"""Baseline pipeline: preflight -> teacher (stage 1) -> students (stage 2) -> evaluation.

    python research/pipeline.py --task skateboard --arm dyn                  # full budget
    python research/pipeline.py --task skateboard --arm dyn --smoke          # ~15 min check

Arms: geo = released config (object pose + geometry; the paper's Geo-Pose ablation),
      dyn = full HAIC (adds object velocity / acceleration to the world-model target),
      dyn-h* = heading-conditioned students trained from the dyn teacher (see research/common.py).

One teacher per (task, arm, teacher seed); each student seed is an independent stage-2
run from that teacher (same protocol as the earlier heading experiments). Every stage
writes complete.json; re-running the same command skips completed stages, and a stage
directory left without complete.json (crash / interruption) is renamed *.incomplete-*
and redone from scratch. Evaluation traces feed research/analyze.py.
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import (REPO, OUT_ROOT, TASKS, TRAIN_EVERY, DYN_FIELDS, execute, checkpoint_info,
                             write_json, read_json, git_state, python, arm_overrides)

EVAL_SENSOR_SEED_OFFSET = 1000  # heading-sensor noise streams differ between training and evaluation


def stage(job: Path):
    """Return the completion record if job finished, else clear a stale attempt."""
    done = job / 'complete.json'
    if done.is_file():
        return read_json(done)
    if job.exists():
        stale = job.with_name(job.name + time.strftime('.incomplete-%Y%m%d-%H%M%S'))
        print(f'Moving unfinished attempt {job} -> {stale.name}', flush=True)
        shutil.move(str(job), str(stale))
    return None


def base_overrides(args, spec):
    return [f'task={spec["task"]}', f'task.command.data_path={REPO / TASKS[args.task]["data"]}',
            'headless=true', 'eval_render=false']


def run_preflight(args, spec, root, env):
    job = root / 'preflight'
    rec = stage(job)
    if rec:
        return rec
    report = job / 'object_obs.json'
    cmd = [python(), '-u', REPO / 'research/preflight.py', 'algo=ppo_haic_train',
           *base_overrides(args, spec), *arm_overrides(spec, args.teacher_seed),
           'task.num_envs=64', f'seed={args.teacher_seed}',
           'wandb.mode=disabled', f'+preflight_out={report}',
           f'hydra.run.dir={job / "hydra"}', 'hydra.job.chdir=false']
    execute(cmd, job, env, timeout=3600)
    rep = read_json(report)
    names = {f['name'] for f in rep['layout']}
    if not rep['pose_tail_ok']:
        raise RuntimeError('object_pos_b / object_ori_b are not the final 12 object_ dims')
    if not rep['all_finite']:
        raise RuntimeError('non-finite object_ values in preflight')
    if 'cuda' not in rep['state_tensor_device']:
        raise RuntimeError(f"simulation state is on {rep['state_tensor_device']}, expected cuda")
    has_dyn = set(DYN_FIELDS) <= names
    if has_dyn != (spec['base'] == 'dyn') or (spec['base'] == 'geo' and names & set(DYN_FIELDS)):
        raise RuntimeError(f'arm {args.arm} has unexpected object_ fields {sorted(names)}')
    groups = rep.get('group_widths', {})
    if spec.get('heading') and groups.get('heading_') != 2:
        raise RuntimeError(f'arm {args.arm} needs a 2-dim heading_ observation group, got {groups}')
    if not spec.get('heading') and 'heading_' in groups:
        raise RuntimeError(f'arm {args.arm} unexpectedly has a heading_ group')
    rec = {'report': rep}
    write_json(job / 'complete.json', rec)
    return rec


def run_train(name, algo, args, spec, seed, frames, root, env, checkpoint=None):
    job = root / name
    rec = stage(job)
    if rec:
        return rec
    cmd = [python(), '-u', REPO / 'scripts/train.py', f'algo={algo}', *base_overrides(args, spec),
           *arm_overrides(spec, seed),
           f'task.num_envs={args.num_envs}', f'total_frames={frames}', f'seed={seed}',
           'wandb.mode=offline', f'save_interval={args.save_interval}',
           f'hydra.run.dir={job / "hydra"}']
    if checkpoint is not None:
        cmd.append(f'checkpoint_path={checkpoint}')
    execute(cmd, job, dict(env, PYTHONHASHSEED=str(seed)))
    found = sorted((job / 'hydra' / 'wandb').glob('*run-*/files/checkpoint_final.pt'))
    if len(found) != 1:
        raise RuntimeError(f'{job}: expected one checkpoint_final.pt, found {len(found)}')
    info = checkpoint_info(found[0])
    want = 'train' if algo == 'ppo_haic_train' else 'finetune'
    if info['last_phase'] != want:
        raise RuntimeError(f'{job}: checkpoint phase {info["last_phase"]} != {want}')
    rec = {'checkpoint': info, 'algo': algo, 'seed': seed, 'frames': frames,
           'iterations': frames // (args.num_envs * TRAIN_EVERY), 'num_envs': args.num_envs,
           'init_from': None if checkpoint is None else str(checkpoint)}
    write_json(job / 'complete.json', rec)
    return rec


def run_eval(job, args, spec, ckpt, algo, seed, env, mode='pred', trace=True, extra=()):
    rec = stage(job)
    if rec:
        return rec
    cmd = [python(), '-u', REPO / 'research/rollout.py', f'algo={algo}', *base_overrides(args, spec),
           *arm_overrides(spec, EVAL_SENSOR_SEED_OFFSET + seed),
           f'checkpoint_path={ckpt}', f'task.num_envs={args.eval_envs}', f'seed={seed}',
           'vecnorm=eval', 'algo.load_noise_scale=null', 'app.enable_cameras=false',
           f'+out_dir={job}', f'+mode={mode}', f'+trace={str(trace).lower()}', *extra,
           f'hydra.run.dir={job / "hydra"}', 'hydra.job.chdir=false']
    execute(cmd, job, dict(env, PYTHONHASHSEED=str(seed)), timeout=4 * 3600)
    if not (job / 'summary.json').is_file():
        raise RuntimeError(f'{job}: the evaluation wrote no summary.json; see {job / "run.log"}')
    s = read_json(job / 'summary.json')
    if s['n_envs'] != args.eval_envs:
        raise RuntimeError(f'{job}: evaluated {s["n_envs"]} envs, expected {args.eval_envs}')
    rec = {k: s[k] for k in ('n_envs', 'n_success', 'success_rate', 'n_strict_success',
                             'strict_success_rate', 'failure_causes', 'mode', 'seed', 'checkpoint_sha256')}
    rec['ablate'] = s.get('ablate')
    write_json(job / 'complete.json', rec)
    print(f'    {job.relative_to(OUT_ROOT)}: {s["n_success"]}/{s["n_envs"]} = {100 * s["success_rate"]:.2f}%', flush=True)
    return rec


def archive(root):
    path = root.with_name(root.name + '-results.zip')
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt', '.wandb', '.zip') and 'incomplete' not in str(p):
                z.write(p, p.relative_to(root.parent))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), default='skateboard')
    p.add_argument('--arm', required=True, help='one of the arms in research/common.py TASKS')
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--student-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--teacher-frames', type=int, default=800_000_000)
    p.add_argument('--student-frames', type=int, default=800_000_000)
    p.add_argument('--eval-envs', type=int, default=1024)
    p.add_argument('--save-interval', type=int, default=500)
    p.add_argument('--skip-teacher-eval', action='store_true')
    p.add_argument('--smoke', action='store_true', help='tiny budgets: checks the whole chain end to end')
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.eval_envs = 64, 64
        args.teacher_frames = args.student_frames = 64 * TRAIN_EVERY * 3
        args.student_seeds, args.eval_seeds = args.student_seeds[:1], args.eval_seeds[:1]
    for frames in (args.teacher_frames, args.student_frames):
        if frames < args.num_envs * TRAIN_EVERY:
            raise SystemExit('frame budget is smaller than one PPO iteration')
    arms = TASKS[args.task]['arms']
    if args.arm not in arms:
        raise SystemExit(f'unknown arm {args.arm}; choose from {sorted(arms)}')
    spec = arms[args.arm]
    task = spec['task']
    if not (REPO / 'cfg/task' / f'{task}.yaml').is_file():
        raise SystemExit(f'missing cfg/task/{task}.yaml')
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('Isaac Sim requires its EULA to be accepted: read it, then export OMNI_KIT_ACCEPT_EULA=YES')

    prefix = 'smoke-' if args.smoke else ''
    root = OUT_ROOT / f'{prefix}{args.task}-{args.arm}' / f'teacher-s{args.teacher_seed}'
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WANDB_MODE='offline', PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1')
    summary = {'task': args.task, 'arm': args.arm, 'task_cfg': task, 'arm_spec': spec, 'args': vars(args),
               'git': git_state(), 'status': 'running', 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(root / 'summary.json', summary)
    print(f'PIPELINE_ROOT={root}', flush=True)
    status = 0
    try:
        summary['preflight'] = run_preflight(args, spec, root, env)['report']['layout']
        if spec.get('teacher_from'):
            src = OUT_ROOT / f'{prefix}{args.task}-{spec["teacher_from"]}' / f'teacher-s{args.teacher_seed}' / 'teacher'
            if not (src / 'complete.json').is_file():
                raise RuntimeError(f'arm {args.arm} reuses the teacher in {src}, which is not finished')
            teacher = dict(read_json(src / 'complete.json'), reused_from=str(src))
        else:
            teacher = run_train('teacher', 'ppo_haic_train', args, spec, args.teacher_seed,
                                args.teacher_frames, root, env)
        t_ckpt = teacher['checkpoint']['path']
        summary['teacher'] = teacher
        write_json(root / 'summary.json', summary)
        if not args.skip_teacher_eval and not spec.get('teacher_from'):
            summary['teacher_eval'] = [run_eval(root / f'teacher-eval-s{e}', args, spec, t_ckpt,
                                                'ppo_haic_train', e, env, trace=False)
                                       for e in args.eval_seeds]
            write_json(root / 'summary.json', summary)
        for ss in args.student_seeds:
            st = run_train(f'student-s{ss}', 'ppo_haic_finetune', args, spec, ss,
                           args.student_frames, root, env, checkpoint=t_ckpt)
            evals = [run_eval(root / f'student-s{ss}' / f'eval-s{e}', args, spec,
                              st['checkpoint']['path'], 'ppo_haic_finetune', e, env)
                     for e in args.eval_seeds]
            n = sum(r['n_envs'] for r in evals)
            k = sum(r['n_success'] for r in evals)
            print(f'STUDENT_RESULT {args.task}-{args.arm} student-s{ss}: {k}/{n} = {100 * k / n:.2f}%', flush=True)
        # aggregate every finished student of this teacher (also those from earlier invocations)
        summary['students'] = {}
        for sd in sorted(root.glob('student-s*')):
            if not (sd / 'complete.json').is_file():
                continue
            evals = [read_json(sd / f'eval-s{e}' / 'complete.json') for e in args.eval_seeds
                     if (sd / f'eval-s{e}' / 'complete.json').is_file()]
            if len(evals) != len(args.eval_seeds):
                continue
            n = sum(r['n_envs'] for r in evals)
            k = sum(r['n_success'] for r in evals)
            summary['students'][sd.name[len('student-s'):]] = {
                'checkpoint': read_json(sd / 'complete.json')['checkpoint'], 'evals': evals, 'success_rate': k / n}
        write_json(root / 'summary.json', summary)
        rates = [v['success_rate'] for v in summary['students'].values()]
        import statistics
        summary['student_mean'] = statistics.mean(rates)
        summary['student_sd'] = statistics.stdev(rates) if len(rates) > 1 else None
        summary['status'] = 'complete'
        sd = '' if summary['student_sd'] is None else f' ± {100 * summary["student_sd"]:.2f}'
        print(f'PIPELINE_RESULT {args.task}-{args.arm}: {100 * summary["student_mean"]:.2f}%{sd} '
              f'over student seeds {sorted(summary["students"])}', flush=True)
    except Exception as exc:
        summary['status'] = 'failed'
        summary['failure'] = f'{type(exc).__name__}: {exc}'
        print('PIPELINE_ERROR ' + summary['failure'], flush=True)
        status = 1
    summary['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_json(root / 'summary.json', summary)
    print(f'PIPELINE_ARCHIVE={archive(root)}', flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
