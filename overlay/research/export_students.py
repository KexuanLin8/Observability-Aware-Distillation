"""Export students as ONNX for the MuJoCo sim-to-sim evaluation (research/sim2sim.py).

    python research/export_students.py                                  # 3 tasks x (dyn, dyn-hboth), student s0
    python research/export_students.py --tasks skateboard --arms dyn dyn-hboth dyn-hnull --student-seeds 0 1 2

Each student is exported by research/export_onnx.py into $HAIC_OUT/sim2sim/<task>/<arm>/st<s>/
(policy.onnx, policy.json, policy_cfg.json, golden.npz); everything is zipped into
$HAIC_OUT/sim2sim/sim2sim-export.zip. Training-free and short (about a minute per student);
needs no cameras. Do not run it while a training job fills the GPU memory.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import OUT_ROOT, REPO, TASKS, execute, python, read_json, arm_overrides
from research.pipeline import base_overrides, EVAL_SENSOR_SEED_OFFSET
from research.showcase import student_ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks', nargs='+', default=['skateboard', 'push_cart', 'box_stairslope'], choices=sorted(TASKS))
    p.add_argument('--arms', nargs='+', default=['dyn', 'dyn-hboth'])
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--student-seeds', type=int, nargs='+', default=[0])
    p.add_argument('--seed', type=int, default=0, help='seed of the golden Isaac rollout')
    p.add_argument('--force', action='store_true', help='export again even if golden.npz exists')
    p.add_argument('--golden-steps', type=int, default=150, help='steps of the golden Isaac rollout (700: whole episode)')
    a = p.parse_args()
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('export OMNI_KIT_ACCEPT_EULA=YES after reading the Isaac Sim EULA')
    out = OUT_ROOT / 'sim2sim'
    for task in a.tasks:
        a.task = task  # base_overrides reads args.task
        for arm in a.arms:
            spec = TASKS[task]['arms'][arm]
            for s in a.student_seeds:
                job = out / task / arm / f'st{s}'
                if (job / 'golden.npz').is_file() and not a.force:
                    print(f'{job}: done', flush=True)
                    continue
                ckpt = student_ckpt(task, arm, a.teacher_seed, s)
                cmd = [python(), '-u', REPO / 'research/export_onnx.py', 'algo=ppo_haic_finetune',
                       *base_overrides(a, spec), *arm_overrides(spec, EVAL_SENSOR_SEED_OFFSET + a.seed),
                       f'checkpoint_path={ckpt}', 'task.num_envs=4', f'seed={a.seed}', 'vecnorm=eval',
                       'algo.load_noise_scale=null', 'app.enable_cameras=false', f'+out_dir={job}', f'+golden_steps={a.golden_steps}',
                       f'hydra.run.dir={job / "hydra"}', 'hydra.job.chdir=false']
                execute(cmd, job, dict(os.environ, WANDB_MODE='disabled', PYTHONUNBUFFERED='1'), timeout=3600)
                if not (job / 'golden.npz').is_file():
                    raise RuntimeError(f'{job}: export failed; see {job / "run.log"}')
                print(f'{task} {arm} st{s}: exported to {job}', flush=True)
    zpath = out / 'sim2sim-export.zip'
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f != zpath and 'hydra' not in f.parts and f.suffix != '.pt':
                z.write(f, f.relative_to(out))
    print(f'SIM2SIM_EXPORT_ZIP={zpath}', flush=True)


if __name__ == '__main__':
    main()
