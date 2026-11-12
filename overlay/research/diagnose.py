"""Training-free mechanism diagnostics on finished pipeline runs.

    python research/diagnose.py --task skateboard --arms geo dyn
    python research/diagnose.py --task skateboard --arms dyn dyn-hwm dyn-hnull

1. For every student of every arm, runs the counterfactual oracle evaluations requested
   by --modes (the normal eval-s*/ traces were already written by pipeline.py; missing
   ones are run here too). pred_dt is the drift-tolerant evaluation (global torso terminations off).
2. Runs research/analyze.py over all arms and writes report.md / figures / metrics.json
   to $HAIC_OUT/diagnostics/<task>-<arms>-<k>students/, and packs it with every evaluation's
   summary / episodes (no traces, no weights) into a zip next to it.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import OUT_ROOT, REPO, TASKS, read_json, execute, python
from research.pipeline import run_eval


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), default='skateboard')
    p.add_argument('--arms', nargs='+', default=['geo', 'dyn'])
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--eval-envs', type=int, default=1024)
    p.add_argument('--modes', nargs='*', default=['oracle_object', 'oracle_priv', 'pred_dt'],
                   choices=['oracle_object', 'oracle_priv', 'pred_dt'])
    p.add_argument('--smoke', action='store_true', help='use the smoke-run roots')
    a = p.parse_args()
    if a.smoke:
        a.eval_envs, a.eval_seeds = 64, a.eval_seeds[:1]
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('export OMNI_KIT_ACCEPT_EULA=YES after reading the Isaac Sim EULA')
    env = dict(os.environ, WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    prefix = 'smoke-' if a.smoke else ''
    roots, n_students = [], []
    for arm in a.arms:
        root = OUT_ROOT / f'{prefix}{a.task}-{arm}' / f'teacher-s{a.teacher_seed}'
        s = read_json(root / 'summary.json')
        if s.get('status') != 'complete':
            raise SystemExit(f'{root} is not complete (status {s.get("status")}); finish pipeline.py first')
        args = SimpleNamespace(task=a.task, eval_envs=a.eval_envs)
        spec = TASKS[a.task]['arms'][arm]
        for ss, st in s['students'].items():
            ckpt = st['checkpoint']['path']
            for e in a.eval_seeds:
                run_eval(root / f'student-s{ss}' / f'eval-s{e}', args, spec, ckpt, 'ppo_haic_finetune', e, env)
                for mode in a.modes:
                    run_eval(root / f'student-s{ss}' / f'eval-{mode}-s{e}', args, spec, ckpt,
                             'ppo_haic_finetune', e, env, mode=mode, trace=False)
        roots.append(root)
        n_students.append(len(s['students']))
    out = OUT_ROOT / 'diagnostics' / f'{prefix}{a.task}-{"-".join(a.arms)}-{min(n_students)}students'
    cmd = [python(), REPO / 'research/analyze.py', '--runs', *roots, '--labels', *a.arms, '--out', out]
    execute(cmd, out / 'analyze-job', env)
    zpath = out.with_name(out.name + '.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
        for root in roots:
            for f in sorted(root.glob('student-s*/eval-*/*')):
                if f.name in ('summary.json', 'episodes.csv', 'wm_stats.json', 'complete.json') and 'incomplete' not in str(f):
                    z.write(f, Path('runs') / f.relative_to(OUT_ROOT))
            if (root / 'summary.json').is_file():
                z.write(root / 'summary.json', Path('runs') / (root / 'summary.json').relative_to(OUT_ROOT))
    print(f'DIAGNOSTICS_REPORT={out / "report.md"}', flush=True)
    print(f'DIAGNOSTICS_ZIP={zpath}', flush=True)


if __name__ == '__main__':
    main()
