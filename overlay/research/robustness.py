"""Heading-sensor robustness: evaluate trained heading students with scaled IMU error models.

    python research/robustness.py --task skateboard --arms dyn-hboth dyn-hact dyn-hwm
    python research/robustness.py --task push_cart --arms dyn-hboth --scales 0 1 2 5 10 20

Every error term of robot_heading_err_imu (alignment error, gyro bias, bias random walk, white
rate noise) is multiplied by the scale at evaluation time only; the policy is unchanged (it was
trained at scale 1). Scale 0 is a perfect heading. The table also gives the expected end-of-episode
heading error (1 sigma) of the sensor model and the arm's dyn baseline for reference.
Training-free: one evaluation per (arm, student, scale, eval seed).
"""
import argparse
import json
import math
import os
import statistics
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import OUT_ROOT, TASKS, HEADING_NOISE_DEFAULTS, read_json, write_json
from research.pipeline import run_eval

SENSOR = 'task.observation.heading_.robot_heading_err_imu'


def noise_overrides(scale):
    return [f'{SENSOR}.{k}={v * scale}' for k, v in HEADING_NOISE_DEFAULTS.items()]


def expected_heading_sigma_deg(scale, duration_s, dt=0.02):
    """1-sigma heading error of the sensor model after duration_s (independent terms)."""
    p = {k: v * scale for k, v in HEADING_NOISE_DEFAULTS.items()}
    var = (p['init_align_std'] ** 2 + (p['bias_std'] * duration_s) ** 2
           + p['bias_rw_std'] ** 2 * duration_s ** 3 / 3 + p['rate_noise_std'] ** 2 * dt * duration_s)
    return math.degrees(math.sqrt(var))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), default='skateboard')
    p.add_argument('--arms', nargs='+', default=['dyn-hboth'])
    p.add_argument('--scales', type=float, nargs='+', default=[0, 1, 2, 5, 10])
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[0])
    p.add_argument('--eval-envs', type=int, default=1024)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.eval_envs, a.eval_seeds, a.scales = 64, a.eval_seeds[:1], a.scales[:2]
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('export OMNI_KIT_ACCEPT_EULA=YES after reading the Isaac Sim EULA')
    env = dict(os.environ, WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    prefix = 'smoke-' if a.smoke else ''
    args = SimpleNamespace(task=a.task, eval_envs=a.eval_envs)
    results, duration = {}, None
    for arm in a.arms:
        spec = TASKS[a.task]['arms'][arm]
        if not spec.get('heading'):
            raise SystemExit(f'{arm} has no heading sensor')
        root = OUT_ROOT / f'{prefix}{a.task}-{arm}' / f'teacher-s{a.teacher_seed}'
        s = read_json(root / 'summary.json')
        if s.get('status') != 'complete':
            raise SystemExit(f'{root} is not complete')
        results[arm] = {}
        for scale in a.scales:
            rates = []
            for ss, st in sorted(s['students'].items()):
                for e in a.eval_seeds:
                    job = root / f'student-s{ss}' / 'robustness' / f'noise{scale:g}-s{e}'
                    rec = run_eval(job, args, spec, st['checkpoint']['path'], 'ppo_haic_finetune', e, env,
                                   trace=False, extra=noise_overrides(scale))
                    rates.append(rec['success_rate'])
                    if duration is None:
                        summ = read_json(job / 'summary.json')
                        duration = max(summ['reference_motion_lengths_steps']) * summ['step_dt']
            results[arm][scale] = rates
    base = OUT_ROOT / f'{prefix}{a.task}-dyn' / f'teacher-s{a.teacher_seed}' / 'summary.json'
    dyn_mean = read_json(base).get('student_mean') if base.is_file() else None

    L = [f'# Heading-sensor robustness: {a.task}', '',
         f'Error terms scaled at evaluation only (trained at scale 1). Eval seeds {a.eval_seeds}; '
         f'mean ± sd over students. dyn baseline (no heading): '
         + ('n/a' if dyn_mean is None else f'{100 * dyn_mean:.2f}%') + '.', '',
         '| scale | expected heading error at episode end (1σ) | ' + ' | '.join(a.arms) + ' |',
         '|---|---|' + '---|' * len(a.arms)]
    for scale in a.scales:
        cells = []
        for arm in a.arms:
            r = results[arm][scale]
            sd = statistics.stdev(r) if len(r) > 1 else float('nan')
            cells.append(f'{100 * statistics.mean(r):.2f}%' + ('' if sd != sd else f' ± {100 * sd:.2f}'))
        L.append(f'| {scale:g}× | {expected_heading_sigma_deg(scale, duration):.1f}° | ' + ' | '.join(cells) + ' |')
    out = OUT_ROOT / 'robustness' / f'{prefix}{a.task}'
    write_json(out / 'robustness.json', {'task': a.task, 'scales': a.scales, 'results': results,
                                         'dyn_baseline': dyn_mean, 'episode_s': duration,
                                         'noise_defaults': HEADING_NOISE_DEFAULTS})
    (out / 'robustness.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    zpath = out.with_name(out.name + '.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            z.write(f, f.relative_to(out.parent))
    print('\n'.join(L), flush=True)
    print(f'ROBUSTNESS_ZIP={zpath}', flush=True)


if __name__ == '__main__':
    main()
