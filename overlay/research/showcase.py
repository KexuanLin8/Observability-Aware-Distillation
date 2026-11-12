"""Paired showcase for the paper figures (Fig. 1): the same episodes under two students.

    python research/showcase.py --task skateboard                       # dyn vs dyn-hboth, student s0
    python research/showcase.py --task push_cart --render-k 2 --video   # two episodes, with videos
    python research/showcase.py --task skateboard --env-id 17           # render a chosen episode

Step 1 (paths): both arms run the same eval seed with --envs parallel episodes and record the
world-frame root / object paths of every episode (paths.npz; feeds paper/scripts/fig_showcase.py).
Step 2 (selection): episodes where the first arm fails by a global torso-position termination
(drift) and the second arm succeeds, ranked by the first arm's final lateral error.
Step 3 (render): each selected episode is rendered for both arms with the same camera
(snapshots every --snap-every s; --video adds an mp4). Rendering never runs in step 1, and
every rendered run checks that the episode outcome matches step 1.

Training-free; needs app.enable_cameras (offscreen RTX rendering) only in step 3. Do not run it
while a training job fills the GPU memory. Outputs: $HAIC_OUT/showcase/<task>/ and a zip.
"""
import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import OUT_ROOT, REPO, TASKS, execute, python, read_json, write_json, arm_overrides
from research.pipeline import base_overrides, EVAL_SENSOR_SEED_OFFSET

GLOBAL_POS = 'term/cum_body_pos_error'


def student_ckpt(task, arm, teacher_seed, student_seed):
    root = OUT_ROOT / f'{task}-{arm}' / f'teacher-s{teacher_seed}'
    s = read_json(root / 'summary.json')
    st = s['students'].get(str(student_seed))
    if st is None:
        raise SystemExit(f'{root}: no student seed {student_seed}')
    return st['checkpoint']['path']


def run(job, a, arm, extra):
    spec = TASKS[a.task]['arms'][arm]
    ckpt = student_ckpt(a.task, arm, a.teacher_seed, a.student_seed)
    if (job / 'summary.json').is_file():
        return read_json(job / 'summary.json')
    cmd = [python(), '-u', REPO / 'research/render.py', 'algo=ppo_haic_finetune',
           *base_overrides(a, spec), *arm_overrides(spec, EVAL_SENSOR_SEED_OFFSET + a.seed),
           f'checkpoint_path={ckpt}', f'task.num_envs={a.envs}', f'seed={a.seed}',
           'vecnorm=eval', 'algo.load_noise_scale=null', f'+out_dir={job}', *extra,
           f'hydra.run.dir={job / "hydra"}', 'hydra.job.chdir=false']
    execute(cmd, job, dict(os.environ, WANDB_MODE='disabled', PYTHONUNBUFFERED='1',
                           PYTHONHASHSEED=str(a.seed)), timeout=2 * 3600)
    if not (job / 'summary.json').is_file():
        raise RuntimeError(f'{job}: no summary.json; see {job / "run.log"}')
    return read_json(job / 'summary.json')


def episodes(job):
    with (job / 'episodes.csv').open(encoding='utf-8') as f:
        return {int(r['env_id']): r for r in csv.DictReader(f)}


def final_lateral(job):
    """Lateral root error w.r.t. the reference heading at the last alive step of every env."""
    z = np.load(job / 'paths.npz')
    d = z['root_xyz'][..., :2] - z['ref_root_xyz'][..., :2]
    yaw = z['ref_root_yaw']
    lat = -np.sin(yaw) * d[..., 0] + np.cos(yaw) * d[..., 1]
    alive = ~np.isnan(lat)
    last = np.where(alive.any(0), alive.shape[0] - 1 - np.argmax(alive[::-1], axis=0), 0)
    return np.abs(lat[last, np.arange(lat.shape[1])])


def pair_select(eb, eg, lat):
    """Episodes where the baseline drifts out (global torso termination) and the better arm succeeds,
    ranked by the baseline's final lateral error."""
    ok = lambda r: int(r['success']) == 1
    cand = [i for i in sorted(eb) if not ok(eb[i]) and int(eb[i].get(GLOBAL_POS, 0)) == 1 and ok(eg[i])]
    cand.sort(key=lambda i: -lat[i])
    return {'candidates': cand, 'final_lateral_m': {i: float(lat[i]) for i in cand},
            'n_base_fail_better_ok': len(cand),
            'n_both_success': sum(ok(eb[i]) and ok(eg[i]) for i in eb),
            'n_base_ok_better_fail': sum(ok(eb[i]) and not ok(eg[i]) for i in eb)}


def fixed_target(job, env_id):
    z = np.load(job / 'paths.npz')
    p = z['ref_root_xyz'][:, env_id]
    p = p[~np.isnan(p).any(-1)]
    c = (p.min(0) + p.max(0)) / 2
    return [round(float(c[0]), 3), round(float(c[1]), 3), 0.8], float(np.linalg.norm(p.max(0)[:2] - p.min(0)[:2]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), default='skateboard')
    p.add_argument('--arms', nargs=2, default=['dyn', 'dyn-hboth'], help='baseline arm, improved arm')
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--student-seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0, help='evaluation seed')
    p.add_argument('--envs', type=int, default=256)
    p.add_argument('--render-k', type=int, default=1, help='number of episodes to render')
    p.add_argument('--env-id', type=int, nargs='*', default=None, help='render these episodes instead')
    p.add_argument('--snap-every', type=float, default=1.5)
    p.add_argument('--video', action='store_true')
    p.add_argument('--camera', choices=['chase', 'follow_ref', 'fixed'], default='chase',
                   help='chase: from behind along the reference heading (lateral drift is left/right)')
    p.add_argument('--eye', type=float, nargs=3, default=None,
                   help='camera offset [m]; default chase [-4.0 0 2.2] (heading frame), else [0 -4.5 1.6]')
    p.add_argument('--markers', choices=['ref', 'both', 'none'], default='ref')
    p.add_argument('--resolution', type=int, nargs=2, default=[1600, 900])
    p.add_argument('--paths-only', action='store_true', help='step 1 and 2 only (no rendering)')
    a = p.parse_args()
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('export OMNI_KIT_ACCEPT_EULA=YES after reading the Isaac Sim EULA')
    out = OUT_ROOT / 'showcase' / a.task
    tag = f'st{a.student_seed}-ev{a.seed}-n{a.envs}'

    # step 1: paths of every episode, no rendering
    jobs = {arm: out / tag / arm / 'paths' for arm in a.arms}
    res = {arm: run(jobs[arm], a, arm, ['app.enable_cameras=false', '+render_env=-1']) for arm in a.arms}
    for arm in a.arms:
        print(f'{arm}: {res[arm]["n_success"]}/{res[arm]["num_envs"]} = {100 * res[arm]["success_rate"]:.2f}%', flush=True)

    # step 2: paired selection
    base, better = a.arms
    sel = pair_select(episodes(jobs[base]), episodes(jobs[better]), final_lateral(jobs[base]))
    cand = sel['candidates']
    both_ok = sel['n_both_success']
    chosen = a.env_id if a.env_id else cand[:a.render_k]
    sel['rendered'] = chosen
    write_json(out / tag / 'selection.json', sel)
    print(f'paired episodes: {base} drift-fails & {better} succeeds = {len(cand)}, both succeed = {both_ok}, '
          f'reverse = {sel["n_base_ok_better_fail"]}; rendering {chosen}', flush=True)
    if not chosen and not a.paths_only:
        raise SystemExit('no paired drift episode; pass --env-id or increase --envs')

    # step 3: render the chosen episodes with identical cameras
    if not a.paths_only:
        for env_id in chosen:
            tgt, span = fixed_target(jobs[base], env_id)
            eye = a.eye or ([-4.0, 0.0, 2.2] if a.camera == 'chase' else [0.0, -4.5, 1.6])
            if a.camera == 'fixed' and a.eye is None:  # back off far enough to see the whole reference path
                eye = [0.0, -max(4.5, 1.1 * span), max(1.6, 0.4 * span)]
            extra = ['app.enable_cameras=true', f'+render_env={env_id}', f'+snap_every_s={a.snap_every}',
                     f'+video={str(a.video).lower()}', f'+camera={a.camera}', f'+markers={a.markers}',
                     f'+eye=[{eye[0]},{eye[1]},{eye[2]}]', f'+target=[{tgt[0]},{tgt[1]},{tgt[2]}]',
                     f'task.viewer.resolution=[{a.resolution[0]},{a.resolution[1]}]']
            for arm in a.arms:
                job = out / tag / arm / f'render-env{env_id}-{a.camera}'
                r = run(job, a, arm, extra)
                want = int(episodes(jobs[arm])[env_id]['success'])
                if r['rendered_success'] != want:
                    print(f'WARNING: {arm} env {env_id}: rendered outcome {r["rendered_success"]} differs from '
                          f'step 1 ({want}); the simulation is not bit-reproducible here', flush=True)
                print(f'{arm} env {env_id}: success={r["rendered_success"]}, {len(r["snapshots"])} snapshots in {job}',
                      flush=True)

    zpath = out.with_name(f'showcase-{a.task}.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix not in ('.pt', '.wandb') and 'hydra' not in f.parts:
                z.write(f, f.relative_to(out.parent))
    print(f'SHOWCASE_ZIP={zpath}', flush=True)


if __name__ == '__main__':
    main()
