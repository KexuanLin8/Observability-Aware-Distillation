"""Paired showcase rollouts for the paper figures: world-frame paths of every env, optionally
rendered snapshots / video of one env. Driven by research/showcase.py; not part of the
evaluation pipeline (research/rollout.py is unchanged).

Same protocol as rollout.py mode=pred (seeds fixed before the simulator is built, first episode
from reference frame 0, deterministic actions), so two arms run with the same seed and env count
see the same domain randomization. Rendering does not touch the physics.

Config (Hydra, on top of the usual task / algo / checkpoint overrides):
    +out_dir=...          output directory
    +render_env=-1        env to render (-1: paths only, no rendering)
    +snap_every_s=1.5     snapshot period [s] (first and last frame always included)
    +video=false          also write every control step to video.mp4 (or frames/ if no ffmpeg)
    +camera=chase         chase: camera behind the reference root along its (smoothed) heading, so
                          lateral drift shows as the robot moving sideways off the reference markers;
                          follow_ref: camera follows the reference root with a fixed world offset;
                          fixed: one camera for the whole episode
    +target=[0,0,0.8]     fixed camera: target relative to the env origin [m]
    +eye=...              camera offset from its target [m]; chase: in the heading frame (default
                          [-4.0,0.0,2.2]), otherwise world frame (default [0.0,-4.5,1.6])
    +yaw_tau_s=1.0        chase: time constant of the heading smoothing [s]
    +markers=ref          ref: red reference keypoints; both: also green robot keypoints; none
    +hide_others=true     hide all other envs in the render
Outputs: paths.npz, episodes.csv, summary.json, snapshots/*.png (+ video.mp4). The *_end.png
snapshot is the last frame before the termination (the env is already reset after it).
"""
import csv
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import active_adaptation  # noqa: F401  registers the algo configs (ppo_haic_*) in Hydra's ConfigStore
import numpy as np
import torch
from omegaconf import OmegaConf

from research.common import seed_all, sha256, git_state, write_json
from research.rollout import make_policy, yaw_of_quat


def hide_envs(n, keep):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    for i in range(n):
        if i != keep:
            prim = stage.GetPrimAtPath(f'/World/envs/env_{i}')
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()


class VideoSink:
    """Stream frames to video.mp4; without an ffmpeg backend, write numbered PNGs to frames/."""

    def __init__(self, out, fps):
        import imageio.v2 as imageio
        self.imageio, self.out, self.i = imageio, out, 0
        try:
            self.writer = imageio.get_writer(out / 'video.mp4', fps=fps, codec='libx264', quality=8)
        except Exception as e:
            print(f'WARNING: no mp4 writer ({e}); writing frames/', flush=True)
            self.writer = None
            (out / 'frames').mkdir(exist_ok=True)

    def add(self, img):
        if self.writer is not None:
            self.writer.append_data(img)
        else:
            self.imageio.imwrite(self.out / 'frames' / f'{self.i:04d}.png', img)
        self.i += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()


@hydra.main(config_path='../cfg', config_name='eval', version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    out = Path(cfg.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    k = int(cfg.get('render_env', -1))
    rendering = k >= 0
    snap_every = float(cfg.get('snap_every_s', 1.5))
    video = bool(cfg.get('video', False))
    camera = str(cfg.get('camera', 'chase'))
    eye = torch.tensor(list(cfg.get('eye', [-4.0, 0.0, 2.2] if camera == 'chase' else [0.0, -4.5, 1.6])),
                       dtype=torch.float32)
    yaw_tau = float(cfg.get('yaw_tau_s', 1.0))
    markers = str(cfg.get('markers', 'ref'))
    hide_others = bool(cfg.get('hide_others', True))
    # fixed camera target relative to the env origin (showcase.py passes the centre of the reference path)
    fixed_target = torch.tensor(list(cfg.get('target', [0.0, 0.0, 0.8])), dtype=torch.float32)
    assert camera in ('chase', 'follow_ref', 'fixed') and markers in ('ref', 'both', 'none')
    assert cfg.vecnorm == 'eval' and not cfg.eval_render
    assert rendering == bool(cfg.app.enable_cameras), 'set app.enable_cameras=true exactly when rendering'
    ckpt = Path(cfg.checkpoint_path).resolve()
    assert ckpt.is_file(), f'Missing checkpoint: {ckpt}'
    seed = int(cfg.seed)
    meta = {'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt), 'seed': seed, 'task': cfg.task.name,
            'render_env': k, 'camera': camera, 'eye': eye.tolist(), 'markers': markers,
            'git': git_state(), 'script_sha256': sha256(__file__)}
    OmegaConf.save(cfg, out / 'resolved_cfg.yaml')

    from isaaclab.app import AppLauncher
    seed_all(seed)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    env = None
    try:
        from helpers import make_env_policy
        from torchrl.envs.utils import set_exploration_type, ExplorationType

        seed_all(seed)  # before creation-time domain randomization
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            env, agent, _ = make_env_policy(cfg)
        bad = [str(w.message) for w in caught if 'Failed to load state dict' in str(w.message)]
        if bad:
            raise RuntimeError('Incomplete model load: ' + '; '.join(bad))
        base = env.base_env
        base.eval(); env.eval(); agent.eval()
        seed_all(seed)
        env.set_seed(seed)
        carry = env.reset()
        cmd = base.command_manager
        n, dev, T = env.num_envs, env.device, int(env.max_episode_length)
        dt = float(base.step_dt)
        origins = base.scene.env_origins.clone()
        policy = make_policy(agent, 'pred')
        meta.update({'num_envs': n, 'step_dt': dt, 'max_episode_length': T})
        if k >= n:
            raise ValueError(f'render_env={k} but num_envs={n}')

        snap_dir = out / 'snapshots'
        if rendering:
            snap_dir.mkdir(exist_ok=True)
            if hide_others:
                hide_envs(n, k)
            snap_steps = set(range(0, T, max(1, round(snap_every / dt))))

            cam_yaw = [None]  # smoothed reference heading of the chase camera (unwrapped)

            def look():
                tgt = (origins[k] + fixed_target if camera == 'fixed' else cmd.ref_root_pos_w[k]).detach().cpu().float()
                off = eye
                if camera == 'chase':
                    y = float(yaw_of_quat(cmd.ref_root_quat_w[k:k + 1])[0])
                    if cam_yaw[0] is None:
                        cam_yaw[0] = y
                    else:  # first-order low pass on the wrapped difference
                        dy = (y - cam_yaw[0] + np.pi) % (2 * np.pi) - np.pi
                        cam_yaw[0] += dy * min(1.0, dt / yaw_tau)
                    c, s = np.cos(cam_yaw[0]), np.sin(cam_yaw[0])
                    off = torch.tensor([c * eye[0] - s * eye[1], s * eye[0] + c * eye[1], eye[2]])
                base.sim.set_camera_view(eye=(tgt + off).tolist(), target=tgt.tolist())

            def draw():
                if markers == 'none' or not hasattr(cmd, 'vis_markers'):
                    return
                m = cmd.all_marker_pos_w[:, k]  # [2, bodies, 3]: robot, reference
                if markers == 'ref':
                    cmd.vis_markers.visualize(translations=m[1], marker_indices=[1] * m.shape[1])
                else:
                    cmd.vis_markers.visualize(translations=m.reshape(-1, 3), marker_indices=[0] * m.shape[1] + [1] * m.shape[1])

            def grab(settle):
                img = None
                for _ in range(settle):
                    img = base.render('rgb_array')
                return np.ascontiguousarray(img)

        nan = float('nan')
        names = ('root', 'ref_root', 'obj', 'ref_obj')
        paths = {f'{p}_xyz': torch.full((T, n, 3), nan, device=dev) for p in names}
        yaws = {f'{p}_yaw': torch.full((T, n), nan, device=dev) for p in ('root', 'ref_root')}
        rows = [None] * n
        seen = torch.zeros(n, dtype=torch.bool, device=dev)
        snaps, last_step = [], 0
        import imageio.v2 as imageio
        sink = VideoSink(out, round(1 / dt)) if rendering and video else None
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for step in range(T):
                alive = ~seen
                a3 = alive.unsqueeze(-1)
                for key, v in (('root_xyz', cmd.robot_root_pos_w), ('ref_root_xyz', cmd.ref_root_pos_w),
                               ('obj_xyz', cmd.object_pos_w), ('ref_obj_xyz', cmd.ref_object_pos_w)):
                    paths[key][step] = torch.where(a3, (v - origins).float(), torch.full_like(v.float(), nan))
                for key, q in (('root_yaw', cmd.robot_root_quat_w), ('ref_root_yaw', cmd.ref_root_quat_w)):
                    y = yaw_of_quat(q).float()
                    yaws[key][step] = torch.where(alive, y, torch.full_like(y, nan))

                if rendering and not seen[k]:
                    look()
                    draw()
                    # every step is rendered: the env resets inside step_and_maybe_reset, so the last
                    # frame before the termination is the one taken here at the terminating step
                    is_snap = step in snap_steps
                    last_img, last_step_k = grab(3 if is_snap else 1), step
                    if video:
                        sink.add(last_img)
                    if is_snap:
                        imageio.imwrite(snap_dir / f'step{step:04d}_t{step * dt:05.2f}s.png', last_img)
                        snaps.append({'step': step, 't': step * dt})

                carry = policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                nxt = td['next']
                done = nxt['done'].reshape(n).bool()
                new = done & ~seen
                if new.any():
                    ids = new.nonzero(as_tuple=False).flatten()
                    stats = nxt['stats'][ids].cpu()
                    for j, env_id in enumerate(ids.cpu().tolist()):
                        row = {'env_id': env_id, 'success': int(stats['success'][j].item() > .5),
                               'episode_steps': int(stats['episode_len'][j].item()), 'end_rollout_step': step}
                        for key, value in stats.items(True, True):
                            parts = key if isinstance(key, tuple) else (key,)
                            if parts[0] == 'termination':
                                row['term/' + '/'.join(parts[1:])] = int(float(value[j].item()) > .5)
                        rows[env_id] = row
                    if rendering and bool(new[k]):  # last frame of the rendered episode, before its reset
                        t_end = last_step_k * dt
                        imageio.imwrite(snap_dir / f'step{last_step_k:04d}_t{t_end:05.2f}s_end.png', last_img)
                        snaps.append({'step': last_step_k, 't': t_end, 'end': True})
                    seen |= new
                last_step = step
                if step % 100 == 0 or seen.all():
                    print(f'RENDER_PROGRESS step={step + 1} done={int(seen.sum())}/{n}', flush=True)
                if seen.all():
                    break

        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            raise RuntimeError(f'{len(missing)} first episodes did not finish within {T} steps')
        keys = sorted({c for r in rows for c in r})
        with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys, restval=0)
            w.writeheader()
            w.writerows(rows)
        L = last_step + 1
        arrays = {key: v[:L].cpu().numpy() for key, v in {**paths, **yaws}.items()}
        np.savez_compressed(out / 'paths.npz', step_dt=dt, **arrays)
        if sink is not None:
            sink.close()
        n_succ = sum(r['success'] for r in rows)
        summary = dict(meta, n_success=n_succ, success_rate=n_succ / n, snapshots=snaps,
                       rendered_success=(rows[k]['success'] if rendering else None))
        write_json(out / 'summary.json', summary)
        print('RENDER_RESULT ' + json.dumps({'n_success': n_succ, 'n_envs': n, 'render_env': k,
                                             'rendered_success': summary['rendered_success']}), flush=True)
        sys.stdout.flush()
        os._exit(0)  # Isaac Sim teardown can hang after a completed rollout
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
