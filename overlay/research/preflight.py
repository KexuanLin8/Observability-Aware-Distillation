"""Preflight: build the task with an untrained teacher, check object_ layout, record ranges.

Also serves as the end-to-end environment check on a new machine: it starts Isaac Sim
headless, builds the scene on the GPU, and steps physics for a few hundred steps.

    python research/preflight.py task=G1/haic/skateboard_dyn +preflight_out=/path/report.json
(called by pipeline.py with the remaining overrides)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import active_adaptation  # noqa: F401  registers the algo configs (ppo_haic_*) in Hydra's ConfigStore
import torch
from omegaconf import OmegaConf

from research.common import REPO, seed_all, object_layout, check_pose_tail, write_json


def quantiles(x):
    a = x.abs().flatten().float()
    if a.numel() > 2_000_000:
        a = a[torch.randperm(a.numel(), device=a.device)[:2_000_000]]
    q = torch.quantile(a, torch.tensor([0.5, 0.99, 0.999], device=a.device)).tolist()
    return {'abs_p50': q[0], 'abs_p99': q[1], 'abs_p999': q[2], 'abs_max': a.max().item()}


@hydra.main(config_path=str(REPO / 'cfg'), config_name='train', version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    out = Path(cfg.preflight_out).resolve()
    steps = int(cfg.get('preflight_steps', 300))
    assert cfg.algo.phase == 'train' and cfg.checkpoint_path is None
    from isaaclab.app import AppLauncher
    seed_all(cfg.seed)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    env = None
    try:
        from helpers import make_env_policy
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        seed_all(cfg.seed)
        env, agent, _ = make_env_policy(cfg)
        base = env.base_env
        carry = env.reset()
        layout = object_layout(base, 'object_')
        width = int(carry['object_'].shape[-1])
        if sum(f['dim'] for f in layout) != width:
            raise RuntimeError(f'object_ field widths do not sum to {width}: {layout}')
        geo_width = int(carry['object_geo_'].shape[-1])
        n_objects = geo_width // (128 * 3)

        policy = agent.get_rollout_policy('train')
        samples, resets = [], 0
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            for _ in range(steps):
                samples.append(carry['object_'].clone())
                carry = policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                resets += int(td['next', 'done'].sum().item())
        data = torch.stack(samples)
        for item in layout:
            sl = data[..., item['start']:item['start'] + item['dim']]
            item.update(quantiles(sl))
            item['finite'] = bool(torch.isfinite(sl).all())
        report = {
            'task': cfg.task.name, 'num_envs': base.num_envs, 'steps': steps,
            'sim_device': str(base.device), 'state_tensor_device': str(carry['policy'].device),
            'episode_resets_during_rollout': resets, 'object_width': width, 'layout': layout,
            'all_finite': bool(torch.isfinite(data).all()),
            'pose_tail_ok': check_pose_tail(layout, n_objects),
            'object_geo_width': geo_width, 'n_objects': n_objects,
            'group_widths': {k: int(v.shape[-1]) for k, v in carry.items()
                             if k in base.observation_funcs},
            'note': 'Untrained teacher with exploration noise; ranges describe falls and resets, not a trained gait.',
        }
        write_json(out, report)
        print('PREFLIGHT ' + json.dumps({k: report[k] for k in (
            'task', 'sim_device', 'object_width', 'pose_tail_ok', 'all_finite', 'n_objects')}), flush=True)
        for item in layout:
            print(f"  {item['name']:<24} dims {item['start']:>3}-{item['start'] + item['dim'] - 1:<3} "
                  f"|x| p50={item['abs_p50']:.3g} p99={item['abs_p99']:.3g} "
                  f"p99.9={item['abs_p999']:.3g} max={item['abs_max']:.3g}", flush=True)
        sys.stdout.flush()
        os._exit(0)  # Isaac Sim teardown can hang after a completed run
    except BaseException:
        # Isaac Sim's shutdown can hang after an exception; report and exit right away
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
