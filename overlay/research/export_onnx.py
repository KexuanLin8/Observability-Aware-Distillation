"""Export one student as ONNX for the MuJoCo sim-to-sim evaluation (research/sim2sim.py).
Driven by research/export_students.py; not part of the evaluation pipeline.

The exported graph is the deployed student (world model, privilege adapter and actor of the
finetune phase) behind the observation normalizer of its checkpoint, so it takes raw observations.
Besides policy.onnx it writes
    policy.json   in/out keys of the graph (the order of the ONNX inputs and outputs)
    policy.yaml   observation terms, joint names, gains and action scaling of the task
    golden.npz    the first steps of one Isaac rollout of the same student: raw (unnormalized) inputs
                  of every in_key, the exported and the rollout action, and the robot state they come from, so that the
                  MuJoCo observation builder can be checked term by term against Isaac
Config (Hydra, on top of the usual task / algo / checkpoint overrides):
    +out_dir=...       output directory
    +golden_steps=150  steps recorded in golden.npz
"""
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import active_adaptation  # noqa: F401  registers the algo configs (ppo_haic_*) in Hydra's ConfigStore
import numpy as np
import torch
from omegaconf import OmegaConf

from research.common import seed_all, sha256, write_json


@hydra.main(config_path='../cfg', config_name='eval', version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    out = Path(cfg.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_golden = int(cfg.get('golden_steps', 150))
    assert cfg.vecnorm == 'eval' and not cfg.app.enable_cameras
    ckpt = Path(cfg.checkpoint_path).resolve()
    assert ckpt.is_file(), f'Missing checkpoint: {ckpt}'
    seed = int(cfg.seed)

    from isaaclab.app import AppLauncher
    seed_all(seed)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    env = None
    try:
        from helpers import make_env_policy, ObsNorm
        from tensordict.nn import TensorDictSequential
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from active_adaptation.utils.export import export_onnx
        from active_adaptation.assets import get_asset_meta

        env, agent, vecnorm = make_env_policy(cfg)
        base = env.base_env
        base.eval(); env.eval(); agent.eval()

        deploy = copy.deepcopy(agent.get_rollout_policy('deploy'))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy.in_keys)
        policy = TensorDictSequential(obs_norm, deploy).cpu()
        fake = env.observation_spec[0].rand().cpu()
        fake['is_init'] = torch.tensor(1, dtype=bool)
        fake = fake.unsqueeze(0)
        export_onnx(policy, fake, str(out / 'policy.onnx'))  # also writes policy.json (in/out keys)

        asset = get_asset_meta(base.scene['robot'])
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        write_json(out / 'policy_cfg.json', {
            'observation': {k: dict_cfg['task']['observation'][k] for k in policy.in_keys
                            if k in dict_cfg['task']['observation']},
            'action_scaling': dict_cfg['task']['action']['action_scaling'],
            'isaac_joint_names': list(asset['joint_names_isaac']),
            'policy_joint_names': list(base.action_manager.joint_names),
            'default_joint_pos': asset['init_state']['joint_pos'],
            'actuators': json.loads(json.dumps(asset['actuators'], default=str)),
            'step_dt': float(base.step_dt),
            'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt), 'task': cfg.task.name,
            'script_sha256': sha256(__file__)})

        # golden rollout of env 0: raw inputs, action and the state they come from
        robot, cmd = base.scene['robot'], base.command_manager
        seed_all(seed)
        env.set_seed(seed)
        carry = env.reset()
        rec = {k: [] for k in ('action', 'rollout_action', 'joint_pos', 'joint_vel', 'root_quat_w', 'root_pos_w', 'root_ang_vel_b',
                               'ref_t', 'ref_root_quat_w', 'ref_root_pos_w', 'applied_torque', 'ankle_pos_w',
                               'obj_pos_w', 'obj_quat_w', 'obj_lin_vel_w')}
        ankles = [robot.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        obj = getattr(cmd, 'object', None)
        rec.update({f'in/{k}': [] for k in policy.in_keys})
        rollout = agent.get_rollout_policy('eval')
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for step in range(n_golden):
                td0 = carry[0:1].cpu().clone()
                # the env normalizes its observations (vecnorm=eval); the exported graph takes raw ones
                for k_in, k_out in zip(obs_norm.in_keys, obs_norm.out_keys):
                    td0.set(k_in, td0[k_out] * obs_norm.scale[k_out].cpu() + obs_norm.loc[k_in].cpu())
                for k in policy.in_keys:
                    rec[f'in/{k}'].append(td0[k][0].float().numpy().copy())
                rec['joint_pos'].append(robot.data.joint_pos[0].cpu().numpy())
                rec['joint_vel'].append(robot.data.joint_vel[0].cpu().numpy())
                rec['root_quat_w'].append(robot.data.root_link_quat_w[0].cpu().numpy())
                rec['root_pos_w'].append((robot.data.root_link_pos_w[0] - base.scene.env_origins[0]).cpu().numpy())
                rec['root_ang_vel_b'].append(robot.data.root_ang_vel_b[0].cpu().numpy())
                rec['applied_torque'].append(robot.data.applied_torque[0].cpu().numpy())
                rec['ankle_pos_w'].append((robot.data.body_pos_w[0, ankles] - base.scene.env_origins[0]).cpu().numpy())
                if obj is not None:
                    rec['obj_pos_w'].append((obj.data.root_pos_w[0] - base.scene.env_origins[0]).cpu().numpy())
                    rec['obj_quat_w'].append(obj.data.root_quat_w[0].cpu().numpy())
                    rec['obj_lin_vel_w'].append(obj.data.root_lin_vel_w[0].cpu().numpy())
                rec['ref_t'].append(int(cmd.t[0]))
                rec['ref_root_quat_w'].append(cmd.ref_root_quat_w[0].cpu().numpy())
                rec['ref_root_pos_w'].append((cmd.ref_root_pos_w[0] - base.scene.env_origins[0]).cpu().numpy())
                exported = policy(td0.select(*policy.in_keys).clone())
                rec['action'].append(exported['action'][0].numpy().copy())
                carry = rollout(carry)
                rec['rollout_action'].append(carry['action'][0].cpu().numpy().copy())
                td, carry = env.step_and_maybe_reset(carry)
                if bool(td['next', 'done'][0]):
                    break
        norm = {f'norm_loc/{k}': obs_norm.loc[k].cpu().numpy() for k in obs_norm.in_keys}
        norm.update({f'norm_scale/{k}': obs_norm.scale[k].cpu().numpy() for k in obs_norm.out_keys})
        # physical parameters of env 0 (its object scale is always 1: spawn.py randomizes envs 1..N-1 only)
        for k in ('mass', 'static_friction', 'dyn_friction'):
            v = getattr(obj, f'_custom_body_{k}', None) if obj is not None else None
            if v is not None:
                norm[f'obj_{k}'] = v[0].cpu().numpy()
        if obj is not None:
            norm['obj_body_masses'] = obj.root_physx_view.get_masses()[0].cpu().numpy()
        norm['robot_body_masses'] = robot.root_physx_view.get_masses()[0].cpu().numpy()
        np.savez_compressed(out / 'golden.npz', **{k: np.asarray(v) for k, v in rec.items()}, **norm,
                            joint_names=np.array(list(robot.joint_names)))
        print(f'EXPORT_RESULT {json.dumps({"out": str(out), "in_keys": [str(k) for k in policy.in_keys], "golden_steps": len(rec["action"])})}',
              flush=True)
        sys.stdout.flush()
        os._exit(0)  # Isaac Sim teardown can hang
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
