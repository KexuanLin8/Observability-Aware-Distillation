"""First-episode evaluation with optional per-step mechanism traces.

Protocol (same as the earlier 35.09% / 83.11% evaluations): RNGs seeded before the
environment is built, eval mode (episodes start at reference frame 0, no reset pose
noise), deterministic actions (distribution mode), only the first episode of every
environment counts. Success = the reference's last frame was reached; strict success
additionally requires that no termination flag fired on that step.

mode=pred            normal student (the deployable policy)
mode=oracle_object   object_pred := true object_ (perfect object adapter), counterfactual
mode=oracle_priv     priv_pred   := teacher's priv_feature (perfect world model), counterfactual
mode=pred_dt         normal student, drift-tolerant success (global torso terminations disabled)
mode=teacher_ablate      teacher (stage-1 checkpoint) with one privileged-input group (+ablate=<group>,
                         see research/common.py ABLATION_GROUPS) replaced by another environment's values
mode=oracle_priv_ablate  oracle_priv student whose teacher feature is computed with that group replaced
The replacement is a fixed derangement of the environments drawn once per run: every environment
reads the whole group from one other environment at the same time step (eval episodes are
time-aligned), so the group stays internally consistent and in distribution but carries no
information about the environment itself.

Outputs in +out_dir: episodes.csv, summary.json, wm_stats.json, trace.npz (if +trace=true).
The extra network passes used for tracing do not consume RNG and do not feed the policy.
"""
import csv
import json
import math
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

from research.common import (REPO, seed_all, sha256, git_state, installed_versions,
                             object_layout, check_pose_tail, write_json, ABLATION_GROUPS)

MODES = ('pred', 'pred_dt', 'oracle_object', 'oracle_priv', 'teacher_ablate', 'oracle_priv_ablate')
ABLATE_MODES = ('teacher_ablate', 'oracle_priv_ablate')
ABLATE_SEED_OFFSET = 12345
# pred_dt: drift-tolerant evaluation; every world-frame tracking termination is disabled (torso and
# object position / orientation w.r.t. their reference trajectories; the object terms are already
# inactive in the skateboard and cart configs), so an episode fails only by losing local posture or
# object contact (a dropped box ends the episode through cum_lost_contact_steps).
GLOBAL_TERMINATIONS = ('cum_body_pos_error', 'cum_body_ori_error', 'cum_object_pos_error', 'cum_object_ori_error')


def yaw_of_quat(q):  # [..., 4] wxyz
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def rot_angle(R_true, R_pred):
    """Geodesic angle between R_true and the closest rotation to R_pred."""
    U, _, Vh = torch.linalg.svd(R_pred)
    d = torch.sign(torch.linalg.det(U @ Vh))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    Rp = U @ D @ Vh
    c = ((R_true.transpose(-1, -2) @ Rp).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(c.clamp(-1, 1))


def derangement(n, seed, device):
    """Random permutation without fixed points (cyclic shift of a random order)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    order = torch.randperm(n, generator=g, device=device)
    perm = torch.empty_like(order)
    perm[order] = order.roll(-1)
    return perm


def resolve_ablation(base, group_name):
    """[(obs group, term key, slice)] of an ablation group in this environment."""
    if group_name not in ABLATION_GROUPS:
        raise ValueError(f'unknown ablation group {group_name}; choose from {sorted(ABLATION_GROUPS)}')
    out = []
    for group, key in ABLATION_GROUPS[group_name]:
        if group not in base.observation_funcs:
            continue
        layout = object_layout(base, group)
        if key == '*':
            out.append((group, '*', slice(0, sum(f['dim'] for f in layout))))
        else:
            out += [(group, f['key'], slice(f['start'], f['start'] + f['dim'])) for f in layout if f['key'] == key]
    if not out:
        raise RuntimeError(f'ablation group {group_name} matches no observation term in this task')
    return out


def make_policy(agent, mode, ablation=None, perm=None):
    from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq

    class PermuteSlices(TensorDictModuleBase):
        def __init__(self, ablation, perm):
            super().__init__()
            self.ablation, self.perm = ablation, perm
            self.in_keys = self.out_keys = sorted({g for g, _, _ in ablation})

        def forward(self, td):
            for key in self.in_keys:
                x = td.get(key).clone()
                for g, _, sl in self.ablation:
                    if g == key:
                        x[:, sl] = x[self.perm, sl]
                td.set(key, x)
            return td

    class CopyKey(TensorDictModuleBase):
        def __init__(self, src, dst):
            super().__init__()
            self.in_keys, self.out_keys = [src], [dst]

        def forward(self, td):
            td.set(self.out_keys[0], td.get(self.in_keys[0]).clone())
            return td

    if mode in ('pred', 'pred_dt'):
        return agent.get_rollout_policy('eval')
    out_keys = ['sample_log_prob', 'action'] + list(agent.dist_keys)
    if mode == 'teacher_ablate':
        assert agent.cfg.phase == 'train', 'teacher_ablate needs a stage-1 (teacher) checkpoint'
        return Seq(PermuteSlices(ablation, perm), agent.object_transform, agent.encoder_priv, agent.actor,
                   selected_out_keys=out_keys)
    assert agent.cfg.phase == 'finetune', 'oracle modes are defined for student checkpoints'
    assert agent.cfg.adapt_module == 'mlp', 'oracle modes assume the MLP privilege adapter'
    mods = [agent.object_adapt_ema]
    if mode == 'oracle_object':
        mods.append(CopyKey('object_', 'object_pred'))
    mods += [agent.object_pred_transform, agent.adapt_ema]
    if mode == 'oracle_priv_ablate':
        mods.append(PermuteSlices(ablation, perm))
    if mode in ('oracle_priv', 'oracle_priv_ablate'):
        mods += [agent.object_transform, agent.encoder_priv, CopyKey('priv_feature', 'priv_pred')]
    mods.append(agent.actor_adapt)
    return Seq(*mods, selected_out_keys=out_keys)


class WMStats:
    """Pooled per-dimension statistics of object_pred vs object_ over alive steps."""

    def __init__(self, dim, device):
        z = lambda: torch.zeros(dim, dtype=torch.float64, device=device)
        self.n = 0
        self.sy, self.syy, self.see, self.sp, self.spy = z(), z(), z(), z(), z()

    def add(self, pred, true, mask):
        p, y = pred[mask].double(), true[mask].double()
        self.n += p.shape[0]
        self.sy += y.sum(0); self.syy += (y * y).sum(0)
        self.see += ((p - y) ** 2).sum(0)
        self.sp += p.sum(0); self.spy += (p * y).sum(0)

    def result(self, layout):
        n = max(self.n, 1)
        var = (self.syy / n - (self.sy / n) ** 2).clamp_min(1e-12)
        mse = self.see / n
        out = {'n_samples': self.n, 'fields': {}}
        for f in layout:
            sl = slice(f['start'], f['start'] + f['dim'])
            out['fields'][f['key']] = {
                'mse_per_dim': mse[sl].tolist(), 'var_per_dim': var[sl].tolist(),
                'r2_per_dim': (1 - mse[sl] / var[sl]).tolist(),
                'r2_pooled': float(1 - mse[sl].sum() / var[sl].sum()),
                'rmse': float(mse[sl].mean().sqrt()),
            }
        return out


@hydra.main(config_path=str(REPO / 'cfg'), config_name='eval', version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    out = Path(cfg.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get('mode', 'pred'))
    trace = bool(cfg.get('trace', True))
    assert mode in MODES, mode
    ablate = cfg.get('ablate', None)
    assert (ablate is not None) == (mode in ABLATE_MODES), 'set +ablate=<group> exactly for the ablation modes'
    ckpt = Path(cfg.checkpoint_path).resolve()
    assert ckpt.is_file(), f'Missing checkpoint: {ckpt}'
    assert cfg.vecnorm == 'eval' and not cfg.eval_render
    assert not cfg.task.command.get('replay_motion', False)
    seed = int(cfg.seed)
    meta = {'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt), 'mode': mode, 'ablate': ablate,
            'seed': seed, 'task': cfg.task.name, 'algo_phase': cfg.algo.phase,
            'git': git_state(), 'versions': installed_versions(),
            'script_sha256': sha256(__file__)}
    OmegaConf.save(cfg, out / 'resolved_cfg.yaml')

    from isaaclab.app import AppLauncher
    seed_all(seed)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    env = None
    try:
        from helpers import make_env_policy
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from isaaclab.utils.math import quat_apply_inverse

        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        meta['checkpoint_last_phase'] = state['policy'].get('last_phase')
        meta['checkpoint_last_iter'] = state['policy'].get('last_iter')
        del state
        expected = 'finetune' if cfg.algo.phase == 'finetune' else 'train'
        if meta['checkpoint_last_phase'] != expected:
            raise RuntimeError(f"checkpoint phase {meta['checkpoint_last_phase']} != {expected}")

        if mode == 'pred_dt':
            disabled = [k for k in GLOBAL_TERMINATIONS if k in cfg.task.termination]
            for k in disabled:
                cfg.task.termination[k].threshold = 1e9
            meta['disabled_terminations'] = disabled
        seed_all(seed)  # before creation-time domain randomization
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            env, agent, _ = make_env_policy(cfg)
        bad = [str(w.message) for w in caught if 'Failed to load state dict' in str(w.message)]
        for w in caught:
            print(f'WARNING: {w.message}', flush=True)
        if bad:
            raise RuntimeError('Incomplete model load: ' + '; '.join(bad))

        base = env.base_env
        base.eval(); env.eval(); agent.eval()
        seed_all(seed)
        env.set_seed(seed)
        carry = env.reset()
        cmd = base.command_manager
        n = env.num_envs
        dev = env.device
        T = int(env.max_episode_length)
        layout = object_layout(base, 'object_')
        n_obj = int(carry['object_geo_'].shape[-1]) // (128 * 3)
        if not check_pose_tail(layout, n_obj):
            raise RuntimeError(f'object_ pose tail check failed: {layout}')
        pos_sl = slice(layout[-2]['start'], layout[-2]['start'] + 3) if n_obj == 1 else None
        ori_sl = slice(layout[-1]['start'], layout[-1]['start'] + 9) if n_obj == 1 else None
        if n_obj != 1:  # multi-object: pose of the primary object is the first pose block
            first = len(layout) - 2 * n_obj
            pos_sl = slice(layout[first]['start'], layout[first]['start'] + 3)
            ori_sl = slice(layout[first + 1]['start'], layout[first + 1]['start'] + 9)
        meta.update({'object_layout': layout, 'n_objects': n_obj, 'num_envs': n,
                     'max_episode_length': T, 'step_dt': float(base.step_dt),
                     'initial_frame_t': cmd.t[:8].tolist(),
                     'reference_motion_lengths_steps': cmd.dataset.lengths.cpu().tolist()})
        term_keys = [k for k, f in base.termination_funcs.items() if hasattr(f, 'error')]

        ablation = perm = None
        if ablate is not None:
            ablation = resolve_ablation(base, ablate)
            perm = derangement(n, seed + ABLATE_SEED_OFFSET, dev)
            meta['ablation_terms'] = [{'group': g, 'key': k, 'start': sl.start, 'stop': sl.stop}
                                      for g, k, sl in ablation]
            print('ABLATION ' + json.dumps(meta['ablation_terms']), flush=True)
        policy = make_policy(agent, mode, ablation, perm)
        probe_mods = [agent.object_transform, agent.encoder_priv, agent.object_adapt_ema,
                      agent.object_pred_transform, agent.adapt_ema]

        nan = float('nan')
        names = (['heading_err', 'root_err_along', 'root_err_lat', 'root_err_z',
                  'obj_pos_err_w', 'obj_yaw_err_w', 'wm_pos_err', 'wm_ori_err', 'wm_yaw_err',
                  'priv_err', 'priv_norm', 'frame']
                 + [f'term_err/{k}' for k in term_keys]
                 + [f'wm_field_err/{f["key"]}' for f in layout])
        vec_names = ['pos_true_b', 'pos_pred_b', 'pos_ref_b']
        buf = {k: torch.full((T, n), nan, device=dev) for k in names} if trace else {}
        vbuf = {k: torch.full((T, n, 3), nan, device=dev) for k in vec_names} if trace else {}
        ybuf = {k: torch.full((T, n), nan, device=dev) for k in ('yaw_true_b', 'yaw_pred_b', 'yaw_ref_b')} if trace else {}
        wm = WMStats(int(carry['object_'].shape[-1]), dev)

        rows = [None] * n
        seen = torch.zeros(n, dtype=torch.bool, device=dev)
        last_step = 0
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for step in range(T):
                alive = ~seen
                probe = carry.clone()
                for m in probe_mods:
                    m(probe)
                true_o, pred_o = probe['object_'], probe['object_pred']
                if step > 0:
                    wm.add(pred_o, true_o, alive)
                if trace:
                    def put(k, v):
                        buf[k][step] = torch.where(alive, v.float(), torch.full_like(v.float(), nan))
                    ref_yaw = yaw_of_quat(cmd.ref_root_quat_w)
                    put('heading_err', wrap(yaw_of_quat(cmd.robot_root_quat_w) - ref_yaw))
                    d = cmd.robot_root_pos_w - cmd.ref_root_pos_w
                    c, s = torch.cos(ref_yaw), torch.sin(ref_yaw)
                    put('root_err_along', c * d[:, 0] + s * d[:, 1])
                    put('root_err_lat', -s * d[:, 0] + c * d[:, 1])
                    put('root_err_z', d[:, 2])
                    put('obj_pos_err_w', (cmd.object_pos_w - cmd.ref_object_pos_w)[:, :2].norm(dim=-1))
                    put('obj_yaw_err_w', wrap(yaw_of_quat(cmd.object_quat_w) - yaw_of_quat(cmd.ref_object_quat_w)))
                    p_t, p_p = true_o[:, pos_sl], pred_o[:, pos_sl]
                    R_t = true_o[:, ori_sl].reshape(n, 3, 3)
                    R_p = pred_o[:, ori_sl].reshape(n, 3, 3)
                    put('wm_pos_err', (p_p - p_t).norm(dim=-1))
                    put('wm_ori_err', rot_angle(R_t, R_p))
                    yt = torch.atan2(R_t[:, 1, 0], R_t[:, 0, 0])
                    yp = torch.atan2(R_p[:, 1, 0], R_p[:, 0, 0])
                    put('wm_yaw_err', wrap(yp - yt))
                    put('priv_err', (probe['priv_pred'] - probe['priv_feature']).norm(dim=-1))
                    put('priv_norm', probe['priv_feature'].norm(dim=-1))
                    put('frame', cmd.t.float())
                    for k in term_keys:
                        put(f'term_err/{k}', base.termination_funcs[k].error)
                    for f in layout:
                        sl = slice(f['start'], f['start'] + f['dim'])
                        put(f'wm_field_err/{f["key"]}', (pred_o[:, sl] - true_o[:, sl]).norm(dim=-1))
                    ref_b = quat_apply_inverse(cmd.robot_root_quat_w, cmd.ref_object_pos_w - cmd.robot_root_pos_w)
                    a3 = alive.unsqueeze(-1)
                    for k, v in (('pos_true_b', p_t), ('pos_pred_b', p_p), ('pos_ref_b', ref_b)):
                        vbuf[k][step] = torch.where(a3, v.float(), torch.full_like(v.float(), nan))
                    yr = wrap(yaw_of_quat(cmd.ref_object_quat_w) - yaw_of_quat(cmd.robot_root_quat_w))
                    for k, v in (('yaw_true_b', yt), ('yaw_pred_b', yp), ('yaw_ref_b', yr)):
                        ybuf[k][step] = torch.where(alive, v.float(), torch.full_like(v.float(), nan))

                frames, lengths, motion_ids = cmd.t.clone(), cmd.motion_len.clone(), cmd.motion_ids.clone()
                torch.compiler.cudagraph_mark_step_begin()
                carry = policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                nxt = td['next']
                done = nxt['done'].reshape(n).bool()
                new = done & ~seen
                if new.any():
                    ids = new.nonzero(as_tuple=False).flatten()
                    stats = nxt['stats'][ids].cpu()
                    term = nxt['terminated'][ids].reshape(-1).cpu().tolist()
                    trunc = nxt['truncated'][ids].reshape(-1).cpu().tolist()
                    f_, ml, mi = frames[ids].cpu().tolist(), lengths[ids].cpu().tolist(), motion_ids[ids].cpu().tolist()
                    for j, env_id in enumerate(ids.cpu().tolist()):
                        steps = int(stats['episode_len'][j].item())
                        row = {'env_id': env_id, 'success': int(stats['success'][j].item() > .5),
                               'episode_steps': steps, 'end_rollout_step': step,
                               'motion_id': mi[j], 'end_frame': f_[j], 'motion_len': ml[j],
                               'end_phase': f_[j] / max(1, ml[j] - 1),
                               'terminated': int(term[j]), 'truncated': int(trunc[j])}
                        flags = 0
                        for key, value in stats.items(True, True):
                            parts = key if isinstance(key, tuple) else (key,)
                            if parts[0] == 'termination':
                                v = int(float(value[j].item()) > .5)
                                row['term/' + '/'.join(parts[1:])] = v
                                flags += v
                        row['strict_success'] = int(row['success'] and flags == 0)
                        rows[env_id] = row
                    seen |= new
                last_step = step
                if step % 100 == 0 or seen.all():
                    print(f'ROLLOUT_PROGRESS step={step + 1} done={int(seen.sum())}/{n}', flush=True)
                if seen.all():
                    break

        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            raise RuntimeError(f'{len(missing)} first episodes did not finish within {T} steps')
        keys = sorted({k for r in rows for k in r})
        with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        n_succ = sum(r['success'] for r in rows)
        n_strict = sum(r['strict_success'] for r in rows)
        fails = [r for r in rows if not r['success']]
        causes = {k[5:]: sum(r.get(k, 0) for r in fails) for k in keys if k.startswith('term/')}
        summary = dict(meta, n_envs=n, n_success=n_succ, success_rate=n_succ / n,
                       n_strict_success=n_strict, strict_success_rate=n_strict / n,
                       failure_causes=causes, n_failures=len(fails),
                       mean_failure_time_s=(float(np.mean([r['episode_steps'] for r in fails])) * meta['step_dt']
                                            if fails else None))
        write_json(out / 'summary.json', summary)
        write_json(out / 'wm_stats.json', wm.result(layout))
        if trace:
            L = last_step + 1
            arrays = {k: v[:L].cpu().numpy() for k, v in buf.items()}
            arrays.update({k: v[:L].cpu().numpy() for k, v in vbuf.items()})
            arrays.update({k: v[:L].cpu().numpy() for k, v in ybuf.items()})
            np.savez_compressed(out / 'trace.npz', **arrays)
        print('ROLLOUT_RESULT ' + json.dumps({'mode': mode, 'seed': seed, 'n_success': n_succ, 'n_envs': n,
                                              'success_rate': n_succ / n, 'failure_causes': causes}), flush=True)
        sys.stdout.flush()
        os._exit(0)  # Isaac Sim teardown can hang after a completed rollout
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
