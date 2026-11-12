"""Shared helpers for the research pipeline (no Isaac Sim imports at module level)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get('HAIC_OUT', '/root/autodl-tmp/haic/runs'))
TRAIN_EVERY = 32
LOAD_FAILURE = 'Failed to load state dict'
# log lines that mean the run is broken even if the process exits 0 (e.g. Isaac Sim could not
# allocate the physics scene because another process holds the GPU memory)
FAILURE_MARKERS = (LOAD_FAILURE, 'Failed to create simulation view')
POSE_TAIL = ('object_pos_b', 'object_ori_b')
DYN_FIELDS = ('object_com_vel_b', 'object_com_acc_b')

# Arms. 'geo' is the released config (pose + geometry, the paper's Geo-Pose ablation),
# 'dyn' is full HAIC (pose + velocity + acceleration + geometry). Arms with 'teacher_from'
# train no teacher: their students start from that arm's stage-1 checkpoint (same teacher,
# same initialization; new heading input columns start at zero), so they are paired with
# that arm's students.
#   dyn-hwm / dyn-hact / dyn-hboth  heading -> world model / student actor / both
#   dyn-hnull                       same network as dyn-hboth, constant heading (capacity control)
#   dyn-hboth-cw / dyn-hwm-cw       + control-weighted latent alignment (algo.priv_loss_weighting)
HEADING_SENSOR = 'task.observation.heading_.robot_heading_err_imu.seed'
# nominal error model of the heading sensor (must match the *_heading.yaml configs; tested)
HEADING_NOISE_DEFAULTS = {'init_align_std': 0.01, 'bias_std': 0.002, 'bias_rw_std': 0.0005,
                          'rate_noise_std': 0.005}
CW = ['algo.priv_loss_weighting=action_jacobian']


def _arms(geo, dyn):
    heading, null = f'{dyn}_heading', f'{dyn}_heading_null'
    h = lambda task, to, extra=(): {'task': task, 'base': 'dyn', 'teacher_from': 'dyn', 'heading': True,
                                    'algo': ['algo.heading_key=heading_', f'algo.heading_to={to}', *extra]}
    return {
        'geo': {'task': geo, 'base': 'geo'},
        'dyn': {'task': dyn, 'base': 'dyn'},
        'dyn-hwm': h(heading, 'adapt'),
        'dyn-hact': h(heading, 'actor'),
        'dyn-hboth': h(heading, 'both'),
        'dyn-hnull': h(null, 'both'),
        'dyn-hboth-cw': h(heading, 'both', CW),
        'dyn-hwm-cw': h(heading, 'adapt', CW),
    }


def _cr_arms(dyn):
    """Contact-and-rolling randomized fine-tuning (only the student; teacher reused from dyn):
    dyn-cr is HAIC and dyn-hboth-cr is OH-Both, both fine-tuned under wider board physics
    (cfg/task/<dyn>_cr.yaml, <dyn>_heading_cr.yaml), so the pair stays comparable."""
    return {
        'dyn-cr': {'task': f'{dyn}_cr', 'base': 'dyn', 'teacher_from': 'dyn', 'cr': True},
        'dyn-hboth-cr': {'task': f'{dyn}_heading_cr', 'base': 'dyn', 'teacher_from': 'dyn', 'heading': True, 'cr': True,
                         'algo': ['algo.heading_key=heading_', 'algo.heading_to=both']},
    }


TASKS = {
    'skateboard': {'data': 'data/motion/g1/mirobotA/board6',
                   'arms': {**_arms('G1/haic/skateboard', 'G1/haic/skateboard_dyn'),
                            **_cr_arms('G1/haic/skateboard_dyn')}},
    'push_cart': {'data': 'data/motion/g1/mirobotA/push_cart',
                  'arms': _arms('G1/haic/push_cart', 'G1/haic/push_cart_dyn')},
    # carry a box over a stair + slope terrain (the paper's hardest multi-terrain task)
    'box_stairslope': {'data': 'data/motion/g1/mirobotA/box_stairslope',
                       'arms': _arms('G1/haic/box_stairslope', 'G1/haic/box_stairslope_dyn')},
}


# Privileged-input groups for the counterfactual ablation (research/ablate.py). Each entry is
# (observation group, term key); '*' means the whole group. Only the teacher's inputs are
# listed: priv and object_ (object_ also drives the projected object geometry).
# Frames: diff_body_*_local compare reference and robot each in its own root yaw frame, so they
# carry no global drift; ref_root_*_future_b give the reference root relative to the robot,
# i.e. exactly the global drift; diff_object_*_future compare the object with its reference
# trajectory, i.e. the object's global drift.
ABLATION_GROUPS = {
    'global_root': [('priv', 'ref_root_pos_future_b'), ('priv', 'ref_root_ori_future_b')],
    'global_root_pos': [('priv', 'ref_root_pos_future_b')],
    'global_root_ori': [('priv', 'ref_root_ori_future_b')],
    'local_tracking': [('priv', k) for k in ('diff_body_pos_future_local', 'diff_body_ori_future_local',
                                             'diff_body_lin_vel_future_local', 'diff_body_ang_vel_future_local')],
    'root_linvel': [('priv', 'root_linvel_b')],
    'feet': [('priv', k) for k in ('body_pos_b', 'body_vel_b', 'body_height')],
    'actuation': [('priv', 'applied_action'), ('priv', 'applied_torque')],
    'proprio_history': [('priv', k) for k in ('root_ang_vel_history', 'projected_gravity_history',
                                              'joint_pos_history')],
    'object_relative': [('priv', 'object_pos_b'), ('priv', 'object_ori_b'), ('object_', '*')],
    'object_drift': [('priv', 'diff_object_pos_future'), ('priv', 'diff_object_ori_future')],
    'contact': [('priv', 'ref_object_contact_future'), ('priv', 'diff_contact_pos_b')],
}


def arm_overrides(spec: dict, seed: int) -> list[str]:
    """Hydra overrides an arm needs in every training and evaluation process."""
    out = list(spec.get('algo', []))
    if spec.get('heading'):
        out.append(f'{HEADING_SENSOR}={seed}')
    return out


def seed_all(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        p = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else 'unavailable'
    return {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty_files': [l for l in run('status', '--porcelain').splitlines() if l]}


def installed_versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for name in ('torch', 'torchrl', 'tensordict', 'isaaclab', 'isaacsim-kernel', 'numpy', 'hydra-core'):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = 'unavailable'
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def execute(cmd, job: Path, env=None, timeout=None) -> str:
    """Run a subprocess, tee its output to job/run.log, fail loudly."""
    job.mkdir(parents=True, exist_ok=True)
    write_json(job / 'command.json', {'cmd': [str(c) for c in cmd]})
    log_path = job / 'run.log'
    env = dict(os.environ if env is None else env)
    env.setdefault('HYDRA_FULL_ERROR', '1')
    env.setdefault('PYTHONUNBUFFERED', '1')
    start = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] START {job}  (log: {log_path})', flush=True)
    last_report = start
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                now = time.time()
                if timeout is not None and now - start > timeout:
                    proc.kill()
                    raise RuntimeError(f'{job} exceeded {timeout} s')
                if now - last_report > 900:
                    last_report = now
                    print(f'    still running {job.name}: {(now - start) / 3600:.2f} h', flush=True)
    text = log_path.read_text(encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        print('\n'.join(text.splitlines()[-60:]), flush=True)
        raise RuntimeError(f'{job} failed with exit code {proc.returncode}; see {log_path}')
    for marker in FAILURE_MARKERS:
        if marker in text:
            raise RuntimeError(f'{job}: "{marker}" in {log_path}. If another training run is using the GPU, '
                               f'wait for it to finish (Isaac Sim needs free GPU memory).')
    print(f'[{time.strftime("%H:%M:%S")}] DONE  {job.name}  {(time.time() - start) / 3600:.2f} h', flush=True)
    return text


def checkpoint_info(path) -> dict:
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if 'vecnorm' not in state:
        raise RuntimeError(f'{path} has no VecNorm state')
    policy = state['policy']
    return {'path': str(path), 'sha256': sha256(path),
            'last_phase': policy.get('last_phase'), 'last_iter': policy.get('last_iter')}


def object_layout(base_env, group='object_') -> list[dict]:
    """Field names/slices of an observation group, in concatenation order."""
    funcs = base_env.observation_funcs[group].funcs
    layout, start = [], 0
    for key, f in funcs.items():
        dim = f.compute().reshape(base_env.num_envs, -1).shape[-1]
        layout.append({'name': type(f).__name__, 'key': key, 'start': start, 'dim': dim})
        start += dim
    return layout


def check_pose_tail(layout: list[dict], n_objects: int = 1) -> bool:
    """TransformObject reads (pos 3, rot 9) blocks from the end of object_."""
    tail = layout[-2 * n_objects:]
    if len(tail) != 2 * n_objects:
        return False
    for i in range(n_objects):
        pos, ori = tail[2 * i], tail[2 * i + 1]
        if pos['dim'] != 3 or ori['dim'] != 9:
            return False
        if not (pos['name'].endswith('pos_b') and ori['name'].endswith('ori_b')):
            return False
    return True


def python() -> str:
    return sys.executable
