"""Hydra composition of every arm, with the overrides the pipeline passes."""
import os
import sys
from pathlib import Path

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import active_adaptation.learning  # noqa: E402,F401  registers algo configs
from research.common import TASKS, arm_overrides, DYN_FIELDS  # noqa: E402

CFG = os.path.join(str(REPO), 'cfg')


def comp(name, ov):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        return compose(config_name=name, overrides=ov)


def task_dict(task, arm, cfgname, extra):
    spec = TASKS[task]['arms'][arm]
    c = comp(cfgname, [f'task={spec["task"]}', 'task.command.data_path=/d', *arm_overrides(spec, 7), *extra])
    t = OmegaConf.to_container(c.task)
    t.pop('name'); t.pop('num_envs')
    return c, t


CASES = [(t, a) for t in sorted(TASKS) for a in sorted(TASKS[t]['arms'])]


@pytest.mark.parametrize('task,arm', CASES)
@pytest.mark.parametrize('cfgname', ['train', 'eval'])
def test_arm_composes(task, arm, cfgname):
    extra = ['algo=ppo_haic_finetune', 'checkpoint_path=/x.pt']
    if cfgname == 'eval':
        extra += ['vecnorm=eval', 'algo.load_noise_scale=null', '+out_dir=/o', '+mode=pred_dt', '+trace=false']
    spec = TASKS[task]['arms'][arm]
    c, t = task_dict(task, arm, cfgname, extra)
    obj = list(t['observation']['object_'])
    n_obj = len(t['observation']['object_geo_'])
    tail = obj[-2 * n_obj:]
    assert tail[0::2] == [k.replace('ori_b', 'pos_b') for k in tail[1::2]] and all(k.endswith('ori_b') for k in tail[1::2])
    assert tail[:2] == ['object_pos_b', 'object_ori_b']
    assert set(DYN_FIELDS) <= set(obj) if spec['base'] == 'dyn' else not set(DYN_FIELDS) & set(obj)
    if spec.get('heading'):
        h = t['observation'].pop('heading_')['robot_heading_err_imu']
        assert h['seed'] == 7 and h['zero_info'] == (arm == 'dyn-hnull')
        assert c.algo.heading_key == 'heading_'
        _, dyn = task_dict(task, 'dyn-cr' if spec.get('cr') else 'dyn', cfgname, extra)
        assert t == dyn  # heading arms differ from dyn (or dyn-cr) only by the heading_ group
    else:
        assert 'heading_' not in t['observation'] and c.algo.heading_key is None
    if spec.get('cr'):  # contact-and-rolling arms differ from dyn only by the board randomization
        _, dyn = task_dict(task, 'dyn', cfgname, extra)
        t.get('observation', {}).pop('heading_', None)
        r, r0 = t.pop('randomization'), dyn.pop('randomization')
        assert t == dyn
        j = r['object_joint_randomization']
        assert j['all_joints'] and j['damping_range'][1] >= 0.004 and j['friction_range'][1] >= 0.01
        assert r['object_body_randomization']['mass_range'] == [1.0, 4.0]
        assert {k: v for k, v in r.items() if k not in ('object_joint_randomization', 'object_body_randomization')} == \
            {k: v for k, v in r0.items() if k not in ('object_joint_randomization', 'object_body_randomization')}
        if spec.get('heading'):  # the repeated heading_ group equals the one of the plain heading arm
            _, ref = task_dict(task, 'dyn-hboth', cfgname, extra)
            assert h == ref['observation']['heading_']['robot_heading_err_imu']


def test_cw_arms_set_weighting():
    for task in TASKS:
        for arm, spec in TASKS[task]['arms'].items():
            c, _ = task_dict(task, arm, 'train', ['algo=ppo_haic_finetune', 'checkpoint_path=/x.pt'])
            assert c.algo.priv_loss_weighting == ('action_jacobian' if arm.endswith('-cw') else 'none')


@pytest.mark.parametrize('task', sorted(TASKS))
def test_heading_noise_defaults_match_configs(task):
    from research.common import HEADING_NOISE_DEFAULTS
    c, t = task_dict(task, 'dyn-hboth', 'train', ['algo=ppo_haic_finetune', 'checkpoint_path=/x.pt'])
    h = t['observation']['heading_']['robot_heading_err_imu']
    assert {k: h[k] for k in HEADING_NOISE_DEFAULTS} == HEADING_NOISE_DEFAULTS


def test_dyn_acceleration_is_control_rate_everywhere():
    for task in TASKS:
        _, t = task_dict(task, 'dyn', 'train', ['algo=ppo_haic_train'])
        assert t['observation']['object_']['object_com_acc_b'] == {'source': 'control_fd', 'clip': 50.0, 'scale': 0.1}


def test_no_null_groups_in_any_task_config():
    """An empty group key (e.g. 'tracking:' with all entries commented out) composes to null and
    crashes environment construction; upstream box_stairslope.yaml had one."""
    import glob
    for f in sorted(glob.glob(os.path.join(CFG, 'task', 'G1', 'haic', '*.yaml'))):
        task = 'G1/haic/' + os.path.basename(f)[:-5]
        c = comp('train', [f'task={task}', 'algo=ppo_haic_train'])
        for sec in ('reward', 'observation', 'termination', 'randomization'):
            for g, v in (c.task.get(sec) or {}).items():
                assert v is not None, f'{task}: {sec}.{g} is null'
