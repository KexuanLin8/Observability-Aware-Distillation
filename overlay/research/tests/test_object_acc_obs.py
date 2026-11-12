"""object_com_acc_b with source=control_fd: finite difference, reset handling, clip/scale.

The HDMI observation module imports Isaac Lab; the few names it needs are stubbed here.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Generic, TypeVar

import torch

REPO = Path(__file__).resolve().parents[2]


def quat_apply_inverse(q, v):  # wxyz, rotate v by q^-1
    w, xyz = q[..., :1], -q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def load_module():
    T = TypeVar('T')

    class Observation(Generic[T]):
        def __init__(self, env):
            self.env, self.command_manager = env, env.command_manager

        num_envs = property(lambda self: self.env.num_envs)
        device = property(lambda self: 'cpu')

        def reset(self, env_ids):
            pass

    stubs = {
        'isaaclab': types.ModuleType('isaaclab'), 'isaaclab.utils': types.ModuleType('isaaclab.utils'),
        'isaaclab.utils.math': types.SimpleNamespace(
            quat_apply_inverse=quat_apply_inverse, quat_apply=None, quat_mul=None, quat_conjugate=None,
            matrix_from_quat=None, yaw_quat=None,
            wrap_to_pi=lambda x: torch.remainder(x + math.pi, 2 * math.pi) - math.pi),
        'active_adaptation.envs.mdp.commands.hdmi.command': types.SimpleNamespace(
            RobotTracking=object, RobotObjectTracking=object),
        'active_adaptation.envs.mdp.base': types.SimpleNamespace(Observation=Observation),
        'active_adaptation.utils.math': types.SimpleNamespace(batchify=lambda f: f),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        path = REPO / 'active_adaptation/envs/mdp/commands/hdmi/observations.py'
        spec = importlib.util.spec_from_file_location('_hdmi_obs_under_test', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def make_env(n=3, dt=0.02):
    data = types.SimpleNamespace(root_com_vel_w=torch.zeros(n, 6), body_com_acc_w=torch.zeros(n, 1, 6))
    cmd = types.SimpleNamespace(object=types.SimpleNamespace(data=data),
                                robot_root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1))
    return types.SimpleNamespace(num_envs=n, step_dt=dt, command_manager=cmd)


def test_control_fd_acceleration():
    mod = load_module()
    env = make_env()
    obs = mod.object_com_acc_b(env=env, source='control_fd', clip=50.0, scale=0.1)
    d = env.command_manager.object.data
    obs.update()  # first step after construction: fresh -> zero
    assert torch.equal(obs.compute(), torch.zeros(3, 6))
    d.root_com_vel_w = torch.tensor([[0.2, 0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    obs.update()  # a = dv/dt = [10, 0, ..., 5], [0], [250 -> clipped 50]; then *0.1
    expect = torch.tensor([[1.0, 0, 0, 0, 0, 0.5], [0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    assert torch.allclose(obs.compute(), expect, atol=1e-5)
    obs.reset(torch.tensor([0]))  # velocity jump caused by a reset must not register as acceleration
    d.root_com_vel_w = torch.tensor([[9.0, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    obs.update()
    assert torch.allclose(obs.compute(), torch.tensor([[0., 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), atol=1e-5)
    d.root_com_vel_w = torch.tensor([[9.2, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    obs.update()  # env 0 differences against its post-reset velocity again
    assert torch.allclose(obs.compute()[0], torch.tensor([1.0, 0, 0, 0, 0, 0]), atol=1e-4)


def test_default_is_physx_passthrough():
    mod = load_module()
    env = make_env()
    obs = mod.object_com_acc_b(env=env)
    env.command_manager.object.data.body_com_acc_w = torch.full((3, 1, 6), 1234.0)
    obs.update()
    assert torch.equal(obs.compute(), torch.full((3, 6), 1234.0))


def yaw_quat_tensor(yaw):
    return torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)


def make_heading_env(robot_yaw, ref_yaw, dt=0.02):
    cmd = types.SimpleNamespace(robot_root_quat_w=yaw_quat_tensor(robot_yaw), ref_root_quat_w=yaw_quat_tensor(ref_yaw))
    return types.SimpleNamespace(num_envs=len(robot_yaw), step_dt=dt, command_manager=cmd)


def test_heading_sensor_noise_free_and_null():
    mod = load_module()
    env = make_heading_env(torch.tensor([0.3, -3.0, 1.0]), torch.tensor([0.1, 3.0, 1.0]))
    s = mod.robot_heading_err_imu(env=env, init_align_std=0, bias_std=0, bias_rw_std=0, rate_noise_std=0)
    s.reset(torch.arange(3))
    s.update()
    e = torch.tensor([0.2, -6.0 + 2 * math.pi, 0.0])  # wrapped
    assert torch.allclose(s.compute(), torch.stack([e.cos(), e.sin()], -1), atol=1e-5)
    n = mod.robot_heading_err_imu(env=env, zero_info=True)
    n.update()
    assert torch.equal(n.compute(), torch.tensor([[1.0, 0.0]] * 3))


def test_heading_sensor_bias_integrates_and_keeps_global_rng():
    mod = load_module()
    env = make_heading_env(torch.zeros(2000), torch.zeros(2000))
    s = mod.robot_heading_err_imu(env=env, init_align_std=0, bias_std=0.01, bias_rw_std=0, rate_noise_std=0, seed=5)
    state = torch.random.get_rng_state()
    s.reset(torch.arange(2000))
    for _ in range(500):  # 10 s
        s.update()
    assert torch.equal(state, torch.random.get_rng_state())  # dedicated generator only
    err = torch.atan2(s.compute()[:, 1], s.compute()[:, 0])
    assert torch.allclose(err, s.bias * 10.0, atol=1e-4)  # constant bias -> linear drift
    assert abs(err.std().item() - 0.1) < 0.01               # 0.01 rad/s * 10 s
    s.reset(torch.tensor([0]))
    assert s.err[0] == 0
