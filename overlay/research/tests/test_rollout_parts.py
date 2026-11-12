"""CPU tests for the simulator-independent parts of research/rollout.py."""
import math
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research import rollout as R  # noqa: E402
from research.tests.test_ppo_haic import make_agent, DIMS, N  # noqa: E402
from active_adaptation.learning.ppo import ppo_haic  # noqa: E402
from torchrl.envs.utils import set_exploration_type, ExplorationType  # noqa: E402


def rotz(a):
    R_ = torch.zeros(a.shape[0], 3, 3)
    R_[:, 0, 0], R_[:, 0, 1], R_[:, 1, 0], R_[:, 1, 1], R_[:, 2, 2] = a.cos(), -a.sin(), a.sin(), a.cos(), 1
    return R_


def test_angles():
    a = torch.tensor([0.1, -2.0, 3.0])
    q = torch.stack([torch.cos(a / 2), torch.zeros(3), torch.zeros(3), torch.sin(a / 2)], -1)
    assert torch.allclose(R.yaw_of_quat(q), a, atol=1e-6)
    assert torch.allclose(R.wrap(torch.tensor([3 * math.pi / 2, -3 * math.pi / 2])),
                          torch.tensor([-math.pi / 2, math.pi / 2]), atol=1e-6)
    ang = R.rot_angle(rotz(torch.zeros(3)), rotz(a) * 1.3)  # scaled: projected back to SO(3)
    assert torch.allclose(ang, a.abs(), atol=1e-5)


def carry():
    g = torch.Generator().manual_seed(0)
    td = TensorDict({k: torch.randn(N, d, generator=g) for k, d in DIMS.items()}, [N])
    td['is_init'] = torch.zeros(N, 1, dtype=torch.bool)
    return td


def test_policies():
    agent = make_agent(ppo_haic)
    torch.nn.init.normal_(agent.actor_adapt.module[0][2].module.actor_mean.weight)  # non-trivial outputs
    for p in agent.object_adapt_ema.parameters():
        torch.nn.init.normal_(p)
    td = carry()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        a_pred = R.make_policy(agent, 'pred')(td.clone())['action']
        a_obj = R.make_policy(agent, 'oracle_object')(td.clone())['action']
        a_priv = R.make_policy(agent, 'oracle_priv')(td.clone())['action']
        # manual oracle_object
        m = td.clone()
        m['object_pred'] = m['object_'].clone()
        agent.object_pred_transform(m); agent.adapt_ema(m); agent.actor_adapt(m)
    assert not torch.allclose(a_pred, a_obj)
    assert torch.allclose(a_obj, m['action'])
    assert a_priv.shape == a_pred.shape


def test_wm_stats():
    layout = [{'key': 'a', 'start': 0, 'dim': 2}, {'key': 'b', 'start': 2, 'dim': 1}]
    s = R.WMStats(3, 'cpu')
    y = torch.randn(1000, 3)
    p = y.clone(); p[:, 2] = 0
    s.add(p, y, torch.ones(1000, dtype=torch.bool))
    res = s.result(layout)['fields']
    assert abs(res['a']['r2_pooled'] - 1.0) < 1e-9
    assert res['b']['r2_pooled'] < 0.01
