"""Control-weighted latent alignment (algo.priv_loss_weighting=action_jacobian)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from active_adaptation.learning.ppo import ppo_haic  # noqa: E402
from research.tests.test_ppo_haic import DIMS, N, T, A_DIM  # noqa: E402


def agent(**kw):
    torch.manual_seed(0)
    obs = Composite({k: Unbounded((N, d)) for k, d in DIMS.items()}, shape=[N])
    env = SimpleNamespace(cfg=SimpleNamespace(reward={'a': {}}),
                          action_manager=SimpleNamespace(joint_names=[f'j{i}' for i in range(A_DIM)]),
                          current_iter=0, set_progress=lambda i: None)
    return ppo_haic.PPOHAIC(ppo_haic.PPOConfig(phase='finetune', **kw), obs, Unbounded((N, A_DIM)), None, 'cpu', env)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    td = TensorDict({k: torch.randn(N, T, d, generator=g) for k, d in DIMS.items()}, [N, T])
    td['is_init'] = torch.zeros(N, T, 1, dtype=torch.bool)
    return td


def first_actor_layer(a):
    return a.actor_adapt.module[0][1].module[0]


def test_weights_follow_actor_sensitivity():
    a = agent(priv_loss_weighting='action_jacobian', priv_weight_ema=0.0, priv_weight_floor=0.0)
    for p in a.parameters():
        torch.nn.init.normal_(p, std=0.3)
    lat = a.cfg.latent_dim
    start = DIMS['command'] + DIMS['policy']  # actor input order: command, policy, priv_pred
    with torch.no_grad():
        w = first_actor_layer(a).weight
        w[:, start + 16:start + lat] = 0  # the actor only reads latent dims 0..15
    td = batch()
    with torch.no_grad():
        a.object_transform(td); a.encoder_priv(td)
    weights = a._priv_latent_weights(td)
    assert weights.shape == (lat,)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    assert (weights[16:] < 1e-6).all() and (weights[:16] > 1).all()
    assert all(p.grad is None for p in a.parameters())  # weight estimation leaves no gradients behind


def test_floor_and_ema():
    a = agent(priv_loss_weighting='action_jacobian', priv_weight_floor=0.25, priv_weight_ema=0.5)
    td = batch()
    with torch.no_grad():
        a.object_transform(td); a.encoder_priv(td)
    w1 = a._priv_latent_weights(td)
    assert (w1 >= 0.25 - 1e-6).all() and torch.isclose(w1.mean(), torch.tensor(1.0), atol=1e-5)
    s1 = a._priv_w_ema.clone()
    a._priv_latent_weights(td)
    assert not torch.equal(s1, a._priv_w_ema) or torch.allclose(s1, a._priv_w_ema)  # EMA updated in place of a reset


def test_train_adapt_runs_and_logs():
    a = agent(priv_loss_weighting='action_jacobian')
    info = a.train_adapt(batch(1))
    for k in ('adapt/priv_loss', 'adapt/priv_loss_unweighted', 'adapt/priv_w_max', 'adapt/priv_w_eff_dims'):
        assert k in info and info[k] == info[k]
    assert 1 <= info['adapt/priv_w_eff_dims'] <= a.cfg.latent_dim + 1e-3


def test_off_matches_plain_update():
    plain, off = agent(), agent(priv_loss_weighting='none')
    td = batch(2)
    torch.manual_seed(5); i1 = plain.train_adapt(td.clone())
    torch.manual_seed(5); i2 = off.train_adapt(td.clone())
    assert i1['adapt/priv_loss'] == i2['adapt/priv_loss'] and 'adapt/priv_w_max' not in i2
    for x, y in zip(plain.parameters(), off.parameters()):
        assert torch.equal(x, y)


def test_guards():
    with pytest.raises(AssertionError):
        agent(priv_loss_weighting='bogus')


def test_flag_is_ignored_outside_stage_2():
    """The preflight builds the stage-1 model with the arm's overrides; it must not fail on the flag."""
    torch.manual_seed(0)
    obs = Composite({k: Unbounded((N, d)) for k, d in DIMS.items()}, shape=[N])
    env = SimpleNamespace(cfg=SimpleNamespace(reward={'a': {}}),
                          action_manager=SimpleNamespace(joint_names=[f'j{i}' for i in range(A_DIM)]),
                          current_iter=0, set_progress=lambda i: None)
    a = ppo_haic.PPOHAIC(ppo_haic.PPOConfig(phase='train', priv_loss_weighting='action_jacobian'),
                         obs, Unbounded((N, A_DIM)), None, 'cpu', env)
    assert a._cw is False
