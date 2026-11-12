"""CPU tests for the PPOHAIC world-model path (no simulator).

* the logging-only edit leaves world-model updates bit-identical to upstream (commit e262500)
* the object_loss_pose / object_loss_other split is consistent with object_loss
* TransformObject reads the final 12 dims as (pos, rot) and applies R @ p + t
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from active_adaptation.learning.ppo import ppo_haic  # noqa: E402

N, T, A_DIM = 64, 32, 5
DIMS = {'command': 7, 'policy': 9, 'priv': 6, 'object_': 22, 'object_geo_': 384, 'ref_joint_pos_': A_DIM}


def load_upstream():
    src = subprocess.run(['git', '-C', str(REPO), 'show', 'e262500:active_adaptation/learning/ppo/ppo_haic.py'],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip('upstream commit e262500 not available')
    path = Path(__file__).with_name('_upstream_ppo_haic.py')
    path.write_text(src.stdout, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('active_adaptation.learning.ppo._upstream_ppo_haic', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path.unlink()
    return mod


def make_agent(module, phase='finetune'):
    torch.manual_seed(0)
    obs = Composite({k: Unbounded((N, d)) for k, d in DIMS.items()}, shape=[N])
    act = Unbounded((N, A_DIM))  # the env passes a plain tensor spec
    env = SimpleNamespace(cfg=SimpleNamespace(reward={'tracking': {}, 'loco': {}}),
                          action_manager=SimpleNamespace(joint_names=[f'j{i}' for i in range(A_DIM)]),
                          current_iter=0, set_progress=lambda i: None)
    cfg = module.PPOConfig(phase=phase)
    return module.PPOHAIC(cfg, obs, act, None, 'cpu', env)


def make_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    td = TensorDict({k: torch.randn(N, T, d, generator=g) for k, d in DIMS.items()}, [N, T])
    td['is_init'] = torch.zeros(N, T, 1, dtype=torch.bool)
    td['is_init'][:, 0] = True
    return td


def test_world_model_update_matches_upstream():
    up = load_upstream()
    a_new, a_old = make_agent(ppo_haic), make_agent(up)
    # copy tensors directly: load_state_dict would apply load_noise_scale to actor_std
    with torch.no_grad():
        for x, y in zip([*a_new.parameters(), *a_new.buffers()], [*a_old.parameters(), *a_old.buffers()]):
            y.copy_(x)
    for x, y in zip(a_new.parameters(), a_old.parameters()):
        assert torch.equal(x, y)
    td = make_batch()
    torch.manual_seed(123); info_new = a_new.train_adapt(td.clone())
    torch.manual_seed(123); info_old = a_old.train_adapt(td.clone())
    for (n1, x), (n2, y) in zip(a_new.named_parameters(), a_old.named_parameters()):
        assert n1 == n2 and torch.equal(x, y), n1
    assert info_new['adapt/object_loss'] == info_old['adapt/object_loss']
    assert info_new['adapt/priv_loss'] == info_old['adapt/priv_loss']


def test_object_loss_split():
    agent = make_agent(ppo_haic)
    info = agent.train_adapt(make_batch(2))
    d, pose = DIMS['object_'], 12
    combined = (pose * info['adapt/object_loss_pose'] + (d - pose) * info['adapt/object_loss_other']) / d
    assert combined == pytest.approx(info['adapt/object_loss'], rel=1e-5)


def test_transform_object():
    tf = ppo_haic.TransformObject(22, ['object_', 'object_geo_'], ['object_trans'])
    pts = torch.randn(N, 128, 3)
    ang = torch.rand(N) * 6.28
    R = torch.zeros(N, 3, 3)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = ang.cos(), -ang.sin(), ang.sin(), ang.cos(), 1
    p = torch.randn(N, 3)
    obj = torch.cat([torch.randn(N, 10), p, R.reshape(N, 9)], -1)
    td = TensorDict({'object_': obj, 'object_geo_': pts.reshape(N, -1)}, [N])
    out = tf(td)['object_trans'].reshape(N, 128, 3)
    expect = torch.einsum('nij,nkj->nki', R, pts) + p[:, None]
    assert torch.allclose(out, expect, atol=1e-5)


# ---------------------------------------------------------------- heading conditioning
def make_heading_agent(phase, heading_to=None, seed=0):
    torch.manual_seed(seed)
    dims = dict(DIMS, heading_=2)
    obs = Composite({k: Unbounded((N, d)) for k, d in dims.items()}, shape=[N])
    act = Unbounded((N, A_DIM))  # the env passes a plain tensor spec
    env = SimpleNamespace(cfg=SimpleNamespace(reward={'tracking': {}, 'loco': {}}),
                          action_manager=SimpleNamespace(joint_names=[f'j{i}' for i in range(A_DIM)]),
                          current_iter=0, set_progress=lambda i: None)
    kw = {} if heading_to is None else {'heading_key': 'heading_', 'heading_to': heading_to}
    return ppo_haic.PPOHAIC(ppo_haic.PPOConfig(phase=phase, **kw), obs, act, None, 'cpu', env)


def heading_batch(seed=3):
    g = torch.Generator().manual_seed(seed)
    td = TensorDict({k: torch.randn(N, d, generator=g) for k, d in dict(DIMS, heading_=2).items()}, [N])
    td['is_init'] = torch.zeros(N, 1, dtype=torch.bool)
    return td


@pytest.mark.parametrize('heading_to', ['adapt', 'actor', 'both'])
def test_heading_student_starts_identical_to_plain_student(heading_to):
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    teacher = make_heading_agent('train')
    for p in teacher.parameters():  # make every layer matter
        torch.nn.init.normal_(p, std=0.3)
    sd = teacher.state_dict()
    plain = make_heading_agent('finetune', seed=1)
    plain.load_state_dict(sd)
    heading = make_heading_agent('finetune', heading_to, seed=2)
    heading.load_state_dict(sd)
    td = heading_batch()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        a = plain.get_rollout_policy('eval')(td.clone())
        b = heading.get_rollout_policy('eval')(td.clone())
        td2 = td.clone()
        td2['heading_'] = torch.randn(N, 2)
        c = heading.get_rollout_policy('eval')(td2)
    assert torch.allclose(a['loc'], b['loc'], atol=1e-6) and torch.allclose(b['loc'], c['loc'], atol=1e-6)
    assert torch.allclose(a['priv_pred'], b['priv_pred'], atol=1e-6)
    # the heading input is actually wired in where requested
    keys_wm = heading.adapt_module.module[0].in_keys + heading.object_adapt.module[0].in_keys
    keys_act = heading.actor_adapt.module[0][0].in_keys
    assert ('heading_' in keys_wm) == (heading_to in ('adapt', 'both'))
    assert ('heading_' in keys_act) == (heading_to in ('actor', 'both'))
    # and it is trainable: after a world-model update the heading columns move away from zero
    if heading_to in ('adapt', 'both'):
        w = heading.adapt_module.module[1].module[0][0].weight
        assert torch.count_nonzero(w[:, -2:]) == 0
        batch = TensorDict({k: torch.randn(N, T, d) for k, d in dict(DIMS, heading_=2).items()}, [N, T])
        batch['is_init'] = torch.zeros(N, T, 1, dtype=torch.bool)
        heading.train_adapt(batch)
        assert torch.count_nonzero(w[:, -2:]) > 0


def test_heading_checkpoint_guards():
    teacher = make_heading_agent('train')
    st = make_heading_agent('finetune', 'adapt')
    st.load_state_dict(teacher.state_dict())
    sd = st.state_dict()
    assert sd['heading_cfg'] == {'key': 'heading_', 'to': 'adapt', 'dim': 2}
    make_heading_agent('finetune', 'adapt').load_state_dict(sd)  # same schema: fine
    with pytest.raises(RuntimeError, match='mismatch'):
        make_heading_agent('finetune', 'both').load_state_dict(sd)
    with pytest.raises(RuntimeError, match='heading inputs'):
        make_heading_agent('finetune').load_state_dict(sd)
    plain_student = make_heading_agent('finetune')
    plain_student.load_state_dict(teacher.state_dict())
    with pytest.raises(RuntimeError, match='stage-1'):
        make_heading_agent('finetune', 'adapt').load_state_dict(plain_student.state_dict())
