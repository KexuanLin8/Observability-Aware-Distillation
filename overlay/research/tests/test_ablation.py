"""Privileged-input ablation: derangement, term resolution, policies, aggregation."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research import rollout as R  # noqa: E402
from research import ablate as AB  # noqa: E402
from research.tests.test_ppo_haic import make_heading_agent, heading_batch, N  # noqa: E402


def test_derangement():
    for seed in range(5):
        p = R.derangement(50, seed, 'cpu')
        assert sorted(p.tolist()) == list(range(50))
        assert not (p == torch.arange(50)).any()
    assert torch.equal(R.derangement(50, 3, 'cpu'), R.derangement(50, 3, 'cpu'))


class Term:
    def __init__(self, dim):
        self.dim = dim

    def compute(self):
        return torch.zeros(4, self.dim)


def fake_base(priv_terms, object_terms):
    funcs = lambda terms: SimpleNamespace(funcs={k: Term(d) for k, d in terms})
    return SimpleNamespace(num_envs=4, observation_funcs={'priv': funcs(priv_terms), 'object_': funcs(object_terms)})


def test_resolve_ablation():
    base = fake_base([('root_ang_vel_history', 3), ('ref_root_pos_future_b', 6), ('ref_root_ori_future_b', 4)],
                     [('object_xy_b', 2), ('object_pos_b', 3)])
    got = R.resolve_ablation(base, 'global_root')
    assert [(g, k, s.start, s.stop) for g, k, s in got] == [('priv', 'ref_root_pos_future_b', 3, 9),
                                                            ('priv', 'ref_root_ori_future_b', 9, 13)]
    got = R.resolve_ablation(base, 'object_relative')  # missing priv object terms are skipped
    assert [(g, k, s.start, s.stop) for g, k, s in got] == [('object_', '*', 0, 5)]
    with pytest.raises(RuntimeError):
        R.resolve_ablation(base, 'actuation')
    with pytest.raises(ValueError):
        R.resolve_ablation(base, 'nope')


def randomized(agent):
    for p in agent.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return agent


def test_ablation_policies():
    teacher = randomized(make_heading_agent('train'))
    student = make_heading_agent('finetune', seed=1)
    student.load_state_dict(teacher.state_dict())
    td = heading_batch()
    perm = R.derangement(N, 0, 'cpu')
    priv_dim = td['priv'].shape[-1]
    ablation = [('priv', 'x', slice(2, 5))]
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        # teacher: ablated action == action on inputs with that slice permuted by hand
        a = R.make_policy(teacher, 'teacher_ablate', ablation, perm)(td.clone())['action']
        manual = td.clone()
        manual['priv'][:, 2:5] = td['priv'][perm, 2:5]
        b = R.make_policy(teacher, 'teacher_ablate', [('priv', 'x', slice(0, 0))], perm)(manual)['action']
        clean = R.make_policy(teacher, 'teacher_ablate', [('priv', 'x', slice(0, 0))], perm)(td.clone())['action']
        assert torch.allclose(a, b, atol=1e-6) and not torch.allclose(a, clean)
        # empty slice == plain teacher policy
        plain = teacher.get_rollout_policy('eval')(td.clone())['action']
        assert torch.allclose(clean, plain, atol=1e-6)
        # student: oracle_priv_ablate with an empty slice == oracle_priv; real slice differs
        o = R.make_policy(student, 'oracle_priv')(td.clone())['action']
        o0 = R.make_policy(student, 'oracle_priv_ablate', [('priv', 'x', slice(0, 0))], perm)(td.clone())['action']
        o1 = R.make_policy(student, 'oracle_priv_ablate', [('priv', 'x', slice(0, priv_dim))], perm)(td.clone())['action']
        assert torch.allclose(o, o0, atol=1e-6) and not torch.allclose(o, o1)
    # permutation did not leak into the caller's tensordict values outside the call
    assert priv_dim == 6


def test_aggregate():
    teacher = {'base': [0.98], 'global_root': [0.40], 'feet': [0.97]}
    students = {'s0': {'pred': [0.78], 'oracle': [0.98], 'global_root': [0.80], 'feet': [0.96]},
                's1': {'pred': [0.76], 'oracle': [0.96], 'global_root': [0.74], 'feet': [0.96]}}
    summary, rows = AB.aggregate(['global_root', 'feet'], teacher, students)
    r = {x['group']: x for x in rows}
    assert math.isclose(r['global_root']['teacher_drop'], 0.58)
    assert math.isclose(r['global_root']['oracle_gain_lost'], (0.9 + 1.1) / 2)
    assert math.isclose(r['feet']['oracle_gain_lost'], (0.1 + 0.0) / 2, abs_tol=1e-12)
    assert math.isclose(summary['pred_base'], 0.77)
    text = AB.report('skateboard', 'dyn', summary, rows, [0])
    assert text.index('global_root') < text.index('| feet')


def test_ablate_main_flow(tmp_path, monkeypatch):
    import json
    root = tmp_path / 'skateboard-dyn' / 'teacher-s0'
    root.mkdir(parents=True)
    (root / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'teacher': {'checkpoint': {'path': '/t.pt'}},
        'students': {'0': {'checkpoint': {'path': '/s0.pt'}}, '1': {'checkpoint': {'path': '/s1.pt'}}}}))
    calls = []

    def fake_run_eval(job, args, spec, ckpt, algo, seed, env, mode='pred', trace=True, extra=()):
        calls.append((job.relative_to(root).as_posix(), ckpt, algo, mode, tuple(extra)))
        rates = {'pred': 0.78, 'oracle_priv': 0.98, 'oracle_priv_ablate': 0.80, 'teacher_ablate': 0.5}
        return {'success_rate': rates.get(mode, 0.99)}

    monkeypatch.setattr(AB, 'OUT_ROOT', tmp_path)
    monkeypatch.setattr(AB, 'run_eval', fake_run_eval)
    monkeypatch.setenv('OMNI_KIT_ACCEPT_EULA', 'YES')
    monkeypatch.setattr(sys, 'argv', ['ablate.py', '--arm', 'dyn', '--groups', 'global_root', 'feet'])
    AB.main()
    jobs = [c[0] for c in calls]
    assert jobs[:3] == ['teacher-eval-s0', 'ablation/teacher-global_root-s0', 'ablation/teacher-feet-s0']
    assert 'student-s1/ablation/oracle_priv-feet-s0' in jobs
    t = [c for c in calls if c[0] == 'ablation/teacher-global_root-s0'][0]
    assert t[1:] == ('/t.pt', 'ppo_haic_train', 'teacher_ablate', ('+ablate=global_root',))
    res = json.loads((tmp_path / 'ablation' / 'skateboard-dyn' / 'ablation.json').read_text())
    assert math.isclose(res['rows'][0]['oracle_gain_lost'], 0.9)
    assert (tmp_path / 'ablation' / 'skateboard-dyn.zip').is_file()


def test_ablate_main_handles_zero_oracle_gain(tmp_path, monkeypatch):
    """Smoke runs: every rate is 0, so the oracle gain is 0 and the ratio is undefined (NaN)."""
    import json
    root = tmp_path / 'skateboard-dyn' / 'teacher-s0'
    root.mkdir(parents=True)
    (root / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'teacher': {'checkpoint': {'path': '/t.pt'}},
        'students': {'0': {'checkpoint': {'path': '/s0.pt'}}}}))
    monkeypatch.setattr(AB, 'OUT_ROOT', tmp_path)
    monkeypatch.setattr(AB, 'run_eval', lambda *a, **k: {'success_rate': 0.0})
    monkeypatch.setenv('OMNI_KIT_ACCEPT_EULA', 'YES')
    monkeypatch.setattr(sys, 'argv', ['ablate.py', '--arm', 'dyn', '--groups', 'global_root'])
    AB.main()
    res = json.loads((tmp_path / 'ablation' / 'skateboard-dyn' / 'ablation.json').read_text())
    assert res['rows'][0]['oracle_gain_lost'] is None and res['rows'][0]['oracle_priv_ablated_sd'] is None
    assert 'n/a' in (tmp_path / 'ablation' / 'skateboard-dyn' / 'ablation.md').read_text(encoding='utf-8')
