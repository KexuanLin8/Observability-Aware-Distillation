"""Control-flow test of research/pipeline.py with the simulator jobs mocked out."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research import pipeline as P  # noqa: E402

LAYOUT_DYN = [{'name': n, 'dim': d} for n, d in (('object_xy_b', 2), ('object_heading_b', 2),
              ('ref_contact_pos_b', 6), ('object_com_vel_b', 6), ('object_com_acc_b', 6),
              ('object_pos_b', 3), ('object_ori_b', 9))]


def fake_execute(calls):
    def run(cmd, job, env=None, timeout=None):
        cmd = [str(c) for c in cmd]
        calls.append((job.name, cmd))
        job.mkdir(parents=True, exist_ok=True)
        script = Path(cmd[2]).name
        kv = dict(c.split('=', 1) for c in cmd[3:] if '=' in c)
        if script == 'preflight.py':
            groups = {'policy': 50, 'object_': 34}
            if 'heading' in kv['task']:
                groups['heading_'] = 2
            Path(kv['+preflight_out']).write_text(json.dumps({
                'layout': LAYOUT_DYN, 'pose_tail_ok': True, 'all_finite': True, 'state_tensor_device': 'cuda:0',
                'group_widths': groups}))
        elif script == 'train.py':
            d = job / 'hydra' / 'wandb' / 'offline-run-x' / 'files'
            d.mkdir(parents=True)
            phase = 'train' if kv['algo'] == 'ppo_haic_train' else 'finetune'
            (d / 'checkpoint_final.pt').write_text(phase)
        elif script == 'rollout.py':
            out = Path(kv['+out_dir'])
            n = int(kv['task.num_envs'])
            (out / 'summary.json').write_text(json.dumps({
                'n_envs': n, 'n_success': n // 2, 'success_rate': 0.5, 'n_strict_success': n // 2,
                'strict_success_rate': 0.5, 'failure_causes': {}, 'mode': kv['+mode'], 'seed': int(kv['seed']),
                'checkpoint_sha256': 'x'}))
        return ''
    return run


@pytest.fixture
def patched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(P, 'OUT_ROOT', tmp_path)
    monkeypatch.setattr(P, 'execute', fake_execute(calls))
    monkeypatch.setattr(P, 'checkpoint_info', lambda p: {
        'path': str(p), 'sha256': 'x', 'last_phase': Path(p).read_text(), 'last_iter': 3})
    monkeypatch.setenv('OMNI_KIT_ACCEPT_EULA', 'YES')
    return calls, tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['pipeline.py', *argv])
    return P.main()


def test_full_flow_and_resume(patched, monkeypatch):
    calls, root = patched
    assert run(monkeypatch, '--arm', 'dyn', '--student-seeds', '0') == 0
    names = [c[0] for c in calls]
    assert names == ['preflight', 'teacher', 'teacher-eval-s0', 'teacher-eval-s1', 'teacher-eval-s2',
                     'student-s0', 'eval-s0', 'eval-s1', 'eval-s2']
    student_cmd = dict(c.split('=', 1) for c in calls[5][1][3:] if '=' in c)
    assert student_cmd['algo'] == 'ppo_haic_finetune' and student_cmd['checkpoint_path'].endswith('checkpoint_final.pt')
    assert student_cmd['task'] == 'G1/haic/skateboard_dyn' and student_cmd['seed'] == '0'
    # second invocation adds students 1, 2 and reuses everything else
    calls.clear()
    assert run(monkeypatch, '--arm', 'dyn', '--student-seeds', '1', '2') == 0
    assert [c[0] for c in calls] == ['student-s1', 'eval-s0', 'eval-s1', 'eval-s2',
                                     'student-s2', 'eval-s0', 'eval-s1', 'eval-s2']
    s = json.loads((root / 'skateboard-dyn' / 'teacher-s0' / 'summary.json').read_text())
    assert s['status'] == 'complete' and sorted(s['students']) == ['0', '1', '2']
    assert s['student_mean'] == 0.5


def test_incomplete_stage_is_redone(patched, monkeypatch):
    calls, root = patched
    stale = root / 'skateboard-geo' / 'teacher-s0' / 'teacher'
    stale.mkdir(parents=True)
    (stale / 'junk').write_text('x')
    monkeypatch.setattr(P, 'run_preflight', lambda *a: {'report': {'layout': []}})
    assert run(monkeypatch, '--arm', 'geo', '--student-seeds', '0', '--skip-teacher-eval') == 0
    assert any(p.name.startswith('teacher.incomplete-') for p in stale.parent.iterdir())
    assert (stale / 'complete.json').is_file()


def test_wrong_arm_fields_rejected(patched, monkeypatch):
    calls, root = patched
    assert run(monkeypatch, '--arm', 'geo', '--student-seeds', '0') == 1  # preflight reports dyn fields
    s = json.loads((root / 'skateboard-geo' / 'teacher-s0' / 'summary.json').read_text())
    assert s['status'] == 'failed' and 'unexpected object_ fields' in s['failure']


def test_heading_arm_reuses_dyn_teacher(patched, monkeypatch):
    calls, root = patched
    assert run(monkeypatch, '--arm', 'dyn-hwm', '--student-seeds', '0') == 1  # dyn teacher missing
    calls.clear()
    assert run(monkeypatch, '--arm', 'dyn', '--student-seeds', '0', '--skip-teacher-eval') == 0
    calls.clear()
    assert run(monkeypatch, '--arm', 'dyn-hwm', '--student-seeds', '0') == 0
    names = [c[0] for c in calls]
    assert 'teacher' not in names and not any(n.startswith('teacher-eval') for n in names)
    assert names[0] == 'student-s0'  # preflight already completed by the first (failed) attempt
    kv = {c[0]: dict(x.split('=', 1) for x in c[1][3:] if '=' in x) for c in calls}
    st = kv['student-s0']
    assert st['task'] == 'G1/haic/skateboard_dyn_heading'
    assert st['algo.heading_key'] == 'heading_' and st['algo.heading_to'] == 'adapt'
    assert st['checkpoint_path'] == str(root / 'skateboard-dyn' / 'teacher-s0' / 'teacher' / 'hydra' / 'wandb'
                                        / 'offline-run-x' / 'files' / 'checkpoint_final.pt')
    assert st['task.observation.heading_.robot_heading_err_imu.seed'] == '0'
    ev = [dict(x.split('=', 1) for x in c[1][3:] if '=' in x) for c in calls if c[0].startswith('eval-s')]
    assert [e['task.observation.heading_.robot_heading_err_imu.seed'] for e in ev] == ['1000', '1001', '1002']
    assert all(e['algo.heading_to'] == 'adapt' for e in ev)
    s = json.loads((root / 'skateboard-dyn-hwm' / 'teacher-s0' / 'summary.json').read_text())
    assert s['status'] == 'complete' and s['teacher']['reused_from'].endswith('teacher')


def test_plain_arm_rejects_heading_group(patched, monkeypatch):
    calls, root = patched
    monkeypatch.setitem(P.TASKS['skateboard']['arms'], 'dyn', {'task': 'G1/haic/skateboard_dyn_heading', 'base': 'dyn'})
    assert run(monkeypatch, '--arm', 'dyn', '--student-seeds', '0') == 1
    s = json.loads((root / 'skateboard-dyn' / 'teacher-s0' / 'summary.json').read_text())
    assert 'unexpectedly has a heading_ group' in s['failure']


@pytest.mark.parametrize('task', ['push_cart', 'box_stairslope'])
def test_cross_task_flow(patched, monkeypatch, task):
    calls, root = patched
    assert run(monkeypatch, '--task', task, '--arm', 'dyn', '--student-seeds', '0', '--skip-teacher-eval') == 0
    assert run(monkeypatch, '--task', task, '--arm', 'dyn-hboth-cw', '--student-seeds', '0') == 0
    kv = [dict(x.split('=', 1) for x in c[1][3:] if '=' in x) for c in calls]
    data = {k['task.command.data_path'] for k in kv}
    assert len(data) == 1 and Path(data.pop()).as_posix().endswith(f'data/motion/g1/mirobotA/{task}')
    tasks = [k['task'] for k in kv if k.get('algo') == 'ppo_haic_finetune']
    assert tasks[0] == f'G1/haic/{task}_dyn' and tasks[-1] == f'G1/haic/{task}_dyn_heading'
    cw = [k for k in kv if k.get('algo.priv_loss_weighting')]
    assert cw and all(k['algo.priv_loss_weighting'] == 'action_jacobian' and k['algo.heading_to'] == 'both' for k in cw)
    teacher = str(root / f'{task}-dyn' / 'teacher-s0' / 'teacher')
    assert all(k['checkpoint_path'].startswith(teacher) for k in kv if k.get('algo') == 'ppo_haic_finetune'
               and 'heading' in k['task'] and '+out_dir' not in k)
