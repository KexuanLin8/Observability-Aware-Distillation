"""CPU tests for research/analyze.py on a synthetic run directory (no simulator)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research import analyze as A  # noqa: E402


def test_auc_basic():
    assert A.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert A.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert A.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(A.auc([1, 2], [1, 1]))


def test_reference_copy_alpha():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(5000, 2))
    r = t + rng.normal(scale=0.3, size=(5000, 2))
    for alpha in (0.0, 0.4, 1.0):
        p = t + alpha * (r - t) + rng.normal(scale=0.01, size=(5000, 2))
        res = A.reference_copy_alpha(t, p, r, min_gap=0.05)
        assert abs(res['alpha'] - alpha) < 0.02, res


def make_eval(d: Path, seed, n=200, T=400, dt=0.02, mode='pred', fail_frac=0.3):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    fail = np.zeros(n, bool)
    fail[: int(fail_frac * n)] = True
    rng.shuffle(fail)
    end = np.where(fail, rng.integers(200, 350, n), T - 1)
    # failing episodes drift in heading from ~1 s on; successful ones stay near zero
    t = np.arange(T)[:, None] * dt
    drift = np.where(fail, rng.uniform(0.05, 0.12, n), rng.uniform(-0.01, 0.01, n))
    heading = drift * np.clip(t - 1.0, 0, None) + rng.normal(scale=0.01, size=(T, n))
    lat = np.cumsum(heading * 0.02, axis=0)
    local = 0.1 + rng.normal(scale=0.02, size=(T, n))
    tr = {k: np.full((T, n), np.nan, np.float32) for k in
          ('heading_err', 'root_err_lat', 'root_err_along', 'root_err_z', 'wm_pos_err', 'wm_ori_err',
           'wm_yaw_err', 'priv_err', 'priv_norm', 'frame', 'obj_pos_err_w', 'obj_yaw_err_w',
           f'term_err/{A.LOCAL_POS}', f'term_err/{A.GLOBAL_POS}', 'yaw_true_b', 'yaw_pred_b', 'yaw_ref_b')}
    for k in ('pos_true_b', 'pos_pred_b', 'pos_ref_b'):
        tr[k] = np.full((T, n, 3), np.nan, np.float32)
    alive = np.arange(T)[:, None] <= end[None, :]
    vals = {'heading_err': heading, 'root_err_lat': lat, 'root_err_along': 0.5 * lat,
            f'term_err/{A.LOCAL_POS}': local, f'term_err/{A.GLOBAL_POS}': np.abs(lat),
            'wm_pos_err': np.abs(rng.normal(0.05, 0.01, (T, n))), 'wm_yaw_err': rng.normal(0, 0.05, (T, n)),
            'priv_err': np.ones((T, n)), 'priv_norm': np.full((T, n), 4.0)}
    for k, v in vals.items():
        tr[k] = np.where(alive, v, np.nan).astype(np.float32)
    true = rng.normal(size=(T, n, 3))
    ref = true + rng.normal(scale=0.2, size=(T, n, 3))
    pred = true + 0.7 * (ref - true)
    for k, v in (('pos_true_b', true), ('pos_ref_b', ref), ('pos_pred_b', pred)):
        tr[k] = np.where(alive[..., None], v, np.nan).astype(np.float32)
    yt = rng.normal(size=(T, n)); yr = yt + rng.normal(scale=0.3, size=(T, n)); yp = yt + 0.7 * (yr - yt)
    for k, v in (('yaw_true_b', yt), ('yaw_ref_b', yr), ('yaw_pred_b', yp)):
        tr[k] = np.where(alive, v, np.nan).astype(np.float32)
    np.savez_compressed(d / 'trace.npz', **tr)
    rows = []
    for i in range(n):
        g = int(fail[i])
        rows.append({'env_id': i, 'success': 1 - g, 'strict_success': 1 - g, 'episode_steps': int(end[i]) + 1,
                     'end_rollout_step': int(end[i]), 'terminated': g, 'truncated': 1 - g,
                     f'term/{A.GLOBAL_POS}': g, f'term/{A.LOCAL_POS}': 0, f'term/{A.LOCAL_ORI}': 0})
    pd.DataFrame(rows).to_csv(d / 'episodes.csv', index=False)
    s = {'mode': mode, 'seed': seed, 'n_envs': n, 'success_rate': 1 - fail.mean(),
         'strict_success_rate': 1 - fail.mean(), 'step_dt': dt}
    (d / 'summary.json').write_text(json.dumps(s))
    wm = {'n_samples': 1, 'fields': {'object_pos_b': {'r2_pooled': 0.8}, 'object_com_acc_b': {'r2_pooled': 0.1}}}
    (d / 'wm_stats.json').write_text(json.dumps(wm))


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / 'skateboard-dyn' / 'teacher-s0'
    for ss in (0, 1):
        for e in (0, 1):
            make_eval(root / f'student-s{ss}' / f'eval-s{e}', seed=10 * ss + e)
        (root / f'student-s{ss}' / 'eval-oracle_object-s0').mkdir(parents=True)
        d = root / f'student-s{ss}' / 'eval-oracle_object-s0'
        (d / 'summary.json').write_text(json.dumps({'mode': 'oracle_object', 'seed': 0, 'n_envs': 200,
                                                    'success_rate': 0.9, 'strict_success_rate': 0.9}))
        pd.DataFrame([{'env_id': 0, 'success': 1, 'end_rollout_step': 10}]).to_csv(d / 'episodes.csv', index=False)
    return root


def test_full_analysis(run_root, tmp_path):
    r = A.analyze_run(run_root)
    agg = r['success_agg'].set_index('mode')
    assert abs(agg.loc['pred', 'success_mean'] - 0.7) < 1e-9
    assert agg.loc['oracle_object', 'success_mean'] == pytest.approx(0.9)
    fb = r['failures']
    assert fb['n_failures'] == 4 * 60
    assert fb['drift_type'] == fb['global_pos_failures'] == 240
    ew = r['early_warning']
    h = ew[(ew.feature == '|heading_err|') & (ew.time_s == 3.0)].iloc[0]
    loc = ew[(ew.feature == 'local_pos_err') & (ew.time_s == 3.0)].iloc[0]
    assert h['auc'] > 0.95 and abs(loc['auc'] - 0.5) < 0.1
    w = r['world_model']
    assert abs(w['alpha_xy']['alpha'] - 0.7) < 0.02
    assert abs(w['alpha_yaw']['alpha'] - 0.7) < 0.05
    assert w['r2_by_field_mean']['object_com_acc_b'] == pytest.approx(0.1)
    out = tmp_path / 'report'
    out.mkdir()
    results = {'dyn': r}
    A.plot(results, out)
    A.report(results, out)
    json.dumps(A.to_jsonable(results), default=float)
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert 'drift-type' in text and (out / 'early_warning_auc.png').is_file()


def test_tracking_errors(run_root):
    r = A.analyze_run(run_root)
    agg = r['tracking']['agg']
    mu, sd = agg['mean']['heading |err| (deg)']
    assert mu > 0 and sd == sd
    assert agg['end_success']['root lateral |err| (m)'][0] >= 0
    assert set(r['tracking']['per_student']) == {'student-s0', 'student-s1'}
