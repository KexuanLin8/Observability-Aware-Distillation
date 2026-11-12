"""Selection logic of research/showcase.py and the figure script, on synthetic paths."""
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from research import showcase as S

ROOT = Path(__file__).resolve().parents[3]


def write_paths(d, root, ref, ref_yaw, success, gterm):
    d.mkdir(parents=True, exist_ok=True)
    T, n = root.shape[:2]
    z = np.zeros((T, n, 3))
    np.savez_compressed(d / 'paths.npz', step_dt=0.02, root_xyz=root, ref_root_xyz=ref, obj_xyz=z, ref_obj_xyz=z,
                        root_yaw=ref_yaw, ref_root_yaw=ref_yaw)
    with (d / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['env_id', 'success', S.GLOBAL_POS])
        w.writeheader()
        for i in range(n):
            w.writerow({'env_id': i, 'success': success[i], S.GLOBAL_POS: gterm[i]})


def synthetic(tmp, yaw=0.0):
    """3 envs moving along the reference heading; env 0 drifts 0.5 m sideways and ends early."""
    T, n = 50, 3
    s = np.linspace(0, 2, T)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    ref = np.zeros((T, n, 3))
    ref[..., :2] = s[:, None, None] * fwd
    root = ref.copy()
    root[:, 0, :2] += np.linspace(0, 0.5, T)[:, None] * left
    root[:, 1, :2] += np.linspace(0, 0.1, T)[:, None] * left
    root[40:, 0] = np.nan
    yaws = np.full((T, n), yaw)
    write_paths(tmp / 'dyn' / 'paths', root, ref, yaws, [0, 1, 0], [1, 0, 0])
    good = ref.copy()
    write_paths(tmp / 'dyn-hboth' / 'paths', good, ref, yaws, [1, 1, 1], [0, 0, 0])
    return tmp


def test_final_lateral_is_heading_frame(tmp_path):
    for yaw in (0.0, 2.0):
        src = synthetic(tmp_path / str(yaw), yaw)
        lat = S.final_lateral(src / 'dyn' / 'paths')
        assert abs(lat[0] - 0.5 * 39 / 49) < 1e-6  # last alive step is 39
        assert abs(lat[1] - 0.1) < 1e-6 and lat[2] < 1e-9


def test_pair_select(tmp_path):
    src = synthetic(tmp_path)
    sel = S.pair_select(S.episodes(src / 'dyn' / 'paths'), S.episodes(src / 'dyn-hboth' / 'paths'),
                        S.final_lateral(src / 'dyn' / 'paths'))
    # env 2 fails without a global-position termination, so it is not a drift episode
    assert sel['candidates'] == [0]
    assert sel['n_both_success'] == 1 and sel['n_base_ok_better_fail'] == 0


def test_fixed_target_centre(tmp_path):
    src = synthetic(tmp_path)
    tgt, span = S.fixed_target(src / 'dyn' / 'paths', 1)
    assert abs(tgt[0] - 1.0) < 1e-6 and abs(tgt[1]) < 1e-6 and abs(span - 2.0) < 1e-6


def test_figure_script_runs(tmp_path):
    fig_script = ROOT / 'paper' / 'scripts' / 'fig_showcase.py'
    if not fig_script.is_file():
        return  # paper folder not checked out next to the repo
    src = synthetic(tmp_path / 'src')
    (src / 'selection.json').write_text('{"candidates": [0], "rendered": [0]}', encoding='utf-8')
    import matplotlib.pyplot as plt
    for arm in ('dyn', 'dyn-hboth'):
        snap = src / arm / 'render-env0-chase' / 'snapshots'
        snap.mkdir(parents=True)
        for step in (0, 75, 150):
            plt.imsave(snap / f'step{step:04d}_t{step * 0.02:05.2f}s.png', np.full((90, 160, 3), 200, np.uint8))
        plt.imsave(snap / 'step0040_t00.80s_end.png', np.zeros((90, 160, 3), np.uint8))
    spec = importlib.util.spec_from_file_location('fig_showcase', fig_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / 'figs'
    sys.argv = ['fig_showcase.py', '--src', str(src), '--out', str(out)]
    mod.main()
    assert (out / 'fig_teaser_paths.pdf').is_file() and (out / 'fig_teaser_snaps.pdf').is_file()
