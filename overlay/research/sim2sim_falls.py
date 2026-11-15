"""Where and why students fall in MuJoCo (research/sim2sim.py results), compared with their golden Isaac rollouts.

    python research/sim2sim_falls.py --export D:/HAIC-Claude/result/sim2sim --arms dyn dyn-hboth dyn-cr dyn-hboth-cr

Prints per arm, pooled over the students:
  fall onset       time at which the pelvis first drops below 0.6 m, for episodes that end fallen
  board speed      horizontal board speed at fixed times, Isaac (golden rollout of each student) and MuJoCo (median)
  pushing          between 4.5 and 7.5 s: lateral offset of the foot on the ground from the board (reference heading
                   frame), share of frames with both feet on the board, backward sweep of the ground foot relative
                   to the board (the push), Isaac and MuJoCo
  before a fall    0.5 s before the onset, fallers vs. episodes that do not fall at the same times: board speed,
                   |pelvis - board| lateral offset, |robot heading - reference heading|
"""
import argparse
import csv
from pathlib import Path

import numpy as np

DT = 0.02
PUSH = slice(int(4.5 / DT), int(7.5 / DT))
TIMES = (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)


def yaw_wxyz(q):
    return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def frame(ryaw):
    ax = np.stack([np.cos(ryaw), np.sin(ryaw)], -1)
    return ax, np.stack([-ax[..., 1], ax[..., 0]], -1)


def pushing(ank, obj, ryaw, vobj):
    """Ground-foot lateral offset, both-feet-on-board share and ground-foot backward sweep over the push phase."""
    lat, sweep, both, n = [], [], 0, 0
    for k in range(1, len(ank) - 1):
        if np.isnan(ank[k]).any() or np.isnan(obj[k]).any():
            continue
        ax, ay = frame(ryaw[k])
        rel, dz = ank[k, :, :2] - obj[k, :2], ank[k, :, 2] - obj[k, 2]
        on = (np.abs(rel @ ax) < 0.45) & (np.abs(rel @ ay) < 0.2) & (dz > -0.03)
        n += 1
        if on.all():
            both += 1
            continue
        f = int(np.argmin(on))
        if ank[k, f, 2] < 0.12:  # the off-board foot is on the ground
            lat.append(abs(rel[f] @ ay))
            vf = (ank[k + 1, f, :2] - ank[k - 1, f, :2]) / (2 * DT)
            sweep.append(-(vf - vobj[k, :2]) @ ax)
    return [np.median(lat) if lat else np.nan, both / max(n, 1), np.median(sweep) if sweep else np.nan]


def episodes(d):
    t = np.load(d / 'trace.npz')
    rows = list(csv.DictReader(open(d / 'episodes.csv', encoding='utf-8')))
    out = []
    for i, r in enumerate(rows):
        z = np.nan_to_num(t['root_xyz'][:, i, 2], nan=9.0)
        k = int(np.argmax(z < 0.6)) if r['fell'] == '1' and (z < 0.6).any() else -1
        out.append({key: t[key][:, i] for key in ('root_xyz', 'obj_xyz', 'ankle_xyz', 'obj_lin_vel', 'root_yaw', 'ref_root_yaw')}
                   | {'onset': k})
    return out


def before(e, k):
    ax, ay = frame(e['ref_root_yaw'][k])
    return [np.linalg.norm(e['obj_lin_vel'][k, :2]), abs((e['root_xyz'][k, :2] - e['obj_xyz'][k, :2]) @ ay),
            abs(np.degrees((e['root_yaw'][k] - e['ref_root_yaw'][k] + np.pi) % (2 * np.pi) - np.pi))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--task', default='skateboard')
    p.add_argument('--tag', default='mujoco-wd0.004-fl0.01-isaac_assets1')
    p.add_argument('--arms', nargs='+', default=['dyn', 'dyn-hboth'])
    p.add_argument('--students', nargs='+', default=['st0', 'st1', 'st2'])
    a = p.parse_args()
    for arm in a.arms:
        eps, iso_v, iso_p, mj_p = [], [], [], []
        for s in a.students:
            d = a.export / a.task / arm / s
            g = np.load(d / 'golden.npz')
            v = np.linalg.norm(g['obj_lin_vel_w'][:, :2], axis=1)
            iso_v.append([v[int(t / DT)] if int(t / DT) < len(v) else np.nan for t in TIMES])
            iso_p.append(pushing(g['ankle_pos_w'][PUSH], g['obj_pos_w'][PUSH], yaw_wxyz(g['ref_root_quat_w'])[PUSH],
                                 g['obj_lin_vel_w'][PUSH]))
            E = episodes(d / a.tag)
            eps += E
            mj_p += [pushing(e['ankle_xyz'][PUSH], e['obj_xyz'][PUSH], e['ref_root_yaw'][PUSH], e['obj_lin_vel'][PUSH])
                     for e in E[::4]]
        mj_v = [np.nanmedian([np.linalg.norm(e['obj_lin_vel'][int(t / DT), :2]) for e in eps]) for t in TIMES]
        fall = [e for e in eps if e['onset'] > 0]
        on = np.array([e['onset'] * DT for e in fall])
        print(f'== {arm}: {len(fall)}/{len(eps)} episodes end fallen'
              + (f', onset median {np.median(on):.2f} s (IQR {np.percentile(on, 25):.2f}-{np.percentile(on, 75):.2f})' if len(on) else ''))
        print(f'   board speed at {TIMES} s: Isaac {np.round(np.nanmean(iso_v, 0), 2)}, MuJoCo {np.round(mj_v, 2)}')
        for name, P in (('Isaac', iso_p), ('MuJoCo', mj_p)):
            P = np.nanmedian(P, 0) if name == 'MuJoCo' else np.nanmean(P, 0)
            print(f'   pushing, {name:6s}: ground-foot lateral offset {P[0]:.3f} m, both feet on board {100 * P[1]:.0f}%, '
                  f'backward sweep rel. board {P[2]:.2f} m/s')
        if fall:
            ks = [e['onset'] - int(0.5 / DT) for e in fall]
            F = np.nanmedian([before(e, k) for e, k in zip(fall, ks) if k > 1], 0)
            rng = np.random.default_rng(0)
            G = [before(e, int(rng.choice(ks))) for e in eps if e['onset'] <= 0]
            G = np.nanmedian([x for x in G if not np.isnan(x).any()], 0)
            print(f'   0.5 s before a fall (fallers | others at the same times): board speed {F[0]:.2f} | {G[0]:.2f} m/s, '
                  f'|pelvis-board lateral| {F[1]:.3f} | {G[1]:.3f} m, |heading error| {F[2]:.0f} | {G[2]:.0f} deg')


if __name__ == '__main__':
    main()
