"""Report of a MuJoCo sim-to-sim evaluation with the metrics fixed in research/SIM2SIM_PROTOCOL.md.

    python research/sim2sim_report.py --export D:/HAIC-Claude/result/sim2sim                       # HAIC vs OH-Both
    python research/sim2sim_report.py --export D:/HAIC-Claude/result/sim2sim --pairs dyn-cr:dyn-hboth-cr dyn:dyn-cr dyn-hboth:dyn-hboth-cr

Reads <export>/<task>/<arm>/<student>/<tag>/episodes.csv and trace.npz written by research/sim2sim.py and prints, per arm
and student, then pooled over students:
  success          Isaac criterion (all terminations, as in the Isaac evaluation)
  drift-tolerant   no local termination and no fall (world-frame terms off)
  cross-track      drift-tolerant and no run of 25 steps with |lateral root error| >= 0.3 m
                   (lateral: perpendicular to the reference heading; the along-track simulator lag is ignored)
  fell             pelvis below 0.4 m at the end
  |lat|, |yaw|     median absolute lateral root and heading errors at 9 s (episodes still running)
and, for each pair A:B, the episode-paired two-sided sign test of success and of cross-track success (episode k of
every student uses the same seed in both arms). Writes the numbers to <export>/sim2sim_report-<tag>.json.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

LOCAL = ('local_pos', 'local_ori')     # terminations that are not world-frame drift (research/sim2sim.py DRIFT)
LAT_THR, LAT_STEPS, T_EVAL = 0.3, 25, 9.0


def episodes(d):
    rows = list(csv.DictReader(open(d / 'episodes.csv', encoding='utf-8')))
    tr = np.load(d / 'trace.npz')
    dx = tr['root_xyz'] - tr['ref_root_xyz']
    yr = tr['ref_root_yaw']
    lat = -np.sin(yr) * dx[..., 0] + np.cos(yr) * dx[..., 1]                       # [T, episodes]
    yaw = np.degrees((tr['root_yaw'] - yr + np.pi) % (2 * np.pi) - np.pi)
    k = int(round(T_EVAL / float(tr['step_dt'])))
    out = []
    for i, r in enumerate(rows):
        far = np.nan_to_num(np.abs(lat[:, i]), nan=0.0) >= LAT_THR
        run, longest = 0, 0
        for f in far:
            run = run + 1 if f else 0
            longest = max(longest, run)
        local = any(r.get(f'term/{t}') == '1' for t in LOCAL)
        fell = r['fell'] == '1'
        out.append({'episode': int(r['episode']), 'success': r['success'] == '1',
                    'drift_tolerant': not local and not fell,
                    'cross_track': not local and not fell and longest < LAT_STEPS, 'fell': fell,
                    'lat9': abs(lat[k, i]) if k < len(lat) else np.nan, 'yaw9': abs(yaw[k, i]) if k < len(yaw) else np.nan})
    return out


def summary(eps):
    f = lambda key: 100 * float(np.mean([e[key] for e in eps]))
    return {'n': len(eps), 'success': f('success'), 'drift_tolerant': f('drift_tolerant'), 'cross_track': f('cross_track'),
            'fell': f('fell'), 'lat9_m': float(np.nanmedian([e['lat9'] for e in eps])),
            'yaw9_deg': float(np.nanmedian([e['yaw9'] for e in eps]))}


def sign_test(a, b, key):
    """Episode-paired two-sided sign test of a binary outcome: (A only, B only, p)."""
    na = sum(x[key] and not y[key] for x, y in zip(a, b))
    nb = sum(y[key] and not x[key] for x, y in zip(a, b))
    n = na + nb
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(na, nb) + 1)) / 2 ** n)
    return na, nb, p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', type=Path, required=True, help='unzipped sim2sim export with MuJoCo results')
    p.add_argument('--task', default='skateboard')
    p.add_argument('--tag', default='mujoco-wd0.004-fl0.01-isaac_assets1', help='result folder of the protocol run')
    p.add_argument('--pairs', nargs='+', default=['dyn:dyn-hboth'], help='A:B arm pairs to compare')
    p.add_argument('--students', nargs='+', default=['st0', 'st1', 'st2'])
    a = p.parse_args()
    arms = list(dict.fromkeys(x for pr in a.pairs for x in pr.split(':')))
    data, report = {}, {'tag': a.tag, 'task': a.task, 'arms': {}, 'pairs': {}}
    hdr = f'{"arm":14s} {"student":7s} {"n":>4s} {"success":>8s} {"drift-tol":>9s} {"x-track":>8s} {"fell":>6s} {"|lat|9s":>8s} {"|yaw|9s":>8s}'
    print(hdr)
    print('-' * len(hdr))
    for arm in arms:
        data[arm] = {}
        for s in a.students:
            d = a.export / a.task / arm / s / a.tag
            if not (d / 'episodes.csv').is_file():
                print(f'{arm:14s} {s:7s} missing: {d}')
                continue
            data[arm][s] = episodes(d)
        report['arms'][arm] = {}
        for s, eps in list(data[arm].items()) + [('pooled', [e for v in data[arm].values() for e in v])]:
            if not eps:
                continue
            m = summary(eps)
            report['arms'][arm][s] = m
            print(f'{arm:14s} {s:7s} {m["n"]:4d} {m["success"]:7.1f}% {m["drift_tolerant"]:8.1f}% {m["cross_track"]:7.1f}% '
                  f'{m["fell"]:5.1f}% {m["lat9_m"]:7.2f}m {m["yaw9_deg"]:7.1f}d')
        print()
    for pr in a.pairs:
        A, B = pr.split(':')
        common = [s for s in a.students if s in data.get(A, {}) and s in data.get(B, {})]
        if not common:
            print(f'{A} vs {B}: no common students')
            continue
        xa = [e for s in common for e in sorted(data[A][s], key=lambda e: e['episode'])]
        xb = [e for s in common for e in sorted(data[B][s], key=lambda e: e['episode'])]
        res = {}
        for key in ('success', 'cross_track'):
            na, nb, pv = sign_test(xa, xb, key)
            ra, rb = 100 * np.mean([e[key] for e in xa]), 100 * np.mean([e[key] for e in xb])
            res[key] = {'A': ra, 'B': rb, 'diff_pp': rb - ra, 'A_only': na, 'B_only': nb, 'p': pv}
            print(f'{B} vs {A}, {key:11s}: {rb:5.1f}% vs {ra:5.1f}% ({rb - ra:+5.1f} pp), discordant {nb}/{na} '
                  f'({B} only / {A} only), sign test p = {pv:.3g}  [{len(xa)} paired episodes, students {", ".join(common)}]')
        report['pairs'][pr] = res
    out = a.export / f'sim2sim_report-{a.tag}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
