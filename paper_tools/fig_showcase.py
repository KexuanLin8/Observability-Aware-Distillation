"""Teaser figure (Fig. 1) from a paired showcase run (research/showcase.py on the server).

    python paper/scripts/fig_showcase.py --src D:/HAIC-Claude/showcase/skateboard/st0-ev0-n256

--src is one <tag> folder of the unzipped showcase-<task>.zip: <arm>/paths/{paths.npz,episodes.csv}
and <arm>/render-env<id>-<camera>/snapshots/*.png, plus selection.json. Outputs:
paper/figures/fig_teaser_paths.pdf  (a) top view of one paired episode, (b) lateral error of all
                                    paired episodes over time
paper/figures/fig_teaser_snaps.pdf  snapshot strip, one row per arm (only if snapshots exist)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

PAPER = Path(__file__).resolve().parents[1]
FIG = PAPER / 'figures'
LABEL = {'dyn': 'HAIC', 'dyn-hboth': 'OH-Both (ours)', 'dyn-hnull': 'Null-heading', 'geo': 'Geo-Pose'}
COL = {'dyn': '#35609A', 'dyn-hboth': '#E8702A', 'dyn-hnull': '#B8B2AC', 'geo': '#8C6D4F'}  # as make_figures.py
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
COLW = 3.45  # IEEE column width (in)
TEXTW_CM = 18.1  # IEEE text width (two columns)
GLOBAL_POS = 'term/cum_body_pos_error'


def load(src, arm):
    d = src / arm / 'paths'
    z = dict(np.load(d / 'paths.npz'))
    with (d / 'episodes.csv').open(encoding='utf-8') as f:
        ep = {int(r['env_id']): r for r in csv.DictReader(f)}
    return z, ep


def lateral(z):
    """Signed lateral root error w.r.t. the reference heading, [T, n] (NaN after the episode ends)."""
    d = z['root_xyz'][..., :2] - z['ref_root_xyz'][..., :2]
    yaw = z['ref_root_yaw']
    return -np.sin(yaw) * d[..., 0] + np.cos(yaw) * d[..., 1]


def end_index(z, i):
    alive = ~np.isnan(z['root_xyz'][:, i, 0])
    return int(np.nonzero(alive)[0][-1])


def draw_topview(ax, data, arms, env_id, upright=False):
    """Root paths of one episode for every arm over the reference, equal axes. upright: the start
    heading (+x) points up and -y to the right, as seen by the chase camera of the snapshots."""
    xy = (lambda p: (-p[:, 1], p[:, 0])) if upright else (lambda p: (p[:, 0], p[:, 1]))
    # the reference is the same for both arms but is only recorded while the env is alive,
    # so take it from the arm that lasted longest
    zr = max((data[a][0] for a in arms), key=lambda z: (~np.isnan(z['ref_root_xyz'][:, env_id, 0])).sum())
    ref = zr['ref_root_xyz'][:, env_id, :2]
    ref = ref[~np.isnan(ref).any(-1)]
    ax.plot(*xy(ref), color='#7F7F7F', ls='--', lw=1.0, label='Reference', zorder=5)
    for a in arms:
        z, ep = data[a]
        p = z['root_xyz'][:, env_id, :2]
        p = p[~np.isnan(p).any(-1)]
        ok = int(ep[env_id]['success'])
        ax.plot(*xy(p), color=COL.get(a, 'C0'), lw=1.4,
                label=f'{LABEL.get(a, a)}: {"success" if ok else "terminated"}')
        if not ok:
            ax.plot(*xy(p[-1:]), marker='x', ms=7, mew=1.6, color=COL.get(a, 'C0'))
    ax.set_aspect('equal', adjustable='datalim')
    if upright:
        ax.set_xlabel('$-y$ [m]'); ax.set_ylabel('$x$ [m]')
    else:
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')


def fig_topview(src, arms, env_id, out, width_cm, height_cm=None, upright=False):
    """Stand-alone top view of the teaser: a strip under the snapshots, or upright beside them."""
    data = {a: load(src, a) for a in arms}
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, (height_cm or 3.0) / 2.54))
    draw_topview(ax, data, arms, env_id, upright)
    ax.set_aspect('equal', adjustable='box')
    if upright:
        ax.set_title('(b) Top view', loc='left')
        # no legend: the colours are those of the snapshot rows, the caption names the dashed reference
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.3, hi + 0.3)
    else:
        ax.set_title('(b) Top view of the same episode', loc='left')
        if width_cm > 16:  # room for the legend beside the plot
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
        else:  # the area below the reference start is empty
            ax.legend(frameon=False, loc='lower left', fontsize=6.5)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_paths(src, arms, env_id, out):
    data = {a: load(src, a) for a in arms}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2 * COLW, 1.9), gridspec_kw={'width_ratios': [1.25, 1]})
    # (a) top view of one paired episode
    draw_topview(ax0, data, arms, env_id)
    zr = data[arms[0]][0]
    ax0.set_title('(a) One episode, top view', loc='left')
    ax0.legend(frameon=False, loc='best')
    # (b) lateral error of every episode over time: thin lines, median and interquartile band
    dt = float(zr['step_dt'])
    for a in arms:
        z, ep = data[a]
        lat = 100 * lateral(z)
        t = np.arange(lat.shape[0]) * dt
        for i in range(lat.shape[1])[:: max(1, lat.shape[1] // 60)]:
            ax1.plot(t, lat[:, i], color=COL.get(a, 'C0'), lw=0.3, alpha=0.25)
        a_abs = np.abs(lat)
        with np.errstate(all='ignore'):
            med = np.nanmedian(a_abs, axis=1)
            q1, q3 = np.nanpercentile(a_abs, 25, axis=1), np.nanpercentile(a_abs, 75, axis=1)
        n = lat.shape[1]
        alive = (~np.isnan(lat)).sum(1) >= min(n, max(3, n // 10))  # statistics need enough live episodes
        ax1.fill_between(t[alive], q1[alive], q3[alive], color=COL.get(a, 'C0'), alpha=0.18, lw=0)
        n_fail = sum(1 for r in ep.values() if not int(r['success']))
        ax1.plot(t[alive], med[alive], color=COL.get(a, 'C0'), lw=1.4,
                 label=f'{LABEL.get(a, a)}: |median|, {n_fail}/{len(ep)} terminated')
    ax1.axhline(0, color='#BBBBBB', lw=0.6)
    ax1.set_xlabel('time [s]'); ax1.set_ylabel('lateral root error [cm]')
    ax1.set_title(f'(b) All {lat.shape[1]} episodes', loc='left')
    ax1.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_snaps(src, arms, env_id, out, n_cols, crop, camera, times=None, width_cm=TEXTW_CM, aspect=16 / 9, cy=0.5):
    rows = []
    for a in arms:
        d = sorted((src / a).glob(f'render-env{env_id}-{camera}/snapshots'))
        if not d:
            return False
        files = sorted(d[0].glob('*.png'))
        rows.append(files)
    # common time instants: those every arm reached, evenly thinned to n_cols, plus the baseline's end
    common = sorted(set.intersection(*[{f.name.split('_')[0] for f in r if not f.stem.endswith('_end')}
                                       for r in rows]))
    common = [c for c in common if c != 'step0000'] or common  # the renderer is not warmed up at step 0
    if times:  # nearest common snapshot to each requested time
        dt_ = {c: float(next(f for f in rows[0] if f.name.startswith(c + '_')).stem.split('_t')[1].split('s')[0])
               for c in common}
        pick = [min(common, key=lambda c: abs(dt_[c] - t)) for t in times]
    else:
        pick = [common[int(round(j))] for j in np.linspace(0, len(common) - 1, min(n_cols, len(common)))]
    fig, axes = plt.subplots(len(arms), len(pick) + 1, figsize=(width_cm / 2.54, 1.28 / aspect * len(arms) * width_cm / 2.54 / (len(pick) + 1)),
                             squeeze=False)
    for r, (a, files) in enumerate(zip(arms, rows)):
        by = {f.name.split('_')[0]: f for f in files if not f.stem.endswith('_end')}
        end = [f for f in files if f.stem.endswith('_end')]
        cells = [by.get(s) for s in pick] + [end[0] if end else None]
        for c, f in enumerate(cells):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(COL.get(a, 'C0') if c == len(cells) - 1 else '#CCCCCC')
            if f is None:
                ax.axis('off')
                continue
            img = plt.imread(f)
            h, w = img.shape[:2]
            cw = int(w * crop)
            ch = min(h, int(cw / aspect))  # width/height of the crop
            y0 = min(max(0, int(cy * h - ch / 2)), h - ch)
            img = img[y0:y0 + ch, (w - cw) // 2:(w + cw) // 2]
            ax.imshow(img)
            t = float(f.stem.split('_t')[1].split('s')[0])
            if c == len(cells) - 1:  # the end differs per arm: label it under the image
                ax.set_xlabel(f'end, t = {t:.1f} s', fontsize=7, labelpad=1)
            elif r == 0:
                ax.set_title(f't = {t:.1f} s', fontsize=7)
        axes[r, 0].set_ylabel(LABEL.get(a, a), color=COL.get(a, 'C0'), fontsize=8)
    fig.subplots_adjust(left=0.7 / width_cm, right=0.995, top=0.9, bottom=0.08, wspace=0.03, hspace=0.22)
    fig.text(0.0, 0.99, '(a) One episode, rendered (red: reference keypoints)', ha='left', va='bottom', fontsize=8)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['dyn', 'dyn-hboth'])
    p.add_argument('--env-id', type=int, default=None, help='default: first rendered episode of selection.json')
    p.add_argument('--cols', type=int, default=5, help='snapshot columns before the final frame')
    p.add_argument('--crop', type=float, default=0.6, help='central crop: fraction of the snapshot width')
    p.add_argument('--aspect', type=float, default=16 / 9, help='width/height of each cropped snapshot')
    p.add_argument('--cy', type=float, default=0.5, help='vertical centre of the crop (fraction of the height)')
    p.add_argument('--times', type=float, nargs='*', default=None,
                   help='snapshot times [s] before the final frame (default: --cols evenly spaced)')
    p.add_argument('--camera', default='chase', help='which render-env<id>-<camera> folder to use')
    p.add_argument('--out', type=Path, default=FIG)
    p.add_argument('--width-cm', type=float, default=TEXTW_CM, help='width of the snapshot and top-view parts')
    p.add_argument('--topview-cm', type=float, nargs=2, default=None,
                   help='width height of an upright top view beside the snapshots (default: strip below)')
    a = p.parse_args()
    sel = json.loads((a.src / 'selection.json').read_text(encoding='utf-8'))
    env_id = a.env_id if a.env_id is not None else (sel['rendered'] or sel['candidates'])[0]
    a.out.mkdir(parents=True, exist_ok=True)
    fig_paths(a.src, a.arms, env_id, a.out / 'fig_teaser_paths.pdf')
    fig_topview(a.src, a.arms, env_id, a.out / 'fig_teaser_topview.pdf', a.topview_cm[0] if a.topview_cm else a.width_cm,
                a.topview_cm[1] if a.topview_cm else None, upright=bool(a.topview_cm))
    made = fig_snaps(a.src, a.arms, env_id, a.out / 'fig_teaser_snaps.pdf', a.cols, a.crop, a.camera, a.times, a.width_cm, a.aspect, a.cy)
    print(f'env {env_id}: wrote fig_teaser_paths.pdf' + (' and fig_teaser_snaps.pdf' if made else
                                                         ' (no snapshots for this episode)'))


if __name__ == '__main__':
    main()
