"""Mechanism diagnostics from rollout traces (pure numpy/pandas; no simulator needed).

    python research/analyze.py --runs RUN_ROOT [RUN_ROOT ...] --labels geo dyn --out DIR

RUN_ROOT is a pipeline root (.../<task>-<arm>/teacher-s0). For every student and eval
seed it reads eval-s*/ (mode=pred, with trace.npz) and eval-<oracle mode>-s*/ if present.

Questions answered (numbers only; interpretation is left to the reader):
  1. Success / strict success per student, mean +- sd over student seeds.
  2. What terminates failed episodes, and are global-position failures "drift" failures
     (local body pose still fine at onset) or balance failures?
  3. Is robot heading error an early warning of later failure (AUC at fixed times,
     compared against local pose error and world-model errors)?
  4. How good is the world model: per-field R^2 of object_pred, and the reference-copy
     coefficient alpha in pred ~ true + alpha * (ref - true) (alpha=0: tracks the true
     object, alpha=1: copies the reference motion).
  5. Counterfactual oracles: success with perfect object state / perfect privileged feature.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

GLOBAL_POS = 'cum_body_pos_error'
LOCAL_POS = 'cum_body_pos_error_local'
LOCAL_ORI = 'cum_body_ori_error_local'
DEFAULT_MIN_STEPS = 25
EARLY_TIMES_S = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
WINDOW_S = 0.5
FEATURES = {
    '|heading_err|': ('heading_err', True),
    '|root_err_lat|': ('root_err_lat', True),
    '|root_err_along|': ('root_err_along', True),
    'local_pos_err': (f'term_err/{LOCAL_POS}', False),
    'local_ori_err': (f'term_err/{LOCAL_ORI}', False),
    'wm_pos_err': ('wm_pos_err', False),
    '|wm_yaw_err|': ('wm_yaw_err', True),
    'priv_rel_err': ('priv_rel_err', False),
}


# ---------------------------------------------------------------- statistics helpers
def auc(scores, labels):
    """Mann-Whitney AUC: P(score of a positive > score of a negative), ties count 1/2."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_ci(scores, labels, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(scores[idx], labels[idx])
        if math.isfinite(a):
            vals.append(a)
    if not vals:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reference_copy_alpha(true, pred, ref, min_gap):
    """Least-squares alpha in pred - true = alpha * (ref - true), over samples whose
    reference and true state differ by more than min_gap (otherwise alpha is unidentified)."""
    t, p, r = (np.asarray(x, float) for x in (true, pred, ref))
    if t.ndim == 1:
        t, p, r = t[:, None], p[:, None], r[:, None]
    gap = r - t
    ok = np.isfinite(gap).all(-1) & np.isfinite(p).all(-1) & (np.linalg.norm(gap, axis=-1) > min_gap)
    if ok.sum() < 10:
        return {'alpha': float('nan'), 'n': int(ok.sum()), 'frac_closer_to_ref': float('nan')}
    g, e = gap[ok], (p - t)[ok]
    alpha = float((g * e).sum() / (g * g).sum())
    closer = np.linalg.norm(p[ok] - r[ok], axis=-1) < np.linalg.norm(e, axis=-1)
    return {'alpha': alpha, 'n': int(ok.sum()), 'frac_closer_to_ref': float(closer.mean())}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def mean_sd(values):
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return float('nan'), float('nan')
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'))


# ---------------------------------------------------------------- loading
def load_eval(d: Path):
    s = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
    ep = pd.read_csv(d / 'episodes.csv').sort_values('env_id').reset_index(drop=True)
    tr = None
    if (d / 'trace.npz').is_file():
        with np.load(d / 'trace.npz') as z:
            tr = {k: z[k] for k in z.files}
        if 'priv_err' in tr:
            tr['priv_rel_err'] = tr['priv_err'] / np.maximum(tr['priv_norm'], 1e-6)
    wm = json.loads((d / 'wm_stats.json').read_text(encoding='utf-8')) if (d / 'wm_stats.json').is_file() else None
    min_steps = DEFAULT_MIN_STEPS
    cfg_path = d / 'resolved_cfg.yaml'
    if cfg_path.is_file():
        try:
            from omegaconf import OmegaConf
            c = OmegaConf.load(cfg_path)
            min_steps = int(c.task.termination[GLOBAL_POS].min_steps)
        except Exception:
            pass
    return {'dir': str(d), 'summary': s, 'episodes': ep, 'trace': tr, 'wm': wm,
            'step_dt': float(s.get('step_dt', 0.02)), 'min_steps': min_steps}


def load_run(root: Path):
    out = {'root': str(root), 'students': {}}
    for sd in sorted(root.glob('student-s*')):
        if not sd.is_dir() or 'incomplete' in sd.name:
            continue
        evals = {}
        for ed in sorted(sd.glob('eval-*')):
            if not (ed / 'summary.json').is_file() or 'incomplete' in ed.name:
                continue
            e = load_eval(ed)
            evals[(e['summary']['mode'], e['summary']['seed'])] = e
        if evals:
            out['students'][sd.name] = evals
    if not out['students']:
        raise SystemExit(f'no evaluated students under {root}')
    return out


# ---------------------------------------------------------------- analyses
def success_table(run):
    rows = []
    for sname, evals in run['students'].items():
        for (mode, seed), e in sorted(evals.items()):
            s = e['summary']
            rows.append({'student': sname, 'mode': mode, 'eval_seed': seed, 'n': s['n_envs'],
                         'success': s['success_rate'], 'strict_success': s['strict_success_rate']})
    df = pd.DataFrame(rows)
    per_student = df.groupby(['mode', 'student'])[['success', 'strict_success']].mean().reset_index()
    agg = per_student.groupby('mode').agg(
        success_mean=('success', 'mean'), success_sd=('success', 'std'),
        strict_mean=('strict_success', 'mean'), n_students=('student', 'count')).reset_index()
    return df, per_student, agg


TRACKING_METRICS = {
    # name: (trace key, transform, unit)
    'heading |err| (deg)': ('heading_err', lambda x: np.degrees(np.abs(x)), 'deg'),
    'root lateral |err| (m)': ('root_err_lat', np.abs, 'm'),
    'root along-track |err| (m)': ('root_err_along', np.abs, 'm'),
    'torso global pos err (m)': (f'term_err/{GLOBAL_POS}', lambda x: x, 'm'),
    'local body pos err (m)': (f'term_err/{LOCAL_POS}', lambda x: x, 'm'),
    'object global xy err (m)': ('obj_pos_err_w', lambda x: x, 'm'),
}


def tracking_errors(run):
    """Time-averaged tracking errors over alive steps (all first episodes), per student, plus the
    errors at the last recorded step of successful episodes. Mean over eval seeds per student."""
    per_student = {}
    for sname, evals in run['students'].items():
        acc = {m: [] for m in TRACKING_METRICS}
        acc_end = {m: [] for m in TRACKING_METRICS}
        for (mode, seed), e in evals.items():
            tr, ep = e['trace'], e['episodes']
            if mode != 'pred' or tr is None:
                continue
            succ = ep[ep['success'] == 1]
            for m, (key, f, _) in TRACKING_METRICS.items():
                if key not in tr:
                    continue
                x = f(tr[key])
                with np.errstate(all='ignore'):
                    acc[m].append(float(np.nanmean(x)))
                if len(succ):
                    end = succ['end_rollout_step'].to_numpy()
                    env = succ['env_id'].to_numpy()
                    acc_end[m].append(float(np.nanmean(x[end, env])))
        per_student[sname] = {'mean': {m: float(np.mean(v)) for m, v in acc.items() if v},
                              'end_success': {m: float(np.mean(v)) for m, v in acc_end.items() if v}}
    agg = {}
    for kind in ('mean', 'end_success'):
        agg[kind] = {m: mean_sd([ps[kind][m] for ps in per_student.values() if m in ps[kind]])
                     for m in TRACKING_METRICS}
    return {'per_student': per_student, 'agg': agg}


def failure_breakdown(run):
    """Termination flags among failures, and drift-vs-balance split of global failures."""
    counts, n_fail, drift, balance, onset_rows = {}, 0, 0, 0, []
    for sname, evals in run['students'].items():
        for (mode, seed), e in evals.items():
            if mode != 'pred':
                continue
            ep, tr = e['episodes'], e['trace']
            fails = ep[ep['success'] == 0]
            n_fail += len(fails)
            for c in [c for c in ep.columns if c.startswith('term/')]:
                counts[c[5:]] = counts.get(c[5:], 0) + int(fails[c].sum())
            gcol = f'term/{GLOBAL_POS}'
            if tr is None or gcol not in ep.columns:
                continue
            thr_local = 0.5  # hdmi-base threshold of cum_body_pos_error_local
            for _, r in fails[fails[gcol] == 1].iterrows():
                end = int(r['end_rollout_step'])
                onset = max(0, end - e['min_steps'] + 1)
                env = int(r['env_id'])
                loc = tr.get(f'term_err/{LOCAL_POS}')
                loc_v = float(loc[onset, env]) if loc is not None else float('nan')
                is_drift = bool(loc_v < 0.5 * thr_local) and int(r.get(f'term/{LOCAL_POS}', 0)) == 0 \
                    and int(r.get(f'term/{LOCAL_ORI}', 0)) == 0
                drift += is_drift
                balance += not is_drift
                onset_rows.append({
                    'student': sname, 'seed': seed, 'env': env, 'onset_step': onset,
                    'onset_time_s': onset * e['step_dt'], 'drift_type': is_drift,
                    'local_pos_err_at_onset': loc_v,
                    'heading_err_deg_at_onset': float(np.degrees(abs(tr['heading_err'][onset, env]))),
                    'lat_err_at_onset': float(abs(tr['root_err_lat'][onset, env])),
                    'along_err_at_onset': float(abs(tr['root_err_along'][onset, env])),
                })
    return {'n_failures': n_fail,
            'flag_fraction_among_failures': {k: (v / n_fail if n_fail else float('nan')) for k, v in counts.items()},
            'global_pos_failures': drift + balance, 'drift_type': drift, 'balance_type': balance,
            'onsets': pd.DataFrame(onset_rows)}


def early_warning(run, times=EARLY_TIMES_S, window_s=WINDOW_S):
    """AUC of each feature (window mean before time t) for predicting failure after t,
    among episodes still alive at t. Pooled over students and eval seeds."""
    pooled = {t: {f: ([], []) for f in FEATURES} for t in times}
    for evals in run['students'].values():
        for (mode, seed), e in evals.items():
            if mode != 'pred' or e['trace'] is None:
                continue
            ep, tr, dt = e['episodes'], e['trace'], e['step_dt']
            end = ep['end_rollout_step'].to_numpy()
            fail = (ep['success'] == 0).to_numpy()
            for t in times:
                k = int(round(t / dt))
                w = max(1, int(round(window_s / dt)))
                if k >= tr['heading_err'].shape[0]:
                    continue
                alive = end > k
                for f, (key, use_abs) in FEATURES.items():
                    if key not in tr:
                        continue
                    x = tr[key][max(0, k - w + 1):k + 1]
                    x = np.abs(x) if use_abs else x
                    with np.errstate(all='ignore'):
                        v = np.nanmean(x, axis=0)
                    pooled[t][f][0].extend(v[alive].tolist())
                    pooled[t][f][1].extend(fail[alive].tolist())
    rows = []
    for t in times:
        for f, (scores, labels) in pooled[t].items():
            if not scores:
                continue
            lo, hi = auc_ci(scores, labels)
            rows.append({'time_s': t, 'feature': f, 'auc': auc(scores, labels), 'ci_lo': lo, 'ci_hi': hi,
                         'n_alive': len(labels), 'n_fail_later': int(np.sum(labels))})
    return pd.DataFrame(rows)


def heading_profiles(run, max_s=12.0):
    """Median / quartiles of |heading error| vs time for successful and failed episodes,
    and failure-aligned profiles (time before termination onset)."""
    succ, fail, aligned = [], [], {'heading_deg': [], 'lat_m': [], 'local_pos_m': []}
    dt = None
    for evals in run['students'].values():
        for (mode, seed), e in evals.items():
            if mode != 'pred' or e['trace'] is None:
                continue
            tr, ep, dt = e['trace'], e['episodes'], e['step_dt']
            h = np.degrees(np.abs(tr['heading_err']))
            s_mask = (ep['success'] == 1).to_numpy()
            succ.append(h[:, s_mask])
            fail.append(h[:, ~s_mask])
            K = int(round(3.0 / dt))
            loc = tr.get(f'term_err/{LOCAL_POS}')
            for _, r in ep[ep['success'] == 0].iterrows():
                onset = int(r['end_rollout_step']) - e['min_steps'] + 1
                env = int(r['env_id'])
                if onset - K < 0:
                    continue
                aligned['heading_deg'].append(h[onset - K:onset + 1, env])
                aligned['lat_m'].append(np.abs(tr['root_err_lat'][onset - K:onset + 1, env]))
                if loc is not None:
                    aligned['local_pos_m'].append(loc[onset - K:onset + 1, env])
    if dt is None:
        return None
    T = max(a.shape[0] for a in succ + fail)
    pad = lambda a: np.vstack([a, np.full((T - a.shape[0], a.shape[1]), np.nan)])
    S = np.hstack([pad(a) for a in succ]) if succ else np.full((T, 0), np.nan)
    F = np.hstack([pad(a) for a in fail]) if fail else np.full((T, 0), np.nan)
    q = lambda M: np.nanpercentile(M, [25, 50, 75], axis=1) if M.shape[1] else np.full((3, T), np.nan)
    with np.errstate(all='ignore'):
        prof = {'t': np.arange(T) * dt, 'success_q': q(S), 'failure_q': q(F),
                'n_success': S.shape[1], 'n_failure': F.shape[1]}
        prof['aligned'] = {k: (np.nanpercentile(np.stack(v), [25, 50, 75], axis=0) if v else None)
                           for k, v in aligned.items()}
    prof['aligned_t'] = (np.arange(int(round(3.0 / dt)) + 1) - int(round(3.0 / dt))) * dt
    return prof


def world_model_fidelity(run):
    r2 = {}
    trues, preds, refs, yt, yp, yr = [], [], [], [], [], []
    for evals in run['students'].values():
        for (mode, seed), e in evals.items():
            if mode != 'pred':
                continue
            if e['wm'] is not None:
                for k, v in e['wm']['fields'].items():
                    r2.setdefault(k, []).append(v['r2_pooled'])
            tr = e['trace']
            if tr is None or 'pos_true_b' not in tr:
                continue
            m = np.isfinite(tr['pos_true_b']).all(-1)
            m[0] = False
            trues.append(tr['pos_true_b'][m][:, :2]); preds.append(tr['pos_pred_b'][m][:, :2])
            refs.append(tr['pos_ref_b'][m][:, :2])
            yt.append(tr['yaw_true_b'][m]); yp.append(tr['yaw_pred_b'][m]); yr.append(tr['yaw_ref_b'][m])
    out = {'r2_by_field_mean': {k: mean_sd(v)[0] for k, v in r2.items()},
           'r2_by_field_sd': {k: mean_sd(v)[1] for k, v in r2.items()}}
    if trues:
        t, p, r = np.vstack(trues), np.vstack(preds), np.vstack(refs)
        out['alpha_xy'] = reference_copy_alpha(t, p, r, min_gap=0.05)
        t, p, r = np.concatenate(yt), np.concatenate(yp), np.concatenate(yr)
        # express yaw differences relative to the true yaw so wrap-around does not matter
        out['alpha_yaw'] = reference_copy_alpha(np.zeros_like(t), wrap(p - t), wrap(r - t), min_gap=0.05)
    return out


# ---------------------------------------------------------------- report
def plot(results, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lab in labels:
        prof = results[lab]['profiles']
        if prof is None:
            continue
        for ax, key, name in ((axes[0], 'success_q', 'success'), (axes[1], 'failure_q', 'failure')):
            q = prof[key]
            ax.plot(prof['t'], q[1], label=f'{lab} (n={prof["n_" + name]})')
            ax.fill_between(prof['t'], q[0], q[2], alpha=0.2)
    for ax, name in zip(axes, ('successful episodes', 'failed episodes')):
        ax.set_title(f'|heading error| vs time, {name}'); ax.set_xlabel('time (s)'); ax.set_ylabel('deg')
        ax.legend()
    fig.tight_layout(); fig.savefig(out / 'heading_vs_time.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for lab in labels:
        prof = results[lab]['profiles']
        if prof is None:
            continue
        for ax, key, unit in zip(axes, ('heading_deg', 'lat_m', 'local_pos_m'), ('deg', 'm', 'm')):
            q = prof['aligned'][key]
            if q is None:
                continue
            ax.plot(prof['aligned_t'], q[1], label=lab)
            ax.fill_between(prof['aligned_t'], q[0], q[2], alpha=0.2)
            ax.set_title(f'{key} before termination onset'); ax.set_xlabel('time to onset (s)'); ax.set_ylabel(unit)
    for ax in axes:
        ax.legend()
    fig.tight_layout(); fig.savefig(out / 'failure_aligned.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        df = results[lab]['early_warning']
        for f, g in df.groupby('feature'):
            ax.plot(g['time_s'], g['auc'], marker='o', label=f)
        ax.axhline(0.5, color='gray', lw=0.8, ls='--')
        ax.set_ylim(0.3, 1.0); ax.set_title(f'{lab}: AUC for later failure'); ax.set_xlabel('time (s)')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'early_warning_auc.png', dpi=150); plt.close(fig)


def fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'n/a'
    return f'{100 * x:.2f}%' if pct else f'{x:.{nd}f}'


def report(results, out: Path):
    L = ['# HAIC mechanism diagnostics', '']
    L += ['## 1. Success rate (student mean ± sd over training seeds; each student averaged over eval seeds)', '',
          '| arm | mode | students | success | sd | strict success |', '|---|---|---|---|---|---|']
    for lab, r in results.items():
        for _, a in r['success_agg'].iterrows():
            L.append(f"| {lab} | {a['mode']} | {int(a['n_students'])} | {fmt(a['success_mean'], True)} | "
                     f"{fmt(a['success_sd'], True)} | {fmt(a['strict_mean'], True)} |")
    labels = list(results)
    if len(labels) > 1:
        ref = labels[0]
        L += ['', f'### Paired differences vs {ref} (same student seed; arms reusing the {ref} teacher share '
              'its teacher and initialization)', '',
              '| arm | mode | student seeds | per-seed diff (pp) | mean diff (pp) |', '|---|---|---|---|---|']
        base = results[ref]['success_per_student'].set_index(['mode', 'student'])['success']
        for lab in labels[1:]:
            cur = results[lab]['success_per_student'].set_index(['mode', 'student'])['success']
            for mode in sorted({m for m, _ in cur.index}):
                common = sorted(s for m, s in cur.index if m == mode and (mode, s) in base.index)
                if not common:
                    continue
                d = [100 * (cur[(mode, s)] - base[(mode, s)]) for s in common]
                L.append(f"| {lab} | {mode} | {', '.join(s.replace('student-s', '') for s in common)} | "
                         f"{', '.join(f'{x:+.2f}' for x in d)} | {np.mean(d):+.2f} |")
    L += ['', '## 1b. Tracking errors (student mean ± sd; time average over alive steps of all first episodes, '
          'and at the last step of successful episodes)', '',
          '| metric | ' + ' | '.join(f'{lab} (avg)' for lab in results) + ' | '
          + ' | '.join(f'{lab} (end, success)' for lab in results) + ' |',
          '|---' * (1 + 2 * len(results)) + '|']
    for m in TRACKING_METRICS:
        cells = []
        for kind in ('mean', 'end_success'):
            for lab, r in results.items():
                mu, sd = r['tracking']['agg'][kind][m]
                cells.append('n/a' if mu != mu else f'{mu:.3f}' + ('' if sd != sd else f' ± {sd:.3f}'))
        L.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 2. What ends failed episodes (fraction of failures carrying each flag; flags can co-occur)', '']
    for lab, r in results.items():
        fb = r['failures']
        L.append(f"**{lab}**: {fb['n_failures']} failures. " + ', '.join(
            f'{k}={fmt(v, True)}' for k, v in sorted(fb['flag_fraction_among_failures'].items(), key=lambda kv: -kv[1])))
        if fb['global_pos_failures']:
            on = fb['onsets']
            L.append(f"- global torso-position failures: {fb['global_pos_failures']}; drift-type (local pose error "
                     f"< 0.25 m at onset and no local termination) {fb['drift_type']} "
                     f"({fmt(fb['drift_type'] / fb['global_pos_failures'], True)}).")
            L.append(f"- at onset: median |heading| {on['heading_err_deg_at_onset'].median():.1f} deg, "
                     f"median |lateral| {on['lat_err_at_onset'].median():.3f} m, "
                     f"median |along-track| {on['along_err_at_onset'].median():.3f} m, "
                     f"median onset time {on['onset_time_s'].median():.2f} s.")
        L.append('')
    L += ['## 3. Early warning: AUC of the feature (0.5 s window mean) at time t for failing after t', '',
          'Episodes alive at t only; 95% bootstrap CI in brackets.', '']
    for lab, r in results.items():
        df = r['early_warning']
        if df.empty:
            continue
        piv = df.assign(cell=df.apply(lambda x: f"{x['auc']:.2f} [{x['ci_lo']:.2f},{x['ci_hi']:.2f}]", axis=1)) \
            .pivot(index='feature', columns='time_s', values='cell')
        L.append(f'**{lab}**'); L.append('')
        L.append('| feature | ' + ' | '.join(f'{t:g} s' for t in piv.columns) + ' |')
        L.append('|---' * (len(piv.columns) + 1) + '|')
        for f, row in piv.iterrows():
            L.append(f'| {f} | ' + ' | '.join(row.fillna('n/a')) + ' |')
        n = df.groupby('time_s')[['n_alive', 'n_fail_later']].first()
        L.append(''); L.append('alive / later-failing: ' + ', '.join(
            f"{t:g}s {int(a)}/{int(b)}" for t, (a, b) in n.iterrows())); L.append('')
    L += ['## 4. World-model fidelity', '',
          'R² of object_pred per object_ field (pooled over alive steps, mean over students × eval seeds). '
          'alpha: pred − true ≈ alpha·(ref − true) over samples where reference and true differ '
          '(> 5 cm / > 0.05 rad); 0 = tracks the true object, 1 = copies the reference.', '']
    for lab, r in results.items():
        w = r['world_model']
        L.append(f'**{lab}**: ' + ', '.join(f'{k} R²={fmt(v)}' for k, v in w['r2_by_field_mean'].items()))
        for k in ('alpha_xy', 'alpha_yaw'):
            if k in w:
                L.append(f"- {k}: {fmt(w[k]['alpha'])} (n={w[k]['n']}, pred closer to reference than to truth in "
                         f"{fmt(w[k]['frac_closer_to_ref'], True)})")
        L.append('')
    L += ['## Figures', '', '![](heading_vs_time.png)', '', '![](failure_aligned.png)', '', '![](early_warning_auc.png)', '']
    (out / 'report.md').write_text('\n'.join(L), encoding='utf-8')


def analyze_run(root: Path):
    run = load_run(root)
    df, per_student, agg = success_table(run)
    return {'run': run, 'success_rows': df, 'success_per_student': per_student, 'success_agg': agg,
            'failures': failure_breakdown(run), 'early_warning': early_warning(run),
            'profiles': heading_profiles(run), 'world_model': world_model_fidelity(run),
            'tracking': tracking_errors(run)}


def to_jsonable(results):
    out = {}
    for lab, r in results.items():
        fb = dict(r['failures']); fb['onsets'] = fb['onsets'].to_dict('records')
        out[lab] = {'root': r['run']['root'], 'success_rows': r['success_rows'].to_dict('records'),
                    'success_agg': r['success_agg'].to_dict('records'), 'failures': fb,
                    'early_warning': r['early_warning'].to_dict('records'), 'world_model': r['world_model'],
                    'tracking': r['tracking']}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    labels = a.labels or [r.parent.name for r in a.runs]
    assert len(labels) == len(a.runs)
    a.out.mkdir(parents=True, exist_ok=True)
    results = {lab: analyze_run(root) for lab, root in zip(labels, a.runs)}
    plot(results, a.out)
    report(results, a.out)
    (a.out / 'metrics.json').write_text(json.dumps(to_jsonable(results), indent=2, default=float), encoding='utf-8')
    print(f'ANALYSIS_REPORT={a.out / "report.md"}')


if __name__ == '__main__':
    main()
