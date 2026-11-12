"""Build every figure, table and number of the paper from the experiment outputs.

    python paper/scripts/make_figures.py

Inputs are the result folders downloaded from the training servers (paths below). Outputs:
paper/figures/*.pdf, paper/generated/numbers.tex (\\newcommand macros used in the text) and
paper/generated/tab_*.tex. Nothing in the paper is typed in by hand.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]          # D:/HAIC-Claude
PAPER = ROOT / 'paper'
sys.path.insert(0, str(ROOT / 'HAIC'))
from research import analyze as A  # noqa: E402

HEAD = ROOT / 'skateboard-dyn-dyn-hwm-dyn-hnull-dyn-hact-dyn-hboth-3students-new' / \
    'skateboard-dyn-dyn-hwm-dyn-hnull-dyn-hact-dyn-hboth-3students' / 'metrics.json'
BASE = ROOT / 'skateboard-geo-dyn-3students' / 'skateboard-geo-dyn-3students' / 'metrics.json'
R2 = ROOT / '_work' / 'r2'                           # round-2 archives (summaries + dyn/geo traces)
ABL = {a: ROOT / f'skateboard-{a}' / f'skateboard-{a}' / 'ablation.json' for a in ('dyn', 'geo')}
ROB = ROOT / 'skateboard' / 'robustness.json'
PC = ROOT / 'result' / 'push-cart'                  # push cart: diagnostics, ablation, robustness
# cross + cross-extra: the cart also has OH-Actor and Geo-Pose (own teacher); dyn / hboth / hnull are unchanged
PC_DIAG = PC / 'push_cart-dyn-dyn-hboth-dyn-hnull-dyn-hact-geo-3students'
PC_HEAD = PC_DIAG / 'push_cart-dyn-dyn-hboth-dyn-hnull-dyn-hact-geo-3students' / 'metrics.json'
PC_ABL = PC / 'push_cart-dyn' / 'push_cart-dyn' / 'ablation.json'
PC_ROB = PC / 'push_cart' / 'robustness.json'
BX = ROOT / 'result' / 'box'                         # box over stair and slope
# cross + cross-extra, as for the cart
BX_DIAG = BX / 'box_stairslope-dyn-dyn-hboth-dyn-hnull-dyn-hact-geo-3students'
BX_HEAD = BX_DIAG / 'box_stairslope-dyn-dyn-hboth-dyn-hnull-dyn-hact-geo-3students' / 'metrics.json'
BX_ABL = BX / 'box_stairslope-dyn' / 'box_stairslope-dyn' / 'ablation.json'
BX_ROB = BX / 'box_stairslope' / 'robustness.json'
CW = ROOT / 'skateboard-dyn-hboth-dyn-hboth-cw-dyn-hwm-dyn-hwm-cw-dyn-3students' / \
    'skateboard-dyn-hboth-dyn-hboth-cw-dyn-hwm-dyn-hwm-cw-dyn-3students' / 'metrics.json'  # control-weighted
FIG, GEN = PAPER / 'figures', PAPER / 'generated'
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

ARMS = ['dyn', 'dyn-hnull', 'dyn-hwm', 'dyn-hact', 'dyn-hboth']
LABEL = {'geo': 'Geo-Pose', 'dyn': 'HAIC', 'dyn-hnull': 'Null-heading', 'dyn-hwm': 'OH-WM',
         'dyn-hact': 'OH-Actor', 'dyn-hboth': 'OH-Both (ours)'}
# One palette for the whole paper (also paper/scripts/fig_showcase.py and the overview figure):
# HAIC blue and OH-Both orange as in Fig. 2, the other heading variants teal / purple, the capacity
# control a neutral gray. Checked for colour-vision deficiency (adjacent pairs: deutan dE >= 8.5).
COL = {'geo': '#8C6D4F', 'dyn': '#35609A', 'dyn-hnull': '#B8B2AC', 'dyn-hwm': '#2B9A8A',
       'dyn-hact': '#9C6FBF', 'dyn-hboth': '#E8702A'}
# evaluation conditions of one student: standard (its arm colour), drift-tolerant (lighter), oracle feature, teacher
COND = {'pred': '#35609A', 'pred_dt': '#8FB4DB', 'oracle_priv': '#D9A93A', 'teacher': '#8A8A8A'}
TASK_COL = {'Skateboard': '#6F5BA8', 'Push cart': '#C99A1E', 'Box, stair & slope': '#3E8E52'}
INK, ERR = '#333333', '#222222'  # text / axes and error bars
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#E5E5E5', 'grid.linewidth': 0.5,
                     'axes.axisbelow': True, 'errorbar.capsize': 2})
BAR = {'edgecolor': 'white', 'linewidth': 0.6}  # thin white gap between adjacent bars
COLW = 3.45  # IEEE column width (in)

head = json.loads(HEAD.read_text(encoding='utf-8'))
base = json.loads(BASE.read_text(encoding='utf-8'))
NUM = {}


def num(name, value, fmt='{:.2f}'):
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value


def per_student(metrics, arm, mode):
    """{student: mean success over eval seeds} for one arm and mode."""
    rows = [r for r in metrics[arm]['success_rows'] if r['mode'] == mode]
    out = {}
    for r in rows:
        out.setdefault(r['student'], []).append(r['success'])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def agg(metrics, arm, mode):
    v = list(per_student(metrics, arm, mode).values())
    return 100 * np.mean(v), 100 * np.std(v, ddof=1) if len(v) > 1 else float('nan')


def paired(a, b):
    """Paired t over student seeds (df = n-1); p exact for df = 2."""
    d = np.array([100 * (a[k] - b[k]) for k in sorted(a)])
    m, s = d.mean(), d.std(ddof=1)
    t = m / (s / math.sqrt(len(d)))
    p = 1 - abs(t) / math.sqrt(t * t + 2) if len(d) == 3 else float('nan')
    return d, m, s, t, p


def teacher_rates(arm):
    s = json.loads((R2 / arm / 'teacher-s0' / 'summary.json').read_text(encoding='utf-8'))
    return [e['success_rate'] for e in s['teacher_eval']]


# ------------------------------------------------------------------ numbers + tables
def macro_name(s):
    return ''.join(ch for ch in s.title() if ch.isalpha())


for arm in ['geo', 'dyn']:
    t = 100 * np.mean(teacher_rates(arm))
    num(f'teacher{macro_name(arm)}', t)
    for mode in ('pred', 'pred_dt', 'oracle_priv', 'oracle_object'):
        m, s = agg(base, arm, mode)
        num(f'{macro_name(arm)}{macro_name(mode)}', m)
        num(f'{macro_name(arm)}{macro_name(mode)}Sd', s)
for arm in ARMS:
    for mode in ('pred', 'pred_dt', 'oracle_priv', 'oracle_object'):
        m, s = agg(head, arm, mode)
        num(f'{macro_name(arm)}{macro_name(mode)}', m)
        num(f'{macro_name(arm)}{macro_name(mode)}Sd', s)

dyn_pred = per_student(head, 'dyn', 'pred')
stats = {}
for arm in ARMS[1:]:
    d, m, s, t, p = paired(per_student(head, arm, 'pred'), dyn_pred)
    stats[arm] = (d, m, s, t, p)
    num(f'{macro_name(arm)}Gain', m)
    num(f'{macro_name(arm)}GainP', p, '{:.3f}')
for a, b in (('dyn-hboth', 'dyn-hnull'), ('dyn-hboth', 'dyn-hact'), ('dyn-hboth', 'dyn-hwm')):
    d, m, s, t, p = paired(per_student(head, a, 'pred'), per_student(head, b, 'pred'))
    num(f'{macro_name(a)}Vs{macro_name(b)}', m)
    num(f'{macro_name(a)}Vs{macro_name(b)}P', p, '{:.3f}')
gap = float(NUM['DynOraclePriv']) - float(NUM['DynPred'])
num('gapDyn', gap)
num('gapClosedBoth', 100 * float(NUM['DynHbothGain']) / gap, '{:.0f}')
fb = base['dyn']['failures']
num('dynDriftShare', 100 * fb['drift_type'] / fb['global_pos_failures'], '{:.0f}')
num('dynGlobalShare', 100 * fb['flag_fraction_among_failures']['cum_body_pos_error'], '{:.0f}')
fbg = base['geo']['failures']
num('geoDriftShare', 100 * fbg['drift_type'] / fbg['global_pos_failures'], '{:.0f}')
num('geoGlobalShare', 100 * fbg['flag_fraction_among_failures']['cum_body_pos_error'], '{:.0f}')
wm = base['dyn']['world_model']['r2_by_field_mean']
for k, n in (('object_com_vel_b', 'rsqVel'), ('object_com_acc_b', 'rsqAcc'), ('object_pos_b', 'rsqPos'),
             ('object_ori_b', 'rsqOri')):
    num(n, wm[k])
num('alphaXyDyn', base['dyn']['world_model']['alpha_xy']['alpha'])
num('alphaYawDyn', base['dyn']['world_model']['alpha_yaw']['alpha'])

# table: main heading results
rows = []
for arm in ARMS:
    m, s = agg(head, arm, 'pred')
    mdt, sdt = agg(head, arm, 'pred_dt')
    if arm == 'dyn':
        gain = '---'
        pval = '---'
    else:
        d, gm, gs, t, p = stats[arm]
        gain = f'{gm:+.1f} ({", ".join(f"{x:+.1f}" for x in d)})'
        pval = f'{p:.3f}'
    tr = head[arm]['tracking']['agg']['mean']
    lat, along = tr['root lateral |err| (m)'][0], tr['root along-track |err| (m)'][0]
    best = ' \\bfseries' if arm == 'dyn-hboth' else ''
    hl = '\\rowcolor{ours} ' if arm == 'dyn-hboth' else ''  # our method: light gray row
    rows.append(f'{hl}{LABEL[arm]} &{best} {m:.2f} $\\pm$ {s:.2f} &{best} {gain} & {pval} & '
                f'{mdt:.1f} $\\pm$ {sdt:.1f} & {100 * lat:.1f} & {100 * along:.1f} \\\\')
(GEN / 'tab_main.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# table: privileged-input ablation
abl = {a: json.loads(p.read_text(encoding='utf-8')) for a, p in ABL.items()}
groups = ['global_root_ori', 'global_root_pos', 'object_drift', 'object_relative', 'local_tracking',
          'proprio_history', 'feet', 'actuation', 'contact', 'root_linvel']
gname = {'global_root_ori': 'Root orientation w.r.t.\\ reference (global)',
         'global_root_pos': 'Root position w.r.t.\\ reference (global)',
         'object_drift': 'Object pose w.r.t.\\ its reference (global)',
         'object_relative': 'Object pose w.r.t.\\ robot',
         'local_tracking': 'Body tracking errors (root frames)',
         'proprio_history': 'Proprioceptive history', 'feet': 'Feet states',
         'actuation': 'Applied actions / torques', 'contact': 'Contact schedule / offsets',
         'root_linvel': 'Root linear velocity'}
rows = []
for g in groups:
    rd = {r['group']: r for r in abl['dyn']['rows']}[g]
    rg = {r['group']: r for r in abl['geo']['rows']}[g]
    rows.append(f'{gname[g]} & {100 * rd["teacher"]:.1f} & {100 * rd["oracle_priv_ablated"]:.1f} & '
                f'{100 * rg["teacher"]:.1f} & {100 * rg["oracle_priv_ablated"]:.1f} \\\\')
(GEN / 'tab_ablation.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
GROUP_MACRO = {'global_root_ori': 'Ori', 'global_root_pos': 'Pos', 'object_drift': 'ObjDrift',
               'object_relative': 'ObjRel', 'local_tracking': 'Local'}
for g, n in GROUP_MACRO.items():
    r = {x['group']: x for x in abl['dyn']['rows']}[g]
    num(f'ablTeacher{n}', 100 * r['teacher'], '{:.0f}')
    num(f'ablOracle{n}', 100 * r['oracle_priv_ablated'], '{:.0f}')
num('ablPredBase', 100 * abl['dyn']['summary']['pred_base'])
ew = {(r['feature'], r['time_s']): r['auc'] for r in base['dyn']['early_warning']}
num('aucPrivSix', ew[('priv_rel_err', 6.0)])
num('aucObjSix', ew[('wm_pos_err', 6.0)])
num('dtFailRatio', (100 - float(NUM['GeoPredDt'])) / (100 - float(NUM['DynPredDt'])), '{:.0f}')
num('ablTeacherBase', 100 * abl['dyn']['summary']['teacher_base'])
for arm, n in (('dyn', 'Dyn'), ('dyn-hboth', 'Both'), ('dyn-hwm', 'Wm'), ('dyn-hact', 'Act')):
    w = head[arm]['world_model']
    num(f'rsqHead{n}', w['r2_by_field_mean']['object_heading_b'])
    num(f'rsqObjOri{n}', w['r2_by_field_mean']['object_ori_b'])
    num(f'alphaYaw{n}', w['alpha_yaw']['alpha'])
    tr = head[arm]['tracking']['agg']
    num(f'latAvg{n}', 100 * tr['mean']['root lateral |err| (m)'][0], '{:.1f}')
    num(f'alongAvg{n}', 100 * tr['mean']['root along-track |err| (m)'][0], '{:.1f}')
    num(f'latEnd{n}', 100 * tr['end_success']['root lateral |err| (m)'][0], '{:.1f}')
    num(f'headEnd{n}', tr['end_success']['heading |err| (deg)'][0], '{:.1f}')
    num(f'torsoEnd{n}', 100 * tr['end_success']['torso global pos err (m)'][0], '{:.1f}')
num('sdRatio', float(NUM['DynPredSd']) / float(NUM['DynHbothPredSd']), '{:.0f}')
num('latReduction', 100 * (1 - float(NUM['latEndBoth']) / float(NUM['latEndDyn'])), '{:.0f}')
fbb = head['dyn-hboth']['failures']
num('bothDriftShare', 100 * fbb['drift_type'] / fbb['global_pos_failures'], '{:.0f}')
num('bothFailRate', 100 - float(NUM['DynHbothPred']), '{:.1f}')
num('ablOracleBase', 100 * abl['dyn']['summary']['oracle_priv_base'])
num('ablGeoTeacherBase', 100 * abl['geo']['summary']['teacher_base'])
num('ablGeoOracleBase', 100 * abl['geo']['summary']['oracle_priv_base'])

# table: Geo-Pose vs HAIC
rows = []
for arm in ('geo', 'dyn'):
    t = 100 * np.mean(teacher_rates(arm))
    cells = [f'{t:.1f}']
    for mode in ('pred', 'pred_dt', 'oracle_priv', 'oracle_object'):
        m, s = agg(base, arm, mode)
        cells.append(f'{m:.1f} $\\pm$ {s:.1f}')
    rows.append(f'{LABEL[arm]} & ' + ' & '.join(cells) + ' \\\\')
(GEN / 'tab_geo_dyn.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# robustness numbers
rob = json.loads(ROB.read_text(encoding='utf-8'))
SCALE_NAME = {0: 'Zero', 1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 20: 'Twenty'}


def rob_rates(arm, s_):
    res = rob['results'][arm]
    return res[str(float(s_))] if str(float(s_)) in res else res[str(s_)]


for sc in rob['scales']:
    if float(sc).is_integer() and int(sc) in SCALE_NAME:
        num(f'robBoth{SCALE_NAME[int(sc)]}', 100 * np.mean(rob_rates('dyn-hboth', sc)))


def end_sigma_deg(r, s_):
    """1-sigma heading-sensor error at the episode end for error scale s_ (same model as fig_robustness)."""
    p = {k: v * s_ for k, v in r['noise_defaults'].items()}
    T = r['episode_s']
    return math.degrees(math.sqrt(p['init_align_std'] ** 2 + (p['bias_std'] * T) ** 2
                                  + p['bias_rw_std'] ** 2 * T ** 3 / 3 + p['rate_noise_std'] ** 2 * 0.02 * T))


num('robSigTwo', end_sigma_deg(rob, 2), '{:.1f}')
num('robSigFive', end_sigma_deg(rob, 5), '{:.1f}')

# ------------------------------------------------------------------ other tasks (push cart, box)
TASK_ARMS = ['dyn', 'dyn-hnull', 'dyn-hboth']


def pstr(p):
    """'=0.012' or '<0.001'; the text writes $p<macro>$ without the relation."""
    return '<0.001' if p < 0.001 else f'={p:.3f}'


def rob_curve(r):
    """{scale: mean OH-Both success in %} of one robustness.json."""
    return {float(k): 100 * np.mean(v) for k, v in r['results']['dyn-hboth'].items()}


def break_even_deg(r, base):
    """1-sigma end heading error at which OH-Both falls to the HAIC level (linear between scales)."""
    c = sorted(rob_curve(r).items())
    for (s0, m0), (s1, m1) in zip(c, c[1:]):
        if m0 >= base > m1:
            return end_sigma_deg(r, s0 + (m0 - base) / (m0 - m1) * (s1 - s0))
    return float('nan')


def task_numbers(pre, met, ab, rb, arms=TASK_ARMS):
    """Macros \\<pre>... of one cross-task replication (diagnostics, ablation, robustness)."""
    for arm in arms:
        for mode in ('pred', 'pred_dt', 'oracle_priv', 'oracle_object'):
            num(f'{pre}{macro_name(arm)}{macro_name(mode)}', agg(met, arm, mode)[0])
    base_ = per_student(met, 'dyn', 'pred')
    st = {arm: paired(per_student(met, arm, 'pred'), base_) for arm in arms if arm != 'dyn'}
    for arm, (d, m, s, t, p) in st.items():
        num(f'{pre}{macro_name(arm)}Gain', m)
        num(f'{pre}{macro_name(arm)}GainP', pstr(p))
    d, m, s, t, p = paired(per_student(met, 'dyn-hboth', 'pred'), per_student(met, 'dyn-hnull', 'pred'))
    num(f'{pre}DynHbothVsDynHnull', m)
    num(f'{pre}DynHbothVsDynHnullP', pstr(p))
    # drift gap = drift-tolerant minus standard success of HAIC (threshold-free measure of drift failures)
    num(f'{pre}DriftGap', float(NUM[f'{pre}DynPredDt']) - float(NUM[f'{pre}DynPred']), '{:.1f}')
    num(f'{pre}DriftGapClosed', 100 * float(NUM[f'{pre}DynHbothGain']) / float(NUM[f'{pre}DriftGap']), '{:.0f}')
    num(f'{pre}OracleObjGain', float(NUM[f'{pre}DynOracleObject']) - float(NUM[f'{pre}DynPred']), '{:.1f}')
    for arm, n in (('dyn', 'Dyn'), ('dyn-hboth', 'Both')):
        tr = met[arm]['tracking']['agg']['end_success']
        num(f'{pre}HeadEnd{n}', tr['heading |err| (deg)'][0], '{:.1f}')
        num(f'{pre}LatEnd{n}', 100 * tr['root lateral |err| (m)'][0], '{:.1f}')
        num(f'{pre}AlongEnd{n}', 100 * tr['root along-track |err| (m)'][0], '{:.1f}')
        f = met[arm]['failures']
        fr = f['flag_fraction_among_failures']
        num(f'{pre}ContactShare{n}', 100 * fr['cum_lost_contact_steps'], '{:.0f}')
        num(f'{pre}ObjPosShare{n}', 100 * fr.get('cum_object_pos_error', 0.0), '{:.0f}')
        num(f'{pre}TorsoPosShare{n}', 100 * fr['cum_body_pos_error'], '{:.0f}')
        num(f'{pre}DriftShare{n}', 100 * f['drift_type'] / f['global_pos_failures'], '{:.0f}')
        num(f'{pre}OnsetAlong{n}', 100 * np.median([abs(o['along_err_at_onset']) for o in f['onsets']]), '{:.0f}')
        num(f'{pre}AlphaYaw{n}', met[arm]['world_model']['alpha_yaw']['alpha'])
    num(f'{pre}LocalAvgDyn', 100 * met['dyn']['tracking']['agg']['mean']['local body pos err (m)'][0], '{:.0f}')
    num(f'{pre}RsqXyDyn', met['dyn']['world_model']['r2_by_field_mean']['object_xy_b'])
    num(f'{pre}RsqXyBoth', met['dyn-hboth']['world_model']['r2_by_field_mean']['object_xy_b'])
    lost = {r['group']: r['oracle_gain_lost'] for r in ab['rows']}
    for g, n in (('global_root_ori', 'Ori'), ('global_root_pos', 'Pos'), ('object_drift', 'ObjDrift')):
        num(f'{pre}Lost{n}', lost[g])
    curve = rob_curve(rb)
    for sc_, name in SCALE_NAME.items():
        if float(sc_) in curve:
            num(f'{pre}RobBoth{name}', curve[float(sc_)])
            num(f'{pre}RobSig{name}', end_sigma_deg(rb, sc_), '{:.1f}')
    num(f'{pre}RobFiveMargin', curve[5.0] - 100 * rb['dyn_baseline'], '{:.0f}')
    num(f'{pre}RobGainTwo', curve[2.0] - 100 * rb['dyn_baseline'], '{:.0f}')
    num(f'{pre}RobBreakEven', break_even_deg(rb, 100 * rb['dyn_baseline']), '{:.0f}')
    return st


def load_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


TASKS_X = {'pc': (load_json(PC_HEAD), load_json(PC_ABL), load_json(PC_ROB)),
           'bx': (load_json(BX_HEAD), load_json(BX_ABL), load_json(BX_ROB))}
XTASK_ARMS = ['geo', 'dyn', 'dyn-hnull', 'dyn-hact', 'dyn-hboth']   # table order of the cart and the box
task_stats = {pre: task_numbers(pre, *v, arms=XTASK_ARMS) for pre, v in TASKS_X.items()}
# extras of both tasks: OH-Both against OH-Actor, the teachers of HAIC and Geo-Pose, Geo-Pose's drift and object estimate
for pre, task, diag in (('pc', 'push_cart', PC_DIAG), ('bx', 'box_stairslope', BX_DIAG)):
    met_ = TASKS_X[pre][0]
    d_, m_, s_, t_, p_ = paired(per_student(met_, 'dyn-hboth', 'pred'), per_student(met_, 'dyn-hact', 'pred'))
    num(f'{pre}DynHbothVsDynHact', m_)
    num(f'{pre}DynHbothVsDynHactP', pstr(p_))
    for arm in ('dyn', 'geo'):
        ts = load_json(diag / 'runs' / f'{task}-{arm}' / 'teacher-s0' / 'summary.json')
        num(f'{pre}Teacher{macro_name(arm)}', 100 * np.mean([e['success_rate'] for e in ts['teacher_eval']]), '{:.1f}')
    fg = met_['geo']['failures']
    num(f'{pre}GeoDriftShare', 100 * fg['drift_type'] / fg['global_pos_failures'], '{:.0f}')
    num(f'{pre}GeoDriftGap', float(NUM[f'{pre}GeoPredDt']) - float(NUM[f'{pre}GeoPred']), '{:.1f}')
    num(f'{pre}RsqXyGeo', met_['geo']['world_model']['r2_by_field_mean']['object_xy_b'])
    dts = [agg(met_, arm, 'pred_dt')[0] for arm in XTASK_ARMS]
    num(f'{pre}DtRange', max(dts) - min(dts), '{:.1f}')
num('skDriftGap', float(NUM['DynPredDt']) - float(NUM['DynPred']), '{:.1f}')
num('skDriftGapClosed', 100 * float(NUM['DynHbothGain']) / float(NUM['skDriftGap']), '{:.0f}')
num('skRobBreakEven', break_even_deg(rob, 100 * rob['dyn_baseline']), '{:.0f}')
sk_lost = {r['group']: r['oracle_gain_lost'] for r in abl['dyn']['rows']}
for g, n in (('global_root_ori', 'Ori'), ('global_root_pos', 'Pos'), ('object_drift', 'ObjDrift')):
    num(f'skLost{n}', sk_lost[g])

# control-weighted latent alignment (skateboard): CW arms against the same arms without weighting
cw = load_json(CW)
for arm, n in (('dyn-hboth', 'Both'), ('dyn-hboth-cw', 'BothCw'), ('dyn-hwm', 'Wm'), ('dyn-hwm-cw', 'WmCw')):
    for mode in ('pred', 'oracle_object'):
        m_, s_ = agg(cw, arm, mode)
        num(f'cw{n}{macro_name(mode)}', m_)
        num(f'cw{n}{macro_name(mode)}Sd', s_)
for a_, b_, n in (('dyn-hboth-cw', 'dyn-hboth', 'BothCwVsBoth'), ('dyn-hwm-cw', 'dyn-hwm', 'WmCwVsWm'),
                  ('dyn-hwm-cw', 'dyn-hboth', 'WmCwVsBoth')):
    d_, m_, s_, t_, p_ = paired(per_student(cw, a_, 'pred'), per_student(cw, b_, 'pred'))
    num(f'cw{n}', m_)
    num(f'cw{n}P', pstr(p_))
    num(f'cw{n}Seeds', ', '.join(f'{x:+.1f}' for x in d_))

# table: cross-task replication
rows = []
for met, st, name, arms in ((head, stats, 'Skateboard', TASK_ARMS),
                            (TASKS_X['pc'][0], task_stats['pc'], 'Push cart', XTASK_ARMS),
                            (TASKS_X['bx'][0], task_stats['bx'], 'Box, stair \\& slope', XTASK_ARMS)):
    if rows:
        rows.append('\\midrule')
    for j, arm in enumerate(arms):
        lead = f'\\multirow{{{len(arms)}}}{{*}}{{{name}}}' if j == 0 else ''
        m, s = agg(met, arm, 'pred')
        mdt, sdt = agg(met, arm, 'pred_dt')
        mob, _ = agg(met, arm, 'oracle_object')
        if arm == 'dyn':
            gain, pval = '---', '---'
        else:
            dd, gm, gs, t, p = st[arm]
            gain = f'{gm:+.1f} ({", ".join(f"{x:+.1f}" for x in dd)})'
            pval = '$<$0.001' if p < 0.001 else f'{p:.3f}'
        tr = met[arm]['tracking']['agg']['end_success']
        best = ' \\bfseries' if arm == 'dyn-hboth' else ''
        lab = LABEL[arm] + ('$^\\dagger$' if arm == 'geo' else '')  # own teacher
        cells = [lab, f'{best} {m:.2f} $\\pm$ {s:.2f}', f'{best} {gain}', pval, f'{mdt:.1f}', f'{mob:.1f}',
                 f'{100 * tr["root lateral |err| (m)"][0]:.1f}', f'{100 * tr["root along-track |err| (m)"][0]:.1f}']
        if arm == 'dyn-hboth':  # our method: light gray, without the task column shared by the rows of a task
            cells = ['\\cellcolor{ours}' + c for c in cells]
        rows.append(f'{lead} & ' + ' & '.join(cells) + ' \\\\')
(GEN / 'tab_tasks.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(FIG / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# Fig: standard vs drift-tolerant success (where the gap lives)
fig, ax = plt.subplots(figsize=(COLW, 1.75))
cats = ['Geo-Pose', 'HAIC']
x = np.arange(2)
w = 0.2
vals = {
    'Student (standard)': [agg(base, a, 'pred') for a in ('geo', 'dyn')],
    'Student (drift-tolerant)': [agg(base, a, 'pred_dt') for a in ('geo', 'dyn')],
    'Student + oracle priv. feature': [agg(base, a, 'oracle_priv') for a in ('geo', 'dyn')],
    'Teacher': [(100 * np.mean(teacher_rates(a)), float('nan')) for a in ('geo', 'dyn')],
}
cols = [COND['pred'], COND['pred_dt'], COND['oracle_priv'], COND['teacher']]
for i, (lab, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1.5) * w, [m for m, _ in v], w, yerr=[0 if s != s else s for _, s in v], color=cols[i],
           label=lab, error_kw={'lw': 0.7, 'ecolor': ERR}, **BAR)
ax.set_xticks(x, cats)
ax.set_ylabel('Success rate (%)')
ax.set_ylim(40, 102)
ax.legend(ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1)
save(fig, 'fig_gap')

# Fig: failure-aligned signals (HAIC students, global-position failures)
run = A.analyze_run(R2 / 'dyn' / 'teacher-s0')
prof = run['profiles']
fig, axes = plt.subplots(1, 3, figsize=(2 * COLW, 1.45))
for ax, key, title, unit in zip(axes, ('heading_deg', 'lat_m', 'local_pos_m'),
                                ('|Heading error|', '|Lateral root error|', 'Local body error (max)'),
                                ('deg', 'm', 'm')):
    q = prof['aligned'][key]
    t = prof['aligned_t']
    ax.plot(t, q[1], color=COL['dyn'], lw=1.2)
    ax.fill_between(t, q[0], q[2], color=COL['dyn'], alpha=0.2, lw=0)
    ax.set_title(title)
    ax.set_xlabel('Time to termination onset (s)')
    ax.set_ylabel(unit)
axes[2].set_ylim(0, 0.5)
axes[2].axhline(0.5, color='gray', lw=0.6, ls='--')
fig.tight_layout(w_pad=1.0)
save(fig, 'fig_failure_aligned')

# Fig: privileged-input ablation (teacher and oracle-feature student, HAIC)
fig, ax = plt.subplots(figsize=(COLW, 2.1))
rd = {r['group']: r for r in abl['dyn']['rows']}
yy = np.arange(len(groups))[::-1]
ax.grid(axis='y', visible=False)
ax.grid(axis='x', visible=True)
# the three groups whose loss breaks the teacher: global references (shaded band)
ax.axhspan(len(groups) - 3.5, len(groups) - 0.5, color='#F3EEE6', lw=0, zorder=0)
ax.barh(yy + 0.18, [100 * rd[g]['teacher'] for g in groups], 0.36, color=COND['teacher'], label='Teacher', **BAR)
ax.barh(yy - 0.18, [100 * rd[g]['oracle_priv_ablated'] for g in groups], 0.36, color=COND['oracle_priv'],
        label='Student + oracle priv. feature', **BAR)
pb = 100 * abl['dyn']['summary']['pred_base']
ax.axvline(pb, color=COL['dyn'], lw=0.9, ls='--')
ax.text(pb - 1, len(groups) - 0.55, 'deployable\nstudent', fontsize=6, va='top', ha='right', color=COL['dyn'])
ax.set_yticks(yy, [gname[g].replace('\\ ', ' ').replace('w.r.t.', 'wrt') for g in groups], fontsize=6)
ax.set_xlabel('Success with the group replaced (%)')
ax.set_xlim(0, 102)
ax.set_xticks([0, 20, 40, 60, pb, 100], ['0', '20', '40', '60', f'{pb:.1f}', '100'])
ax.get_xticklabels()[4].set_color(COL['dyn'])
for yi, g in zip(yy, groups):
    for off, v in ((0.18, rd[g]['teacher']), (-0.18, rd[g]['oracle_priv_ablated'])):
        if 100 * v < 2:
            ax.text(1, yi + off, f'{100 * v:.0f}%', fontsize=5.5, va='center')
ax.legend(frameon=False, ncol=2, fontsize=6, loc='lower center', bbox_to_anchor=(0.3, 1.0))
save(fig, 'fig_ablation')

# Fig: main paired results
fig, axes = plt.subplots(1, 2, figsize=(2 * COLW, 1.7), gridspec_kw={'width_ratios': [3, 2]})
ax = axes[0]
for i, arm in enumerate(ARMS):
    ps = list(per_student(head, arm, 'pred').values())
    m, s = agg(head, arm, 'pred')
    ax.bar(i, m, 0.6, color=COL[arm], **BAR)
    ax.errorbar(i, m, yerr=s, color=ERR, lw=0.7)
    ax.scatter(np.full(len(ps), i) + np.linspace(-0.12, 0.12, len(ps)), [100 * p for p in ps], s=7, color='white',
               edgecolor=ERR, lw=0.5, zorder=3)
op = float(NUM['DynOraclePriv'])
ax.axhline(op, color=COND['oracle_priv'], ls='--', lw=0.9)
ax.text(-0.45, op + 0.8, 'HAIC + oracle priv. feature', fontsize=6, color='#9A7420')
ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], rotation=15)
ax.set_ylabel('Success rate (%)')
ax.set_ylim(65, 101)
ax.set_yticks([70, 80, 90, op], ['70', '80', '90', f'{op:.1f}'])
ax.get_yticklabels()[3].set_color('#9A7420')
ax = axes[1]
tr = {a: head[a]['tracking']['agg']['mean'] for a in ARMS}
xx = np.arange(len(ARMS))
plt.rcParams['hatch.color'] = 'white'
for j, (k, lab, hatch) in enumerate((('root lateral |err| (m)', 'lateral', ''), ('root along-track |err| (m)', 'along-track', '////'))):
    ax.bar(xx + (j - 0.5) * 0.38, [100 * tr[a][k][0] for a in ARMS], 0.38,
           yerr=[100 * tr[a][k][1] for a in ARMS], color=[COL[a] for a in ARMS], hatch=hatch,
           error_kw={'lw': 0.6, 'ecolor': ERR, 'capsize': 1.5}, **BAR)
ax.set_xticks(xx, [LABEL[a].replace(' (ours)', '') for a in ARMS], rotation=30, fontsize=6)
ax.set_ylabel('Root error, time avg. (cm)')
ax.set_ylim(0, 13)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(facecolor='#8A8A8A', label='lateral'),
                   Patch(facecolor='#8A8A8A', hatch='////', edgecolor='white', label='along-track')],
          frameon=False, fontsize=6, ncol=2, loc='upper right')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig_main')

# Fig: robustness to heading-sensor error
fig, ax = plt.subplots(figsize=(COLW, 1.6))
sc = rob['scales']
sig = []
for s_ in sc:
    p = {k: v * s_ for k, v in rob['noise_defaults'].items()}
    T = rob['episode_s']
    sig.append(math.degrees(math.sqrt(p['init_align_std'] ** 2 + (p['bias_std'] * T) ** 2
                                      + p['bias_rw_std'] ** 2 * T ** 3 / 3 + p['rate_noise_std'] ** 2 * 0.02 * T)))
for arm in ('dyn-hboth', 'dyn-hact', 'dyn-hwm'):
    r = [rob_rates(arm, s_) for s_ in sc]
    m = [100 * np.mean(v) for v in r]
    s = [100 * np.std(v, ddof=1) for v in r]
    ax.errorbar(sig, m, yerr=s, color=COL[arm], marker={'dyn-hboth': 'o', 'dyn-hact': 's', 'dyn-hwm': '^'}[arm],
                ms=3.5, lw=1.1, label=LABEL[arm])
hb = 100 * rob['dyn_baseline']
num('robHaic', hb, '{:.1f}')
ax.axhline(hb, color=COL['dyn'], ls='--', lw=0.9, label=f'HAIC, no heading ({hb:.1f}%)')
ax.axvspan(0, sig[sc.index(2)] if 2 in sc else 3.4, color='#EDEDED', lw=0, zorder=0)
ax.set_xlabel('Heading-sensor error at episode end, 1$\\sigma$ (deg)')
ax.set_ylabel('Success rate (%)')
ax.legend(frameon=False, fontsize=6, loc='lower left')
save(fig, 'fig_robustness')

# Fig: OH-Both gain over HAIC under heading-sensor error, all tasks
fig, ax = plt.subplots(figsize=(COLW, 1.6))
for name, r, mk in (('Skateboard', rob, 'o'), ('Push cart', TASKS_X['pc'][2], 's'),
                     ('Box, stair & slope', TASKS_X['bx'][2], '^')):
    c = TASK_COL[name]
    sc_ = sorted(float(k) for k in r['results']['dyn-hboth'])
    v = [r['results']['dyn-hboth'][k] for k in (str(int(x)) if str(int(x)) in r['results']['dyn-hboth'] else str(x)
                                               for x in sc_)]
    g = [100 * (np.mean(x) - r['dyn_baseline']) for x in v]
    e = [100 * np.std(x, ddof=1) for x in v]
    ax.errorbar([end_sigma_deg(r, x) for x in sc_], g, yerr=e, color=c, marker=mk, ms=3, lw=1, capsize=2, label=name)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.text(0.3, 1.5, 'HAIC level', fontsize=6)
ax.set_xlabel('Heading-sensor error at episode end, 1$\\sigma$ (deg)')
ax.set_ylabel('OH-Both $-$ HAIC (pp)')
ax.legend(frameon=False, fontsize=6, loc='lower left')
save(fig, 'fig_robustness_tasks')

# Fig 1 (c): success of HAIC and OH-Both on the three tasks, with HAIC's drift-tolerant success
TEASER_C_CM = (4.0, 5.3)  # right column of the teaser, next to the snapshots and the top view
fig, ax = plt.subplots(figsize=(TEASER_C_CM[0] / 2.54, TEASER_C_CM[1] / 2.54))
tasks = (('Skate-\nboard', head), ('Push\ncart', TASKS_X['pc'][0]), ('Box,\nstair', TASKS_X['bx'][0]))
w = 0.38
for i, (name, met) in enumerate(tasks):
    for j, arm in enumerate(('dyn', 'dyn-hboth')):
        m, s = agg(met, arm, 'pred')
        ps = list(per_student(met, arm, 'pred').values())
        x = i + (j - 0.5) * w
        ax.bar(x, m, w, color=COL[arm], label=LABEL[arm] if i == 0 else None, **BAR)
        ax.scatter(np.full(len(ps), x), [100 * p for p in ps], s=5, color='white', edgecolor=ERR, lw=0.4, zorder=3)
        ax.text(x, max(m, 100 * max(ps)) + 0.8, f'{m:.0f}', ha='center', va='bottom', fontsize=6)
    mdt, _ = agg(met, 'dyn', 'pred_dt')
    ax.plot([i - w, i], [mdt, mdt], color='k', ls=':', lw=1.0,
            label='HAIC, drift-tolerant' if i == 0 else None)
ax.set_xticks(range(len(tasks)), [t for t, _ in tasks], fontsize=7)
ax.set_ylabel('Success rate (%)')
ax.set_ylim(60, 118)
ax.set_yticks(range(60, 101, 10))
ax.legend(frameon=False, fontsize=6, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.02))
ax.set_title('(c) Three tasks', loc='left')
save(fig, 'fig_teaser_tasks')

# ------------------------------------------------------------------ write macros
lines = ['% generated by paper/scripts/make_figures.py -- do not edit']
for k, v in sorted(NUM.items()):
    assert k.isalpha(), k
    lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
(GEN / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'{len(NUM)} numbers, figures in {FIG}')
for k in sorted(NUM):
    print(f'  {k} = {NUM[k]}')
