"""Counterfactual privileged-input ablation: which privileged information does the student lack?

    python research/ablate.py --task skateboard --arm dyn
    python research/ablate.py --task skateboard --arm geo --groups global_root local_tracking

For every group in research/common.py ABLATION_GROUPS (training-free, one evaluation each):
  teacher           the stage-1 teacher with that group replaced by another environment's values
                    -> how much the teacher relies on it
  oracle_priv       each student with the teacher's privileged feature (oracle_priv), the feature
                    computed with that group replaced
                    -> how much of the oracle benefit that group's information carries

  oracle gain lost = (oracle_priv - oracle_priv_ablated) / (oracle_priv - pred)
  1.0: without this group's true values the oracle is no better than the deployable student,
       i.e. this is the information the student's world model fails to recover; 0.0: irrelevant.

Writes $HAIC_OUT/ablation/<task>-<arm>/ablation.md, ablation.json and a zip next to it.
"""
import argparse
import json
import os
import statistics
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import OUT_ROOT, TASKS, ABLATION_GROUPS, read_json, write_json
from research.pipeline import run_eval


def finite_or_none(obj):
    """JSON-safe copy: NaN / inf (e.g. a zero oracle gain, a single-sample sd) become null."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float('inf'), float('-inf')) else None
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    return obj


def mean(xs):
    return statistics.mean(xs) if xs else float('nan')


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float('nan')


def aggregate(groups, teacher, students):
    """teacher: {'base': [rate per eval seed], group: [...]} or None;
    students: {student: {'pred': [...], 'oracle': [...], group: [...]}}."""
    rows = []
    t_base = mean(teacher['base']) if teacher else float('nan')
    for g in groups:
        row = {'group': g, 'terms': ABLATION_GROUPS[g]}
        if teacher and g in teacher:
            row['teacher'] = mean(teacher[g])
            row['teacher_drop'] = t_base - row['teacher']
        per_student, lost = [], []
        for st in students.values():
            if g not in st:
                continue
            o, p, a = mean(st['oracle']), mean(st['pred']), mean(st[g])
            per_student.append(a)
            if o - p > 1e-9:
                lost.append((o - a) / (o - p))
        row['oracle_priv_ablated'] = mean(per_student)
        row['oracle_priv_ablated_sd'] = sd(per_student)
        row['oracle_gain_lost'] = mean(lost)
        row['oracle_gain_lost_sd'] = sd(lost)
        rows.append(row)
    summary = {
        'teacher_base': t_base,
        'oracle_priv_base': mean([mean(s['oracle']) for s in students.values()]),
        'pred_base': mean([mean(s['pred']) for s in students.values()]),
        'n_students': len(students),
    }
    return summary, rows


def report(task, arm, summary, rows, eval_seeds):
    pct = lambda x: 'n/a' if x != x else f'{100 * x:.2f}%'
    L = [f'# Privileged-input ablation: {task}-{arm}', '',
         f"Baselines: teacher {pct(summary['teacher_base'])}, student oracle_priv {pct(summary['oracle_priv_base'])}, "
         f"student pred {pct(summary['pred_base'])} ({summary['n_students']} students, eval seeds {eval_seeds}).", '',
         'Each group is replaced by the values of one other environment at the same step (fixed derangement). '
         '"oracle gain lost" = (oracle_priv - ablated) / (oracle_priv - pred): 1 = without the true values of this '
         'group the oracle is no better than the deployable student.', '',
         '| group | teacher | teacher drop | oracle_priv ablated | oracle gain lost |', '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -(r['oracle_gain_lost'] if r['oracle_gain_lost'] == r['oracle_gain_lost'] else -9)):
        lost = r['oracle_gain_lost']
        lost_s = 'n/a' if lost != lost else f"{lost:.2f}" + ('' if r['oracle_gain_lost_sd'] != r['oracle_gain_lost_sd']
                                                             else f" ± {r['oracle_gain_lost_sd']:.2f}")
        L.append(f"| {r['group']} | {pct(r.get('teacher', float('nan')))} | {pct(r.get('teacher_drop', float('nan')))} | "
                 f"{pct(r['oracle_priv_ablated'])} | {lost_s} |")
    L += ['', 'Terms per group:', '']
    L += [f"- {r['group']}: " + ', '.join(f'{g}.{k}' for g, k in r['terms']) for r in rows]
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=sorted(TASKS), default='skateboard')
    p.add_argument('--arm', default='dyn')
    p.add_argument('--teacher-seed', type=int, default=0)
    p.add_argument('--groups', nargs='+', default=list(ABLATION_GROUPS), choices=list(ABLATION_GROUPS))
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[0])
    p.add_argument('--eval-envs', type=int, default=1024)
    p.add_argument('--skip-teacher', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.eval_envs, a.eval_seeds, a.groups = 64, a.eval_seeds[:1], a.groups[:2]
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() not in ('YES', 'Y', '1', 'TRUE'):
        raise SystemExit('export OMNI_KIT_ACCEPT_EULA=YES after reading the Isaac Sim EULA')
    env = dict(os.environ, WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    prefix = 'smoke-' if a.smoke else ''
    root = OUT_ROOT / f'{prefix}{a.task}-{a.arm}' / f'teacher-s{a.teacher_seed}'
    s = read_json(root / 'summary.json')
    if s.get('status') != 'complete':
        raise SystemExit(f'{root} is not complete; finish pipeline.py first')
    spec = TASKS[a.task]['arms'][a.arm]
    args = SimpleNamespace(task=a.task, eval_envs=a.eval_envs)
    rate = lambda rec: rec['success_rate']

    teacher = None
    if not a.skip_teacher and not spec.get('teacher_from'):
        t_ckpt = s['teacher']['checkpoint']['path']
        teacher = {'base': [rate(run_eval(root / f'teacher-eval-s{e}', args, spec, t_ckpt, 'ppo_haic_train', e, env,
                                          trace=False)) for e in a.eval_seeds]}
        for g in a.groups:
            teacher[g] = [rate(run_eval(root / 'ablation' / f'teacher-{g}-s{e}', args, spec, t_ckpt, 'ppo_haic_train',
                                        e, env, mode='teacher_ablate', trace=False, extra=[f'+ablate={g}']))
                          for e in a.eval_seeds]

    students = {}
    for ss, st in sorted(s['students'].items()):
        ckpt = st['checkpoint']['path']
        sd_ = root / f'student-s{ss}'
        r = {'pred': [rate(run_eval(sd_ / f'eval-s{e}', args, spec, ckpt, 'ppo_haic_finetune', e, env))
                      for e in a.eval_seeds],
             'oracle': [rate(run_eval(sd_ / f'eval-oracle_priv-s{e}', args, spec, ckpt, 'ppo_haic_finetune', e, env,
                                      mode='oracle_priv', trace=False)) for e in a.eval_seeds]}
        for g in a.groups:
            r[g] = [rate(run_eval(sd_ / 'ablation' / f'oracle_priv-{g}-s{e}', args, spec, ckpt, 'ppo_haic_finetune', e,
                                  env, mode='oracle_priv_ablate', trace=False, extra=[f'+ablate={g}']))
                    for e in a.eval_seeds]
        students[f'student-s{ss}'] = r

    summary, rows = aggregate(a.groups, teacher, students)
    out = OUT_ROOT / 'ablation' / f'{prefix}{a.task}-{a.arm}'
    write_json(out / 'ablation.json', finite_or_none({
        'summary': summary, 'rows': rows, 'teacher': teacher, 'students': students,
        'eval_seeds': a.eval_seeds, 'eval_envs': a.eval_envs}))
    (out / 'ablation.md').write_text(report(a.task, a.arm, summary, rows, a.eval_seeds), encoding='utf-8')
    zpath = out.with_name(out.name + '.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            z.write(f, f.relative_to(out.parent))
        for f in sorted(root.rglob('ablation/*/summary.json')):
            z.write(f, Path('runs') / f.relative_to(OUT_ROOT))
    print(f'ABLATION_REPORT={out / "ablation.md"}', flush=True)
    print(f'ABLATION_ZIP={zpath}', flush=True)


if __name__ == '__main__':
    main()
