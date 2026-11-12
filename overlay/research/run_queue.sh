#!/usr/bin/env bash
# Sequential job queue for one GPU. Run inside tmux so it survives SSH disconnects:
#   tmux new -s haic
#   bash research/run_queue.sh round1      # dyn + geo: teacher and student seed 0, then diagnostics
#   bash research/run_queue.sh round2      # student seeds 1 2 for both arms, then diagnostics
#   bash research/run_queue.sh round3      # heading arms, student seed 0 (reuse the dyn teacher)
#   bash research/run_queue.sh round4      # heading arms, student seeds 1 2
#   bash research/run_queue.sh ablate      # training-free privileged-input ablation (dyn, geo)
#   TASK=push_cart bash research/run_queue.sh cross    # cross-task replication (also box_stairslope)
#   bash research/run_queue.sh cw          # contribution 3a (control-weighted latent alignment)
#   bash research/run_queue.sh robust      # heading-sensor noise sweep (evaluation only)
#   bash research/run_queue.sh cr          # contact-randomized HAIC / OH-Both fine-tuning + export (MuJoCo transfer)
# Every step is resumable: re-running the same round skips finished stages.
source "$(dirname "$0")/setup/haic_env.sh"  # before set -u: conda activation uses unset variables
set -uo pipefail
TASK="${TASK:-skateboard}"
LOG="$HAIC_OUT/queue-$(date +%Y%m%d-%H%M%S).log"
echo "queue log: $LOG"

step() {  # step <description> <command...>; stop the queue on failure
  echo "[$(date '+%F %T')] >>> $1" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] <<< exit $rc" | tee -a "$LOG"
  [ "$rc" -eq 0 ] || { echo "queue stopped" | tee -a "$LOG"; exit "$rc"; }
}

case "${1:-}" in
  smoke)
    step "smoke dyn" python research/pipeline.py --task "$TASK" --arm dyn --smoke
    step "smoke geo" python research/pipeline.py --task "$TASK" --arm geo --smoke
    step "smoke diagnostics" python research/diagnose.py --task "$TASK" --arms geo dyn --smoke
    ;;
  round1)
    step "dyn teacher + student s0" python research/pipeline.py --task "$TASK" --arm dyn --student-seeds 0
    step "geo teacher + student s0" python research/pipeline.py --task "$TASK" --arm geo --student-seeds 0
    step "diagnostics (1 student per arm)" python research/diagnose.py --task "$TASK" --arms geo dyn
    ;;
  round2)
    step "dyn students s1 s2" python research/pipeline.py --task "$TASK" --arm dyn --student-seeds 1 2
    step "geo students s1 s2" python research/pipeline.py --task "$TASK" --arm geo --student-seeds 1 2
    step "diagnostics (3 students per arm)" python research/diagnose.py --task "$TASK" --arms geo dyn
    ;;
  smoke-heading)
    step "smoke heading arm (needs smoke dyn)" python research/pipeline.py --task "$TASK" --arm dyn-hwm --smoke
    step "smoke heading null arm" python research/pipeline.py --task "$TASK" --arm dyn-hnull --smoke
    step "smoke diagnostics (heading)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hwm dyn-hnull --smoke
    ;;
  round3)
    step "drift-tolerant eval of the baselines" python research/diagnose.py --task "$TASK" --arms geo dyn
    for arm in dyn-hwm dyn-hnull dyn-hact dyn-hboth; do
      step "$arm student s0" python research/pipeline.py --task "$TASK" --arm "$arm" --student-seeds 0
    done
    step "diagnostics (heading arms, seed 0)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hwm dyn-hnull dyn-hact dyn-hboth
    ;;
  round4)
    for arm in dyn-hwm dyn-hnull dyn-hact dyn-hboth; do
      step "$arm students s1 s2" python research/pipeline.py --task "$TASK" --arm "$arm" --student-seeds 1 2
    done
    step "diagnostics (heading arms, 3 seeds)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hwm dyn-hnull dyn-hact dyn-hboth
    ;;
  smoke-ablate)
    step "smoke privileged-input ablation" python research/ablate.py --task "$TASK" --arm dyn --smoke
    ;;
  ablate)
    step "privileged-input ablation, dyn" python research/ablate.py --task "$TASK" --arm dyn
    step "privileged-input ablation, geo" python research/ablate.py --task "$TASK" --arm geo
    ;;
  smoke-cross)
    # whole cross-task chain at tiny budgets: TASK=push_cart or TASK=box_stairslope
    step "smoke dyn ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn --smoke
    step "smoke dyn-hboth ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hboth --smoke
    step "smoke dyn-hnull ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hnull --smoke
    step "smoke diagnostics ($TASK)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hboth dyn-hnull --smoke
    step "smoke ablation ($TASK)" python research/ablate.py --task "$TASK" --arm dyn --smoke
    step "smoke robustness ($TASK)" python research/robustness.py --task "$TASK" --arms dyn-hboth --smoke
    ;;
  cross)
    # cross-task replication: own dyn teacher, then dyn / dyn-hboth / dyn-hnull x 3 student seeds
    step "dyn teacher + students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn --student-seeds 0 1 2
    step "dyn-hboth students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hboth --student-seeds 0 1 2
    step "dyn-hnull students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hnull --student-seeds 0 1 2
    step "diagnostics ($TASK)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hboth dyn-hnull
    step "privileged-input ablation ($TASK)" python research/ablate.py --task "$TASK" --arm dyn
    step "heading robustness ($TASK)" python research/robustness.py --task "$TASK" --arms dyn-hboth
    ;;
  cross-extra)
    # optional, after cross: heading-to-actor arm and the released (geo) baseline with its own teacher
    step "dyn-hact students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hact --student-seeds 0 1 2
    step "geo teacher + students ($TASK)" python research/pipeline.py --task "$TASK" --arm geo --student-seeds 0 1 2
    step "diagnostics, all arms ($TASK)" python research/diagnose.py --task "$TASK" --arms dyn dyn-hboth dyn-hnull dyn-hact geo
    ;;
  smoke-cw)
    step "smoke control-weighted arm (needs smoke dyn)" python research/pipeline.py --task "$TASK" --arm dyn-hboth-cw --smoke
    ;;
  cw)
    # contribution 3a on the skateboard: control-weighted latent alignment on top of the heading arms
    step "dyn-hboth-cw students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hboth-cw --student-seeds 0 1 2
    step "dyn-hwm-cw students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hwm-cw --student-seeds 0 1 2
    step "diagnostics, cw ($TASK)" python research/diagnose.py --task "$TASK" --arms dyn-hboth dyn-hboth-cw dyn-hwm dyn-hwm-cw dyn
    ;;
  smoke-cr)
    step "smoke contact-randomized arms (needs smoke dyn)" python research/pipeline.py --task "$TASK" --arm dyn-hboth-cr --smoke
    ;;
  cr)
    # MuJoCo transfer: HAIC and OH-Both fine-tuned under wider board physics (skateboard only), then exported
    step "dyn-hboth-cr students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-hboth-cr --student-seeds 0 1 2
    step "dyn-cr students ($TASK)" python research/pipeline.py --task "$TASK" --arm dyn-cr --student-seeds 0 1 2
    step "export for MuJoCo ($TASK)" python research/export_students.py --tasks "$TASK" --arms dyn-cr dyn-hboth-cr --student-seeds 0 1 2 --golden-steps 700
    ;;
  robust)
    step "heading robustness ($TASK)" python research/robustness.py --task "$TASK" --arms dyn-hboth dyn-hact dyn-hwm
    ;;
  *)
    echo "usage: bash research/run_queue.sh <target>   (set TASK=skateboard|push_cart|box_stairslope, default skateboard)"
    echo "  smoke round1 round2 smoke-heading round3 round4 smoke-ablate ablate"
    echo "  smoke-cross cross cross-extra smoke-cw cw robust smoke-cr cr"
    exit 2
    ;;
esac
echo "[$(date '+%F %T')] queue '$1' finished" | tee -a "$LOG"
