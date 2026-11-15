# MuJoCo sim-to-sim protocol (skateboard), fixed 2026-10-31 before the OH-Both evaluation

Scene: HOIC-baseline `g1_29dof_rev_1_0_skateboard.xml` with `isaac_assets=1` (foot boxes, wheels, trucks,
deck and wheel inertia of the Isaac training assets, research/dump_usd.py). Wheel frictionloss 0.01 (scene value).

Wheel damping: 0.004 N m s/rad, identified on the HAIC (dyn) students only, 30 episodes each, as the grid value
(0.003, 0.004, 0.005, 0.006, 0.008) whose median signed along-track root error at 9 s is closest to zero
(st0 -0.04, st1 +0.16, st2 +0.02 m). The OH-Both students did not inform any choice.

Known residual gap: from the identical initial state of an Isaac rollout (golden_init=1), MuJoCo tracks Isaac within
a few cm until the first push (4.3 s); the push then accelerates the board to about 2/3 of Isaac's speed, leaving a
constant ~0.2 s along-track lag shared by both arms (contact impulse of the push, PhysX vs MuJoCo).

Evaluation: dyn and dyn-hboth, students st0-st2, 100 episodes each, per-episode randomization of sim2sim.py.

Metrics (declared before seeing the dyn-hboth results):
1. Isaac success and drift-tolerant success (as in the Isaac evaluation).
2. Cross-track success: no local termination, no fall, and no run of 25 consecutive steps with
   |lateral root error| >= 0.3 m (lateral = perpendicular to the reference heading). The along-track
   component is excluded because it is the shared simulator lag above, unobservable to any onboard sensor.
3. Median |lateral root error| and |yaw error| at 9 s (end of the glide), per arm.
Comparison: per-student differences and the episode-paired (same seed per episode index) sign test over the
pooled 300 episodes, as for the first uncalibrated run.

## Result (2026-10-31, mujoco-wd0.004-fl0.01-isaac_assets1)

| arm / student | success | drift-tolerant | cross-track | fell | median |lat| at 9 s | median |yaw| at 9 s |
|---|---|---|---|---|---|---|
| HAIC st0 / st1 / st2 | 0.43 / 0.07 / 0.73 | 0.90 / 1.00 / 0.98 | 0.60 / 0.46 / 0.85 | 0.10 / 0.00 / 0.02 | 0.14 / 0.22 / 0.06 m | 7.5 / 4.7 / 3.1 deg |
| OH-Both st0 / st1 / st2 | 0.25 / 0.08 / 0.01 | 0.90 / 0.27 / 0.03 | 0.26 / 0.08 / 0.01 | 0.10 / 0.71 / 0.95 | 0.27 / 0.29 / 1.30 m | 2.6 / 4.7 / 29.8 deg |

Pooled (300 paired episodes): success 41.0% vs 11.3% (-29.7 pp, sign test p < 1e-15); cross-track 63.7% vs 11.7%.
OH-Both falls (st1, st2) occur at 6-8 s, in the high-speed push phase. The same students in the unmodified HOIC
scene (wheel damping 0.003, before isaac_assets) fell 1% and 31%: the outcome is highly sensitive to the scene.

## Addendum: contact-and-rolling randomized fine-tuning (declared 2026-11-12, before training)

Hypothesis: the OH-Both falls in MuJoCo come from a push response that the heading fine-tuning never saw
(the Isaac board rolls almost freely: wheel damping <= 1e-5 on one wheel only).
Arms (skateboard, teacher reused from dyn, 3 students each): dyn-cr (HAIC) and dyn-hboth-cr (OH-Both), fine-tuned
with cfg/task/G1/haic/skateboard_dyn_cr.yaml / skateboard_dyn_heading_cr.yaml: wheel damping 0-0.008 N m s/rad
and Coulomb friction 0-0.02 N m on all four wheels, board mass 1.0-4.0 kg; everything else unchanged.
The ranges cover the rolling resistance identified on the HAIC students only (0.004 / 0.01 above), which is
disclosed in the paper.
MuJoCo evaluation: exactly the scene, settings, episodes and metrics of the main run above
(isaac_assets=1, wheel damping 0.004, frictionloss 0.01, 100 episodes x 3 students, seed 0).
Reported regardless of the outcome:
1. fall rate, Isaac success, drift-tolerant and cross-track success of dyn-cr and dyn-hboth-cr, per student;
2. pooled paired comparison dyn-hboth-cr vs dyn-cr (sign test, as above);
3. the same students in Isaac (standard evaluation of research/pipeline.py, in the randomized task).

### Result of the addendum (2026-11-15)
Pooled over 3 students x 100 episodes: dyn-cr success 40.0%, cross-track 64.0%, fell 4.0% (dyn: 41.0 / 63.7 / 4.0);
dyn-hboth-cr 22.0 / 28.3 / 46.0 (dyn-hboth: 11.3 / 11.7 / 58.7). The randomization reduces the OH-Both falls
(sign test vs dyn-hboth: success p = 1.0e-4, cross-track p = 2.9e-9) without changing HAIC, but OH-Both-CR still falls
far more than HAIC-CR (success 22.0 vs 40.0%, p = 2.0e-6). Rolling resistance and board mass explain only part of it.

## Diagnostic 2: wheel-ground sliding friction (declared 2026-11-15, before running)

Finding: Isaac's ground material combines friction by multiplication (active_adaptation/envs/terrain/regular.py,
ground 1.0), so the wheels slide with their own dynamic friction, 0.01-0.1 (static 30-60x); the HOIC scene has its
wheel contact pairs commented out and uses the default sliding friction 1.0, i.e. lateral skidding of the board costs
about 20x more force in MuJoCo. Hypothesis: the heading students correct heading partly by skidding the board, which
MuJoCo resists; HAIC, which does not correct heading, is unaffected.
Test (no retraining): the protocol scene with the wheel-ground sliding friction set to Isaac's dynamic range,
--solver isaac_assets=1,wheel_mu=0.05 (midpoint) and, as a bound, wheel_mu=0.3 (lowest Isaac static friction);
arms dyn, dyn-hboth, dyn-cr, dyn-hboth-cr; same episodes, seeds and metrics; reported regardless of the outcome.
The hypothesis is supported if the OH-Both falls drop to the HAIC level at wheel_mu=0.05 while HAIC stays similar.

### Result of diagnostic 2 (2026-11-15)
Pooled over 3 students x 100 episodes (success / cross-track / fell, |lat| and |yaw| at 9 s):
wheel_mu=0.05: dyn 0.0 / 0.0 / 22.3% (1.19 m, 18.3 deg), dyn-hboth 3.0 / 4.0 / 33.0% (0.51 m, 27.3 deg),
               dyn-cr 0.0 / 0.0 / 37.0%, dyn-hboth-cr 7.3 / 1.3 / 45.0%: every arm slides off the reference.
wheel_mu=0.3:  dyn 5.0 / 3.7 / 0.3% (0.85 m, 15.8 deg), dyn-hboth 17.3 / 12.3 / 43.0% (0.30 m, 4.6 deg),
               dyn-cr 1.0 / 2.7 / 1.7% (0.62 m, 17.5 deg), dyn-hboth-cr 42.0 / 44.3 / 24.0% (0.15 m, 5.8 deg);
               dyn-hboth-cr vs dyn-cr: success +41.0 pp (124/1 discordant, p = 5.9e-36).
The declared criterion is not met: at 0.05 HAIC collapses as well (MuJoCo has no static friction, so 0.05 lets the
board slide freely, unlike Isaac's 0.3-6 static friction). Wheel friction strongly changes which arm wins: with less
lateral grip HAIC drifts far off the reference while the heading students hold it (|lat| 0.15 vs 0.62 m at 0.3),
but in every scene the heading students fall more often than HAIC, so the fall fragility is not explained by
wheel friction alone.

### Why the heading students fall (research/sim2sim_falls.py, 2026-11-15)
Falls happen in the push phase (onset median 6.5 s for dyn-hboth, 7.7 s for dyn-hboth-cr). In MuJoCo every push
delivers less thrust (ground foot sweeps back relative to the board at 0.13-0.19 m/s vs 0.36-0.44 m/s in Isaac).
HAIC adapts (both feet on the board 11% of 4.5-7.5 s vs 5% in Isaac, ground foot closer, board reaches Isaac speed);
OH-Both keeps its wide one-leg stance (both feet on board 3%, ground foot 0.33 m beside the board). 0.5 s before a
fall its board moves at 0.66 m/s (others 0.95), its pelvis is 0.16 m off the board centre line and its heading has
turned by 41 deg (others 11 deg). Contact-and-rolling randomization restores part of the board speed and of the falls.
