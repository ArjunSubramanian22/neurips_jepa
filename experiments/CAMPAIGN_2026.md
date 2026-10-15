# Campaign through 4 November 2026

Registered on 15 October 2026, before any production training on the new arm split. Pilots may update the queue. They may not move the test boundary or the metrics below.

## Hardware

Measured on this machine: Apple M1, 8 CPU cores, 8 GB RAM, MPS available, about 15 GB free of 228 GB. Checkpoints already on disk are about 12 MB for a high-excitation JEPA and 2 MB for an arm JEPA. The queue is sized for that machine. A job that would need a second large training run in parallel with simulation does not get scheduled.

## Hypotheses

These are the claims the production runs are allowed to address. A result that goes the other way is reported.

1. On the existing high-excitation MPM checkpoints, open-loop VICReg directional cosine falls from one step to ten while the response ratio stays near one. MSE and the RSSM stochastic mean do not show that split. This is a statement about these checkpoints and this simulator, not a ranking of objectives in general.
2. Giving RSSM the same final goal image that JEPA sees, with no goal trajectory and no generating actions, changes its candidate ranking relative to the old privileged-target score. Physical endpoint error is the decision metric. Latent top-1 is only candidate identity.
3. The RSSM stochastic target cosine depends on the images, the actions, or both. Blank images, the other fork's images, zero actions, and swapped actions are the controls. The intact mode is the comparison, not a new headline.
4. The arm campaign either reproduces a direction-versus-magnitude split under varied initial states or shows that the MPM result does not travel. The 32-episode arm file remains a pilot and is not this test.

The VICReg objective in this repository is invariance, variance, and covariance between the predictor output and a detached EMA target. It is not the standard two-view VICReg objective, and it is not a claim about every JEPA system.

## Metrics and rules

- Directional cosine stays undefined when either contrast has norm at or below the existing threshold. Undefined is not zero.
- Response ratio is reported beside cosine. A large ratio with a falling cosine is the MPM pattern under test, not a win.
- Planning reports latent top-1 and physical endpoint RMS separately. Random top-1 is one over the number of candidates.
- Candidate ranking and closed-loop execution are different experiments. Closed-loop is not started until the matched finite-candidate pipeline has a measured runtime.
- Arm normalization uses training episodes only. Split `all` drops the test suffix. The test is not used for training, early stopping, or threshold choices.
- Arm checkpoints are reported as a pair: the validation-MSE checkpoint and the final epoch. The one-seed pilot showed these can disagree, with directional cosine higher at the final epoch and validation MSE lower earlier. The production grid does not drop either checkpoint after seeing that cosine.
- The arm test stays unread by model selection until a validation freeze is written. That freeze is not written in this session.
- Original high-excitation branch rows stay as the v1 record. Corrected physical columns are the v2 siblings described in `experiments/PHYSICAL_INDEX_CORRECTION.md`.
- No architecture outside the four-model core is added unless a pilot shows it answers one of the hypotheses above. The four are action-conditioned VICReg, action-conditioned MSE, no-action VICReg, and RSSM.

## Queue

Status is updated as pilots finish. Production arm training is five seeds of the four core models, and only after a one-epoch learnability check on the training split.

| Order | Job | Role | Status |
| --- | --- | --- | --- |
| 1 | Evaluator index, test isolation, v2 physical columns | Correctness | done |
| 2 | Arm probe, 6/6/6 episodes, length 8 | Stability and timing | passed in 0.59 s, 0.033 s/episode |
| 3 | Arm full file, 480 train / 120 validation / 120 test, length 16 | Second environment | written to `outputs/mujoco_arm_campaign` in 19.2 s. Push median displacement 0.184, hold 0. Replay of the first train, validation, and test episodes matched. |
| 4 | Matched-goal planning pilot, episodes 1440-1441, seed 42 | Timing | 15 s. Scaled job is affordable. |
| 5 | RSSM target ablations, episodes 1440-1619, branches 0 and 3, five seeds | Hypothesis 3 | done. The four-episode claim that H10 cross-fork cosine is negative for every seed does not survive. Seed 45 is +0.156 and the pooled median is -0.041. Blank-image and zero-action patterns do survive. Report: `experiments/runs/rssm_target_controls_val/report.md`. |
| 6 | One-epoch MSE on the arm training split | Learnability | passed. 16 s/epoch on MPS. Train/val was 480/120; the 120 test episodes were not in the loader. Val MSE 0.0076. A no-action MSE reached val MSE 0.0050 with a zero action gradient, so one-step loss does not show action use. On six validation forks the action model had median cosine 0.29 at H1 and 0.31 at H10, with response ratios 0.25 and 0.20. The no-action predicted delta was 0 and its cosine was undefined. |
| 7 | Matched planning, episodes 1440-1463, five seeds, four models, constant and sequence libraries | Hypothesis 2 | done in 294 s. Report: `experiments/runs/mpm_high_planning_matched/report.md`. RSSM constant-action top-1 falls from 35.8%/42.5% with privileged goals to 15.0%/20.0% with a final image. JEPA is unchanged because it already saw only that image. |
| 8 | Five-seed arm training | Hypothesis 4 | not before 19 October. A 20-epoch MSE seed takes 171 s and validation MSE rises after epoch 2, so the grid is affordable and must keep both the validation-MSE checkpoint and the final checkpoint. |
| 9 | Closed-loop | Separate from ranking | not launched. `scripts/run_planning_benchmark.py` drives a different checkpoint list and CEM goal protocol. It is not a matched-image evaluation of the high-excitation models. |

## Development results so far

RSSM posterior targets were repeated on all 180 validation episodes. See `experiments/runs/rssm_target_controls_val/report.md`. The four-episode file remains a pilot record and is not the claim.

- Intact stochastic cosine stays high at H1 and H10 for every seed.
- Blanking the target images leaves H1 high and lowers H10 for every seed.
- Zeroing the target action drops H1 cosine for every seed, into roughly 0.15-0.35.
- Replacing the target image with the other fork's image does not have one sign at H10. Four seeds are slightly negative and seed 45 is positive.

`swapped_action` is not a separate result. It negates the `cross_fork_image` target delta, so its cosine is the negation. The summary file records that dependence.

The matched-planning pilot on two episodes is too small to rank models. It did show that latent top-1 and physical regret can be computed under a shared final-image goal, including for 16 time-varying sequences. The 24-episode run is the one that can be compared with the old eight-action table, and only as a different goal-information condition.

Augmentation stays off for the arm campaign. An augmentation-on control is a later, separate run, and only if the unaugmented baseline learns.
