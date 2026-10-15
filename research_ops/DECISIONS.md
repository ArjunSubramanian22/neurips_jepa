# Decisions

15 October 2026.

1. The v1 branch CSVs, the protocol freeze, `PRETEST_FINDINGS.md`, and `TEST_EVALUATED.lock` stay as historical records. Corrected physical columns live in `branch_rows_physindex_v2.csv`.
2. New scorer rows use metric version `mpm_multih_v2_physindex`. The frozen test is not rerun to produce them.
3. Matched planning gives every model the current frame, the candidate actions, and the final goal image. RSSM does not receive the goal trajectory or the generating actions.
4. `swapped_action` negates the `cross_fork_image` target delta. It is one contrast, reported once.
5. The H10 cross-fork sign is not a result. The full validation slice disagrees across seeds.
6. Arm episodes 600-719 are a sealed test. Training split `all` drops them. No model selection uses them until a validation freeze exists. That freeze is not written yet.
7. Arm production reports both the validation-MSE checkpoint and the final epoch. The one-seed disagreement was observed before the five-seed grid and is not used to drop the worse-looking cosine.
8. The five-seed arm grid waits until 19 October. One-epoch and one 20-epoch MSE seed were pilots, not that grid.
9. VICReg statements refer to this repository's predictor-versus-detached-EMA objective.
10. Closed-loop is a separate experiment from finite-candidate ranking and has not been run on these checkpoints.
