# Directional counterfactuals

`delta_target` is the target-encoder difference between the recorded next frame and the next frame after a raw force reversal. `delta_pred` is the predictor difference for those two actions from the same encoded current frame. Actions are z-scored with training episodes `[0, 1800)` only.

Cosine, angular error, and signed projection are null when either vector has norm at or below `1e-8`. They are not filled with zero. Normalized vector error and the magnitude ratio stay numeric because their denominators add `1e-12`. A mean of those two on the full table is not stable: near-zero targets hit that floor and inflate the ratio. The medians below use the nonzero-observable rows, where the target norm clears the threshold.

Status priority is nonfinite, then `target_near_zero`, then `target_invisible`, then `valid_directional_contrast`. A row can carry both the near-zero flag and the invisible flag. The exclusive status records the priority winner, and the flags keep the overlap.

## Run

`counterfactual_direction_v1` evaluated the ten no-crop `last.pt` checkpoints on legacy validation episodes 1800–1999. `is_test_set` is false. There are 15 action steps per episode and 3000 rows per checkpoint, 30000 rows in total. The table is `experiments/runs/counterfactual_direction_v1/transitions.csv` and `transitions.parquet`. Slice summaries and episode bootstrap intervals are in `summary.json`. Figures are in `figures/counterfactual_direction/`.

On every checkpoint, 1561 of 3000 transitions have an identical physical raster under the force reversal. Those same 1561 rows are `target_near_zero` and also have the invisible flag. None are invisible with a target norm above `1e-8`. The other 1439 rows are `valid_directional_contrast`. High visibility means an affected-pixel fraction of at least 0.05, fixed before the run: 749 rows. Action-change boundaries are steps with `t % 3 == 0`: 1000 rows, of which 474 have a valid cosine.

## Nonzero-observable cosine

Valid cosine counts are 1439 for each action-conditioned checkpoint and 0 for each no-action checkpoint. The no-action predictor delta is at or below `1e-8` on every row, so cosine is undefined there. Magnitude ratio and signed projection on those rows are 0, and the normalized vector error on the 1439 nonzero targets is 1 within `1e-8`.

| Checkpoint | Median cosine | Episode-mean cosine | 95% CI | Episodes in the CI |
| --- | --- | --- | --- | --- |
| vicreg_action_s42 | 0.0643 | 0.0687 | 0.0494 to 0.0877 | 198 |
| vicreg_action_s43 | 0.0826 | 0.0779 | 0.0601 to 0.0945 | 198 |
| vicreg_action_s44 | 0.1058 | 0.0915 | 0.0760 to 0.1083 | 198 |
| vicreg_action_s45 | 0.0371 | 0.0384 | 0.0241 to 0.0527 | 198 |
| mse_action_s42 | 0.1723 | 0.1384 | 0.1226 to 0.1542 | 198 |
| mse_action_s43 | 0.1642 | 0.1537 | 0.1355 to 0.1708 | 198 |
| four vicreg_noaction seeds | undefined | undefined | not computed | 0 of 200 |

Episodes 1957 and 1963 have no valid cosine for the action-conditioned checkpoints. They are listed in `excluded_episode_ids` and left out of the resample.

On the same 1439 rows, median magnitude ratios are between 0.093 and 0.219. Median normalized vector error is about 1. Median signed projection is between 0.0003 and 0.0025. High-visibility medians are in `summary.json` under the `high_visibility` slice and are slightly higher for cosine than the full nonzero set. Action-distance quintile edges, from the raw action L2 on this validation sample, are 0.530, 0.812, 1.053, and 1.410.
