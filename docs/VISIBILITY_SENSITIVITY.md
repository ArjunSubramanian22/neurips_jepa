# Visibility sensitivity

The historical headline cut keeps the top third of nonzero raster-MSE counterfactuals, quantile 0.67. That rule is unchanged. Its numeric value is fit on `cvpr_v1` validation visibility only, from a one-step force reversal with no predictor. Episodes 1800–1999 are the legacy reporting table and are not used to choose the cut. Test episodes are not used. `selected_by_outcome` is false.

`visibility_sensitivity_v1` measured 180 validation episodes. Raster MSE was zero on 146 of them. The nonzero population is 34 episodes. The primary threshold is 3.798e-6. `saw_test` is false.

The sensitivity tables apply that frozen threshold, plus top 10, 20, 25, 33, 50, and 75 percent of nonzero visibility, plus every nonzero row. The same percentiles are repeated for raster MSE, RGB MSE, and affected-pixel fraction. Absolute floors are the predeclared grids `1e-6` through `1e-2` for the two MSE metrics and `0.01, 0.02, 0.05, 0.10, 0.20` for affected pixels. Percentile edges are taken from the fork, so every checkpoint is scored on the same episodes. Rows outside a cut stay in `n_excluded`.

State/action sensitivity is `not_available` on the horizon-audit table. That column is filled only when a row already carries `state_action_ratio`.

## Persistent reversal at horizon 10

For `vicreg_action_s42`, open-loop top-1 stays near one half from the top 10 percent (18 episodes, 0.500) through every nonzero row (180 episodes, 0.528). The primary cut selects 162 episodes and has open-loop top-1 0.525. Teacher-forced top-1 on the same cuts is 1.000, 0.839, and 0.846. The top-10 teacher-forced value is one hit on 18 episodes, not a separate headline.

Open-loop median cosine on those cuts stays between -0.043 and 0.049. Teacher-forced median cosine stays between 0.964 and 0.973. `vicreg_noaction_s42` has no defined open-loop cosine on any of these cuts. Its teacher-forced cosine median is 0.969 on all 180 nonzero rows, and its teacher-forced top-1 is 0.828. Reading the true frame produces that alignment without an action input.

Spearman correlation of raster MSE with cosine, on the 180 defined open-loop pairs, is 0.059. With retrieval it is -0.027. With the predictor/target ratio it is -0.541, while the median ratio itself stays near 0.03 to 0.05. The trapezoid average of the open-loop top-1 curve across the predeclared fractions is 0.511. The teacher-forced average is 0.915. Neither integral was used to pick a threshold.

The supplement tables are `sensitivity_slices.csv`, `threshold_free.csv`, and `visibility_bins.csv` under `experiments/runs/visibility_sensitivity_v1/`. The validation fork measurements are `validation_visibility.csv`.

```bash
python3.13 scripts/run_visibility_sensitivity.py
```
