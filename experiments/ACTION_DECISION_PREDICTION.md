# Intervention probes and physical action-selection regret

16 October 2026. Branch `research/intervention-action-prediction` at base `b338136`. Decision: **no-go**.

The open-loop force-reversal decomposition does not predict held-out physical regret beyond validation prediction loss, directional cosine, normalized effect error, and effect RMSE. Adding log-ratio distance, orthogonal energy, and parallel gain lowers holdout R² by 0.032. The episode-cluster 95% interval is −0.079 to +0.037.

## Question and design

Unit: one checkpoint on one validation episode. Checkpoints are the final VICReg and MSE action-conditioned weights, seeds 42–46. RSSM and the no-action control are excluded: RSSM's matched planning target is a different state from its training target, and the no-action cosine is undefined.

Outcome: physical endpoint regret of matched final-image ranking among 16 time-varying sequences at horizon 10. The generating sequence is in the library, so the oracle endpoint error is 0 and regret equals the chosen candidate's particle RMS. This is finite-candidate ranking. It is not closed-loop control.

What the planner receives: the current frame, the candidate action sequence, and the final goal image. What the probe adds, and the planner does not see: the target-encoder displacement between the recorded future and the force-reversed future. That probe needs a restorable simulator state. Validation prediction loss is the final-epoch training term already in `history.json` (`val_invariance` for VICReg, `val_loss` for MSE). The library's pairwise median endpoint distance is a difficulty control, not a model diagnostic.

Fit episodes are 1440–1451. Holdout episodes are 1452–1463. Coefficients, scaling, and the 0.05 incremental-R² bar were fixed before the fit (`protocol.json`). Episodes 1620 and above are refused. The reader counted and dropped 64,800 test-regime rows in the multi-horizon file. The arm test was not opened.

The baseline is one linear model: validation prediction loss, cosine, normalized vector error, effect RMSE. The diagnostic model adds |log response ratio|, orthogonal energy, and parallel gain, all medians over the six open-loop force-reversal branch times. A go required holdout incremental R² of at least 0.05, a bootstrap interval above zero, full-model R² of at least 0.05, and the same bar inside each model family, after the difficulty control, on regret divided by pairwise median, on the reversed episode split, and after dropping any one seed from the holdout.

## Result

240 primary rows, 120 fit and 120 holdout. Mean holdout regret is 0.0103. Runtime was 2.6 seconds on CPU. No training job was started.

| Check | Baseline R² | Full R² | Incremental R² | 95% interval |
| --- | ---: | ---: | ---: | --- |
| Primary, sequence H10 | 0.272 | 0.240 | −0.032 | −0.079 to 0.037 |
| VICReg only | −0.080 | −0.164 | −0.084 | −0.239 to 0.022 |
| MSE only | −0.536 | −0.497 | 0.039 | −0.149 to 0.282 |
| Plus pairwise median | 0.297 | 0.267 | −0.030 | −0.075 to 0.043 |
| Regret / pairwise median | 0.270 | 0.240 | −0.030 | −0.070 to 0.033 |
| Reversed episode split | 0.058 | 0.079 | 0.021 | −0.035 to 0.064 |

Leaving any one seed out of the holdout keeps the incremental R² negative. The worst fold is −0.039. The diagnostic model increases absolute error on 78 of 120 holdout rows.

Holdout mean regret is 0.0194 for VICReg and 0.0012 for MSE. A model that only knows the family, using the fit-block family mean, has holdout R² 0.271. The four-feature baseline has holdout R² 0.272. On the fit block, the baselines already span orthogonal energy at R² 0.974 and parallel gain at R² 0.972. Log-ratio distance is only partly spanned (R² 0.494). That residual still does not improve the holdout.

The largest misses are VICReg selections whose regret is 0.04–0.08 while both linear models predict about 0.01–0.02. Episode 1461 is a miss for every VICReg seed. The probe median does not identify those scenes.

Horizon 5 and the eight constant actions were scored and were not decision criteria. Horizon 5 incremental R² is +0.034 with both R² values negative. The constant library incremental R² is −0.015.

## Checkpoint summary, and why it is not a result

Probe medians on episodes 1440–1451 versus mean regret on 1452–1463, ten checkpoints: parallel-gain Spearman −0.97, validation prediction loss and effect RMSE 0.85, cosine −0.80. The two families do not overlap. VICReg parallel gain is 0.28–0.33 and MSE parallel gain is 0.72–0.78. The same separation is present in validation loss, cosine, and effect RMSE.

Inside one family there are five checkpoints. VICReg parallel-gain Spearman is −0.90 and effect-RMSE Spearman is 0.80. MSE parallel-gain Spearman is −0.82, while cosine and effect RMSE are about 0.15 and −0.05. Log-ratio distance is −0.70 inside VICReg and +0.72 inside MSE, so the ratio feature changes sign. The pre-specified within-family regressions have negative holdout R². Five-point rank correlations are not a second analysis.

## Literature, checked against the papers

Terver, Yang, Ponce, Bardes, and LeCun (arXiv:2512.24497, v2) write that faithful unrolling of many actions does not by itself produce planning success, and that the validation prediction task is not fully aligned with goal-conditioned planning. In appendix E.3 the mean negative Spearman with smoothed success is highest for visual-embedding L1 loss: 0.47 on Metaworld, 0.86 on Push-T, 0.81 on Wall, and 0.57 on Point Maze. Their unit is an epoch of a training run. This pilot's saved rows do not contain that unrolled visual-embedding loss. The included validation prediction loss is the final-epoch training embedding term, one number per checkpoint.

Do-JEPA (Resani and Shi, arXiv:2609.37378) reports effect RMSE and effect cosine as separate metrics. Full fine-tuning lowered effect RMSE from 0.2145 to 0.1522 and planning success from the 94% base model to 54% (the matched Aug control was 58%). With a centered adapter, effect RMSE fell about 5–6% and planning stayed within 3 percentage points of the base planner. Effect error and planning success already come apart in that paper. Parallel gain and orthogonal energy are rearrangements of cosine and the response ratio; they are not a new effect metric.

## Decision

No-go. There is no confirmation study.

The strongest alternative explanation is model identity. VICReg is worse at the force-reversal contrast and worse at sequence ranking. Family membership accounts for the pooled R². Inside a family, the linear baseline does not generalize, and the decomposition does not repair it.

A later study would need new checkpoints in which parallel gain varies inside one objective, episode-level visual-embedding loss in the baseline, and a frozen within-family test on episodes outside 1440–1463 and outside the opened test. That study is not justified by this pilot. The machine was already near full load, and this run used no new training.

## Reproduce

```bash
cd /Users/Arjun/cursor/neurips_jepa_action_prediction
python3.13 -m unittest tests.test_action_decision_prediction
python3.13 scripts/run_action_decision_prediction.py
```

Outputs: `experiments/runs/action_decision_prediction_pilot/`. Figures: `figures/incremental_r2.png` and `figures/holdout_predictions.png`. The original checkout was not modified.
