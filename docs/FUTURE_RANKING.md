# Counterfactual future ranking

`future_ranking_v1` failed while scoring the first checkpoint: a status constant was not imported. No summary was written. The partial `queries.csv` in that run is incomplete and is not a result. `future_ranking_v2` is the same predeclared protocol after that import fix.

The question is whether, from one starting state, the predicted future under action `a` is closer to the true future under `a` than to the true futures under the other candidate actions.

Each candidate is held constant for the horizon. This is not the logged sequence of actions. Horizons are 1, 5, and 10 observation frames. Start times are `t = 0` and `t = 5` on legacy validation episodes 1800–1999. `is_test_set` is false.

The four candidates are the recorded action, a deranged other validation episode's recorded action at the same time, a force reversal, and a half-scale force. Actions are z-scored with training episodes `[0, 1800)` only. The same moments are used for `mpm_medium_excitation`, which these checkpoints were not trained on.

Distances are latent MSE, Euclidean L2, variance-normalized MSE, and cosine distance. L2 is a positive scaling of MSE for a fixed dimension, so it induces the same ranking. Cosine distance is `1 - cosine` and is null when either vector has norm at or below `1e-8`. The variance scalar is the mean per-dimension variance of the recorded-action target embeddings on this validation sample. It is a reporting scale, not a selection statistic. A non-positive variance leaves the normalized distance undefined.

A query is `observationally_ambiguous` when its raster equals another candidate's raster. That includes futures whose particles moved and whose pixels did not. Those queries stay in the table, with `physically_distinct_observationally_identical` set when the particle RMS is positive, and they are not in the top-1 denominator. A distance tie between different rasters is a non-hit, not an ambiguity. Top-k is reported for `k = 2` because there are four candidates. The margin is the nearest other distance minus the correct distance. Pairwise probability is

```
P(distance(pred(a), target(a)) < distance(pred(a), target(a')))
```

over pairs whose rasters differ. Equal distances count as pairwise ties, not successes.

Controls, from the same futures: shuffled prediction rows, the target embeddings themselves, and a constant prediction repeated across actions. Chance top-1 for four candidates is 1/4. Episodes with no scored query are listed in the bootstrap and are not imputed.

Visibility bins use an affected-pixel threshold of 0.05, fixed before the run. Ambiguous queries are not placed in the low-visibility bin. Physical bins are zero, at most 0.001 RMS, and above that.

## Measured result

`future_ranking_v2` scored all ten no-crop `last.pt` checkpoints. Each environment has 1600 queries per horizon (200 episodes, two start times, four actions). `l2_mse_top1_hits_agree` is true. Variance-normalized MSE and cosine distance give the same top-1 as MSE on the medium-excitation horizon-1 slices that were checked in `summary.json`. These numbers are legacy validation, not a test set. The checkpoints were trained on `mpm_legacy` only.

On `mpm_legacy`, every ambiguous query is also physically distinct. Horizon 1 leaves 1170 of 1600 queries ambiguous and 430 scored. Horizon 5 leaves 691 ambiguous and 909 scored. Horizon 10 leaves 473 ambiguous and 1127 scored. Forty-five episodes have no scored query at horizon 1 and are excluded from that bootstrap.

On `mpm_medium_excitation`, horizon 1 has 2 ambiguous queries and 1598 scored. Horizons 5 and 10 have 1600 scored and no ambiguous queries.

The oracle, which copies the target embeddings, has top-1 1 and pairwise probability 1 on every scored slice. Shuffled predictions and the constant prediction stay with the trained model. The largest model-minus-shuffled gap in pooled MSE top-1 is 0.008 (`mse_action_s43`, legacy, horizon 5). No-action checkpoints match their shuffled and constant controls exactly, as they must when the prediction does not depend on the action.

Pooled MSE top-1 for the model, with the shuffled control in parentheses:

| Checkpoint | Legacy h=1 | Medium h=1 | Legacy h=10 | Medium h=10 |
| --- | --- | --- | --- | --- |
| vicreg_action_s42 | 0.319 (0.316) | 0.250 (0.250) | 0.272 (0.272) | 0.251 (0.250) |
| vicreg_action_s43 | 0.333 (0.335) | 0.250 (0.250) | 0.279 (0.275) | 0.250 (0.250) |
| vicreg_action_s44 | 0.333 (0.333) | 0.250 (0.249) | 0.276 (0.274) | 0.250 (0.250) |
| vicreg_action_s45 | 0.328 (0.326) | 0.250 (0.251) | 0.271 (0.272) | 0.251 (0.251) |
| vicreg_noaction_s42 | 0.314 (0.314) | 0.250 (0.250) | 0.275 (0.275) | 0.250 (0.250) |
| vicreg_noaction_s43 | 0.300 (0.300) | 0.250 (0.250) | 0.263 (0.263) | 0.250 (0.250) |
| vicreg_noaction_s44 | 0.330 (0.330) | 0.250 (0.250) | 0.268 (0.268) | 0.250 (0.250) |
| vicreg_noaction_s45 | 0.349 (0.349) | 0.250 (0.250) | 0.262 (0.262) | 0.250 (0.250) |
| mse_action_s42 | 0.335 (0.335) | 0.252 (0.250) | 0.277 (0.275) | 0.253 (0.250) |
| mse_action_s43 | 0.321 (0.326) | 0.252 (0.250) | 0.259 (0.259) | 0.253 (0.247) |

On medium excitation the episode-mean top-1 sits on 1/4. For `vicreg_action_s42` at horizon 1 the equal-episode mean is 0.250 with a 95% interval 0.249 to 0.250, and no episode is excluded. The legacy horizon-1 interval for that checkpoint is 0.295 to 0.395 around an episode mean of 0.344, with 45 episodes excluded. That interval is not evidence of retrieval: the shuffled control has the same pooled rate, and the scored queries are the minority whose rasters differ.

Median MSE margins are negative, so the nearest wrong future is closer than the correct one. For `vicreg_action_s42` at horizon 1 the median margin is -0.00019 on legacy and -0.0050 on medium excitation. The oracle margins on those slices are positive. High-visibility legacy queries are not more accurate than low-visibility ones. The per-query table is `experiments/runs/future_ranking_v2/queries.csv`.
