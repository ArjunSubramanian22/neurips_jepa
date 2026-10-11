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
