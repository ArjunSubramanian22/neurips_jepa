# Episode bootstrap

Resamples: 5000
Seed: 0
Unit: validation episode (transitions within an episode are not treated as independent).
Statistic: mean of per-episode means (also reported in CSV).
CI: percentile 95%.
Do not interpret a tiny CI around a tiny effect as practical action dependence.

The **mean capture ratio is unstable**: pairs with `tgt_mse` just above 1e-8 produce huge ratios. Prefer the median capture ratio in `aggregate_metrics.json`.
