# Results

Generated from saved experiment files. No significance test was added.

Regenerate: `python3.13 scripts/generate_scientific_report.py`

Rows: 144. Flags: 42.

## Flags

- undefined_cosine: vicreg_noaction_s42
- undefined_cosine: vicreg_noaction_s43
- undefined_cosine: vicreg_noaction_s44
- undefined_cosine: vicreg_noaction_s45
- missing_seed: vicreg_action 46
- missing_seed: vicreg_noaction 46
- missing_seed: mse_action 44
- missing_seed: mse_action 45
- missing_seed: mse_action 46
- missing_seed: vicreg_action 46
- missing_seed: vicreg_noaction 46
- missing_seed: mse_action 44
- missing_seed: mse_action 45
- missing_seed: mse_action 46
- not_run: conditioning_film@42
- not_run: conditioning_gate@42
- not_run: conditioning_cross_attn@42
- not_run: model_rssm@42
- not_run: model_transformer@42
- not_run: model_pixel@42
- not_run: w0
- not_run: w0.01
- not_run: w0.1
- not_run: w1
- not_run: probe
- not_run: base
- not_run: direction
- not_run: magnitude
- not_run: direction_magnitude
- not_run: fraction_0.5
- not_run: fraction_1
- not_run: cls_uniform
- not_run: pool_uniform
- not_run: pool_soft
- not_run: pool_topk
- not_run: patch_tokens
- not_run: local_head
- not_run: small
- not_run: large
- unequal_sample_count: ranking_top1 [4, 4, 2]
- unequal_sample_count: cosine [4, 0, 2]
- suspicious_duplicate: final_distance_mean ['vicreg_noaction_s42', 'vicreg_noaction_s43', 'vicreg_noaction_s44', 'vicreg_noaction_s45']

## environment_visibility

OBSERVED RESULT

environment_visibility has 8 rows. Sample sizes present: 6. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Pixel differences describe the rendered forks. They do not say whether a learned model uses the action.

## model_rollout

OBSERVED RESULT

model_rollout has 20 rows. Sample sizes present: 1. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Latent MSE is on each checkpoint's own scale. A smaller number is not compared across objectives here.

## directional_counterfactual

OBSERVED RESULT

directional_counterfactual has 21 rows. Sample sizes present: 1. NaN values kept: 0. Undefined cosines kept: 4.

POSSIBLE INTERPRETATION

An undefined cosine is a zero difference vector, not a cosine of zero. Open-loop rows are the ones that do not read the true next frame.

## retrieval

OBSERVED RESULT

retrieval has 21 rows. Sample sizes present: 1, 3. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Retrieval is a hit rate on the scored rows stored in the source. Ties were not counted as hits by that source.

## planning

OBSERVED RESULT

planning has 25 rows. Sample sizes present: 1, 5. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Success and distance are the stored closed-loop outcomes. The report does not add a significance test.

## metric_planning_correlation

OBSERVED RESULT

metric_planning_correlation has 9 rows. Sample sizes present: 10, 6. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Spearman values and bootstrap intervals are copied from the source file. An interval is not labeled significant by this report.

## seed_variability

OBSERVED RESULT

seed_variability has 6 rows. Sample sizes present: 0, 2, 4. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Means use completed seeds only where the source says so. Missing seeds stay listed and are not imputed.

## architecture_ablation

OBSERVED RESULT

architecture_ablation has 6 rows. Sample sizes present: 0. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Queue status is not a trained metric. An unrun conditioning or baseline has no accuracy to compare.

## training_loss_ablation

OBSERVED RESULT

training_loss_ablation has 17 rows. Sample sizes present: 0. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

Unrun loss variants stay in the table with status NOT_RUN. No weight or lambda was selected.

## resolution_scale

OBSERVED RESULT

resolution_scale has 11 rows. Sample sizes present: 0, 4, 6. NaN values kept: 0. Undefined cosines kept: 0.

POSSIBLE INTERPRETATION

RGB rows are observation differences. Width rows with n 0 were not trained, so no scale trend is identified.
