# Evaluation protocol

This repository separates five layers. A result at one layer is not evidence about the layers below or above it.

1. **Physical response.** Particle-state change under an action intervention. Reported as RMS position and, when velocities are present, RMS velocity.
2. **Observation visibility.** Whether that change appears in the rendered raster. Reported as the fraction of identical elements and the mean squared error. Integer frames are compared exactly and scaled by 255 before the squared error.
3. **Representation response.** Change in an encoder embedding of the two futures. This layer stays `not_measured` until an encoder checkpoint is actually run.
4. **Predictor action response.** Change in the action-conditioned predictor when the same latent state is paired with two actions, including the cosine between the predictor difference and the target-encoder difference. A zero difference vector has cosine `null` and status `undefined_zero_vector`. It is not stored as zero.
5. **Downstream decision utility.** Whether a planner recovers a designated action. This is a top-1 hit against uniform chance. It is not a latent MSE.

`assemble_cascade` has no combined score. Variance-normalized latent MSE divides by a declared reference variance. A non-positive variance is `undefined_nonpositive_variance` and is counted by `summarize_defined` rather than dropped.

Latent distance matches the held-out evaluator in `scripts/evaluate_final.py`:

```
D = mean_i (u_i - v_i)^2
```

## Splits

`hist_v1_last_fraction_val` (`configs/protocols/hist_v1_split.yaml`) is the historical loader rule. For the 2000-episode dump and `val_fraction=0.1` the intervals are train `[0, 1800)` and validation `[1800, 2000)`. There is no test interval. Training can write `best.pt` from validation loss, and the revised paper evaluates that same tail. Do not call it an untouched test set, and do not repartition it after seeing reported metrics.

`fresh_pool_v2` (`configs/protocols/fresh_pool_v2_split.yaml`) is an 80/10/10 contiguous split for a newly generated pool whose length is a multiple of 60. The suffix is test and is closed to training, checkpoint selection, and hyperparameter search. `fresh_pool_v2_split(2000)` raises, because this protocol does not carve a test set out of the historical dump.

Action z-scoring for historical evaluations uses training episodes only, as in `scripts/evaluate_final.py`.

## What evaluates the revised-paper checkpoints

`scripts/evaluate_final.py` is the val-only evaluator for the ten `last.pt` checkpoints: rollout MSE, variance-normalized MSE, per-horizon rank, and probes on episodes 1800–1999. It needs `outputs/physics_mpm/observations.npy`.

`src/eval.py:evaluate_model` averages rollout MSE over the episodes it is given and is not the source of the revised-paper tables.

## Commands

Layer metrics and known-value tests:

```bash
python3.13 -m unittest tests.test_cascade_metrics tests.test_splits tests.test_planning_oracle tests.test_bootstrap tests.test_registry tests.test_cascade_protocol
```

Reduced-simulator cascade check, then the figure from the saved pairs:

```bash
python3.13 scripts/run_cascade_protocol_smoke.py
python3.13 scripts/plot_cascade_layers.py
```

Planner sanity check on identity residual dynamics:

```bash
python3.13 scripts/run_planning_oracle.py
```

Worktree hash inventory:

```bash
python3.13 scripts/audit_reproducibility_baseline.py
```

Reproduce the archived val-only MSE table into a new directory:

```bash
python3.13 scripts/run_heldout_reproduction.py
```

`heldout_reproduction_v1` already occupies that experiment id. A further reproduction needs a new id.

The reduced simulator in the cascade smoke uses `n_grid=16`, `img_size=32`, and `substeps=4`. The historical dump uses `n_grid=32`, `img_size=64`, and `substeps=8`. Those RMS values are not interchangeable.
