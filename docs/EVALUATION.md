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

`docs/DATA_SPLITS.md` is the split protocol. `legacy_v1` keeps train `[0, 1800)` and validation `[1800, 2000)` and has no test ids. Numbers on that tail stay legacy validation. `cvpr_v1` is the future study split: exact 80/10/10 of episodes `[0, 1800)`, with `[1800, 2000)` excluded. Headline metrics for `cvpr_v1` use the test ids only. Historical checkpoints are not scored on those test ids.

Action z-scoring uses training episodes only.

## What evaluates the revised-paper checkpoints

`scripts/evaluate_final.py` is the `legacy_v1` evaluator for the ten `last.pt` checkpoints: rollout MSE, variance-normalized MSE, per-horizon rank, and probes on validation episodes 1800–1999. It needs `outputs/physics_mpm/observations.npy`. Its output sets `is_test_set` false.

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
