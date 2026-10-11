# Multi-horizon counterfactual audit

Branch A follows the recorded action sequence from a shared simulator state. Branch B is one labeled intervention. A metric is never pooled across protocols or across rollout modes.

The fork starts at `t = 0`, which is a command boundary for the legacy hold of 3. Horizons are 1, 3, 5, and 10 observation frames. The recorded sequence has 15 actions, so those horizons fit. A horizon that does not fit the action sequence raises instead of being dropped.

## Protocols

| Protocol | Action sequence on branch B |
| --- | --- |
| `one_step_replacement` | Force reversal at step 0 only, then the recorded actions |
| `command_block_replacement` | Force reversal held for the first command block (3 steps), then the recorded actions |
| `persistent_alternative` | Force reversal at every step |
| `alternative_then_recorded` | A shuffled validation action at step 0, then the recorded actions |

The force reversal negates the force channels of the recorded action at the fork. The shuffled action is another validation episode's recorded action at the same time, from a derangement with seed 0. `one_step_replacement` and `alternative_then_recorded` share a first-step mask and differ in the action source. Both stay labeled.

## Rollout modes

`open_loop` encodes the shared start frame and then feeds the predictor its own previous prediction for each action in the sequence.

`teacher_forced` predicts the frame at horizon `h` from the true frame at `h - 1` and the action that produced it. It does not consume earlier predictions.

Ordinary rollout error is the latent MSE of the recorded branch. It does not depend on the intervention. Counterfactual rollout error is the latent MSE of branch B and does depend on the protocol.

## Layers

Physical RMS uses particle positions and velocities. Observation metrics are reported separately for the physical raster the model was trained on and for `rgb_v1`. Identical rasters make retrieval `observationally_ambiguous` even when the particles differ; those rows stay in the table and out of the top-1 denominator. Target direction compares the horizon-`h` target delta with the horizon-1 target delta, and is `not_applicable` at horizon 1. Predictor cosine is null when either delta has norm at or below `1e-8`.

Episodes 1800–1999 are legacy validation. `is_test_set` is false. These checkpoints were trained on `mpm_legacy`.
