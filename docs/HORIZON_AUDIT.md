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

## Measured result

`horizon_audit_v1` has 64000 rows: 10 checkpoints, 200 episodes, four protocols, four horizons, and both rollout modes. `recorded_rollout_identical_across_protocols` is true. At horizon 1 the three force-reversal protocols match, and open-loop matches teacher forcing, because only one action has been applied.

Particle separation then splits by protocol. Median RMS position at horizon 10 is 4.72e-5 for one-step replacement, 1.42e-4 for the command block, 3.01e-4 for the persistent reversal, and 3.52e-5 for the shuffled-then-recorded action. The command block matches the persistent reversal at horizon 3, the end of the three-step block, and then falls behind. Every observationally ambiguous query in these slices is also physically distinct. At horizon 1, 142 of 200 force-reversal episodes have identical physical rasters. At horizon 10 the persistent reversal leaves 20 of 200 ambiguous.

Recorded-branch rollout MSE grows under open-loop recursion and stays flat under teacher forcing. For `vicreg_action_s42` the open-loop median is 0.073 at horizon 1 and 1.197 at horizon 10. The teacher-forced median stays between 0.073 and 0.074. The MSE checkpoints grow more slowly and remain lower.

Predictor cosine does not follow that physical separation under open-loop rollout. No-action checkpoints have an undefined open-loop cosine at every horizon, because the predicted delta is at or below `1e-8`. Action-conditioned open-loop cosines are mostly defined and stay near zero. Teacher-forced cosine is high for both action and no-action checkpoints, because that diagnostic reads the true frame. It is not evidence that the action input is being used.

Persistent reversal, horizon 10, pooled top-1 on scored queries. Chance for two futures is 1/2. Twenty ambiguous episodes are excluded from every checkpoint, not imputed.

| Checkpoint | Open-loop top-1 | Open-loop cosine | Teacher-forced top-1 | Teacher-forced cosine |
| --- | --- | --- | --- | --- |
| vicreg_action_s42 | 0.528 | 0.031 | 0.839 | 0.964 |
| vicreg_action_s43 | 0.450 | -0.015 | 0.756 | 0.954 |
| vicreg_action_s44 | 0.500 | 0.012 | 0.844 | 0.970 |
| vicreg_action_s45 | 0.439 | -0.001 | 0.833 | 0.969 |
| vicreg_noaction_s42 | 0.444 | undefined (0/200) | 0.828 | 0.969 |
| vicreg_noaction_s43 | 0.411 | undefined (0/200) | 0.822 | 0.973 |
| vicreg_noaction_s44 | 0.500 | undefined (0/200) | 0.828 | 0.969 |
| vicreg_noaction_s45 | 0.483 | undefined (0/200) | 0.839 | 0.965 |
| mse_action_s42 | 0.456 | 0.066 | 0.844 | 0.987 |
| mse_action_s43 | 0.433 | 0.081 | 0.867 | 0.991 |

For `vicreg_action_s42` the equal-episode open-loop interval is 0.456 to 0.600 around 0.528, and the teacher-forced interval is 0.783 to 0.889 around 0.839. Both use 180 episodes and list the other 20. Target-direction cosine against horizon 1 is defined for 58 of 200 persistent episodes for that checkpoint, median 0.410. The forks and target embeddings are under `experiments/runs/horizon_audit_v1/cache` and `targets`.
