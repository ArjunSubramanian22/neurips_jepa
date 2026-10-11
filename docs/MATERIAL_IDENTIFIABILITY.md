# Material identifiability

This study asks whether a force protocol can make stiffness visible in the dynamics and in the raster, before any new world model is trained. It does not write the stiffness label into the image or the action.

`legacy_random` resamples commands from the deformed state, so later actions can depend on stiffness. It is a negative control and is not selectable. The other schedules are open-loop: the contact is the rest centroid, and the force is a function of time. A matched-geometry check confirms that changing Young's modulus does not change those actions.

## Protocol

Each protocol rolls 120 episodes of length 24. The split is `fresh_pool_v2`: train `[0, 96)`, validation `[96, 108)`, test `[108, 120)`. Class and geometry counts match inside each block because each length is a multiple of 6. Probes are fit on train only. Features are standardized from the training rows only.

The readout is fixed: 80 full-batch AdamW steps, learning rate 0.01, weight decay 1e-4, seed 0. The MLP has one hidden layer of width 128. Raster probes see a fixed 48-dimensional Gaussian projection of the full 64×64 frame. The projection seed is 0 and does not depend on labels. Mechanical features are `x, v, C, F` for up to 64 spatially strided particles, ordered by the rest position. `mu` and `lambda` are not included in the mechanical features.

History length `k` is the first `k` saved frames after the rest frame. Saved times are 0, 4, 8, 12, 16, 20, and 23. The selection history length is 6.

Regenerate:

```bash
python3.13 experiments/material_identifiability/run_study.py
```

That id is already complete. A changed probe or a changed oracle feature needs a new experiment id.

## Selection rule

A selectable, stable protocol is admissible only if validation linear accuracy meets every bar:

- privileged `(mu, lambda)` stiffness at least 0.99
- geometry from the rest uint8 frame and from the rest mechanical state at least 0.90
- mechanical-history stiffness and uint8-history stiffness at least chance + 0.20, with chance `1/3`

The designated regime would have been the admissible protocol with the highest validation uint8-history accuracy. Test accuracy is stored in `probe_accuracy.csv` and is not an input to `select_protocol`.

## Result

`material_identifiability_v1` returned **NONE**. Every protocol was numerically stable. Geometry linear validation accuracy was 1.0 on both rest uint8 and rest mechanical state. The linear privileged probe was 0.667 on train, validation, and test for every protocol. `mu` and `lambda` are both proportional to Young's modulus, so those two coordinates are one degree of freedom, and this fixed linear probe did not separate the middle class. The MLP on the same privileged features was 1.0 on validation for every protocol, so the class is identified by the material parameters under the nonlinear readout.

Validation linear stiffness accuracy at history length 6:

| Protocol | Mechanical history | uint8 history | Selectable |
| --- | --- | --- | --- |
| legacy_random | 0.250 | 0.500 | no |
| legacy_orbit | 0.917 | 0.500 | yes |
| medium_orbit | 1.000 | 0.667 | yes |
| high_orbit | 1.000 | 0.917 | yes |
| high_push_release | 0.917 | 0.917 | yes |
| high_bend | 1.000 | 1.000 | yes |
| persistent_compression | 0.917 | 0.500 | yes |

`high_bend`, `high_orbit`, and `high_push_release` clear the mechanical-history and uint8-history bars on these 12 validation episodes. They are not designated regimes, because the predeclared rule also required the linear oracle, and that bar failed. The rule was not edited after the table was seen.

The sealed test split, which was not used to choose, gives the same pattern for the linear history-6 probes: `high_bend` is 1.0 for both mechanical history and uint8 history (12/12). `legacy_random` test uint8 history is 0.50 and mechanical history is 0.417. With 12 test episodes, a perfect count is not a large-sample rate.

## Figures

- `figures/material_identifiability/accuracy_vs_excitation.png` — validation linear stiffness accuracy by protocol
- `figures/material_identifiability/accuracy_vs_history_uint8_val.png`
- `figures/material_identifiability/accuracy_vs_history_mechanical_val.png`
- `figures/material_identifiability/accuracy_vs_history_high_bend.png` — validation and confirmatory test for the highest validation uint8-history protocol; it is not a designated regime
- confusion matrices for that same protocol's linear history-6 probes, validation and test

Raw rows: `experiments/runs/material_identifiability_v1/probe_accuracy.csv`.

## What this does not say

This run does not establish that an encoder discards stiffness. The legacy random protocol still leaves mechanical-history stiffness at or below chance, which is why a near-chance latent probe on that simulator cannot be read as encoder loss. Stronger open-loop bending and orbiting make stiffness linearly readable from the uint8 history in this small split. A follow-up that replaces the collinear `(mu, lambda)` oracle, or that accepts the MLP oracle, is a new experiment.
