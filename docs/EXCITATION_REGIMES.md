# Excitation regimes

`mpm_legacy` is the historical simulator. The other named regimes come from `excitation_calibration_v1`. That run did not train a model and did not read `cvpr_v1` test episodes.

Regenerate the tables and the figure:

```bash
python3.13 experiments/calibrate_excitation.py
```

The command refuses to overwrite `experiments/runs/excitation_calibration_v1`. A new sweep needs a new experiment id and a new criteria file. Do not edit the bins after reading the table.

## What is held fixed

Every candidate uses the historical grid (`n_grid=32`), timestep, substeps, render, and Young's moduli. The swept controls are damping, `action_gain`, contact radius, command hold, contact-weight mode, center influence, and force influence. Defaults of `MPMConfig()` stay on the legacy actuator path, so a default rollout matches `mpm_legacy`.

The descriptive horizontal axis in `excitation_curves.png` is

```
(action_gain / 55) * (0.08 / damping) * (command_hold / 3) * (contact_radius / 0.10)
```

Selection does not minimize or maximize that index.

## Forks

From each of the six class-by-geometry states, after one warmup frame, the factual command is held and then resampled every `command_hold` frames. The compared command replaces only that first hold block:

- `identical` — same command. Legacy rows have RMS position 0 and uint8 equality 1.
- `force_reversed` — negate the force channels. Regime selection uses this fork.
- `contact_shifted` — add `contact_radius` to `cx`. This is the center-coordinate probe.
- `zero_force` — zero the force channels.

Action-distance correlation uses every non-identical fork. A zero-variance sample is `undefined_zero_variance`, not a correlation of 0.

## Stability screen

A state is unstable if it is non-finite, more than 5% of particles sit on the box clip, more than 20% of deformation-gradient singular values sit on the SVD clip `[0.2, 4]`, or speed exceeds 25. Candidates whose force-reversed median RMS position exceeds 0.20 are also ineligible. All 58 environments in this sweep passed the stability screen.

## Visibility bins

A one-state range check, before the 58-environment run, showed uint8 change leveling off near 0.18 while pixel RMS still grew from gain 800 to gain 3200. The bins are therefore pixel-RMS intervals with a raster-change cap, not a search for the largest raster difference. They are recorded in `configs/experiments/excitation_calibration_v1.yaml`.

| Regime | Rule |
| --- | --- |
| low | pixel RMS ≤ 0.02 and raster change ≤ 0.10. Prefer `mpm_legacy` when it qualifies. |
| medium | pixel RMS in [0.08, 0.35], raster change in [0.05, 0.40], closest to pixel RMS 0.16 |
| high | pixel RMS in [0.45, 2], raster change in [0.05, 0.50], closest to pixel RMS 0.75 |
| persistent | hold ≥ 8, damping ≤ 0.03, pixel RMS in [0.08, 2], closest to pixel RMS 0.40 |

Persistent is chosen first so it is not reused as medium or high. An empty bin would stay `UNFILLED`.

## Selected settings

Medians below are the force-reversed fork over the six states. Source: `experiments/runs/excitation_calibration_v1/regime_summary.md`.

| Name | Damping | Gain | Radius | Hold | Pixel RMS | Raster change | RMS position |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mpm_legacy` | 0.08 | 55 | 0.10 | 3 | 0.00611 | 0.00631 | 9.69e-05 |
| `mpm_medium_excitation` | 0.03 | 800 | 0.10 | 8 | 0.156 | 0.158 | 0.00248 |
| `mpm_persistent_excitation` | 0.03 | 3200 | 0.16 | 8 | 0.590 | 0.173 | 0.00937 |
| `mpm_high_excitation` | 0.01 | 3200 | 0.10 | 8 | 0.770 | 0.175 | 0.0122 |

`local_peak` weights at gain 200, damping 0.03, hold 3 reached a larger pixel RMS (1.63) than the high regime. It was eligible for the high band and was not selected, because 1.63 is farther from the target 0.75 than 0.770. Files:

- `configs/environments/mpm_legacy.yaml`
- `configs/environments/mpm_medium_excitation.yaml`
- `configs/environments/mpm_high_excitation.yaml`
- `configs/environments/mpm_persistent_excitation.yaml`

Raw rows are `forks.csv` and `forks.parquet` in the run directory. Aggregates are `candidates.csv` and `candidates.parquet`.

## How to read this

These regimes separate a weak physical fork (`mpm_legacy`) from forks whose particle motion is a larger fraction of a pixel. Raster equality does not keep rising with that motion under the current uint8 renderer, so a high pixel RMS here is not a claim that most pixels change. No predictor was run, so the table does not say whether a learned action path uses the extra excitation.
