# Research changelog

## 2026-10-11

The worktree index had been emptied (every tracked path staged for deletion) while the files remained on disk, and `src/`, `scripts/`, `tests/`, and part of `results/` were missing. The index was rebuilt from `HEAD` `c9daa63` and the missing blobs were restored. History was not rewritten.

Added a layer-separated metric API, a historical-versus-fresh split helper, CEM and linear-dynamics planner checks, an equal-episode bootstrap, and an append-only experiment registry. These do not change `hist_v1` episode indices or `scripts/evaluate_final.py`.

Registered and completed, at code commit `949564e`:

- `cascade_protocol_smoke_v1` — reduced-simulator forks. Learned layers left `not_measured`.
- `planning_oracle_sanity_v1` — identity-dynamics recovery and random-ranking chance check. `success_criterion_met` true.
- `reproducibility_baseline_v1` — 12/12 checkpoint hashes matched. `observations.npy` absent.

No paper checkpoint was retrained or re-scored. No test-set protocol was edited after looking at test metrics. `fresh_pool_v2` was defined and not applied to the historical 2000 episodes.

`EXPERIMENT_MANIFEST.csv` was left unchanged. It still says the stiffness and state-perturbation controls were not run, while later files under `results/audit_phase2/` are present. That discrepancy is recorded in `docs/EXPERIMENTS.md` rather than silently rewritten.

`dump_regeneration_hashcheck_v1` regenerated the seed-0 dump with one worker. Observations and actions matched `DATA_MANIFEST.csv`, and the observation tensor was copied to `outputs/physics_mpm/observations.npy`. Episode 1800 of that file matches `replay_episode`.

`heldout_reproduction_v1` then recomputed the ten no-crop checkpoints with `scripts/evaluate_final.py`. MSE and variance-normalized MSE at horizons 1, 5, and 10 matched `results/final_checkpoint_results.csv` with absolute delta 0, inside the predeclared tolerance 0.001. The archive CSV was not overwritten.

## 2026-10-11 split protocol

Added `cvpr_v1` without moving `legacy_v1`. Episodes 1800–1999 stay legacy validation and are excluded from the new train, validation, and test roles. `cvpr_v1` is an exact 80/10/10 split of episodes `[0, 1800)`, with ids saved in `configs/protocols/cvpr_v1_episode_ids.json`. Archived validation numbers were not relabeled as test numbers. Historical checkpoints are refused on the `cvpr_v1` test ids because those checkpoints trained on them.

## 2026-10-11 excitation regimes

`MPMConfig()` remains the historical actuator. `excitation_calibration_v1` measured 58 environments and 1392 forks with no learned model. All 58 passed the stability screen. Selected regimes, from predeclared pixel-RMS bands on the force-reversed fork, are `mpm_legacy` (low), `mpm_medium_excitation`, `mpm_persistent_excitation`, and `mpm_high_excitation`. See `docs/EXCITATION_REGIMES.md`.

## 2026-10-11 material identifiability

`material_identifiability_v1` probed open-loop force schedules without training a world model. The predeclared validation rule returned NONE because the linear readout of collinear `(mu, lambda)` stayed at 0.667. Mechanical-history and uint8-history linear probes were above the stiffness bar for the stronger open-loop protocols, including `high_bend`. That protocol was not designated after the fact. No encoder-level stiffness-loss claim was added. See `docs/MATERIAL_IDENTIFIABILITY.md`.

## 2026-10-11 visibility sensitivity

`visibility_sensitivity_v1` kept the historical top-third raster-MSE rule and fit its value on `cvpr_v1` validation forks. 146 of 180 validation episodes had zero raster MSE. The nonzero quantile is 3.798e-6. `saw_test` is false, and the cut was not chosen from cosine or retrieval. On the persistent horizon-10 reporting rows, open-loop top-1 for `vicreg_action_s42` stays near one half from the top 10 percent through every nonzero row. Teacher-forced alignment stays high for the no-action checkpoints as well. See `docs/VISIBILITY_SENSITIVITY.md`.

## 2026-10-11 multi-horizon audit

`horizon_audit_v1` extended the counterfactual audit to horizons 1, 3, 5, and 10 under four labeled interventions. Open-loop recorded rollout error grows with horizon. Teacher-forced error stays at the one-step level. Open-loop counterfactual retrieval stays near one half, including the no-action checkpoints, whose open-loop cosine is undefined. Teacher-forced cosine is high for those same no-action checkpoints because they read the true frame. See `docs/HORIZON_AUDIT.md`.

## 2026-10-11 future ranking

`future_ranking_v1` failed during the first checkpoint because a status name was not imported. No summary was written. `future_ranking_v2` repeated the same protocol. On medium excitation, where nearly every held-action future is visible, pooled MSE top-1 is 0.25 and matches the shuffled and constant controls. The oracle that copies target embeddings scores 1 on every scored slice. Legacy validation leaves most horizon-1 queries observationally ambiguous even though the particles moved; those rows are counted and are not treated as misses. See `docs/FUTURE_RANKING.md`.

## 2026-10-11 directional counterfactuals

`counterfactual_direction_v1` scored force reversals for the ten no-crop `last.pt` checkpoints on legacy validation episodes 1800–1999. Each of the 30000 transitions is in the table. Cosine is null when either latent delta is at or below `1e-8`. The four no-action checkpoints have no valid cosine. Episodes 1957 and 1963 are listed in the bootstrap exclusions and are not imputed. These numbers are not a test set. See `docs/COUNTERFACTUAL_DIRECTION.md`.

## 2026-10-11 RGB observations

Added `rgb_v1` beside the density/velocity raster. `rgb_visibility_v1` compared force-reversal visibility on legacy and high excitation at RGB 64, 128, and 256. LPIPS was left unavailable. No world model was trained. See `docs/RGB_OBSERVATIONS.md`.
