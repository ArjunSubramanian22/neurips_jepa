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
