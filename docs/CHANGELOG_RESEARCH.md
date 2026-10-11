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
