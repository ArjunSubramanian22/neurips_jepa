# Reproducibility

Use Python 3.13. The lockfile versions used for the runs in `experiments/registry.jsonl` are torch 2.12.1, NumPy 2.5.2, matplotlib 3.11.1, and PyYAML 6.0.3 (`requirements.txt`). `TrainingConfig.device="auto"` selects CUDA, then Apple MPS, then CPU, so a training rerun is device-dependent even with the same seed. The new registry runs in this pass were evaluation and inventory only.

## Registry

Every new run is created with `src.registry.RunRecorder`. The record contains:

- git commit, dirty flag, and dirty count
- config
- seed
- model
- environment
- dataset split
- checkpoint path, or an explicit note when there is no checkpoint
- command
- software versions
- hardware (CPU count, memory when `sysconf` provides it, CUDA and MPS availability)
- start and end timestamps
- output directory
- status: `RUNNING`, `COMPLETE`, or `FAILED`
- metrics
- failure reason when status is `FAILED`

Terminal lines are appended to `experiments/registry.jsonl`. `record.json` is written first with status `RUNNING`. An exception inside the recorder context marks the run `FAILED` and keeps the reason. Starting an id that already has a record raises `FileExistsError`. Do not delete a failed record to hide it. Use a new id.

Non-finite floats are rejected by `to_jsonable`. They are not written as null.

## Regenerating the new artifacts

From a clean checkout of commit `949564e` or later:

```bash
python3.13 -m unittest tests.test_cascade_metrics tests.test_splits \
  tests.test_planning_oracle tests.test_bootstrap tests.test_registry \
  tests.test_cascade_protocol
python3.13 scripts/run_cascade_protocol_smoke.py
python3.13 scripts/plot_cascade_layers.py
python3.13 scripts/run_planning_oracle.py
python3.13 scripts/audit_reproducibility_baseline.py
```

The smoke and planning commands refuse to overwrite `cascade_protocol_smoke_v1` and `planning_oracle_sanity_v1`. To rerun, choose a new experiment id in a new config rather than editing the completed JSON.

The figure is redrawn from `experiments/runs/cascade_protocol_smoke_v1/pairs.json` only. It does not resimulate.

## Historical paper artifacts

`python3.13 reproduce_paper_results.py` still regenerates the cached reviewer-strengthening analysis described in the root `REPRODUCIBILITY.md`. It does not train.

`scripts/evaluate_final.py` is the val-only evaluator for the ten no-crop `last.pt` checkpoints. It requires `outputs/physics_mpm/observations.npy`. That file is gitignored (about 375 MB) and was absent when `reproducibility_baseline_v1` ran. `dump_regeneration_hashcheck_v1` later regenerated it with `--workers 1` and `--seed 0`. The observations and actions matched `DATA_MANIFEST.csv` (`ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f` for observations), and the tensor was copied into place. Do not substitute a dump that fails that comparison.

Regeneration command, written beside the canonical dump only after the hashes match the manifest:

```bash
python3.13 scripts/generate_trajectories.py \
  --n-episodes 2000 --episode-length 16 --img-size 64 \
  --output outputs/physics_mpm --seed 0 --workers 1
```

Per-episode RNG is `seed + 10007 * episode_index`. The Linux replay tensor is not in this worktree. A local regeneration matching the manifest does not prove byte identity with that Linux file.

## Splits

Do not select hyperparameters on episodes 1800–1999 and then report those episodes as a test set. Future dump experiments declare `split_version: cvpr_v1` and follow `docs/DATA_SPLITS.md`. Historical tables stay `legacy_v1`.
