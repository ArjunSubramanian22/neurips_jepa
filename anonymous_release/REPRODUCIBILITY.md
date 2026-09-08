# Reproducibility

Paper-facing analysis in this artifact can be regenerated from **cached results**
without retraining. Training and dump generation are explicit, expensive stages
and are not required to inspect the shipped tables.

Use Python 3.13 with PyTorch. Cached analysis runs on CPU. The shipped
checkpoints were trained with `device=auto` (MPS on Apple Silicon in the
recorded `history.json` files).

## Canonical held-out evaluation

The evaluator that reproduces the ten-checkpoint held-out tables is:

```bash
python scripts/evaluate_final.py --help
```

It uses validation episodes 1800–1999, VN-MSE, and per-horizon rank. Cached
output is `results/final_checkpoint_results.csv`. Re-running it requires
`outputs/physics_mpm/observations.npy` (omitted; SHA-256 in `DATA_MANIFEST.csv`).

Do **not** use `src/eval.py::evaluate_model` for those tables. That path can
average rollout MSE over all loaded episodes and does not implement VN-MSE.

## Cached regeneration (no training, no dump)

```bash
python scripts/regenerate_cached_tables.py
python scripts/analyze_simulator_counterfactual.py
```

The second command rebuilds figures from the Phase-1 CSVs already present under
`results/reviewer_strengthening/`. It does not overwrite the raw per-example CSVs.

## Optional expensive stages (not needed for shipped tables)

```bash
python reproduce_paper_results.py --stage analysis
bash scripts/train_reviewer_seed.sh 44 vicreg action \
  results/reviewer_strengthening/05_more_seeds vicreg_action_s44
```

Do not overwrite the shipped `last.pt` files if the goal is to match the
cached tables.

## Data

Dump seed 0, `outputs/physics_mpm/`. SHA-256 of `observations.npy` is recorded in
`DATA_MANIFEST.csv` and `results/reviewer_strengthening/00_baseline_audit/baseline_reproduction.json`.

Episode split: last 200 of 2000 episodes are validation (`val_fraction=0.1`).
Action z-score statistics are train-only.

## Counterfactual protocol

See `results/reviewer_strengthening/01_simulator_counterfactual/protocol.md`.

## Environment lock

`requirements-lock.txt` records versions from the audit environment. It is not a
fully frozen pip hash set for every transitive wheel.
