# Reproducibility (internal + blinded)

The evaluation framework added on 2026-10-11 is documented in `docs/REPRODUCIBILITY.md`, `docs/EVALUATION.md`, and `docs/EXPERIMENTS.md`. Registered runs live in `experiments/registry.jsonl`. The sections below still describe the cached reviewer-strengthening package.

This repository's paper-facing analysis can be regenerated from **cached artifacts**
without retraining. Training and dump generation are explicit, expensive stages.

Use Python 3.13 with PyTorch. The workshop machine used in the audit is Apple
Silicon with MPS; CPU works for analysis.

```bash
python3.13 reproduce_paper_results.py --stage audit
python3.13 reproduce_paper_results.py --stage analysis
python3.13 reproduce_paper_results.py --stage counterfactual --max-val-episodes 200
```

Training a matched corrected no-crop run (does **not** write into `outputs/`):

```bash
bash scripts/train_reviewer_seed.sh 44 vicreg action \
  results/reviewer_strengthening/05_more_seeds vicreg_action_s44
```

## Stages

| Stage | Writes | Requires |
| --- | --- | --- |
| `audit` | `results/reviewer_strengthening/00_baseline_audit/` | existing `outputs/` checkpoints + dump |
| `counterfactual` | `01_simulator_counterfactual/` then 02/03 figures | dump + `outputs/vicreg_nocrop{,_seed43}/last.pt` |
| `analysis` | 02/03/paper_figures from Phase 1 CSVs | Phase 1 CSVs |
| `train-seed` | new `last.pt` under `results/reviewer_strengthening/` | dump, MPS/CPU, ~2.5 h/seed |

## Do not

- Overwrite `outputs/vicreg_nocrop/`, `outputs/vicreg_nocrop_seed43/`, or the physics dump.
- Mix `best.pt` (`outputs/ablation_twoseed.json`) with corrected `last.pt` headlines.
- Treat VN-MSE as implemented (`src/eval.py` does not compute it).

## Data

Dump seed 0, `outputs/physics_mpm/`. SHA-256 of `observations.npy` is recorded in
`results/reviewer_strengthening/00_baseline_audit/baseline_reproduction.json`.

Episode split: last 200 of 2000 episodes are validation (`val_fraction=0.1`).
Action z-score statistics are train-only.

## Counterfactual protocol

See `results/reviewer_strengthening/01_simulator_counterfactual/protocol.md`.

## Environment lock

`requirements-lock.txt` records versions from the audit machine. It is not a
fully frozen pip hash set for every transitive wheel.
