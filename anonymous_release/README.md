# Anonymous artifact — PhysWorldAI workshop submission

This directory is a **double-blind** snapshot of the codebase and cached
results. It is not a git repository. The submission studies action-conditioned
joint-embedding predictors on a synthetic MLS-MPM deformable-object dump.

Nothing in this artifact identifies the authors. Third-party citation authors,
package names, and license credits are retained on purpose.

## Directory structure

| Path | Role |
| --- | --- |
| `src/` | Model, data, training, and evaluation library |
| `configs/workshop_vicreg.yaml` | Canonical 200-epoch no-crop VICReg schedule |
| `scripts/evaluate_final.py` | **Canonical held-out evaluator** for the ten checkpoints |
| `scripts/regenerate_cached_tables.py` | Rebuild markdown tables from cached CSVs |
| `scripts/check_anonymity.py` | Recursive anonymity / secret scanner |
| `scripts/generate_trajectories.py` | NumPy MLS-MPM dump generator |
| `tests/` | Unit tests (no training) |
| `outputs/` | Canonical `last.pt` checkpoints and dump sidecars |
| `results/` | Cached metrics, counterfactuals, figures |
| `CHECKPOINT_MANIFEST.csv` | SHA-256 for every paper checkpoint |
| `DATA_MANIFEST.csv` | SHA-256 for dump tensors, including omitted files |
| `EXPERIMENT_MANIFEST.csv` | Which runs are in this artifact |

## Environment

Python 3.13 and PyTorch 2.12 are the recorded versions (`requirements.txt`,
`requirements-lock.txt`). CPU is sufficient to inspect cached tables and to
import the library. Re-encoding the dump or re-evaluating checkpoints needs
the omitted `observations.npy` plus a GPU or MPS device for speed, not for
correctness of the cached CSVs.

```bash
pip install -r requirements.txt
python -c "import src; print(src.__version__)"
python scripts/evaluate_final.py --help
python scripts/regenerate_cached_tables.py
python scripts/check_anonymity.py
```

Optional W&B logging in `src/train.py` activates only if `WANDB_API_KEY` is
set in the environment. This artifact does not contain keys.

## Canonical checkpoint set

All of the following are epoch 200 with `crop_scale_min = 1.0`. SHA-256 values
are in `CHECKPOINT_MANIFEST.csv`.

| Condition | Seeds | Path |
| --- | --- | --- |
| VICReg + action | 42, 43, 44, 45 | `outputs/vicreg_nocrop/last.pt`, `outputs/vicreg_nocrop_seed43/last.pt`, `results/reviewer_strengthening/05_more_seeds/vicreg_action_s{44,45}/last.pt` |
| VICReg no-action | 42, 43, 44, 45 | `outputs/no_action/last.pt`, `outputs/vicreg_nocrop_noaction_seed43/last.pt`, `results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s{44,45}/last.pt` |
| MSE-only + action | 42, 43 | `results/reviewer_strengthening/06_corrected_objective/mse_action_s{42,43}/last.pt` |

Obsolete cropped checkpoints (`crop_scale_min = 0.75`) are **not** shipped.

## Canonical evaluator

**`scripts/evaluate_final.py`** is the evaluator for the ten-checkpoint held-out
tables:

- data root: `outputs/physics_mpm` (original dump, not a Linux replay tensor)
- split: probes on episodes `0:1800`; rollout MSE / VN-MSE / rank on `1800:2000`
- writes: `results/final_evaluation/*.json` and `results/final_checkpoint_results.csv`

Do **not** use `src/eval.py::evaluate_model` for those tables. That helper can
pool train+val episodes and does not implement VN-MSE.

Re-running `evaluate_final.py` requires `outputs/physics_mpm/observations.npy`.
That file is omitted (see below). The cached CSV is the reproduction path for
reviewers who do not have the dump.

## Regenerating figures and tables from cache

Without checkpoints or the observation dump:

```bash
python scripts/regenerate_cached_tables.py
```

writes `results/regenerated/heldout_eval_table.md` from
`results/final_checkpoint_results.csv`.

Counterfactual figures can be rebuilt from the shipped per-example CSVs:

```bash
python scripts/analyze_simulator_counterfactual.py
```

That command does not re-simulate and does not overwrite the raw per-example
CSVs. It may refresh derived figures under `results/reviewer_strengthening/`.

This artifact does **not** include manuscript LaTeX, a compiled workshop PDF,
or a bibliography. Inspect cached CSVs, JSON, and figure files instead.

## Cached results that correspond to the reported experiments

| Object | Cached file |
| --- | --- |
| Held-out MSE / VN-MSE / rank / probes | `results/final_checkpoint_results.csv` |
| Per-run JSON | `results/final_evaluation/*.json` |
| Physical / target / predictor counterfactuals | `results/reviewer_strengthening/01_simulator_counterfactual/` |
| Action-distance bins | `results/reviewer_strengthening/02_action_distance/` |
| Episode bootstrap | `results/reviewer_strengthening/03_episode_bootstrap/` |
| Candidate figures | `results/reviewer_strengthening/paper_figures/` |
| Actuator / damping / visibility | `results/audit_phase1/` |
| VN + state perturbation; stiffness controls | `results/audit_phase2/` |
| Dump replay identity (local RNG replay) | `results/replay_diff_report.csv` |

## Large artifacts omitted

| Artifact | Bytes | SHA-256 | Why omitted |
| --- | --- | --- | --- |
| `outputs/physics_mpm/observations.npy` | 393216128 | `ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f` | ~375 MB raster dump |
| Linux replay `observations.npy` | (not shipped) | `aedda53c0486f2dea3903d78bf360208e57040665defbeb00064180716485a7b` | Cross-machine replay tensor; not in the working tree used for this snapshot |
| Cropped `outputs/vicreg/last.pt` and `outputs/mse/last.pt` | — | see `CHECKPOINT_MANIFEST.csv` (`used_in_revised_paper=False`) | Superseded `crop_scale_min=0.75` runs |

Canonical `last.pt` files (~12 MB each, ten files) **are** included.

## Reproduction limits

- Exact re-evaluation of Table-scale held-out metrics needs the omitted
  `observations.npy` whose hash is above. Cached CSVs are bitwise copies of the
  original working-tree results.
- Local RNG replay of the dump matches the stored observations on the machine
  that produced `results/replay_diff_report.csv`. That does **not** claim
  byte-identity with the Linux replay tensor.
- `scripts/generate_trajectories.py` can build a new dump; bitwise identity
  with the omitted file is not claimed across OS / BLAS / worker settings.
- Do not retrain to match the shipped tables. Different kernels (CUDA vs MPS
  vs CPU) change training even with the same seed.
- Headline ten-checkpoint numbers live in `results/final_checkpoint_results.csv`.

This artifact does not claim PhysWorld / PhysTwin SOTA numbers, does not claim
that MSE-only low error is better physics, and does not claim that stiffness
is linearly readable from the encoder under this excitation.

## Anonymity and licenses

```bash
python scripts/check_anonymity.py
```

See `ANONYMITY_AUDIT.md` and `LICENSE_AUDIT.md`. Third-party names in code
comments and license notes (cited papers, package authors) are not submitting-
author identity.
