# Artifact provenance map (reproducibility audit)

**Audit date:** 2026-09-04  
**Git HEAD:** see `git rev-parse HEAD` (paper macro `\GitHead`).  
**Working tree:** live `outputs/vicreg/` is still a moving checkpoint (PID 92746, `train_and_eval`) until 200 epochs. After that job exits, `scripts/finish_sept9.sh` trains seed-43 VICReg+MSE, writes effective-rank probes, regenerates figures, syncs the paper, and commits each milestone. `outputs/physics_mpm/observations.npy` is gitignored (GitHub 100 MB limit); regenerate with `scripts/generate_trajectories.py`.  
**Python / torch:** 3.13.2 / torch 2.12.1 (macOS arm64).  
**`TrainingConfig.device="auto"`:** CUDA if available, else Apple MPS, else CPU. Different machines therefore train different kernels even with the same seed.  
**Sept 9 scope:** 2-seed (42, 43) full VICReg and MSE only. The 3-seed × 4-variant grid is skipped. Report mean ± half-range, labeled as a 2-seed range, not a std.

Do **not** copy numbers from flagged rows into the paper.

---

## Do not paper over (untraceable or mixed-scale)

| Artifact | Problem |
| --- | --- |
| Entire `outputs/` tree vs git | Code that produced later artifacts is uncommitted; HEAD is an older snapshot. |
| `outputs/ablation_multiseed.json` / `ablation_summary.csv` | **Skipped** for the Sept 9 deadline. Paper uses `outputs/ablation_twoseed.json` (VICReg vs MSE, seeds 42 and 43, mean ± half-range). |
| `outputs/ablation_drift.png` | Regenerated from a **temporary 3-seed 1-epoch dummy** run, not from `ablation_results.csv` (single-seed, n_val=12) and not from workshop JSON. The PNG and the CSV describe different experiments. |
| `outputs/vicreg/{best,last}.pt`, `history.json` | In-progress **200-epoch workshop** job on MPS; `history.json` still records `epochs: 1` from an earlier write. Checkpoint is a moving target. |
| `outputs/probe_diagnostics.json` → `vicreg_physics_mpm` | Explicitly copied from that in-progress `best.pt`, not a finished 200-epoch model (`checkpoint_note` in the JSON). |
| `outputs/paper_tables.tex` probes / latency | Mix of dummy-ablation CSV MSE and a later live eval of 1-epoch MPS checkpoints; n_val=12. |
| `outputs/figures/rollout_comparison_*.png` | VICReg decode uses the in-progress workshop checkpoint + a **post-hoc** decoder; Pixel-AE row is a short qualitative fit, not Variant C at 200 epochs. |
| `outputs/physics_smoke/` | One-off CPU smoke against a physics path; not referenced by paper tables. |
| `outputs/.mplconfig/` | Matplotlib cache, not a result. |

---

## Global seeds (when a factory was used)

| Factory | Training seed | Data seed | Device in factory |
| --- | --- | --- | --- |
| `src/train.py:build_smoke_config` | **42** | **0** | `auto` (resolved per machine) |
| `configs/workshop_vicreg.yaml` | **42** | dump seed **0** (see physics metadata) | `auto` |
| `src/baselines.py:build_baseline_config` | **42** (training) | **0** (dataset) | CLI `--device` (reported run: **cpu**) |
| `src/ablation.py:build_ablation_config` | **42** | smoke data **0** | CLI `--device` (reported dummy run: **mps**) |
| Physics dump | — | **0** (`metadata.json`) | NumPy CPU, `workers=7` |

Per-episode physics RNG is `seed + 10007 * episode_id` (`scripts/generate_trajectories.py`), so worker count should not change the dump if that formula is kept.

---

## Mapping: paper-facing figure/table → file → config → command

Format: **paper slot ← file ← config ← command (device, seed)**.

### Data (Prompt 1)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Physics dump used by workshop train / probes / rollout figures | `outputs/physics_mpm/{observations,actions,stiffness_labels,stiffness_values,geometry_labels}.npy` + `metadata.json` | In-repo NumPy MLS-MPM (`scripts/generate_trajectories.py`); **not** a Taichi install (taichi absent). `seed=0`, N=2000, T=16, 64×64, E∈{18,70,220}, workers=**7** | `python scripts/generate_trajectories.py --n-episodes 2000 --output outputs/physics_mpm --seed 0` (workers defaulted to auto=7) |
| Dataset contract stats | `outputs/dataset_report.json` | Same dump | `python scripts/validate_dataset.py --dataset-path outputs/physics_mpm --report outputs/dataset_report.json` |

**Flag:** dump is uncommitted; command was not recorded in a run script at generation time. Metadata is the only seed record.

### Smoke train + archival eval (Prompt 2, dummy lattice)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| **Do not use as powered probes.** Chance-level curves with n_val=12 | `outputs/metrics.json`, `outputs/metrics.png`, `outputs/run_report.json` | `build_smoke_config(epochs=5)` — 48 eps, T=11, 32×32, seed_train=42, seed_data=0 | `python src/train.py --mode train_and_eval` (defaults). Device in `run_report.json`: **cpu**, amp false. VICReg + MSE, 5 epochs each. |
| Smoke checkpoints | `outputs/vicreg/last.pt` **(overwritten later by workshop)**; `outputs/mse/{last,best}.pt` | Same smoke factory | Same command. **`outputs/vicreg/` is no longer this run** — workshop MPS job replaced it. MSE dir still looks like the 5-epoch CPU smoke (`history.json` epochs=5, device=cpu). |

**Flag:** `metrics.json` VICReg numbers cannot be re-eval’d from the current `outputs/vicreg/` checkpoint.

### Workshop train (Prompt 2, intended paper scale)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Intended VICReg/MSE/Pixel-AE 200-epoch models | `outputs/vicreg/*` (in progress); MSE/pixel_ae workshop **not started** | `configs/workshop_vicreg.yaml` (seed 42, batch 128, 200 epochs, amp true, `device: auto` → **MPS** on this Mac) | `python src/train.py --config configs/workshop_vicreg.yaml --mode train_and_eval` |
| `outputs/vicreg/history.json` | claims `epochs: 1` | same | **Stale/partial write** while training continued. Not a finished log. |

### Dummy ablation CSV / drift (early Prompt 5, not workshop)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Point-estimate ablation MSE/cosine in `paper_tables.tex` (k=1,5,10,20) | `outputs/ablation_results.csv`, `outputs/ablation_report.json` | `build_ablation_config` (smoke lattice, T≥21, **1 epoch** per `history.json`) | `python src/ablation.py` with then-default smoke path. Device **mps**, amp true, n_val=**12**. Seeds: training 42. |
| Variant checkpoints | `outputs/ablation/{full_vicreg,variant_a_no_cov,variant_b_mse,variant_c_pixel_ae}/` | Same | Same. `full_vicreg` best_val is null (NaN val loss that run). |
| `outputs/ablation_drift.png` | **does not match the CSV** | temporary 3-seed 1-epoch dummy via `plot_multiseed_drift` | Prompt 6 regenerated error bands from a tempfile multi-seed smoke, then copied the PNG. **Do not caption this as workshop mean±std.** |

### Probe diagnostics (Prompt 3)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Smoke probe numbers (`mse_smoke`, `vicreg_smoke`) | `outputs/probe_diagnostics.json` | smoke checkpoints + `build_smoke_config` data | `python scripts/diagnose_probes.py` on CPU. n_val=12. |
| Physics probe numbers (`vicreg_physics_mpm`) | same JSON | workshop YAML data (`outputs/physics_mpm`), **early** `outputs/vicreg/best.pt` | Same script; copied checkpoint to a temp file. Device CPU for probes. Seed of the *model* is workshop 42; dump seed 0. |
| Limitation paragraph | `paper_limitation` field in that JSON | — | Hand-written in `scripts/diagnose_probes.py`, numbers filled from the eval. |

### Baselines (Prompt 4)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Within-simulator table | `outputs/baseline_comparison.csv`, `.json`, `.md` | `build_baseline_config(epochs=8, n_episodes=160)` — n_grid=8, T=12, val_fraction=0.25, train seed **42**, data seed **0** | `python src/baselines.py --epochs 8 --n-episodes 160 --device cpu` |
| Per-method checkpoints | `outputs/baselines/{ac_jepa_vicreg,ac_jepa_no_cov,ac_jepa_mse,pixel_ae,no_action,particle_gnn}/` | same | same. Device **cpu**, 8 epochs, n_train=120, n_val=40. |

**Flag:** mass-spring lattice, **not** `outputs/physics_mpm`. GNN `rollout_mse` is a different space (see the md file).

### Qualitative figures (Prompt 6)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| Human-checkable rollouts, stiffness 18/70/220 | `outputs/figures/rollout_comparison_{1800,1801,1802,1803}.png` | workshop dump + in-progress VICReg + post-hoc `PixelDecoder` | `python scripts/render_rollout_comparison.py --dataset-path outputs/physics_mpm --device cpu` |
| Worst k=10 latent MSE | `outputs/figures/failure_case.png` (episode **1841**, E=220) | same | same script (argmax on val) |
| Caption | `outputs/figures/failure_case_caption.txt` | text grounded in `probe_diagnostics.json` | written by that script |
| Post-hoc decoder weights | `outputs/figures/latent_decoder.pt` | not part of AC-JEPA | trained during render; **not** a reported model |
| Pixel-AE row | `outputs/figures/pixel_ae_qual/last.pt` | short CPU fit, same dump | **not** workshop Variant C |

### Paper export (Prompt 5/6)

| Paper slot | File | Config | Command / notes |
| --- | --- | --- | --- |
| `tab:ac-jepa-ablation` | `outputs/paper_tables.tex` | MSE from dummy `ablation_results.csv`; probes/latency from live checkpoint eval | `python src/utils/export_paper_artifacts.py` |
| `fig:ac-jepa-latent-traj` | `outputs/paper_latent_trajectories.png` / `.pdf` | PCA/t-SNE of dummy full_vicreg (or first available ckpt) | same exporter (`live_eval`) |
| Multi-seed mean±std table | **not generated (skipped)** | would need `outputs/ablation_multiseed.json` | 3-seed × 4-variant grid skipped for Sept 9 |
| 2-seed VICReg vs MSE range | `outputs/ablation_twoseed.json` (pending until seed 43 finishes) | `configs/workshop_vicreg.yaml`, seeds 42 and 43 | `scripts/finish_sept9.sh` → `scripts/two_seed_range.py` |
| Effective rank | `effective_rank` key in `outputs/probe_diagnostics.json` (pending final ckpts) | val-split covariance participation ratio | `scripts/diagnose_probes.py --physics-only` after 200-epoch VICReg+MSE |

---

## Per-file checklist (`outputs/*.json`, `outputs/*.csv`)

| File | Seed | Device | Epochs / N | Traceable command? |
| --- | --- | --- | --- | --- |
| `physics_mpm/metadata.json` | dump 0 | CPU (NumPy) | N=2000 | Partial (no logged argv; workers=7) |
| `dataset_report.json` | — | — | N=2000 | Yes, validate_dataset |
| `metrics.json` | train 42 / data 0 | cpu | 5 / n_val=12 | Yes, smoke train_and_eval — **but vicreg ckpt overwritten** |
| `run_report.json` | same | cpu | 5 | Yes, same |
| `mse/history.json` | 42 | cpu | 5 | Yes, smoke MSE |
| `vicreg/history.json` | 42 (yaml) | mps | log says 1; job still running | **Broken log / in-progress** |
| `ablation_results.csv` | 42 | mps | 1 / n_val=12 | Yes, early ablation.py |
| `ablation_report.json` | 42 | mps | 1 | Yes, same |
| `ablation/*/history.json` | 42 | mps | 1 | Yes |
| `baseline_comparison.csv` | 42 / data 0 | cpu | 8 / n_val=40 | Yes, baselines.py |
| `baseline_comparison.json` | same | cpu | 8 | Yes |
| `baselines/*/history.json` | 42 | cpu | 8 | Yes |
| `probe_diagnostics.json` | mixed | cpu eval | mixed n_val 12 and 200 | Script yes; **physics model unfinished** |
| `physics_smoke/vicreg/history.json` | 42 | cpu | 1 | Ad-hoc; not a paper table |
| `paper_tables.tex` | mixed | mps live eval | dummy | Export yes; **inputs not paper-scale** |

No `outputs/*.csv` besides `ablation_results.csv` and `baseline_comparison.csv`.

---

## How to get a clean chain

`scripts/reproduce_all.sh` writes a fresh tree (default `outputs_repro/`) and refuses to overwrite `outputs/`. It records `git_commit`, dirty-file count, torch version, and the resolved device in `PROVENANCE.json`.

1. Commit the current code (or tag) **before** claiming numbers.
2. `SCALE=workshop DEVICE=auto OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh`
   - Prompt 1 dump uses `--workers 1` (the audited dump used auto workers=7; per-episode RNG is `seed + 10007 * episode_id`).
   - Prompt 2 trains VICReg then MSE then Pixel-AE from `configs/workshop_vicreg.yaml`.
   - Prompt 4 baselines stay on the mass-spring lattice (CPU), not the physics dump.
   - Prompt 5 workshop 3-seed × 4-variant grid is **skipped**. Sept 9 uses `scripts/finish_sept9.sh` (seed-42 flagship `train_and_eval`, then seed-43 VICReg+MSE only).
   - Prompt 3 (`scripts/diagnose_probes.py --physics-only`) then paper export then Prompt 6 figures from **final** checkpoints.
3. Cite only files under that `outputs_repro/` together with `PROVENANCE.json`.
4. Do not put mean±std ablation tables in the paper. The workshop table is a 2-seed range from `outputs/ablation_twoseed.json`.

A smoke-scale `outputs_repro/` **will not** `diff` equal to the audited `outputs/` tree: the original mix is 5-epoch CPU smoke + 1-epoch MPS dummy ablations + 8-epoch CPU baselines + an in-progress 200-epoch MPS VICReg job. The smoke recipe trains 2 epochs on a 48-episode dump (`--workers 1`) so it is a pipeline check, not a number match.

```bash
SCALE=smoke DEVICE=cpu OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
diff -rq outputs outputs_repro   # expected to differ; see flags above
```
