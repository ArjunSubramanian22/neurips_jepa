# Reproducibility matrix (Phase 0 freeze)

Canonical sources, in this order:

1. **Checkpoints and original dump** — this git workspace (`outputs/`, `results/reviewer_strengthening/05_more_seeds/`, `06_corrected_objective/`, `outputs/physics_mpm/`).
2. **Revised manuscript numbers** — `/Users/Arjun/Downloads/Counterfactual_Action_Evaluation__Observation_Bottlenecks__and_Representation_Geometry_in_Joint_Embedding_Predictive_World_Models__1_.zip` (`data/final_run_summary.csv`, `data/heldout_evaluation/`, `main.tex`).
3. **Revised PDF** — `PhysWorldAI_Revised__2_.pdf` (attachment).
4. **Original fork physics** — `results/reviewer_strengthening/01_simulator_counterfactual/` in this workspace (byte-identical to the zip copies used by the paper).

Do **not** treat `paper/main.tex` in this git repo as the submission. It is the older “Action-Conditioned Latent World Models” draft.

Do **not** treat `src/eval.py:evaluate_model` as the paper evaluator. It still averages rollout MSE over all 2000 episodes and does not implement VN-MSE. The paper’s Table 3 comes from `scripts/evaluate_completed_runs.py` inside the revised zip.

| Paper object | Claimed method | Generating artifact | Status vs this repo |
| --- | --- | --- | --- |
| Table 3 / Fig 4–5 rollout MSE | val episodes 1800–1999; last.pt; no-crop | zip `data/final_run_summary.csv` | Numbers exist; **canonical script not in git** |
| VN-MSE | \(E_k / (d^{-1}\mathrm{tr}\,Cov_e[z^-_{e,k}])\) | zip evaluator `vn=err.mean(0)/var` | Implemented in zip; **absent from `src/eval.py`** |
| Effective rank / top-1 | pooled horizons 1–10 | zip `effective_rank(pv)` flatten `(200,10,128)` | Matches Issue 10 concern |
| Target rank | same pooling | zip `effective_rank(tv)`; in CSV not Table 3 | Issue 8 |
| Table 4 probes | linear probe 80 steps; train 0–1799 / val 1800–1999 | zip evaluator | Geometry/stiffness/contact present |
| Fig 2 / Table 5 physical CF | original dump forks | `results/reviewer_strengthening/01_simulator_counterfactual/` | In git |
| Fig 3 action quintiles | original dump + s42/s43 last.pt | same folder + `02_action_distance/` | In git |
| Appendix C replay hash | Linux recorded-action replay | **observations not in this workspace** | SHA mismatch disclosed; local 20-ep RNG replay is bitwise identical to dump |
| Appendix F stiffness | 16×16 raster + EMA | zip `data/observability_replay/` | Particle/float raster **not run** |
| Checklist Q5 | No | appendix justification | Correct until anonymous large-artifact host exists |

## Reviewer one-liner (not yet true of this git tree)

```
pip install -r requirements.txt
python scripts/download_data.py   # does not exist
python scripts/reproduce_paper.py # does not exist as a single entrypoint matching Table 3
```

Existing: `reproduce_paper_results.py` and `REPRODUCIBILITY.md` cover the **original fork** package, not the ten-checkpoint held-out evaluator.

## Checkpoint freeze (verified 2026-09-07 local)

All ten paper checkpoints are epoch 200 and `crop_scale_min=1.0`. SHA-256 values match `data/training_registry.json` in the revised zip. See `CHECKPOINT_MANIFEST.csv`.
