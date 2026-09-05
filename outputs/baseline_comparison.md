# Within-simulator baseline comparison

This table compares world-model families **inside the same mass-spring
lattice simulator** (the in-process `DataConfig.n_grid` particles and
`_edge_list` spring connectivity in `src/dataset.py`).

## What is being compared

| method | family | dynamics |
| --- | --- | --- |
| `ac_jepa_vicreg` | AC-JEPA | action-conditioned latent predictor + VICReg |
| `ac_jepa_no_cov` | AC-JEPA ablation | VICReg without covariance |
| `ac_jepa_mse` | AC-JEPA ablation | latent MSE only |
| `pixel_ae` | pixel autoencoder | decode `Predictor(z_t, a_t)` in image space |
| `no_action` | AC-JEPA ablation | predictor sees `z_t` only (`condition_on_action=False`) |
| `particle_gnn` | GNS / DPI-Net family | message passing on `_edge_list`, next-position MSE |

All rows use the **same held-out episodes**, the **same** `evaluate_model`
metrics (`rollout_mse`, `stiffness_probe_acc`, `contact_probe_acc`), and
horizons `k = 1 … 10`.

The particle GNN is scored in its pooled graph-readout space (not the ViT
latent). Magnitudes of `rollout_mse` are therefore **not** interchangeable
across methods; ranking within a method over horizon, and probe accuracies
across methods, are the fair columns.

## Domain gap (limitation)

Do **not** read this table as state-of-the-art vs [PhysWorld](https://physworld.github.io/)
or PhysTwin. Those systems:

* train / evaluate on **real captured video** of deformable objects, and
* couple a learned model to an **MPM-optimized digital twin**.

Our GNN sees privileged particle states from a synthetic Hookean lattice.
That is a different data regime. This file is a *within-simulator* sanity
check that AC-JEPA is compared against an established particle-graph
baseline family, not a claim of outperforming those papers on their
captures.
