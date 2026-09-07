SUPPORTED STRONG CLAIMS

- Dump geometry matches the brief: 2000 episodes, T=16, actions (2000,15,4), split 1800/200, observations SHA-256 `ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f`. Artifact: `results/reviewer_strengthening/00_baseline_audit/baseline_reproduction.json`.
- Unit tests pass (83/83), including exact MPM snapshot/clone/restore and dump replay equality. Artifact: `00_baseline_audit/baseline_audit.md`, `tests/test_counterfactual_fork.py`.
- The action pathway is wired: changing `a` changes the action-conditioned predictor; no-action models ignore `a`; gradients reach `action_embed`. Artifact: `09_action_path_audit/audit.md`.
- Same-state forks with alternate actions change particle **velocity** at h=1 (median RMS vel Δ vs shuffled = 0.002385589526966214). Artifact: `01_simulator_counterfactual/aggregate_metrics.json`.
- Replacing normalized actions at inference changes the predictor by ~1e-6 MSE (s42 median 1.9272439431006205e-06; s43 2.587943640719459e-06) on 3000 val transitions. Artifact: same, plus `predictor_divergence_per_example.csv`.
- Shuffled actions are not copies of the true action (Spearman of normalized action L2 vs physical RMS pos = 0.6840317911146434, n=3000). Artifact: `02_action_distance/correlations.csv`.

SUPPORTED BUT MUST BE QUALIFIED

- “The predictor is nearly invariant to supplied actions” — true at the scale of ~1e-6 latent MSE versus rollout MSE@1 ≈ 0.07, including command-change times. It is **not** bitwise invariance; gradients and weights are nonzero.
- “Geometry is linearly recoverable; stiffness is near chance” — true for the frozen linear probes in `outputs/corrected_seed43_audit.json` (geometry ≈ 0.9985–1.0; stiffness ≈ 0.30). Nonlinear/history/raw probes were **not** executed in this package.
- “The physical system responds to actions” — particle-state yes (especially velocity). **Raster observations often do not**: 41.533% of h=1 shuffled pairs have identical uint8 frames; median position Δ is 1.19e-5 vs pixel size ~0.016.
- “Target latents distinguish true vs counterfactual futures” — median target MSE is ~1e-6–3e-6, and 1256–1273 / 3000 pairs are excluded because tgt_mse ≤ 1e-8. Distinction is weak at h=1 after rasterization.
- Median capture ratio (pred/target, nonzero denom) is 0.01647 (s42) and 0.06699 (s43). Mean capture ratio bootstrap CIs are **not** usable as typical values.
- Corrected no-crop VICReg seeds 42/43 match the brief’s sanity-check rollout numbers to rounding. Predicted rank is seed-unstable (12.27 vs 5.78).
- Seed-42 “no-action matches/beats action on MSE@10” does **not** replicate on seed 43 (`outputs/corrected_seed43_audit.json`).

NOT ESTABLISHED BY THESE EXPERIMENTS

- Action neglect on four corrected training seeds (seed 44+ not finished at package time).
- Matched no-crop MSE-only vs VICReg (legacy comparison used crop; new trains not finished).
- VN-MSE (not implemented in `src/eval.py`).
- Stiffness unrecoverable even with MLP / history / raw rasters (Phase 7 not run).
- Incremental predictive information of action in a linear z_t → z_{t+1} probe (Phase 8 not run).
- Jacobian Frobenius norms (Phase 4 not run).
- Bitwise-deterministic full dump regeneration (only per-episode replay tested).
- Planning / control performance, real video, or PhysWorld SOTA.

CONTRADICTED BY CURRENT EVIDENCE

- “Shuffled actions miss the object, so of course nothing happens.” Predeclared effective-contact is true for 3000/3000 val transitions; shuffled actions still differ in L2 and move particle velocities.
- “A low capture-ratio mean CI proves the predictor copies the target’s action sensitivity.” The mean is denominator-unstable; medians are 0.016–0.067 among valid pairs, and many pairs have ~0 target change.
- Mixing `outputs/ablation_twoseed.json` (best.pt, cropped) with corrected last.pt no-crop headlines.
- Brief legacy MSE-only MSE@10 ≈ 0.090 / 0.046 as the best.pt two-seed file (0.114 / 0.184). Those are different checkpoints/protocols.
