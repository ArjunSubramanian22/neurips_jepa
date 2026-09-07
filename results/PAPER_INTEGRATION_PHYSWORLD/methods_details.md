# Methods details (facts only)

## Simulator
- MLS-MPM 2D NumPy, `scripts/generate_trajectories.py`, dump seed 0.
- State forked: particle `x, v, C, F, μ, λ`. Grid rebuilt each substep. `substep` has no RNG.
- `n_grid=32`, `dt=1e-3`, `substeps=8`, `action_gain=55`, `contact_radius=0.10`, `damping=0.08`.
- Actuator: Gaussian weights **normalized** so total impulse is applied even for distant contacts.
- Effective contact (predeclared): `min_i ||x_i-contact|| ≤ 2 * contact_radius`.

## Fork / intervention
- Replay episode e; snapshot at observation t; restore independent clones.
- Primary: TRUE recorded action; SHUFFLED-SAME-TIME deranged perm `Generator(0+1009*t)` over 200 val episodes at the same t; NEUTRAL = train-split raw action mean.
- Secondary (not pooled): force reverse; 90° CCW `(cx,cy,-fy,fx)`; force ×1.5; `cx += contact_radius` clipped to [0.08,0.92].
- After intervene, hold alt command for `hold - (t % 3)` frames (`hold=3`), then resume recorded commands.
- Horizons h ∈ {1,3,5} observation steps.

## Data
- `outputs/physics_mpm/observations.npy` SHA-256 `ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f`.
- Split: last 200 of 2000 episodes. Action z-score from train episodes only, `std` with `ddof=0`, clamp 1e-6.
- Canonical workshop crop: `crop_scale_min=1.0` (crop skipped).

## Training (existing corrected runs)
- ViT 8×8, width 128, depth 4, heads 4, concat action, EMA 0.996, VICReg 25/25/1, batch 128, 200 epochs, lr 2e-4, cosine, 20 warmup, wd 0.05, clip 1.0.
- Checkpoint policy: `last.pt` epoch 200. Do not use `best.pt` for corrected tables.
- Trainable params excluding EMA: 874368.

## Counterfactual metrics
- Particle RMS pos/vel, max pos, ΔKE, COM; uint8 raster MSE on [0,1] after /255.
- Target: EMA encoder MSE/cosine on rasterized true vs alt futures.
- Predictor: `g(z_t, a_true)` vs `g(z_t, a_alt)` with **normalized** actions.
- Capture ratio: pred_mse / tgt_mse if tgt_mse > 1e-8; exclusions counted.

## Statistics
- Bootstrap: 5000, seed 0, unit = validation episode, statistic = mean of per-episode means, percentile 95% CI.
- Spearman: rank correlation on n=3000 shuffled pairs.
- Quintiles of normalized action L2 defined before inspecting predictor trends.

## Probes (existing)
- Frozen linear probes in `src/eval.py`, 80 epochs, reported in `outputs/corrected_seed43_audit.json`.
- Phase 7 MLP/history/raw: not run at package time.

## Compute
- Python 3.13.2, torch 2.12.1, numpy 2.5.2, sklearn 1.9.0, matplotlib 3.11.1, Darwin arm64, 8 GB, MPS.
- CF experiment: 876.73 s CPU.
- Prior seed-42 no-crop train: 9046.35 s MPS (`outputs/vicreg_nocrop/history.json`).
