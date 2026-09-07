# Statistical summary (Phase 1) — executed, not estimated

N validation episodes: 200 (indices 1800–1999)
N transitions: 3000
N command-change transitions (t in {3,6,9,12}): 800
N effective-contact transitions (predeclared `min_dist <= 2 * contact_radius`): **3000 / 3000**
Horizons: h ∈ {1,3,5}
Wall-clock: 876.73 s on CPU

Primary interventions are not pooled with OOD force/contact diagnostics.

## Physical response (TRUE vs SHUFFLED, h=1)

- median RMS particle **position** Δ: `1.1906067811651155e-05`
- command-change subset: `1.1807631835836219e-05`
- median RMS particle **velocity** Δ: `0.002385589526966214`
- median uint8 raster MSE: `8.885939450919977e-08`
- fraction **identical uint8 rasters**: `0.41533333333333333`

TRUE vs NEUTRAL h=1: median RMS pos `7.876719791966025e-06`; identical-raster fraction `0.5793333333333334`.

One 64×64 pixel is about 0.016 in simulator units. Median position Δ is ~0.00075 pixels. Velocity Δ is visible in particle state but often below one gray level after tanh + uint8 rendering (see debug: force reversal at h≤5 can leave rasters bitwise identical).

## Target-encoder future Δ (TRUE vs SHUFFLED, h=1)

- `vicreg_nocrop_s42` median MSE: `1.8069437714984815e-06`
- `vicreg_nocrop_s43` median MSE: `3.127198965557909e-06`
- near-zero denominator exclusions (`tgt_mse ≤ 1e-8`): 1273 / 3000 (s42), 1256 / 3000 (s43)

## Predictor Δ (same z_t, TRUE vs SHUFFLED normalized actions)

- s42 median MSE: `1.9272439431006205e-06`
- s43 median MSE: `2.587943640719459e-06`

Compare to rollout MSE@1 ≈ 0.07 on these checkpoints: predictor action-swap MSE is ~10^4 times smaller.

## Capture ratio = pred_mse / tgt_mse (exclude tgt_mse ≤ 1e-8)

- median s42: `0.016469133928909738`
- median s43: `0.06698928339760657`

Episode-mean of this ratio is **not** a stable summary: tiny denominators just above 1e-8 inflate the mean (bootstrap mean ≈ 5.3 / 3.1). Prefer the **median**. Do not interpret the mean CI as typical capture.

## Action distance (Phase 2)

Normalized action L2 vs physical RMS pos Spearman = `0.6840317911146434` (n=3000). Shuffled actions are not duplicates of the original.

Largest action-L2 quintile (n=600):
- s42 median pred MSE `5.172790224605706e-06` vs median target MSE `9.034944741870277e-05`
- s43 median pred MSE `7.0072603648441145e-06` vs median target MSE `3.975741674366873e-05`

Even the strongest in-distribution action swaps move the predictor by ~1e-6–1e-5 MSE while the target future, when it moves, moves more.

## Conclusion supported

- Exact-state forks are valid (Phase 1A tests).
- Particle **velocity** responds to alternate actions at h=1.
- Shuffled actions differ from true actions (contact/force geometry + Spearman).
- The action-conditioned predictor changes only ~1e-6 MSE when those actions are replaced, including at command-change times (see per-example CSVs).
- A large share of h=1 counterfactual **observations** are identical after uint8 rasterization, so the target encoder often cannot see the physical fork.

## Conclusion NOT supported

- That the physical system is action-insensitive (particle velocity RMS Δ ≈ 2.4e-3).
- That raster observations faithfully transmit one-step action effects (41.5% identical uint8 under shuffle).
- That the predeclared effective-contact flag isolates missed contacts (it is true for every val transition).
- That mean capture-ratio bootstrap CIs describe typical pairs.
- That action neglect is *only* a predictor failure vs *also* an observation/quantization/time-scale limit.

OOD force-reverse/rotate/scale/contact-shift are diagnostic and are not pooled into the primary medians.
