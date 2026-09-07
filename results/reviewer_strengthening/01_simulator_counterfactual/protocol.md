# Simulator counterfactual protocol (predeclared)

This document is frozen **before** looking at model sensitivity trends.

## Physical state

`MLSMPM2D` particle state is `(x, v, C, F, μ, λ)`.
Grid mass/velocity is rebuilt every `substep` from particles and is **not** persistent.
`substep` uses no RNG. Rest-shape jitter and action sampling use `numpy.Generator(seed=dump_seed + 10007 * episode_index)`.

Dump seed = 0. Validation episodes = `[1800, 2000)` (episode split, `val_fraction=0.1`).

## Fork

Replay episode `e` with `replay_episode` (unit-tested equal to stored `observations.npy` / `actions.npy`).
Snapshot at observation index `t`. Clone. Apply interventions. Original snapshot is never mutated by the branch.

## Interventions (simulator-space actions, layout `cx, cy, fx, fy`)

Primary (paper-facing):

1. **TRUE** — recorded `actions[e, t]`
2. **SHUFFLED-SAME-TIME** — `actions[val_perm_t[i], t]` with `perm_t = Generator(0 + 1009*t).permutation(n_val)`, deranged if `perm[i]==i`
3. **NEUTRAL** — training-split raw action mean (z-scored zeros / `dataset.action_mean`)

Secondary (diagnostic / more OOD; **not** pooled with primary):

4. **FORCE-REVERSED** — `(cx, cy, -fx, -fy)`
5. **FORCE-ROTATED** — 90° CCW: `(cx, cy, -fy, fx)`
6. **FORCE-SCALED** — magnitude ×1.5: `(cx, cy, 1.5 fx, 1.5 fy)`
7. **CONTACT-PERTURBED** — `cx' = clip(cx + contact_radius, 0.08, 0.92)`, force unchanged

## Horizons and hold

Observation step = `substeps=8` MPM substeps.
Horizons `h ∈ {1, 3, 5}` frames, requiring `t+h < T`.

After the intervention frame, **hold the alternate command** for the rest of the control block
`hold - (t % hold)` frames (`hold=3`), then resume the **recorded** command sequence.

## Command-change set

`t % 3 == 0` and `t > 0` → `{3, 6, 9, 12}` (and `t=0` reported separately).

## Effective contact (mechanics, not tuned on outcomes)

Actuator force is a **normalized** Gaussian, so total impulse is applied even for distant contacts.
`effective_contact := min_i ||x_i - contact|| ≤ 2 * contact_radius` (`contact_radius=0.10`).

## Metrics

Particle: RMS position Δ, mean per-particle L2, max L2, RMS velocity Δ, Δ kinetic energy (`0.5 m ||v||²`), COM Δ.
Raster: density MSE (ch0), velocity MSE (ch1–2), full MSE on `[0,1]` float rasters.
Latent: target-encoder MSE/cosine on true vs alt futures; predictor MSE/cosine of `g(z_t, a_true)` vs `g(z_t, a_alt)` with **normalized** actions.

Ratio: `MSE(pred_true, pred_alt) / MSE(z_tgt_true, z_tgt_alt)` when denominator `> 1e-8`.
Count exclusions. Do not hide near-zero denominators.

## Checkpoints

Final-epoch `last.pt` only:

- `outputs/vicreg_nocrop/last.pt` (seed 42, action-conditioned)
- `outputs/vicreg_nocrop_seed43/last.pt` (seed 43, action-conditioned)

Do not overwrite these files.
