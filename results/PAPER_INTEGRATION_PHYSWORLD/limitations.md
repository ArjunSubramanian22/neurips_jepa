# Limitations (do not minimize)

- Synthetic 2D MLS-MPM rasters, T=16, two shapes, three modulus values.
- Action policy resamples every 3 steps near the object; persistence is structural.
- Normalized Gaussian contact never drops total impulse; “missed contact” is not in this actuator.
- Predeclared effective-contact threshold did not stratify the val set (3000/3000 true).
- One-step rasterization + tanh velocity + uint8 hides much of the particle response at h=1.
- Only two corrected action-conditioned training seeds (42, 43) finished; 44+ were in progress.
- No matched corrected MSE-only objective at package time.
- No VN-MSE implementation.
- Linear stiffness probes only; nonlinear/raw observability not run.
- No planning/control benchmark; no real video.
- Representation vs predictor vs observation-operator ambiguity is only partially separated: target medians are also tiny at h=1.
- Intervention distribution: shuffled-same-time preserves the marginal but is not an optimal control contrast.
- 8 GB unified memory: do not overlap two workshop ViT trains.
- Double-blind: do not upload `environment.json` git metadata if the venue objects.
