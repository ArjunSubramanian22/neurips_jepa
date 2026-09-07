# Repository map (Phase 0)

Internal audit only. Paths are repo-relative. No remotes or usernames.

## Simulator
- `scripts/generate_trajectories.py` — MLS-MPM (`MLSMPM2D`), dump generation, snapshot/clone/restore, replay
- dump: `outputs/physics_mpm/` — observations/actions/labels/metadata (dump seed 0)

## Dataset / split / augs
- `src/dataset.py` — `DeformablePhysicsDataset`, episode split, train-only action z-score, `warp_contact_xy`, `crop_scale_min>=1` skips crop
- `configs/workshop_vicreg.yaml` — canonical no-crop (`crop_scale_min: 1.0`)

## Model / objective / train
- `src/models/ac_jepa.py` — ViT encoder, EMA target, concat action predictor
- `src/losses.py` — VICReg vs `mse` (invariance only)
- `src/train.py` — AdamW, cosine+warmup, clip 1.0, `--no-action`, last.pt policy
- `src/eval.py` — rollout MSE, effective rank, linear probes (no VN-MSE)

## Existing eval artifacts (do not overwrite)
- `outputs/corrected_seed43_audit.json` — matched last.pt 4-way + shuffle
- `outputs/mpm_action_ablation.json` — seed-42 cropped / nocrop / no-action last.pt
- `outputs/ablation_twoseed.json` — **best.pt** legacy two-seed (not last.pt)

## Tests
- `tests/test_*.py` including `test_counterfactual_fork.py`, `test_action_path.py`

## Checkpoints (read-only)
- `outputs/vicreg_nocrop/last.pt` seed 42 action VICReg no-crop
- `outputs/vicreg_nocrop_seed43/last.pt` seed 43
- `outputs/no_action/last.pt` seed 42 no-action
- `outputs/vicreg_nocrop_noaction_seed43/last.pt` seed 43 no-action
- `outputs/vicreg/last.pt` seed 42 cropped VICReg (legacy)
- `outputs/mse/last.pt` seed 42 cropped MSE (legacy)
