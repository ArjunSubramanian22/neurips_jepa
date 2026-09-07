# Minimum reproduction bundle

## Without giant tensors
- `results/reviewer_strengthening/` analysis CSVs/JSON/figures
- `outputs/corrected_seed43_audit.json` and other metric JSON
- code + `configs/workshop_vicreg.yaml` + tests

## Requires checkpoints (~12 MB each last.pt)
- rollout / probes / shuffle / Jacobian / latent counterfactuals

## Requires observations.npy (~375 MB)
- any encoding of real dump frames, probes, identifiability, eval

## Dataset regeneration
`python scripts/generate_trajectories.py --n-episodes 2000 --seed 0`

Full-dump bitwise regeneration is **not** claimed. Per-episode replay vs stored dump is tested in `tests/test_counterfactual_fork.py`.
