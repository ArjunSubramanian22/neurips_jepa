The canonical observation dump `observations.npy` (~375 MB) is omitted from this
artifact. SHA-256:

  ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f

Shipped here: actions.npy, stiffness_labels.npy, stiffness_values.npy,
geometry_labels.npy, metadata.json.

To regenerate a dump locally (not claimed bitwise-identical to the omitted file
on every machine):

  python scripts/generate_trajectories.py --n-episodes 2000 --seed 0 --output outputs/physics_mpm

`scripts/evaluate_final.py` requires the original observations.npy to recompute
held-out metrics. Cached metrics are in results/final_checkpoint_results.csv.
