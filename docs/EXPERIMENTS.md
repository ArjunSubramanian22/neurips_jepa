# Experiments

Each registered run writes `experiments/runs/<experiment_id>/record.json` and one line in `experiments/registry.jsonl`. A finished id is not overwritten. Measured numbers below are copied from those files. Regenerate them with the commands in `docs/EVALUATION.md` rather than editing the JSON.

Code commit for the three runs below: `949564e11baa7f787a6f2ea71a3ce3be9a8b50f0`. The cascade smoke started from a clean tree. The planning run and the baseline inventory started while `experiments/` was still untracked, so their records have `git_dirty: true`.

## cascade_protocol_smoke_v1

- **Status.** COMPLETE. Failure reason: none.
- **Research question.** Can one action fork store physical response, raster visibility, and the learned-model layers as separate fields, including an explicit `not_measured` when no checkpoint is loaded?
- **Hypothesis.** An identical action leaves the reduced MLS-MPM state and its uint8 raster unchanged. A different action has nonzero action L2. The size of the particle and raster changes is measured, not assumed.
- **Implementation.** `configs/experiments/cascade_protocol_smoke.yaml`, `src/cascade_protocol.py`. Seed 0, disk geometry, Young's modulus 70, two observation frames after one warmup frame. Interventions: `identical`, `force_reversed`, `zero_force`. Reduced simulator: `n_grid=16`, `img_size=32`, `substeps=4`, `damping=0.08`.
- **Controls.** Identical action. Constant predictor stand-in, which emits the factual particle embedding for every action. Oracle stand-in, which emits the privileged embedding of the rolled future. Synthetic checks call the same metric functions on vectors with known cosine.
- **Confounds.** One initial state. Reduced resolution, so the displacements are not comparable to the historical dump. Particle coordinates are simulator state, not a learned encoder. No AC-JEPA checkpoint is loaded. Two frames under the normalized actuator and damping.
- **Success criterion.** Identical action L2, RMS position, and RMS velocity are 0, and the identical raster fraction is 1. Synthetic aligned cosine is 1 and opposite cosine is -1.
- **Recorded outcome.** `experiments/runs/cascade_protocol_smoke_v1/pairs.csv` meets that criterion: the identical row is 0 for action L2, RMS position, RMS velocity, privileged MSE, and both stand-in MSEs, with fraction identical 1. Oracle cosine on that row is undefined because the difference vector is zero. Force reversal action L2 is 0.8694835462362778, RMS position is 3.264588857076061e-05, and the identical-uint8 fraction is 0.9482421875. The constant stand-in MSE stays 0. The oracle cosine on the two changed actions is 1. Learned representation, learned predictor, and decision are `not_measured`.
- **Permitted interpretation.** The metric plumbing distinguishes a changed action, a small particle displacement, and a raster that remains mostly identical, and it refuses to fill the learned-model layers. It is not a measurement of AC-JEPA.

Figure, from `pairs.json`:

```bash
python3.13 scripts/plot_cascade_layers.py
```

Output: `experiments/runs/cascade_protocol_smoke_v1/cascade_layers.png`.

## planning_oracle_sanity_v1

- **Status.** COMPLETE. `success_criterion_met` is true. Failure reason: none.
- **Research question.** Before any learned dynamics model is scored, does random shooting recover the unique action of a known linear residual model, does CEM reduce a known quadratic, and is random ranking near chance?
- **Hypothesis.** Identity dynamics `z + a` make the designated candidate the unique zero-cost action. CEM from a zero mean with unit scale reduces `||a - target||^2` for `target = (0.4, -0.2, 0.7, 0.1)`. Uniform random scores with four candidates land near top-1 chance 0.25.
- **Implementation.** `configs/experiments/planning_oracle_sanity.yaml`, `scripts/run_planning_oracle.py`. Seed 0. Eight candidates, latent dimension 4, correct index 3. CEM population 128, 8 iterations, elite fraction 0.25. Random ranking uses 4000 trials and 4 candidates.
- **Controls.** The correct action is chosen before the planner runs. The planner sees costs only. No world-model checkpoint and no episode split.
- **Confounds.** Identity residual dynamics are not the MLS-MPM simulator. CEM's final cost is the best sample, not a cost recomputed at the elite mean. `argmax` breaks score ties toward the first maximum; continuous scores make ties negligible.
- **Success criterion.** Oracle top-1 is 1 and the minimum is unique. CEM final cost is below the initial cost 0.7. Absolute error of the random top-1 rate against 0.25 is at most 0.03.
- **Recorded outcome.** `experiments/runs/planning_oracle_sanity_v1/metrics.json`: chosen index 3, unique minimum, top-1 1.0, chance 0.125. CEM final cost 0.00010014623116759674 from initial cost 0.7. Random ranking: 1001/4000 hits, top-1 0.25025, chance 0.25.
- **Permitted interpretation.** The planner code recovers a known linear action and does not treat random scores as better than chance on this draw. It is not evidence about a learned world model.

## reproducibility_baseline_v1

- **Status.** COMPLETE. Failure reason: none. This run loads no weights and computes no model metric.
- **Research question.** Which declared checkpoint and data files are on disk, and do their SHA-256 values match `CHECKPOINT_MANIFEST.csv` and `DATA_MANIFEST.csv`?
- **Hypothesis.** None. This is an inventory.
- **Implementation.** `scripts/audit_reproducibility_baseline.py` hashes a file only when it exists and the manifest records a hash.
- **Controls.** Absent paths stay `sha256_match: null`. The script does not copy a recorded hash into `measured_sha256`.
- **Confounds.** `EXPERIMENT_MANIFEST.csv` is an earlier inventory. Empty artifact cells are reported as missing even when a later file exists under another path.
- **Success criterion.** Every present checkpoint with a recorded hash matches. Missing canonical files are listed.
- **Recorded outcome.** `experiments/runs/reproducibility_baseline_v1/baseline.json`: 12 checkpoint rows, 12 present, 12 hashes matched, 0 mismatched. `outputs/physics_mpm/observations.npy` is absent. `linux_replay_observations.npy`, `particle_positions.npy`, and `particle_velocities.npy` are absent. Actions, stiffness, geometry, metadata, and the three counterfactual CSVs match their recorded hashes. `heldout_linux_replay_eval` and `appendix_F_probes` point at a Downloads zip that is not in this worktree.
- **Bookkeeping gap, not a new measurement.** `EXPERIMENT_MANIFEST.csv` still marks `stiffness_particle_state_probe` and `state_perturbation_control` as `not_run` with an empty path. The files `results/audit_phase2/stiffness_positive_controls.csv`, `results/audit_phase2/vn_state_perturbation.csv`, and `results/final_checkpoint_results.csv` are present. This baseline did not recompute those analyses.
- **Permitted interpretation.** The twelve paper checkpoints on disk match the manifest. The original observation tensor is not in the worktree, so `scripts/evaluate_final.py` cannot be rerun until that tensor is restored or regenerated and its hash is checked against `ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f`.

## Historical dump and revised-paper evaluations

These were not rerun in this pass. Their protocol and caveats are in `AUDIT.md`, `REPRODUCIBILITY_MATRIX.md`, and `configs/protocols/hist_v1_split.yaml`. Do not mix cropped `outputs/vicreg/last.pt` and `outputs/mse/last.pt` (`crop_scale_min=0.75`) with the no-crop `last.pt` checkpoints.
