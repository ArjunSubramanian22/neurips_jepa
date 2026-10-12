# Engineering and scientific-consistency audit

This note records what the repository can support after the checks below. Completed measurements are the only basis for a finding. Code that has not been trained or scored is listed as incomplete.

Audit date: 2026-10-12. Test command: `MPLCONFIGDIR=outputs/.mplconfig python3.13 -m unittest discover -s tests -v`. Result: 358 tests, OK. Smoke: `integration_smoke_v1`, status COMPLETE, git commit `5a99f94f9f52efeda3a5494a4f1376c0903fe528`, git dirty false. That smoke is a reduced-resolution protocol check. It did not load a learned checkpoint.

## Checklist

1. **Audit layers.** The MLS-MPM cascade stores physical, observation, representation, predictor, and decision separately (`src/metrics/cascade.py`). MuJoCo forks use the same cascade functions through `src/envs/fork_audit.py` and the benchmark schema (`physical_response`, `observation_visibility`, `target_response`, `predictor_response`, direction, ranking, multi-step, planning). `target_response` is the privileged tracked state. A learned encoder enters the predictor and direction records when a checkpoint is attached. New benchmark writes set the layer status from the learned record. An oracle cosine of 1 no longer marks an unmeasured predictor as measured. Files written before commit `75ce2a3` were not rewritten.

2. **Common adapter.** `benchmark.adapters` loads `jepa` (VICReg, action on), `mse` (MSE, action on), `noaction` (VICReg, action off), `rssm`, and `latent_tf`, and refuses a checkpoint whose class, loss, or action switch does not match the name. `obs_pred` is refused: it predicts pixels, and the latent fork adapter does not score it.

3. **Headline splits.** `scripts/evaluate_final.py` scores legacy validation episodes 1800–1999 and records `is_test_set: false`. `assert_checkpoint_eval_pairing` refuses a legacy checkpoint under `cvpr_v1`. The arm and the box pusher have no declared test split; `python -m benchmark.audit --split test` is rejected for them. The arm dataset tail is `arm_tail_v1` (validation), not `legacy_v1` and not a test set.

4. **Seeds.** `configs/seeds/headline_v1.yaml` requires seeds 42–46. Completed: VICReg action and no-action at 42–45; MSE action at 42–43. Missing, and still listed: VICReg action 46, VICReg no-action 46, MSE action 44, 45, and 46. RSSM, latent-transformer, observation-predictor, and width-64/256 MPM runs have no checkpoints in their seed files. Their evaluation status files say `trained: false`.

5. **Dropped failures.** Report tables keep NaN and undefined cosines. `future_ranking_v1` stays FAILED in `experiments/registry.jsonl`. Unrun inverse, alignment, and target-mask rows stay in the loss table with status `NOT_RUN`. The ablation ledger lists BLOCKED jobs instead of omitting them.

6. **Figures.** Qualitative panels are drawn by `scripts/render_qualitative_figures.py` from saved queries, caches, and the planning archive. The figure JSON files record the source episode, time, seed, checkpoint, and intervention.

7. **Regeneration.** Each qualitative JSON has `regeneration_command`. The results note has `Regenerate: python3.13 scripts/generate_scientific_report.py`.

8. **Provenance.** Registry records store git commit, config, split, seed, checkpoint, and environment. New benchmark writes add git commit, dirty flag, and config path. Older benchmark JSON files have environment, checkpoint, split, and seed, and do not contain a git commit. They were not backfilled. Some earlier registry rows were saved with `git_dirty: true` (`planning_oracle_sanity_v1`, `reproducibility_baseline_v1`, `headline_seed_uncertainty_v1`).

9. **Legacy reproduction.** `heldout_reproduction_v1` compared the ten scheduled no-crop `last.pt` checkpoints with the archived legacy-validation table. `n_scheduled` 10, `n_missing` 0, absolute tolerance 0.001, `success_criterion_met` true. Those episodes are legacy validation.

10. **CVPR versus legacy.** Completed headline tables use `legacy_v1` and say they are not a test set. `cvpr_v1` is the split named on the unrun inverse, alignment, target-mask, and action-conditioning configs. No completed training run has been scored as a `cvpr_v1` test result.

11. **Search.** No `TODO`, `FIXME`, or `HACK` in the Python, YAML, or Markdown sources. The latent adapter now rejects a mismatched loss or action switch. Visibility thresholds refuse test episodes and the legacy reporting tail. Seed 46 remains in the registry when it is missing. No seed is dropped to improve a mean.

## Completed capabilities

- Deterministic MLS-MPM clone, restore, and raster rendering, covered by `tests/scientific`.
- Legacy validation evaluator `scripts/evaluate_final.py`, reproduced for the ten scheduled checkpoints inside tolerance 0.001.
- Layer-separated counterfactual metrics, ranking, horizon audit, and visibility threshold fit on validation only.
- Planning benchmark on the historical MPM dump for the ten no-crop checkpoints, plus analytic CEM and oracle checks.
- MuJoCo box push (physical and image forks; no learned checkpoint) and MuJoCo arm (dataset validation, then JEPA, RSSM, and latent transformer, seed 0, 20 epochs, 24 training episodes).
- Shared benchmark audit and plan commands for the arm and the box pusher.
- Scientific report, ablation ledger, local queue, qualitative figure script, and reproduction packaging.
- Scientific regression suite and `.github/workflows/scientific_regression.yml`.

## Incomplete experiments

Headline seeds still null in `configs/seeds/headline_v1.yaml`:

- `vicreg_action` seed 46
- `vicreg_noaction` seed 46
- `mse_action` seeds 44, 45, 46

Not trained on the MPM dump, status `not_measured` or `NOT_RUN`:

- FiLM, gate, and cross-attention conditioning
- Inverse-dynamics weights 0, 0.01, 0.1, 1, and the probe
- Action-alignment direction, magnitude, and combined losses
- Target-mask variants
- RSSM, latent transformer, and observation U-Net at the headline width and seeds 42–46
- Capacity widths 64 and 256 (width 128 is the existing VICReg action run)
- Medium-excitation, high-excitation, and resolution 128/256 training jobs

The box-push environment has no trained checkpoint. Its audit file is the pre-model fork record.

The arm `validation.json` field `learned_model: not_measured` is the pre-training check. Later numbers are in `experiments/runs/mujoco_arm_v1/benchmark_*.json`. That validation file was not overwritten.

## Failed experiments

`future_ranking_v1` is FAILED. Reason: `NameError: name 'STATUS_UNDEFINED' is not defined`. The record was kept. `future_ranking_v2` completed and is the ranking result to use.

The ablation ledger has 0 FAILED jobs, 12 NOT_RUN, and 7 BLOCKED.

## Known bugs and remaining gaps

- Benchmark JSON written before `75ce2a3` still stores the old parent layer status. New writes use the learned status. The stored arm files were not regenerated.
- Those same benchmark files do not record a git commit. New benchmark writes do.
- `experiments/runs/mujoco_box_push_v1/audit.json` is not in the benchmark schema, so `reports/latest` does not ingest it. The file itself keeps learned metrics as `not_measured`.
- The observation U-Net is outside the latent adapter on purpose.
- MLS-MPM forks are run through the cascade scripts, not `python -m benchmark.audit`.
- A headline mean over seeds 42–46 is not available until the missing seeds exist. The seed summary already withholds that interval.

## Compute-heavy jobs still pending

The ablation ledger estimates 49642.6 seconds remaining on the measured JEPA scale, and 81910.3 seconds if the unmeasured multipliers are used. Those are planning estimates from `experiments/ablations/v1/ledger.jsonl`, not new timings.

Dry-run the queue before launching:

```bash
python3.13 -m experiments.queue --config configs/experiments/queue_example.yaml --dry-run
python3.13 scripts/plan_ablations.py
```

NOT_RUN commands from the ledger:

```bash
bash scripts/train_reviewer_seed.sh 46 vicreg action results/ablations/v1/runs headline_jepa_s46
bash scripts/train_reviewer_seed.sh 46 vicreg noaction results/ablations/v1/runs headline_noaction_s46
bash scripts/train_reviewer_seed.sh 44 mse action results/ablations/v1/runs headline_mse_s44
bash scripts/train_reviewer_seed.sh 45 mse action results/ablations/v1/runs headline_mse_s45
bash scripts/train_reviewer_seed.sh 46 mse action results/ablations/v1/runs headline_mse_s46
python3.13 -m src.train --mode train --config results/ablations/v1/configs/conditioning_film_s42.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name conditioning_film_s42 --num-workers 0
python3.13 -m src.train --mode train --config results/ablations/v1/configs/conditioning_gate_s42.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name conditioning_gate_s42 --num-workers 0
python3.13 -m src.train --mode train --config results/ablations/v1/configs/conditioning_cross_attn_s42.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name conditioning_cross_attn_s42 --num-workers 0
python3.13 -m src.train --mode train --config results/ablations/v1/configs/loss_inverse_s42.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name loss_inverse_s42 --num-workers 0
python3.13 -m src.train --mode train --config configs/experiments/rssm_v1.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name model_rssm_s42 --num-workers 0
python3.13 -m src.train --mode train --config configs/experiments/latent_transformer_v1.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name model_transformer_s42 --num-workers 0
python3.13 -m src.train --mode train --config configs/experiments/obs_pred_v1.yaml --seed 42 --output-dir results/ablations/v1/runs --run-name model_pixel_s42 --num-workers 0
```

BLOCKED, and not launched:

- Direction, magnitude, and combined alignment losses: `outputs/counterfactual_pairs/force_reversal_train.npz` is not on disk.
- Medium and high excitation training: no trajectory dump is registered for those regimes.
- Resolution 128 and 256: RGB training needs particle positions, which the dump does not store.

Do not start `train_reviewer_seed.sh` unless those seeds are meant to be trained. Each is about the measured VICReg wall clock (roughly 2.5–3 hours per seed on the machine that produced seeds 42–45).

## Smoke experiment

`python3.13 scripts/run_cascade_protocol_smoke.py --experiment-id integration_smoke_v1`

Record: `experiments/runs/integration_smoke_v1/record.json`. Status COMPLETE. Learned model evaluated: false. Three forks on a reduced disk (`n_grid` 16, image 32, substeps 4). The identical action left position RMS at 0 and the raster fraction identical at 1. Learned representation, predictor, and decision are `not_measured`. The oracle cosine on the reversed and zero-force forks is 1 with status `ok`. The identical fork cosine is undefined because the difference is zero. This is not an AC-JEPA result and not a `legacy_v1` or `cvpr_v1` number. The original `cascade_protocol_smoke_v1` record was left in place.

## Findings supported by completed experiments

- On the historical MPM dump, legacy validation episodes 1800–1999, the archived rollout table for the ten no-crop checkpoints is reproduced within absolute tolerance 0.001 (`heldout_reproduction_v1`).
- Open-loop counterfactual retrieval and cosine on that same legacy validation slice stay near chance, including on high-visibility and persistent-reversal slices. Teacher-forced scores look strong because the predictor reads the true next frame. Those teacher-forced scores are not evidence of action conditioning.
- For the ten checkpoints, MPC success on the planning report seeds is 0. Success therefore has no correlation with any audit metric. The magnitude-response Spearman against final goal distance is positive (about 0.83, n=10) and its bootstrap interval excludes 0. That sign means a larger response goes with a larger distance. Dropping the no-action family moves that Spearman to about 0.26. Latent-MSE correlations with final distance have intervals that include 0. The source headline is in `analysis/metric_planning_headline.json`. This report does not add a significance test.
- On the arm, seed 0, 24 training episodes, the fork files show the RSSM push cosine near 0.96 with retrieval 1, and the JEPA push cosine near 0 with retrieval 0. That is one seed and one small dataset. It does not establish that RSSM fixes action under-use on the MPM dump. The MPM RSSM was not trained.
- Identical actions on the reduced smoke simulator match. A changed force can move particles. Many MPM forks remain physically distinct while the raster is identical; those rows are ambiguity, and they are counted rather than dropped.

## Findings that are not supported

- A `cvpr_v1` test result. No completed headline checkpoint was trained and scored under that split.
- A five-seed uncertainty interval for the headline models. Seed 46 is missing, and MSE is missing three seeds.
- That FiLM, gating, cross-attention, inverse dynamics, alignment losses, target masks, or a wider or narrower JEPA improve planning or counterfactual direction. Those runs were not trained.
- That the observation U-Net, the MPM RSSM, or the MPM latent transformer uses actions. Their MPM evaluation status is `trained: false`.
- That a learned planner beats the oracle or the heuristic on the historical dump. Learned success on the report seeds is 0. The heuristic is the controller that succeeds. The oracle is a partial positive control.
- That the arm RSSM result transfers to the paper dump or to a held-out test set.
- That this repository is a finished CVPR submission. The code paths exist. The missing seeds, blocked dumps, and untrained ablations are still open.
