# Data splits

Two episode-level protocols exist. A trajectory in this dump is one episode, so disjoint episode ids are disjoint trajectories. Transition-level random splits are not used.

## legacy_v1

`legacy_v1` is the original loader split. `hist_v1_last_fraction_val` is the same set of ids.

| Role | Episodes | Count |
| --- | --- | --- |
| train | `[0, 1800)` | 1800 |
| validation | `[1800, 2000)` | 200 |
| test | none | 0 |

Normalization for this protocol uses episodes `[0, 1800)` only. Checkpoint selection used validation loss on `[1800, 2000)`. The revised-paper tables also report that same tail. Those numbers are **legacy validation**. They are not `cvpr_v1` test numbers, and this file does not relabel them.

`scripts/evaluate_final.py` and `configs/workshop_vicreg.yaml` stay on `legacy_v1`. `configs/experiments/heldout_reproduction_v1.yaml` compares against the archive under `headline_role: legacy_validation` and `is_test_set: false`.

## cvpr_v1

Future training and headline reporting on the canonical 2000-episode dump (seed 0) use `cvpr_v1`.

The legacy validation episodes `[1800, 2000)` were already used for selection and for published metrics. Putting them in a new test set would keep tuning and final reporting on the same episodes. They are therefore excluded from `cvpr_v1` train, validation, and test.

The remaining episodes `[0, 1800)` are 300 complete stiffness-class by geometry cycles of length 6 (`episode % 3` and `(episode // 3) % 2`). An exact 80/10/10 split of those 1800 episodes is 1440 / 180 / 180, and each count is a multiple of 6. An 80/10/10 split of all 2000 episodes is not a multiple of that cycle (10% of 2000 is 200, and `200 % 6 = 2`).

Assignment, fixed before any `cvpr_v1` metric is read:

- block = `episode // 6` for `episode` in `[0, 1800)`
- validation when `block % 10 == 8`
- test when `block % 10 == 9`
- train otherwise

| Role | Count | Fraction of the 1800 eligible episodes | Fraction of the full dump |
| --- | --- | --- | --- |
| train | 1440 | 0.80 | 0.72 |
| validation | 180 | 0.10 | 0.09 |
| test | 180 | 0.10 | 0.09 |
| excluded (legacy validation) | 200 | — | 0.10 |

Exact ids are stored in `configs/protocols/cvpr_v1_episode_ids.json`. `cvpr_v1_split()` rebuilds those tuples. The rule does not look at losses, visibility, or any other outcome.

Within each of train, validation, and test, stiffness class counts are equal and geometry counts are equal.

### What each role is allowed to do

- **Normalization.** Action mean and population standard deviation are computed from the 1440 training episodes only. Validation, test, and excluded episodes do not enter the moments. `src.eval_protocol.train_action_statistics` and `DeformablePhysicsDataset` with `split_version="cvpr_v1"` both do this.
- **Model selection.** Validation episodes only.
- **Headline metrics.** Test episodes only.
- **Visibility thresholds and other hyperparameters.** Fit on train or validation. `fit_visibility_threshold` raises if a test id or an excluded legacy-reporting id is present, and it raises if the requested role is the test set.

`DeformablePhysicsDataset(..., split_version="cvpr_v1")` accepts `split` of `train`, `val`, or `test`. `split="all"` raises, so the test episodes are not silently appended to validation.

### Historical checkpoints

Checkpoints trained under `legacy_v1` saw every episode in `[0, 1800)`, including the `cvpr_v1` test ids. `assert_checkpoint_eval_pairing` rejects scoring them under `cvpr_v1`. Compare them only under `legacy_v1`, and keep the legacy-validation label. A model trained under `cvpr_v1` is scored under `cvpr_v1`, not under the old tail.

## Other configs

`configs/experiments/cascade_protocol_smoke.yaml` and `configs/experiments/planning_oracle_sanity.yaml` set `split_version: not_applicable`. They do not read the canonical dump. Their results are neither legacy validation nor `cvpr_v1` test numbers.

`fresh_pool_v2` remains a contiguous 80/10/10 rule for a newly generated pool whose length is a multiple of 60. It is not the study split for the existing 2000 episodes. `fresh_pool_v2_split(2000)` raises.
