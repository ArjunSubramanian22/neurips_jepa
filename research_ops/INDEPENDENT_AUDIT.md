# Independent audit

15 October 2026. Source code and saved rows were checked directly. Project summaries were not treated as evidence. No training was started, the held-out MPM test was not rescored, and historical CSVs were not edited. Focused reruns wrote only under `/tmp/jepa_audit`.

Verdict: **needs bounded repairs**. The measured MPM open-loop pattern, the physical-index repair, and the matched-planning arithmetic hold. Several sentences are broader than the procedure that produced them. The arm result is a six-episode, one-seed pilot. This is not a certification of the paper or of selection odds.

## Verified findings, by severity

### 1. The matched RSSM score is not the target RSSM was trained to match

`scripts/run_matched_planning.py` gives JEPA and RSSM the same final goal image and does not pass the goal trajectory or the generating actions. That part is real: all 1,920 saved rows have `goal_information=final_image`, and `_score_rssm_goal_image` never calls `observe` with an action (`tests/test_matched_planning.py`).

The RSSM target is then `CompactRSSM.encode`, which filters the goal image from the learned initial state with `action is None` (`src/models/rssm.py`, `encode`). The candidate side is `imagine` over the candidate actions. Training compares a prior and a posterior along the same action sequence (`CompactRSSM.loss`). This planning score compares those two different states.

The numerical drop is real for this score. On the saved table, mean top-1 for eight constant actions is 15.0% at H5 and 20.0% at H10, against 35.8% and 42.5% in `experiments/runs/mpm_high_planning/seed_summary.csv`. The old scorer, `scripts/run_mpm_high_planning.py` `_score_rssm`, still feeds `goal_actions` into `observe`. Re-executing episodes 1440 and 1441, seed 42, constant library, reproduced the saved miss at episode 1441: goal 1, chosen 5, rank 7, endpoint RMS `0.016404900699853897` at H5, and the same chosen index, rank, and RMS at H10.

Smallest correction: describe the result as candidate ranking under a final-image encoding from the initial state. Do not describe it as RSSM's training loss or as closed-loop control. A filtered goal-image target, with no future actions, would be a new measurement.

### 2. The v1 teacher-forced physical columns are still wrong, and the v2 repair matches the cache

For episode 1620, branch 0, `vicreg_action`, seed 42, teacher-forced, the cache file `outputs/counterfactual_cache/mpm_high_test/ep1620.npz` has position RMS 0.000755 at frame index 0, 0.002743 at index 1, 0.005785 at index 2, 0.014311 at index 4, and 0.041612 at index 9.

The v1 row stores H5 as the index-1 value and H10 as the index-2 value. The v2 row stores H5 as index 4 and H10 as index 9. H1 matches index 0 in both files. The cosine string is unchanged (`0.7397764526277933`, `0.9620636811750225`, `0.9206592142875448`).

Across the files, v1 and v2 have the same number of rows, no extra trailing rows, and no cosine, response-ratio, or prediction-norm mismatches. RMS or raster mismatches are exactly the teacher-forced non-RSSM H5/H10 rows: 32,400 of 129,600 in the test file and 54,000 of 194,400 in the validation/legacy file. Those v2 rows are marked `v2_recomputed`. Open-loop and RSSM rows are `v1_unchanged`.

`scripts/run_mpm_high_pretest.py` now uses `physical_step` (horizon minus 1) for positions and rasters and `latent_step` for the teacher-forced JEPA latent list. The saved test rows still carry metric version `mpm_multih_v1_rssm_z` because they were not rescored.

Smallest correction: cite `branch_rows_physindex_v2.csv` for teacher-forced physical or raster numbers. Leave the v1 files in place.

### 3. The open-loop direction split is in the rows, within every high-excitation seed

Recomputed from `branch_rows_physindex_v2.csv`, not from `headline.json`. For each seed: median over episodes of the median over six branch times, using only `cosine_status=ok`. The table value is the median of the five seeds.

Held-out test, episodes 1620–1799, open loop, 180 episodes, six branches, cosine defined on all 180 episodes for every action-conditioned seed:

| Model | H1 cosine (seed range) | H1 response ratio | H10 cosine (seed range) | H10 response ratio |
| --- | --- | ---: | --- | ---: |
| VICReg + action | 0.618 (0.585–0.636) | 0.914 | 0.308 (0.282–0.329) | 1.138 |
| MSE + action | 0.727 (0.720–0.747) | 0.807 | 0.810 (0.799–0.828) | 0.954 |
| RSSM stochastic mean | 0.924 (0.872–0.966) | 0.840 | 0.914 (0.873–0.963) | 0.711 |
| No action | undefined (0/180) | 0 | undefined (0/180) | 0 |

Each of the five VICReg seeds falls from H1 to H10. Each MSE seed rises. RSSM stays high. Validation (`mpm_high`, episodes 1440–1619) has the same pattern: VICReg 0.626 to 0.316, ratio 0.913 to 1.140. No-action cosine is undefined because `pair_metrics` withholds cosine at norm `1e-8`; the ratio is 0 because the predicted delta is at that floor and the target is not. There were no duplicate keys and no validation-file rows with regime `mpm_high` and episode ≥ 1620.

The RSSM number is the intact contrast. In `_rssm_sequences`, the target `observe` receives the true action and the true frame. It is not the blank-image or zero-action control.

Smallest correction: keep this table, and point the RSSM column at finding 4 so it is not read as image-only prediction.

### 4. RSSM target controls: two patterns hold on 180 validation episodes; the H10 sign does not

`experiments/runs/rssm_target_controls_val/rows.csv` has 1,800 rows per mode and horizon (180 episodes × 2 branch times × 5 seeds). Episodes are 1440–1619. For every cell, `swapped_action` cosine is the exact negation of `cross_fork_image` (maximum absolute sum 0). That is the algebra of swapping one input and subtracting the forks. It is not a second result.

H10 cross-fork seed medians are -0.158, -0.043, -0.048, +0.156, -0.118. Seed 45 is positive. A claim that every seed reverses sign is false on this file. Blank-image H1 stays high and H10 is lower for every seed; zero-action H1 falls to roughly 0.15–0.35 for every seed. Those two patterns match the code in `ablate`.

Smallest correction: quote the full-validation report and drop the four-episode sign sentence. Do not cite `swapped_action` as confirmation.

### 5. The arm numbers do not carry the MPM conclusion

`outputs/mujoco_arm_campaign` is 720 episodes, shape `(720, 16, 64, 64, 3)` and actions `(720, 15, 2)`, with metadata `n_train=480`, `n_val=120`, `n_test=120`. `DeformablePhysicsDataset._take_arm_campaign_v1` builds action moments from indices `[0, n_train)` and `split="all"` stops at `n_train+n_val`. The test suffix is 600–719.

The directional numbers are episodes 480–485, which sit inside validation, for one MSE seed. `best.pt` was chosen by validation loss on that same split (`outputs/mujoco_arm_campaign_runs/mse_s42/history.json`: best validation loss about `1.5e-4` near epoch 2; epoch 20 validation loss about 0.10). The six-episode cosine is 0.34/0.45 at `best.pt` and 0.85/0.56 at `last.pt`. The no-action one-epoch run has predicted-delta norm 0 and undefined cosine. That shows an action path can be knocked out. It does not show that the MPM seed pattern repeats.

Smallest correction: keep the arm file labeled as a pilot. Do not import its cosine into the MPM table. The sealed test stays unread until a validation freeze exists.

### 6. Legacy is a different, thinner design

Legacy open-loop VICReg uses four seeds, MSE two, and there is no RSSM. At H1, cosine is defined for 141 of 180 episodes, not 180. The median above drops the undefined episodes. Particle RMS at H10 is about `2.3e-4`, versus 0.062 on the high-excitation test. The low legacy cosine is a different excitation regime, not a failed replication of the high-excitation split.

Smallest correction: print the seed counts and the defined-episode counts next to any legacy number.

### 7. Subset reruns of matched planning change the goal

`goal_index` is `episodes.index(episode) % len(library)` in `run_matched_planning.py`. Episode 1441 is goal 1 in the saved list 1440–1463. A rerun with only 1441 assigns goal 0 and returns a different choice. With the original two-episode prefix, the saved 1441 row reproduced exactly.

The saved 24-episode file has 1,920 rows, no duplicate keys, episodes 1440–1463, seeds 42–46, and library episodes 0, 80, 160, 240, 320, 400, 480, 560, all below the validation start of 1440 (`src/splits.py`, `fresh_pool_v2_split`: for 1,800 episodes, train is `[0, 1440)`, validation `[1440, 1620)`, test `[1620, 1800)`). `physical_top1` is 1 on all 1,920 rows because the generating sequence is one of the candidates and its endpoint distance to itself is 0. That is a library identity, not a model result.

Checkpoint `outputs/mpm_high_runs/jepa_s42/last.pt` still has SHA-256 `1a485b9ff312…`, matching the hash stored on the seed-42 VICReg test rows. Git commit on those rows is `49ed22b7d990308270518f6387ac1d55feaad2ac`.

Smallest correction: set the goal from the episode id, or refuse a subset whose order does not preserve the published indices. The published means do not need to be recomputed.

### 8. The index tests catch the repair function, not every future wiring mistake

`tests/test_physical_index.py` builds a fake cache and checks that horizon 5 is rewritten from frame 1 to frame 4. `test_teacher_forced_physical_index_is_not_the_latent_slot` checks the two helpers. Neither test executes `score_checkpoint`. The current loop does call both helpers (lines 402–404). The freeze test patches the freeze path, expects `PROTOCOL_FREEZE`, and the on-disk lock bytes were unchanged after the test ran.

These six tests passed in this audit: the two physical-index tests, the two matched-planning tests, the helper test, and the freeze test.

Smallest correction: add a test that a teacher-forced JEPA row and an open-loop row read different frame indices from one fake bundle. Not required to correct the current loop.

## Claims

Supported, on the stated slice:

- High-excitation open-loop VICReg cosine falls from H1 to H10 on validation and on the held-out test, in every seed, while the response ratio stays near 1. MSE cosine does not fall. No-action cosine is undefined.
- Teacher-forced JEPA H5/H10 physical and raster values in the v1 CSVs are frames H2 and H3. The v2 files match the cache at frames H5 and H10. Latent columns were not changed.
- Matched final-image ranking, eight constant actions, 24 validation episodes, five seeds: VICReg top-1 91.7% / 75.8% and MSE 99.2% / 99.2%, the same as the older JEPA table. RSSM is 15.0% / 20.0% under the initial-state encoding, versus 35.8% / 42.5% when the old scorer also saw the goal trajectory and its actions.
- Sixteen time-varying sequences: VICReg 65.0% / 36.7%, MSE 82.5% / 87.5%, RSSM 5.8% / 7.5%, no-action 8.3% / 8.3%. Chance is 6.25%. This is one-shot ranking. Endpoint RMS is the particle error of the chosen sequence, not a receding-horizon controller.
- Fork snapshots copy particle state (`MLSMPM2D.snapshot` / `restore`). Recorded and reversed branches are separate arrays.
- Action moments for `fresh_pool_v2` and `arm_campaign_v1` are fit on the training block. The planning library is inside that training block.

Overstated if said without the qualifier:

- RSSM open-loop cosine shows that the stochastic state tracks the image rather than the action. The target filter receives the action. The blank-image control leaves H1 cosine high.
- RSSM cannot use its goal information for planning. The matched score uses a different target than training.
- The H10 cross-fork cosine is negative for every seed.
- `swapped_action` independently confirms the cross-fork result.
- The arm pilot shows the same dissociation as the MPM test.
- Legacy is a replication of the high-excitation result.
- Oracle physical top-1 of 1 is evidence about the models.

Unresolved:

- A matched RSSM target that filters the goal image without future actions.
- Any closed-loop execution on these checkpoints. `scripts/run_planning_benchmark.py` is a different CEM protocol and was not run.
- Whether the arm pattern survives five seeds and the sealed test. That test was not opened here.

## What was rerun

- Row-level comparison of v1 and v2 CSVs, cache RMS for episode 1620, duplicate-key and split checks, planning arithmetic, RSSM negation check, and the `jepa_s42` hash. Output: `/tmp/jepa_audit/audit_numbers.json`.
- Matched planning for episodes 1440–1441, seed 42, RSSM, constant library, written to `/tmp/jepa_audit/plan_spot_pair`. The 1441 row matched the saved file. The other 1,918 rows were not re-inferred.
- Six unit tests, listed above.

Not checked: every checkpoint hash, every line of training, pixel-level replay of all 720 arm episodes, or the scientific status of the work outside these measurements.
