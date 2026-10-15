# Physical index correction

The v1 scorer stored teacher-forced JEPA physical and raster measurements at the latent-list index. For horizons (1, 5, 10) that index is (0, 1, 2), so the columns labeled H5 and H10 were frames H2 and H3. Open-loop rows and RSSM rows already used frame `horizon - 1`. Latent columns were not affected.

`scripts/run_mpm_high_pretest.py` now uses `physical_step` for those columns and `latent_step` for the latent tensors. New scores use metric version `mpm_multih_v2_physindex`.

The original CSVs are unchanged:

- `experiments/runs/mpm_high_multih/branch_rows.csv`
- `experiments/runs/mpm_high_test_final/branch_rows.csv`
- `experiments/runs/mpm_high_bottleneck/branch_rows.csv`

Corrected siblings are `branch_rows_physindex_v2.csv` beside each file. They were recomputed from `outputs/counterfactual_cache` on 15 October 2026. Every open-loop row and every RSSM row was checked against the cache and left unchanged. Latent columns were copied.

| File | Rows | Recomputed | Checked unchanged | Max abs RMS position change |
| --- | ---: | ---: | ---: | ---: |
| validation multih | 194400 | 54000 | 140400 | 0.221 |
| held-out test | 129600 | 32400 | 97200 | 0.199 |
| bottleneck copy | 129600 | 32400 | 97200 | 0.221 |

Recomputed rows are teacher-forced, non-RSSM, horizon greater than 1. Their `physical_metric_version` is `v2_recomputed`. Bottleneck labels did not change: the wrong frame and the true frame fell on the same side of the predeclared thresholds. Do not read H5 or H10 teacher-forced physical or raster values from the v1 files.

The protocol freeze and `experiments/PRETEST_FINDINGS.md` stay as the historical record. This note does not reopen the test.
