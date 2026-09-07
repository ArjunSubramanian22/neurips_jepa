# Baseline audit

**BASELINE AUDIT: PASS**

## Tests
- ran: 83
- passed: 83
- failed: 0
- skipped: 0

## Data
- observations shape: [2000, 16, 3, 64, 64]
- actions shape: [2000, 15, 4]
- split: 1800/200
- observations sha256: `ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f`
- claimed sanity-check sha256 match: True

## Model
- trainable params excluding EMA: 874368
- total params including EMA: 1437952

## Headline source
Read from `outputs/corrected_seed43_audit.json` (already executed last.pt eval).
This audit does **not** re-run 200-epoch training or overwrite those JSON files.

## Discrepancies
[]

## Protocol notes
- Corrected paper-facing numbers use **last.pt** (epoch 200).
- `outputs/ablation_twoseed.json` uses **best.pt** and must not be mixed into corrected tables.
- Probe accuracies differ slightly across eval reruns (`mpm_action_ablation.json` vs audit JSON).
- VN-MSE is not implemented.
- Legacy MSE@10 brief values (~0.090 / 0.046) are **not** reproduced by `ablation_twoseed.json` best.pt (0.114 / 0.184). Document, do not silently retarget.

## Environment
See `environment.json` (no hostname/username/remote).
