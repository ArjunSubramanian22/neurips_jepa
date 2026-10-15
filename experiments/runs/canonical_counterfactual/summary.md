# Canonical open-loop counterfactual summary

Aggregation: for each seed, the median across episodes of the median across six branch times. The reported value is the median of those seed estimates. Cosine uses only rows with `cosine_status=ok`. Undefined cosine is omitted, not replaced by zero.

Sources are the v2 physical-index files. Open-loop latent columns were copied from v1. Teacher-forced physical and raster medians now match the open-loop medians at every horizon, which is the check that H5 and H10 read frames 5 and 10.

Held-out episodes are 1620-1799. They were already scored. This file does not rescore them and does not select checkpoints.

## High-excitation held-out test, open loop, five seeds

| Model | H1 cosine | H1 ratio | H10 cosine | H10 ratio |
| --- | ---: | ---: | ---: | ---: |
| VICReg + action | 0.618 | 0.914 | 0.308 | 1.138 |
| MSE + action | 0.727 | 0.807 | 0.810 | 0.954 |
| RSSM stochastic mean | 0.924 | 0.840 | 0.914 | 0.711 |
| No action | undefined | 0 | undefined | 0 |

VICReg's response ratio stays near 1 while its directional cosine falls. MSE and RSSM cosine stay high. The same pattern is on the validation slice (VICReg 0.626 to 0.316). Legacy excitation, a separate regime with much smaller particle motion, does not show this split: VICReg cosine is 0.073 at H1 and 0.016 at H10, and the response ratio falls with it. Legacy uses four VICReg seeds and two MSE seeds, and it has no RSSM.

Physical position RMS at H10 is 0.062 on the test and 0.061 on validation. On legacy it is 0.00023.

Full seed rows: `seed_summary.json`. Headline table: `headline.json`.
