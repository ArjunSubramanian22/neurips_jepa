# Matched-information planning

Validation episodes 1440-1463. Five seeds. The goal for every model is the final image only. RSSM does not receive the goal trajectory or the actions that generated it. Physical oracle top-1 is 1 for every group: the generating action is identifiable from the endpoint particles. Chance is 12.5% for eight constant actions and 6.25% for sixteen sequences.

Numbers are the mean of the five seed means. Source: `seed_summary.csv`. This is not the held-out test.

## Constant actions, eight candidates

Same library shape as `experiments/runs/mpm_high_planning`. JEPA already used a final image there, so its top-1 is unchanged. RSSM previously saw the goal trajectory and the generating actions.

| Model | H5 top-1 | H5 endpoint RMS | H10 top-1 | H10 endpoint RMS | Old privileged top-1, H5 / H10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSE + action | 99.2% | 0.0001 | 99.2% | 0.0003 | 99.2% / 99.2% |
| VICReg + action | 91.7% | 0.0010 | 75.8% | 0.0097 | 91.7% / 75.8% |
| RSSM stochastic mean | 15.0% | 0.0149 | 20.0% | 0.0432 | 35.8% / 42.5% |
| No action | 12.5% | 0.0129 | 12.5% | 0.0429 | 12.5% / 12.5% |

RSSM seed 44 was 70.8% at both horizons under the privileged target. With a final image only it is 12.5% at H5 and 25.0% at H10. The earlier RSSM planning scores were not a matched comparison.

## Time-varying sequences, sixteen candidates

Two windows from each of eight training episodes. The initial state is still the start of the validation episode.

| Model | H5 top-1 | H5 endpoint RMS | H10 top-1 | H10 endpoint RMS |
| --- | ---: | ---: | ---: | ---: |
| MSE + action | 82.5% | 0.0011 | 87.5% | 0.0023 |
| VICReg + action | 65.0% | 0.0027 | 36.7% | 0.0169 |
| RSSM stochastic mean | 5.8% | 0.0141 | 7.5% | 0.0428 |
| No action | 8.3% | 0.0108 | 8.3% | 0.0362 |

Latent top-1 is whether the generating sequence was selected. Endpoint RMS is the physical error of that selection. VICReg's H10 sequence top-1 falls while MSE's does not. RSSM stays at chance. This is still finite-candidate selection, not closed-loop control.
