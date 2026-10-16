# Response decomposition: decision

16 October 2026. Code base `b338136`. The new scorer and runs are uncommitted on top of that baseline.

## What the repository had already established

Action-conditioned VICReg, MSE, a no-action VICReg control, and an RSSM were trained on the high-excitation MPM dump and on a MuJoCo arm. On the high-excitation test, open-loop VICReg directional cosine falls from 0.618 at one step to 0.308 at ten while the median response ratio stays near 1.15. MSE and the RSSM stochastic mean keep cosine high. Teacher forcing restores VICReg cosine to about 0.94. The arm does not copy that split: VICReg's median ratio is about 50 to 60 and its cosine stays near zero. Matched final-image planning on 24 validation episodes ranks MSE above VICReg and leaves the image-only RSSM near chance. Those test sets were already opened. They are historical context, not confirmation of the comparison below.

Episode-level force-reversal cosine on those 24 planning episodes has Spearman correlation about 0 with planning regret. The planning task is often at zero regret. Cosine does not predict which of those episodes the planner misses.

## Choice

Primary question: when does a median response ratio near one describe a calibrated intervention, and when does it describe an orthogonal residual that has replaced the parallel gain?

Rejected for this round:

- A larger arm or MPM grid. The arm already bounds the MPM curve.
- A new planner. The existing finite-candidate task is too easy for within-episode prediction, and closed-loop control was a separate unrun experiment.
- Training the existing direction/magnitude alignment losses. Do-JEPA (Resani and Shi, arXiv:2609.37378) already supervises the latent effect of a same-state action pair and reports effect RMSE separately from effect cosine. An auxiliary loss that wins on the same pair is not a new claim until it changes a decision those metrics do not. That training was not run. One high-excitation epoch is about 38 s, so a 200-epoch ablation is about two hours per variant on this machine.

Closest published measurements, checked against the papers:

- Terver, Yang, Ponce, Bardes, and LeCun (arXiv:2512.24497, reviewed at https://openreview.net/forum?id=cHZn5Gdh8e) write that faithful open-loop unrolling does not by itself produce planning success, and that the validation prediction task is not fully aligned with goal-conditioned planning. In their Spearman tables the visual embedding loss is the metric most correlated with success. That dissociation is already published. This note does not claim that a latent error predicts planning.
- ActSWM (Gan et al., arXiv:2607.26712) names Context Collapse: a rollout can stay close to the encoded future while recorded-action and zero-action futures become nearly indistinguishable. Their action gap is a different cut from the angle between a predicted intervention and the encoder's intervention.
- Delta-JEPA (Zhang et al., arXiv:2606.31232) decodes the action from the latent displacement and probes physical displacement from encoder displacements. That is a claim about true latent differences, not about a predictor whose open-loop cosine falls while its median response ratio stays near one.
- Do-JEPA (Resani and Shi, arXiv:2609.37378) supervises the latent effect of a same-state action pair and reports effect cosine separately from effect RMSE. Effect RMSE ranks the open-loop and teacher-forced conditions here in the same order as cosine. The median ratio is the summary that does not.
- Parallel gain is cosine times the response ratio. It is an interpretation of those two numbers, not a third measurement. The empirical content is when the ratio sits near one while the cosine does not.

## Protocol frozen before the replication

`experiments/runs/intervention_decomposition_protocol.json`.

A material trap at horizon 10 means the open-loop median ratio is closer to one than the teacher-forced ratio by at least 0.05 in absolute log distance, while the median normalized vector error is smaller under teacher forcing. The 0.05 margin was set on the development slice, where an MSE reversal gap of 0.006 was a near-tie. Identity actions are the null: cosine stays undefined and both predicted norms are zero.

Development was dump seed 0, episodes 1440–1455, branches 0 and 4, training seed 42. The force-reversal cosine and ratio on that slice match the archived validation table to three decimals.

## Result

Held-out trajectories use dump seed 1, episodes 0–23, branches 0 and 4, training seeds 42 and 43. They were simulated for this comparison and are not the training dump. Normalized vector error prefers teacher forcing in every non-null cell. The ratio does not.

VICReg, horizon 10, held-out median cosine / ratio / normalized error, open loop then teacher forcing:

| Family | Seed 42 | Seed 43 | Material trap |
| --- | --- | --- | --- |
| Half force | 0.260 / 0.725 / 1.20 → 0.844 / 1.261 / 0.46 | 0.227 / 0.779 / 1.19 → 0.826 / 1.248 / 0.45 | no |
| Zero force | 0.284 / 1.006 / 1.35 → 0.930 / 1.258 / 0.25 | 0.299 / 0.871 / 1.28 → 0.937 / 1.244 / 0.23 | yes, both seeds |
| Reversal | 0.361 / 1.170 / 1.48 → 0.938 / 1.279 / 0.24 | 0.328 / 1.100 / 1.61 → 0.927 / 1.272 / 0.26 | yes, both seeds |
| 90° rotation | 0.314 / 1.117 / 1.48 → 0.931 / 1.302 / 0.29 | 0.324 / 1.043 / 1.46 → 0.931 / 1.262 / 0.24 | yes, both seeds |

MSE on the same held-out episodes has no material trap. Its open-loop cosine stays between 0.60 and 0.81, and the ratio gap does not prefer the higher-error condition by 0.05. The same pattern is on the replication block, validation episodes 1456–1471, seeds 42 and 43.

Parallel gain is cosine times the response ratio. On the development slice at ten steps, teacher-forced VICReg gain stays about 1.07 to 1.20 for every family. Open-loop gain is 0.17 for a half-force change, 0.25 for zero force, 0.36 for a rotation, and 0.44 for a reversal. Where the open-loop ratio is near one, the missing parallel component has been replaced by an orthogonal residual of similar norm. The half-force case is the boundary: the ratio also moves away from one, so the ratio and the error agree, while the gain is still the worst of the four.

The identity intervention produces a zero predicted difference for both models. The physical position RMS on the development slice rises from 0 at identity to 0.017, 0.035, 0.048, and 0.069 across half force, zero force, rotation, and reversal. The target-norm medians follow that order.

## Decision

Go for this diagnostic, with the scope above. The median response ratio is not a calibration check for these open-loop VICReg rollouts. Parallel gain, or cosine together with the ratio, shows when a ratio near one is orthogonal compensation.

No-go for a CVPR main-conference method or benchmark claim from this evidence alone. Effect RMSE and cosine already rank teacher forcing ahead of the open-loop rollout. The ratio is the statistic that fails. Episode-level cosine did not predict planning regret on the existing 24-episode task. No training intervention was shown to remove the orthogonal residual, and Do-JEPA already trains on paired effects. A competitive next experiment would have to change a planning or control outcome that cosine and effect RMSE do not already predict, or show that the orthogonal residual has a structure those scalars miss.

## What was not claimed

These checkpoints do not rank objectives in general. Teacher forcing is a different input, not a deployable policy. The held-out set is 24 new episodes of the same simulator, two training seeds, and two branch times. The arm ratio explosion is a separate regime and was not re-scored here. The high-excitation and arm test locks were not opened.
