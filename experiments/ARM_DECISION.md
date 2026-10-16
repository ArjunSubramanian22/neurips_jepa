# Arm decision

15 October 2026. The five-seed grid was started the same day, without waiting for 19 October. Episodes 600–719 were scored once after this decision was fixed. The lock is `experiments/runs/arm_campaign_v2/test_once/TEST_EVALUATED.lock`.

## Decision

The arm supports the diagnostic as a measurement, and it is a boundary for the MPM pattern.

The controls transfer. Reflected joint targets change the image at one step in 84% of test episodes, while the object moves in only 40%. By ten steps the object moves in 72% of episodes and the image changes in 82%. Those image rates are the fraction of episodes with raster MSE above 0.01 (101/120 at one step, 98/120 at ten). Raster MSE is positive on all 120 test episodes at both horizons, so the frames are not identical in the other 16% and 18%. Object motion is object L2 above 1e-6 (48/120 at one step, 86/120 at ten). The no-action VICReg predicts a zero delta at every horizon, seed, and checkpoint, so its cosine stays undefined. Action-conditioned models do not.

The MPM pattern does not transfer. On the high-excitation test, VICReg cosine falls from 0.618 to 0.308 while the response ratio stays near 1. On the arm test, VICReg cosine stays near zero (0.016 at one step and 0.039 at ten steps for the validation-MSE checkpoint) and the response ratio is about 50 to 60. MSE cosine rises with horizon on the validation-MSE checkpoint, from 0.178 to 0.405, while its response ratio falls from 0.69 to 0.45. The final MSE checkpoint has higher cosine and much larger latent norms. Both checkpoints were kept, as preregistered. RSSM cosine stays near 0.6 to 0.7, with seed 46 lower at one step (0.16).

No further environment will be built to recover the MPM curve. The sealed test was not used to drop a model or a checkpoint.
