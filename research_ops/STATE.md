# State

Updated 15 October 2026, after the arm test was opened once.

The October 19 wait was dropped. The five-seed arm grid has finished. Validation was scored before the test. The test lock is `experiments/runs/arm_campaign_v2/test_once/TEST_EVALUATED.lock`. Do not score those episodes again.

Decision: the arm diagnostic runs, and the MPM direction-versus-magnitude pattern does not appear there. No new environment is planned. The write-up is `paper/diagnostic.tex`. The planning separability audit is `experiments/runs/mpm_high_planning_separability/report.md`.

MPM checkpoints were not retrained.

16 October 2026. A separate diagnostic is in `experiments/RESPONSE_DECOMPOSITION.md`. On newly simulated dump-seed-1 episodes, open-loop VICReg at ten steps has a median response ratio closer to one than teacher forcing for zeroed, reversed, and rotated forces, while normalized vector error and cosine prefer teacher forcing. Halving the force does not produce that reversal. MSE does not produce it. The high-excitation and arm test locks were not opened. This is a measurement result. It is not a claim that the decomposition predicts planning beyond cosine, and the alignment-loss ablation was not trained.
