# High-excitation protocol freeze

Git commit `217899d2bf2329efbfeb9979a1d95cf78ba087ba`.

High-excitation test episodes 1620–1799 have not been scored.

Validation episodes are 1440-1619. Branch times are 0, 1, 2, 3, 4, and 5. Horizons are 1, 5, and 10.
The intervention reverses action components 2: for the whole horizon.
Open loop rolls the model state forward. Teacher forcing refreshes the state from the physical fork.
RSSM cosine uses the stochastic mean. The deterministic state is reported separately because it is shared at one step.
Undefined cosine stays null.
Branch times are aggregated inside each episode before the episode bootstrap. Seeds are the replication unit.
Bottleneck thresholds: position RMS 1e-5, target norm 1e-8, success cosine 0.5, success response ratio 0.2.
Visibility uses one physical-fork threshold: all forks, non-identical rasters, and the top third of nonzero raster MSE.
Planning uses eight actions from training episodes 0, 80, 160, 240, 320, 400, 480, and 560, on validation episodes 1440-1463.
No architecture was selected on the test set. FiLM, gate, cross-attention, and alignment training stayed deferred.
