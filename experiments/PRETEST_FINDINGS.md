# Pre-test findings

Git commit `217899d2bf2329efbfeb9979a1d95cf78ba087ba`. High-excitation test episodes 1620-1799 were not scored.

## RSSM audit

One-step observe and imagine apply the same GRU update before the posterior sees the frame. concat(h, z) therefore shares delta_h exactly. The corrected predictor contrast is the stochastic mean only.
Concat cosines: [0.997282142666164, 0.9983410958341569, 0.9993197412237649, 0.9996931242483081, 0.9976688208973385].
Stochastic-mean cosines: [0.9143820898398167, 0.9653611693987317, 0.9817829424739709, 0.9925980127589035, 0.9409360534618483].
Shuffle-versus-original-target stochastic cosines: [0.3657340215077771, 0.3436207028709247, 0.44608863724423065, 0.458499528570281, 0.45256013945813084].
The published H1 screen files were not rewritten. Later RSSM rows use the stochastic mean.

## Horizon results, high excitation, open loop

- vicreg_action H1 cosine: 42=0.5951, 43=0.6381, 44=0.652, 45=0.6262, 46=0.6192
- vicreg_action H1 projection: 42=0.493, 43=0.5956, 44=0.5718, 45=0.5384, 46=0.6151
- vicreg_action H1 response ratio: 42=0.8396, 43=0.9748, 44=0.9128, 45=0.8919, 46=1.039
- vicreg_action H5 cosine: 42=0.5511, 43=0.5537, 44=0.5286, 45=0.5391, 46=0.5531
- vicreg_action H5 projection: 42=0.5474, 43=0.5868, 44=0.563, 45=0.54, 46=0.6309
- vicreg_action H5 response ratio: 42=1.04, 43=1.06, 44=1.075, 45=1.007, 46=1.189
- vicreg_action H10 cosine: 42=0.3492, 43=0.3157, 44=0.2814, 45=0.3268, 46=0.296
- vicreg_action H10 projection: 42=0.3497, 43=0.3452, 44=0.3057, 45=0.364, 46=0.3516
- vicreg_action H10 response ratio: 42=1.094, 43=1.098, 44=1.14, 45=1.145, 46=1.183

- vicreg_noaction H1 cosine: 42=null, 43=null, 44=null, 45=null, 46=null
- vicreg_noaction H1 projection: 42=0, 43=0, 44=0, 45=0, 46=0
- vicreg_noaction H1 response ratio: 42=0, 43=0, 44=0, 45=0, 46=0
- vicreg_noaction H5 cosine: 42=null, 43=null, 44=null, 45=null, 46=null
- vicreg_noaction H5 projection: 42=0, 43=0, 44=0, 45=0, 46=0
- vicreg_noaction H5 response ratio: 42=0, 43=0, 44=0, 45=0, 46=0
- vicreg_noaction H10 cosine: 42=null, 43=null, 44=null, 45=null, 46=null
- vicreg_noaction H10 projection: 42=0, 43=0, 44=0, 45=0, 46=0
- vicreg_noaction H10 response ratio: 42=0, 43=0, 44=0, 45=0, 46=0

- mse_action H1 cosine: 42=0.725, 43=0.7307, 44=0.7341, 45=0.7369, 46=0.742
- mse_action H1 projection: 42=0.5704, 43=0.5635, 44=0.5749, 45=0.5919, 46=0.6067
- mse_action H1 response ratio: 42=0.7922, 43=0.7795, 44=0.7883, 45=0.815, 46=0.8075
- mse_action H5 cosine: 42=0.8523, 43=0.8545, 44=0.8398, 45=0.8636, 46=0.8676
- mse_action H5 projection: 42=0.7828, 43=0.7791, 44=0.7658, 45=0.8055, 46=0.7972
- mse_action H5 response ratio: 42=0.9304, 43=0.919, 44=0.9189, 45=0.9364, 46=0.932
- mse_action H10 cosine: 42=0.8071, 43=0.8202, 44=0.8225, 45=0.8248, 46=0.8419
- mse_action H10 projection: 42=0.7913, 43=0.7853, 44=0.7743, 45=0.7952, 46=0.8087
- mse_action H10 response ratio: 42=0.966, 43=0.9523, 44=0.9509, 45=0.9642, 46=0.9577

- rssm H1 cosine: 42=0.8698, 43=0.927, 44=0.9643, 45=0.9471, 46=0.9022
- rssm H1 projection: 42=0.6351, 43=0.7731, 44=0.8722, 45=0.8754, 46=0.6944
- rssm H1 response ratio: 42=0.748, 43=0.8368, 44=0.9096, 45=0.9241, 46=0.8109
- rssm H5 cosine: 42=0.8884, 43=0.9436, 44=0.9756, 45=0.9472, 46=0.9275
- rssm H5 projection: 42=0.5536, 43=0.676, 44=0.778, 45=0.7407, 46=0.6222
- rssm H5 response ratio: 42=0.6143, 43=0.714, 44=0.8001, 45=0.7996, 46=0.6893
- rssm H10 cosine: 42=0.8769, 43=0.927, 44=0.9616, 45=0.9361, 46=0.9124
- rssm H10 projection: 42=0.5688, 43=0.6593, 44=0.7884, 45=0.7524, 46=0.6337
- rssm H10 response ratio: 42=0.6487, 43=0.7369, 44=0.8417, 45=0.8215, 46=0.7154

## Open loop versus teacher forced

- vicreg_action H1 teacher-forced cosine: 42=0.5951, 43=0.6381, 44=0.652, 45=0.6262, 46=0.6192
- vicreg_action H5 teacher-forced cosine: 42=0.9366, 43=0.9331, 44=0.9326, 45=0.9401, 46=0.9393
- vicreg_action H10 teacher-forced cosine: 42=0.9385, 43=0.9352, 44=0.9353, 45=0.9355, 46=0.9381

- mse_action H1 teacher-forced cosine: 42=0.725, 43=0.7307, 44=0.7341, 45=0.7369, 46=0.742
- mse_action H5 teacher-forced cosine: 42=0.9683, 43=0.9669, 44=0.9686, 45=0.9706, 46=0.9732
- mse_action H10 teacher-forced cosine: 42=0.9685, 43=0.968, 44=0.9686, 45=0.9718, 46=0.9737

- rssm H1 teacher-forced cosine: 42=0.8698, 43=0.927, 44=0.9643, 45=0.9471, 46=0.9022
- rssm H5 teacher-forced cosine: 42=0.9656, 43=0.9786, 44=0.992, 45=0.9827, 46=0.9755
- rssm H10 teacher-forced cosine: 42=0.9457, 43=0.969, 44=0.9882, 45=0.9769, 46=0.961

- vicreg_noaction H1 teacher-forced cosine: 42=null, 43=null, 44=null, 45=null, 46=null
- vicreg_noaction H5 teacher-forced cosine: 42=0.9342, 43=0.9375, 44=0.9358, 45=0.9364, 46=0.9321
- vicreg_noaction H10 teacher-forced cosine: 42=0.9396, 43=0.939, 44=0.9383, 45=0.9385, 46=0.9383

For a one-step feedforward predictor, open loop and teacher forcing use the same state, so H=1 matches.
Teacher-forced no-action contrast at H>1 can be nonzero because the two forks have different observed states. That is not action-argument sensitivity. The same-state action norm is the action-pathway check.

## Legacy versus high excitation

These are two controlled simulator regimes, not a randomized treatment.

- mpm_legacy H1: physical 1.652e-05, raster 3.504e-08, target 0.004406, predictor 0.01039, cosine 0.07301, ratio 0.1205
- mpm_legacy H5: physical 0.0001337, raster 4.434e-06, target 0.2685, predictor 0.02816, cosine 0.03405, ratio 0.06569
- mpm_legacy H10: physical 0.0002315, raster 8.905e-06, target 0.7414, predictor 0.04011, cosine 0.01562, ratio 0.05269
- mpm_high H1: physical 0.001126, raster 0.0007027, target 4.613, predictor 4.044, cosine 0.6321, ratio 0.9023
- mpm_high H5: physical 0.02153, raster 0.01216, target 11.2, predictor 11.57, cosine 0.5451, ratio 1.05
- mpm_high H10: physical 0.06114, raster 0.02257, target 11.83, predictor 13.26, cosine 0.3212, ratio 1.119

## Planning

- vicreg_action seed 42 H5: top1 0.875 top3 1 regret 0.001626
- vicreg_action seed 42 H10: top1 0.7917 top3 1 regret 0.008034
- vicreg_action seed 43 H5: top1 0.9583 top3 1 regret 0.0003198
- vicreg_action seed 43 H10: top1 0.6667 top3 0.9583 regret 0.01508
- vicreg_action seed 44 H5: top1 0.9167 top3 1 regret 0.001261
- vicreg_action seed 44 H10: top1 0.8333 top3 1 regret 0.006276
- vicreg_action seed 45 H5: top1 1 top3 1 regret 0
- vicreg_action seed 45 H10: top1 0.7917 top3 0.9583 regret 0.007324
- vicreg_action seed 46 H5: top1 0.8333 top3 1 regret 0.001836
- vicreg_action seed 46 H10: top1 0.7083 top3 1 regret 0.01163
- vicreg_noaction seed 42 H5: top1 0.125 top3 0.375 regret 0.01294
- vicreg_noaction seed 42 H10: top1 0.125 top3 0.375 regret 0.04285
- vicreg_noaction seed 43 H5: top1 0.125 top3 0.375 regret 0.01294
- vicreg_noaction seed 43 H10: top1 0.125 top3 0.375 regret 0.04285
- vicreg_noaction seed 44 H5: top1 0.125 top3 0.375 regret 0.01294
- vicreg_noaction seed 44 H10: top1 0.125 top3 0.375 regret 0.04285
- vicreg_noaction seed 45 H5: top1 0.125 top3 0.375 regret 0.01294
- vicreg_noaction seed 45 H10: top1 0.125 top3 0.375 regret 0.04285
- vicreg_noaction seed 46 H5: top1 0.125 top3 0.375 regret 0.01294
- vicreg_noaction seed 46 H10: top1 0.125 top3 0.375 regret 0.04285
- mse_action seed 42 H5: top1 1 top3 1 regret 0
- mse_action seed 42 H10: top1 1 top3 1 regret 0
- mse_action seed 43 H5: top1 1 top3 1 regret 0
- mse_action seed 43 H10: top1 0.9583 top3 1 regret 0.001439
- mse_action seed 44 H5: top1 1 top3 1 regret 0
- mse_action seed 44 H10: top1 1 top3 1 regret 0
- mse_action seed 45 H5: top1 1 top3 1 regret 0
- mse_action seed 45 H10: top1 1 top3 1 regret 0
- mse_action seed 46 H5: top1 0.9583 top3 0.9583 regret 0.0003528
- mse_action seed 46 H10: top1 1 top3 1 regret 0
- rssm seed 42 H5: top1 0.1667 top3 0.5 regret 0.01101
- rssm seed 42 H10: top1 0.2917 top3 0.5417 regret 0.03098
- rssm seed 43 H5: top1 0.2917 top3 0.5 regret 0.01123
- rssm seed 43 H10: top1 0.4167 top3 0.6667 regret 0.0271
- rssm seed 44 H5: top1 0.7083 top3 0.9583 regret 0.002544
- rssm seed 44 H10: top1 0.7083 top3 1 regret 0.007826
- rssm seed 45 H5: top1 0.3333 top3 0.5833 regret 0.01061
- rssm seed 45 H10: top1 0.4583 top3 0.75 regret 0.02398
- rssm seed 46 H5: top1 0.2917 top3 0.375 regret 0.01138
- rssm seed 46 H10: top1 0.25 top3 0.4167 regret 0.03936

## Arm

Reflected joint-position targets on episodes 24-31. Not an MPM force reversal.

- jepa H1 episode-median cosines by seed: [0.02455729074784579, 0.047567756897925015, -0.10176643732755211, -0.10421493083037281, 0.09602416400467897]
- jepa H7 episode-median cosines by seed: [0.0370170406486109, 0.04109028208669996, 0.04363300897253871, 0.059145398339691194, -0.11623567531272526]
- rssm H1 episode-median cosines by seed: [0.1368584999407359, 0.11224485258576786, 0.3340970171983437, 0.17207191716445708, -0.0018041939443461772]
- rssm H7 episode-median cosines by seed: [0.07951764065120825, 0.03060541764858224, 0.2703989658867094, 0.18401546454400444, 0.03681573809471602]
- latent_tf H1 episode-median cosines by seed: [-0.08193367846362962, 0.01912218158887231, 0.005935258330729121, -0.14921137700442094, 0.10163987667608074]
- latent_tf H7 episode-median cosines by seed: [-0.11889131859010102, -0.0720476921798949, 0.04166639619426402, -0.14362763263322753, 0.08138158079122784]

## Visibility

Top-third raster-MSE threshold, fit once on the physical forks: {'top_third_mse': 0.0010203398999850484, 'n_nonzero': 1079}.
Seed-level rank correlations are in experiments/runs/mpm_high_multih/visibility.json.

## Limitations

- The original H1 screen used 24 episodes at t=0. The multi-horizon table uses 180 validation episodes and six branch times, aggregated within episode.
- JEPA seeds 42 and 43 and no-action seed 42 still lack episode-level bootstrap intervals in the old 24-episode screen. The new table bootstraps episodes.
- RSSM concat cosine at H=1 is not an independent measurement. The stochastic mean is the reported RSSM cosine.
- Legacy checkpoints do not include seed 46 or an RSSM. The regime table uses JEPA action seeds 42-45.
- Legacy validation was used to choose historical checkpoints. It is a regime comparison, not a sealed test.
- Planning uses 24 validation episodes and eight training actions. Five seeds is a small replicate for the alignment-versus-planning correlation.
- The arm has eight validation episodes and no held-out test. No sealed arm dump was generated.
- No-action teacher-forced multi-step contrast is not a pure action-null.
