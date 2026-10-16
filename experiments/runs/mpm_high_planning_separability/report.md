# Planning-task separability

Same 24 validation episodes, libraries, and goal indices as the matched-planning table. Models were not reloaded. A candidate pair is close when the oracle margin is under 25% of that episode's median pairwise endpoint distance. Easy, medium, and hard are tertiles of the margin.

- constant H5: median margin 0.007915 (p10 0.005176, p90 0.01398), pairwise median 0.01592, close 0.0%, oracle is goal 100.0%
- constant H10: median margin 0.02661 (p10 0.0175, p90 0.0432), pairwise median 0.05222, close 0.0%, oracle is goal 100.0%
- sequence H5: median margin 0.002678 (p10 0.00127, p90 0.004329), pairwise median 0.0143, close 62.5%, oracle is goal 100.0%
- sequence H10: median margin 0.01571 (p10 0.01027, p90 0.02581), pairwise median 0.04136, close 20.8%, oracle is goal 100.0%

The eight-action task is not a near-tie. Its smallest H5 margin is 0.00465 and its smallest H10 margin is 0.0162, against pairwise medians of 0.0159 and 0.0522. No constant-action episode meets the close rule. MSE stays at 97.5–100% in the hard tertile, where the runner-up is nearest. The sixteen-sequence task is more crowded: 62.5% of H5 episodes and 20.8% of H10 episodes are close. MSE remains well above the 6.25% chance line on those hard episodes too. Each tertile is only eight episodes, so the stratum percentages are noisy. The no-action model stays near chance overall.

## Top-1 by margin tertile

- vicreg_action constant H5 easy: top1 92.5% over 8 episodes (margin median 0.01319)
- vicreg_action constant H5 medium: top1 90.0% over 8 episodes (margin median 0.007915)
- vicreg_action constant H5 hard: top1 92.5% over 8 episodes (margin median 0.006045)
- vicreg_noaction constant H5 easy: top1 0.0% over 8 episodes (margin median 0.01319)
- vicreg_noaction constant H5 medium: top1 37.5% over 8 episodes (margin median 0.007915)
- vicreg_noaction constant H5 hard: top1 0.0% over 8 episodes (margin median 0.006045)
- mse_action constant H5 easy: top1 100.0% over 8 episodes (margin median 0.01319)
- mse_action constant H5 medium: top1 97.5% over 8 episodes (margin median 0.007915)
- mse_action constant H5 hard: top1 100.0% over 8 episodes (margin median 0.006045)
- rssm constant H5 easy: top1 20.0% over 8 episodes (margin median 0.01319)
- rssm constant H5 medium: top1 20.0% over 8 episodes (margin median 0.007915)
- rssm constant H5 hard: top1 5.0% over 8 episodes (margin median 0.006045)
- vicreg_action constant H10 easy: top1 75.0% over 8 episodes (margin median 0.03581)
- vicreg_action constant H10 medium: top1 82.5% over 8 episodes (margin median 0.02661)
- vicreg_action constant H10 hard: top1 70.0% over 8 episodes (margin median 0.02058)
- vicreg_noaction constant H10 easy: top1 0.0% over 8 episodes (margin median 0.03581)
- vicreg_noaction constant H10 medium: top1 37.5% over 8 episodes (margin median 0.02661)
- vicreg_noaction constant H10 hard: top1 0.0% over 8 episodes (margin median 0.02058)
- mse_action constant H10 easy: top1 100.0% over 8 episodes (margin median 0.03581)
- mse_action constant H10 medium: top1 97.5% over 8 episodes (margin median 0.02661)
- mse_action constant H10 hard: top1 100.0% over 8 episodes (margin median 0.02058)
- rssm constant H10 easy: top1 35.0% over 8 episodes (margin median 0.03581)
- rssm constant H10 medium: top1 22.5% over 8 episodes (margin median 0.02661)
- rssm constant H10 hard: top1 2.5% over 8 episodes (margin median 0.02058)
- vicreg_action sequence H5 easy: top1 67.5% over 8 episodes (margin median 0.004275)
- vicreg_action sequence H5 medium: top1 65.0% over 8 episodes (margin median 0.002678)
- vicreg_action sequence H5 hard: top1 62.5% over 8 episodes (margin median 0.001322)
- vicreg_noaction sequence H5 easy: top1 0.0% over 8 episodes (margin median 0.004275)
- vicreg_noaction sequence H5 medium: top1 25.0% over 8 episodes (margin median 0.002678)
- vicreg_noaction sequence H5 hard: top1 0.0% over 8 episodes (margin median 0.001322)
- mse_action sequence H5 easy: top1 82.5% over 8 episodes (margin median 0.004275)
- mse_action sequence H5 medium: top1 70.0% over 8 episodes (margin median 0.002678)
- mse_action sequence H5 hard: top1 95.0% over 8 episodes (margin median 0.001322)
- rssm sequence H5 easy: top1 10.0% over 8 episodes (margin median 0.004275)
- rssm sequence H5 medium: top1 5.0% over 8 episodes (margin median 0.002678)
- rssm sequence H5 hard: top1 2.5% over 8 episodes (margin median 0.001322)
- vicreg_action sequence H10 easy: top1 57.5% over 8 episodes (margin median 0.02481)
- vicreg_action sequence H10 medium: top1 37.5% over 8 episodes (margin median 0.01571)
- vicreg_action sequence H10 hard: top1 15.0% over 8 episodes (margin median 0.01073)
- vicreg_noaction sequence H10 easy: top1 12.5% over 8 episodes (margin median 0.02481)
- vicreg_noaction sequence H10 medium: top1 12.5% over 8 episodes (margin median 0.01571)
- vicreg_noaction sequence H10 hard: top1 0.0% over 8 episodes (margin median 0.01073)
- mse_action sequence H10 easy: top1 92.5% over 8 episodes (margin median 0.02481)
- mse_action sequence H10 medium: top1 77.5% over 8 episodes (margin median 0.01571)
- mse_action sequence H10 hard: top1 92.5% over 8 episodes (margin median 0.01073)
- rssm sequence H10 easy: top1 10.0% over 8 episodes (margin median 0.02481)
- rssm sequence H10 medium: top1 12.5% over 8 episodes (margin median 0.01571)
- rssm sequence H10 hard: top1 0.0% over 8 episodes (margin median 0.01073)

