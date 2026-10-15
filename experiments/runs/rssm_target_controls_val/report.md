# RSSM target controls on the full validation slice

Episodes 1440-1619, branch times 0 and 3, five seeds. 1800 contrasts per mode and horizon. The test split was not read. `swapped_action` is the negation of `cross_fork_image` and is not a separate result.

Seed medians of stochastic cosine:

| Mode | H1 seeds | H10 seeds |
| --- | --- | --- |
| Intact | 0.900, 0.955, 0.966, 0.977, 0.918 | 0.870, 0.936, 0.956, 0.930, 0.921 |
| Blank image | 0.865, 0.919, 0.907, 0.914, 0.888 | 0.588, 0.708, 0.629, 0.693, 0.629 |
| Other fork's image | 0.483, 0.525, 0.513, 0.700, 0.467 | -0.158, -0.043, -0.048, 0.156, -0.118 |
| Zero action | 0.220, 0.350, 0.180, 0.155, 0.259 | 0.548, 0.561, 0.557, 0.419, 0.555 |

The four-episode pilot said the H10 cross-fork cosine was negative for every seed. On all 180 validation episodes, seed 45 is positive (0.156) and the pooled median is -0.041. That sign claim is withdrawn.

What still holds for every seed: blanking the image leaves H1 cosine high and lowers H10; zeroing the action drops H1 into the 0.15-0.35 range. The one-step target alignment is carried largely by the action. At ten steps the image matters, but swapping in the other fork's image does not produce a seed-universal reversal.
