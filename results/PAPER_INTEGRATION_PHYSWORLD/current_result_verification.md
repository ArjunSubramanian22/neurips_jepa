# Current result verification

Compared the brief’s sanity-check numbers to `outputs/corrected_seed43_audit.json` (last.pt) and the dump.

| Item | Brief (sanity) | Measured | Verdict |
| --- | --- | --- | --- |
| obs sha256 | ee3fe0…4cb0759f | identical | match |
| shapes / split | 2000, 16, 1800/200 | identical | match |
| s42 MSE@1/5/10 | 0.07571 / 0.62213 / 1.19953 | 0.07571088522672653 / 0.6221303343772888 / 1.199525237083435 | rounding |
| s43 MSE@1/5/10 | 0.06946 / 0.61565 / 1.10682 | 0.0694643184542656 / 0.6156516075134277 / 1.1068207025527954 | rounding |
| s42 pred rank / top-1 | 12.27 / 0.178 | 12.267736855841244 / 0.17782228767885566 | rounding |
| s43 pred rank / top-1 | 5.78 / 0.333 | 5.776180055689214 / 0.3327033324340438 | rounding |
| s42 geo/stiff/contact | 0.9985 / 0.3025 / 0.3955 | 0.9985000014305114 / 0.3024999976158142 / 0.3954999953508377 | rounding |
| s42 shuffle % t=0 / change | 0.01604 / 0.01634 | 0.01604054419760376 / 0.016341159419844684 | rounding (units: percent relative error change) |
| Legacy cropped VICReg last.pt MSE@10 | ~0.899 | 0.8932738304138184 (`mpm_action_ablation.json`) | small; last.pt |
| Legacy MSE@10 brief 0.090 / 0.046 | vs `ablation_twoseed.json` best.pt 0.114 / 0.184 | **protocol mismatch, not a silent retarget** |

VN-MSE: not in `src/eval.py`.

`outputs/mpm_action_ablation.json` geometry 0.9995 vs audit 0.9985 for seed-42 nocrop: probe refits, not dump drift.
