# Issues

| ID | Priority | Status | Issue |
| --- | --- | --- | --- |
| I1 | done | closed | Teacher-forced JEPA physical/raster rows used the latent-list index. Fixed in the scorer. v2 CSVs recomputed from cache. v1 files kept. |
| I2 | done | closed | Pytest collected `test_once_main`, and the freeze guard required the real freeze to be absent. Alias renamed. Guard uses a temp freeze path and does not touch the lock. |
| I3 | done | closed | RSSM planning saw the goal trajectory and generating actions. Matched final-image run is in `experiments/runs/mpm_high_planning_matched/`. |
| I4 | done | narrowed | RSSM posterior targets are action-conditioned. Full validation controls replace the four-episode sign claim. |
| I5 | open | waiting | Arm five-seed comparison is the second-environment test of hypothesis 4. Not started. Gate is 19 October. Test episodes 600-719 stay closed. |
| I6 | open | recorded | Arm validation MSE and directional cosine disagree across epochs on one seed. Production reports both checkpoints. |
| I7 | open | deferred | Closed-loop control is not the same as candidate ranking. `scripts/run_planning_benchmark.py` does not evaluate the high-excitation checkpoints under a matched final image. |
| I8 | open | not started | `paper/main.tex` still describes the older stiffness-probe manuscript. Rewrite waits on the arm result and the frozen claims. |
| I9 | open | noted | Legacy rows have four VICReg seeds, two MSE seeds, and no RSSM. Do not treat that slice as the same design as the five-seed high-excitation test. |
