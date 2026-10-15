# State

Updated 15 October 2026, after the validation-wide RSSM target controls. No training or evaluation process is running. Confirmatory-queue pid files are stale.

Phase: development. The high-excitation test set is already opened and is not used for selection. The arm test suffix, episodes 600-719, has not been used for training or selection.

Machine: Apple M1, 8 GB RAM, MPS, about 14 GB free.

## Established this session

- Open-loop held-out summary is in `experiments/runs/canonical_counterfactual/summary.md`. VICReg cosine on the test falls from 0.618 at H1 to 0.308 at H10 while the response ratio stays near 1. Validation matches. Legacy does not.
- Teacher-forced physical medians in the v2 files match open-loop physical medians. The index correction holds.
- RSSM target controls on episodes 1440-1619 withdraw the four-episode sign claim at H10. Blank-image and zero-action patterns remain.

## Next

Do not start the five-seed arm grid before 19 October. When it starts, train the four core models, keep both the validation-MSE checkpoint and the final checkpoint, and leave episodes 600-719 unread until a validation freeze is written.

Closed-loop stays unlaunched. The existing benchmark script is a different protocol.

## Not running

Nothing. Do not resume `experiments/confirmatory_queue` pid files.
