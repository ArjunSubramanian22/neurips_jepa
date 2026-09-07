# Anonymity audit

Scanned `results/reviewer_strengthening/` text artifacts (md/json/csv/tex) for home paths, emails, GitHub remotes, and key assignments.

## Hits

_none_ in those artifacts.

Internal git commit hashes appear in `00_baseline_audit/environment.json` (`git.commit`). That file is for the reproducibility engineer, not for OpenReview upload. Strip git hashes from any file copied into a public anonymous zip if required by the venue.

Do not copy `environment.json` into the OpenReview supplementary if the program chair treats commit IDs as identity-adjacent; the hash does not include a username.
