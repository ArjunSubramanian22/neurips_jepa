# Anonymous Release Audit

## Scope

This snapshot is a code-and-results artifact for double-blind review. It does
**not** include manuscript LaTeX, a compiled workshop PDF, a bibliography, or
NeurIPS style files.

`python scripts/check_anonymity.py` recursively scans the entire tree
(not only `results/`). After removing the manuscript, the scanner reported
PASS on the remaining files (Python, Markdown, JSON, YAML, CSV, shell, logs,
and binary checkpoint/figure metadata).

## Excluded files/directories

Removed from this snapshot, including in this pass:

- `paper/` (LaTeX source, generated macros/tables, style files, bibliography,
  compiled `main.pdf`, claim checker)
- Candidate `.tex` table snippets under `results/`
- Git history, virtualenvs, smoke-test outputs, obsolete cropped checkpoints
- `outputs/physics_mpm/observations.npy` (~375 MB; hash in `DATA_MANIFEST.csv`)
- Identity-leaking working-tree files (absolute home paths, zip enumerations,
  training logs, launch JSON with git commits)

Retained experimental plots live under
`results/reviewer_strengthening/paper_figures/` (PNG/PDF figures generated from
cached CSVs). Those are not a compiled manuscript.

## Sanitized files

Identity-only edits (scientific CSV/JSON/checkpoint bytes unchanged):

- `scripts/train_reviewer_seed.sh` — removed absolute interpreter path and git
  commit recording
- `scripts/reproduce_all.sh` — removed git provenance
- `DATA_MANIFEST.csv` / `EXPERIMENT_MANIFEST.csv` — portable notes only
- `results/reviewer_strengthening/00_baseline_audit/environment.json` — git
  metadata omitted; library versions unchanged

## Identity checks

- personal names: PASS
- emails: PASS
- absolute paths: PASS
- institutional affiliations: PASS
- git metadata: PASS
- PDF metadata: PASS (figure PDFs credit Matplotlib only; no manuscript PDF)
- secrets: PASS

## Reproducibility checks

- canonical evaluator present: `scripts/evaluate_final.py`
- checkpoint manifest present: `CHECKPOINT_MANIFEST.csv`
- figure/table regeneration from cache: `scripts/regenerate_cached_tables.py`
- relative paths: scripts use `Path(__file__).resolve().parents[1]`

## License status

See `LICENSE_AUDIT.md`. Project LICENSE absent. MLS-MPM heritage vs the
Taichi 88-line demo is UNRESOLVED.

## Remaining limitations

- `observations.npy` is omitted; cached CSVs reproduce reported metrics.
- This artifact does not ship the workshop manuscript.

## Final verdict

ANONYMOUS RELEASE PASS
