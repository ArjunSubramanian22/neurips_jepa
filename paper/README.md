# Workshop paper (4-page NeurIPS format)

Tables in `main.tex` are `\input{}` from `generated/`, which `sync_tables.py`
rebuilds from `outputs/*.csv` and `outputs/*.json`. Do not edit generated files.

```bash
python paper/sync_tables.py
python paper/check_claims.py
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`neurips_workshop.sty` matches NeurIPS letter / 5.5in text width. For an
official submission, replace that package with `neurips_2025`.

## Honesty constraints (from MAPPING.md)

- No PhysWorld / PhysTwin SOTA claim.
- `outputs/ablation_multiseed.json` is missing: the paper includes a flag table,
  not invented mean±std numbers.
- `tab:ac-jepa-ablation` is dummy-scale (`n_val` and epoch count come from
  `ablation_results.csv` / `history.json`).
- Probe numbers come from `probe_diagnostics.json` (in-progress VICReg ckpt).
- Do not invent mean±std numbers; `generated/multiseed_table.tex` is a flag table
  until `outputs/ablation_multiseed.json` exists.

This environment does not have `pdflatex` installed; compile locally with the
Makefile `pdf` target.
