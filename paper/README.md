# PhysWorldAI workshop paper (NeurIPS 2026 style)

CFP ([Call for Papers](https://physworld-org.github.io/physworld.github.io/cfp)):
extended abstract **up to 4 pages** or long paper **up to 8 pages**; references
and appendix excluded. Style file is official `neurips_2026.sty` from
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
with `[dblblindworkshop]`. `neurips_workshop.sty` is a geometry stand-in — do
not upload a PDF built with it.

Tables in `main.tex` are `\input{}` from `generated/`, which `sync_tables.py`
rebuilds from `outputs/*.csv` and `outputs/*.json`. Do not edit generated files.

```bash
cd paper && make pdf
```

## Honesty constraints (from MAPPING.md)

- No PhysWorld / PhysTwin SOTA claim.
- Workshop ablations are a 2-seed range (`outputs/ablation_twoseed.json`), not mean±std.
- The 3-seed × 4-variant grid was skipped.
- Probe numbers must come from the finished 200-epoch checkpoint before camera-ready.

`make pdf` uses `pdflatex` if present, otherwise `tectonic`.
