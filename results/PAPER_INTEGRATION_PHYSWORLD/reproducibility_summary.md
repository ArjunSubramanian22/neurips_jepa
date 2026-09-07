# Reproducibility summary

Entry: `python reproduce_paper_results.py --stage audit` then `--stage analysis`.

Counterfactual generation: `python scripts/run_simulator_counterfactual.py --device cpu --max-val-episodes 200` then `python scripts/analyze_simulator_counterfactual.py`.

Do not write into `outputs/vicreg_nocrop/` or the physics dump.

Lockfile: `requirements-lock.txt` (Python 3.13.2, torch 2.12.1, numpy 2.5.2, sklearn 1.9.0, matplotlib 3.11.1).

Full dump regeneration bitwise identity: **not tested**. Per-episode replay identity: tested.

See `REPRODUCIBILITY.md`.
