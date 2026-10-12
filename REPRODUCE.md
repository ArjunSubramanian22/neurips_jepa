# Reproduce

Source commit: `see MANIFEST.json`.

The project tree has no root LICENSE. `anonymous_release/LICENSE_AUDIT.md` records third-party status and says redistribution beyond confidential review is not authorized by that note. This bundle does not change that. MuJoCo, if installed, is Apache-2.0.

Hashes are SHA-256 in `MANIFEST.json`. `small_artifact` stores hashes of checkpoints and large arrays. `full_artifact` copies a file once when it is under 52428800 bytes. Larger arrays stay hashed, with the command that regenerates them.

## 1. Environment setup

```bash
python3.13 -m pip install -r requirements.txt
python3.13 -m pip install 'mujoco==3.3.7'
```

The second install is optional and is only needed for the MuJoCo box and arm. Device selection is CUDA, then MPS, then CPU.

## 2. Baseline reproduction

```bash
SCALE=workshop DEVICE=auto OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
```

This writes a new directory. It does not replace `outputs/`.

## 3. One lightweight smoke run

```bash
SCALE=smoke DEVICE=cpu OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
```

The smoke scale builds a 48-episode dump and a short train. It is not a headline result.

## 4. Evaluator reproduction from supplied checkpoints

```bash
python3.13 scripts/evaluate_final.py --data outputs/physics_mpm --repo .
python3.13 -m benchmark.audit --config configs/benchmark/arm_jepa_audit.yaml
```

`evaluate_final.py` scores the legacy validation episodes 1800-1999. That split is not a test set. The audit command scores the arm checkpoint already named in the config.

## 5. Figure and table regeneration

```bash
MPLCONFIGDIR=outputs/.mplconfig python3.13 scripts/render_qualitative_figures.py
python3.13 scripts/generate_scientific_report.py
python3.13 scripts/plan_ablations.py
```

Figures read saved caches and tables. The ablation command is a dry run and does not train.
