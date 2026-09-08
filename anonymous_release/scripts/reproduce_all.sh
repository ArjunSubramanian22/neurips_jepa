#!/usr/bin/env bash
# Reproduce AC-JEPA artifacts into a fresh directory (default: outputs_repro/).
#
#   SCALE=smoke     short synthetic + tiny physics dump (default; can be diffed)
#   SCALE=workshop  Prompt-1 dump + 200-epoch train + 3-seed ablations (hours–days)
#
#   OUTPUT_ROOT=outputs_repro DEVICE=cpu bash scripts/reproduce_all.sh
#   SCALE=workshop DEVICE=auto OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
#
# Does not set WANDB_API_KEY. TrainingConfig.device=auto picks CUDA, then MPS, then CPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SCALE="${SCALE:-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_repro}"
DEVICE="${DEVICE:-}"
export PYTHONHASHSEED=0

if [[ -z "$DEVICE" ]]; then
  if [[ "$SCALE" == "workshop" ]]; then
    DEVICE="auto"
  else
    DEVICE="cpu"
  fi
fi

if [[ "$OUTPUT_ROOT" == "outputs" ]]; then
  echo "Refusing to write SCALE=$SCALE into outputs/ (that tree is the audit target)." >&2
  echo "Set OUTPUT_ROOT=outputs_repro (or another path)." >&2
  exit 1
fi

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

torch_ver="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
resolved_device="$("$PYTHON" -c 'from src.config import resolve_device; print(resolve_device("'"$DEVICE"'"))' 2>/dev/null || echo unknown)"

{
  echo "{"
  echo "  \"scale\": \"$SCALE\","
  echo "  \"output_root\": \"$OUTPUT_ROOT\","
  echo "  \"device_arg\": \"$DEVICE\","
  echo "  \"resolved_device\": \"$resolved_device\","
  echo "  \"torch\": \"$torch_ver\","
  echo "  \"python\": \"$PYTHON\","
  echo "  \"command\": \"SCALE=$SCALE OUTPUT_ROOT=$OUTPUT_ROOT DEVICE=$DEVICE bash scripts/reproduce_all.sh\""
  echo "}"
} > "$OUTPUT_ROOT/PROVENANCE.json"

echo "== provenance  scale=$SCALE  out=$OUTPUT_ROOT  device=$DEVICE"

if [[ "$SCALE" == "workshop" ]]; then
  echo "== [1] physics dump (Prompt 1, N=2000)"
  "$PYTHON" scripts/generate_trajectories.py \
    --n-episodes 2000 --episode-length 16 --img-size 64 --seed 0 --workers 1 \
    --output "$OUTPUT_ROOT/physics_mpm"
  "$PYTHON" scripts/validate_dataset.py \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --report "$OUTPUT_ROOT/dataset_report.json"

  echo "== [2] workshop train + eval (Prompt 2, 200 epochs × vicreg/mse/pixel_ae)"
  "$PYTHON" src/train.py \
    --config configs/workshop_vicreg.yaml \
    --mode train_and_eval \
    --output-dir "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --metrics-png "$OUTPUT_ROOT/metrics.png"

  echo "== [3] within-simulator baselines (Prompt 4)"
  "$PYTHON" src/baselines.py \
    --epochs 8 --n-episodes 160 \
    --output-dir "$OUTPUT_ROOT" \
    --device cpu \
    --csv "$OUTPUT_ROOT/baseline_comparison.csv" \
    --readme "$OUTPUT_ROOT/baseline_comparison.md"

  echo "== [4] multi-seed ablations (Prompt 5)"
  "$PYTHON" src/ablation.py \
    --scale workshop \
    --seeds 42,43,44 \
    --ema-sweep 0.99,0.996,0.999 \
    --output-dir "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm"

  echo "== [5] probe eval (Prompt 3)"
  "$PYTHON" scripts/diagnose_probes.py \
    --output-dir "$OUTPUT_ROOT" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --probe-epochs 80

  echo "== [6] paper export (Prompt 5/6)"
  "$PYTHON" src/utils/export_paper_artifacts.py \
    --csv "$OUTPUT_ROOT/ablation_results.csv" \
    --multiseed-json "$OUTPUT_ROOT/ablation_multiseed.json" \
    --output-dir "$OUTPUT_ROOT" \
    --no-live-eval

  echo "== [7] qualitative rollouts (Prompt 6)"
  "$PYTHON" scripts/render_rollout_comparison.py \
    --output-dir "$OUTPUT_ROOT" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --device cpu \
    --n-episodes 4
else
  echo "== [1] tiny physics dump (smoke stand-in for Prompt 1)"
  "$PYTHON" scripts/generate_trajectories.py \
    --n-episodes 48 --episode-length 16 --img-size 64 --seed 0 --workers 1 \
    --output "$OUTPUT_ROOT/physics_mpm"
  "$PYTHON" scripts/validate_dataset.py \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --report "$OUTPUT_ROOT/dataset_report.json" \
    --min-episodes 48

  echo "== [2] smoke train + eval (Prompt 2 factory on the smoke dump)"
  "$PYTHON" src/train.py \
    --mode train_and_eval \
    --epochs 2 \
    --output-dir "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --metrics-png "$OUTPUT_ROOT/metrics.png"

  echo "== [3] within-simulator baselines (Prompt 4, shortened)"
  "$PYTHON" src/baselines.py \
    --epochs 2 --n-episodes 32 \
    --output-dir "$OUTPUT_ROOT" \
    --device cpu \
    --csv "$OUTPUT_ROOT/baseline_comparison.csv" \
    --readme "$OUTPUT_ROOT/baseline_comparison.md"

  echo "== [4] smoke ablations (Prompt 5 CLI --scale smoke, same dump)"
  "$PYTHON" src/ablation.py \
    --scale smoke \
    --epochs 2 \
    --output-dir "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --horizons 1,5,10

  echo "== [5] probe eval (Prompt 3)"
  "$PYTHON" scripts/diagnose_probes.py \
    --output-dir "$OUTPUT_ROOT" \
    --dataset-path "$OUTPUT_ROOT/physics_mpm" \
    --probe-epochs 20

  echo "== [6] paper export"
  if [[ -f "$OUTPUT_ROOT/ablation_results.csv" ]]; then
    "$PYTHON" src/utils/export_paper_artifacts.py \
      --csv "$OUTPUT_ROOT/ablation_results.csv" \
      --output-dir "$OUTPUT_ROOT" \
      --no-live-eval
  fi

  echo "== [7] qualitative rollouts (smoke checkpoint + smoke dump)"
  if [[ -f "$OUTPUT_ROOT/vicreg/last.pt" ]]; then
    "$PYTHON" scripts/render_rollout_comparison.py \
      --output-dir "$OUTPUT_ROOT" \
      --dataset-path "$OUTPUT_ROOT/physics_mpm" \
      --vicreg-ckpt "$OUTPUT_ROOT/vicreg/last.pt" \
      --device cpu \
      --n-episodes 4 \
      --decoder-steps 40 \
      --pixel-ae-steps 40 \
      --decoder-epochs 1 \
      --pixel-ae-epochs 1
  fi
fi

echo "== done  artifacts in $OUTPUT_ROOT"
echo "Diff against the audited tree with:  diff -rq outputs $OUTPUT_ROOT"
echo "Those trees will NOT match unless SCALE and every seed/device/epoch match the original run."
