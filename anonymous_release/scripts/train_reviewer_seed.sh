#!/usr/bin/env bash
# Serial corrected no-crop training into results/reviewer_strengthening/.
# Does not write into outputs/vicreg_nocrop or other original run directories.
# The paper tables use the shipped last.pt files; do not retrain to reproduce them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export MPLCONFIGDIR="$ROOT/results/reviewer_strengthening/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

SEED="${1:?seed}"
LOSS="${2:?vicreg|mse}"
ACTION="${3:?action|noaction}"
OUT_ROOT="${4:?output root under results/}"
RUN_NAME="${5:?run name}"

if [[ "$ACTION" == "noaction" ]]; then
  NA="--no-action"
else
  NA=""
fi

mkdir -p "$OUT_ROOT"
META="$OUT_ROOT/${RUN_NAME}_launch.json"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" - "$META" "$START" "$SEED" "$LOSS" "$ACTION" "$OUT_ROOT" "$RUN_NAME" <<'PY'
import json, sys, platform
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "start_time_utc": sys.argv[2],
    "seed": int(sys.argv[3]),
    "objective": sys.argv[4],
    "condition_on_action": sys.argv[5] == "action",
    "output_root": sys.argv[6],
    "run_name": sys.argv[7],
    "epochs": 200,
    "crop_scale_min": 1.0,
    "device": "mps",
    "checkpoint_policy": "last.pt final epoch",
    "os": platform.system(),
    "machine": platform.machine(),
}, indent=2) + "\n", encoding="utf-8")
PY

"$PYTHON" src/train.py \
  --config configs/workshop_vicreg.yaml \
  --mode train \
  --loss-type "$LOSS" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --device mps \
  --dataset-path outputs/physics_mpm \
  --num-workers 0 \
  --skip-baseline \
  --skip-pixel-ae \
  $NA

END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" - "$META" "$END" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["end_time_utc"] = sys.argv[2]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
echo "TRAIN_COMPLETE seed=$SEED loss=$LOSS action=$ACTION run=$RUN_NAME"
