#!/usr/bin/env bash
# Sequential matched seed-43 trains (action, then no-action) + audit eval.
# Does not write into seed-42 run dirs or paper artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13}"
export PYTHONUNBUFFERED=1

LOG_DIR="$ROOT/outputs/jobs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/corrected_seed43_pipeline.log"

SEED42_AC="$ROOT/outputs/vicreg_nocrop"
SEED42_NA="$ROOT/outputs/no_action"
SEED43_AC="$ROOT/outputs/vicreg_nocrop_seed43"
SEED43_NA="$ROOT/outputs/vicreg_nocrop_noaction_seed43"

forbid_overwrite() {
  local path="$1"
  if [[ ! -f "$path/last.pt" ]]; then
    echo "FATAL: missing protected checkpoint $path/last.pt"
    exit 1
  fi
}

epoch_of() {
  local ckpt="$1"
  "$PYTHON" - <<PY
from src.train import peek_checkpoint_epoch
print(peek_checkpoint_epoch("$ckpt"))
PY
}

train_if_needed() {
  local run_name="$1"
  local extra_args="$2"
  local out_dir="$ROOT/outputs/$run_name"
  if [[ -f "$out_dir/last.pt" ]]; then
    local ep
    ep="$(epoch_of "$out_dir/last.pt")"
    if [[ "$ep" -ge 200 ]]; then
      echo "$(date) skip train $run_name (last.pt epoch=$ep)"
      return 0
    fi
    echo "FATAL: incomplete $out_dir/last.pt epoch=$ep; trainer has no resume. Refusing to wipe metrics."
    exit 1
  fi
  if [[ -e "$SEED42_AC" && "$out_dir" -ef "$SEED42_AC" ]]; then
    echo "FATAL: refusing to overwrite seed-42 action dir"
    exit 1
  fi
  if [[ -e "$SEED42_NA" && "$out_dir" -ef "$SEED42_NA" ]]; then
    echo "FATAL: refusing to overwrite seed-42 no-action dir"
    exit 1
  fi
  echo "$(date) START train $run_name"
  # shellcheck disable=SC2086
  "$PYTHON" src/train.py \
    --config configs/workshop_vicreg.yaml \
    --mode train \
    --loss-type vicreg \
    --seed 43 \
    --output-dir outputs \
    --run-name "$run_name" \
    --device mps \
    --dataset-path outputs/physics_mpm \
    --skip-baseline \
    --skip-pixel-ae \
    $extra_args
  echo "$(date) FINISHED train $run_name"
}

exec > >(tee -a "$LOG") 2>&1
echo "===== corrected seed-43 pipeline $(date) PYTHON=$PYTHON ====="
forbid_overwrite "$SEED42_AC"
forbid_overwrite "$SEED42_NA"

  echo "Config assert (seed-42 reconstructed CLI vs last.pt; seed-43 vs seed-42)..."
  "$PYTHON" - <<'PY'
from pathlib import Path
import torch
from src.config import load_experiment_config
from src.train import apply_dataset_path

DUMP = "outputs/physics_mpm"
YAML = "configs/workshop_vicreg.yaml"
ALLOWED = {"training.seed"}

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

def resolve(seed, no_action):
    cfg = load_experiment_config(YAML)
    cfg.training.output_dir = "outputs"
    cfg.training.device = "mps"
    cfg.training.seed = seed
    cfg.training.loss_type = "vicreg"
    if no_action:
        cfg.model.condition_on_action = False
    cfg = apply_dataset_path(cfg, DUMP, num_workers=2)
    cfg.training.device = "mps"
    cfg.training.output_dir = "outputs"
    cfg.training.seed = seed
    cfg.training.loss_type = "vicreg"
    return cfg.to_dict()

pairs = [
    ("action", Path("outputs/vicreg_nocrop/last.pt"), False),
    ("no_action", Path("outputs/no_action/last.pt"), True),
]
for name, ckpt, no_action in pairs:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert int(payload["epoch"]) == 200, (name, payload["epoch"])
    flat42 = flatten(payload["config"])
    rec42 = flatten(resolve(42, no_action))
    rec43 = flatten(resolve(43, no_action))
    rec_vs_ckpt = {k for k in set(flat42) | set(rec42) if flat42.get(k) != rec42.get(k)}
    d43 = {k for k in set(flat42) | set(rec43) if flat42.get(k) != rec43.get(k)}
    if rec_vs_ckpt:
        raise SystemExit(f"{name}: reconstructed seed-42 CLI != checkpoint: {rec_vs_ckpt}")
    if d43 != ALLOWED:
        raise SystemExit(f"{name}: seed-43 diffs {d43} != {ALLOWED}")
    print(f"ASSERT_OK {name} only_diff={sorted(d43)}")
PY

  train_if_needed "vicreg_nocrop_seed43" ""
  train_if_needed "vicreg_nocrop_noaction_seed43" "--no-action"

echo "$(date) START audit eval (CPU, last.pt only)"
"$PYTHON" scripts/corrected_seed43_audit.py --device cpu --output outputs/corrected_seed43_audit.json
echo "$(date) PIPELINE_COMPLETE"
