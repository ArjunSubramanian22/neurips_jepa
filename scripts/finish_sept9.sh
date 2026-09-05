#!/usr/bin/env bash
# After the live seed-42 train_and_eval exits: verify 200-epoch VICReg+MSE,
# train seed 43, run probes/rank, regenerate figures, sync the paper, commit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13}"
WAIT_PID="${WAIT_PID:-92746}"
LOG="${LOG:-outputs/sept9_finish.log}"
mkdir -p outputs
exec >>"$LOG" 2>&1

git_commit() {
  local msg="$1"
  git add -A
  if git diff --cached --quiet; then
    echo "== skip commit (empty): $msg"
    return 0
  fi
  git commit -m "$msg"
  echo "== committed: $msg  $(git rev-parse --short HEAD)"
}

echo "== finish_sept9 start $(date)  wait_pid=$WAIT_PID"

if kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "== waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    echo "== still waiting $(date)"
    sleep 60
  done
fi
echo "== train PID $WAIT_PID is gone at $(date)"

check_epochs() {
  "$PYTHON" - <<'PY'
import json, sys
from pathlib import Path

def max_epoch(path):
    best = 0
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("split") == "train_epoch":
                best = max(best, int(rec["epoch"]))
    return best

vic = max_epoch("outputs/vicreg/metrics.jsonl")
mse = max_epoch("outputs/mse/metrics.jsonl")
print(f"vicreg_completed_epochs={vic} mse_completed_epochs={mse}")
if vic < 200:
    sys.exit(2)
if mse < 200:
    sys.exit(3)
PY
}

set +e
check_epochs
epoch_rc=$?
set -e
if [[ "$epoch_rc" -eq 2 ]]; then
  echo "VICReg did not reach 200 epochs; not auto-resuming a mid-run job."
  exit 1
fi
if [[ "$epoch_rc" -eq 3 ]]; then
  echo "== seed-42 MSE incomplete; training workshop MSE now $(date)"
  "$PYTHON" src/train.py \
    --config configs/workshop_vicreg.yaml \
    --mode train \
    --loss-type mse \
    --seed 42 \
    --output-dir outputs \
    --device auto \
    --dataset-path outputs/physics_mpm \
    --skip-baseline \
    --skip-pixel-ae
  check_epochs
fi

git_commit "$(cat <<'EOF'
Record seed-42 200-epoch VICReg and MSE workshop checkpoints

EOF
)"

echo "== seed 43 VICReg $(date)"
"$PYTHON" src/train.py \
  --config configs/workshop_vicreg.yaml \
  --mode train \
  --loss-type vicreg \
  --seed 43 \
  --output-dir outputs/seed43 \
  --device auto \
  --dataset-path outputs/physics_mpm \
  --skip-baseline \
  --skip-pixel-ae

echo "== seed 43 MSE $(date)"
"$PYTHON" src/train.py \
  --config configs/workshop_vicreg.yaml \
  --mode train \
  --loss-type mse \
  --seed 43 \
  --output-dir outputs/seed43 \
  --device auto \
  --dataset-path outputs/physics_mpm \
  --skip-baseline \
  --skip-pixel-ae

git_commit "$(cat <<'EOF'
Record seed-43 200-epoch VICReg and MSE workshop runs

EOF
)"

echo "== probes + effective rank on final seed-42 ckpts $(date)"
"$PYTHON" scripts/diagnose_probes.py \
  --output-dir outputs \
  --dataset-path outputs/physics_mpm \
  --physics-only \
  --probe-epochs 80

echo "== two-seed range table $(date)"
"$PYTHON" scripts/two_seed_range.py \
  --output-dir outputs \
  --dataset-path outputs/physics_mpm \
  --device cpu

PIXEL_AE=""
if [[ -f outputs/pixel_ae/last.pt ]]; then
  PIXEL_AE="--pixel-ae-ckpt outputs/pixel_ae/last.pt"
fi
echo "== regenerate rollout figures from final ckpts $(date)"
"$PYTHON" scripts/render_rollout_comparison.py \
  --output-dir outputs \
  --dataset-path outputs/physics_mpm \
  --vicreg-ckpt outputs/vicreg/best.pt \
  --device cpu \
  --n-episodes 4 \
  $PIXEL_AE

git_commit "$(cat <<'EOF'
Regenerate probes, 2-seed range, and rollout figures from final 200-epoch checkpoints

EOF
)"

echo "== paper sync $(date)"
"$PYTHON" paper/sync_tables.py
"$PYTHON" paper/check_claims.py

git_commit "$(cat <<'EOF'
Sync paper macros and tables to finished 200-epoch artifacts

EOF
)"

FREEZE=$(git rev-parse HEAD)
FREEZE_SHORT=$(git rev-parse --short HEAD)
MAPPING_FREEZE="$FREEZE" MAPPING_SHORT="$FREEZE_SHORT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

head = os.environ["MAPPING_FREEZE"]
short = os.environ["MAPPING_SHORT"]
root = Path("MAPPING.md")
old = root.read_text(encoding="utf-8").splitlines()
out = []
replaced_head = replaced_tree = False
bt = "`"
for line in old:
    if line.startswith("**Git HEAD:**"):
        out.append(
            f"**Git HEAD:** {bt}{head}{bt} ({bt}{short}{bt}). Paper macros, probes, figures, "
            "and 200-epoch VICReg/MSE checkpoints are in this commit."
        )
        replaced_head = True
        continue
    if line.startswith("**Working tree:**"):
        out.append(
            "**Working tree:** clean after finish_sept9 freeze (this file may be one commit later). "
            f"{bt}outputs/physics_mpm/observations.npy{bt} remains gitignored (GitHub 100 MB limit); "
            f"regenerate with {bt}scripts/generate_trajectories.py{bt}."
        )
        replaced_tree = True
        continue
    out.append(line)
if not replaced_head or not replaced_tree:
    raise SystemExit("MAPPING.md Git HEAD / Working tree lines not found")
root.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"updated MAPPING.md git HEAD -> {head}")
PY

git_commit "$(cat <<EOF
Record artifact-freeze git HEAD ${FREEZE_SHORT} in MAPPING.md

EOF
)"

echo "== finish_sept9 done $(date)  HEAD=$(git rev-parse --short HEAD)"
echo "== freeze cited in MAPPING.md: $FREEZE_SHORT"
