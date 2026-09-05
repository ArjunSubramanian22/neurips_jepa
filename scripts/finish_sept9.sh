#!/usr/bin/env bash
# Orchestrate remaining workshop trains. Concurrency is decided by
# scripts/hw_train_plan.py from actual CUDA/MPS/RAM — never assumed.
#
# Env:
#   WAIT_PID          live seed-42 train_and_eval PID (default 92746)
#   SKIP_SEED43=1     drop the 2-seed range; paper stays single-seed
#   SKIP_PIXEL_AE=1   SIGTERM WAIT_PID once VICReg+MSE hit 200 (do not wait for Pixel-AE)
#   FORCE_CPU_PARALLEL=1  allow CPU sidecar only if RAM ≥ 24 GiB (see hw_train_plan.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13}"
export PYTHONUNBUFFERED=1
WAIT_PID="${WAIT_PID:-92746}"
LOG="${LOG:-outputs/sept9_finish.log}"
JOBS="$ROOT/outputs/jobs"
mkdir -p outputs "$JOBS"
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

completed_epochs() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
best = 0
if path.is_file():
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("split") == "train_epoch":
                best = max(best, int(rec["epoch"]))
print(best)
PY
}

run_job() {
  local name="$1"
  shift
  rm -f "$JOBS/${name}.done" "$JOBS/${name}.fail" "$JOBS/${name}.pid"
  echo "== launch job $name: $*"
  (
    set +e
    "$@" >"$JOBS/${name}.log" 2>&1
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" >"$JOBS/${name}.done"
    else
      echo "$rc" >"$JOBS/${name}.fail"
    fi
    exit "$rc"
  ) &
  echo $! >"$JOBS/${name}.pid"
}

wait_jobs() {
  local name pid
  for name in "$@"; do
    echo "== wait job $name"
    while true; do
      if [[ -f "$JOBS/${name}.done" ]]; then
        echo "== job $name done"
        break
      fi
      if [[ -f "$JOBS/${name}.fail" ]]; then
        echo "FATAL: job $name failed rc=$(cat "$JOBS/${name}.fail"). Log: $JOBS/${name}.log"
        tail -50 "$JOBS/${name}.log" || true
        exit 1
      fi
      if [[ -f "$JOBS/${name}.pid" ]]; then
        pid="$(cat "$JOBS/${name}.pid")"
        if ! kill -0 "$pid" 2>/dev/null; then
          sleep 2
          if [[ -f "$JOBS/${name}.done" ]]; then
            break
          fi
          echo "FATAL: job $name PID $pid exited without a done marker."
          tail -50 "$JOBS/${name}.log" || true
          exit 1
        fi
      fi
      sleep 20
    done
  done
}

milestone_paper() {
  local tag="$1"
  local commit_msg="$2"
  echo "== milestone $tag $(date)"
  git_commit "$commit_msg"
  echo "== paper/sync_tables.py after $tag"
  "$PYTHON" paper/sync_tables.py
  set +e
  "$PYTHON" paper/check_claims.py
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FATAL: paper/check_claims.py failed after $tag (exit $rc). Stopping; not deleting the check or papering over it."
    exit "$rc"
  fi
  git_commit "Sync paper macros after ${tag}"
}

echo "== finish_sept9 start $(date)  wait_pid=$WAIT_PID"
"$PYTHON" scripts/hw_train_plan.py
PARALLEL42="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["parallel_seed42_mse_with_vicreg"] and "1" or "0")')"
PARALLEL43="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["parallel_seed43_vicreg_mse"] and "1" or "0")')"
DEV42MSE="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["seed42_mse_device"])')"
DEV43VIC="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["seed43_vic_device"])')"
DEV43MSE="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["seed43_mse_device"])')"
SKIP43="$("$PYTHON" -c 'import json; print("1" if json.load(open("outputs/hw_train_plan.json"))["skip_seed43"] else "0")')"
SKIPPAE="$("$PYTHON" -c 'import json; print("1" if json.load(open("outputs/hw_train_plan.json"))["skip_pixel_ae"] else "0")')"
NACCEL="$("$PYTHON" -c 'import json; print(json.load(open("outputs/hw_train_plan.json"))["n_accelerators"])')"
echo "== plan parallel42=$PARALLEL42 parallel43=$PARALLEL43 n_accel=$NACCEL skip43=$SKIP43 skip_pixel_ae=$SKIPPAE"

# --- seed-42 VICReg tail + optional concurrent MSE on a second device ---
# The live WAIT_PID is --mode train_and_eval, which itself writes
# outputs/mse/metrics.jsonl after VICReg. A second MSE job into that
# file would violate the one-writer rule. Only pair MSE with VICReg when
# that PID is already gone (or never started).
if [[ "$PARALLEL42" == "1" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
  mse_ep="$(completed_epochs outputs/mse/metrics.jsonl)"
  if [[ "$mse_ep" -lt 200 ]]; then
    run_job seed42_mse $PYTHON src/train.py \
      --config configs/workshop_vicreg.yaml \
      --mode train \
      --loss-type mse \
      --seed 42 \
      --output-dir outputs \
      --device "$DEV42MSE" \
      --dataset-path outputs/physics_mpm \
      --skip-baseline \
      --skip-pixel-ae
  fi
elif [[ "$PARALLEL42" == "1" ]]; then
  echo "== not pairing seed-42 MSE with live PID $WAIT_PID (train_and_eval owns outputs/mse)"
fi

if kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "== waiting for live train PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    vic_ep="$(completed_epochs outputs/vicreg/metrics.jsonl)"
    mse_ep="$(completed_epochs outputs/mse/metrics.jsonl)"
    echo "== still waiting PID $WAIT_PID  $(date)  vicreg=$vic_ep mse=$mse_ep"
    if [[ "$SKIPPAE" == "1" && "$vic_ep" -ge 200 && "$mse_ep" -ge 200 ]]; then
      echo "== SKIP_PIXEL_AE: VICReg+MSE at 200; SIGTERM $WAIT_PID"
      kill -TERM "$WAIT_PID" 2>/dev/null || true
      break
    fi
    sleep 60
  done
fi
echo "== live train PID $WAIT_PID is gone at $(date)"

if [[ "$PARALLEL42" == "1" && -f "$JOBS/seed42_mse.pid" ]]; then
  wait_jobs seed42_mse
fi

vic_ep="$(completed_epochs outputs/vicreg/metrics.jsonl)"
mse_ep="$(completed_epochs outputs/mse/metrics.jsonl)"
echo "vicreg_completed_epochs=$vic_ep mse_completed_epochs=$mse_ep"
if [[ "$vic_ep" -lt 200 ]]; then
  echo "FATAL: VICReg did not reach 200 epochs (got $vic_ep); not auto-resuming a mid-run job."
  exit 1
fi
if [[ "$mse_ep" -lt 200 ]]; then
  echo "== seed-42 MSE incomplete; training on $DEV42MSE $(date)"
  run_job seed42_mse $PYTHON src/train.py \
    --config configs/workshop_vicreg.yaml \
    --mode train \
    --loss-type mse \
    --seed 42 \
    --output-dir outputs \
    --device "$DEV42MSE" \
    --dataset-path outputs/physics_mpm \
    --skip-baseline \
    --skip-pixel-ae
  wait_jobs seed42_mse
  mse_ep="$(completed_epochs outputs/mse/metrics.jsonl)"
  if [[ "$mse_ep" -lt 200 ]]; then
    echo "FATAL: seed-42 MSE still at $mse_ep after job."
    exit 1
  fi
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" >"$JOBS/seed42_vicreg.done"
date -u +"%Y-%m-%dT%H:%M:%SZ" >"$JOBS/seed42_mse.done"
milestone_paper seed42-ckpts "$(cat <<'EOF'
Record seed-42 200-epoch VICReg and MSE workshop checkpoints

EOF
)"

# --- seed 43: parallel only if hw_train_plan said two accelerators (or CPU sidecar) ---
if [[ "$SKIP43" == "1" ]]; then
  echo "== SKIP_SEED43=1: not training seed 43"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$JOBS/seed43_skipped.done"
else
  run_job seed43_vicreg $PYTHON src/train.py \
    --config configs/workshop_vicreg.yaml \
    --mode train \
    --loss-type vicreg \
    --seed 43 \
    --output-dir outputs/seed43 \
    --device "$DEV43VIC" \
    --dataset-path outputs/physics_mpm \
    --skip-baseline \
    --skip-pixel-ae
  if [[ "$PARALLEL43" == "1" ]]; then
    run_job seed43_mse $PYTHON src/train.py \
      --config configs/workshop_vicreg.yaml \
      --mode train \
      --loss-type mse \
      --seed 43 \
      --output-dir outputs/seed43 \
      --device "$DEV43MSE" \
      --dataset-path outputs/physics_mpm \
      --skip-baseline \
      --skip-pixel-ae
    wait_jobs seed43_vicreg seed43_mse
  else
    wait_jobs seed43_vicreg
    run_job seed43_mse $PYTHON src/train.py \
      --config configs/workshop_vicreg.yaml \
      --mode train \
      --loss-type mse \
      --seed 43 \
      --output-dir outputs/seed43 \
      --device "$DEV43MSE" \
      --dataset-path outputs/physics_mpm \
      --skip-baseline \
      --skip-pixel-ae
    wait_jobs seed43_mse
  fi
  milestone_paper seed43-ckpts "$(cat <<'EOF'
Record seed-43 200-epoch VICReg and MSE workshop runs

EOF
)"
fi

echo "== probes + effective rank on final seed-42 ckpts $(date)"
run_job probes $PYTHON scripts/diagnose_probes.py \
  --output-dir outputs \
  --dataset-path outputs/physics_mpm \
  --physics-only \
  --probe-epochs 80 \
  --require-final
wait_jobs probes

if [[ "$SKIP43" != "1" ]]; then
  echo "== two-seed range table $(date)"
  run_job twoseed $PYTHON scripts/two_seed_range.py \
    --output-dir outputs \
    --dataset-path outputs/physics_mpm \
    --device cpu
  wait_jobs twoseed
else
  echo "== SKIP_SEED43: not running two_seed_range.py"
fi

PIXEL_AE=""
if [[ -f outputs/pixel_ae/last.pt ]]; then
  PIXEL_AE="--pixel-ae-ckpt outputs/pixel_ae/last.pt"
fi
echo "== regenerate rollout figures from final ckpts $(date)"
# shellcheck disable=SC2086
run_job figures $PYTHON scripts/render_rollout_comparison.py \
  --output-dir outputs \
  --dataset-path outputs/physics_mpm \
  --vicreg-ckpt outputs/vicreg/best.pt \
  --device cpu \
  --n-episodes 4 \
  $PIXEL_AE
wait_jobs figures

milestone_paper eval-artifacts "$(cat <<'EOF'
Regenerate probes, 2-seed range, and rollout figures from final 200-epoch checkpoints

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
echo "== hw plan: $(cat outputs/hw_train_plan.json)"
