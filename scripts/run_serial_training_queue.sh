#!/usr/bin/env bash
# Serial MPS training queue. One ViT at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/reviewer_strengthening/05_more_seeds/serial_train.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "QUEUE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash scripts/train_reviewer_seed.sh 44 vicreg action results/reviewer_strengthening/05_more_seeds vicreg_action_s44
  bash scripts/train_reviewer_seed.sh 45 vicreg action results/reviewer_strengthening/05_more_seeds vicreg_action_s45
  bash scripts/train_reviewer_seed.sh 42 mse action results/reviewer_strengthening/06_corrected_objective mse_action_s42
  bash scripts/train_reviewer_seed.sh 43 mse action results/reviewer_strengthening/06_corrected_objective mse_action_s43
  bash scripts/train_reviewer_seed.sh 44 vicreg noaction results/reviewer_strengthening/05_more_seeds vicreg_noaction_s44
  bash scripts/train_reviewer_seed.sh 45 vicreg noaction results/reviewer_strengthening/05_more_seeds vicreg_noaction_s45
  echo "QUEUE_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"
