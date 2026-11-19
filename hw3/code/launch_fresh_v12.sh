#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_TAG="${RUN_TAG:-fresh_v12_$(date +%Y%m%d_%H%M%S)}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe run tag" >&2; exit 2; }
[[ -f training_data_fresh_v12/manifest.json ]] || { echo "Prepare and audit v12 data first" >&2; exit 2; }
SESSION="hw3_${RUN_TAG}"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session already exists: $SESSION" >&2; exit 2; }
mkdir -p logs runs
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && RUN_TAG='$RUN_TAG' DATA_DIR=training_data_fresh_v12 bash code/run_v12_training.sh >> 'logs/${RUN_TAG}.log' 2>&1"
printf '%s\n' "$RUN_TAG" > runs/latest_v12_run.txt
printf '%s\n' "logs/${RUN_TAG}.log" > logs/latest_v12_log.txt
echo "Started $SESSION"
echo "Log: logs/${RUN_TAG}.log"
