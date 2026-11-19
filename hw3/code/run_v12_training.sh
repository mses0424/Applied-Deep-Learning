#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_TAG="${RUN_TAG:-fresh_v12_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ta/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
DATA_DIR="${DATA_DIR:-training_data_fresh_v12}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe run tag" >&2; exit 2; }
[[ "$DATA_DIR" =~ ^training_data_[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe dataset directory" >&2; exit 2; }
[[ -f "$DATA_DIR/manifest.json" ]] || { echo "Missing fresh dataset: $DATA_DIR" >&2; exit 2; }
[[ -f results/stratified_tools_600.ids ]] || "$PYTHON_BIN" code/build_stratified_ids.py
"$PYTHON_BIN" code/check_environment.py --require-ta-exact
FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
[[ "$FREE_KB" -ge 7500000 ]] || { echo "Need at least 7.5 GB free disk" >&2; exit 2; }
mkdir -p "runs/$RUN_TAG" logs results preds
cp -- "$DATA_DIR/manifest.json" "runs/$RUN_TAG/data_manifest.json"
sha256sum "$DATA_DIR/train.jsonl" "$DATA_DIR/validation.jsonl" > "runs/$RUN_TAG/data_sha256.txt"
printf 'started %s dataset=%s\n' "$(date --iso-8601=seconds)" "$DATA_DIR" > "runs/$RUN_TAG/status.txt"
trap 'rc=$?; printf "failed rc=%s %s\n" "$rc" "$(date --iso-8601=seconds)" > "runs/$RUN_TAG/status.txt"; exit "$rc"' ERR

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC="${#GPU_ARRAY[@]}"
if (( NPROC > 1 )); then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" code/train.py \
    --train "$DATA_DIR/train.jsonl" \
    --validation "$DATA_DIR/validation.jsonl" \
    --output-dir "runs/$RUN_TAG/train" --final-adapter "runs/$RUN_TAG/final_adapter" \
    --rank 128 --alpha 192 --dropout 0.05 --max-length 3072 \
    --epochs 2.0 --batch-size 1 --gradient-accumulation 8 \
    --learning-rate 1.2e-4 --warmup-ratio 0.04 \
    --save-steps 600 --save-total-limit 4 --logging-steps 20 \
    --load-best-model-at-end
else
  CUDA_VISIBLE_DEVICES="$GPU_IDS" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" code/train.py \
    --train "$DATA_DIR/train.jsonl" \
    --validation "$DATA_DIR/validation.jsonl" \
    --output-dir "runs/$RUN_TAG/train" --final-adapter "runs/$RUN_TAG/final_adapter" \
    --rank 128 --alpha 192 --dropout 0.05 --max-length 3072 \
    --epochs 2.0 --batch-size 1 --gradient-accumulation 16 \
    --learning-rate 1.2e-4 --warmup-ratio 0.04 \
    --save-steps 600 --save-total-limit 4 --logging-steps 20 \
    --load-best-model-at-end --load-in-4bit
fi

"$PYTHON_BIN" code/validate_adapter.py "runs/$RUN_TAG/final_adapter" \
  > "runs/$RUN_TAG/adapter_validation.json"

score_value() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["headline"]["with_first_step_rule"]["all"]["weighted_step_acc"])
PY
}

TABLE="results/${RUN_TAG}_selection.tsv"
printf 'adapter\tweighted_step_accuracy\n' > "$TABLE"

evaluate_subset() {
  local label="$1"
  local adapter="$2"
  local merged="runs/$RUN_TAG/selection_merged"
  local prediction="preds/${RUN_TAG}_${label}_subset.jsonl"
  local score="results/${RUN_TAG}_${label}_subset.json"
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" eval.py generate \
    --data data/public_test.jsonl.gz --ids-file results/stratified_tools_600.ids \
    --out "$prediction" --adapter "$adapter" --merge-dir "$merged" --gpu-util 0.70
  "$PYTHON_BIN" eval.py score --data data/public_test.jsonl.gz \
    --pred "$prediction" --only-predicted --out "$score" \
    > "logs/${RUN_TAG}_${label}_subset.log"
  printf '%s\t%s\n' "$label" "$(score_value "$score")" >> "$TABLE"
  rm -rf -- "$merged"
}

evaluate_subset final "runs/$RUN_TAG/final_adapter"
while IFS= read -r checkpoint; do
  evaluate_subset "$(basename "$checkpoint")" "$checkpoint"
done < <(find "runs/$RUN_TAG/train" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

{ head -n 1 "$TABLE"; tail -n +2 "$TABLE" | sort -t $'\t' -k2,2nr; } > "${TABLE}.sorted"
mv -- "${TABLE}.sorted" "$TABLE"
cat "$TABLE"
BEST_LABEL="$(sed -n '2{s/\t.*//;p;}' "$TABLE")"
if [[ "$BEST_LABEL" == "final" ]]; then
  BEST_ADAPTER="runs/$RUN_TAG/final_adapter"
elif [[ "$BEST_LABEL" == checkpoint-* ]]; then
  BEST_ADAPTER="runs/$RUN_TAG/train/$BEST_LABEL"
else
  echo "Unexpected v12 selection label: $BEST_LABEL" >&2
  exit 2
fi
printf '%s\n' "$BEST_ADAPTER" > "runs/$RUN_TAG/selected_adapter_path.txt"
MERGED="runs/$RUN_TAG/full_eval_merged"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" eval.py generate \
  --data data/public_test.jsonl.gz --out "preds/${RUN_TAG}_tools_full.jsonl" \
  --adapter "$BEST_ADAPTER" --merge-dir "$MERGED" --gpu-util 0.70
"$PYTHON_BIN" eval.py score --data data/public_test.jsonl.gz \
  --pred "preds/${RUN_TAG}_tools_full.jsonl" \
  --out "results/${RUN_TAG}_tools_full.json"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" eval.py generate \
  --data data/public_test_negotiation.jsonl.gz \
  --out "preds/${RUN_TAG}_negotiation_full.jsonl" --model "$MERGED" --gpu-util 0.70
"$PYTHON_BIN" eval.py score --data data/public_test_negotiation.jsonl.gz \
  --pred "preds/${RUN_TAG}_negotiation_full.jsonl" \
  --out "results/${RUN_TAG}_negotiation_full.json"
"$PYTHON_BIN" code/analyze_public_errors.py \
  --tools-pred "preds/${RUN_TAG}_tools_full.jsonl" \
  --negotiation-pred "preds/${RUN_TAG}_negotiation_full.jsonl" \
  --out "results/${RUN_TAG}_error_analysis.json"
rm -rf -- "$MERGED"
printf 'completed %s\n' "$(date --iso-8601=seconds)" > "runs/$RUN_TAG/status.txt"
trap - ERR
