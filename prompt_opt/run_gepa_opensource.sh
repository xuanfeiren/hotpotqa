#!/bin/bash
# Run GEPA HotpotQA Prompt Optimization with open-source reflection model
# Usage: bash prompt_opt/run_gepa_opensource.sh [RUN_NUM]

RUN_NUM=${1:-1}
REFLECTION_MODEL="gpt-oss-20b"
OUTPUT_DIR="prompt_opt/results/gepa_${REFLECTION_MODEL}/gepa_${RUN_NUM}"
LOG_DIR="${OUTPUT_DIR}/logs"
REFLECTION_API_BASE="http://127.0.0.1:8000/v1"

set -e

cd "$(dirname "$0")/.."  # cd to project root

echo "=== Running GEPA HotpotQA Prompt Optimization (Run ${RUN_NUM}) ==="
echo "Task model: gemini-2.5-flash-lite"
echo "Reflection model: ${REFLECTION_MODEL} @ ${REFLECTION_API_BASE}"
echo "Target: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

uv run python prompt_opt/gepa_opt.py \
    --num_tasks 100 \
    --num_val_tasks 100 \
    --max_metric_calls 2000 \
    --reflection_minibatch_size 10 \
    --num_threads 10 \
    --run_num "${RUN_NUM}" \
    --reflection_model "${REFLECTION_MODEL}" \
    --reflection_api_base "${REFLECTION_API_BASE}" \
    --reflection_api_key "EMPTY" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"

echo ""
echo "=== Pushing results to git ==="
git add "${OUTPUT_DIR}"
git commit -m "GEPA ${REFLECTION_MODEL} run ${RUN_NUM} results"
git pull --rebase
git push
