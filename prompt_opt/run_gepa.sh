#!/bin/bash
# Run GEPA HotpotQA Prompt Optimization
# Usage: bash prompt_opt/run_gepa.sh [RUN_NUM]

RUN_NUM=${1:-1}
OUTPUT_DIR="prompt_opt/results/gepa_batchsize_10/gepa_${RUN_NUM}"
LOG_DIR="${OUTPUT_DIR}/logs"

set -e

cd "$(dirname "$0")/.."  # cd to project root

echo "=== Running GEPA HotpotQA Prompt Optimization (Run ${RUN_NUM}) ==="
echo "Model: gemini-2.5-flash-lite"
echo "Target: ${OUTPUT_DIR}"
echo ""

# Ensure directory exists so tee doesn't fail
mkdir -p "${OUTPUT_DIR}"

# Python script handles backup if folder exists
uv run python prompt_opt/gepa_opt.py \
    --num_tasks 100 \
    --num_val_tasks 100 \
    --max_metric_calls 2000 \
    --reflection_minibatch_size 10 \
    --num_threads 10 \
    --run_num "${RUN_NUM}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
