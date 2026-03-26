#!/bin/bash
# Run Trace PrioritySearch optimization for HotpotQA
# Usage: bash prompt_opt/run_trace.sh [RUN_NUM]

RUN_NUM=${1:-1}
OUTPUT_DIR="prompt_opt/results/trace_${RUN_NUM}"

set -e

cd "$(dirname "$0")/.."  # cd to project root

echo "=== Running Trace HotpotQA Prompt Optimization (Run ${RUN_NUM}) ==="
echo "Model: gemini-2.5-flash-lite"
echo "Target: ${OUTPUT_DIR}"
echo ""

# Ensure directory exists so tee doesn't fail
mkdir -p "${OUTPUT_DIR}"

# Python script handles backup if folder exists
uv run python prompt_opt/trace_opt.py \
    --num_train_samples 100 \
    --num_test_samples 100 \
    --num_candidates 5 \
    --batch_size 2 \
    --num_batches 1 \
    --num_steps 100 \
    --num_threads 10 \
    --num_eval_samples 5 \
    --run_num "${RUN_NUM}" \
    --log_frequency 1 \
    --algorithm PS_epsNet_Summarizer \
    --epsilon 0.1 \
    --project_name hotpotqa_opensource \
    --run_name run_${RUN_NUM}_$(date +%m%d_%H%M) \
    --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
