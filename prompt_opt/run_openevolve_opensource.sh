#!/bin/bash
# Run OpenEvolve prompt optimization for HotpotQA with open-source model
# Usage: bash prompt_opt/run_openevolve_opensource.sh [RUN_NUM]

RUN_NUM=${1:-1}
MODEL="gpt-oss-20b"
export OPENEVOLVE_OUTPUT_DIR="prompt_opt/results/openevolve_${MODEL}/openevolve_${RUN_NUM}"

set -e

cd "$(dirname "$0")/.."  # cd to project root

echo "=== Running OpenEvolve HotpotQA Prompt Optimization (Run ${RUN_NUM}) ==="
echo "Model: ${MODEL} @ http://127.0.0.1:8000/v1"
echo "Target: ${OPENEVOLVE_OUTPUT_DIR}"
echo ""

# Ensure fresh start: backup instead of delete
if [ -d "${OPENEVOLVE_OUTPUT_DIR}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="${OPENEVOLVE_OUTPUT_DIR}_bak_${TIMESTAMP}"
    echo "Backing up existing output directory to: ${BACKUP_DIR}"
    mv "${OPENEVOLVE_OUTPUT_DIR}" "${BACKUP_DIR}"
fi

mkdir -p "${OPENEVOLVE_OUTPUT_DIR}/logs"

uv run python ../openevolve/openevolve-run.py \
    prompt_opt/openevolve_initial_prompt.txt \
    prompt_opt/openevolve_opt.py \
    --config prompt_opt/openevolve_config.yaml \
    --output "${OPENEVOLVE_OUTPUT_DIR}" \
    --iterations 20 2>&1 | tee "${OPENEVOLVE_OUTPUT_DIR}/run.log"

echo ""
echo "=== Pushing results to git ==="
git add -f "${OPENEVOLVE_OUTPUT_DIR}"
git commit -m "OpenEvolve ${MODEL} run ${RUN_NUM} results"
git pull --rebase
git push
