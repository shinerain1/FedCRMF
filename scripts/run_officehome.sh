#!/usr/bin/env bash
set -euo pipefail

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=4
fi

SEED="${1:-42}"
TARGET="${2:-all}"
METHOD="${3:-fedcrmf}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
SAVE_SINGLE_MODEL="${SAVE_SINGLE_MODEL:-1}"

SAVE_ARGS=()
if [[ "$SAVE_SINGLE_MODEL" == "1" ]]; then
  SAVE_ARGS+=(--save-single-model)
elif [[ "$SAVE_SINGLE_MODEL" != "0" ]]; then
  echo "SAVE_SINGLE_MODEL must be 0 or 1, got: $SAVE_SINGLE_MODEL" >&2
  exit 2
fi

mapfile -t CONFIGS < <(python scripts/make_pacs_configs.py \
  --dataset officehome \
  --seed "$SEED" \
  --target "$TARGET" \
  --method "$METHOD" \
  --dataset-path "$DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  "${SAVE_ARGS[@]}")

for config in "${CONFIGS[@]}"; do
  config="${config//$'\r'/}"
  echo "Running $config"
  python -u main.py --no_wandb --config_file "$config"
done
