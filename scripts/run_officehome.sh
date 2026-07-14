#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-all}"
METHOD="${3:-fedcrmf}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

mapfile -t CONFIGS < <(python scripts/make_pacs_configs.py \
  --dataset officehome \
  --seed "$SEED" \
  --target "$TARGET" \
  --method "$METHOD" \
  --dataset-path "$DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --save-single-model)

for config in "${CONFIGS[@]}"; do
  config="${config//$'\r'/}"
  echo "Running $config"
  python -u main.py --no_wandb --config_file "$config"
done
