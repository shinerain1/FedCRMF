#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-all}"
METHOD="${3:-fedcrmf}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

python scripts/make_pacs_configs.py \
  --seed "$SEED" \
  --target "$TARGET" \
  --method "$METHOD" \
  --dataset-path "$DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --save-single-model

CONFIG_DIR="configs/pacs_seed${SEED}"
for config in "$CONFIG_DIR"/*.json; do
  echo "Running $config"
  python -u main.py --no_wandb --config_file "$config"
done
