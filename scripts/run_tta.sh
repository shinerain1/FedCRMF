#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-pac_s}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
MODE="${MODE:-pl_full_tta,fedcrmf_gated_pl_full_tta}"
TTA_LR="${TTA_LR:-0.002}"
TTA_POWER="${TTA_POWER:-1}"
TTA_RHO="${TTA_RHO:-1.0}"
TTA_RUN_NAME="${TTA_RUN_NAME:-fedcrmf_tta}"

python scripts/make_pacs_configs.py \
  --seed "$SEED" \
  --target "$TARGET" \
  --method fedcrmf \
  --dataset-path "$DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --save-single-model

CONFIG="configs/pacs_seed${SEED}/fedcrmf_${TARGET}_seed${SEED}.json"
CHECKPOINT="${OUTPUT_ROOT}/pacs/fedcrmf_seed${SEED}/${TARGET}/fedcrmf_${TARGET}_seed${SEED}/checkpoint/model.pt"
TTA_CONFIG="configs/pacs_seed${SEED}/${TTA_RUN_NAME}_${TARGET}_seed${SEED}.json"

python - "$CONFIG" "$TTA_CONFIG" "$CHECKPOINT" "$OUTPUT_ROOT" "$SEED" "$TARGET" "$MODE" "$TTA_LR" "$TTA_POWER" "$TTA_RHO" "$TTA_RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

(src, dst, checkpoint, output_root, seed, target, modes, lr, power, rho, run_name) = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    cfg = json.load(handle)
cfg.update(
    {
        "id": f"{run_name}_{target}_seed{seed}",
        "variant_name": run_name,
        "data_path": str(Path(output_root) / "pacs" / f"{run_name}_seed{seed}" / target / f"{run_name}_{target}_seed{seed}"),
        "tta_only": 1,
        "tta_eval": 1,
        "checkpoint_file": checkpoint,
        "tta_split": "test",
        "tta_modes": modes,
        "tta_param_scope": "all" if "_full_tta" in modes else "bn_affine",
        "tta_optimizer": "sgd",
        "tta_conf_threshold": 0.9,
        "tta_gate_mode": "enhance",
        "tta_gate_transform": "square_norm",
        "tta_gate_power": float(power),
        "tta_rho": float(rho),
        "tta_lr": float(lr),
    }
)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(dst)
PY

python -u main.py --no_wandb --config_file "$TTA_CONFIG"