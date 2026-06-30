#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-pac_s}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
MODE="${MODE:-pl_full_tta,fedcrmf_gated_pl_full_tta}"
TTA_LR="${TTA_LR:-0.002}"
TTA_POWER="${TTA_POWER:-1}"

python scripts/make_pacs_configs.py \
  --seed "$SEED" \
  --target "$TARGET" \
  --method fedcrmf \
  --dataset-path "$DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --save-single-model

CONFIG="configs/pacs_seed${SEED}/fedcrmf_${TARGET}_seed${SEED}.json"
CHECKPOINT="${OUTPUT_ROOT}/pacs/fedcrmf_seed${SEED}/${TARGET}/fedcrmf_${TARGET}_seed${SEED}/checkpoint/model.pt"
TTA_CONFIG="configs/pacs_seed${SEED}/fedcrmf_tta_${TARGET}_seed${SEED}.json"

python - "$CONFIG" "$TTA_CONFIG" "$CHECKPOINT" "$OUTPUT_ROOT" "$SEED" "$TARGET" "$MODE" "$TTA_LR" "$TTA_POWER" <<'PY'
import json
import sys
from pathlib import Path

src, dst, checkpoint, output_root, seed, target, modes, lr, power = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    cfg = json.load(handle)
cfg.update(
    {
        "id": f"fedcrmf_tta_{target}_seed{seed}",
        "variant_name": "fedcrmf_tta",
        "data_path": str(Path(output_root) / "pacs" / f"fedcrmf_tta_seed{seed}" / target / f"fedcrmf_tta_{target}_seed{seed}"),
        "tta_only": 1,
        "tta_eval": 1,
        "checkpoint_file": checkpoint,
        "tta_split": "test",
        "tta_modes": modes,
        "tta_param_scope": "all" if "pl_full" in modes else "bn_affine",
        "tta_optimizer": "sgd",
        "tta_conf_threshold": 0.9,
        "tta_gate_mode": "enhance",
        "tta_gate_transform": "square_norm",
        "tta_gate_power": float(power),
        "tta_rho": 1.0,
        "tta_lr": float(lr),
    }
)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(dst)
PY

python -u main.py --no_wandb --config_file "$TTA_CONFIG"
