#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-all}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
MODE="${MODE:-pl_full_tta,fedcrmf_gated_pl_full_tta}"
TTA_LR="${TTA_LR:-0.002}"
TTA_POWER="${TTA_POWER:-1}"

if [ "$TARGET" = "all" ]; then
  TARGETS=(cpr_a apr_c acr_p acp_r)
else
  TARGETS=("$TARGET")
fi

for target in "${TARGETS[@]}"; do
  python scripts/make_pacs_configs.py \
    --dataset officehome \
    --seed "$SEED" \
    --target "$target" \
    --method fedcrmf \
    --dataset-path "$DATASET_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --save-single-model

  CONFIG="configs/officehome_seed${SEED}/fedcrmf_${target}_seed${SEED}.json"
  CHECKPOINT="${OUTPUT_ROOT}/officehome/fedcrmf_seed${SEED}/${target}/fedcrmf_${target}_seed${SEED}/checkpoint/model.pt"
  TTA_CONFIG="configs/officehome_seed${SEED}/fedcrmf_tta_${target}_seed${SEED}.json"

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT"
    echo "Run FedCRMF training first:"
    echo "  DATASET_PATH=$DATASET_PATH bash scripts/run_officehome.sh $SEED $target fedcrmf"
    exit 1
  fi

python - "$CONFIG" "$TTA_CONFIG" "$CHECKPOINT" "$OUTPUT_ROOT" "$SEED" "$target" "$MODE" "$TTA_LR" "$TTA_POWER" <<'PY'
import json
import sys
from pathlib import Path

(
    src,
    dst,
    checkpoint,
    output_root,
    seed,
    target,
    modes,
    lr,
    power,
) = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    cfg = json.load(handle)

cfg.update(
    {
        "id": f"fedcrmf_tta_{target}_seed{seed}",
        "variant_name": "fedcrmf_tta",
        "data_path": str(
            Path(output_root)
            / "officehome"
            / f"fedcrmf_tta_seed{seed}"
            / target
            / f"fedcrmf_tta_{target}_seed{seed}"
        ),
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
        "tta_gate_norm_scope": "layer",
        "tta_rho": 1.0,
        "tta_lr": float(lr),
    }
)

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(dst)
PY

  echo "Running TTA: $TTA_CONFIG"
  python -u main.py --no_wandb --config_file "$TTA_CONFIG"
done
