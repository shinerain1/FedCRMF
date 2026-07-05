#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
TARGET="${2:-all}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
MODE="${MODE:-pl_full_tta,fedcrmf_gated_pl_full_tta}"
TTA_LR="${TTA_LR:-0.002}"
TTA_POWER="${TTA_POWER:-1}"
TTA_RHO="${TTA_RHO:-1.0}"
TTA_RUN_NAME="${TTA_RUN_NAME:-fedcrmf_tta}"
TTA_LABELED_PER_CLASS="${TTA_LABELED_PER_CLASS:-5}"
TTA_LABELED_ADAPT_EPOCHS="${TTA_LABELED_ADAPT_EPOCHS:-1}"
TTA_BETA="${TTA_BETA:-0}"

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
  TTA_CONFIG="configs/officehome_seed${SEED}/${TTA_RUN_NAME}_${target}_seed${SEED}.json"

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT"
    echo "Run FedCRMF training first:"
    echo "  DATASET_PATH=$DATASET_PATH bash scripts/run_officehome.sh $SEED $target fedcrmf"
    exit 1
  fi

python - "$CONFIG" "$TTA_CONFIG" "$CHECKPOINT" "$OUTPUT_ROOT" "$SEED" "$target" "$MODE" "$TTA_LR" "$TTA_POWER" "$TTA_RHO" "$TTA_RUN_NAME" "$TTA_LABELED_PER_CLASS" "$TTA_LABELED_ADAPT_EPOCHS" "$TTA_BETA" <<'PY'
import json
import sys
from pathlib import Path

(src, dst, checkpoint, output_root, seed, target, modes, lr, power, rho, run_name, labeled_per_class, labeled_adapt_epochs, beta) = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    cfg = json.load(handle)

cfg.update(
    {
        "id": f"{run_name}_{target}_seed{seed}",
        "variant_name": run_name,
        "data_path": str(
            Path(output_root)
            / "officehome"
            / f"{run_name}_seed{seed}"
            / target
            / f"{run_name}_{target}_seed{seed}"
        ),
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
        "tta_beta": float(beta),
        "tta_labeled_per_class": int(labeled_per_class),
        "tta_labeled_adapt_epochs": int(labeled_adapt_epochs),
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