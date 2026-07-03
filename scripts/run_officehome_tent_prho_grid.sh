#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-1}"
TARGET="${2:-all}"

DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
TTA_LR="${TTA_LR:-0.001}"
POWERS="${POWERS:-1,2,4,6}"
RHOS="${RHOS:-0,0.25,0.5,0.75,1.0}"

if [ "$TARGET" = "all" ]; then
  TARGETS=(cpr_a apr_c acr_p acp_r)
else
  TARGETS=("$TARGET")
fi

IFS=',' read -ra POWER_LIST <<< "$POWERS"
IFS=',' read -ra RHO_LIST <<< "$RHOS"

for power in "${POWER_LIST[@]}"; do
  for rho in "${RHO_LIST[@]}"; do
    power_name="${power//./p}"
    rho_name="${rho//./p}"
    lr_name="${TTA_LR//./p}"
    run_name="tent_prho_grid_lr${lr_name}_p${power_name}_rho${rho_name}"

    for target in "${TARGETS[@]}"; do
      python scripts/make_pacs_configs.py \
        --dataset officehome \
        --seed "$SEED" \
        --target "$target" \
        --method fedcrmf \
        --dataset-path "$DATASET_PATH" \
        --output-root "$OUTPUT_ROOT" \
        --save-single-model

      src="configs/officehome_seed${SEED}/fedcrmf_${target}_seed${SEED}.json"
      dst="configs/officehome_seed${SEED}/${run_name}_${target}_seed${SEED}.json"
      ckpt="${OUTPUT_ROOT}/officehome/fedcrmf_seed${SEED}/${target}/fedcrmf_${target}_seed${SEED}/checkpoint/model.pt"

      if [ ! -f "$ckpt" ]; then
        echo "Missing checkpoint: $ckpt"
        echo "Run FedCRMF training first:"
        echo "  DATASET_PATH=$DATASET_PATH bash scripts/run_officehome.sh $SEED $target fedcrmf"
        exit 1
      fi

      python - "$src" "$dst" "$ckpt" "$OUTPUT_ROOT" "$SEED" "$target" "$run_name" "$TTA_LR" "$power" "$rho" <<'PY'
import json
import sys
from pathlib import Path

src, dst, ckpt, output_root, seed, target, run_name, lr, power, rho = sys.argv[1:]

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
        "checkpoint_file": ckpt,
        "tta_split": "test",
        "tta_modes": "tent_frozen_bn,fedcrmf_gated_tent_frozen_bn",
        "tta_param_scope": "bn_affine",
        "tta_optimizer": "sgd",
        "tta_gate_mode": "enhance",
        "tta_gate_transform": "square_norm",
        "tta_gate_power": float(power),
        "tta_rho": float(rho),
        "tta_lr": float(lr),
    }
)

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

print(dst)
PY

      echo "=== Running target=$target lr=$TTA_LR p=$power rho=$rho run=$run_name ==="
      python -u main.py --no_wandb --config_file "$dst"
    done
  done
done
