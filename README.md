# FedCRMF

This is a slim PACS implementation of the current FedCRMF method:

- FedAvg baseline
- FedCRMF coordinate-response gated aggregation
- FrozenBN Tent TTA
- High-confidence pseudo-label TTA
- FedCRMF-gated TTA

Old experimental methods, old configs, and historical outputs are intentionally
not included.

## Expected PACS layout

```text
dataset/
`-- pacs/
    `-- images/
        |-- art_painting/
        |-- cartoon/
        |-- photo/
        `-- sketch/
```

The PACS split CSV files are stored in `resources/pacs_v1.0`.

## Environment

```bash
conda create -n fedcrmf python=3.8 -y
conda activate fedcrmf
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Train

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all fedcrmf
```

Run FedAvg:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all fedavg
```

## TTA

Train FedCRMF first with `scripts/run_pacs.sh`, then:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_tta.sh 42 pac_s
```

For FrozenBN Tent:

```bash
MODE=tent_frozen_bn,fedcrmf_gated_tent_frozen_bn TTA_LR=0.001 \
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_tta.sh 42 pac_s
```
