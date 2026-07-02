# FedCRMF

This is a slim implementation of the current FedCRMF method for PACS and
OfficeHome:

- FedAvg baseline
- FedCRMF coordinate-response gated aggregation
- FrozenBN Tent TTA
- High-confidence pseudo-label TTA
- FedCRMF-gated TTA

Old experimental methods, old configs, and historical outputs are intentionally
not included.

## Expected dataset layout

```text
dataset/
|-- pacs/
|   `-- images/
|       |-- art_painting/
|       |-- cartoon/
|       |-- photo/
|       `-- sketch/
`-- office_home_v1.0/
    |-- images/
    `-- metadata.csv
```

The PACS split CSV files are stored in `resources/pacs_v1.0`.
OfficeHome uses `office_home_v1.0/metadata.csv`; LODO splits are constructed
inside the dataset loader.

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

Run OfficeHome:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_officehome.sh 42 all fedcrmf
```

Run FedAvg:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all fedavg
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_officehome.sh 42 all fedavg
```

Summarize training results:

```bash
python scripts/summarize_training.py --dataset officehome --method fedcrmf --seed 42
python scripts/summarize_training.py --dataset pacs --method fedcrmf --seed 42
```

## TTA

Train FedCRMF first with `scripts/run_pacs.sh`, then:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_tta.sh 42 pac_s
```

Run OfficeHome TTA after `scripts/run_officehome.sh` has saved checkpoints:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_officehome_tta.sh 42 all
```

Use layer-wise gate normalization and clip large TTA gate multipliers:

```bash
TTA_GATE_NORM_SCOPE=layer TTA_GATE_CLIP_MIN=0.5 TTA_GATE_CLIP_MAX=2.0 \
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_officehome_tta.sh 42 all
```

Summarize OfficeHome TTA:

```bash
python scripts/summarize_officehome_tta.py --seed 42
```

For FrozenBN Tent:

```bash
MODE=tent_frozen_bn,fedcrmf_gated_tent_frozen_bn TTA_LR=0.001 \
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_tta.sh 42 pac_s
```
