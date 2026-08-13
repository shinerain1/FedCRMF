# FedCRMF

This is a slim federated domain-generalization implementation for PACS,
OfficeHome, VLCS, and DomainNet:

- FedAvg baseline
- FedSR stochastic-representation baseline
- FedIIR invariant-gradient baseline
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

Run the FedDG baselines separately:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all fedsr
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all fediir
```

Or run FedSR followed by FedIIR:

```bash
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_pacs.sh 42 all feddg_baselines
```

The same method argument works with `run_officehome.sh`, `run_vlcs.sh`, and
`run_domainnet.sh`. FedIIR performs an additional pass over the selected source
clients each round to estimate its classifier-gradient reference, so it is
slower than FedAvg and FedSR.

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

FedCRMF-gated TTA uses layer-wise gate normalization without clipping by default.
Unlabeled TTA uses a deterministic 50/50 target holdout by default: the first
partition is used without labels for adaptation, and accuracy is reported only
on the disjoint evaluation partition. The default comparison contains plain,
gradient-norm-matched, and FedCRMF-gated pseudo-label TTA. It also reports
source-domain accuracy before and after adaptation on `id_test`; positive
`source_forgetting` means source accuracy decreased.

The holdout fraction and source-retention split can be changed with
`TTA_TARGET_ADAPT_FRACTION` and `TTA_SOURCE_SPLIT`. Set
`TTA_STRICT_TARGET_HOLDOUT=0` only to reproduce the legacy same-batch
transductive protocol.

Summarize OfficeHome TTA:

```bash
python scripts/summarize_officehome_tta.py --seed 42
```

For FrozenBN Tent:

```bash
MODE=tent_frozen_bn,fedcrmf_gated_tent_frozen_bn TTA_LR=0.001 \
DATASET_PATH=/root/autodl-tmp/dataset bash scripts/run_tta.sh 42 pac_s
```
