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
# Additional Federated DG Baselines

The shared experiment protocol also supports three protocol-adapted baselines:

- `fedprox`: FedAvg aggregation with the standard local proximal objective
  (`fedprox_mu=0.1`). It uses gradient accumulation over microbatches of eight
  while preserving the configured effective batch size and optimizer-step count.
- `fedomg`: the official low-dimensional FedOMG update solver with global
  learning rate `0.05`, search radius `0.5`, solver learning rate `25`, and
  `21` solver iterations. Local ERM uses microbatch gradient accumulation on
  Blackwell GPUs while preserving the effective batch and optimizer-step count.
- `fedga`: Generalization Adjustment using only each source domain's `id_val`
  split. The initial adjustment step is `0.2` and decays linearly. Its local ERM
  uses the same protocol-preserving microbatch accumulation.

All three use the repository's common ResNet50, FrozenBN, Adam, domain-client,
LODO, local-epoch, and communication-round protocol. They are adaptations to
this benchmark rather than claims of reproducing the original papers' complete
training protocols. Their Blackwell-safe path synchronizes CUDA once per
optimizer step; it does not change the optimization objective or step count.

Run all three methods for one seed on PACS or OfficeHome:

```bash
SAVE_SINGLE_MODEL=0 DATASET_PATH=/root/autodl-tmp/dataset \
  bash scripts/run_pacs.sh 0 all additional_baselines

SAVE_SINGLE_MODEL=0 DATASET_PATH=/root/autodl-tmp/dataset \
  bash scripts/run_officehome.sh 0 all additional_baselines
```
