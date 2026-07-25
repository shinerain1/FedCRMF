import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_eval_loader

import src.datasets as my_datasets
from src.client import ERM
from src.dataset_bundle import DomainNet, OfficeHome, PACS, VLCS
from src.fedcrmf import FedCRMFServer
from src.server import FedAvg
from src.splitter import DomainBalancedSplitter, NonIIDSplitter
from src.utils import set_seed


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _collect_split_domain_ids(dataset):
    split_array = np.asarray(dataset._split_array)
    metadata_array = dataset.metadata_array
    if isinstance(metadata_array, torch.Tensor):
        metadata_array = metadata_array.detach().cpu().numpy()
    domain_index = dataset._metadata_fields.index("domain")
    split_domains = {}
    for split_name, split_id in dataset._split_dict.items():
        mask = split_array == split_id
        split_domains[split_name] = sorted(
            int(value)
            for value in np.unique(metadata_array[mask, domain_index])
        )
    return split_domains


def _validate_dataset_protocol(dataset, hparam):
    dataset_name = str(hparam.get("dataset", "PACS")).lower()
    dataset_label = {
        "officehome": "OfficeHome",
        "vlcs": "VLCS",
        "domainnet": "DomainNet",
    }.get(dataset_name, "PACS")
    configured_scheme = str(hparam.get("split_scheme", "official"))
    loaded_scheme = str(getattr(dataset, "_split_scheme", "unknown"))
    if loaded_scheme != configured_scheme:
        raise RuntimeError(
            "Dataset protocol mismatch: config split_scheme="
            f"{configured_scheme!r}, loaded split_scheme={loaded_scheme!r}."
        )

    split_domains = _collect_split_domain_ids(dataset)
    train_domains = set(split_domains.get("train", []))
    val_domains = set(split_domains.get("val", []))
    test_domains = set(split_domains.get("test", []))
    id_val_domains = set(split_domains.get("id_val", []))
    id_test_domains = set(split_domains.get("id_test", []))

    if not train_domains:
        raise RuntimeError(f"{dataset_label} protocol has no training domains.")
    if train_domains & val_domains:
        raise RuntimeError(f"{dataset_label} train and validation domains overlap.")
    if train_domains & test_domains:
        raise RuntimeError(f"{dataset_label} train and test domains overlap.")
    if dataset_name not in {"officehome", "vlcs", "domainnet"} and val_domains & test_domains:
        raise RuntimeError(f"{dataset_label} validation and test domains overlap.")
    if not id_val_domains.issubset(train_domains):
        raise RuntimeError(f"{dataset_label} id_val contains a non-source domain.")
    if not id_test_domains.issubset(train_domains):
        raise RuntimeError(f"{dataset_label} id_test contains a non-source domain.")

    strict_domain_clients = _as_bool(hparam.get("strict_domain_clients", True))
    clients_per_domain = int(hparam.get("clients_per_domain", 1))
    if clients_per_domain < 1:
        raise RuntimeError("clients_per_domain must be >= 1.")
    if strict_domain_clients and float(hparam.get("iid", 1.0)) == 0.0:
        expected = len(train_domains) * clients_per_domain
        if int(hparam.get("num_clients", 0)) != expected:
            raise RuntimeError(
                f"Strict {dataset_label} protocol requires num_clients = "
                f"num_source_domains * clients_per_domain = {expected}."
            )
    if not test_domains:
        raise RuntimeError(f"{dataset_label} experiment protocol requires a test domain.")
    if str(hparam.get("server_method", "")) in {"FedCRMF", "FedCRMFServer"}:
        if float(hparam.get("fraction", 1.0)) != 1.0:
            raise RuntimeError("FedCRMF response history requires fraction=1.0.")

    hparam["loaded_split_scheme"] = loaded_scheme
    hparam["loaded_split_domain_ids"] = {
        key: list(value) for key, value in split_domains.items()
    }
    hparam["strict_domain_clients"] = strict_domain_clients
    hparam["clients_per_domain"] = clients_per_domain
    hparam["pacs_total_domain_subclients"] = (
        len(train_domains | val_domains | test_domains) * clients_per_domain
    )
    hparam["protocol_validation_status"] = "passed"
    print(
        "[protocol] "
        f"split={loaded_scheme} train_domains={sorted(train_domains)} "
        f"val_domains={sorted(val_domains)} test_domains={sorted(test_domains)}"
    )


def _validate_client_shards(training_datasets, dataset, hparam):
    domain_index = dataset._metadata_fields.index("domain")
    client_domains = []
    client_indices = []
    for shard in training_datasets:
        metadata = shard.metadata_array
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.detach().cpu().numpy()
        client_domains.append(
            sorted(int(value) for value in np.unique(metadata[:, domain_index]))
        )
        client_indices.extend(int(value) for value in shard.indices)

    train_indices = sorted(int(value) for value in dataset.get_subset("train").indices)
    if len(client_indices) != len(set(client_indices)):
        raise RuntimeError("PACS client shards contain duplicate examples.")
    if sorted(client_indices) != train_indices:
        raise RuntimeError("PACS client shards do not exactly cover train split.")
    if _as_bool(hparam.get("strict_domain_clients", False)):
        invalid = [domains for domains in client_domains if len(domains) != 1]
        if invalid:
            raise RuntimeError(
                "Strict PACS protocol requires every client shard to contain "
                f"exactly one source domain; got {client_domains}."
            )
    hparam["loaded_client_domain_ids"] = client_domains
    print(f"[protocol] client_domain_ids={client_domains}")


def _normalize_output_path(hparam):
    data_path = os.path.normpath(hparam["data_path"])
    dataset_name = str(hparam.get("dataset", "pacs")).lower()
    dataset_dir = {
        "officehome": "officehome",
        "vlcs": "vlcs",
        "domainnet": "domainnet",
    }.get(dataset_name, "pacs")
    parts = list(Path(data_path).parts)
    outputs_idx = next((i for i, p in enumerate(parts) if p.lower() == "outputs"), -1)
    if outputs_idx != -1:
        has_dataset_level = (
            outputs_idx + 1 < len(parts)
            and parts[outputs_idx + 1].lower() == dataset_dir
        )
        if not has_dataset_level:
            parts = parts[: outputs_idx + 1] + [dataset_dir] + parts[outputs_idx + 1 :]
            data_path = os.path.join(*parts)
    hparam["data_path"] = data_path


def main(args):
    hparam = vars(args)
    with open(args.config_file, encoding="utf-8-sig") as fh:
        config = json.load(fh)
    hparam.update(config)
    hparam.update(getattr(args, "_cli_overrides", {}))
    hparam["wandb"] = not bool(args.no_wandb)

    excluded = hparam.get("eval_exclude_splits", [])
    if isinstance(excluded, str):
        excluded = [item.strip() for item in excluded.split(",") if item.strip()]
    hparam["eval_exclude_splits"] = sorted(set(excluded))
    _normalize_output_path(hparam)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(int(hparam["seed"]))
    hparam["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    hparam["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)

    if hparam["optimizer"] == "torch.optim.SGD":
        hparam["optimizer_config"] = {
            "lr": hparam["lr"],
            "momentum": hparam.get("momentum", 0.0),
            "weight_decay": hparam.get("weight_decay", 0.0),
        }
    elif hparam["optimizer"] in {"torch.optim.Adam", "torch.optim.AdamW"}:
        hparam["optimizer_config"] = {
            "lr": hparam["lr"],
            "eps": hparam.get("eps", 1e-8),
            "weight_decay": hparam.get("weight_decay", 0.0),
        }
    else:
        raise ValueError(f"Unsupported optimizer: {hparam['optimizer']}")

    dataset_name = str(hparam.get("dataset", "PACS")).lower()
    dataset_classes = {
        "pacs": (my_datasets.PACS, PACS),
        "officehome": (my_datasets.OfficeHome, OfficeHome),
        "vlcs": (my_datasets.VLCS, VLCS),
        "domainnet": (my_datasets.DomainNet, DomainNet),
    }
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {hparam.get('dataset')}")
    dataset_cls, bundle_cls = dataset_classes[dataset_name]
    dataset = dataset_cls(
        version="1.0",
        root_dir=hparam["dataset_path"],
        download=True,
        split_scheme=hparam["split_scheme"],
    )
    _validate_dataset_protocol(dataset, hparam)
    ds_bundle = bundle_cls(
        dataset,
        probabilistic=False,
        backbone=hparam.get("backbone", "resnet50"),
        freeze_bn=_as_bool(hparam.get("freeze_bn", 1)),
    )

    total_subset = dataset.get_subset("train", transform=ds_bundle.train_transform)
    testloader = {}
    for split in dataset.split_names:
        if split == "train" or split in hparam["eval_exclude_splits"]:
            continue
        ds = dataset.get_subset(split, transform=ds_bundle.test_transform)
        if len(ds) == 0:
            continue
        testloader[split] = get_eval_loader(
            loader="standard",
            dataset=ds,
            batch_size=hparam["batch_size"],
        )

    sampler = RandomSampler(total_subset, replacement=True)
    _ = DataLoader(total_subset, batch_size=hparam["batch_size"], sampler=sampler)

    num_shards = int(hparam["num_clients"])
    if num_shards == 1:
        training_datasets = [total_subset]
    elif (
        _as_bool(hparam.get("strict_domain_clients", False))
        and float(hparam.get("iid", 1.0)) == 0.0
        and int(hparam.get("clients_per_domain", 1)) > 1
    ):
        training_datasets = DomainBalancedSplitter(
            shards_per_domain=int(hparam["clients_per_domain"]),
            seed=int(hparam["seed"]),
        ).split(
            dataset.get_subset("train"),
            ds_bundle.groupby_fields,
            transform=ds_bundle.train_transform,
        )
    else:
        training_datasets = NonIIDSplitter(
            num_shards=num_shards,
            iid=hparam["iid"],
            seed=int(hparam["seed"]),
        ).split(
            dataset.get_subset("train"),
            ds_bundle.groupby_fields,
            transform=ds_bundle.train_transform,
        )
    _validate_client_shards(training_datasets, dataset, hparam)

    clients = [
        ERM(k, device, training_datasets[k], ds_bundle, hparam)
        for k in tqdm(range(num_shards), leave=False)
    ]
    server_classes = {
        "FedAvg": FedAvg,
        "FedCRMF": FedCRMFServer,
        "FedCRMFServer": FedCRMFServer,
    }
    server_method = hparam.get("server_method", "FedCRMF")
    if server_method not in server_classes:
        raise ValueError(f"Unsupported server_method: {server_method}")
    central_server = server_classes[server_method](device, ds_bundle, hparam)
    central_server.setup_model(None, 0)
    central_server.register_clients(clients)
    central_server.register_testloader(testloader)

    if _as_bool(hparam.get("tta_only", False)):
        central_server.load_single_checkpoint(hparam.get("checkpoint_file", None))
        hparam["tta_eval"] = 1
        from src.tta import run_tta_comparison

        run_tta_comparison(central_server, os.path.join(hparam["data_path"], "tta"))
    else:
        central_server.fit()

    logging.info("done")
    time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--data_path", default="./outputs/run")
    parser.add_argument("--dataset_path", default="./dataset/")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_clients", default=3, type=int)
    parser.add_argument("--clients_per_domain", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--iid", default=0.0, type=float)
    parser.add_argument("--server_method", default="FedCRMF")
    parser.add_argument("--fraction", default=1.0, type=float)
    parser.add_argument("--num_rounds", default=40, type=int)
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--freeze_bn", default=1, type=int)
    parser.add_argument("--split_scheme", default="pac-s")
    parser.add_argument("--local_epochs", default=1, type=int)
    parser.add_argument("--n_groups_per_batch", default=2, type=int)
    parser.add_argument("--optimizer", default="torch.optim.Adam")
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--hparam1", default=0.0, type=float)
    parser.add_argument("--eval_exclude_splits", default="")
    parser.add_argument("--strict_domain_clients", default=1, type=int)
    parser.add_argument("--save_single_model", default=0, type=int)
    parser.add_argument("--checkpoint_file", default="")
    parser.add_argument("--tta_only", default=0, type=int)
    parser.add_argument("--fedcrmf_history_length", default=2, type=int)
    parser.add_argument("--fedcrmf_warmup_rounds", default=1, type=int)
    parser.add_argument("--fedcrmf_mu", default=10000.0, type=float)
    parser.add_argument("--fedcrmf_alpha_mode", default="uniform")
    parser.add_argument("--fedcrmf_hist_keys", default="ALL")
    parser.add_argument("--fedcrmf_hist_max_numel", default=5000000, type=int)
    parser.add_argument("--tta_eval", default=0, type=int)
    parser.add_argument("--tta_split", default="test")
    parser.add_argument(
        "--tta_modes",
        default="pl_full_tta,fedcrmf_gated_pl_full_tta",
    )
    parser.add_argument("--tta_param_scope", default="bn_affine")
    parser.add_argument("--tta_optimizer", default="sgd")
    parser.add_argument("--tta_conf_threshold", default=0.9, type=float)
    parser.add_argument("--tta_gate_mode", default="enhance")
    parser.add_argument("--tta_gate_transform", default="square_norm")
    parser.add_argument("--tta_gate_power", default=2.0, type=float)
    parser.add_argument("--tta_rho", default=1.0, type=float)
    parser.add_argument("--tta_lr", default=1e-4, type=float)
    parser.add_argument("--tta_beta", default=0.0, type=float)
    parser.add_argument("--tta_max_batches", default=0, type=int)
    parser.add_argument("--tta_reset_each_batch", default=0, type=int)
    parser.add_argument("--tta_labeled_per_class", default=0, type=int)
    parser.add_argument("--tta_labeled_adapt_epochs", default=1, type=int)
    args = parser.parse_args()
    default_args = parser.parse_args(["--config_file", "__dummy__"])
    args._cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config_file", "_cli_overrides"}
        and value != getattr(default_args, key, None)
    }
    main(args)
