import argparse
import json
from pathlib import Path


DATASETS = {
    "pacs": {
        "display_name": "PACS",
        "targets": {
            "acs_p": {"scheme": "acs-p", "code": "P", "sources": ["A", "C", "S"]},
            "pcs_a": {"scheme": "pcs-a", "code": "A", "sources": ["P", "C", "S"]},
            "pac_s": {"scheme": "pac-s", "code": "S", "sources": ["P", "A", "C"]},
            "pas_c": {"scheme": "pas-c", "code": "C", "sources": ["P", "A", "S"]},
        },
        "best_params": {
            "P": {"history_length": 1, "mu": 10000},
            "A": {"history_length": 3, "mu": 10000},
            "S": {"history_length": 3, "mu": 10000},
            "C": {"history_length": 2, "mu": 10000},
        },
    },
    "officehome": {
        "display_name": "OfficeHome",
        "targets": {
            "cpr_a": {"scheme": "cpr-a", "code": "A", "sources": ["C", "P", "R"]},
            "apr_c": {"scheme": "apr-c", "code": "C", "sources": ["A", "P", "R"]},
            "acr_p": {"scheme": "acr-p", "code": "P", "sources": ["A", "C", "R"]},
            "acp_r": {"scheme": "acp-r", "code": "R", "sources": ["A", "C", "P"]},
        },
        # OfficeHome is newly added; use a conservative untuned default.
        "best_params": {
            "A": {"history_length": 2, "mu": 10000},
            "C": {"history_length": 2, "mu": 10000},
            "P": {"history_length": 2, "mu": 10000},
            "R": {"history_length": 2, "mu": 10000},
        },
    },
}

RESPONSE_GATE_DROPOUT_ABLATIONS = {
    "fedcrmf_rgd_l1_p0p2": {"history_length": 1, "dropout_p": 0.2},
    "fedcrmf_rgd_l1_p0p5": {"history_length": 1, "dropout_p": 0.5},
    "fedcrmf_rgd_l3_p0p5": {"history_length": 3, "dropout_p": 0.5},
}


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pacs", choices=sorted(DATASETS))
    parser.add_argument("--seed", default="42")
    parser.add_argument("--target", default="all")
    parser.add_argument(
        "--method",
        default="fedcrmf",
        choices=[
            "fedavg",
            "fedcrmf",
            "fedcrmf_uniform_shrinkage",
            "fedcrmf_one_round",
            "fedcrmf_permuted_gate",
            "fedcrmf_no_centering",
            "fedcrmf_response_gate_dropout",
            *RESPONSE_GATE_DROPOUT_ABLATIONS,
            "response_gate_dropout_ablation",
            "all",
            "ablation_core",
        ],
    )
    parser.add_argument("--gate-dropout-p", default=0.5, type=float)
    parser.add_argument("--dataset-path", default="./dataset")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--config-root", default="./configs")
    parser.add_argument("--clients-per-domain", default=1, type=int)
    parser.add_argument("--num-rounds", default=40, type=int)
    parser.add_argument("--save-single-model", action="store_true")
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    dataset_meta = DATASETS[dataset_key]
    all_targets = dataset_meta["targets"]
    if args.target == "all":
        targets = list(all_targets.keys())
    else:
        if args.target not in all_targets:
            valid = ", ".join(["all", *all_targets.keys()])
            raise ValueError(f"Unsupported target={args.target!r} for {dataset_key}. Valid: {valid}")
        targets = [args.target]

    seed = int(args.seed)
    if args.method == "all":
        methods = ["fedavg", "fedcrmf"]
    elif args.method == "ablation_core":
        methods = [
            "fedcrmf_uniform_shrinkage",
            "fedcrmf_one_round",
            "fedcrmf_permuted_gate",
            "fedcrmf_no_centering",
        ]
    elif args.method == "response_gate_dropout_ablation":
        methods = list(RESPONSE_GATE_DROPOUT_ABLATIONS)
    else:
        methods = [args.method]
    for target in targets:
        meta = all_targets[target]
        num_clients = len(meta["sources"]) * int(args.clients_per_domain)
        for method in methods:
            if method == "fedavg":
                run_name = "fedavg"
                server_method = "FedAvg"
            else:
                run_name = method
                server_method = "FedCRMF"
            exp_id = f"{run_name}_{target}_seed{seed}"
            config = {
                "log_path": "./log",
                "data_path": str(
                    Path(args.output_root)
                    / dataset_key
                    / f"{run_name}_seed{seed}"
                    / target
                    / exp_id
                ),
                "dataset_path": args.dataset_path,
                "id": exp_id,
                "variant_name": run_name,
                "experiment_protocol": f"{dataset_key}_fedcrmf_v1",
                "implementation_revision": "fedcrmf_slim_v2",
                "dataset": dataset_meta["display_name"],
                "backbone": "resnet50",
                "freeze_bn": 1,
                "split_scheme": meta["scheme"],
                "server_method": server_method,
                "client_method": "ERM",
                "hparam1": 0.0,
                "num_clients": num_clients,
                "clients_per_domain": int(args.clients_per_domain),
                "strict_domain_clients": True,
                "fraction": 1.0,
                "iid": 0.0,
                "num_rounds": int(args.num_rounds),
                "local_epochs": 1,
                "batch_size": 16,
                "optimizer": "torch.optim.Adam",
                "lr": 3e-5,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "seed": seed,
                "disable_pairwise_update_metrics": bool(method.startswith("fedcrmf_")),
            }
            if method != "fedavg":
                best = dataset_meta["best_params"][meta["code"]]
                history_length = int(best["history_length"])
                gate_variant = "full"
                if method == "fedcrmf_uniform_shrinkage":
                    gate_variant = "uniform_shrinkage"
                elif method == "fedcrmf_permuted_gate":
                    gate_variant = "permuted_gate"
                elif method == "fedcrmf_no_centering":
                    gate_variant = "no_centering"
                elif method == "fedcrmf_response_gate_dropout":
                    gate_variant = "response_gate_dropout"
                    history_length = 1
                elif method in RESPONSE_GATE_DROPOUT_ABLATIONS:
                    gate_variant = "response_gate_dropout"
                    history_length = int(
                        RESPONSE_GATE_DROPOUT_ABLATIONS[method]["history_length"]
                    )
                elif method == "fedcrmf_one_round":
                    history_length = 1
                elif method != "fedcrmf":
                    raise ValueError(f"Unsupported FedCRMF method variant: {method}")
                config.update(
                    {
                        "fedcrmf_history_length": history_length,
                        "fedcrmf_warmup_rounds": max(history_length - 1, 0),
                        "fedcrmf_mu": float(best["mu"]),
                        "fedcrmf_gate_variant": gate_variant,
                        "fedcrmf_gate_dropout_p": float(
                            RESPONSE_GATE_DROPOUT_ABLATIONS.get(
                                method, {"dropout_p": args.gate_dropout_p}
                            )["dropout_p"]
                        ),
                        "fedcrmf_alpha_mode": "uniform",
                        "fedcrmf_hist_keys": "ALL",
                        "fedcrmf_hist_max_numel": 5000000,
                        "save_single_model": 1 if args.save_single_model else 0,
                    }
                )
            path = Path(args.config_root) / f"{dataset_key}_seed{seed}" / f"{exp_id}.json"
            write_config(path, config)
            print(path)


if __name__ == "__main__":
    main()
