import argparse
import json
from pathlib import Path


TARGETS = {
    "acs_p": {"scheme": "acs-p", "code": "P"},
    "pcs_a": {"scheme": "pcs-a", "code": "A"},
    "pac_s": {"scheme": "pac-s", "code": "S"},
    "pas_c": {"scheme": "pas-c", "code": "C"},
}

BEST_PARAMS = {
    "P": {"history_length": 1, "mu": 10000},
    "A": {"history_length": 3, "mu": 10000},
    "S": {"history_length": 3, "mu": 10000},
    "C": {"history_length": 2, "mu": 10000},
}


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="42")
    parser.add_argument("--target", default="all", choices=["all", *TARGETS.keys()])
    parser.add_argument("--method", default="fedcrmf", choices=["fedavg", "fedcrmf", "all"])
    parser.add_argument("--dataset-path", default="./dataset")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--config-root", default="./configs")
    parser.add_argument("--clients-per-domain", default=1, type=int)
    parser.add_argument("--num-rounds", default=40, type=int)
    parser.add_argument("--save-single-model", action="store_true")
    args = parser.parse_args()

    seed = int(args.seed)
    targets = list(TARGETS.keys()) if args.target == "all" else [args.target]
    methods = ["fedavg", "fedcrmf"] if args.method == "all" else [args.method]
    num_clients = 3 * int(args.clients_per_domain)
    for target in targets:
        meta = TARGETS[target]
        for method in methods:
            if method == "fedavg":
                run_name = "fedavg"
                server_method = "FedAvg"
            else:
                run_name = "fedcrmf"
                server_method = "FedCRMF"
            exp_id = f"{run_name}_{target}_seed{seed}"
            config = {
                "log_path": "./log",
                "data_path": str(Path(args.output_root) / "pacs" / f"{run_name}_seed{seed}" / target / exp_id),
                "dataset_path": args.dataset_path,
                "id": exp_id,
                "variant_name": run_name,
                "experiment_protocol": "pacs_fedcrmf_v1",
                "implementation_revision": "fedcrmf_slim_v1",
                "dataset": "PACS",
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
            }
            if method == "fedcrmf":
                best = BEST_PARAMS[meta["code"]]
                history_length = int(best["history_length"])
                config.update(
                    {
                        "fedcrmf_history_length": history_length,
                        "fedcrmf_warmup_rounds": max(history_length - 1, 0),
                        "fedcrmf_mu": float(best["mu"]),
                        "fedcrmf_alpha_mode": "uniform",
                        "fedcrmf_hist_keys": "ALL",
                        "fedcrmf_hist_max_numel": 5000000,
                        "save_single_model": 1 if args.save_single_model else 0,
                    }
                )
            path = Path(args.config_root) / f"pacs_seed{seed}" / f"{exp_id}.json"
            write_config(path, config)
            print(path)


if __name__ == "__main__":
    main()
