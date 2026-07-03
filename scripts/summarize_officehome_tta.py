import argparse
import json
from pathlib import Path


DEFAULT_TARGETS = ("cpr_a", "apr_c", "acr_p", "acp_r")


def _acc_from_metric(metric):
    if isinstance(metric, dict):
        if "acc_avg" in metric:
            return metric["acc_avg"]
        if "acc" in metric:
            return metric["acc"]
    return None


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _read_rows(root, seed, targets, run_name):
    rows = []
    for target in targets:
        path = (
            root
            / f"{run_name}_seed{seed}"
            / target
            / f"{run_name}_{target}_seed{seed}"
            / "tta"
            / "tta_summary.json"
        )
        if not path.exists():
            print(f"Missing: {path}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            before = _acc_from_metric(row.get("metric_before"))
            online_before = _acc_from_metric(row.get("metric_online_before"))
            after = _acc_from_metric(row.get("metric_after"))
            delta = None if before is None or after is None else float(after) - float(before)
            rows.append(
                {
                    "target": target,
                    "mode": row.get("mode", ""),
                    "before": before,
                    "online_before": online_before,
                    "after": after,
                    "delta": delta,
                    "selected_ratio": row.get("selected_ratio"),
                    "entropy_before": row.get("entropy_before"),
                    "entropy_after": row.get("entropy_after"),
                }
            )
    return rows


def _print_rows(rows):
    print(
        "target,mode,acc_before,acc_online_before,acc_after,"
        "delta_acc,selected_ratio,entropy_before,entropy_after"
    )
    for row in rows:
        print(
            ",".join(
                [
                    row["target"],
                    str(row["mode"]),
                    _fmt(row["before"]),
                    _fmt(row["online_before"]),
                    _fmt(row["after"]),
                    _fmt(row["delta"]),
                    _fmt(row["selected_ratio"]),
                    _fmt(row["entropy_before"]),
                    _fmt(row["entropy_after"]),
                ]
            )
        )


def _print_means(rows):
    by_mode = {}
    for row in rows:
        if row["before"] is not None and row["after"] is not None:
            by_mode.setdefault(row["mode"], []).append(row)

    print("\nMean over targets:")
    for mode, values in by_mode.items():
        n = len(values)
        mean_before = sum(float(v["before"]) for v in values) / n
        mean_online_before = sum(float(v["online_before"]) for v in values) / n
        mean_after = sum(float(v["after"]) for v in values) / n
        mean_delta = sum(float(v["delta"]) for v in values) / n
        mean_selected = sum(float(v["selected_ratio"]) for v in values) / n
        print(
            f"{mode}: "
            f"before={mean_before:.6f}, "
            f"online_before={mean_online_before:.6f}, "
            f"after={mean_after:.6f}, "
            f"delta={mean_delta:.6f}, "
            f"selected_ratio={mean_selected:.6f}, "
            f"n={n}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--root",
        default="outputs/officehome",
        type=Path,
        help="Directory containing TTA run outputs.",
    )
    parser.add_argument("--run-name", default="fedcrmf_tta")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated OfficeHome target names.",
    )
    args = parser.parse_args()

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    rows = _read_rows(args.root, args.seed, targets, args.run_name)
    _print_rows(rows)
    _print_means(rows)


if __name__ == "__main__":
    main()
