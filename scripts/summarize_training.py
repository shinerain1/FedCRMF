import argparse
import csv
from pathlib import Path


TARGETS = {
    "pacs": ("acs_p", "pcs_a", "pac_s", "pas_c"),
    "officehome": ("cpr_a", "apr_c", "acr_p", "acp_r"),
}


def _fmt(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except ValueError:
        return str(value)


def _read_target_row(root, method, seed, target):
    path = root / f"{method}_seed{seed}" / target / "summary_all_results.csv"
    if not path.exists():
        print(f"Missing: {path}")
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"Empty: {path}")
        return None

    preferred = [row for row in rows if row.get("variant_name") == method]
    row = preferred[0] if preferred else rows[0]
    return {
        "target": target,
        "variant_name": row.get("variant_name", ""),
        "method": row.get("method", ""),
        "seed": row.get("seed", seed),
        "selected_round": row.get("selected_round", ""),
        "selected_lodo_test": row.get("selected_lodo_test", ""),
        "final_lodo_test": row.get("final_lodo_test", ""),
        "best_lodo_test": row.get("best_lodo_test", ""),
        "experiment_id": row.get("experiment_id", ""),
    }


def _mean(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value not in (None, "", "N/A"):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="officehome", choices=sorted(TARGETS))
    parser.add_argument("--method", default="fedcrmf")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output-root", default="outputs", type=Path)
    parser.add_argument("--targets", default="")
    args = parser.parse_args()

    targets = (
        [item.strip() for item in args.targets.split(",") if item.strip()]
        if args.targets
        else list(TARGETS[args.dataset])
    )
    root = args.output_root / args.dataset

    rows = []
    for target in targets:
        row = _read_target_row(root, args.method, args.seed, target)
        if row is not None:
            rows.append(row)

    print(
        "target,variant_name,method,seed,selected_round,"
        "selected_lodo_test,final_lodo_test,best_lodo_test,experiment_id"
    )
    for row in rows:
        print(
            ",".join(
                [
                    row["target"],
                    row["variant_name"],
                    row["method"],
                    str(row["seed"]),
                    str(row["selected_round"]),
                    _fmt(row["selected_lodo_test"]),
                    _fmt(row["final_lodo_test"]),
                    _fmt(row["best_lodo_test"]),
                    row["experiment_id"],
                ]
            )
        )

    print("\nMean over targets:")
    print(
        f"selected_lodo_test={_fmt(_mean(rows, 'selected_lodo_test'))}, "
        f"final_lodo_test={_fmt(_mean(rows, 'final_lodo_test'))}, "
        f"best_lodo_test={_fmt(_mean(rows, 'best_lodo_test'))}, "
        f"n={len(rows)}"
    )


if __name__ == "__main__":
    main()
