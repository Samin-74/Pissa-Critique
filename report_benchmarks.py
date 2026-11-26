from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.benchmarks import BENCHMARK_ORDER, resolve_benchmark

MODES: Tuple[str, ...] = (
    "uniform_lora",
    "uniform_pissa",
    "entropy_pissa",
    "entropy_pissa_inverted",
)

MODE_LABELS: Dict[str, str] = {
    "uniform_lora": "Uniform LoRA (r=16)",
    "uniform_pissa": "Uniform PiSSA (r=16)",
    "entropy_pissa": "Entropy-PiSSA (high entropy first)",
    "entropy_pissa_inverted": "Entropy-PiSSA (inverted)",
    "base_model": "Base model (no fine-tuning)",
}


def log(message: str) -> None:
    print(f"[report] {message}", flush=True)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def eval_path(run_dir: str, mode: str, benchmark: str) -> str:
    suffix = "" if benchmark in ("", "gsm8k") else f"_{benchmark}"
    return os.path.join(run_dir, f"eval_{mode}{suffix}.json")


def discover_groups(output_dir: str) -> List[str]:
    groups = []
    for entry in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, entry)
        if os.path.isdir(path) and ("_seed" in entry or entry.startswith("base_reference")):
            groups.append(entry)
    return groups


def is_base_group(group: str) -> bool:
    return group.startswith("base_reference")


def collect(output_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in discover_groups(output_dir):
        if is_base_group(group):
            # Base refs sit in the group dir, and the family suffix keeps the columns apart
            model = group[len("base_reference_"):] or "base"
            seed, modes = 0, ("base_model",)
        else:
            model, _, seed_text = group.rpartition("_seed")
            seed = int(seed_text) if seed_text.isdigit() else 0
            modes = MODES
        for mode in modes:
            run_dir = (
                os.path.join(output_dir, group)
                if is_base_group(group)
                else os.path.join(output_dir, group, mode)
            )
            for benchmark in BENCHMARK_ORDER:
                payload = load_json(eval_path(run_dir, mode, benchmark))
                if payload is None:
                    continue
                num_examples = int(payload.get("num_examples", 0))
                num_unparsed = int(payload.get("num_unparsed", 0))
                parsed = max(num_examples - num_unparsed, 0)
                # Older artifacts predate these keys, so derive them from the counts
                among_parsed = payload.get("accuracy_pct_among_parsed")
                if among_parsed is None:
                    among_parsed = (
                        100.0 * int(payload.get("num_correct", 0)) / parsed if parsed else 0.0
                    )
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "group": group,
                        "mode": mode,
                        "benchmark": benchmark,
                        "accuracy_pct": float(payload.get("accuracy_pct", 0.0)),
                        "accuracy_pct_among_parsed": float(among_parsed),
                        "num_examples": num_examples,
                        "num_correct": int(payload.get("num_correct", 0)),
                        "num_unparsed": num_unparsed,
                        "records": payload.get("records"),
                    }
                )
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["benchmark"], row["model"], row["mode"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for (benchmark, model, mode), items in sorted(buckets.items()):
        accs = [i["accuracy_pct"] for i in items]
        unparsed = sum(i["num_unparsed"] for i in items)
        total = sum(i["num_examples"] for i in items)
        out.append(
            {
                "benchmark": benchmark,
                "model": model,
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "n_seeds": len(items),
                "seeds": sorted(i["seed"] for i in items),
                "accuracy_pct_mean": statistics.mean(accs),
                "accuracy_pct_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "accuracy_pct_values": accs,
                "accuracy_pct_among_parsed_mean": statistics.mean(
                    [i["accuracy_pct_among_parsed"] for i in items]
                ),
                "num_examples": total,
                "unparsed_pct": 100.0 * unparsed / max(total, 1),
            }
        )
    return out


def binom_two_sided(k: int, n: int) -> float:
    # Exact binomial test against p=0.5, used for the paired discordant counts
    if n == 0:
        return 1.0
    probs = [comb(n, i) * 0.5**n for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def paired_test(rows: Sequence[Dict[str, Any]], baseline: str, treatment: str) -> List[Dict[str, Any]]:
    # Compares two modes on the exact same items, which is far more sensitive than seed means
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        by_key[(row["group"], row["mode"], row["benchmark"])] = row

    results: List[Dict[str, Any]] = []
    seen = {(row["benchmark"], row["model"], row["group"]) for row in rows}
    for benchmark, model, group in sorted(seen):
        base = by_key.get((group, baseline, benchmark))
        treat = by_key.get((group, treatment, benchmark))
        if not base or not treat:
            continue
        if not base.get("records") or not treat.get("records"):
            continue
        base_correct = {r["index"]: r["correct"] for r in base["records"]}
        treat_correct = {r["index"]: r["correct"] for r in treat["records"]}
        common = sorted(set(base_correct) & set(treat_correct))
        treat_only = sum(1 for i in common if treat_correct[i] and not base_correct[i])
        base_only = sum(1 for i in common if base_correct[i] and not treat_correct[i])
        discordant = treat_only + base_only
        results.append(
            {
                "benchmark": benchmark,
                "model": model,
                "group": group,
                "baseline": baseline,
                "treatment": treatment,
                "n_items": len(common),
                "baseline_only_correct": base_only,
                "treatment_only_correct": treat_only,
                "delta_pct": 100.0 * (treat_only - base_only) / max(len(common), 1),
                "p_value": binom_two_sided(treat_only, discordant),
            }
        )
    return results


def write_outputs(
    summary: Sequence[Dict[str, Any]],
    tests: Sequence[Dict[str, Any]],
    output_dir: str,
) -> None:
    from src.utils import ensure_dir, save_csv, save_json

    results_dir = os.path.join(output_dir, "results")
    ensure_dir(results_dir)

    save_csv(
        [
            {
                "benchmark": row["benchmark"],
                "model": row["model"],
                "mode": row["mode"],
                "label": row["label"],
                "n_seeds": row["n_seeds"],
                "accuracy_pct_mean": round(row["accuracy_pct_mean"], 4),
                "accuracy_pct_std": round(row["accuracy_pct_std"], 4),
                "accuracy_pct_among_parsed_mean": round(
                    row["accuracy_pct_among_parsed_mean"], 4
                ),
                "num_examples": row["num_examples"],
                "unparsed_pct": round(row["unparsed_pct"], 4),
            }
            for row in summary
        ],
        os.path.join(results_dir, "benchmark_summary.csv"),
        fieldnames=[
            "benchmark",
            "model",
            "mode",
            "label",
            "n_seeds",
            "accuracy_pct_mean",
            "accuracy_pct_std",
            "accuracy_pct_among_parsed_mean",
            "num_examples",
            "unparsed_pct",
        ],
    )
    save_json(list(summary), os.path.join(results_dir, "benchmark_summary.json"))

    if tests:
        save_csv(
            [
                {
                    "benchmark": row["benchmark"],
                    "model": row["model"],
                    "group": row["group"],
                    "n_items": row["n_items"],
                    "baseline_only_correct": row["baseline_only_correct"],
                    "treatment_only_correct": row["treatment_only_correct"],
                    "delta_pct": round(row["delta_pct"], 4),
                    "p_value": round(row["p_value"], 6),
                }
                for row in tests
            ],
            os.path.join(results_dir, "benchmark_paired_tests.csv"),
            fieldnames=[
                "benchmark",
                "model",
                "group",
                "n_items",
                "baseline_only_correct",
                "treatment_only_correct",
                "delta_pct",
                "p_value",
            ],
        )
        save_json(list(tests), os.path.join(results_dir, "benchmark_paired_tests.json"))


def print_tables(summary: Sequence[Dict[str, Any]], tests: Sequence[Dict[str, Any]]) -> None:
    print()
    for benchmark in BENCHMARK_ORDER:
        block = [row for row in summary if row["benchmark"] == benchmark]
        if not block:
            continue
        spec = resolve_benchmark(benchmark)
        print(f"=== {spec.display_name} ===")
        models = sorted({row["model"] for row in block})
        print(f"{'mode':<38}" + "".join(f"{m:<14}" for m in models))
        # Include the base reference row when it exists, otherwise just the four modes
        row_modes = list(MODES) + [
            m for m in {"base_model"} if any(r["mode"] == m for r in block)
        ]
        for mode in row_modes:
            cells = []
            for model in models:
                match = [r for r in block if r["model"] == model and r["mode"] == mode]
                if not match:
                    cells.append("-".ljust(14))
                else:
                    row = match[0]
                    cells.append(
                        f"{row['accuracy_pct_mean']:.2f}+-{row['accuracy_pct_std']:.2f}".ljust(14)
                    )
            print(f"{MODE_LABELS.get(mode, mode):<38}" + "".join(cells))
        print()

    if any(row["unparsed_pct"] > 1.0 for row in summary):
        print("=== Format compliance (share of replies in the expected answer format) ===")
        print(f"{'benchmark':<16}{'group':<18}{'mode':<30}{'parsed%':>9}{'acc|parsed':>12}")
        for row in summary:
            if row["unparsed_pct"] <= 1.0:
                continue
            print(
                f"{row['benchmark']:<16}{row['model']:<18}{row['mode']:<30}"
                f"{100.0 - row['unparsed_pct']:>9.1f}{row['accuracy_pct_among_parsed_mean']:>12.2f}"
            )
        print()

    if tests:
        print("=== Paired comparison vs uniform PiSSA (same items) ===")
        print(f"{'benchmark':<16}{'group':<18}{'delta':>8}{'p':>10}")
        for row in tests:
            print(
                f"{row['benchmark']:<16}{row['group']:<18}"
                f"{row['delta_pct']:>+8.2f}{row['p_value']:>10.4f}"
            )
        print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-benchmark comparison for the Entropy-PiSSA runs"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--baseline", type=str, default="uniform_pissa")
    parser.add_argument("--treatment", type=str, default="entropy_pissa_inverted")
    args = parser.parse_args(argv)

    rows = collect(args.output_dir)
    if not rows:
        log(f"no eval artifacts found under {args.output_dir}")
        return 1
    summary = summarize(rows)
    tests = paired_test(rows, args.baseline, args.treatment)
    write_outputs(summary, tests, args.output_dir)
    print_tables(summary, tests)
    log(f"benchmarks={sorted({r['benchmark'] for r in rows})} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
