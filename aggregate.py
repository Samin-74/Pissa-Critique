from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.rank_allocator import (
    budget_breakdown,
    load_allocation,
    summarize_by_block_type,
    summarize_by_module,
    summarize_ranks_by_block_type,
    summarize_ranks_by_module,
)
from src.utils import (
    ensure_dir,
    plot_accuracy_comparison,
    plot_budget_comparison,
    plot_energy_retention,
    plot_entropy_by_block,
    plot_entropy_by_layer,
    plot_entropy_heatmap,
    plot_entropy_vs_rank,
    plot_loss_curves,
    plot_rank_distribution,
    plot_rank_distribution_comparison,
    plot_rank_heatmap,
    save_csv,
    save_json,
)

MODES: Tuple[str, ...] = (
    "uniform_lora",
    "uniform_pissa",
    "entropy_pissa",
    "entropy_pissa_inverted",
)

MODE_LABELS_FULL: Dict[str, str] = {
    "uniform_lora": "Uniform LoRA (r=16)",
    "uniform_pissa": "Uniform PiSSA (r=16)",
    "entropy_pissa": "Entropy-PiSSA (high entropy first)",
    "entropy_pissa_inverted": "Entropy-PiSSA (inverted)",
}

MODEL_TAGS: Tuple[str, ...] = ("qwen", "phi")
SEEDS: Tuple[int, ...] = (42, 123)


def log(message: str) -> None:
    print(f"[aggregate] {message}", flush=True)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_runs(output_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag in MODEL_TAGS:
        for seed in SEEDS:
            group = f"{tag}_seed{seed}"
            for mode in MODES:
                run_dir = os.path.join(output_dir, group, mode)
                metrics = load_json(os.path.join(run_dir, "train_metrics.json"))
                evaluation = load_json(os.path.join(run_dir, f"eval_{mode}.json"))
                losses = load_json(os.path.join(run_dir, "loss_curve.json"))
                if metrics is None or evaluation is None:
                    log(f"missing artifacts for {group}/{mode}")
                    continue
                records = losses or []
                grad = sorted(
                    r["grad_norm"] for r in records if r.get("grad_norm") is not None
                )
                tok_acc = [
                    r["mean_token_accuracy"]
                    for r in records
                    if r.get("mean_token_accuracy") is not None
                ]
                tok_ent = [r["entropy"] for r in records if r.get("entropy") is not None]
                rows.append(
                    {
                        "model": tag,
                        "seed": seed,
                        "group": group,
                        "mode": mode,
                        "label": MODE_LABELS_FULL[mode],
                        "trainable_parameters": int(metrics.get("trainable_parameters", 0)),
                        "total_parameters": int(metrics.get("total_parameters", 0)),
                        "trainable_pct": float(metrics.get("trainable_ratio_pct", 0.0)),
                        "final_loss": float(metrics.get("train_loss", 0.0)),
                        "train_seconds": float(metrics.get("train_elapsed_seconds", 0.0)),
                        "peak_vram_gb": float(
                            (metrics.get("gpu_memory") or {}).get("max_allocated_gb", 0.0)
                        ),
                        "accuracy": float(evaluation.get("accuracy", 0.0)),
                        "accuracy_pct": 100.0 * float(evaluation.get("accuracy", 0.0)),
                        "num_examples": int(evaluation.get("num_examples", 0)),
                        "num_correct": int(evaluation.get("num_correct", 0)),
                        "num_unparsed": int(evaluation.get("num_unparsed", 0)),
                        "grad_norm_median": grad[len(grad) // 2] if grad else 0.0,
                        "grad_norm_p90": grad[int(len(grad) * 0.9)] if grad else 0.0,
                        "grad_norm_max": max(grad) if grad else 0.0,
                        "grad_steps_over_50": sum(1 for g in grad if g > 50),
                        "final_token_entropy": tok_ent[-1] if tok_ent else 0.0,
                        "max_token_entropy": max(tok_ent) if tok_ent else 0.0,
                        "final_token_accuracy": tok_acc[-1] if tok_acc else 0.0,
                        "loss_records": records,
                    }
                )
    return rows


def average_loss_records(per_seed: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # Average logged losses at matching steps so seeds do not double-plot
    buckets: Dict[int, List[float]] = {}
    for records in per_seed:
        for record in records:
            step = record.get("step")
            loss = record.get("loss")
            if step is None or loss is None:
                continue
            buckets.setdefault(int(step), []).append(float(loss))
    return [
        {"step": step, "loss": statistics.mean(values)}
        for step, values in sorted(buckets.items())
    ]


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["model"], row["mode"]), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (model, mode), items in sorted(buckets.items()):
        accs = [i["accuracy_pct"] for i in items]
        losses = [i["final_loss"] for i in items]
        params = [i["trainable_parameters"] for i in items]
        summary.append(
            {
                "model": model,
                "mode": mode,
                "label": MODE_LABELS_FULL[mode],
                "n_seeds": len(items),
                "seeds": sorted(i["seed"] for i in items),
                "accuracy_pct_mean": statistics.mean(accs),
                "accuracy_pct_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "accuracy_pct_values": accs,
                "final_loss_mean": statistics.mean(losses),
                "final_loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "trainable_parameters": int(statistics.mean(params)),
                "trainable_pct": statistics.mean([i["trainable_pct"] for i in items]),
                "train_seconds_mean": statistics.mean([i["train_seconds"] for i in items]),
                "peak_vram_gb_max": max(i["peak_vram_gb"] for i in items),
                "num_examples": items[0]["num_examples"],
                "num_unparsed_total": sum(i["num_unparsed"] for i in items),
                "grad_norm_median_mean": statistics.mean([i["grad_norm_median"] for i in items]),
                "grad_norm_p90_mean": statistics.mean([i["grad_norm_p90"] for i in items]),
                "grad_norm_max_max": max(i["grad_norm_max"] for i in items),
                "grad_steps_over_50_total": sum(i["grad_steps_over_50"] for i in items),
                "final_token_entropy_mean": statistics.mean(
                    [i["final_token_entropy"] for i in items]
                ),
                "max_token_entropy_max": max(i["max_token_entropy"] for i in items),
                "final_token_accuracy_mean": statistics.mean(
                    [i["final_token_accuracy"] for i in items]
                ),
            }
        )
    return summary


def write_tables(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], output_dir: str) -> None:
    results_dir = os.path.join(output_dir, "results")
    ensure_dir(results_dir)

    per_run_fields = [
        "model",
        "seed",
        "group",
        "mode",
        "label",
        "trainable_parameters",
        "trainable_pct",
        "final_loss",
        "train_seconds",
        "peak_vram_gb",
        "accuracy",
        "accuracy_pct",
        "num_examples",
        "num_correct",
        "num_unparsed",
        "grad_norm_median",
        "grad_norm_p90",
        "grad_norm_max",
        "grad_steps_over_50",
        "final_token_entropy",
        "max_token_entropy",
        "final_token_accuracy",
    ]
    save_csv(rows, os.path.join(results_dir, "runs_all.csv"), fieldnames=per_run_fields)

    for tag in MODEL_TAGS:
        subset = [r for r in rows if r["model"] == tag]
        if subset:
            save_csv(
                subset,
                os.path.join(results_dir, f"comparison_{tag}.csv"),
                fieldnames=per_run_fields,
            )
    save_csv(rows, os.path.join(results_dir, "comparison_all.csv"), fieldnames=per_run_fields)

    summary_fields = [
        "model",
        "mode",
        "label",
        "n_seeds",
        "accuracy_pct_mean",
        "accuracy_pct_std",
        "final_loss_mean",
        "final_loss_std",
        "trainable_parameters",
        "trainable_pct",
        "train_seconds_mean",
        "peak_vram_gb_max",
        "num_examples",
        "num_unparsed_total",
        "grad_norm_median_mean",
        "grad_norm_p90_mean",
        "grad_norm_max_max",
        "grad_steps_over_50_total",
        "final_token_entropy_mean",
        "max_token_entropy_max",
        "final_token_accuracy_mean",
    ]
    save_csv(summary, os.path.join(results_dir, "summary_table.csv"), fieldnames=summary_fields)
    save_json(
        {"per_run": rows, "aggregated": summary},
        os.path.join(results_dir, "summary_metrics.json"),
    )
    log(f"tables written to {results_dir}")


def write_plots(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], output_dir: str) -> None:
    plots_dir = os.path.join(output_dir, "plots")
    ensure_dir(plots_dir)

    for tag in MODEL_TAGS:
        curves: Dict[str, List[Dict[str, Any]]] = {}
        for mode in MODES:
            per_seed = [
                row["loss_records"]
                for row in rows
                if row["model"] == tag and row["mode"] == mode and row["loss_records"]
            ]
            if per_seed:
                curves[mode] = average_loss_records(per_seed)
        if curves:
            plot_loss_curves(curves, os.path.join(plots_dir, f"loss_curves_{tag}.png"))

        results = {
            row["mode"]: {"accuracy": row["accuracy_pct_mean"] / 100.0}
            for row in summary
            if row["model"] == tag
        }
        if results:
            plot_accuracy_comparison(
                results, os.path.join(plots_dir, f"accuracy_{tag}.png")
            )

        budgets = {
            row["mode"]: row["trainable_parameters"]
            for row in summary
            if row["model"] == tag
        }
        if budgets:
            plot_budget_comparison(
                budgets,
                os.path.join(plots_dir, f"parameter_budget_{tag}.png"),
                target_budget=budgets.get("uniform_lora"),
            )

    combined = {
        f"{row['model']}_{row['mode']}": {"accuracy": row["accuracy_pct_mean"] / 100.0}
        for row in summary
    }
    if combined:
        combined_order = [
            f"{tag}_{mode}" for tag in MODEL_TAGS for mode in MODES
            if f"{tag}_{mode}" in combined
        ]
        combined_labels = {
            f"{tag}_{mode}": f"{tag.upper()} {MODE_LABELS_FULL[mode]}"
            for tag in MODEL_TAGS
            for mode in MODES
        }
        plot_accuracy_comparison(
            combined,
            os.path.join(plots_dir, "accuracy_all.png"),
            labels=combined_labels,
            order=combined_order,
        )

    for tag in MODEL_TAGS:
        alloc_path = os.path.join(output_dir, "analysis", tag, "allocation.json")
        if not os.path.isfile(alloc_path):
            continue
        allocation = load_allocation(alloc_path)
        plot_entropy_heatmap(
            allocation.spectra, os.path.join(plots_dir, f"spectral_entropy_heatmap_{tag}.png")
        )
        plot_entropy_by_layer(
            allocation.spectra, os.path.join(plots_dir, f"spectral_entropy_by_layer_{tag}.png")
        )
        plot_entropy_by_block(
            summarize_by_block_type(allocation.spectra),
            os.path.join(plots_dir, f"spectral_entropy_by_block_{tag}.png"),
        )
        plot_rank_heatmap(allocation, os.path.join(plots_dir, f"rank_heatmap_{tag}.png"))
        plot_rank_distribution(
            allocation,
            os.path.join(plots_dir, f"rank_distribution_{tag}.png"),
            uniform_rank=allocation.uniform_rank,
        )
        plot_rank_distribution_comparison(
            allocation,
            os.path.join(plots_dir, f"rank_distribution_comparison_{tag}.png"),
            uniform_rank=allocation.uniform_rank,
        )
        plot_entropy_vs_rank(
            allocation, os.path.join(plots_dir, f"entropy_vs_rank_{tag}.png")
        )
        plot_energy_retention(
            allocation, os.path.join(plots_dir, f"energy_retention_{tag}.png")
        )

        inverted_path = os.path.join(output_dir, "analysis", f"{tag}_inverted", "allocation.json")
        if os.path.isfile(inverted_path):
            inverted = load_allocation(inverted_path)
            plot_rank_heatmap(
                inverted, os.path.join(plots_dir, f"rank_heatmap_{tag}_inverted.png")
            )
            plot_rank_distribution(
                inverted,
                os.path.join(plots_dir, f"rank_distribution_{tag}_inverted.png"),
                uniform_rank=inverted.uniform_rank,
            )
            plot_rank_distribution_comparison(
                inverted,
                os.path.join(plots_dir, f"rank_distribution_comparison_{tag}_inverted.png"),
                uniform_rank=inverted.uniform_rank,
            )
            plot_entropy_vs_rank(
                inverted, os.path.join(plots_dir, f"entropy_vs_rank_{tag}_inverted.png")
            )
            plot_energy_retention(
                inverted, os.path.join(plots_dir, f"energy_retention_{tag}_inverted.png")
            )
    log(f"plots written to {plots_dir}")


def write_analysis_summary(output_dir: str) -> None:
    for tag in MODEL_TAGS:
        for suffix, inverted in (("", False), ("_inverted", True)):
            alloc_path = os.path.join(output_dir, "analysis", f"{tag}{suffix}", "allocation.json")
            if not os.path.isfile(alloc_path):
                continue
            allocation = load_allocation(alloc_path)
            payload = {
                "model_tag": tag,
                "inverted": inverted,
                "target_budget": allocation.target_budget,
                "achieved_budget": allocation.achieved_budget,
                "budget_error_pct": allocation.budget_error_pct,
                "within_tolerance": allocation.within_tolerance,
                "uniform_rank": allocation.uniform_rank,
                "mean_rank": allocation.mean_rank,
                "min_rank": allocation.min_rank,
                "max_rank": allocation.max_rank,
                "rank_std": allocation.rank_std,
                "r_min": allocation.r_min,
                "entropy_temperature": allocation.entropy_temperature,
                "high_entropy_more_rank": allocation.high_entropy_more_rank,
                "num_matrices": len(allocation.rank_pattern),
                "entropy_by_block": summarize_by_block_type(allocation.spectra),
                "entropy_by_module": summarize_by_module(allocation.spectra),
                "rank_by_block": summarize_ranks_by_block_type(allocation),
                "rank_by_module": summarize_ranks_by_module(allocation),
                "budget_by_module": budget_breakdown(allocation),
            }
            save_json(
                payload,
                os.path.join(output_dir, "analysis", f"{tag}{suffix}", "allocation_summary.json"),
            )
    log("analysis summaries written")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Entropy-PiSSA experiment results")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args(argv)

    rows = collect_runs(args.output_dir)
    if not rows:
        log("no completed runs found")
        return 1
    log(f"collected {len(rows)} runs")
    summary = aggregate(rows)
    write_tables(rows, summary, args.output_dir)
    write_analysis_summary(args.output_dir)
    write_plots(rows, summary, args.output_dir)
    log("aggregation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
