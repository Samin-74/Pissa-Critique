from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch

from src.rank_allocator import (
    DEFAULT_ENTROPY_TEMPERATURE,
    DEFAULT_R_MIN,
    DEFAULT_TOLERANCE,
    DEFAULT_UNIFORM_RANK,
    allocate_from_model,
    budget_breakdown,
    save_allocation,
    summarize_by_block_type,
    summarize_by_module,
    summarize_ranks_by_block_type,
    summarize_ranks_by_module,
    to_peft_rank_pattern,
)
from src.utils import (
    ensure_dir,
    entropy_table_rows,
    plot_energy_retention,
    plot_entropy_by_block,
    plot_entropy_by_layer,
    plot_entropy_heatmap,
    plot_entropy_vs_rank,
    plot_rank_distribution,
    plot_rank_distribution_comparison,
    plot_rank_heatmap,
    rank_table_rows,
    save_csv,
    save_json,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectral entropy and the Entropy-PiSSA rank allocation"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    parser.add_argument("--uniform_rank", type=int, default=DEFAULT_UNIFORM_RANK)
    parser.add_argument("--r_min", type=int, default=DEFAULT_R_MIN)
    parser.add_argument("--r_max", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--entropy_temperature", type=float, default=DEFAULT_ENTROPY_TEMPERATURE)
    parser.add_argument(
        "--high_entropy_more_rank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[analysis] {message}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM

    log(f"loading {args.model_name} on CPU")
    # Keep the checkpoint dtype: compute_spectrum upcasts one matrix at a time
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()

    log("running SVD and rank allocation")
    allocation = allocate_from_model(
        model,
        uniform_rank=args.uniform_rank,
        r_min=args.r_min,
        r_max=args.r_max,
        high_entropy_more_rank=args.high_entropy_more_rank,
        tolerance=args.tolerance,
        target_modules=None,
        progress_every=28,
        entropy_temperature=args.entropy_temperature,
    )
    log(
        f"target={allocation.target_budget} achieved={allocation.achieved_budget} "
        f"error={allocation.budget_error_pct:.4f}% within_tolerance={allocation.within_tolerance}"
    )

    ensure_dir(args.output_dir)
    save_allocation(
        allocation,
        os.path.join(args.output_dir, "allocation.json"),
        include_spectrum_values=True,
    )
    save_json(
        {
            "model_name": args.model_name,
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
            "r_max": allocation.r_max,
            "tolerance": allocation.tolerance,
            "entropy_temperature": allocation.entropy_temperature,
            "high_entropy_more_rank": allocation.high_entropy_more_rank,
            "num_layers": len(allocation.rank_pattern),
            "entropy_by_block": summarize_by_block_type(allocation.spectra),
            "entropy_by_module": summarize_by_module(allocation.spectra),
            "rank_by_block": summarize_ranks_by_block_type(allocation),
            "rank_by_module": summarize_ranks_by_module(allocation),
            "budget_by_module": budget_breakdown(allocation),
        },
        os.path.join(args.output_dir, "allocation_summary.json"),
    )
    save_csv(
        entropy_table_rows(allocation.spectra),
        os.path.join(args.output_dir, "spectral_entropy.csv"),
        fieldnames=[
            "layer",
            "module",
            "name",
            "out_features",
            "in_features",
            "entropy",
            "effective_rank",
            "block",
        ],
    )
    save_csv(
        rank_table_rows(allocation),
        os.path.join(args.output_dir, "rank_allocation.csv"),
        fieldnames=[
            "layer",
            "module",
            "name",
            "entropy",
            "rank",
            "uniform_rank",
            "delta",
            "params",
            "energy_pct",
        ],
    )
    save_json(
        to_peft_rank_pattern(allocation.rank_pattern),
        os.path.join(args.output_dir, "peft_rank_pattern.json"),
    )

    if not args.skip_plots:
        plots_dir = os.path.join(args.output_dir, "plots")
        ensure_dir(plots_dir)
        plot_entropy_heatmap(
            allocation.spectra, os.path.join(plots_dir, "spectral_entropy_heatmap.png")
        )
        plot_entropy_by_layer(
            allocation.spectra, os.path.join(plots_dir, "spectral_entropy_by_layer.png")
        )
        plot_entropy_by_block(
            summarize_by_block_type(allocation.spectra),
            os.path.join(plots_dir, "spectral_entropy_by_block.png"),
        )
        plot_rank_heatmap(allocation, os.path.join(plots_dir, "rank_heatmap.png"))
        plot_rank_distribution(
            allocation,
            os.path.join(plots_dir, "rank_distribution.png"),
            uniform_rank=args.uniform_rank,
        )
        plot_rank_distribution_comparison(
            allocation,
            os.path.join(plots_dir, "rank_distribution_comparison.png"),
            uniform_rank=args.uniform_rank,
        )
        plot_entropy_vs_rank(allocation, os.path.join(plots_dir, "entropy_vs_rank.png"))
        plot_energy_retention(allocation, os.path.join(plots_dir, "energy_retention.png"))
        log(f"plots written to {plots_dir}")

    log("analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
