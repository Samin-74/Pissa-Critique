from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Expandable segments stop allocator fragmentation from spilling to shared memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from src.evaluator import evaluate_gsm8k, save_eval_result
from src.rank_allocator import (
    DEFAULT_ENTROPY_TEMPERATURE,
    DEFAULT_R_MIN,
    DEFAULT_TOLERANCE,
    DEFAULT_UNIFORM_RANK,
    AllocationResult,
    allocate_from_model,
    budget_breakdown,
    detect_target_modules,
    load_allocation,
    save_allocation,
    summarize_by_block_type,
    summarize_ranks_by_block_type,
    to_peft_rank_pattern,
)
from src.utils import (
    DEFAULT_SYSTEM_PROMPT,
    MODE_LABELS,
    MODE_ORDER,
    LossRecorder,
    build_loss_callback,
    build_training_text,
    count_trainable_parameters,
    ensure_dir,
    entropy_table_rows,
    format_param_count,
    gpu_memory_report,
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
    rank_table_rows,
    save_csv,
    save_json,
    set_seed,
)

MODES: Tuple[str, ...] = MODE_ORDER

MODE_ALIASES: Dict[str, str] = {
    "entropy_pissa_inverted": "entropy_pissa",
    "inverted": "entropy_pissa",
}


@dataclass(frozen=True)
class RunSpec:
    # label names the output directory and eval artifact, config_mode drives the adapter
    label: str
    config_mode: str
    high_entropy_more_rank: bool


def resolve_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def build_run_specs(requested: Sequence[str], default_direction: bool) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for mode in requested:
        config_mode = resolve_mode(mode)
        if config_mode == "entropy_pissa":
            direction = False if mode in MODE_ALIASES else bool(default_direction)
        else:
            direction = bool(default_direction)
        specs.append(
            RunSpec(
                label=mode,
                config_mode=config_mode,
                high_entropy_more_rank=direction,
            )
        )
    return specs


def model_tag_for(args: argparse.Namespace) -> str:
    group = (args.run_name or "").split("/")[0]
    return re.sub(r"_seed\d+$", "", group) or "model"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entropy-PiSSA: spectral entropy guided rank allocation for PiSSA"
    )
    parser.add_argument(
        "--mode",
        choices=sorted(set(MODES) | {"all", "inverted"}),
        default="entropy_pissa",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--allocation_json", type=str, default="")
    parser.add_argument("--uniform_rank", type=int, default=DEFAULT_UNIFORM_RANK)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--r_min", type=int, default=DEFAULT_R_MIN)
    parser.add_argument("--r_max", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--entropy_temperature", type=float, default=DEFAULT_ENTROPY_TEMPERATURE)
    parser.add_argument(
        "--high_entropy_more_rank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--eval_limit", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--cpu_only", action="store_true")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[entropy-pissa] {message}", flush=True)


def run_dir_for(args: argparse.Namespace, label: str) -> str:
    # run_name lets several seeds share one output tree
    return os.path.join(args.output_dir, args.run_name or label)


def analysis_dir_for(args: argparse.Namespace, inverted: bool) -> str:
    suffix = "_inverted" if inverted else ""
    return os.path.join(args.output_dir, "analysis", f"{model_tag_for(args)}{suffix}")


def resolve_dtype(cpu_only: bool) -> torch.dtype:
    if cpu_only or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    # Phi ships a slow sentencepiece tokenizer that cannot be converted to fast here
    is_phi = "phi" in args.model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True if is_phi else args.trust_remote_code,
        use_fast=False if is_phi else True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(args: argparse.Namespace, on_cpu: bool = False):
    from transformers import AutoModelForCausalLM

    # CPU allocation runs keep the checkpoint dtype; SVD upcasts one matrix at a time
    dtype = torch.bfloat16 if on_cpu else resolve_dtype(args.cpu_only)
    # Phi-3 remote code is incompatible with transformers 5 caches, the native class is not
    is_phi = "phi" in args.model_name.lower()
    trust = False if is_phi else args.trust_remote_code
    attn = "eager" if is_phi else None
    # Load on CPU then move, which avoids the caching allocator warmup doubling VRAM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=None,
        trust_remote_code=trust,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    if not on_cpu and not args.cpu_only and torch.cuda.is_available():
        model = model.to("cuda:0")
    model.config.use_cache = False
    return model


def load_gsm8k(args: argparse.Namespace):
    from datasets import Dataset, DatasetDict, load_dataset

    name = args.dataset_name
    if name.endswith(".json") or name.endswith(".jsonl"):
        with open(name, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "question" not in payload:
            splits = DatasetDict(
                {split: Dataset.from_list(rows) for split, rows in payload.items()}
            )
        else:
            splits = DatasetDict({"train": Dataset.from_list(payload)})
        train_split = splits["train"]
        test_split = splits["test"] if "test" in splits else splits["train"]
    else:
        dataset = load_dataset(name, args.dataset_config or None)
        train_split = dataset["train"]
        test_split = dataset["test"]
    if args.train_limit and args.train_limit > 0:
        train_split = train_split.select(range(min(args.train_limit, len(train_split))))
    return train_split, test_split


def build_text_dataset(train_split, tokenizer, args: argparse.Namespace):
    from datasets import Dataset

    system_prompt = args.system_prompt or None
    texts = [
        build_training_text(
            row["question"],
            row["answer"],
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        for row in train_split
    ]
    return Dataset.from_dict({"text": texts})


def build_lora_config(
    mode: str,
    args: argparse.Namespace,
    allocation: Optional[AllocationResult],
    target_modules: Sequence[str],
):
    from peft import LoraConfig

    # PiSSA adapters carry the principal weight components, so dropout would corrupt the init
    dropout = 0.0 if mode in ("uniform_pissa", "entropy_pissa") else args.lora_dropout
    if mode in ("uniform_pissa", "entropy_pissa") and args.lora_dropout > 0.0:
        log(f"mode={mode} forcing lora_dropout=0.0 to preserve the PiSSA initialization")

    if mode == "uniform_lora":
        return LoraConfig(
            r=args.uniform_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
        )
    if mode == "uniform_pissa":
        return LoraConfig(
            r=args.uniform_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="pissa",
        )
    if allocation is None:
        raise ValueError("entropy_pissa requires a rank allocation")
    rank_pattern = to_peft_rank_pattern(allocation.rank_pattern)
    # Constant alpha/r scaling keeps the update magnitude comparable across ranks
    alpha_pattern = {
        key: max(1, int(round(args.lora_alpha * rank / float(args.uniform_rank))))
        for key, rank in rank_pattern.items()
    }
    return LoraConfig(
        r=args.uniform_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="pissa",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )


def apply_adapter(model, mode: str, args: argparse.Namespace, allocation):
    from peft import get_peft_model

    target_modules = detect_target_modules(model)
    config = build_lora_config(mode, args, allocation, target_modules)
    peft_model = get_peft_model(model, config)
    return peft_model, config


def build_sft_config(args: argparse.Namespace, label: str):
    import inspect

    from trl import SFTConfig

    run_dir = run_dir_for(args, label)
    use_cuda = torch.cuda.is_available() and not args.cpu_only
    # transformers 5 merged warmup_ratio into warmup_steps: values below 1 are ratios
    warmup = float(args.warmup_ratio) if args.warmup_ratio < 1.0 else int(args.warmup_ratio)
    kwargs: Dict[str, Any] = {
        "output_dir": run_dir,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "save_strategy": "no",
        "warmup_steps": warmup,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "bf16": use_cuda,
        "fp16": False,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": args.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        "completion_only_loss": False,
        "report_to": args.report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "use_cpu": args.cpu_only,
        "remove_unused_columns": True,
        "dataloader_pin_memory": use_cuda,
        "logging_first_step": True,
        "disable_tqdm": False,
    }
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return SFTConfig(**filtered)


def train_single_mode(
    spec: RunSpec,
    args: argparse.Namespace,
    tokenizer,
    train_dataset,
    test_split,
    allocation: Optional[AllocationResult],
) -> Dict[str, Any]:
    from trl import SFTTrainer

    mode = spec.config_mode
    label = spec.label
    set_seed(args.seed)
    log(f"loading base model for mode={label}")
    model = load_base_model(args)
    model, lora_config = apply_adapter(model, mode, args, allocation)
    trainable, total = count_trainable_parameters(model)
    log(
        f"mode={label} trainable={format_param_count(trainable)} "
        f"total={format_param_count(total)} ratio={100.0 * trainable / max(total, 1):.3f}%"
    )

    recorder = LossRecorder()
    sft_config = build_sft_config(args, label)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[build_loss_callback(recorder)],
    )

    run_dir = run_dir_for(args, label)
    ensure_dir(run_dir)
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    loss_json = os.path.join(run_dir, "loss_curve.json")
    loss_csv = os.path.join(run_dir, "loss_curve.csv")
    if not recorder.records:
        for entry in getattr(trainer.state, "log_history", []) or []:
            recorder.add(dict(entry), step=entry.get("step"))
    recorder.save(loss_json, loss_csv)

    adapter_dir = os.path.join(run_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = dict(train_result.metrics or {})
    metrics["train_elapsed_seconds"] = elapsed
    metrics["trainable_parameters"] = trainable
    metrics["total_parameters"] = total
    metrics["trainable_ratio_pct"] = 100.0 * trainable / max(total, 1)
    metrics["mode"] = label
    metrics["config_mode"] = mode
    metrics["high_entropy_more_rank"] = spec.high_entropy_more_rank
    metrics["lora_r"] = lora_config.r
    metrics["init_lora_weights"] = str(lora_config.init_lora_weights)
    metrics["rank_pattern_size"] = len(lora_config.rank_pattern or {})
    metrics["gpu_memory"] = gpu_memory_report()
    save_json(metrics, os.path.join(run_dir, "train_metrics.json"))

    log(f"mode={label} training finished in {elapsed / 60.0:.2f} min")
    summary: Dict[str, Any] = {
        "mode": label,
        "metrics": metrics,
        "loss_records": recorder.to_rows(),
        "run_dir": run_dir,
        "trainable": trainable,
        "total": total,
    }
    # Drop the trainer before eval so optimizer state and cached blocks are released
    del trainer
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    release_model(None)
    if not args.skip_eval:
        summary["eval"] = evaluate_single_mode(spec, model, tokenizer, test_split, args)
    del model
    release_model(None)
    return summary


def evaluate_single_mode(
    spec: RunSpec,
    model,
    tokenizer,
    test_split,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    log(f"evaluating mode={spec.label} on GSM8K test split")
    result = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        dataset=test_split,
        mode=spec.label,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        limit=args.eval_limit if args.eval_limit > 0 else None,
        system_prompt=args.system_prompt or None,
    )
    run_dir = run_dir_for(args, spec.label)
    paths = save_eval_result(result, run_dir)
    log(
        f"mode={spec.label} accuracy={result.accuracy_pct:.2f}% "
        f"({result.num_correct}/{result.num_examples}) unparsed={result.num_unparsed}"
    )
    payload = result.to_dict(include_records=False)
    payload["paths"] = paths
    return payload


def release_model(model) -> None:
    # Callers must drop their own reference; this only clears allocator caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def compute_allocations(
    args: argparse.Namespace,
    directions: Sequence[bool],
) -> Dict[bool, AllocationResult]:
    # One model load serves every direction, which halves host memory and SVD time
    wanted = sorted(set(bool(d) for d in directions))
    results: Dict[bool, AllocationResult] = {}
    if args.allocation_json:
        log(f"loading cached allocation from {args.allocation_json}")
        cached = load_allocation(args.allocation_json)
        for direction in wanted:
            if bool(cached.high_entropy_more_rank) != direction:
                raise ValueError(
                    f"cached allocation {args.allocation_json} was built with "
                    f"high_entropy_more_rank={cached.high_entropy_more_rank}, "
                    f"but this run needs {direction}"
                )
            results[direction] = cached
        return results

    log("computing spectral entropy and rank allocation on CPU")
    # Load straight to CPU so the model is never duplicated in host memory
    model = load_base_model(args, on_cpu=True)
    start = time.time()
    for direction in wanted:
        allocation = allocate_from_model(
            model,
            uniform_rank=args.uniform_rank,
            r_min=args.r_min,
            r_max=args.r_max,
            high_entropy_more_rank=direction,
            tolerance=args.tolerance,
            target_modules=None,
            progress_every=28,
            entropy_temperature=args.entropy_temperature,
        )
        results[direction] = allocation
        log(
            f"allocation direction={direction} target={allocation.target_budget} "
            f"achieved={allocation.achieved_budget} "
            f"error={allocation.budget_error_pct:.4f}%"
        )
    log(f"allocation done in {time.time() - start:.1f}s")
    del model
    release_model(None)
    return results


def write_allocation_artifacts(
    allocation: AllocationResult,
    args: argparse.Namespace,
    inverted: bool,
) -> None:
    analysis_dir = analysis_dir_for(args, inverted)
    ensure_dir(analysis_dir)
    # Keep singular values so energy_retention plots survive a reload
    save_allocation(
        allocation,
        os.path.join(analysis_dir, "allocation.json"),
        include_spectrum_values=True,
    )
    save_json(
        {
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
            "rank_by_block": summarize_ranks_by_block_type(allocation),
            "budget_by_module": budget_breakdown(allocation),
        },
        os.path.join(analysis_dir, "allocation_summary.json"),
    )
    save_csv(
        entropy_table_rows(allocation.spectra),
        os.path.join(analysis_dir, "spectral_entropy.csv"),
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
        os.path.join(analysis_dir, "rank_allocation.csv"),
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
        os.path.join(analysis_dir, "peft_rank_pattern.json"),
    )


def write_analysis_plots(
    allocation: AllocationResult,
    args: argparse.Namespace,
    inverted: bool,
) -> None:
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_dir(plots_dir)
    suffix = "_inverted" if inverted else ""
    tag = model_tag_for(args)
    plot_entropy_heatmap(
        allocation.spectra,
        os.path.join(plots_dir, f"spectral_entropy_heatmap_{tag}.png"),
    )
    plot_entropy_by_layer(
        allocation.spectra,
        os.path.join(plots_dir, f"spectral_entropy_by_layer_{tag}.png"),
    )
    plot_entropy_by_block(
        summarize_by_block_type(allocation.spectra),
        os.path.join(plots_dir, f"spectral_entropy_by_block_{tag}.png"),
    )
    plot_rank_heatmap(
        allocation,
        os.path.join(plots_dir, f"rank_heatmap_{tag}{suffix}.png"),
    )
    plot_rank_distribution(
        allocation,
        os.path.join(plots_dir, f"rank_distribution_{tag}{suffix}.png"),
        uniform_rank=args.uniform_rank,
    )
    plot_rank_distribution_comparison(
        allocation,
        os.path.join(plots_dir, f"rank_distribution_comparison_{tag}{suffix}.png"),
        uniform_rank=args.uniform_rank,
    )
    plot_entropy_vs_rank(
        allocation,
        os.path.join(plots_dir, f"entropy_vs_rank_{tag}{suffix}.png"),
    )
    plot_energy_retention(
        allocation,
        os.path.join(plots_dir, f"energy_retention_{tag}{suffix}.png"),
    )
    log(f"analysis plots written to {plots_dir}")


def write_comparison_plots(
    run_summaries: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_dir(plots_dir)
    tag = model_tag_for(args)
    curves = {
        mode: summary.get("loss_records", [])
        for mode, summary in run_summaries.items()
        if summary.get("loss_records")
    }
    if curves:
        plot_loss_curves(curves, os.path.join(plots_dir, f"loss_curves_{tag}.png"))
    budgets = {
        mode: int(summary["trainable"])
        for mode, summary in run_summaries.items()
        if summary.get("trainable")
    }
    if budgets:
        target = budgets.get("uniform_lora") or budgets.get("uniform_pissa")
        plot_budget_comparison(
            budgets,
            os.path.join(plots_dir, f"parameter_budget_{tag}.png"),
            target_budget=target,
        )
    results = {
        mode: summary["eval"]
        for mode, summary in run_summaries.items()
        if summary.get("eval")
    }
    if results:
        plot_accuracy_comparison(
            results, os.path.join(plots_dir, f"accuracy_{tag}.png")
        )


def write_results_table(run_summaries: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> None:
    rows = []
    for mode in MODES:
        summary = run_summaries.get(mode)
        if not summary:
            continue
        metrics = summary.get("metrics", {})
        evaluation = summary.get("eval", {})
        rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "trainable_parameters": summary.get("trainable", 0),
                "trainable_pct": round(float(metrics.get("trainable_ratio_pct", 0.0)), 4),
                "final_loss": round(float(metrics.get("train_loss", 0.0)), 6),
                "train_seconds": round(float(metrics.get("train_elapsed_seconds", 0.0)), 2),
                "accuracy": round(float(evaluation.get("accuracy", 0.0)), 6),
                "accuracy_pct": round(float(evaluation.get("accuracy_pct", 0.0)), 4),
                "num_examples": int(evaluation.get("num_examples", 0)),
                "num_correct": int(evaluation.get("num_correct", 0)),
                "num_unparsed": int(evaluation.get("num_unparsed", 0)),
            }
        )
    if not rows:
        return
    results_dir = os.path.join(args.output_dir, "results")
    ensure_dir(results_dir)
    tag = model_tag_for(args)
    save_csv(
        rows,
        os.path.join(results_dir, f"comparison_{tag}.csv"),
        fieldnames=[
            "mode",
            "label",
            "trainable_parameters",
            "trainable_pct",
            "final_loss",
            "train_seconds",
            "accuracy",
            "accuracy_pct",
            "num_examples",
            "num_correct",
            "num_unparsed",
        ],
    )
    save_json(rows, os.path.join(results_dir, f"comparison_{tag}.json"))
    log(f"comparison table written to {results_dir}")


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    requested = list(MODES) if args.mode == "all" else [args.mode]
    specs = build_run_specs(requested, args.high_entropy_more_rank)
    log(
        f"modes={[s.label for s in specs]} model={args.model_name} "
        f"directions={[s.high_entropy_more_rank for s in specs]}"
    )

    tokenizer = load_tokenizer(args)
    train_split, test_split = load_gsm8k(args)
    log(f"dataset sizes train={len(train_split)} test={len(test_split)}")
    train_dataset = build_text_dataset(train_split, tokenizer, args)

    allocations: Dict[bool, AllocationResult] = {}
    entropy_specs = [s for s in specs if s.config_mode == "entropy_pissa"]
    if entropy_specs:
        allocations = compute_allocations(
            args, [s.high_entropy_more_rank for s in entropy_specs]
        )
        for direction, allocation in allocations.items():
            write_allocation_artifacts(allocation, args, inverted=not direction)
            if not args.skip_plots:
                write_analysis_plots(allocation, args, inverted=not direction)

    run_summaries: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        allocation = allocations.get(spec.high_entropy_more_rank)
        if args.skip_train:
            summary: Dict[str, Any] = {"mode": spec.label}
            if not args.skip_eval:
                model = load_trained_adapter(args, spec.label)
                summary["eval"] = evaluate_single_mode(spec, model, tokenizer, test_split, args)
                del model
                release_model(None)
        else:
            summary = train_single_mode(
                spec, args, tokenizer, train_dataset, test_split, allocation
            )
        run_summaries[spec.label] = summary

    if not args.skip_plots:
        write_comparison_plots(run_summaries, args)
    write_results_table(run_summaries, args)
    log("pipeline complete")
    return 0


def load_trained_adapter(args: argparse.Namespace, label: str):
    from peft import PeftModel

    adapter_dir = os.path.join(run_dir_for(args, label), "adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")
    model = load_base_model(args)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
