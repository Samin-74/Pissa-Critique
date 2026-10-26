from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .rank_allocator import (  # noqa: E402
    ATTENTION_MODULES,
    MLP_MODULES,
    AllocationResult,
    LayerSpectrum,
)

MODULE_ORDER: Tuple[str, ...] = ATTENTION_MODULES + MLP_MODULES

MODE_COLORS: Dict[str, str] = {
    "uniform_lora": "#4C72B0",
    "uniform_pissa": "#DD8452",
    "entropy_pissa": "#55A868",
    "entropy_pissa_inverted": "#C44E52",
}

MODE_LABELS: Dict[str, str] = {
    "uniform_lora": "Uniform LoRA (r=16)",
    "uniform_pissa": "Uniform PiSSA (r=16)",
    "entropy_pissa": "Entropy-PiSSA (high entropy first)",
    "entropy_pissa_inverted": "Entropy-PiSSA (inverted)",
}

MODE_ORDER: Tuple[str, ...] = (
    "uniform_lora",
    "uniform_pissa",
    "entropy_pissa",
    "entropy_pissa_inverted",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step and finish "
    "with the final numeric answer on its own line."
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_json(payload: Any, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(rows: Sequence[Dict[str, Any]], path: str, fieldnames: Sequence[str]) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


@dataclass
class LossRecorder:
    # Captures trainer log events step by step
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, logs: Dict[str, Any], step: Optional[int] = None) -> None:
        if not logs or "loss" not in logs:
            return
        resolved_step = logs.get("step", step)
        if resolved_step is None:
            return
        try:
            loss = float(logs["loss"])
        except (TypeError, ValueError):
            return
        if not math.isfinite(loss):
            return
        self.records.append(
            {
                "step": int(resolved_step),
                "epoch": float(logs.get("epoch", 0.0) or 0.0),
                "loss": loss,
                "learning_rate": _optional_float(logs.get("learning_rate")),
                "grad_norm": _optional_float(logs.get("grad_norm")),
                "entropy": _optional_float(logs.get("entropy")),
                "mean_token_accuracy": _optional_float(logs.get("mean_token_accuracy")),
                "num_tokens": _optional_float(logs.get("num_tokens")),
            }
        )

    def to_rows(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def save(self, json_path: str, csv_path: str) -> None:
        save_json(self.records, json_path)
        save_csv(
            self.records,
            csv_path,
            fieldnames=["step", "epoch", "loss", "learning_rate", "grad_norm"],
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_loss_callback(recorder: LossRecorder):
    from transformers import TrainerCallback

    class _LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                recorder.add(dict(logs), step=getattr(state, "global_step", None))

    return _LossCallback()


def smooth_curve(values: Sequence[float], window: int = 10) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    smoothed: List[float] = []
    running: List[float] = []
    for value in values:
        running.append(float(value))
        if len(running) > window:
            running.pop(0)
        smoothed.append(sum(running) / len(running))
    return smoothed


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
ANSWER_MARKERS: Tuple[str, ...] = (
    "####",
    "final answer is",
    "the answer is",
    "final answer:",
    "answer:",
    "answer is",
    "= ",
)


def normalize_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = str(text).strip()
    cleaned = cleaned.replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.replace("\u2212", "-").replace("\u2013", "-")
    cleaned = cleaned.strip().rstrip(".").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _last_number(text: str) -> Optional[str]:
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def extract_final_answer(text: str) -> Optional[str]:
    # Prefer the GSM8K #### marker, then explicit answer phrasing, then last number
    if not text:
        return None
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        value = _last_number(tail)
        if value is not None:
            return value
    lowered = text.lower()
    for marker in ANSWER_MARKERS[1:]:
        position = lowered.rfind(marker)
        if position == -1:
            continue
        tail = text[position + len(marker) :]
        value = _last_number(tail)
        if value is not None:
            return value
    return _last_number(text)


def answers_match(prediction: Optional[str], reference: Optional[str]) -> bool:
    if prediction is None or reference is None:
        return False
    return prediction == reference


def build_prompt(question: str, tokenizer=None, system_prompt: Optional[str] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    if system_prompt:
        return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_training_text(
    question: str,
    answer: str,
    tokenizer=None,
    system_prompt: Optional[str] = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    if system_prompt:
        return f"{system_prompt}\n\nQuestion: {question}\nAnswer: {answer}"
    return f"Question: {question}\nAnswer: {answer}"


def _spectrum_grid(
    spectra: Sequence[LayerSpectrum],
    values: Dict[str, float],
) -> Tuple[np.ndarray, List[int], List[str]]:
    layers = sorted({s.layer_index for s in spectra})
    modules = [m for m in MODULE_ORDER if any(s.module_type == m for s in spectra)]
    layer_pos = {layer: idx for idx, layer in enumerate(layers)}
    module_pos = {module: idx for idx, module in enumerate(modules)}
    grid = np.full((len(modules), len(layers)), np.nan, dtype=float)
    for spectrum in spectra:
        if spectrum.layer_index not in layer_pos or spectrum.module_type not in module_pos:
            continue
        grid[module_pos[spectrum.module_type], layer_pos[spectrum.layer_index]] = values.get(
            spectrum.name, np.nan
        )
    return grid, layers, modules


def _layer_ticks(layers: Sequence[int]) -> List[int]:
    if not layers:
        return []
    stride = max(1, len(layers) // 12)
    return list(range(0, len(layers), stride))


def _save(fig, output_path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_entropy_heatmap(
    spectra: Sequence[LayerSpectrum],
    output_path: str,
    title: str = "Spectral entropy across layers and projections",
) -> str:
    grid, layers, modules = _spectrum_grid(spectra, {s.name: s.entropy for s in spectra})
    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=200)
    image = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=9)
    ticks = _layer_ticks(layers)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(layers[i]) for i in ticks], fontsize=8)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Projection")
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Normalized spectral entropy H(W)", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_rank_heatmap(
    allocation: AllocationResult,
    output_path: str,
    title: str = "Entropy-PiSSA rank allocation",
) -> str:
    grid, layers, modules = _spectrum_grid(allocation.spectra, allocation.rank_pattern)
    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=200)
    image = ax.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=9)
    ticks = _layer_ticks(layers)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(layers[i]) for i in ticks], fontsize=8)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Projection")
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Allocated rank r", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_rank_distribution(
    allocation: AllocationResult,
    output_path: str,
    uniform_rank: int = 16,
) -> str:
    # Per-layer rank curves for each projection type
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.2), dpi=200)
    layers = sorted({s.layer_index for s in allocation.spectra})
    modules = [m for m in MODULE_ORDER if any(s.module_type == m for s in allocation.spectra)]
    lookup = {(s.layer_index, s.module_type): s.name for s in allocation.spectra}

    palette = plt.get_cmap("tab10")
    for idx, module in enumerate(modules):
        series = [
            allocation.rank_pattern.get(lookup.get((layer, module), ""), np.nan)
            for layer in layers
        ]
        axes[0].plot(
            layers,
            series,
            marker="o",
            markersize=2.6,
            linewidth=1.3,
            color=palette(idx % 10),
            label=module,
        )
    axes[0].axhline(
        uniform_rank,
        color="#4C72B0",
        linestyle="--",
        linewidth=1.6,
        label=f"uniform r={uniform_rank}",
    )
    axes[0].set_ylabel("Allocated rank")
    axes[0].set_xlabel("Transformer layer index")
    axes[0].set_title("Entropy-PiSSA per-layer rank versus uniform baseline", fontsize=12)
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", frameon=False)
    axes[0].grid(alpha=0.25, linestyle=":")

    module_means = []
    for module in modules:
        values = [
            allocation.rank_pattern[s.name]
            for s in allocation.spectra
            if s.module_type == module and s.name in allocation.rank_pattern
        ]
        module_means.append(float(np.mean(values)) if values else 0.0)
    positions = np.arange(len(modules))
    bars = axes[1].bar(
        positions,
        module_means,
        color=[palette(i % 10) for i in range(len(modules))],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axhline(uniform_rank, color="#4C72B0", linestyle="--", linewidth=1.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(modules, fontsize=9)
    axes[1].set_xlabel("Projection type")
    axes[1].set_ylabel("Mean rank")
    axes[1].set_title("Mean allocated rank per projection type", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25, linestyle=":")
    for bar, value in zip(bars, module_means):
        axes[1].annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return _save(fig, output_path)


def plot_rank_distribution_comparison(
    allocation: AllocationResult,
    output_path: str,
    uniform_rank: int = 16,
) -> str:
    # Histogram of allocated ranks against the uniform baseline
    ranks = list(allocation.rank_pattern.values())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=200)

    if ranks:
        low = int(min(ranks)) - 2
        high = int(max(ranks)) + 2
        bins = np.arange(low, high + 2, 2)
    else:
        bins = np.arange(0, 20, 2)
    axes[0].hist(
        ranks,
        bins=bins,
        color=MODE_COLORS["entropy_pissa"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].axvline(
        uniform_rank,
        color=MODE_COLORS["uniform_lora"],
        linestyle="--",
        linewidth=1.8,
        label=f"uniform r={uniform_rank}",
    )
    axes[0].set_xlabel("Allocated rank r")
    axes[0].set_ylabel("Number of projections")
    axes[0].set_title("Rank histogram (Entropy-PiSSA)", fontsize=12)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", alpha=0.25, linestyle=":")

    blocks = ["attention", "mlp"]
    data = [
        [
            allocation.rank_pattern[s.name]
            for s in allocation.spectra
            if s.block_type == block and s.name in allocation.rank_pattern
        ]
        for block in blocks
    ]
    positions = np.arange(len(blocks))
    box = axes[1].boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, color in zip(box["boxes"], ["#4C72B0", "#55A868"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1].axhline(uniform_rank, color=MODE_COLORS["uniform_lora"], linestyle="--", linewidth=1.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([b.upper() for b in blocks], fontsize=10)
    axes[1].set_ylabel("Allocated rank r")
    axes[1].set_title("Rank spread by block type", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    return _save(fig, output_path)


def plot_entropy_vs_rank(
    allocation: AllocationResult,
    output_path: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    palette = plt.get_cmap("tab10")
    modules = [m for m in MODULE_ORDER if any(s.module_type == m for s in allocation.spectra)]
    for idx, module in enumerate(modules):
        xs = [
            s.entropy
            for s in allocation.spectra
            if s.module_type == module and s.name in allocation.rank_pattern
        ]
        ys = [
            allocation.rank_pattern[s.name]
            for s in allocation.spectra
            if s.module_type == module and s.name in allocation.rank_pattern
        ]
        ax.scatter(xs, ys, s=26, alpha=0.8, color=palette(idx % 10), label=module)
    ax.set_xlabel("Normalized spectral entropy H(W)")
    ax.set_ylabel("Allocated rank r")
    ax.set_title("Entropy to rank mapping", fontsize=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_loss_curves(
    curves: Dict[str, Sequence[Dict[str, Any]]],
    output_path: str,
    smooth_window: int = 10,
) -> str:
    # Shrink the window when few points are logged so the average stays meaningful
    longest = max((len(r) for r in curves.values() if r), default=0)
    window = max(1, min(int(smooth_window), max(1, longest // 4)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)
    for mode in MODE_ORDER:
        records = curves.get(mode)
        if not records:
            continue
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        color = MODE_COLORS.get(mode, None)
        label = MODE_LABELS.get(mode, mode)
        axes[0].plot(
            steps,
            losses,
            marker="o",
            markersize=3.4,
            linewidth=1.4,
            color=color,
            label=label,
        )
        axes[1].plot(
            steps,
            smooth_curve(losses, window),
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[0].set_title("Training loss (logged points)", fontsize=12)
    axes[1].set_title(f"Training loss (moving average, window={window})", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Cross entropy loss")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_budget_comparison(
    budgets: Dict[str, int],
    output_path: str,
    target_budget: Optional[int] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    modes = [m for m in MODE_ORDER if m in budgets] or list(budgets.keys())
    values = [budgets[m] / 1e6 for m in modes]
    colors = [MODE_COLORS.get(m, "#888888") for m in modes]
    bars = ax.bar(
        range(len(modes)),
        values,
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    if target_budget:
        ax.axhline(
            target_budget / 1e6,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="uniform r=16 budget",
        )
        ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=8, rotation=12)
    ax.set_ylabel("Trainable LoRA parameters (millions)")
    ax.set_title("Parameter budget parity across modes", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    # Zoom the axis so sub-percent budget differences stay visible
    reference = target_budget / 1e6 if target_budget else max(values)
    low = min(values + [reference])
    high = max(values + [reference])
    span = max(high - low, reference * 0.01)
    ax.set_ylim(max(0.0, low - 2.0 * span), high + 2.0 * span)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}M",
            (bar.get_x() + bar.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return _save(fig, output_path)


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    labels: Optional[Dict[str, str]] = None,
    order: Optional[Sequence[str]] = None,
) -> str:
    # order lets callers plot keys that are not bare mode names
    candidates = list(order) if order is not None else list(MODE_ORDER)
    modes = [m for m in candidates if m in results and "accuracy" in results[m]]
    if not modes:
        return output_path
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    values = [100.0 * float(results[m]["accuracy"]) for m in modes]
    colors = [MODE_COLORS.get(m, "#888888") for m in modes]
    bars = ax.bar(
        range(len(modes)),
        values,
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(range(len(modes)))
    tick_labels = [
        (labels or {}).get(m) or MODE_LABELS.get(m, m) for m in modes
    ]
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=12)
    ax.set_ylabel("GSM8K exact-match accuracy (%)")
    ax.set_title("GSM8K accuracy across training modes", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return _save(fig, output_path)


def plot_entropy_by_block(
    block_summary: Dict[str, Dict[str, float]],
    output_path: str,
) -> str:
    if not block_summary:
        return output_path
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    blocks = list(block_summary.keys())
    means = [block_summary[b]["mean"] for b in blocks]
    stds = [block_summary[b]["std"] for b in blocks]
    colors = ["#4C72B0" if b == "attention" else "#55A868" for b in blocks]
    bars = ax.bar(
        range(len(blocks)),
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([b.upper() for b in blocks], fontsize=10)
    ax.set_ylabel("Mean normalized spectral entropy")
    ax.set_title("Attention versus MLP spectral entropy", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    for bar, value in zip(bars, means):
        ax.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return _save(fig, output_path)


def plot_entropy_by_layer(
    spectra: Sequence[LayerSpectrum],
    output_path: str,
) -> str:
    # Layer-wise entropy trend for attention and MLP blocks
    layers = sorted({s.layer_index for s in spectra})
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=200)
    for block, color in (("attention", "#4C72B0"), ("mlp", "#55A868")):
        means, lows, highs = [], [], []
        for layer in layers:
            values = [
                s.entropy
                for s in spectra
                if s.layer_index == layer and s.block_type == block
            ]
            if values:
                means.append(float(np.mean(values)))
                lows.append(float(np.min(values)))
                highs.append(float(np.max(values)))
            else:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
        ax.plot(layers, means, marker="o", markersize=3.2, linewidth=1.6, color=color, label=block)
        ax.fill_between(layers, lows, highs, color=color, alpha=0.16)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Normalized spectral entropy H(W)")
    ax.set_title("Layer-wise spectral entropy (mean with min-max band)", fontsize=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_energy_retention(
    allocation: AllocationResult,
    output_path: str,
) -> str:
    # Frobenius energy retained by the allocated rank per projection type
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    palette = plt.get_cmap("tab10")
    modules = [m for m in MODULE_ORDER if any(s.module_type == m for s in allocation.spectra)]
    for idx, module in enumerate(modules):
        xs, ys = [], []
        for spectrum in allocation.spectra:
            if spectrum.module_type != module or not spectrum.singular_values:
                continue
            rank = allocation.rank_pattern.get(spectrum.name)
            if rank is None:
                continue
            xs.append(rank)
            ys.append(100.0 * spectrum.energy_at(rank))
        if xs:
            ax.scatter(xs, ys, s=30, alpha=0.8, color=palette(idx % 10), label=module)
    ax.set_xlabel("Allocated rank r")
    ax.set_ylabel("Frobenius energy retained (%)")
    ax.set_title("Energy retention at the allocated rank", fontsize=12)
    ax.grid(alpha=0.25, linestyle=":")
    if any(ax.collections):
        ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, output_path)


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_param_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1e6:.3f}M"
    if value >= 1_000:
        return f"{value / 1e3:.1f}K"
    return str(value)


def gpu_memory_report() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def entropy_table_rows(spectra: Sequence[LayerSpectrum]) -> List[Dict[str, Any]]:
    rows = []
    for spectrum in sorted(spectra, key=lambda s: (s.layer_index, s.module_type)):
        rows.append(
            {
                "layer": spectrum.layer_index,
                "module": spectrum.module_type,
                "name": spectrum.name,
                "out_features": spectrum.shape[0],
                "in_features": spectrum.shape[1],
                "entropy": round(spectrum.entropy, 8),
                "effective_rank": round(spectrum.effective_rank, 4),
                "block": spectrum.block_type,
            }
        )
    return rows


def rank_table_rows(allocation: AllocationResult) -> List[Dict[str, Any]]:
    rows = []
    for spectrum in sorted(allocation.spectra, key=lambda s: (s.layer_index, s.module_type)):
        rank = allocation.rank_pattern.get(spectrum.name, 0)
        rows.append(
            {
                "layer": spectrum.layer_index,
                "module": spectrum.module_type,
                "name": spectrum.name,
                "entropy": round(spectrum.entropy, 8),
                "rank": rank,
                "uniform_rank": allocation.uniform_rank,
                "delta": rank - allocation.uniform_rank,
                "params": rank * spectrum.unit_cost,
                "energy_pct": round(100.0 * spectrum.energy_at(rank), 4),
            }
        )
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def geometric_mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values if v > 0]
    if not items:
        return 0.0
    return math.exp(sum(math.log(v) for v in items) / len(items))
