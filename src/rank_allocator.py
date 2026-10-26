from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Architectures such as Phi-3 fuse q/k/v and gate/up into single matrices
FUSED_MODULES: Tuple[str, ...] = ("qkv_proj", "gate_up_proj")

ALL_MODULES: Tuple[str, ...] = TARGET_MODULES + FUSED_MODULES

ATTENTION_MODULES: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
MLP_MODULES: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")

DEFAULT_EPS = 1e-12
DEFAULT_TOLERANCE = 0.005
DEFAULT_R_MIN = 4
DEFAULT_UNIFORM_RANK = 16
DEFAULT_ENTROPY_TEMPERATURE = 0.5


@dataclass
class LayerSpectrum:
    name: str
    module_type: str
    layer_index: int
    shape: Tuple[int, int]
    entropy: float
    singular_values: List[float] = field(default_factory=list)
    top_singular_values: List[float] = field(default_factory=list)

    @property
    def is_attention(self) -> bool:
        return self.module_type in ATTENTION_MODULES

    @property
    def is_mlp(self) -> bool:
        return self.module_type in MLP_MODULES

    @property
    def block_type(self) -> str:
        if self.is_attention:
            return "attention"
        if self.is_mlp:
            return "mlp"
        return "other"

    @property
    def max_rank(self) -> int:
        return int(min(self.shape))

    @property
    def unit_cost(self) -> int:
        return int(self.shape[0]) + int(self.shape[1])

    @property
    def effective_rank(self) -> float:
        # Perplexity of the singular value distribution
        if self.max_rank <= 1:
            return 1.0
        return float(self.max_rank ** self.entropy)

    def energy_at(self, rank: int) -> float:
        # Fraction of Frobenius energy kept by the top-r truncation
        if not self.singular_values:
            return 0.0
        sigma = torch.tensor(self.singular_values, dtype=torch.float64)
        total = float((sigma**2).sum())
        if total <= 0.0:
            return 0.0
        kept = float((sigma[: max(1, int(rank))] ** 2).sum())
        return kept / total

    def to_dict(self, include_spectrum: bool = False) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "module_type": self.module_type,
            "layer_index": self.layer_index,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "entropy": float(self.entropy),
            "block_type": self.block_type,
            "max_rank": self.max_rank,
            "unit_cost": self.unit_cost,
            "effective_rank": self.effective_rank,
        }
        if include_spectrum:
            payload["singular_values"] = [float(v) for v in self.singular_values]
        return payload


@dataclass
class AllocationResult:
    rank_pattern: Dict[str, int]
    entropies: Dict[str, float]
    spectra: List[LayerSpectrum]
    target_budget: int
    achieved_budget: int
    uniform_rank: int
    high_entropy_more_rank: bool
    r_min: int
    r_max: Optional[int] = None
    continuous_ranks: Dict[str, float] = field(default_factory=dict)
    fractional_parts: Dict[str, float] = field(default_factory=dict)
    entropy_weights: Dict[str, float] = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    entropy_temperature: float = DEFAULT_ENTROPY_TEMPERATURE

    @property
    def budget_error_pct(self) -> float:
        if self.target_budget == 0:
            return 0.0
        return 100.0 * abs(self.achieved_budget - self.target_budget) / self.target_budget

    @property
    def within_tolerance(self) -> bool:
        return self.budget_error_pct <= 100.0 * self.tolerance

    @property
    def mean_rank(self) -> float:
        if not self.rank_pattern:
            return 0.0
        return sum(self.rank_pattern.values()) / len(self.rank_pattern)

    @property
    def min_rank(self) -> int:
        return min(self.rank_pattern.values()) if self.rank_pattern else 0

    @property
    def max_rank(self) -> int:
        return max(self.rank_pattern.values()) if self.rank_pattern else 0

    @property
    def rank_std(self) -> float:
        if len(self.rank_pattern) <= 1:
            return 0.0
        values = torch.tensor(list(self.rank_pattern.values()), dtype=torch.float64)
        return float(values.std(unbiased=False))

    def to_dict(
        self,
        include_spectra: bool = True,
        include_spectrum_values: bool = False,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "rank_pattern": {k: int(v) for k, v in self.rank_pattern.items()},
            "entropies": {k: float(v) for k, v in self.entropies.items()},
            "target_budget": int(self.target_budget),
            "achieved_budget": int(self.achieved_budget),
            "budget_error_pct": self.budget_error_pct,
            "within_tolerance": self.within_tolerance,
            "uniform_rank": int(self.uniform_rank),
            "mean_rank": self.mean_rank,
            "min_rank": self.min_rank,
            "max_rank": self.max_rank,
            "rank_std": self.rank_std,
            "r_min": int(self.r_min),
            "r_max": self.r_max,
            "tolerance": float(self.tolerance),
            "entropy_temperature": float(self.entropy_temperature),
            "high_entropy_more_rank": bool(self.high_entropy_more_rank),
            "num_layers": len(self.rank_pattern),
            "continuous_ranks": {k: float(v) for k, v in self.continuous_ranks.items()},
            "fractional_parts": {k: float(v) for k, v in self.fractional_parts.items()},
            "entropy_weights": {k: float(v) for k, v in self.entropy_weights.items()},
        }
        if include_spectra:
            payload["spectra"] = [
                s.to_dict(include_spectrum=include_spectrum_values) for s in self.spectra
            ]
        return payload


def module_type_of(name: str) -> str:
    parts = name.split(".")
    for part in reversed(parts):
        if part in ALL_MODULES:
            return part
    return parts[-1] if parts else name


def detect_target_modules(model: torch.nn.Module) -> Tuple[str, ...]:
    # Returns the projection names this architecture actually exposes
    present = set()
    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.dim() != 2:
            continue
        module = module_type_of(name)
        if module in ALL_MODULES:
            present.add(module)
    ordered = tuple(m for m in ALL_MODULES if m in present)
    if not ordered:
        raise ValueError("no known projection modules found in model")
    return ordered


def layer_index_of(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part in ("layers", "h", "blocks") and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                continue
    return -1


def spectral_entropy(
    singular_values: torch.Tensor,
    shape: Optional[Tuple[int, int]] = None,
    eps: float = DEFAULT_EPS,
) -> float:
    # Normalized spectral entropy H(W) in [0, 1]
    if singular_values.numel() == 0:
        return 0.0
    sigma = singular_values.detach().to(device="cpu", dtype=torch.float64).abs()
    if shape is not None:
        normalizer_rank = int(min(int(shape[0]), int(shape[1])))
    else:
        normalizer_rank = int(sigma.numel())
    if normalizer_rank <= 1:
        return 0.0
    total = float(sigma.sum())
    if total <= eps:
        return 0.0
    p = sigma / (total + eps)
    p = p[p > 0]
    if p.numel() <= 1:
        return 0.0
    entropy = float(-(p * torch.log2(p + eps)).sum())
    normalizer = math.log2(normalizer_rank)
    if normalizer <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, entropy / normalizer)))


def compute_spectrum(
    weight: torch.Tensor,
    name: str,
    eps: float = DEFAULT_EPS,
    top_k: int = 0,
    keep_spectrum: bool = True,
) -> LayerSpectrum:
    # CPU-only SVD keeps GPU memory free
    matrix = weight.detach().to(device="cpu", dtype=torch.float32)
    if matrix.dim() != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    sigma = torch.linalg.svdvals(matrix)
    entropy = spectral_entropy(sigma, shape=shape, eps=eps)
    values = [float(v) for v in sigma]
    top_values = values[: int(top_k)] if top_k > 0 else []
    return LayerSpectrum(
        name=name,
        module_type=module_type_of(name),
        layer_index=layer_index_of(name),
        shape=shape,
        entropy=entropy,
        singular_values=values if keep_spectrum else [],
        top_singular_values=top_values,
    )


def collect_target_weights(
    model: torch.nn.Module,
    target_modules: Sequence[str] = TARGET_MODULES,
) -> Dict[str, torch.Tensor]:
    # Returns detached references, conversion to CPU happens per matrix
    targets = set(target_modules)
    weights: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if module_type_of(name) not in targets:
            continue
        if param.dim() != 2:
            continue
        weights[name] = param.detach()
    return weights


def compute_spectra(
    weights: Mapping[str, torch.Tensor],
    eps: float = DEFAULT_EPS,
    top_k: int = 0,
    progress_every: int = 0,
    keep_spectrum: bool = True,
) -> List[LayerSpectrum]:
    spectra: List[LayerSpectrum] = []
    names = sorted(weights.keys())
    for idx, name in enumerate(names, start=1):
        spectra.append(
            compute_spectrum(
                weights[name],
                name=name,
                eps=eps,
                top_k=top_k,
                keep_spectrum=keep_spectrum,
            )
        )
        if progress_every and idx % progress_every == 0:
            print(f"[spectral] {idx}/{len(names)} matrices processed", flush=True)
    return spectra


def lora_param_count(shape: Tuple[int, int], rank: int) -> int:
    out_features, in_features = shape
    return int(rank) * (int(out_features) + int(in_features))


def uniform_budget(shapes: Mapping[str, Tuple[int, int]], uniform_rank: int) -> int:
    return sum(lora_param_count(shape, uniform_rank) for shape in shapes.values())


def round_to_even(value: float, minimum: int = 2) -> int:
    # Nearest even integer with a hard floor
    rounded = int(math.floor(float(value) / 2.0 + 0.5)) * 2
    return max(int(minimum), rounded)


def entropy_weights(
    spectra: Sequence[LayerSpectrum],
    high_entropy_more_rank: bool = True,
    temperature: float = DEFAULT_ENTROPY_TEMPERATURE,
) -> torch.Tensor:
    # Scale-invariant softmax over standardized entropy
    count = len(spectra)
    if count == 0:
        return torch.zeros(0, dtype=torch.float64)
    entropies = torch.tensor([s.entropy for s in spectra], dtype=torch.float64)
    centered = entropies - entropies.mean()
    std = float(centered.std(unbiased=False))
    if std <= 1e-12:
        return torch.full((count,), 1.0 / count, dtype=torch.float64)
    z = centered / std
    if not high_entropy_more_rank:
        z = -z
    logits = float(temperature) * z
    logits = logits - logits.max()
    weights = torch.exp(logits)
    return weights / weights.sum()


def _rank_caps(
    spectra: Sequence[LayerSpectrum],
    r_max: Optional[int],
) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for spectrum in spectra:
        cap = spectrum.max_rank if r_max is None else min(int(r_max), spectrum.max_rank)
        if cap % 2 != 0:
            cap -= 1
        caps[spectrum.name] = max(2, cap)
    return caps


def _allocate_continuous(
    spectra: Sequence[LayerSpectrum],
    target_budget: int,
    r_min: int,
    r_max: Optional[int],
    high_entropy_more_rank: bool,
    temperature: float,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int], Dict[str, float]]:
    # Rank_i = r_min + scale * (r_uniform - r_min) * N * w_i, scale keeps the budget exact
    weights = entropy_weights(spectra, high_entropy_more_rank, temperature)
    caps = _rank_caps(spectra, r_max)
    floors = {s.name: min(int(r_min), caps[s.name]) for s in spectra}
    count = len(spectra)
    total_cost = sum(s.unit_cost for s in spectra)
    weighted_cost = float(
        sum(float(w) * s.unit_cost for w, s in zip(weights.tolist(), spectra))
    )
    if weighted_cost <= 0.0:
        scale = 1.0
    else:
        scale = float(total_cost) / (float(count) * weighted_cost)

    span = float(target_budget) / float(total_cost) - float(r_min)
    continuous: Dict[str, float] = {}
    for spectrum, weight in zip(spectra, weights.tolist()):
        rank = float(r_min) + scale * span * float(count) * float(weight)
        continuous[spectrum.name] = float(
            min(float(caps[spectrum.name]), max(float(floors[spectrum.name]), rank))
        )
    weight_map = {s.name: float(w) for s, w in zip(spectra, weights.tolist())}
    return continuous, floors, caps, weight_map


def _budget_of(ranks: Mapping[str, int], spectra: Sequence[LayerSpectrum]) -> int:
    lookup = {s.name: s.unit_cost for s in spectra}
    return sum(int(rank) * lookup[name] for name, rank in ranks.items())


def _greedy_adjust(
    ranks: Dict[str, int],
    spectra: Sequence[LayerSpectrum],
    residual: Mapping[str, float],
    target_budget: int,
    floors: Mapping[str, int],
    caps: Mapping[str, int],
    tolerance: float,
) -> Dict[str, int]:
    # Largest-remainder repair: one step per layer per pass, best residual first
    lookup = {s.name: s.unit_cost for s in spectra}
    current = _budget_of(ranks, spectra)
    limit = abs(float(target_budget)) * float(tolerance)
    grow_order = sorted(ranks.keys(), key=lambda n: residual.get(n, 0.0), reverse=True)
    shrink_order = sorted(ranks.keys(), key=lambda n: residual.get(n, 0.0))
    max_passes = 4 * len(ranks) + 64

    for _ in range(max_passes):
        error = current - target_budget
        if abs(error) <= limit:
            break
        step = 2 if error < 0 else -2
        order = grow_order if step > 0 else shrink_order
        moved = False
        for name in order:
            if abs(current - target_budget) <= limit:
                break
            candidate = ranks[name] + step
            if candidate < floors[name] or candidate > caps[name]:
                continue
            if abs(current + step * lookup[name] - target_budget) >= abs(current - target_budget):
                continue
            ranks[name] = candidate
            current += step * lookup[name]
            moved = True
        if not moved:
            break
    return ranks


def allocate_ranks(
    spectra: Sequence[LayerSpectrum],
    uniform_rank: int = DEFAULT_UNIFORM_RANK,
    r_min: int = DEFAULT_R_MIN,
    r_max: Optional[int] = None,
    high_entropy_more_rank: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    eps: float = DEFAULT_EPS,
    entropy_temperature: float = DEFAULT_ENTROPY_TEMPERATURE,
) -> AllocationResult:
    if not spectra:
        raise ValueError("spectra must contain at least one layer")
    if r_min < 2 or r_min % 2 != 0:
        raise ValueError("r_min must be an even integer >= 2")
    if uniform_rank < r_min:
        raise ValueError("uniform_rank must be >= r_min")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if entropy_temperature < 0.0:
        raise ValueError("entropy_temperature must be non-negative")

    shapes = {s.name: s.shape for s in spectra}
    target_budget = uniform_budget(shapes, uniform_rank)

    caps = _rank_caps(spectra, r_max)
    min_budget = sum(min(int(r_min), caps[s.name]) * s.unit_cost for s in spectra)
    max_budget = sum(caps[s.name] * s.unit_cost for s in spectra)
    if target_budget < min_budget or target_budget > max_budget:
        raise ValueError(
            "uniform budget is infeasible under the rank constraints: "
            f"target={target_budget} feasible=[{min_budget}, {max_budget}]"
        )

    continuous, floors, caps, weight_map = _allocate_continuous(
        spectra=spectra,
        target_budget=target_budget,
        r_min=r_min,
        r_max=r_max,
        high_entropy_more_rank=high_entropy_more_rank,
        temperature=entropy_temperature,
    )

    ranks: Dict[str, int] = {}
    residual: Dict[str, float] = {}
    for name, value in continuous.items():
        rounded = min(caps[name], round_to_even(value, floors[name]))
        ranks[name] = rounded
        # Positive residual means the layer was rounded down and deserves the next step
        residual[name] = float(value) - float(rounded)

    ranks = _greedy_adjust(
        ranks=ranks,
        spectra=spectra,
        residual=residual,
        target_budget=target_budget,
        floors=floors,
        caps=caps,
        tolerance=tolerance,
    )

    achieved = _budget_of(ranks, spectra)
    return AllocationResult(
        rank_pattern=ranks,
        entropies={s.name: s.entropy for s in spectra},
        spectra=list(spectra),
        target_budget=target_budget,
        achieved_budget=achieved,
        uniform_rank=int(uniform_rank),
        high_entropy_more_rank=bool(high_entropy_more_rank),
        r_min=int(r_min),
        r_max=None if r_max is None else int(r_max),
        continuous_ranks=continuous,
        fractional_parts=residual,
        entropy_weights=weight_map,
        tolerance=float(tolerance),
        entropy_temperature=float(entropy_temperature),
    )


def allocate_from_model(
    model: torch.nn.Module,
    uniform_rank: int = DEFAULT_UNIFORM_RANK,
    r_min: int = DEFAULT_R_MIN,
    r_max: Optional[int] = None,
    high_entropy_more_rank: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    target_modules: Optional[Sequence[str]] = None,
    eps: float = DEFAULT_EPS,
    top_k: int = 0,
    progress_every: int = 0,
    entropy_temperature: float = DEFAULT_ENTROPY_TEMPERATURE,
    keep_spectrum: bool = True,
) -> AllocationResult:
    # None means auto-detect, which handles fused architectures like Phi-3
    resolved = tuple(target_modules) if target_modules else detect_target_modules(model)
    weights = collect_target_weights(model, target_modules=resolved)
    if not weights:
        raise ValueError("no target projection weights found in model")
    spectra = compute_spectra(
        weights,
        eps=eps,
        top_k=top_k,
        progress_every=progress_every,
        keep_spectrum=keep_spectrum,
    )
    return allocate_ranks(
        spectra=spectra,
        uniform_rank=uniform_rank,
        r_min=r_min,
        r_max=r_max,
        high_entropy_more_rank=high_entropy_more_rank,
        tolerance=tolerance,
        eps=eps,
        entropy_temperature=entropy_temperature,
    )


def to_peft_rank_pattern(rank_pattern: Mapping[str, int]) -> Dict[str, int]:
    # PEFT matches module names without the trailing .weight suffix
    pattern: Dict[str, int] = {}
    for name, rank in rank_pattern.items():
        key = name[: -len(".weight")] if name.endswith(".weight") else name
        pattern[key] = int(rank)
    return pattern


def to_peft_alpha_pattern(
    rank_pattern: Mapping[str, int],
    multiplier: float = 2.0,
) -> Dict[str, int]:
    pattern: Dict[str, int] = {}
    for name, rank in rank_pattern.items():
        key = name[: -len(".weight")] if name.endswith(".weight") else name
        pattern[key] = max(1, int(round(float(multiplier) * int(rank))))
    return pattern


def summarize_by_module(spectra: Sequence[LayerSpectrum]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for spectrum in spectra:
        buckets.setdefault(spectrum.module_type, []).append(spectrum.entropy)
    return {module: _stats(values) for module, values in sorted(buckets.items())}


def summarize_by_block_type(spectra: Sequence[LayerSpectrum]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {"attention": [], "mlp": []}
    for spectrum in spectra:
        if spectrum.is_attention:
            buckets["attention"].append(spectrum.entropy)
        elif spectrum.is_mlp:
            buckets["mlp"].append(spectrum.entropy)
    return {block: _stats(values) for block, values in buckets.items() if values}


def summarize_ranks_by_block_type(
    allocation: AllocationResult,
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {"attention": [], "mlp": []}
    for spectrum in allocation.spectra:
        if spectrum.name not in allocation.rank_pattern:
            continue
        if spectrum.is_attention:
            buckets["attention"].append(float(allocation.rank_pattern[spectrum.name]))
        elif spectrum.is_mlp:
            buckets["mlp"].append(float(allocation.rank_pattern[spectrum.name]))
    return {block: _stats(values) for block, values in buckets.items() if values}


def summarize_ranks_by_module(
    allocation: AllocationResult,
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for spectrum in allocation.spectra:
        if spectrum.name not in allocation.rank_pattern:
            continue
        buckets.setdefault(spectrum.module_type, []).append(
            float(allocation.rank_pattern[spectrum.name])
        )
    return {module: _stats(values) for module, values in sorted(buckets.items())}


def budget_breakdown(allocation: AllocationResult) -> Dict[str, Dict[str, float]]:
    # Parameter spend per projection type
    buckets: Dict[str, Dict[str, float]] = {}
    for spectrum in allocation.spectra:
        rank = allocation.rank_pattern.get(spectrum.name)
        if rank is None:
            continue
        entry = buckets.setdefault(
            spectrum.module_type,
            {"params": 0.0, "uniform_params": 0.0, "mean_rank": 0.0, "count": 0.0},
        )
        entry["params"] += float(rank * spectrum.unit_cost)
        entry["uniform_params"] += float(allocation.uniform_rank * spectrum.unit_cost)
        entry["mean_rank"] += float(rank)
        entry["count"] += 1.0
    for entry in buckets.values():
        if entry["count"] > 0:
            entry["mean_rank"] /= entry["count"]
    return buckets


def _stats(values: Sequence[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "count": float(len(values)),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)) if len(values) > 1 else 0.0,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def energy_captured(weight: torch.Tensor, rank: int) -> float:
    # Fraction of Frobenius energy kept by the top-r truncation
    matrix = weight.detach().to(device="cpu", dtype=torch.float32)
    if matrix.dim() != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    sigma = torch.linalg.svdvals(matrix).double()
    total = float((sigma**2).sum())
    if total <= 0.0:
        return 0.0
    kept = float((sigma[: max(1, int(rank))] ** 2).sum())
    return kept / total


def save_allocation(
    result: AllocationResult,
    path: str,
    include_spectrum_values: bool = False,
) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            result.to_dict(
                include_spectra=True,
                include_spectrum_values=include_spectrum_values,
            ),
            handle,
            indent=2,
            ensure_ascii=True,
        )
    return path


def load_allocation(path: str) -> AllocationResult:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    spectra = [
        LayerSpectrum(
            name=item["name"],
            module_type=item["module_type"],
            layer_index=int(item["layer_index"]),
            shape=(int(item["shape"][0]), int(item["shape"][1])),
            entropy=float(item["entropy"]),
            singular_values=[float(v) for v in item.get("singular_values", [])],
        )
        for item in payload.get("spectra", [])
    ]
    return AllocationResult(
        rank_pattern={k: int(v) for k, v in payload["rank_pattern"].items()},
        entropies={k: float(v) for k, v in payload.get("entropies", {}).items()},
        spectra=spectra,
        target_budget=int(payload["target_budget"]),
        achieved_budget=int(payload["achieved_budget"]),
        uniform_rank=int(payload.get("uniform_rank", DEFAULT_UNIFORM_RANK)),
        high_entropy_more_rank=bool(payload.get("high_entropy_more_rank", True)),
        r_min=int(payload.get("r_min", DEFAULT_R_MIN)),
        r_max=payload.get("r_max"),
        continuous_ranks={
            k: float(v) for k, v in payload.get("continuous_ranks", {}).items()
        },
        fractional_parts={
            k: float(v) for k, v in payload.get("fractional_parts", {}).items()
        },
        entropy_weights={
            k: float(v) for k, v in payload.get("entropy_weights", {}).items()
        },
        tolerance=float(payload.get("tolerance", DEFAULT_TOLERANCE)),
        entropy_temperature=float(
            payload.get("entropy_temperature", DEFAULT_ENTROPY_TEMPERATURE)
        ),
    )
