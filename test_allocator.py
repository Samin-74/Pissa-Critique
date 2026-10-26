from __future__ import annotations

import math
import sys
from typing import List, Optional

import torch

from src.rank_allocator import (
    allocate_from_model,
    allocate_ranks,
    compute_spectrum,
    detect_target_modules,
    entropy_weights,
    lora_param_count,
    round_to_even,
    spectral_entropy,
    summarize_by_block_type,
    to_peft_rank_pattern,
    uniform_budget,
)
from src.utils import answers_match, extract_final_answer, normalize_number

FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def test_entropy_bounds() -> None:
    uniform = torch.ones(8)
    check("entropy uniform equals 1", close(spectral_entropy(uniform, shape=(8, 8)), 1.0, 1e-9))

    rank_one = torch.zeros(8)
    rank_one[0] = 5.0
    check("entropy rank one equals 0", close(spectral_entropy(rank_one, shape=(8, 8)), 0.0))

    zeros = torch.zeros(8)
    check("entropy zero matrix equals 0", close(spectral_entropy(zeros, shape=(8, 8)), 0.0))

    sigma = torch.tensor([2.0, 1.0, 1.0, 0.0])
    p = sigma / sigma.sum()
    expected = float(-(p[p > 0] * torch.log2(p[p > 0])).sum()) / math.log2(4)
    check("entropy matches manual formula", close(spectral_entropy(sigma, shape=(4, 4)), expected, 1e-9))

    for trial in range(20):
        generator = torch.Generator().manual_seed(trial)
        values = torch.rand(16, generator=generator) + 1e-3
        value = spectral_entropy(values, shape=(16, 16))
        if not (0.0 <= value <= 1.0):
            check(f"entropy in range trial {trial}", False, f"got {value}")
            return
    check("entropy always in [0, 1]", True)


def test_rounding() -> None:
    check("round 4.4 to 4", round_to_even(4.4, 4) == 4)
    check("round 5.0 to 6", round_to_even(5.0, 4) == 6)
    check("round 5.1 to 6", round_to_even(5.1, 4) == 6)
    check("round 3.0 floors to 4", round_to_even(3.0, 4) == 4)
    check("round 0.0 floors to 4", round_to_even(0.0, 4) == 4)
    check("round 16.0 to 16", round_to_even(16.0, 4) == 16)


def test_weights() -> None:
    class Fake:
        def __init__(self, entropy: float):
            self.entropy = entropy

    spectra = [Fake(0.9), Fake(0.95), Fake(0.99)]
    weights = entropy_weights(spectra, high_entropy_more_rank=True, temperature=0.5)
    check("weights sum to 1", close(float(weights.sum()), 1.0, 1e-9))
    check("weights monotone in entropy", bool(weights[0] < weights[1] < weights[2]))

    inverted = entropy_weights(spectra, high_entropy_more_rank=False, temperature=0.5)
    check("inverted weights sum to 1", close(float(inverted.sum()), 1.0, 1e-9))
    check("inverted weights reversed", bool(inverted[0] > inverted[1] > inverted[2]))

    flat = entropy_weights([Fake(0.97), Fake(0.97), Fake(0.97)], True, 0.5)
    check("flat entropy gives uniform weights", all(close(float(w), 1.0 / 3.0, 1e-9) for w in flat))

    zero_temp = entropy_weights(spectra, True, 0.0)
    check("zero temperature gives uniform weights", all(close(float(w), 1.0 / 3.0, 1e-9) for w in zero_temp))


def build_model(layers: int = 4, dim: int = 128, hidden: int = 256) -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            for _ in range(layers):
                block = torch.nn.Module()
                block.self_attn = torch.nn.Module()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(block.self_attn, name, torch.nn.Linear(dim, dim, bias=False))
                block.mlp = torch.nn.Module()
                block.mlp.gate_proj = torch.nn.Linear(dim, hidden, bias=False)
                block.mlp.up_proj = torch.nn.Linear(dim, hidden, bias=False)
                block.mlp.down_proj = torch.nn.Linear(hidden, dim, bias=False)
                self.model.layers.append(block)

    torch.manual_seed(0)
    return Model()


def test_budget() -> None:
    model = build_model()
    for temperature in (0.0, 0.25, 0.5, 1.0, 2.0):
        for direction in (True, False):
            result = allocate_from_model(
                model,
                uniform_rank=16,
                r_min=4,
                entropy_temperature=temperature,
                high_entropy_more_rank=direction,
            )
            label = f"temp={temperature} hemr={direction}"
            check(f"budget within tolerance {label}", result.within_tolerance, f"{result.budget_error_pct:.4f}%")
            check(
                f"ranks even {label}",
                all(rank % 2 == 0 for rank in result.rank_pattern.values()),
            )
            check(
                f"ranks above floor {label}",
                all(rank >= 4 for rank in result.rank_pattern.values()),
            )
            check(
                f"ranks below cap {label}",
                all(
                    rank <= min(shape)
                    for rank, shape in zip(
                        result.rank_pattern.values(),
                        [s.shape for s in result.spectra],
                    )
                ),
            )

    shapes = {s.name: s.shape for s in result.spectra}
    expected = uniform_budget(shapes, 16)
    check("uniform budget matches manual sum", expected == sum(16 * (m + n) for m, n in shapes.values()))
    check("target budget equals uniform budget", result.target_budget == expected)


def test_continuous_exactness() -> None:
    model = build_model()
    result = allocate_from_model(model, uniform_rank=16, r_min=4, entropy_temperature=0.5)
    total = sum(
        result.continuous_ranks[s.name] * s.unit_cost for s in result.spectra
    )
    check(
        "continuous allocation is budget exact",
        close(total, float(result.target_budget), 1e-3),
        f"{total} vs {result.target_budget}",
    )


def test_infeasible() -> None:
    model = build_model()
    raised = False
    try:
        allocate_from_model(model, uniform_rank=16, r_min=4, r_max=8)
    except ValueError:
        raised = True
    check("infeasible r_max raises", raised)

    raised = False
    try:
        allocate_from_model(model, uniform_rank=2, r_min=4)
    except ValueError:
        raised = True
    check("uniform_rank below r_min raises", raised)

    raised = False
    try:
        allocate_ranks([], uniform_rank=16)
    except ValueError:
        raised = True
    check("empty spectra raises", raised)


def test_peft_pattern() -> None:
    model = build_model()
    result = allocate_from_model(model, uniform_rank=16, r_min=4)
    pattern = to_peft_rank_pattern(result.rank_pattern)
    check("peft pattern drops .weight", all(not key.endswith(".weight") for key in pattern))
    check("peft pattern same size", len(pattern) == len(result.rank_pattern))
    check(
        "peft pattern values preserved",
        all(pattern[k[: -len(".weight")]] == v for k, v in result.rank_pattern.items()),
    )


def test_spectrum() -> None:
    weight = torch.eye(32)
    spectrum = compute_spectrum(weight, "model.layers.0.self_attn.q_proj.weight")
    check("identity entropy is 1", close(spectrum.entropy, 1.0, 1e-9))
    check("module type parsed", spectrum.module_type == "q_proj")
    check("layer index parsed", spectrum.layer_index == 0)
    check("shape parsed", spectrum.shape == (32, 32))
    check("unit cost", spectrum.unit_cost == 64)
    check("max rank", spectrum.max_rank == 32)
    check("energy at full rank is 1", close(spectrum.energy_at(32), 1.0, 1e-9))

    low_rank = torch.zeros(32, 32)
    low_rank[0, 0] = 3.0
    low_spectrum = compute_spectrum(low_rank, "model.layers.1.mlp.down_proj.weight")
    check("rank one entropy is 0", close(low_spectrum.entropy, 0.0))
    check("rank one energy at r=2 is 1", close(low_spectrum.energy_at(2), 1.0, 1e-9))


def test_block_summary() -> None:
    model = build_model()
    result = allocate_from_model(model, uniform_rank=16, r_min=4)
    summary = summarize_by_block_type(result.spectra)
    check("block summary has attention", "attention" in summary)
    check("block summary has mlp", "mlp" in summary)
    check("attention count", int(summary["attention"]["count"]) == 16)
    check("mlp count", int(summary["mlp"]["count"]) == 12)


def test_answer_extraction() -> None:
    cases = [
        ("#### 42", "42"),
        ("The answer is 18.", "18"),
        ("Final answer: -7", "-7"),
        ("#### 1,234", "1234"),
        ("#### $1,000", "1000"),
        ("#### 12.50", "12.5"),
        ("answer: 2.0", "2"),
        ("no digits", None),
        ("", None),
    ]
    for text, expected in cases:
        check(f"extract {text[:24]!r}", extract_final_answer(text) == expected, f"got {extract_final_answer(text)}")

    check("normalize integer", normalize_number("42.0") == "42")
    check("normalize decimal", normalize_number("3.50") == "3.5")
    check("normalize comma", normalize_number("1,000") == "1000")
    check("match equal", answers_match("42", "42"))
    check("match unequal", not answers_match("42", "43"))
    check("match none", not answers_match(None, "42"))


def test_param_count() -> None:
    check("lora param count", lora_param_count((256, 256), 16) == 16 * 512)
    check("lora param count rectangular", lora_param_count((512, 256), 8) == 8 * 768)


def build_fused_model(layers: int = 3, dim: int = 128, hidden: int = 256) -> torch.nn.Module:
    # Mirrors Phi-3 style fused qkv_proj and gate_up_proj
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            for _ in range(layers):
                block = torch.nn.Module()
                block.self_attn = torch.nn.Module()
                block.self_attn.qkv_proj = torch.nn.Linear(dim, 3 * dim, bias=False)
                block.self_attn.o_proj = torch.nn.Linear(dim, dim, bias=False)
                block.mlp = torch.nn.Module()
                block.mlp.gate_up_proj = torch.nn.Linear(dim, 2 * hidden, bias=False)
                block.mlp.down_proj = torch.nn.Linear(hidden, dim, bias=False)
                self.model.layers.append(block)

    torch.manual_seed(0)
    return Model()


def test_fused_architecture() -> None:
    model = build_fused_model()
    detected = detect_target_modules(model)
    check(
        "fused detection finds qkv_proj",
        "qkv_proj" in detected,
        f"got {detected}",
    )
    check(
        "fused detection finds gate_up_proj",
        "gate_up_proj" in detected,
        f"got {detected}",
    )
    check(
        "fused detection excludes absent q_proj",
        "q_proj" not in detected,
        f"got {detected}",
    )
    check("fused detection count", len(detected) == 4, f"got {detected}")

    result = allocate_from_model(model, uniform_rank=16, r_min=4)
    check("fused allocation covers all matrices", len(result.rank_pattern) == 12)
    check("fused budget within tolerance", result.within_tolerance, f"{result.budget_error_pct:.4f}%")
    check(
        "fused ranks even",
        all(rank % 2 == 0 for rank in result.rank_pattern.values()),
    )
    check(
        "fused ranks above floor",
        all(rank >= 4 for rank in result.rank_pattern.values()),
    )

    summary = summarize_by_block_type(result.spectra)
    check("fused attention bucket populated", "attention" in summary)
    check("fused mlp bucket populated", "mlp" in summary)
    check("fused attention count", int(summary["attention"]["count"]) == 6)
    check("fused mlp count", int(summary["mlp"]["count"]) == 6)

    pattern = to_peft_rank_pattern(result.rank_pattern)
    check(
        "fused peft pattern keys",
        all("qkv_proj" in k or "gate_up_proj" in k or "o_proj" in k or "down_proj" in k for k in pattern),
    )


def test_detect_dense() -> None:
    model = build_model()
    detected = detect_target_modules(model)
    check("dense detection count", len(detected) == 7, f"got {detected}")
    check("dense detection has q_proj", "q_proj" in detected)
    check("dense detection has down_proj", "down_proj" in detected)


def main(argv: Optional[List[str]] = None) -> int:
    test_entropy_bounds()
    test_rounding()
    test_weights()
    test_budget()
    test_continuous_exactness()
    test_infeasible()
    test_peft_pattern()
    test_spectrum()
    test_block_summary()
    test_answer_extraction()
    test_param_count()
    test_detect_dense()
    test_fused_architecture()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
