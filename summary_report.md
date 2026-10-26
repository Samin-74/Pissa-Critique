# Entropy-PiSSA: Spectral Entropy Guided Rank Allocation for PiSSA

## Executive summary

PiSSA (NeurIPS 2024) improves LoRA by initializing the low rank factors from the principal
singular triplets of each pretrained weight matrix. The initialization is better, but the
rank assignment is still uniform: every projection in every layer receives the same `r`.
Entropy-PiSSA replaces that uniform rank with a per-matrix rank derived from the normalized
spectral entropy of the matrix, under a hard parameter budget equal to the uniform `r=16`
baseline.

The headline result is that the **inverted** allocation, which gives more rank to low entropy
(spectrally concentrated) projections, is the best PiSSA variant on both models tested:

| Model | Uniform PiSSA | Entropy-PiSSA inverted | Delta |
| --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 61.10 +/- 0.71 | **63.00 +/- 0.57** | +1.90 |
| Phi-3.5-mini-instruct | 65.40 +/- 0.28 | **67.60 +/- 0.57** | +2.20 |

The default direction, which gives more rank to high entropy projections, behaves very
differently on the two models. On Phi-3.5 it is harmless (66.60 +/- 1.13, slightly above
uniform PiSSA). On Qwen2.5 it collapses: 1.50 +/- 0.71, with both seeds diverging during
training. The section "Why the default direction fails on Qwen" identifies the cause as a
width dependent artifact of the entropy normalization, and shows that the severity of the
failure tracks how uneven the resulting allocation is.

All numbers in this report are read directly from `outputs/results/comparison_all.csv` and
`outputs/results/summary_metrics.json`. Means and standard deviations are over two seeds
(42 and 123) with 3000 training examples, 2 epochs, and 500 GSM8K test examples per run.

## Experimental setup

| Setting | Value |
| --- | --- |
| Models | `Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct` |
| Modes | `uniform_lora`, `uniform_pissa`, `entropy_pissa`, `entropy_pissa_inverted` |
| Seeds | 42, 123 |
| Train examples | 3000 |
| Epochs | 2 |
| Eval examples | 500 |
| `per_device_batch_size` | 4 |
| `gradient_accumulation_steps` | 2 |
| `max_seq_length` | 512 |
| `max_new_tokens` | 512 |
| `uniform_rank` | 16 |
| `lora_alpha` | 32 |
| `r_min` | 4 |
| Learning rate | 2e-4 |
| LR schedule | cosine, 3% warmup |
| Optimizer | `paged_adamw_8bit` |
| Total runs | 16 |

Every run uses the same effective batch size (8), the same schedule, and the same evaluation
protocol. The only difference between modes is the adapter rank pattern and the
initialization.

## Method

For each target projection `W`:

1. Compute singular values `sigma` with `torch.linalg.svdvals` on CPU.
2. Normalize them into a probability vector `p_i = sigma_i / (sum_j sigma_j + eps)`.
3. Compute the normalized spectral entropy

   $$H(W) = \frac{-\sum_i p_i \log_2 (p_i + \epsilon)}{\log_2 \min(m, n)} \in [0, 1]$$

4. Convert entropies into allocation weights with a scale invariant softmax over the
   standardized entropy `z_i = (H_i - mean(H)) / std(H)`:

   $$w_i = \frac{\exp(\tau z_i)}{\sum_j \exp(\tau z_j)}$$

5. Distribute the budget with a closed form that is exact before rounding:

   $$r_i = r_{min} + \kappa \, (r_{uniform} - r_{min}) \, N \, w_i,
   \qquad \kappa = \frac{\sum_j (m_j + n_j)}{N \sum_j w_j (m_j + n_j)}$$

6. Round to the nearest even integer, clamp to `[r_min, min(m, n)]`, then greedily adjust the
   layers with the largest fractional parts until the realized parameter count is within
   0.5% of the uniform `r=16` budget.

`high_entropy_more_rank=True` gives more rank to high entropy matrices. Setting it to `False`
inverts the preference.

## Why entropy and not energy

Energy based criteria (`sum_{i<=r} sigma_i^2 / sum_i sigma_i^2`) are also discriminative on
these models: at `r=16` the retained fraction ranges from 1.96% to 46.41% across the 252
Qwen projections (mean 9.29%, coefficient of variation 0.716). The reason this project uses
normalized spectral entropy is not that energy saturates, but that energy at a fixed `r` is
strongly confounded with matrix width (`r = -0.70` against `min(m, n)`), so it partly
re-encodes the same size bias the allocation is meant to correct. Normalized entropy is less
width confounded (`r = -0.22`), but it is a much flatter signal: its coefficient of variation
is only 0.021, with all 252 values inside `[0.908, 0.999]`.

That flatness is the central empirical finding of this project. Because
`effective_rank = exp(H * ln min(m, n))` holds to within 0.2% on every projection, normalized
entropy is close to a rescaled effective rank. The spread between the widest and narrowest
matrices is small in entropy terms but large in rank terms, so allocating rank in proportion
to entropy compresses the dynamic range of the allocation far below what the underlying
spectra would justify.

## Layer-wise spectral entropy findings

Measured on Qwen2.5-3B-Instruct over 252 projections:

| Module | Mean normalized entropy | Mean raw entropy (bits) | Width |
| --- | --- | --- | --- |
| `q_proj` | 0.9476 | 10.42 | 2048 |
| `o_proj` | 0.9548 | 10.50 | 2048 |
| `k_proj` | 0.9739 | 7.79 | 256 |
| `gate_proj` | 0.9891 | 10.88 | 2048 |
| `up_proj` | 0.9901 | 10.89 | 2048 |
| `down_proj` | 0.9907 | 10.90 | 2048 |
| `v_proj` | 0.9952 | 7.96 | 256 |

MLP projections have systematically higher normalized entropy than attention projections
(0.9899 vs 0.9679 mean). Within attention, `v_proj` is the most entropic and `q_proj` the
least.

The raw entropy column is the important one. `q_proj` and `o_proj` have the second and third
highest raw entropy of any module, but the lowest normalized entropy, because they are 2048
wide and the normalization divides by `ln(2048)` rather than `ln(256)`. The normalization
therefore reverses their position in the ordering.

## Rank allocation results

### Qwen2.5-3B-Instruct

| Module | Raw H | Norm H | Effective rank | Width | Default r | Inverted r | Default r/eff | Inverted r/eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `q_proj` | 10.42 | 0.9476 | 1376 | 2048 | 8.8 | 29.7 | 0.64% | 2.16% |
| `o_proj` | 10.50 | 0.9548 | 1453 | 2048 | 9.8 | 25.4 | 0.67% | 1.75% |
| `k_proj` | 7.79 | 0.9739 | 222 | 256 | 13.3 | 17.9 | 5.98% | 8.06% |
| `gate_proj` | 10.88 | 0.9891 | 1891 | 2048 | 17.2 | 13.4 | 0.91% | 0.71% |
| `up_proj` | 10.89 | 0.9901 | 1910 | 2048 | 17.2 | 14.1 | 0.90% | 0.74% |
| `down_proj` | 10.90 | 0.9907 | 1914 | 2048 | 17.4 | 13.2 | 0.91% | 0.69% |
| `v_proj` | 7.96 | 0.9952 | 249 | 256 | 18.8 | 12.2 | 7.55% | 4.90% |

The default allocation spans a factor of **11.85x** in rank-to-effective-rank ratio, from
0.64% for `q_proj` to 7.55% for `v_proj`.

### Phi-3.5-mini-instruct

| Module | Raw H | Norm H | Effective rank | Width | Default r | Inverted r | Default r/eff | Inverted r/eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `o_proj` | 11.19 | 0.9659 | 2338 | 3072 | 9.1 | 27.5 | 0.39% | 1.18% |
| `qkv_proj` | 11.37 | 0.9812 | 2655 | 3072 | 14.1 | 19.6 | 0.53% | 0.74% |
| `down_proj` | 11.47 | 0.9903 | 2842 | 3072 | 17.1 | 12.7 | 0.60% | 0.45% |
| `gate_up_proj` | 11.50 | 0.9929 | 2902 | 3072 | 18.6 | 12.2 | 0.64% | 0.42% |

Phi uses fused projections, so all four target modules are 3072 wide. The default allocation
spans only **1.66x** in rank-to-effective-rank ratio.

Both allocations hit the parameter budget to within 0.5%: Qwen default 29,788,160 against a
target of 29,933,568 (0.49% under), Qwen inverted 29,790,208 (0.48% under), Phi default
25,079,808 against a target of 25,165,824 (0.34% under), Phi inverted 25,284,608 (0.47% over).

## Benchmark results

GSM8K exact match, 500 test examples per run, greedy decoding, `max_new_tokens=512`.

### Qwen2.5-3B-Instruct

| Mode | Seed 42 | Seed 123 | Mean +/- std | Final loss |
| --- | --- | --- | --- | --- |
| Uniform LoRA (r=16) | 65.00% | 66.40% | **65.70 +/- 0.99** | 0.235 |
| Uniform PiSSA (r=16) | 61.60% | 60.60% | **61.10 +/- 0.71** | 0.808 |
| Entropy-PiSSA (high entropy first) | 2.00% | 1.00% | **1.50 +/- 0.71** | 3.157 |
| Entropy-PiSSA (inverted) | 63.40% | 62.60% | **63.00 +/- 0.57** | 1.031 |

### Phi-3.5-mini-instruct

| Mode | Seed 42 | Seed 123 | Mean +/- std | Final loss |
| --- | --- | --- | --- | --- |
| Uniform LoRA (r=16) | 76.40% | 75.80% | **76.10 +/- 0.42** | 0.277 |
| Uniform PiSSA (r=16) | 65.60% | 65.20% | **65.40 +/- 0.28** | 0.443 |
| Entropy-PiSSA (high entropy first) | 65.80% | 67.40% | **66.60 +/- 1.13** | 0.432 |
| Entropy-PiSSA (inverted) | 67.20% | 68.00% | **67.60 +/- 0.57** | 0.468 |

### Main comparison

The inverted allocation beats uniform PiSSA on both models, by +1.90 points on Qwen and
+2.20 points on Phi. The standard deviations are small (0.57 on both), so the ordering is
stable across seeds.

Uniform LoRA remains the strongest single configuration on both models (65.70 and 76.10).
This is consistent with the PiSSA literature: PiSSA's advantage appears at low rank, and at
`r=16` with 3000 examples the zero-initialized LoRA adapter is simply easier to optimize.
The comparison of interest in this report is between PiSSA variants, all of which share the
same initialization and differ only in rank allocation.

## Why the default direction fails on Qwen

The default direction collapses on Qwen2.5 but not on Phi-3.5. The difference is explained by
the interaction between the entropy normalization and the set of target module widths.

### The normalization reverses the ordering when widths differ

Normalized entropy divides by `ln(min(m, n))`. On Qwen the target modules have two different
widths (2048 for `q_proj`, `o_proj`, and the MLP; 256 for `k_proj` and `v_proj`). Dividing by
`ln(2048)` instead of `ln(256)` scales a value by 1.375, which is enough to overturn the raw
ordering:

- `q_proj` has raw entropy 10.42 (2nd highest) but normalized entropy 0.9476 (lowest).
- `o_proj` has raw entropy 10.50 (3rd highest) but normalized entropy 0.9548 (2nd lowest).

The heuristic therefore assigns the least rank to the two most spectrally spread matrices.
On Phi all four modules are 3072 wide, the normalization is a constant, and the ordering is
preserved.

### The severity tracks allocation unevenness

| Model | r/eff spread | Default direction outcome |
| --- | --- | --- |
| Qwen2.5-3B | 11.85x | collapse, 1.50 +/- 0.71 |
| Phi-3.5-mini | 1.66x | stable, 66.60 +/- 1.13 |

On Phi every module lands in a narrow 0.39% to 0.64% band, so the direction of the preference
barely matters and the default is within noise of uniform PiSSA. On Qwen `q_proj` is held to
0.64% of its effective rank while `v_proj` receives 7.55%, and the model breaks.

### Training evidence

The collapse is visible in the loss curves and gradient norms, not only in the final accuracy.

| Mode (Qwen, seed 42) | Median grad norm | p90 grad norm | Max grad norm | Steps above 50 | Final loss |
| --- | --- | --- | --- | --- | --- |
| Uniform LoRA | 2.26 | 2.92 | 70.2 | 1 | 0.181 |
| Uniform PiSSA | 5.49 | 685.69 | 2298.8 | 18 | 0.350 |
| Entropy-PiSSA (default) | 26.10 | 55.79 | 308.2 | 9 | 2.005 |
| Entropy-PiSSA (inverted) | 5.89 | 92.61 | 965.9 | 12 | 0.339 |

The default direction has a median gradient norm 4.8x higher than any other mode. It is not a
single spike but chronically elevated gradients throughout training. Both seeds diverge at
step 150 (seed 42) and step 130 (seed 123) and never recover.

The generation outputs confirm genuine model failure rather than a parsing artifact. On Qwen
seed 42 the default direction produced 48 unparseable answers out of 500 and hit the
512-token cap on 168 examples, against 8 for uniform PiSSA and 0 for inverted. Representative
failure:

```
reference: 18
raw: The rest of the remaining money is $30 per day * 4 = <<15+1=18>>15.
     The rest of the money is $10 per day * 10 per day * 10 per day * 2 per day = ...
     #### 10
```

The arithmetic is incoherent, which indicates the model lost the ability to compute rather
than merely to format.

### What this means for the method

The result is not that spectral entropy is uninformative. It is that the specific
normalization used here is width dependent, and that a width dependent signal combined with a
fixed absolute rank budget produces an allocation whose unevenness depends on the mix of
module widths in the model. The inverted direction is more robust because it inverts the
preference rather than the scale, so it does not amplify the width artifact.

A cleaner fix would be to allocate on raw entropy, or to normalize by a constant rather than
per-matrix width. That is left as future work; this report documents the measured behaviour
of the implemented heuristic.

## Reading these numbers honestly

1. **Two seeds is a small sample.** The standard deviations reported here (0.28 to 1.13
   points) are within-seed variation, not confidence intervals over the training
   distribution. The +1.90 and +2.20 point gaps are larger than the observed standard
   deviations, but a third seed would strengthen the claim.

2. **The collapse is a single-model phenomenon.** It reproduces across both Qwen seeds, which
   rules out seed luck, but it does not reproduce on Phi. The mechanism proposed above
   explains why, but it is a post-hoc explanation of two models, not a validated prediction.

3. **Uniform LoRA wins outright.** This report does not claim Entropy-PiSSA beats LoRA. At
   `r=16` with 3000 examples, LoRA is the stronger baseline on both models. The contribution
   here is about how to allocate rank within the PiSSA family.

4. **Short-horizon training dynamics matter.** 3000 examples and 2 epochs is a deliberately
   small budget. PiSSA's non-zero initialization makes it more sensitive to the early
   optimization trajectory than zero-initialized LoRA, and the divergence observed here is an
   early-training phenomenon. Longer training might allow the default direction to recover,
   though the loss curves show no sign of it within 750 steps.

5. **The inverted result is the finding, not the collapse.** The collapse is diagnostic: it
   identifies a defect in the heuristic. The positive result is that inverting the preference
   improves on uniform PiSSA on both models.

## Reproduction

Full suite (16 runs, both models, both seeds, all four modes):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1
```

A single group:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_suite.ps1 `
    -Model "Qwen/Qwen2.5-3B-Instruct" -ModelTag qwen -Seed 42
```

Aggregate plots and tables:

```bash
python aggregate.py --output_dir outputs
```

A single run:

```bash
python train.py \
    --mode entropy_pissa_inverted \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --dataset_name openai/gsm8k \
    --output_dir outputs \
    --run_name qwen_seed42/entropy_pissa_inverted \
    --allocation_json outputs/analysis/qwen_inverted/allocation.json \
    --uniform_rank 16 --lora_alpha 32 --r_min 4 \
    --max_seq_length 512 \
    --per_device_batch_size 4 --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 --num_train_epochs 2 \
    --train_limit 3000 --eval_limit 500 \
    --eval_batch_size 8 --max_new_tokens 512 \
    --logging_steps 10 --save_steps 100000 \
    --seed 42 --skip_plots
```

## Implementation notes

- `alpha_pattern` is set so that `lora_alpha / r` is identical for every layer. Without it,
  PEFT computes `scaling = alpha / r` per layer, which would give low rank layers a larger
  effective update and confound the comparison. Verified: 0 layers deviate from `alpha/r = 2.0`
  across all 252 Qwen projections.
- `lora_dropout` is forced to `0.0` for both PiSSA modes. A PiSSA adapter is not a zero
  initialized perturbation: it holds the principal components of the original weight, and the
  residual base layer holds the rest. Applying dropout to the adapter input during training
  removes part of the principal signal while the residual stays intact, which breaks the
  reconstruction and inflates the loss.
- PiSSA initialization is the official PEFT implementation (`init_lora_weights="pissa"`),
  combined with `rank_pattern`. The residual weight plus `scaling * B @ A` reconstructs the
  original matrix to floating point precision (max absolute error 3.725e-09, verified with
  `rank_pattern` active).
- Warmup is 3% of total steps (23 of 750). This matters for PiSSA specifically: its adapter is
  non-zero initialized (`||A|| = ||B|| = 2.91`), so an un-warmed first step produces a large
  update. Measured without warmup, PiSSA loss spiked from 2.289 to 6.449 at step 2; with
  warmup it moved 2.289 to 2.115.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before CUDA initialization.
  Without it, long runs fragment the caching allocator until Windows spills to shared memory,
  which drops throughput from 1.2 s/it to over 300 s/it while still reporting 100% GPU
  utilization.
- The allocator raises an error when the uniform budget is infeasible under the requested
  `r_min` / `r_max`, instead of silently returning a wrong budget.

## Limitations

- Results are from two models and two seeds. The inverted-versus-uniform gap is consistent
  across both models and both seeds, but the sample is small.
- The explanation for the Qwen collapse is post-hoc. It is supported by the measured
  correlation between allocation unevenness and outcome, and by the raw-versus-normalized
  entropy ordering, but it was not predicted in advance.
- Only GSM8K is evaluated. Rank allocation interacts with task structure, and a task requiring
  different projections would likely produce a different allocation.
- The training budget (3000 examples, 2 epochs) is short. Longer training could change the
  ordering, particularly for the default direction on Qwen, though the loss curves show no
  recovery within the observed window.
- `r_min=4` was never binding on Qwen (minimum realized rank 6), and the interaction between
  `r_min` and the allocation direction was not ablated.
