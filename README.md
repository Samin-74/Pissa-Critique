# Entropy-PiSSA

Spectral entropy guided rank allocation for PiSSA under a fixed parameter budget.

PiSSA (NeurIPS 2024) initializes LoRA adapters from the principal singular triplets of each
weight matrix, but it assigns the same rank `r` to every projection in every layer. Weight
spectra are not uniform: some matrices concentrate energy in a few directions, others spread
it almost evenly. Entropy-PiSSA measures that concentration per matrix and reallocates the
rank budget accordingly, keeping the total number of trainable parameters equal to the
uniform `r=16` baseline.

## Headline result

The inverted allocation (more rank to spectrally concentrated projections) is the best PiSSA
variant on both models tested, and the default allocation fails on one of them:

| Model | Uniform PiSSA | Entropy-PiSSA inverted | Delta |
| --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 61.10 +/- 0.71 | 63.00 +/- 0.57 | +1.90 |
| Phi-3.5-mini-instruct | 65.40 +/- 0.28 | 67.60 +/- 0.57 | +2.20 |

The default direction scores 1.50 +/- 0.71 on Qwen2.5 (training diverges on both seeds) but
66.60 +/- 1.13 on Phi-3.5. The difference is traced to the `ln(min(m, n))` normalization
interacting with the mix of module widths in each model. `summary_report.md` has the full
analysis.

## Method

For each target projection `W` in `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}`:

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

The `kappa` factor makes the continuous allocation budget exact: `sum_i r_i (m_i + n_i)` equals
the uniform budget by construction, so rounding only introduces a small residual that the
greedy repair removes.

`high_entropy_more_rank=True` gives more rank to high entropy (energy spread out) matrices.
`False` inverts the preference and gives more rank to low entropy (strongly concentrated)
matrices.

## Why entropy and not energy

Energy based criteria (`sum_{i<=r} sigma_i^2 / sum_i sigma_i^2`) are also discriminative on
these models: at `r=16` the retained fraction ranges from 1.96% to 46.41% across the 252
Qwen projections (mean 9.29%, coefficient of variation 0.716). The reason this project uses
normalized spectral entropy is not that energy saturates, but that energy at a fixed `r` is
strongly confounded with matrix width (`r = -0.70` against `min(m, n)`), so it partly
re-encodes the same size bias that the allocation is meant to correct. Normalized entropy is
less width-confounded (`r = -0.22`), but it is a much flatter signal: its coefficient of
variation is only 0.021, with all 252 values inside `[0.908, 0.999]`.

That flatness is the central empirical finding of this project. Because
`effective_rank = exp(H * ln min(m, n))` holds exactly by construction (it is the perplexity
of the singular value distribution), normalized entropy is a rescaled effective rank, and the
spread between the widest and narrowest matrices is small in entropy terms but large in rank
terms. Allocating rank in proportion to entropy therefore compresses the dynamic range of the
allocation far below what the underlying spectra would justify, which is what the inverted
direction corrects. See `summary_report.md` for the measured consequences.

## Project layout

```
.
|-- train.py                four-mode training and evaluation pipeline
|-- analyze.py              standalone spectral analysis and allocation
|-- aggregate.py            cross-run tables and plots
|-- test_allocator.py       23 allocator checks
|-- run_all.ps1             full 16-run suite
|-- run_suite.ps1           one model and seed group
|-- launch_all.cmd          Windows wrapper for run_all.ps1
|-- launch_suite.cmd        Windows wrapper for run_suite.ps1
|-- requirements.txt
|-- README.md
|-- summary_report.md
|-- src/
|   |-- __init__.py
|   |-- rank_allocator.py   CPU SVD, spectral entropy, budget allocation, PEFT patterns
|   |-- evaluator.py        batched greedy GSM8K generation and exact-match scoring
|   `-- utils.py            prompt construction, answer extraction, loss recording, plots
`-- outputs/
    |-- analysis/           per-model allocation artifacts
    |-- plots/              33 PNG figures at 200 dpi
    |-- results/            comparison CSVs and summary_metrics.json
    |-- qwen_seed42/        uniform_lora, uniform_pissa, entropy_pissa, entropy_pissa_inverted
    |-- qwen_seed123/
    |-- phi_seed42/
    `-- phi_seed123/
```

Each run directory holds `adapter/adapter_config.json`, `loss_curve.{json,csv}`,
`train_metrics.json`, and `eval_<mode>.{json,csv}`. Adapter weight files and per-run logs are
not committed; `outputs/logs/` is created when you run the suite.

## Hardware requirements

A single 16 GB GPU is enough for both models at the reported settings. The project was
developed on an RTX 4080 Super with 32 GB of system RAM.

- bf16 weights and bf16 autocast, no 4-bit quantization needed at 3B.
- Gradient checkpointing enabled with `use_reentrant=False`.
- Micro batch 4 with 2 accumulation steps (effective batch 8).
- SVD runs on CPU tensors, so the allocator never competes with the model for VRAM.
- `paged_adamw_8bit` keeps optimizer state small.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before CUDA initialization.

Measured peak allocated VRAM across the 16 reported runs: 8.37 GB for Qwen2.5-3B and 9.67 GB
for Phi-3.5-mini at `max_seq_length=512`, micro batch 4. Reserved memory reaches 13.4 to 14.8
GB, which is why `expandable_segments` matters: without it the caching allocator fragments
over a long run until Windows spills to shared memory, and throughput collapses from 1.2 s/it
to over 300 s/it while still reporting 100% GPU utilization.

GSM8K prompts never exceed 567 tokens, so 512 is enough for that dataset. 4096 will not fit at
micro batch 4; use micro batch 1 with more accumulation steps.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux or macOS
pip install -r requirements.txt
```

Set `HF_HOME` to control where base model weights are cached. The `launch_*.cmd` wrappers
default it to `hf_cache/` next to the repository if it is unset.

## Reproduce

Run the full 16-run suite (2 models, 2 seeds, 4 modes):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1
```

Or one model and seed group:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_suite.ps1 `
    -Model "Qwen/Qwen2.5-3B-Instruct" -ModelTag qwen -Seed 42
```

Generate all tables and figures from completed runs:

```bash
python aggregate.py --output_dir outputs
```

The suite is resumable: a run that already produced `eval_<mode>.json` is skipped.

A single run without a driver script:

```bash
python analyze.py --model_name Qwen/Qwen2.5-3B-Instruct --output_dir outputs/analysis/qwen
python analyze.py --model_name Qwen/Qwen2.5-3B-Instruct --output_dir outputs/analysis/qwen_inverted --no-high_entropy_more_rank

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

## Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `entropy_pissa` | `uniform_lora`, `uniform_pissa`, `entropy_pissa`, `entropy_pissa_inverted` (or `inverted`), or `all` |
| `--uniform_rank` | `16` | Baseline rank and the budget reference |
| `--lora_alpha` | `32` | Base alpha, scaled per layer to keep `alpha/r` constant |
| `--lora_dropout` | `0.0` | LoRA dropout, forced to 0 for the PiSSA modes |
| `--r_min` | `4` | Hard lower bound on any allocated rank |
| `--r_max` | `None` | Optional upper bound, validated against budget feasibility |
| `--tolerance` | `0.005` | Allowed relative deviation from the uniform budget |
| `--entropy_temperature` | `0.5` | Softmax temperature over standardized entropy |
| `--high_entropy_more_rank` | `True` | Direction of the entropy preference |
| `--max_seq_length` | `1024` | SFT sequence length |
| `--per_device_batch_size` | `2` | Micro batch size |
| `--gradient_accumulation_steps` | `4` | Effective batch is 8 |
| `--eval_limit` | `200` | GSM8K test examples to score |
| `--cpu_only` | off | Force float32 on CPU for debugging |

The reported experiments override the batch defaults to `--per_device_batch_size 4
--gradient_accumulation_steps 2` (same effective batch of 8) and use `--max_seq_length 512
--eval_limit 500 --max_new_tokens 512 --num_train_epochs 2 --train_limit 3000`.

## Results

GSM8K exact match, 500 test examples per run, greedy decoding, mean +/- std over seeds 42
and 123.

| Model | Mode | Accuracy |
| --- | --- | --- |
| Qwen2.5-3B-Instruct | Uniform LoRA | 65.70 +/- 0.99 |
| Qwen2.5-3B-Instruct | Uniform PiSSA | 61.10 +/- 0.71 |
| Qwen2.5-3B-Instruct | Entropy-PiSSA (default) | 1.50 +/- 0.71 |
| Qwen2.5-3B-Instruct | Entropy-PiSSA (inverted) | **63.00 +/- 0.57** |
| Phi-3.5-mini-instruct | Uniform LoRA | 76.10 +/- 0.42 |
| Phi-3.5-mini-instruct | Uniform PiSSA | 65.40 +/- 0.28 |
| Phi-3.5-mini-instruct | Entropy-PiSSA (default) | 66.60 +/- 1.13 |
| Phi-3.5-mini-instruct | Entropy-PiSSA (inverted) | **67.60 +/- 0.57** |

Uniform LoRA remains the strongest single configuration on both models. At `r=16` with 3000
examples the zero-initialized adapter is easier to optimize than PiSSA's non-zero
initialization. The comparison of interest here is between PiSSA variants, which share the
same initialization and differ only in rank allocation.

## Tests

```bash
python test_allocator.py
```

Covers budget exactness, even-rank rounding, the `r_min` floor, fused architectures such as
Phi-3 `qkv_proj` and `gate_up_proj`, and PEFT pattern export.

## Notes on correctness

- `alpha_pattern` is set so that `lora_alpha / r` is identical for every layer. Without it,
  PEFT computes `scaling = alpha / r` per layer, which would give low rank layers a larger
  effective update and confound the comparison.
- `lora_dropout` is forced to `0.0` for both PiSSA modes. A PiSSA adapter is not a zero
  initialized perturbation: it holds the principal components of the original weight, and the
  residual base layer holds the rest. Applying dropout to the adapter input during training
  removes part of the principal signal while the residual stays intact, which breaks the
  reconstruction and inflates the loss. Measured on Qwen2.5-3B-Instruct with two GSM8K
  examples, `lora_dropout=0.05` raised the step-0 training loss from 2.331 to 4.280, while
  `lora_dropout=0.0` kept it at 2.331.
- PiSSA initialization is the official PEFT implementation (`init_lora_weights="pissa"`),
  combined with `rank_pattern`. The residual weight plus `scaling * B @ A` reconstructs the
  original matrix to floating point precision (max absolute error 3.725e-09 with
  `rank_pattern` active).
- Warmup is 3% of total steps. This matters for PiSSA specifically: its adapter is non-zero
  initialized, so an un-warmed first step produces a large update.
- The allocator raises an error when the uniform budget is infeasible under the requested
  `r_min` / `r_max`, instead of silently returning a wrong budget.
