# Entropy-PiSSA

Spectral entropy guided rank allocation for PiSSA under a fixed parameter budget.

PiSSA (NeurIPS 2024) initializes LoRA adapters from the principal singular triplets of each
weight matrix, but it assigns the same rank `r` to every projection in every layer. Weight
spectra are not uniform: some matrices concentrate energy in a few directions, others spread
it almost evenly. Entropy-PiSSA measures that concentration per matrix and reallocates the
rank budget accordingly, keeping the total number of trainable parameters equal to the
uniform `r=16` baseline.

## Headline result

On GSM8K, the inverted allocation (more rank to spectrally concentrated projections) is the
best PiSSA variant on both models tested, and the default allocation fails on one of them:

| Model | Uniform PiSSA | Entropy-PiSSA inverted | Delta |
| --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 61.10 +/- 0.71 | 63.00 +/- 0.57 | +1.90 |
| Phi-3.5-mini-instruct | 65.40 +/- 0.28 | 67.60 +/- 0.57 | +2.20 |

The default direction scores 1.50 +/- 0.71 on Qwen2.5 (training diverges on both seeds) but
66.60 +/- 1.13 on Phi-3.5. The difference is traced to the `log2(min(m, n))` normalization
interacting with the mix of module widths in each model.

**The GSM8K advantage does not generalize.** The same adapters, scored without retraining on
ARC-Challenge science questions, show no replicable advantage for the inverted allocation:
-15.10 on Qwen, and on Phi it swings from -11.80 to +14.60 across the two seeds. On GSM8K the
base model also outscores every fine-tuned variant (82.60 vs 65.70 on Qwen, 84.20 vs 76.10 on
Phi), and on ARC the base model wins for Phi (89.60 vs 85.60) but loses for Qwen (73.20 vs
79.40 uniform LoRA). At 3000 examples the fine-tuning therefore improves one of these four
model/benchmark pairs. `summary_report.md` has the full analysis and the caveats.

## Method

For each target projection `W` in `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}`,
or in the fused equivalents `{qkv_proj, gate_up_proj}` that Phi-3.5 uses (detected automatically):

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
`effective_rank = exp(H * ln min(m, n))` holds by construction when `H` is the normalized
entropy (it is the perplexity of the singular value distribution), normalized entropy is a
rescaled effective rank, and the spread
between the widest and narrowest matrices is small in entropy terms but large in rank terms.
Allocating rank in proportion to entropy therefore compresses the dynamic range of the
allocation far below what the underlying spectra would justify.

Inverting the preference (`high_entropy_more_rank=False`) does not restore that range: it
negates the standardized scores, so the allocation has the same spread and only the
preference is reversed. What inversion changes is which modules are favoured. The measured
consequence is in `summary_report.md`: inverting gives the widest, highest effective-rank
attention projections the largest ranks instead of the smallest, and that is the direction
that stays stable on Qwen.

## Project layout

```
.
|-- train.py                four-mode training and evaluation pipeline
|-- analyze.py              standalone spectral analysis and allocation
|-- aggregate.py            cross-run tables and plots
|-- report_benchmarks.py    cross-benchmark tables and paired tests
|-- test_allocator.py       23 allocator checks + benchmark scorer checks
|-- run_all.ps1             full 16-run suite
|-- run_suite.ps1           one model and seed group
|-- run_extra_benchmarks.ps1  score trained adapters on extra benchmarks
|-- launch_all.cmd          Windows wrapper for run_all.ps1
|-- launch_suite.cmd        Windows wrapper for run_suite.ps1
|-- requirements.txt
|-- README.md
|-- summary_report.md
|-- src/
|   |-- __init__.py
|   |-- rank_allocator.py   CPU SVD, spectral entropy, budget allocation, PEFT patterns
|   |-- benchmarks.py       benchmark registry, loaders, and per-format answer scoring
|   |-- evaluator.py        batched greedy generation and benchmark scoring
|   `-- utils.py            prompt construction, answer extraction, loss recording, plots
`-- outputs/
    |-- analysis/           per-model allocation artifacts (compact + full spectra)
    |-- plots/              33 PNG figures at 200 dpi
    |-- results/            comparison CSVs, summary_metrics.json, benchmark tables
    |-- base_reference_qwen/  un-finetuned scores (zero-shot floor)
    |-- base_reference_phi/
    |-- qwen_seed42/        uniform_lora, uniform_pissa, entropy_pissa, entropy_pissa_inverted
    |-- qwen_seed123/
    |-- phi_seed42/
    `-- phi_seed123/
```

Each run directory holds `adapter/adapter_config.json`, `loss_curve.{json,csv}`,
`train_metrics.json`, and `eval_<mode>.{json,csv}`. Extra benchmarks add a suffix
(`eval_<mode>_arc_challenge.{json,csv}`). Adapter weight files and per-run logs are not
committed; `outputs/logs/` is created when you run the suite, and `outputs/results/` holds
only the aggregated tables.

### What works in a fresh clone

The repository ships 219 files (73 MB) with every result, plot, and allocation artifact, so
the numbers can be inspected and the tables rebuilt without a GPU.

| Task | Fresh clone | Notes |
| --- | --- | --- |
| Inspect results and read plots | yes | `outputs/results/`, `outputs/plots/`, `outputs/analysis/` |
| Rebuild all tables and figures | yes | `python aggregate.py` and `python report_benchmarks.py`, CPU only, about 30 s |
| Run the test suite | yes | `python test_allocator.py`, CPU only |
| Score adapters with `--skip_train` | no | needs `adapter_model.safetensors`, which is gitignored |
| Train from scratch | yes | needs a GPU and the base weights from the Hub |

`--skip_train` reads a saved adapter, and only `adapter_config.json` is committed, not the
weights. The scorer exits with an explicit message rather than a stack trace when they are
absent. To reproduce the evaluation end to end, retrain first: `run_extra_benchmarks.ps1`
then re-scores the adapters it produced.

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

Measured over the 3000 training examples at `max_seq_length=512`, the tokenized chat text has
median length 196, a 99th percentile of 421, and a maximum of 567 tokens. Exactly one example
exceeds 512, and its `#### <answer>` span still falls inside the kept window. GSM8K test
prompts are shorter still: the maximum across every evaluation here is 206 tokens, so 512 is
sufficient for this dataset without being a universal limit. 4096 will not fit at micro batch
4; use micro batch 1 with more accumulation steps.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux or macOS
pip install -r requirements.txt
```

`transformers>=5.0.0` and `trl>=1.0.0` are hard requirements, not preferences. The trainer
uses `SFTTrainer(processing_class=...)` and `SFTConfig(warmup_steps=...)` only, because trl 1
removed the older `tokenizer` and `warmup_ratio` config arguments. Installing the previously
documented `transformers>=4.40` / `trl>=0.8.6` produces a `TypeError` at trainer construction.

The CLI keeps a `--warmup_ratio` flag (default `0.03`) for readability, but it is **not** passed
to `SFTConfig` as a ratio. Values below 1.0 are forwarded to `warmup_steps`, which trl 1 also
interprets as a ratio; values of 1.0 or more are treated as absolute step counts. The name is
therefore a convenience alias, not the removed trl argument.

Tested on Windows 11, Python 3.12, torch 2.14.1+cu132, transformers 5.20.2, peft 0.21.2,
trl 1.16.0, on an RTX 4080 Super (16 GB).

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

## Additional benchmarks (no retraining)

GSM8K is a single task family. Because the rank allocation is derived from the base model
weights only, it is identical for every training set, so a second training set would test the
same allocation. What changes the claim is a benchmark that measures a *different* kind of
reasoning. `arc_challenge` plays that role in this repository; `bbh` is implemented but was
not run.

| Benchmark | Status | Task | Format |
| --- | --- | --- | --- |
| `gsm8k` | **reported** | grade-school math | `#### <number>` |
| `arc_challenge` | **reported** | ARC science questions | 4-way choice |
| `bbh` | **implemented, not run** | BIG-Bench Hard, 24 non-arithmetic subtasks | letter / yes-no / free text |

Only `gsm8k` and `arc_challenge` have results in this repository; the tables in "Results" come
from those two. `bbh` is registered and its scorer is unit-tested in `test_allocator.py`, but
scoring it needs about 15 GPU hours for 2400 items across the 16 adapters, so it was deferred
and has no numbers here. Do not cite BBH results from this repository.

Score the existing adapters on the reported second benchmark (no training step):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_extra_benchmarks.ps1 `
    -Model "Qwen/Qwen2.5-3B-Instruct" -ModelTag qwen -Seeds "42,123" `
    -Benchmarks "arc_challenge"
```

Or score one adapter directly, reusing its saved rank pattern:

```bash
python train.py \
    --mode entropy_pissa_inverted \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --run_name qwen_seed42/entropy_pissa_inverted \
    --output_dir outputs \
    --eval_benchmarks arc_challenge \
    --eval_limit 500 --eval_batch_size 8 --max_new_tokens 512 \
    --seed 42 --skip_train --skip_plots
```

`--skip_train` skips both the SVD allocation and the training split, so only the adapter and
the benchmark load. Extra benchmarks are written next to the GSM8K artifact with a suffix
(`eval_<mode>_arc_challenge.json`), so existing results are never overwritten.

To generate BBH as well, add `bbh` to `--eval_benchmarks` and allow roughly 15 GPU hours for
all 16 adapters. Each of the 24 subtasks is capped by `--bbh_max_per_subtask` so the aggregate
stays balanced.

Then build the cross-benchmark tables and paired tests:

```bash
python report_benchmarks.py --output_dir outputs
```

Paired tests compare two modes on the *same items* (exact binomial over discordant pairs),
which is much more sensitive than comparing seed means.

### Base-model reference

An absolute accuracy on a new benchmark is only interpretable against the un-finetuned model:
without it there is no way to tell whether GSM8K training helped or hurt science reasoning.
`--base_model_reference` scores the raw weights and writes to `outputs/base_reference_<family>/`,
so each model keeps its own reference and the four-mode comparison table is left untouched.

```bash
python train.py \
    --mode uniform_lora \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --run_name base_reference \
    --output_dir outputs \
    --eval_benchmarks arc_challenge \
    --eval_limit 500 --eval_batch_size 8 --max_new_tokens 512 \
    --seed 42 --skip_train --skip_plots --base_model_reference
```

The driver script runs this automatically when the reference artifact is missing.

### Format collapse

These models were trained to answer with `#### <number>`, and they carry that habit to other
benchmarks: on ARC they often emit `#### 3` instead of `(C)`. The scorers accept the native
option formats and the explicit `The answer is ...` phrasing, so a reply that names the option
in prose is still scored. A reply that only produces a bare number without naming the option
is scored as wrong, and it is reported in `num_unparsed`. `num_unparsed` is therefore a real
finding about instruction-following transfer, not just a parsing artifact, and it is printed
next to every accuracy.

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
    --logging_steps 10 \
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
| `--eval_benchmarks` | `gsm8k` | Comma separated: `gsm8k`, `arc_challenge` (both reported), `bbh` (implemented, not run), or `all` |
| `--bbh_max_per_subtask` | `100` | Cap per BIG-Bench Hard subtask so the aggregate stays balanced (only used by `bbh`) |
| `--base_model_reference` | off | Also score the un-finetuned base model, the zero-shot floor for absolute scores |
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

### ARC-Challenge (second benchmark, no retraining)

Same adapters, scored on ARC-Challenge science questions. 500 items per run, greedy decoding,
mean +/- std over seeds 42 and 123. The base row is the un-finetuned model, which is the
zero-shot floor that makes the absolute numbers interpretable.

| Model | Mode | ARC accuracy |
| --- | --- | --- |
| Qwen2.5-3B-Instruct | Base model (no fine-tuning) | 73.20 |
| Qwen2.5-3B-Instruct | Uniform LoRA | **79.40 +/- 0.57** |
| Qwen2.5-3B-Instruct | Uniform PiSSA | 54.60 +/- 21.21 |
| Qwen2.5-3B-Instruct | Entropy-PiSSA (default) | 0.00 +/- 0.00 |
| Qwen2.5-3B-Instruct | Entropy-PiSSA (inverted) | 39.50 +/- 18.81 |
| Phi-3.5-mini-instruct | Base model (no fine-tuning) | 89.60 |
| Phi-3.5-mini-instruct | Uniform LoRA | 85.60 +/- 0.28 |
| Phi-3.5-mini-instruct | Uniform PiSSA | 66.40 +/- 6.79 |
| Phi-3.5-mini-instruct | Entropy-PiSSA (default) | 74.40 +/- 3.96 |
| Phi-3.5-mini-instruct | Entropy-PiSSA (inverted) | 67.80 +/- 11.88 |

Three things this benchmark shows that GSM8K alone could not:

1. **The PiSSA ranking does not transfer.** On GSM8K the inverted allocation beats uniform
   PiSSA on both models (+1.90 Qwen, +2.20 Phi). On ARC the Qwen gap reverses (-15.10) and the
   Phi gap is inconsistent across seeds (-11.80 and +14.60, mean +1.40), so the paired tests
   are significant in opposite directions at the seed level. The GSM8K advantage is therefore
   task-specific, not a general property of the allocation.
2. **Fine-tuning does not improve these benchmarks.** The base model beats every fine-tuned
   variant on GSM8K (Qwen 82.60 vs 65.70 best adapter; Phi 84.20 vs 76.10) and on ARC for Phi
   (89.60 vs 85.60). ARC Qwen is the one exception, where uniform LoRA lifts 73.20 to 79.40.
   Three of four model/benchmark pairs therefore lose accuracy to fine-tuning, which bounds
   what any mode comparison here can claim.
3. **Seed variance is much larger than on GSM8K.** Uniform PiSSA on Qwen ranges 39.60 to
   69.60 across seeds, and entropy-inverted ranges 26.20 to 52.80. Two seeds cannot support a
   claim on this benchmark; the GSM8K deltas are stable by comparison.

The base model also scores 82.60 on GSM8K (Qwen) and 84.20 on GSM8K (Phi), both above their
fine-tuned counterparts, so the fine-tuning is not improving GSM8K either. That is a
limitation of 3000 examples at 2 epochs rather than of the rank allocation, and it is the
first thing to address before drawing conclusions about the allocation itself.

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
