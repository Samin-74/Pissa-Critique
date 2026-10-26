from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .utils import (
    answers_match,
    build_prompt,
    ensure_dir,
    extract_final_answer,
    save_csv,
    save_json,
)

GSM8K_REFERENCE_PATTERN = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


@dataclass
class EvalRecord:
    index: int
    question: str
    reference: Optional[str]
    prediction: Optional[str]
    raw_output: str
    correct: bool
    prompt_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class EvalResult:
    mode: str
    accuracy: float
    num_examples: int
    num_correct: int
    num_unparsed: int
    records: List[EvalRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    max_new_tokens: int = 512
    batch_size: int = 8

    @property
    def accuracy_pct(self) -> float:
        return 100.0 * self.accuracy

    def to_dict(self, include_records: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "accuracy": self.accuracy,
            "accuracy_pct": self.accuracy_pct,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "num_unparsed": self.num_unparsed,
            "elapsed_seconds": self.elapsed_seconds,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
        }
        if include_records:
            payload["records"] = [
                {
                    "index": r.index,
                    "question": r.question,
                    "reference": r.reference,
                    "prediction": r.prediction,
                    "raw_output": r.raw_output,
                    "correct": r.correct,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                }
                for r in self.records
            ]
        return payload


def extract_reference(answer_field: str) -> Optional[str]:
    if not answer_field:
        return None
    match = GSM8K_REFERENCE_PATTERN.search(answer_field)
    if match:
        return extract_final_answer("#### " + match.group(1))
    return extract_final_answer(answer_field)


def _truncate_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    cut = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        position = text.find(stop)
        if position != -1:
            cut = min(cut, position)
    return text[:cut]


def _patch_legacy_cache_api() -> None:
    # Phi-3 remote code still reads Cache.seen_tokens and get_usable_length, removed in transformers 5
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return
    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(Cache, "get_usable_length"):
        Cache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length()
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length()
        )


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    tokenizer,
    questions: Sequence[str],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    system_prompt: Optional[str] = None,
    stop_strings: Sequence[str] = ("\nQuestion:", "\n\nQuestion:"),
    temperature: float = 0.0,
    progress_every: int = 20,
    max_prompt_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    # Batched greedy decoding with left padding
    _patch_legacy_cache_api()
    model.eval()
    device = next(model.parameters()).device
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: List[Dict[str, Any]] = []
    total = len(questions)
    start = time.time()
    use_cache = True
    for offset in range(0, total, batch_size):
        batch = list(questions[offset : offset + batch_size])
        prompts = [
            build_prompt(q, tokenizer=tokenizer, system_prompt=system_prompt) for q in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": use_cache,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        try:
            generated = model.generate(**encoded, **generation_kwargs)
        except (ValueError, AttributeError) as error:
            # Some remote-code architectures break with the transformers 5 cache
            if not use_cache:
                raise
            print(f"[eval] cache disabled after {type(error).__name__}: {error}", flush=True)
            use_cache = False
            generation_kwargs["use_cache"] = False
            generated = model.generate(**encoded, **generation_kwargs)
        except torch.cuda.OutOfMemoryError:
            # Halve the batch and retry so one oversized prompt cannot abort the run
            if len(batch) == 1:
                raise
            print(f"[eval] OOM at batch={len(batch)}, retrying with {len(batch) // 2}", flush=True)
            torch.cuda.empty_cache()
            half = len(batch) // 2
            for chunk in (batch[:half], batch[half:]):
                outputs.extend(
                    generate_predictions(
                        model,
                        tokenizer,
                        chunk,
                        max_new_tokens=max_new_tokens,
                        batch_size=len(chunk),
                        system_prompt=system_prompt,
                        stop_strings=stop_strings,
                        temperature=temperature,
                        progress_every=0,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                )
            continue
        prompt_width = encoded["input_ids"].shape[1]
        for row in generated:
            new_tokens = row[prompt_width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = _truncate_at_stop(text, stop_strings)
            outputs.append(
                {
                    "raw_output": text,
                    "prompt_tokens": int(prompt_width),
                    "generated_tokens": int(new_tokens.shape[0]),
                }
            )
        if progress_every and (offset // batch_size + 1) % progress_every == 0:
            done = min(offset + batch_size, total)
            rate = done / max(time.time() - start, 1e-6)
            print(f"[eval] {done}/{total} prompts ({rate:.2f}/s)", flush=True)

    tokenizer.padding_side = original_padding
    return outputs


def evaluate_gsm8k(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    mode: str,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    progress_every: int = 20,
) -> EvalResult:
    rows = list(dataset)
    if limit is not None:
        rows = rows[: int(limit)]
    questions = [row["question"] for row in rows]
    references = [extract_reference(row["answer"]) for row in rows]

    start = time.time()
    generations = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        system_prompt=system_prompt,
        progress_every=progress_every,
    )
    elapsed = time.time() - start

    records: List[EvalRecord] = []
    correct = 0
    unparsed = 0
    for idx, (question, reference, generation) in enumerate(
        zip(questions, references, generations)
    ):
        prediction = extract_final_answer(generation["raw_output"])
        if prediction is None:
            unparsed += 1
        is_correct = answers_match(prediction, reference)
        correct += int(is_correct)
        records.append(
            EvalRecord(
                index=idx,
                question=question,
                reference=reference,
                prediction=prediction,
                raw_output=generation["raw_output"],
                correct=is_correct,
                prompt_tokens=generation["prompt_tokens"],
                generated_tokens=generation["generated_tokens"],
            )
        )

    total = max(len(records), 1)
    return EvalResult(
        mode=mode,
        accuracy=correct / total,
        num_examples=len(records),
        num_correct=correct,
        num_unparsed=unparsed,
        records=records,
        elapsed_seconds=elapsed,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


def save_eval_result(result: EvalResult, output_dir: str) -> Dict[str, str]:
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"eval_{result.mode}.json")
    csv_path = os.path.join(output_dir, f"eval_{result.mode}.csv")
    save_json(result.to_dict(include_records=True), json_path)
    save_csv(
        [
            {
                "index": r.index,
                "reference": r.reference,
                "prediction": r.prediction,
                "correct": r.correct,
                "generated_tokens": r.generated_tokens,
                "question": r.question.replace("\n", " "),
                "raw_output": r.raw_output.replace("\n", " "),
            }
            for r in result.records
        ],
        csv_path,
        fieldnames=[
            "index",
            "reference",
            "prediction",
            "correct",
            "generated_tokens",
            "question",
            "raw_output",
        ],
    )
    return {"json": json_path, "csv": csv_path}


def load_eval_result(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_eval_results(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for mode, payload in results.items():
        rows.append(
            {
                "mode": mode,
                "accuracy": payload.get("accuracy", 0.0),
                "accuracy_pct": 100.0 * float(payload.get("accuracy", 0.0)),
                "num_examples": payload.get("num_examples", 0),
                "num_correct": payload.get("num_correct", 0),
                "num_unparsed": payload.get("num_unparsed", 0),
                "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
            }
        )
    return rows
