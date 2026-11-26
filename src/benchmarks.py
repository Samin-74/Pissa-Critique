from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .utils import extract_final_answer, normalize_number

# Task families decide how a reference is compared against generated text
TASK_NUMERIC = "numeric"
TASK_CHOICE = "choice"
TASK_BOOLEAN = "boolean"
TASK_STRING = "string"

FINISH_INSTRUCTION = (
    "Think step by step if needed, then end your response with a single final line "
    "of the form 'The answer is <ANSWER>.'"
)

_CHOICE_REFERENCE = re.compile(r"^\(?([A-Z])\)?$")
_NUMERIC_REFERENCE = re.compile(r"^-?\d[\d,]*\.?\d*$")
_ANSWER_IS_LETTER = re.compile(r"answer\s*(?:is|:)\s*\(?([A-Z])\b", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")
_YES_NO_WORD = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUE_FALSE_WORD = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return _WHITESPACE.sub(" ", _PUNCT.sub(" ", str(text)).lower()).strip()


def _last_letter_answer(text: str) -> Optional[str]:
    # Prefer an explicit "answer is (C)" phrase, then the last parenthesized letter
    matches = _ANSWER_IS_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    matches = _PAREN_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def _last_boolean_answer(text: str, vocabulary: Sequence[str]) -> Optional[str]:
    pattern = _YES_NO_WORD if set(vocabulary) == {"yes", "no"} else _TRUE_FALSE_WORD
    matches = pattern.findall(text)
    if matches:
        return matches[-1].lower()
    return None


def _last_number(text: str) -> Optional[str]:
    from .utils import NUMBER_PATTERN

    matches = NUMBER_PATTERN.findall(text or "")
    if not matches:
        return None
    return normalize_number(matches[-1])


@dataclass(frozen=True)
class Example:
    question: str
    reference: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    task_type: str
    supports_training: bool
    load_examples: Callable[[int, Optional[int]], List[Example]]

    def effective_type(self, reference: str) -> str:
        # References are more reliable than subtask names: BBH mixes formats inside a task
        ref = (reference or "").strip()
        if _CHOICE_REFERENCE.match(ref):
            return TASK_CHOICE
        if ref.lower() in ("yes", "no", "true", "false"):
            return TASK_BOOLEAN
        if _NUMERIC_REFERENCE.match(ref):
            return TASK_NUMERIC
        return self.task_type

    def judge(self, raw_output: str, reference: str) -> Tuple[Optional[str], bool]:
        # Returns the parsed prediction and whether it matches the reference
        task_type = self.effective_type(reference)
        if task_type == TASK_CHOICE:
            prediction = _last_letter_answer(raw_output)
            expected = (reference or "").strip().strip("()").upper()
            return prediction, prediction == expected
        if task_type == TASK_BOOLEAN:
            expected = normalize_text(reference)
            vocabulary = ["yes", "no"] if expected in ("yes", "no") else ["true", "false"]
            prediction = _last_boolean_answer(raw_output, vocabulary)
            return prediction, prediction == expected
        if task_type == TASK_NUMERIC and self.name == "gsm8k":
            prediction = extract_final_answer(raw_output)
            expected = normalize_number(reference) or normalize_text(reference)
            return prediction, prediction is not None and prediction == expected
        if task_type == TASK_NUMERIC:
            expected = normalize_number(reference)
            prediction = _last_number(raw_output)
            return prediction, prediction is not None and prediction == expected
        # Whole-word match covers short labels (valid/invalid) and long strings (word_sorting)
        expected_text = normalize_text(reference)
        haystack = normalize_text(raw_output)
        if expected_text and re.search(
            r"(?<!\w)" + re.escape(expected_text) + r"(?!\w)", haystack
        ):
            return normalize_text(reference), True
        tail = normalize_text(raw_output.splitlines()[-1]) if raw_output.strip() else ""
        return tail or None, False

    def summarize(self, records: Sequence[Any]) -> Dict[str, Any]:
        # Per-subtask accuracy so a single collapsed subtask cannot hide in the mean
        buckets: Dict[str, List[bool]] = {}
        for record in records:
            subtask = (getattr(record, "meta", None) or {}).get("subtask", "all")
            buckets.setdefault(subtask, []).append(bool(record.correct))
        return {
            subtask: {
                "num_examples": len(values),
                "num_correct": sum(values),
                "accuracy_pct": 100.0 * sum(values) / max(len(values), 1),
            }
            for subtask, values in sorted(buckets.items())
        }


def _gsm8k_examples(limit: int, max_per_subtask: Optional[int]) -> List[Example]:
    from datasets import load_dataset

    from .evaluator import extract_reference

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(dataset)
    if limit and limit > 0:
        rows = rows[:limit]
    return [
        Example(
            question=row["question"],
            reference=extract_reference(row["answer"]),
            meta={"subtask": "gsm8k"},
        )
        for row in rows
    ]


# Non-arithmetic BBH subtasks: logic, tracking, causal, temporal, and linguistic reasoning
DEFAULT_BBH_SUBTASKS: Tuple[str, ...] = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

BBH_BOOLEAN_SUBTASKS = frozenset(
    {
        "boolean_expressions",
        "causal_judgement",
        "navigate",
        "sports_understanding",
        "web_of_lies",
    }
)
BBH_STRING_SUBTASKS = frozenset(
    {
        "formal_fallacies",
        "hyperbaton",
        "movie_recommendation",
        "ruin_names",
        "snarks",
        "word_sorting",
    }
)


def _bbh_task_type(subtask: str) -> str:
    if subtask in BBH_BOOLEAN_SUBTASKS:
        return TASK_BOOLEAN
    if subtask in BBH_STRING_SUBTASKS:
        return TASK_STRING
    if subtask == "object_counting":
        return TASK_NUMERIC
    return TASK_CHOICE


def _bbh_examples(limit: int, max_per_subtask: Optional[int]) -> List[Example]:
    from datasets import load_dataset

    examples: List[Example] = []
    for subtask in DEFAULT_BBH_SUBTASKS:
        dataset = load_dataset("lukaemon/bbh", subtask, split="test")
        rows = list(dataset)
        caps = [c for c in (max_per_subtask, limit) if c and c > 0]
        # Applying the cap per subtask keeps the aggregate balanced under any limit
        capped = min(len(rows), min(caps)) if caps else len(rows)
        for row in rows[:capped]:
            examples.append(
                Example(
                    question=f"{row['input'].strip()}\n\n{FINISH_INSTRUCTION}",
                    reference=str(row["target"]).strip(),
                    meta={"subtask": subtask, "task_type": _bbh_task_type(subtask)},
                )
            )
    return examples


def _arc_examples(limit: int, max_per_subtask: Optional[int]) -> List[Example]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = list(dataset)
    if limit and 0 < limit < len(rows):
        # ARC rows are topic-grouped, so shuffle with a fixed seed instead of slicing
        import random

        shuffled = list(rows)
        random.Random(1234).shuffle(shuffled)
        rows = shuffled[:limit]
    examples: List[Example] = []
    for row in rows:
        choices = row["choices"]
        option_lines = [
            f"({label}) {text}" for label, text in zip(choices["label"], choices["text"])
        ]
        question = (
            f"{row['question'].strip()}\n\n"
            + "\n".join(option_lines)
            + f"\n\n{FINISH_INSTRUCTION}"
        )
        examples.append(
            Example(
                question=question,
                reference=str(row["answerKey"]).strip(),
                meta={"subtask": "arc_challenge"},
            )
        )
    return examples


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        display_name="GSM8K (grade-school math)",
        task_type=TASK_NUMERIC,
        supports_training=True,
        load_examples=_gsm8k_examples,
    ),
    "bbh": BenchmarkSpec(
        name="bbh",
        display_name="BIG-Bench Hard (non-arithmetic reasoning)",
        task_type=TASK_STRING,
        supports_training=False,
        load_examples=_bbh_examples,
    ),
    "arc_challenge": BenchmarkSpec(
        name="arc_challenge",
        display_name="ARC-Challenge (science reasoning)",
        task_type=TASK_CHOICE,
        supports_training=False,
        load_examples=_arc_examples,
    ),
}

BENCHMARK_ORDER: Tuple[str, ...] = ("gsm8k", "bbh", "arc_challenge")


def resolve_benchmark(name: str) -> BenchmarkSpec:
    key = (name or "").strip().lower()
    if key not in BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}"
        )
    return BENCHMARKS[key]


def evaluate_spec(spec: BenchmarkSpec, rows: Sequence[Any]) -> Dict[str, Any]:
    # Kept for callers that already have (example, raw_output) pairs
    correct = 0
    per_task: Dict[str, List[bool]] = {}
    for row, raw in rows:
        _, is_correct = spec.judge(raw, row.reference)
        correct += int(is_correct)
        kind = spec.effective_type(row.reference)
        per_task.setdefault(kind, []).append(is_correct)
    total = max(len(rows), 1)
    return {
        "accuracy": correct / total,
        "num_correct": correct,
        "num_examples": len(rows),
        "by_task_type": {
            key: {
                "num_examples": len(values),
                "accuracy_pct": 100.0 * sum(values) / max(len(values), 1),
            }
            for key, values in sorted(per_task.items())
        },
    }
