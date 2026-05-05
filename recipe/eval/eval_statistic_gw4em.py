"""Evaluation statistics for injection tests.

Reads the summary_gw4em.tsv and corresponding jsonl files to compute:
- Recall rate (correct counterpart predictions)
- Token usage statistics (mean and std for input/output tokens)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


SUMMARY_TSV_PATH = Path("recipe/eval/eval_data/summary_gw4em.tsv")
EVAL_DATA_DIR = Path("recipe/eval/eval_data/gw4em")


@dataclass
class EvalResult:
    gw_event: str
    waveform: str
    gold_counterpart: str
    predicted_counterpart: Optional[str]
    is_correct: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]


def parse_counterpart_from_jsonl(jsonl_path: Path) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse the predicted counterpart, input_tokens, and output_tokens from a jsonl file.

    Returns:
        (predicted_counterpart, input_tokens, output_tokens)
    """
    predicted_counterpart = None
    input_tokens = None
    output_tokens = None

    if not jsonl_path.exists():
        return predicted_counterpart, input_tokens, output_tokens

    content = jsonl_path.read_text(encoding="utf-8")

    # Find Counterpart: XXX pattern
    counterpart_match = re.search(r"Counterpart:\s*(\S+)", content)
    if counterpart_match:
        predicted_counterpart = counterpart_match.group(1)

    # Find [input_tokens] XXX pattern
    input_match = re.search(r"\[input_tokens\]\s*(\d+)", content)
    if input_match:
        input_tokens = int(input_match.group(1))

    # Find [output_tokens] XXX pattern
    output_match = re.search(r"\[output_tokens\]\s*(\d+)", content)
    if output_match:
        output_tokens = int(output_match.group(1))

    return predicted_counterpart, input_tokens, output_tokens


def load_summary_tsv(summary_path: Path) -> List[Tuple[str, str, str]]:
    """Load summary.tsv file.

    Returns:
        List of (gw_event, waveform, gold_counterpart) tuples
    """
    results = []
    if not summary_path.exists():
        return results

    with summary_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            gw_event = parts[0].strip()
            waveform = parts[1].strip()
            gold_counterpart = parts[2].strip()
            results.append((gw_event, waveform, gold_counterpart))

    return results


def compute_statistics(results: List[EvalResult]) -> dict:
    """Compute recall rate and token statistics."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    recall_rate = correct / total if total > 0 else 0.0

    # Token statistics
    input_tokens_list = [r.input_tokens for r in results if r.input_tokens is not None]
    output_tokens_list = [r.output_tokens for r in results if r.output_tokens is not None]

    def mean_std(values: List[int]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        return mean, std

    input_mean, input_std = mean_std(input_tokens_list)
    output_mean, output_std = mean_std(output_tokens_list)

    return {
        "total_events": total,
        "correct_predictions": correct,
        "recall_rate": recall_rate,
        "input_tokens_mean": input_mean,
        "input_tokens_std": input_std,
        "output_tokens_mean": output_mean,
        "output_tokens_std": output_std,
    }


def main():
    # Load summary.tsv
    summary = load_summary_tsv(SUMMARY_TSV_PATH)
    if not summary:
        print(f"No data found in {SUMMARY_TSV_PATH}")
        return

    print(f"Loaded {len(summary)} events from summary.tsv")

    # Evaluate each event
    results: List[EvalResult] = []
    for gw_event, waveform, gold_counterpart in summary:
        jsonl_path = EVAL_DATA_DIR / f"{gw_event}.jsonl"
        predicted, input_tokens, output_tokens = parse_counterpart_from_jsonl(jsonl_path)

        is_correct = predicted == gold_counterpart

        results.append(EvalResult(
            gw_event=gw_event,
            waveform=waveform,
            gold_counterpart=gold_counterpart,
            predicted_counterpart=predicted,
            is_correct=is_correct,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

        status = "✓" if is_correct else "✗"
        print(f"  {status} {gw_event} ({waveform}): gold={gold_counterpart}, pred={predicted}")

    # Compute and print statistics
    stats = compute_statistics(results)

    print("\n" + "=" * 50)
    print("Evaluation Statistics")
    print("=" * 50)
    print(f"Total events:        {stats['total_events']}")
    print(f"Correct predictions: {stats['correct_predictions']}")
    print(f"Recall rate:         {stats['recall_rate']:.2%}")
    print("-" * 50)
    print(f"Input tokens:  mean = {stats['input_tokens_mean']:.1f}, std = {stats['input_tokens_std']:.1f}")
    print(f"Output tokens: mean = {stats['output_tokens_mean']:.1f}, std = {stats['output_tokens_std']:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()