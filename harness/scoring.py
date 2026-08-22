"""Scoring and evaluation metrics for experiment results."""

import math
import re
from typing import Any, Optional


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Args:
        text: Raw answer text

    Returns:
        Normalized answer (uppercase, stripped, alphanumeric only)
    """
    # Convert to uppercase
    text = text.upper().strip()

    # Extract first letter if it looks like a choice (A, B, C, D, etc.)
    match = re.match(r'^([A-Z])\b', text)
    if match:
        return match.group(1)

    # Otherwise return cleaned text
    return re.sub(r'[^A-Z0-9]', '', text)


def exact_match(predicted: str, gold: str) -> bool:
    """
    Check if predicted answer exactly matches gold answer.

    Args:
        predicted: Predicted answer
        gold: Gold standard answer

    Returns:
        True if answers match after normalization
    """
    return normalize_answer(predicted) == normalize_answer(gold)


def accuracy(predictions: list[str], golds: list[str]) -> dict[str, Any]:
    """
    Calculate accuracy metrics.

    Args:
        predictions: List of predicted answers
        golds: List of gold standard answers

    Returns:
        Dict with n, n_correct, accuracy
    """
    if len(predictions) != len(golds):
        raise ValueError("Predictions and golds must have same length")

    n = len(predictions)
    n_correct = sum(exact_match(pred, gold) for pred, gold in zip(predictions, golds))

    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n > 0 else 0.0
    }


def abstention_correctness(
    predictions: list[str],
    golds: list[str],
    abstention_flags: list[bool],
    regime_labels: list[str]
) -> dict[str, Any]:
    """
    Calculate correctness and abstention rates by regime.

    Args:
        predictions: List of predicted answers
        golds: List of gold answers
        abstention_flags: List of bools indicating if model abstained
        regime_labels: List of regime labels (e.g., "normal", "contradictory", "underspecified")

    Returns:
        Dict with per-regime and overall metrics
    """
    if not (len(predictions) == len(golds) == len(abstention_flags) == len(regime_labels)):
        raise ValueError("All input lists must have same length")

    # Initialize counters per regime
    regimes = set(regime_labels)
    regime_stats = {
        regime: {"n": 0, "n_correct": 0, "n_abstain": 0}
        for regime in regimes
    }

    # Count per regime
    for pred, gold, abstain, regime in zip(predictions, golds, abstention_flags, regime_labels):
        regime_stats[regime]["n"] += 1

        if abstain:
            regime_stats[regime]["n_abstain"] += 1
        elif exact_match(pred, gold):
            regime_stats[regime]["n_correct"] += 1

    # Calculate rates per regime
    results = {}
    for regime, stats in regime_stats.items():
        n = stats["n"]
        results[regime] = {
            "n": n,
            "n_correct": stats["n_correct"],
            "n_abstain": stats["n_abstain"],
            "accuracy": stats["n_correct"] / n if n > 0 else 0.0,
            "abstention_rate": stats["n_abstain"] / n if n > 0 else 0.0
        }

    # Overall stats
    total_n = len(predictions)
    total_correct = sum(stats["n_correct"] for stats in regime_stats.values())
    total_abstain = sum(stats["n_abstain"] for stats in regime_stats.values())

    results["overall"] = {
        "n": total_n,
        "n_correct": total_correct,
        "n_abstain": total_abstain,
        "accuracy": total_correct / total_n if total_n > 0 else 0.0,
        "abstention_rate": total_abstain / total_n if total_n > 0 else 0.0
    }

    return results


def flip_rate(preds_original: list[str], preds_permuted: list[str]) -> dict[str, Any]:
    """
    Calculate rate at which answers flip between original and permuted choices.

    Args:
        preds_original: Predictions with original choice order
        preds_permuted: Predictions with permuted choice order

    Returns:
        Dict with n, n_flipped, flip_rate
    """
    if len(preds_original) != len(preds_permuted):
        raise ValueError("Prediction lists must have same length")

    n = len(preds_original)
    n_flipped = sum(
        normalize_answer(orig) != normalize_answer(perm)
        for orig, perm in zip(preds_original, preds_permuted)
    )

    return {
        "n": n,
        "n_flipped": n_flipped,
        "flip_rate": n_flipped / n if n > 0 else 0.0
    }


def self_consistency(predictions_per_run: list[list[str]]) -> dict[str, Any]:
    """
    Calculate self-consistency across multiple runs.

    Args:
        predictions_per_run: List of prediction lists, one per run

    Returns:
        Dict with n_items, mean_agreement, per_item_agreement
    """
    if not predictions_per_run:
        return {
            "n_items": 0,
            "mean_agreement": 0.0,
            "per_item_agreement": []
        }

    n_runs = len(predictions_per_run)
    n_items = len(predictions_per_run[0])

    # Check all runs have same length
    if not all(len(run) == n_items for run in predictions_per_run):
        raise ValueError("All runs must have same number of items")

    # Calculate agreement per item
    per_item_agreement = []

    for item_idx in range(n_items):
        # Get all predictions for this item
        item_preds = [run[item_idx] for run in predictions_per_run]

        # Normalize
        item_preds_norm = [normalize_answer(p) for p in item_preds]

        # Most common answer
        from collections import Counter
        counts = Counter(item_preds_norm)
        if counts:
            most_common_count = counts.most_common(1)[0][1]
            agreement = most_common_count / n_runs
        else:
            agreement = 0.0

        per_item_agreement.append(agreement)

    return {
        "n_items": n_items,
        "mean_agreement": sum(per_item_agreement) / n_items if n_items > 0 else 0.0,
        "per_item_agreement": per_item_agreement
    }


def matrix_faithfulness(
    matrix_outputs: list[dict[str, Any]],
    final_answers: list[str]
) -> dict[str, Any]:
    """
    Calculate faithfulness of final answers to ACH matrix recommendations.

    For ACH conditions, checks if final answer matches hypothesis with
    fewest inconsistencies in the matrix.

    Args:
        matrix_outputs: List of parsed ACH outputs with inconsistency_counts
        final_answers: List of final answers

    Returns:
        Dict with n, n_faithful, faithfulness_rate
    """
    if len(matrix_outputs) != len(final_answers):
        raise ValueError("Matrix outputs and answers must have same length")

    n = len(matrix_outputs)
    n_faithful = 0

    for matrix_out, answer in zip(matrix_outputs, final_answers):
        if matrix_out is None:
            continue

        # Get inconsistency counts
        inconsistency_counts = matrix_out.get("inconsistency_counts", {})
        if not inconsistency_counts:
            continue

        # Find hypothesis with minimum inconsistencies
        min_hypothesis = min(inconsistency_counts.items(), key=lambda x: x[1])[0]

        # Extract answer choice from hypothesis (assumes format like "Answer is A")
        hypothesis_match = re.search(r'\b([A-Z])\b', min_hypothesis)
        if hypothesis_match:
            recommended_answer = hypothesis_match.group(1)

            # Check if answer matches recommendation
            if normalize_answer(answer) == normalize_answer(recommended_answer):
                n_faithful += 1

    return {
        "n": n,
        "n_faithful": n_faithful,
        "faithfulness_rate": n_faithful / n if n > 0 else 0.0
    }


def calibration_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10
) -> dict[str, Any]:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        confidences: List of confidence scores (0-100)
        correctness: List of correctness flags
        n_bins: Number of bins for calibration

    Returns:
        Dict with ece, bin_accuracies, bin_confidences, bin_counts
    """
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have same length")

    # Normalize confidences to 0-1
    confidences_norm = [c / 100.0 for c in confidences]

    # Create bins
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        # Find items in this bin
        in_bin = [
            j for j, conf in enumerate(confidences_norm)
            if bin_edges[i] <= conf < bin_edges[i + 1]
        ]

        # Handle last bin edge inclusively
        if i == n_bins - 1:
            in_bin = [
                j for j, conf in enumerate(confidences_norm)
                if bin_edges[i] <= conf <= bin_edges[i + 1]
            ]

        if not in_bin:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            bin_counts.append(0)
            continue

        # Calculate bin accuracy and average confidence
        bin_correct = sum(correctness[j] for j in in_bin)
        bin_conf = sum(confidences_norm[j] for j in in_bin)

        bin_count = len(in_bin)
        bin_accuracy = bin_correct / bin_count
        bin_avg_conf = bin_conf / bin_count

        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_avg_conf)
        bin_counts.append(bin_count)

        # Add to ECE
        ece += (bin_count / n_total) * abs(bin_accuracy - bin_avg_conf)

    return {
        "ece": ece,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }


def brier_score(confidences: list[float], correctness: list[bool]) -> dict[str, Any]:
    """
    Calculate Brier score for probability predictions.

    Args:
        confidences: List of confidence scores (0-100)
        correctness: List of correctness flags

    Returns:
        Dict with brier_score, n
    """
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have same length")

    n = len(confidences)
    if n == 0:
        return {"brier_score": 0.0, "n": 0}

    # Normalize confidences to probabilities
    probs = [c / 100.0 for c in confidences]

    # Calculate Brier score
    brier = sum((p - int(c)) ** 2 for p, c in zip(probs, correctness)) / n

    return {
        "brier_score": brier,
        "n": n
    }


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Args:
        n_success: Number of successes
        n_total: Total number of trials
        z: Z-score for confidence level (1.96 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n_total == 0:
        return (0.0, 0.0)

    p = n_success / n_total
    z2 = z * z

    denominator = 1 + z2 / n_total
    centre = (p + z2 / (2 * n_total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / n_total + z2 / (4 * n_total * n_total))) / denominator

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin)
    )


def aggregate_by_condition(scored_trials: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate trial results by condition.

    Args:
        scored_trials: List of trial records with scoring results

    Returns:
        Dict mapping condition_id -> aggregated metrics
    """
    # Group by condition
    by_condition = {}

    for trial in scored_trials:
        condition_id = trial.get("condition_id")
        if condition_id not in by_condition:
            by_condition[condition_id] = []
        by_condition[condition_id].append(trial)

    # Aggregate per condition
    results = {}

    for condition_id, trials in by_condition.items():
        n = len(trials)

        # Extract predictions and golds
        predictions = [t.get("parsed_result", {}).get("answer", "") for t in trials]
        golds = [t.get("gold_answer", "") for t in trials]

        # Calculate accuracy
        n_correct = sum(exact_match(pred, gold) for pred, gold in zip(predictions, golds))
        acc = n_correct / n if n > 0 else 0.0

        # Wilson CI
        ci_lower, ci_upper = wilson_ci(n_correct, n)

        # Parse failure rate
        n_parse_fail = sum(1 for t in trials if not t.get("parsed_result", {}).get("success", False))
        parse_fail_rate = n_parse_fail / n if n > 0 else 0.0

        # Mean confidence (if available)
        confidences = [
            t.get("parsed_result", {}).get("data", {}).get("confidence")
            for t in trials
        ]
        confidences = [c for c in confidences if c is not None]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None

        # Mean calibration error (if available)
        if confidences and len(confidences) == n:
            correctness = [exact_match(pred, gold) for pred, gold in zip(predictions, golds)]
            cal_metrics = calibration_ece(confidences, correctness)
            mean_calibration_error = cal_metrics["ece"]
        else:
            mean_calibration_error = None

        results[condition_id] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": acc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "parse_fail_rate": parse_fail_rate,
            "mean_confidence": mean_confidence,
            "mean_calibration_error": mean_calibration_error
        }

    return results
