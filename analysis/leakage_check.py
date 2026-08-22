#!/usr/bin/env python3
"""
Leakage detection script for T2 dataset.

This script implements multiple baseline methods to detect potential data leakage
in the T2 dataset by checking if simple heuristics can predict correct answers
above chance level.

Supports T2 items with:
  - hypotheses: list[str]
  - gold_answer: str
  - narrative: str
"""

import argparse
import json
import sys
from collections import Counter
from typing import List, Dict, Any


def load_items_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load items from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of item dictionaries
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def majority_class_baseline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute majority class baseline accuracy.

    Predicts the most common gold answer for all items.

    Args:
        items: List of item dictionaries with 'gold_answer' field

    Returns:
        Dictionary with accuracy and most_common_answer
    """
    if not items:
        return {"accuracy": 0.0, "most_common_answer": None, "n_items": 0}

    gold_answers = [item.get("gold_answer", "") for item in items]
    counter = Counter(gold_answers)
    most_common_answer, most_common_count = counter.most_common(1)[0]

    acc = most_common_count / len(items)

    return {
        "accuracy": acc,
        "most_common_answer": most_common_answer,
        "n_items": len(items)
    }


def lexical_overlap_heuristic(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict hypothesis whose text appears most often in narrative.

    For each item, count how many times each hypothesis text appears in the
    narrative and predict the one with the highest count.

    Args:
        items: List of item dictionaries with hypotheses as list[str]

    Returns:
        Dictionary with accuracy and per-item predictions
    """
    correct = 0
    predictions = []

    for item in items:
        narrative = item.get("narrative", "").lower()
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        # Count mentions of each hypothesis text in narrative
        mention_counts = {}
        for hyp in hypotheses:
            hyp_lower = hyp.lower()
            # Extract the key name (e.g., "Alex Chen" from "Alex Chen is responsible")
            parts = hyp_lower.split(" is responsible")
            name = parts[0] if len(parts) > 1 else hyp_lower
            count = narrative.count(name)
            mention_counts[hyp] = count

        # Predict hypothesis with most mentions
        predicted = max(mention_counts.keys(), key=lambda k: mention_counts[k])
        predictions.append({
            "item_id": item.get("id", ""),
            "predicted": predicted,
            "correct": gold_answer,
            "mention_counts": {k: v for k, v in mention_counts.items()},
        })

        if predicted == gold_answer:
            correct += 1

    acc = correct / len(items) if items else 0.0

    return {
        "accuracy": acc,
        "n_items": len(items),
        "n_correct": correct,
    }


def simple_word_frequency_baseline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple word frequency baseline (fallback when sklearn unavailable).

    For each hypothesis, compute word overlap between narrative and hypothesis text.
    Predict hypothesis with highest overlap.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy
    """
    correct = 0

    for item in items:
        narrative_words = set(item.get("narrative", "").lower().split())
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        overlap_scores = {}
        for hyp in hypotheses:
            hyp_words = set(hyp.lower().split())
            overlap = len(narrative_words & hyp_words)
            overlap_scores[hyp] = overlap

        predicted = max(overlap_scores.keys(), key=lambda k: overlap_scores[k])
        if predicted == gold_answer:
            correct += 1

    acc = correct / len(items) if items else 0.0

    return {
        "accuracy": acc,
        "n_items": len(items),
        "n_correct": correct,
        "method": "simple_word_frequency"
    }


def bow_logistic_baseline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """TF-IDF + LogisticRegression baseline with cross-validation.

    Falls back to simple_word_frequency_baseline if sklearn is unavailable
    or there are too few items.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and method used
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        print("Warning: sklearn not available, falling back to simple_word_frequency_baseline",
              file=sys.stderr)
        return simple_word_frequency_baseline(items)

    if len(items) < 5:
        print("Warning: too few items for cross-validation, falling back to simple baseline",
              file=sys.stderr)
        return simple_word_frequency_baseline(items)

    # Prepare data: narrative text and gold answer labels
    texts = []
    labels = []

    for item in items:
        texts.append(item.get("narrative", ""))
        labels.append(item.get("gold_answer", ""))

    # Convert to numeric labels
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return simple_word_frequency_baseline(items)

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return simple_word_frequency_baseline(items)

    # Logistic regression with k-fold CV
    clf = LogisticRegression(max_iter=1000, random_state=42)
    k = min(5, min(Counter(y).values()), len(items))
    if k < 2:
        return simple_word_frequency_baseline(items)

    try:
        scores = cross_val_score(clf, X, y, cv=k, scoring='accuracy')
        acc = float(np.mean(scores))

        return {
            "accuracy": acc,
            "n_items": len(items),
            "method": "tfidf_logistic_cv",
            "cv_scores": [float(s) for s in scores],
            "cv_std": float(np.std(scores))
        }
    except Exception as e:
        print(f"Warning: cross-validation failed ({e}), falling back to simple baseline",
              file=sys.stderr)
        return simple_word_frequency_baseline(items)


def run_leakage_check(items: List[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """Run all leakage detection methods and compute PASS/FAIL verdict.

    Args:
        items: List of item dictionaries
        alpha: Threshold margin above chance level for FAIL verdict

    Returns:
        Dictionary with results from all methods and overall verdict
    """
    # Compute chance level (1 / average number of hypotheses per item)
    n_hypotheses_per_item = [len(item.get("hypotheses", [])) for item in items]
    avg_n_hypotheses = sum(n_hypotheses_per_item) / len(items) if items else 1
    chance_level = 1.0 / avg_n_hypotheses
    threshold = chance_level + alpha

    # Run all baseline methods
    majority_result = majority_class_baseline(items)
    lexical_result = lexical_overlap_heuristic(items)
    bow_result = bow_logistic_baseline(items)

    # Determine PASS/FAIL for each method
    methods_results = {
        "majority_class": {
            **majority_result,
            "verdict": "FAIL" if majority_result["accuracy"] > threshold else "PASS"
        },
        "lexical_overlap": {
            **lexical_result,
            "verdict": "FAIL" if lexical_result["accuracy"] > threshold else "PASS"
        },
        "bow_logistic": {
            **bow_result,
            "verdict": "FAIL" if bow_result["accuracy"] > threshold else "PASS"
        }
    }

    # Overall verdict: FAIL if any method fails
    overall_verdict = "FAIL" if any(
        m["verdict"] == "FAIL" for m in methods_results.values()
    ) else "PASS"

    report = {
        "n_items": len(items),
        "avg_hypotheses_per_item": avg_n_hypotheses,
        "chance_level": chance_level,
        "threshold": threshold,
        "alpha": alpha,
        "methods": methods_results,
        "overall_verdict": overall_verdict
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run leakage detection checks on T2 dataset"
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to output JSON report (optional)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Threshold margin above chance level (default: 0.05)"
    )
    args = parser.parse_args()

    # Load items and run leakage check
    items = load_items_from_jsonl(args.input)
    report = run_leakage_check(items, alpha=args.alpha)

    # Print summary to stdout
    print("Leakage Check Report")
    print("====================")
    print(f"Items: {report['n_items']}")
    print(f"Chance level: {report['chance_level']:.3f}")
    print(f"Threshold: {report['threshold']:.3f}")
    print(f"\nResults:")
    for method, result in report["methods"].items():
        print(f"  {method}: {result['accuracy']:.3f} [{result['verdict']}]")
    print(f"\nOverall verdict: {report['overall_verdict']}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.output}")

    # Exit with error code if failed
    sys.exit(1 if report["overall_verdict"] == "FAIL" else 0)
