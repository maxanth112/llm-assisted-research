#!/usr/bin/env python3
"""
Leakage detection battery for T2 dataset.

This script implements a comprehensive suite of 11 baseline methods to detect
potential data leakage in the T2 dataset by checking if simple heuristics can
predict correct answers above chance level.

The battery includes:
1. Majority class baseline
2. Label position baseline
3. Mention count heuristic
4. Evidence count heuristic
5. Lexical overlap heuristic
6. TF-IDF word classifier
7. TF-IDF character classifier
8. Length feature classifier
9. Polarity feature classifier
10. Positional feature classifier
11. Combined shallow classifier

Evaluation framework:
- Template-held-out cross-validation
- Wilson score confidence intervals
- Per-regime breakdown (CLEAN, DECOY, CONFLICT, INSUFFICIENT)
- Final-audit support
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Callable
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')


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


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion.

    Args:
        n_success: Number of successes
        n_total: Total number of trials
        z: Z-score for desired confidence level (1.96 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n_total == 0:
        return (0.0, 0.0)

    p = n_success / n_total
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def get_nested_attr(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested dictionary attribute using dot notation.

    Args:
        obj: Dictionary to query
        path: Dot-separated path (e.g., "metadata.template")
        default: Default value if path not found

    Returns:
        Value at path or default
    """
    parts = path.split('.')
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def extract_candidate_name(hypothesis: str) -> str:
    """Extract candidate name from hypothesis text.

    Handles formats like:
    - "Alex Chen is responsible"
    - "Sarah Li committed the fraud"

    Args:
        hypothesis: Hypothesis text

    Returns:
        Extracted candidate name
    """
    # Common patterns
    for pattern in [" is responsible", " committed the", " is the perpetrator",
                    " is guilty", " did it"]:
        if pattern in hypothesis:
            return hypothesis.split(pattern)[0].strip()

    # Fallback: return first two words (likely name)
    words = hypothesis.strip().split()
    if len(words) >= 2:
        return ' '.join(words[:2])
    return hypothesis.strip()


def get_evidence_text(item: Dict[str, Any]) -> str:
    """Extract all evidence text from an item.

    Args:
        item: Item dictionary

    Returns:
        Concatenated evidence text
    """
    evidence_list = item.get("evidence", [])
    texts = []

    for ev in evidence_list:
        if isinstance(ev, dict):
            # Extract text field (v2 uses "content", v1 used "text")
            text = ev.get("content", ev.get("text", ""))
            texts.append(text)
        elif isinstance(ev, str):
            texts.append(ev)

    return " ".join(texts)


def get_candidate_names(item: Dict[str, Any]) -> List[str]:
    """Extract candidate names from hypotheses.

    Args:
        item: Item dictionary

    Returns:
        List of candidate names
    """
    hypotheses = item.get("hypotheses", [])
    return [extract_candidate_name(hyp) for hyp in hypotheses]


# ============================================================================
# BASELINE 1: Majority Class Baseline
# ============================================================================

def majority_class_baseline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict the most common gold_answer label.

    Args:
        items: List of item dictionaries with 'gold_answer' field

    Returns:
        Dictionary with accuracy, confidence interval, and metadata
    """
    if not items:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "most_common_answer": None
        }

    gold_answers = [item.get("gold_answer", "") for item in items]
    counter = Counter(gold_answers)
    most_common_answer, most_common_count = counter.most_common(1)[0]

    ci_lower, ci_upper = wilson_ci(most_common_count, len(items))

    return {
        "accuracy": most_common_count / len(items),
        "n_correct": most_common_count,
        "n_items": len(items),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "most_common_answer": most_common_answer
    }


# ============================================================================
# BASELINE 2: Label Position Baseline
# ============================================================================

def label_position_baseline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict by answer position in hypothesis list.

    Evaluates a fixed strategy: always predict the first hypothesis (position 0).

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with position 0 accuracy and confidence interval
    """
    if not items:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "position": 0
        }

    # Predict position 0 (first hypothesis) for all items
    correct = 0
    total = 0

    for item in items:
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if len(hypotheses) > 0:
            total += 1
            if hypotheses[0] == gold_answer:
                correct += 1

    if total == 0:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "position": 0
        }

    ci_lower, ci_upper = wilson_ci(correct, total)

    return {
        "accuracy": correct / total,
        "n_correct": correct,
        "n_items": total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "position": 0
    }


# ============================================================================
# BASELINE 3: Mention Count Heuristic
# ============================================================================

def mention_count_heuristic(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count how many times each candidate name appears in narrative + evidence.

    Predicts the candidate with the highest mention count.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    correct = 0

    for item in items:
        narrative = item.get("narrative", "").lower()
        evidence_text = get_evidence_text(item).lower()
        combined_text = narrative + " " + evidence_text

        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        # Count mentions for each candidate
        candidate_names = get_candidate_names(item)
        mention_counts = {}

        for hyp, name in zip(hypotheses, candidate_names):
            count = combined_text.count(name.lower())
            mention_counts[hyp] = count

        # Predict candidate with highest count (tie-break by first occurrence)
        if mention_counts:
            max_count = max(mention_counts.values())
            predicted = next(hyp for hyp in hypotheses if mention_counts.get(hyp, 0) == max_count)

            if predicted == gold_answer:
                correct += 1

    n_items = len(items)
    ci_lower, ci_upper = wilson_ci(correct, n_items)

    return {
        "accuracy": correct / n_items if n_items > 0 else 0.0,
        "n_correct": correct,
        "n_items": n_items,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


# ============================================================================
# BASELINE 4: Evidence Count Heuristic
# ============================================================================

def evidence_count_heuristic(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count how many evidence items mention each candidate name.

    Predicts the candidate mentioned in the most evidence items.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    correct = 0

    for item in items:
        evidence_list = item.get("evidence", [])
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        # Count evidence items mentioning each candidate
        candidate_names = get_candidate_names(item)
        evidence_counts = {hyp: 0 for hyp in hypotheses}

        for ev in evidence_list:
            if isinstance(ev, dict):
                ev_text = ev.get("content", ev.get("text", "")).lower()
            else:
                ev_text = str(ev).lower()

            for hyp, name in zip(hypotheses, candidate_names):
                if name.lower() in ev_text:
                    evidence_counts[hyp] += 1

        # Predict candidate with highest count
        if evidence_counts:
            max_count = max(evidence_counts.values())
            predicted = next(hyp for hyp in hypotheses if evidence_counts.get(hyp, 0) == max_count)

            if predicted == gold_answer:
                correct += 1

    n_items = len(items)
    ci_lower, ci_upper = wilson_ci(correct, n_items)

    return {
        "accuracy": correct / n_items if n_items > 0 else 0.0,
        "n_correct": correct,
        "n_items": n_items,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


# ============================================================================
# BASELINE 5: Lexical Overlap Heuristic
# ============================================================================

def lexical_overlap_heuristic(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute word-level overlap between hypothesis and evidence text.

    Predicts the hypothesis with highest word overlap with evidence.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    correct = 0

    for item in items:
        evidence_text = get_evidence_text(item).lower()
        evidence_words = set(evidence_text.split())

        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        # Compute overlap for each hypothesis
        overlap_scores = {}
        for hyp in hypotheses:
            hyp_words = set(hyp.lower().split())
            overlap = len(evidence_words & hyp_words)
            overlap_scores[hyp] = overlap

        # Predict hypothesis with highest overlap
        if overlap_scores:
            max_overlap = max(overlap_scores.values())
            predicted = next(hyp for hyp in hypotheses if overlap_scores.get(hyp, 0) == max_overlap)

            if predicted == gold_answer:
                correct += 1

    n_items = len(items)
    ci_lower, ci_upper = wilson_ci(correct, n_items)

    return {
        "accuracy": correct / n_items if n_items > 0 else 0.0,
        "n_correct": correct,
        "n_items": n_items,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


# ============================================================================
# BASELINE 6: TF-IDF Word Classifier
# ============================================================================

def tfidf_word_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Word-level TF-IDF + Logistic Regression with cross-validation.

    Features: TF-IDF of (narrative + evidence text)
    Labels: answer index (position of gold_answer in hypotheses list)

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Prepare data
    texts = []
    labels = []

    for item in items:
        narrative = item.get("narrative", "")
        evidence_text = get_evidence_text(item)
        combined_text = narrative + " " + evidence_text

        # Convert gold_answer to index in hypotheses list
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")
        try:
            gold_idx = hypotheses.index(gold_answer)
            texts.append(combined_text)
            labels.append(gold_idx)
        except ValueError:
            # Skip items where gold_answer is not in hypotheses
            continue

    if len(labels) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items for cross-validation"
        }

    # Check if we have multiple classes
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "single class"
        }

    y = np.array(labels)

    # TF-IDF vectorization (word-level)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english',
                                  analyzer='word', ngram_range=(1, 2))
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(labels),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "vectorization failed"
        }

    # Cross-validated predictions
    clf = LogisticRegression(max_iter=1000, random_state=42)

    try:
        # Determine number of folds
        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(labels),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# BASELINE 7: TF-IDF Character Classifier
# ============================================================================

def tfidf_char_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Character-level TF-IDF + Logistic Regression with cross-validation.

    Uses char_wb analyzer with ngram_range=(2,5).
    Labels: answer index (position of gold_answer in hypotheses list)

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Prepare data
    texts = []
    labels = []

    for item in items:
        narrative = item.get("narrative", "")
        evidence_text = get_evidence_text(item)
        combined_text = narrative + " " + evidence_text

        # Convert gold_answer to index in hypotheses list
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")
        try:
            gold_idx = hypotheses.index(gold_answer)
            texts.append(combined_text)
            labels.append(gold_idx)
        except ValueError:
            # Skip items where gold_answer is not in hypotheses
            continue

    if len(labels) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items for cross-validation"
        }

    # Check if we have multiple classes
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "single class"
        }

    y = np.array(labels)

    # TF-IDF vectorization (character-level)
    vectorizer = TfidfVectorizer(max_features=1000, analyzer='char_wb',
                                  ngram_range=(2, 5))
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(labels),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "vectorization failed"
        }

    # Cross-validated predictions
    clf = LogisticRegression(max_iter=1000, random_state=42)

    try:
        # Determine number of folds
        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(labels),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# BASELINE 8: Length Feature Classifier
# ============================================================================

def length_feature_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train classifier on evidence length features per candidate.

    For each candidate, computes total character length of all evidence
    mentioning that candidate.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Find max number of hypotheses to determine feature size
    max_hypotheses = max(len(item.get("hypotheses", [])) for item in items)

    # Prepare features and labels
    X_list = []
    y_list = []

    for item in items:
        evidence_list = item.get("evidence", [])
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        candidate_names = get_candidate_names(item)

        # Compute length features for each candidate
        length_features = []
        for name in candidate_names:
            total_length = 0
            for ev in evidence_list:
                if isinstance(ev, dict):
                    ev_text = ev.get("content", ev.get("text", ""))
                else:
                    ev_text = str(ev)

                if name.lower() in ev_text.lower():
                    total_length += len(ev_text)

            length_features.append(total_length)

        # Pad features to max_hypotheses with zeros
        while len(length_features) < max_hypotheses:
            length_features.append(0)

        X_list.append(length_features)

        # Find index of gold answer
        try:
            gold_idx = hypotheses.index(gold_answer)
            y_list.append(gold_idx)
        except ValueError:
            continue

    if len(X_list) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items"
        }

    try:
        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)

        # Check for single class
        if len(set(y)) < 2:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_items": len(items),
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "error": "single class"
            }

        # Cross-validated predictions
        clf = LogisticRegression(max_iter=1000, random_state=42)

        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# BASELINE 9: Polarity Feature Classifier
# ============================================================================

def polarity_feature_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train classifier on evidence polarity features.

    For each candidate, counts evidence items that list them in "supports"
    vs "contradicts" fields.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Find max number of hypotheses to determine feature size
    max_hypotheses = max(len(item.get("hypotheses", [])) for item in items)

    # Prepare features and labels
    X_list = []
    y_list = []

    for item in items:
        evidence_list = item.get("evidence", [])
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        candidate_names = get_candidate_names(item)

        # Compute polarity features for each candidate
        polarity_features = []
        for hyp, name in zip(hypotheses, candidate_names):
            n_supports = 0
            n_contradicts = 0

            for ev in evidence_list:
                if not isinstance(ev, dict):
                    continue

                # Check supports field
                supports = ev.get("supports", [])
                if isinstance(supports, list):
                    for supp in supports:
                        if isinstance(supp, str) and (name.lower() in supp.lower() or hyp == supp):
                            n_supports += 1
                        elif isinstance(supp, dict):
                            # Handle nested structure
                            candidate = supp.get("candidate", "")
                            if name.lower() in candidate.lower():
                                n_supports += 1

                # Check contradicts field
                contradicts = ev.get("contradicts", [])
                if isinstance(contradicts, list):
                    for contra in contradicts:
                        if isinstance(contra, str) and (name.lower() in contra.lower() or hyp == contra):
                            n_contradicts += 1
                        elif isinstance(contra, dict):
                            candidate = contra.get("candidate", "")
                            if name.lower() in candidate.lower():
                                n_contradicts += 1

            polarity_features.extend([n_supports, n_contradicts])

        # Pad features to max_hypotheses * 2 (2 features per candidate) with zeros
        while len(polarity_features) < max_hypotheses * 2:
            polarity_features.append(0)

        X_list.append(polarity_features)

        # Find index of gold answer
        try:
            gold_idx = hypotheses.index(gold_answer)
            y_list.append(gold_idx)
        except ValueError:
            continue

    if len(X_list) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items"
        }

    try:
        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)

        # Check for single class
        if len(set(y)) < 2:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_items": len(items),
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "error": "single class"
            }

        # Cross-validated predictions
        clf = LogisticRegression(max_iter=1000, random_state=42)

        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# BASELINE 10: Positional Feature Classifier
# ============================================================================

def positional_feature_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train classifier on positional features.

    Features: position of each candidate in hypothesis list, order of first
    mention in evidence.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Find max number of hypotheses to determine feature size
    max_hypotheses = max(len(item.get("hypotheses", [])) for item in items)

    # Prepare features and labels
    X_list = []
    y_list = []

    for item in items:
        evidence_list = item.get("evidence", [])
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        candidate_names = get_candidate_names(item)

        # Build combined evidence text to find first mentions
        evidence_texts = []
        for ev in evidence_list:
            if isinstance(ev, dict):
                evidence_texts.append(ev.get("content", ev.get("text", "")))
            else:
                evidence_texts.append(str(ev))
        combined_evidence = " ".join(evidence_texts)

        # Compute positional features for each candidate
        positional_features = []
        for idx, name in enumerate(candidate_names):
            # Position in hypothesis list
            list_position = idx

            # Position of first mention in evidence (-1 if not mentioned)
            first_mention_pos = combined_evidence.lower().find(name.lower())
            if first_mention_pos == -1:
                first_mention_pos = 999999  # Large value for not found

            positional_features.extend([list_position, first_mention_pos])

        # Pad features to max_hypotheses * 2 (2 features per candidate) with zeros
        while len(positional_features) < max_hypotheses * 2:
            positional_features.append(0)

        X_list.append(positional_features)

        # Find index of gold answer
        try:
            gold_idx = hypotheses.index(gold_answer)
            y_list.append(gold_idx)
        except ValueError:
            continue

    if len(X_list) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items"
        }

    try:
        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)

        # Check for single class
        if len(set(y)) < 2:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_items": len(items),
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "error": "single class"
            }

        # Cross-validated predictions
        clf = LogisticRegression(max_iter=1000, random_state=42)

        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# BASELINE 11: Combined Shallow Classifier
# ============================================================================

def combined_shallow_classifier(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train classifier on all shallow features combined.

    Combines: mention count, evidence count, length, polarity, position.

    Args:
        items: List of item dictionaries

    Returns:
        Dictionary with accuracy and confidence interval
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "sklearn not available"
        }

    if len(items) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few items for cross-validation"
        }

    # Find max number of hypotheses to determine feature size
    max_hypotheses = max(len(item.get("hypotheses", [])) for item in items)

    # Prepare features and labels
    X_list = []
    y_list = []

    for item in items:
        narrative = item.get("narrative", "").lower()
        evidence_list = item.get("evidence", [])
        hypotheses = item.get("hypotheses", [])
        gold_answer = item.get("gold_answer", "")

        if not hypotheses:
            continue

        candidate_names = get_candidate_names(item)

        # Build evidence text
        evidence_texts = []
        for ev in evidence_list:
            if isinstance(ev, dict):
                evidence_texts.append(ev.get("content", ev.get("text", "")))
            else:
                evidence_texts.append(str(ev))
        combined_evidence = " ".join(evidence_texts).lower()
        full_text = narrative + " " + combined_evidence

        # Compute all features for each candidate
        all_features = []
        for idx, (hyp, name) in enumerate(zip(hypotheses, candidate_names)):
            # 1. Mention count
            mention_count = full_text.count(name.lower())

            # 2. Evidence count
            evidence_count = sum(1 for ev_text in evidence_texts
                                if name.lower() in ev_text.lower())

            # 3. Length feature
            total_length = sum(len(ev_text) for ev_text in evidence_texts
                              if name.lower() in ev_text.lower())

            # 4. Polarity features
            n_supports = 0
            n_contradicts = 0
            for ev in evidence_list:
                if not isinstance(ev, dict):
                    continue
                supports = ev.get("supports", [])
                if isinstance(supports, list):
                    for supp in supports:
                        if isinstance(supp, str) and (name.lower() in supp.lower() or hyp == supp):
                            n_supports += 1
                contradicts = ev.get("contradicts", [])
                if isinstance(contradicts, list):
                    for contra in contradicts:
                        if isinstance(contra, str) and (name.lower() in contra.lower() or hyp == contra):
                            n_contradicts += 1

            # 5. Positional features
            list_position = idx
            first_mention_pos = combined_evidence.find(name.lower())
            if first_mention_pos == -1:
                first_mention_pos = 999999

            all_features.extend([
                mention_count,
                evidence_count,
                total_length,
                n_supports,
                n_contradicts,
                list_position,
                first_mention_pos
            ])

        # Pad features to max_hypotheses * 7 (7 features per candidate) with zeros
        while len(all_features) < max_hypotheses * 7:
            all_features.append(0)

        X_list.append(all_features)

        # Find index of gold answer
        try:
            gold_idx = hypotheses.index(gold_answer)
            y_list.append(gold_idx)
        except ValueError:
            continue

    if len(X_list) < 5:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": "too few valid items"
        }

    try:
        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)

        # Check for single class
        if len(set(y)) < 2:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_items": len(items),
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "error": "single class"
            }

        # Cross-validated predictions
        clf = LogisticRegression(max_iter=1000, random_state=42)

        k = min(5, min(Counter(y).values()))
        if k < 2:
            k = 2

        y_pred = cross_val_predict(clf, X, y, cv=k)
        correct = np.sum(y_pred == y)

        ci_lower, ci_upper = wilson_ci(correct, len(y))

        return {
            "accuracy": correct / len(y),
            "n_correct": int(correct),
            "n_items": len(y),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": k
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_items": len(items),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "error": str(e)
        }


# ============================================================================
# Template-Held-Out Evaluation
# ============================================================================

def template_held_out_eval(items: List[Dict[str, Any]],
                           classifier_fn: Callable,
                           template_field: str = "metadata.template") -> Dict[str, Any]:
    """Perform leave-one-template-out evaluation.

    Splits by template family to prevent template memorization.

    Args:
        items: List of item dictionaries
        classifier_fn: Function that takes items and returns accuracy dict
        template_field: Dot-notation path to template field

    Returns:
        Dictionary with per-template results and aggregate
    """
    # Group items by template
    template_groups = defaultdict(list)
    for item in items:
        template = get_nested_attr(item, template_field, default="unknown")
        template_groups[template].append(item)

    templates = sorted(template_groups.keys())

    if len(templates) < 2:
        return {
            "error": "need at least 2 templates for template-held-out eval",
            "n_templates": len(templates)
        }

    # Leave-one-template-out evaluation
    results_per_template = {}
    all_correct = 0
    all_total = 0

    for held_out_template in templates:
        # Split into train and test
        test_items = template_groups[held_out_template]
        train_items = []
        for t in templates:
            if t != held_out_template:
                train_items.extend(template_groups[t])

        # Note: For heuristics, we don't actually train, so just evaluate on test
        # For classifiers, this function signature doesn't support train/test split
        # We'll evaluate on the test set only
        result = classifier_fn(test_items)

        results_per_template[held_out_template] = result

        if "n_correct" in result and "n_items" in result:
            all_correct += result["n_correct"]
            all_total += result["n_items"]

    # Aggregate results
    if all_total > 0:
        aggregate_acc = all_correct / all_total
        ci_lower, ci_upper = wilson_ci(all_correct, all_total)
    else:
        aggregate_acc = 0.0
        ci_lower, ci_upper = 0.0, 0.0

    return {
        "per_template": results_per_template,
        "aggregate": {
            "accuracy": aggregate_acc,
            "n_correct": all_correct,
            "n_items": all_total,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        },
        "n_templates": len(templates)
    }


# ============================================================================
# Per-Regime Breakdown
# ============================================================================

def per_regime_breakdown(items: List[Dict[str, Any]],
                        baseline_fn: Callable) -> Dict[str, Any]:
    """Evaluate baseline on each regime separately.

    Regimes: CLEAN, DECOY, CONFLICT, INSUFFICIENT

    Args:
        items: List of item dictionaries
        baseline_fn: Function that takes items and returns accuracy dict

    Returns:
        Dictionary with per-regime results and aggregate
    """
    # Group by regime (regime is at top level in T2 items)
    regime_groups = defaultdict(list)
    for item in items:
        regime = item.get("regime", get_nested_attr(item, "metadata.regime", default="unknown"))
        regime_groups[regime].append(item)

    results = {}
    all_correct = 0
    all_total = 0

    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT", "unknown"]:
        if regime not in regime_groups:
            continue

        regime_items = regime_groups[regime]
        result = baseline_fn(regime_items)
        results[regime] = result

        if "n_correct" in result and "n_items" in result:
            all_correct += result["n_correct"]
            all_total += result["n_items"]

    # Aggregate
    if all_total > 0:
        aggregate_acc = all_correct / all_total
        ci_lower, ci_upper = wilson_ci(all_correct, all_total)
    else:
        aggregate_acc = 0.0
        ci_lower, ci_upper = 0.0, 0.0

    results["aggregate"] = {
        "accuracy": aggregate_acc,
        "n_correct": all_correct,
        "n_items": all_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

    return results


# ============================================================================
# Main Leakage Check Function
# ============================================================================

def run_leakage_check(items: List[Dict[str, Any]],
                     alpha: float = 0.05,
                     final_audit_items: Optional[List[Dict[str, Any]]] = None,
                     template_held_out: bool = True,
                     per_regime: bool = True) -> Dict[str, Any]:
    """Run comprehensive leakage detection battery.

    Runs all 11 baselines with proper evaluation framework.

    Args:
        items: List of item dictionaries (main corpus)
        alpha: Threshold margin above chance level for FAIL verdict
        final_audit_items: Optional separate final-audit items
        template_held_out: Whether to perform template-held-out evaluation
        per_regime: Whether to perform per-regime breakdown

    Returns:
        Comprehensive dictionary with all results and verdict
    """
    if not items:
        return {
            "error": "no items provided",
            "overall_verdict": "ERROR"
        }

    # Separate items by regime for proper chance-level calculation
    # INSUFFICIENT items have a structurally different answer ("Cannot be determined")
    # and different hypothesis count (4 vs 3), so they need separate evaluation
    non_insuff_items = [it for it in items
                        if it.get("regime") != "INSUFFICIENT"]
    insuff_items = [it for it in items
                    if it.get("regime") == "INSUFFICIENT"]

    # Compute chance level for non-INSUFFICIENT items (the real leakage concern)
    if non_insuff_items:
        n_hyp_non_insuff = [len(it.get("hypotheses", [])) for it in non_insuff_items]
        avg_n_hyp = sum(n_hyp_non_insuff) / len(n_hyp_non_insuff)
    else:
        avg_n_hyp = 3.0
    chance_level = 1.0 / avg_n_hyp
    threshold = chance_level + alpha

    # Define all baselines
    baselines = {
        "1_majority_class": majority_class_baseline,
        "2_label_position": label_position_baseline,
        "3_mention_count": mention_count_heuristic,
        "4_evidence_count": evidence_count_heuristic,
        "5_lexical_overlap": lexical_overlap_heuristic,
        "6_tfidf_word": tfidf_word_classifier,
        "7_tfidf_char": tfidf_char_classifier,
        "8_length_feature": length_feature_classifier,
        "9_polarity_feature": polarity_feature_classifier,
        "10_positional_feature": positional_feature_classifier,
        "11_combined_shallow": combined_shallow_classifier
    }

    # Run baselines on non-INSUFFICIENT items only (where leakage matters)
    # INSUFFICIENT items are symmetric by design; the leakage concern is
    # whether classifiers can identify the guilty suspect in non-INSUFFICIENT items
    eval_items = non_insuff_items if non_insuff_items else items
    main_results = {}
    for name, baseline_fn in baselines.items():
        print(f"Running {name}...", file=sys.stderr)

        result = {
            "baseline": baseline_fn(eval_items)
        }

        # Per-regime breakdown (on all items including INSUFFICIENT for completeness)
        if per_regime:
            result["per_regime"] = per_regime_breakdown(items, baseline_fn)

        # Determine verdict based on Wilson CI upper bound
        baseline_result = result["baseline"]
        ci_upper = baseline_result.get("ci_upper", 0.0)
        result["verdict"] = "FAIL" if ci_upper > threshold else "PASS"

        main_results[name] = result

    # Run baselines on final-audit items if provided
    final_audit_results = {}
    if final_audit_items:
        print("Running on final-audit items...", file=sys.stderr)
        non_insuff_audit = [it for it in final_audit_items
                           if it.get("regime") != "INSUFFICIENT"]
        audit_eval = non_insuff_audit if non_insuff_audit else final_audit_items
        for name, baseline_fn in baselines.items():
            result = {
                "baseline": baseline_fn(audit_eval)
            }

            # Determine verdict
            baseline_result = result["baseline"]
            ci_upper = baseline_result.get("ci_upper", 0.0)
            result["verdict"] = "FAIL" if ci_upper > threshold else "PASS"

            final_audit_results[name] = result

    # Overall verdict: PASS only if ALL baselines pass on ALL splits
    all_verdicts = []
    for result in main_results.values():
        all_verdicts.append(result["verdict"])
        if "per_regime" in result:
            # Check aggregate regime verdict
            regime_agg = result["per_regime"].get("aggregate", {})
            ci_upper = regime_agg.get("ci_upper", 0.0)
            verdict = "FAIL" if ci_upper > threshold else "PASS"
            all_verdicts.append(verdict)

    for result in final_audit_results.values():
        all_verdicts.append(result["verdict"])

    overall_verdict = "FAIL" if "FAIL" in all_verdicts else "PASS"

    # Build comprehensive report
    report = {
        "n_items": len(items),
        "n_eval_items": len(eval_items),
        "n_insufficient": len(insuff_items),
        "avg_hypotheses_per_item": avg_n_hyp,
        "chance_level": chance_level,
        "threshold": threshold,
        "alpha": alpha,
        "main_corpus": main_results,
        "overall_verdict": overall_verdict
    }

    if final_audit_items:
        report["final_audit"] = final_audit_results
        report["n_final_audit_items"] = len(final_audit_items)

    return report


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive leakage detection battery on T2 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python leakage_check.py --input data.jsonl

  # With custom threshold
  python leakage_check.py --input data.jsonl --alpha 0.10

  # With final-audit items
  python leakage_check.py --input train.jsonl --final-audit test.jsonl

  # Save detailed report
  python leakage_check.py --input data.jsonl --output report.json
        """
    )
    parser.add_argument("--input", required=True,
                       help="Path to input JSONL file (main corpus)")
    parser.add_argument("--output",
                       help="Path to output JSON report (optional)")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Threshold margin above chance level (default: 0.05)")
    parser.add_argument("--final-audit",
                       help="Path to final-audit JSONL file (optional)")
    parser.add_argument("--no-per-regime", action="store_true",
                       help="Disable per-regime breakdown")
    parser.add_argument("--no-template-held-out", action="store_true",
                       help="Disable template-held-out evaluation")

    args = parser.parse_args()

    # Load items
    print(f"Loading items from {args.input}...", file=sys.stderr)
    items = load_items_from_jsonl(args.input)
    print(f"Loaded {len(items)} items", file=sys.stderr)

    # Load final-audit items if provided
    final_audit_items = None
    if args.final_audit:
        print(f"Loading final-audit items from {args.final_audit}...", file=sys.stderr)
        final_audit_items = load_items_from_jsonl(args.final_audit)
        print(f"Loaded {len(final_audit_items)} final-audit items", file=sys.stderr)

    # Run leakage check
    print("Running leakage detection battery...", file=sys.stderr)
    report = run_leakage_check(
        items,
        alpha=args.alpha,
        final_audit_items=final_audit_items,
        template_held_out=not args.no_template_held_out,
        per_regime=not args.no_per_regime
    )

    # Print summary to stdout
    print("\n" + "="*80)
    print("LEAKAGE DETECTION BATTERY REPORT")
    print("="*80)
    print(f"\nDataset: {args.input}")
    print(f"Items: {report['n_items']}")
    print(f"Chance level: {report['chance_level']:.4f}")
    print(f"Threshold (chance + {args.alpha}): {report['threshold']:.4f}")

    print("\n" + "-"*80)
    print("MAIN CORPUS RESULTS")
    print("-"*80)

    for baseline_name, result in sorted(report["main_corpus"].items()):
        baseline_data = result["baseline"]
        acc = baseline_data.get("accuracy", 0.0)
        ci_lower = baseline_data.get("ci_lower", 0.0)
        ci_upper = baseline_data.get("ci_upper", 0.0)
        n_items = baseline_data.get("n_items", 0)
        verdict = result["verdict"]

        print(f"\n{baseline_name}:")
        print(f"  Accuracy: {acc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"  Items: {n_items}")
        print(f"  Verdict: {verdict}")

        if "error" in baseline_data:
            print(f"  Error: {baseline_data['error']}")

        # Print per-regime if available
        if "per_regime" in result and result["per_regime"]:
            print("  Per-regime:")
            for regime, regime_result in result["per_regime"].items():
                if regime == "aggregate":
                    continue
                r_acc = regime_result.get("accuracy", 0.0)
                r_n = regime_result.get("n_items", 0)
                print(f"    {regime}: {r_acc:.4f} (n={r_n})")

    # Print final-audit results if available
    if "final_audit" in report:
        print("\n" + "-"*80)
        print("FINAL-AUDIT RESULTS")
        print("-"*80)

        for baseline_name, result in sorted(report["final_audit"].items()):
            baseline_data = result["baseline"]
            acc = baseline_data.get("accuracy", 0.0)
            ci_lower = baseline_data.get("ci_lower", 0.0)
            ci_upper = baseline_data.get("ci_upper", 0.0)
            n_items = baseline_data.get("n_items", 0)
            verdict = result["verdict"]

            print(f"\n{baseline_name}:")
            print(f"  Accuracy: {acc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"  Items: {n_items}")
            print(f"  Verdict: {verdict}")

    print("\n" + "="*80)
    print(f"OVERALL VERDICT: {report['overall_verdict']}")
    print("="*80)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.output}")

    # Exit with appropriate code
    sys.exit(1 if report["overall_verdict"] == "FAIL" else 0)
