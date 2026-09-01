"""
v3.1 Diagnostic baselines — complement the existing 11-baseline battery.

These baselines disentangle three potential leakage channels:

  D1 (option_only): Uses ONLY the hypothesis texts (no narrative, no evidence).
     Measures how much the option SET alone leaks the answer.  If this
     baseline exceeds chance, the hypothesis wording is not lexically
     parallel enough.

  D2 (context_regime): Uses ONLY narrative + evidence context (no options)
     to predict whether the item is INSUFFICIENT vs answerable.  This is
     a binary classifier, NOT a candidate-aware argmax.  If accuracy
     significantly exceeds 50%, the context alone reveals the regime,
     which would let a model "cheat" by identifying INSUFFICIENT items
     and always choosing abstention.

  D3 (full_candidate): The complete candidate-aware TARGET-normalized
     pipeline (same as the existing TF-IDF baselines in run_leakage_eval).
     Included here for comparison against D1 and D2.

Design:
  - D1 and D3 predict hypothesis indices (same as existing baselines).
  - D2 predicts a binary label (0=answerable, 1=INSUFFICIENT).
  - All use template-held-out cross-validation.
  - All are deterministic from frozen seeds.
"""

import gc
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from analysis.run_leakage_eval import (
    extract_name,
    ev_content_text,
    gold_index,
    wilson_ci,
    chance_level_correct,
    _extract_all_names,
    _target_normalized_candidate_text,
    _prepare_candidate_rows,
)

warnings.filterwarnings('ignore')


# ================================================================
# D1: Option-only baseline
# ================================================================

def pred_option_only(train_items, test_items) -> np.ndarray:
    """Predict gold answer using ONLY the hypothesis texts.

    For each candidate row: the feature text is JUST the hypothesis string
    (no narrative, no evidence).  Trained via the same candidate-aware
    argmax-of-probabilities pipeline.

    If hypothesis wording is truly lexically parallel, this should be at
    chance (0.25 for 4-option items).
    """
    builder = _OptionOnlyFeatureBuilder()
    return _candidate_predict_diagnostic(train_items, test_items, builder)


class _OptionOnlyFeatureBuilder:
    """TF-IDF features from hypothesis text only (no context)."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=200, analyzer='word', ngram_range=(1, 2),
            stop_words='english', dtype=np.float32,
        )
        self._fitted = False

    def __call__(self, items, is_train=False):
        texts = []
        labels = []
        item_ids = []
        valid_mask = np.ones(len(items), dtype=bool)

        for i, it in enumerate(items):
            gi = gold_index(it)
            if gi < 0:
                valid_mask[i] = False
            for j, hyp in enumerate(it["hypotheses"]):
                # ONLY the hypothesis text — no context
                texts.append(hyp)
                labels.append(1 if (gi >= 0 and j == gi) else 0)
                item_ids.append(i)

        if is_train:
            X = self.vec.fit_transform(texts)
            self._fitted = True
        else:
            X = self.vec.transform(texts)

        return X, np.array(labels), np.array(item_ids), valid_mask


# ================================================================
# D2: Context-only regime classifier
# ================================================================

def pred_context_regime(train_items, test_items) -> np.ndarray:
    """Binary classifier: predict INSUFFICIENT (1) vs answerable (0).

    Uses ONLY narrative + evidence context (no hypothesis texts).
    Returns binary predictions (not hypothesis indices).
    """
    # Build features: TF-IDF on narrative + evidence only
    vec = TfidfVectorizer(
        max_features=300, analyzer='word', ngram_range=(1, 2),
        stop_words='english', dtype=np.float32,
    )

    train_texts = [
        it.get("narrative", "") + " " + ev_content_text(it)
        for it in train_items
    ]
    test_texts = [
        it.get("narrative", "") + " " + ev_content_text(it)
        for it in test_items
    ]

    train_labels = np.array([
        1 if it.get("regime") == "INSUFFICIENT" else 0
        for it in train_items
    ])
    test_labels = np.array([
        1 if it.get("regime") == "INSUFFICIENT" else 0
        for it in test_items
    ])

    tr_X = vec.fit_transform(train_texts)
    te_X = vec.transform(test_texts)

    if len(set(train_labels.tolist())) < 2:
        # Degenerate: predict majority
        return np.full(len(test_items), int(np.argmax(np.bincount(train_labels))))

    clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
    clf.fit(tr_X, train_labels)

    preds = clf.predict(te_X)
    del clf, tr_X, te_X
    gc.collect()

    return preds


def evaluate_context_regime(items, alpha=0.05):
    """Template-held-out evaluation for the context-regime classifier.

    Returns:
        dict with accuracy, ci_lower, ci_upper, chance (0.50 for binary),
        threshold, verdict, and per-regime breakdown.
    """
    by_template = defaultdict(list)
    for it in items:
        t = it.get("metadata", {}).get("template", "unknown")
        by_template[t].append(it)
    templates = sorted(by_template.keys())

    # Collect predictions across folds
    ordered_items = []
    fold_preds = []

    for t in templates:
        test_items = by_template[t]
        train_items = []
        for ot in templates:
            if ot != t:
                train_items.extend(by_template[ot])

        if not train_items:
            # Skip if no train data
            for it in test_items:
                ordered_items.append(it)
                fold_preds.append(0)
            continue

        preds = pred_context_regime(train_items, test_items)
        for it, p in zip(test_items, preds):
            ordered_items.append(it)
            fold_preds.append(int(p))

    # Compute accuracy
    true_labels = np.array([
        1 if it.get("regime") == "INSUFFICIENT" else 0
        for it in ordered_items
    ])
    preds_arr = np.array(fold_preds)

    n = len(true_labels)
    k = int(np.sum(preds_arr == true_labels))
    acc = k / n if n > 0 else 0
    ci_lo, ci_hi = wilson_ci(k, n)

    # Chance for binary = 0.50
    chance = 0.50
    threshold = chance + alpha

    result = {
        "n_items": n,
        "n_correct": k,
        "accuracy": round(acc, 5),
        "ci_lower": round(ci_lo, 5),
        "ci_upper": round(ci_hi, 5),
        "chance": chance,
        "threshold": round(threshold, 5),
        "verdict": "FAIL" if ci_hi > threshold else "PASS",
    }

    # Per-regime breakdown (binary accuracy within each regime)
    per_regime = {}
    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
        mask = np.array([
            it.get("regime") == regime for it in ordered_items
        ])
        if not mask.any():
            continue
        r_true = true_labels[mask]
        r_pred = preds_arr[mask]
        r_n = int(mask.sum())
        r_k = int(np.sum(r_pred == r_true))
        r_acc = r_k / r_n if r_n > 0 else 0
        r_ci_lo, r_ci_hi = wilson_ci(r_k, r_n)
        per_regime[regime] = {
            "n_items": r_n,
            "n_correct": r_k,
            "accuracy": round(r_acc, 5),
            "ci_lower": round(r_ci_lo, 5),
            "ci_upper": round(r_ci_hi, 5),
        }

    result["per_regime"] = per_regime
    return result


# ================================================================
# D3: Full candidate-aware TF-IDF (same as existing baseline 6)
# ================================================================

def pred_full_candidate(train_items, test_items) -> np.ndarray:
    """Full candidate-aware TF-IDF word baseline.

    This is the same as pred_tfidf_word in run_leakage_eval.py,
    included here for apples-to-apples comparison with D1 and D2.
    """
    builder = _FullCandidateFeatureBuilder()
    return _candidate_predict_diagnostic(train_items, test_items, builder)


class _FullCandidateFeatureBuilder:
    """TF-IDF features from TARGET-normalized hypothesis + context."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=200, analyzer='word', ngram_range=(1, 2),
            stop_words='english', dtype=np.float32,
        )
        self._fitted = False

    def __call__(self, items, is_train=False):
        texts, labels, item_ids, valid_mask = _prepare_candidate_rows(items)

        if is_train:
            X = self.vec.fit_transform(texts)
            self._fitted = True
        else:
            X = self.vec.transform(texts)

        return X, labels, item_ids, valid_mask


# ================================================================
# Shared candidate-aware predict pipeline
# ================================================================

def _candidate_predict_diagnostic(train_items, test_items, build_features_fn):
    """Generic candidate-aware train/predict pipeline.

    Same logic as _candidate_predict in run_leakage_eval.py.
    """
    tr_X, tr_labels, tr_item_ids, tr_valid = build_features_fn(
        train_items, is_train=True
    )
    te_X, te_labels, te_item_ids, te_valid = build_features_fn(
        test_items, is_train=False
    )

    tr_row_valid = np.array([tr_valid[iid] for iid in tr_item_ids])
    tr_X_v = tr_X[tr_row_valid]
    tr_y_v = tr_labels[tr_row_valid]

    if len(set(tr_y_v.tolist())) < 2:
        return np.zeros(len(test_items), dtype=int)

    clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
    clf.fit(tr_X_v, tr_y_v)

    del tr_X, tr_X_v, tr_y_v, tr_labels
    gc.collect()

    col_1 = clf.classes_.tolist().index(1) if 1 in clf.classes_ else -1
    if col_1 >= 0:
        probs = clf.predict_proba(te_X)[:, col_1]
    else:
        probs = np.zeros(te_X.shape[0])
    del clf
    gc.collect()

    full_preds = np.full(len(test_items), -1, dtype=int)
    for item_idx in range(len(test_items)):
        row_mask = te_item_ids == item_idx
        if not row_mask.any():
            continue
        item_probs = probs[row_mask]
        full_preds[item_idx] = int(np.argmax(item_probs))

    return full_preds


DIAGNOSTIC_BASELINE_NAMES = [
    "D1_option_only",
    "D2_context_regime",
    "D3_full_candidate",
]
