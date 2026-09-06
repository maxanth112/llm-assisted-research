#!/usr/bin/env python3
"""
Baselines B1-B7 plus always-abstain and option-length.

Each baseline is a fitted classifier evaluated via leave-one-family-out
cross-validation. Classifiers are TRAINED on the train fold and PREDICT
on the held-out fold. No precomputed predictions are reused.

Target: binary classification ANSWERABLE(1) vs INSUFFICIENT(0).
Only CLEAN and INSUFFICIENT items are used for baseline evaluation
(they form counterfactual pairs sharing the same token multiset).

Correction 3a: CONTEXT-ONLY features strip answer options from feature text.
Correction 4: Per-fold provenance (train IDs, test IDs, train families,
held-out family, feature dim, hyperparameters) saved to machine-readable records.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple


def tokenize_words(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def char_ngrams(text: str, ns=(3, 4, 5)) -> Counter:
    c = Counter()
    t = text.lower()
    for n in ns:
        for i in range(len(t) - n + 1):
            c[t[i:i+n]] += 1
    return c


# ================================================================
# CONTEXT-ONLY TEXT EXTRACTION (Correction 3a)
# ================================================================

def extract_context_only_text(item) -> str:
    """Extract ONLY the context text (preamble + narrative + question +
    evidence statements), REMOVING the answer options entirely.

    This isolates the regime-leakage question: can a classifier predict
    ANSWERABLE vs INSUFFICIENT from CONTEXT ALONE (without seeing options)?
    """
    prompt = item.metadata.get("serialized_prompt", "")
    # Remove option lines: lines starting with (A), (B), (C), (D)
    lines = prompt.split("\n")
    context_lines = [l for l in lines if not re.match(r'^\([A-D]\)\s', l)]
    return "\n".join(context_lines)


# ================================================================
# FEATURE EXTRACTORS (context-only for B1-B7)
# ================================================================

def extract_b1_features(item) -> Counter:
    """B1: Word unigram features from context-only text (no options)."""
    text = extract_context_only_text(item)
    return Counter(tokenize_words(text))


def extract_b2_features(item) -> Counter:
    """B2: Character 3-5-gram features from context-only text."""
    text = extract_context_only_text(item)
    return char_ngrams(text, ns=(3, 4, 5))


def extract_b3_features(item) -> np.ndarray:
    """B3: Token/option-length features from context-only text.
    Since options are removed for context-only evaluation,
    this extracts evidence-level token statistics instead.
    """
    text = extract_context_only_text(item)
    tokens = tokenize_words(text)
    evidence = item.evidence
    ev_lens = [len(tokenize_words(e["content"])) for e in evidence]
    features = [
        len(tokens),               # total context token count
        np.mean(ev_lens) if ev_lens else 0,  # mean evidence length
        np.std(ev_lens) if ev_lens else 0,   # std evidence length
        max(ev_lens) - min(ev_lens) if ev_lens else 0,  # range
    ]
    return np.array(features, dtype=float)


def extract_b4_features(item) -> np.ndarray:
    """B4: Evidence-length features per suspect (context-only)."""
    evidence = item.evidence
    names = sorted(item.metadata.get("name_frequencies", {}).keys())
    features = []
    for name in names:
        lens = [len(e["content"]) for e in evidence if name in e["content"]]
        features.append(sum(lens) / len(lens) if lens else 0)
        features.append(max(lens) if lens else 0)
        features.append(len(lens))
    while len(features) < 9:
        features.append(0)
    return np.array(features[:9], dtype=float)


def extract_b5_features(item) -> np.ndarray:
    """B5: Name-frequency and co-occurrence features (context-only)."""
    evidence = item.evidence
    names = sorted(item.metadata.get("name_frequencies", {}).keys())
    freqs = []
    for name in names:
        count = sum(1 for e in evidence if name in e["content"])
        freqs.append(count)
    cooc = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = sum(1 for e in evidence
                    if names[i] in e["content"] and names[j] in e["content"])
            cooc.append(c)
    features = freqs + cooc
    while len(features) < 6:
        features.append(0)
    return np.array(features[:6], dtype=float)


POLARITY_POS = {"confirmed", "verified", "approved", "authorized", "valid",
                "successful", "positive", "cleared", "compliant", "normal"}
POLARITY_NEG = {"flagged", "suspicious", "unauthorized", "violation", "breach",
                "deviation", "anomaly", "discrepancy", "irregular", "contaminated"}


def extract_b6_features(item) -> np.ndarray:
    """B6: Polarity/sentiment features from context-only text."""
    text = extract_context_only_text(item)
    words = set(tokenize_words(text))
    # Per-evidence polarity
    ev_pos = []
    ev_neg = []
    for e in item.evidence:
        ew = set(tokenize_words(e["content"]))
        ev_pos.append(len(ew & POLARITY_POS))
        ev_neg.append(len(ew & POLARITY_NEG))
    features = [
        len(words & POLARITY_POS),
        len(words & POLARITY_NEG),
        sum(ev_pos),
        sum(ev_neg),
        np.std(ev_pos) if ev_pos else 0,
        np.std(ev_neg) if ev_neg else 0,
    ]
    return np.array(features, dtype=float)


# ================================================================
# VECTORIZATION (Counter -> numpy array with shared vocab)
# ================================================================

def vectorize_counters(train_counters, test_counters, max_features=500):
    """Convert Counter features to numpy arrays using shared vocabulary.
    Build vocab from train, apply to both train and test.
    Returns (train_matrix, test_matrix, vocab_size).
    """
    total = Counter()
    for c in train_counters:
        total.update(c)
    vocab = [w for w, _ in total.most_common(max_features)]
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    def to_array(counter):
        arr = np.zeros(len(vocab))
        for w, cnt in counter.items():
            if w in vocab_idx:
                arr[vocab_idx[w]] = cnt
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    train_mat = np.array([to_array(c) for c in train_counters])
    test_mat = np.array([to_array(c) for c in test_counters])
    return train_mat, test_mat, len(vocab)


# ================================================================
# CLASSIFIER (sklearn-free ridge regression)
# ================================================================

def fit_and_predict_ridge(X_train, y_train, X_test, alpha=1.0):
    """Ridge regression classifier (closed-form, no sklearn needed).
    Returns (predicted_probs, coef_vector, n_features).
    """
    n, d = X_train.shape
    if d == 0 or n == 0:
        return np.full(len(X_test), 0.5), np.zeros(0), 0
    X_tr = np.hstack([X_train, np.ones((n, 1))])
    X_te = np.hstack([X_test, np.ones((len(X_test), 1))])
    try:
        XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
        Xty = X_tr.T @ y_train
        w = np.linalg.solve(XtX, Xty)
        preds = X_te @ w
        preds = np.clip(preds, 0, 1)
    except np.linalg.LinAlgError:
        w = np.zeros(X_tr.shape[1])
        preds = np.full(len(X_test), 0.5)
    return preds, w, d


# ================================================================
# LEAVE-ONE-FAMILY-OUT EVALUATION (with Correction 4 provenance)
# ================================================================

def lofo_evaluate(items, labels, families, extract_fn, baseline_name,
                  use_counter_vectorize=False, alpha=1.0, max_features=500):
    """
    Leave-one-family-out evaluation with full provenance recording.

    Returns list of dicts with per-item predictions AND fold provenance.
    """
    unique_families = sorted(set(families))
    all_predictions = []
    fold_provenance = []

    for held_out_fam in unique_families:
        train_mask = [f != held_out_fam for f in families]
        test_mask = [f == held_out_fam for f in families]

        train_items = [it for it, m in zip(items, train_mask) if m]
        test_items = [it for it, m in zip(items, test_mask) if m]
        y_train = np.array([l for l, m in zip(labels, train_mask) if m], dtype=float)
        y_test = np.array([l for l, m in zip(labels, test_mask) if m], dtype=float)
        train_families = sorted(set(f for f, m in zip(families, train_mask) if m))
        train_ids = [it.id for it in train_items]
        test_ids = [it.id for it in test_items]

        if len(train_items) == 0 or len(test_items) == 0:
            continue

        if use_counter_vectorize:
            train_counters = [extract_fn(it) for it in train_items]
            test_counters = [extract_fn(it) for it in test_items]
            X_train, X_test, vocab_size = vectorize_counters(
                train_counters, test_counters, max_features=max_features)
            feat_dim = vocab_size
        else:
            X_train = np.array([extract_fn(it) for it in train_items])
            X_test = np.array([extract_fn(it) for it in test_items])
            feat_dim = X_train.shape[1] if X_train.ndim > 1 else 1

        preds, coef, n_features = fit_and_predict_ridge(
            X_train, y_train, X_test, alpha=alpha)

        # Record fold provenance (Correction 4)
        fold_prov = {
            "baseline": baseline_name,
            "held_out_family": held_out_fam,
            "train_families": train_families,
            "train_item_ids": train_ids,
            "test_item_ids": test_ids,
            "n_train": len(train_items),
            "n_test": len(test_items),
            "feature_dim": feat_dim,
            "n_features_fitted": n_features,
            "model_type": "ridge_regression",
            "hyperparameters": {"alpha": alpha,
                                "max_features": max_features if use_counter_vectorize else None},
            "coef_norm": float(np.linalg.norm(coef)) if len(coef) > 0 else 0.0,
            "coef_n_nonzero": int(np.sum(np.abs(coef) > 1e-10)) if len(coef) > 0 else 0,
            "prediction_variance": float(np.var(preds)) if len(preds) > 1 else 0.0,
        }
        fold_provenance.append(fold_prov)

        for it, pred, true in zip(test_items, preds, y_test):
            all_predictions.append({
                "item_id": it.id,
                "regime": it.regime,
                "family": it.metadata.get("template", "unknown"),
                "held_out_family": held_out_fam,
                "baseline": baseline_name,
                "true_label": int(true),
                "predicted_prob": float(pred),
                "predicted_label": int(pred >= 0.5),
                "train_families": train_families,
                "train_item_ids": train_ids,
            })

    return all_predictions, fold_provenance


def compute_balanced_accuracy(predictions):
    """Compute balanced accuracy from a list of prediction dicts."""
    if not predictions:
        return 0.5
    true = np.array([p["true_label"] for p in predictions])
    pred = np.array([p["predicted_label"] for p in predictions])
    pos_mask = true == 1
    neg_mask = true == 0
    tpr = np.mean(pred[pos_mask] == 1) if np.any(pos_mask) else 0.5
    tnr = np.mean(pred[neg_mask] == 0) if np.any(neg_mask) else 0.5
    return 0.5 * (tpr + tnr)


def orientation_invariant_separability(predictions):
    """Compute S = max(BA, 1-BA) = 0.5 + abs(BA - 0.5).
    This is orientation-invariant: perfect prediction in EITHER direction
    scores 1.0. Chance = 0.5. A raw BA of 0.0 (perfect reversal) gives S=1.0.
    """
    ba = compute_balanced_accuracy(predictions)
    return 0.5 + abs(ba - 0.5)


# ================================================================
# SPECIAL BASELINES
# ================================================================

def always_abstain_evaluate(items, labels, families):
    """Always-abstain baseline: always predict INSUFFICIENT (label=0)."""
    predictions = []
    for it, label, fam in zip(items, labels, families):
        predictions.append({
            "item_id": it.id,
            "regime": it.regime,
            "family": it.metadata.get("template", "unknown"),
            "held_out_family": "N/A",
            "baseline": "always_abstain",
            "true_label": int(label),
            "predicted_prob": 0.0,
            "predicted_label": 0,
        })
    return predictions, []


def option_length_evaluate(items, labels, families):
    """Option-length baseline: predict the longest option as gold.
    Maps to ANSWERABLE(1) if longest is a suspect hypothesis,
    INSUFFICIENT(0) if longest is abstention.
    """
    predictions = []
    for it, label, fam in zip(items, labels, families):
        hyp_lens = [len(h) for h in it.hypotheses]
        longest_idx = int(np.argmax(hyp_lens))
        longest_hyp = it.hypotheses[longest_idx]
        is_abstention = "no listed suspect" in longest_hyp
        pred_label = 0 if is_abstention else 1
        predictions.append({
            "item_id": it.id,
            "regime": it.regime,
            "family": it.metadata.get("template", "unknown"),
            "held_out_family": "N/A",
            "baseline": "option_length",
            "true_label": int(label),
            "predicted_prob": 0.0 if is_abstention else 1.0,
            "predicted_label": pred_label,
        })
    return predictions, []


# ================================================================
# ANSWER-CHOICE BASELINES (4-class, all regimes, chance=0.25)
# ================================================================

def answer_choice_always_abstain(items):
    """Always predict the abstention option (contains 'no listed suspect').
    Returns list of per-item prediction dicts.
    """
    predictions = []
    for item in items:
        gold_idx = item.metadata["gold_position"]
        # Find the abstention option
        abstain_idx = next(
            (i for i, h in enumerate(item.hypotheses)
             if "no listed suspect" in h), 0)
        predictions.append({
            "item_id": item.id,
            "regime": item.regime,
            "family": item.metadata.get("template", "unknown"),
            "gold_option_index": gold_idx,
            "gold_option_text": item.hypotheses[gold_idx][:60],
            "predicted_option_index": abstain_idx,
            "predicted_option_text": item.hypotheses[abstain_idx][:60],
            "correct": abstain_idx == gold_idx,
            "baseline": "ac_always_abstain",
        })
    return predictions


def answer_choice_longest_option(items):
    """Tie-aware expected-accuracy baseline for the longest option.

    For each item, identifies the set of options that achieve the maximum
    character length (the "tied candidate set").  Expected credit =
    1/|tied_set| if the gold option is in the tied set, else 0.

    With length-matched construction, all 4 options have equal char length,
    so the tied set is always {0,1,2,3} and expected credit is always 0.25.

    Persists auditable fields: per-option char lengths, tied_candidate_set,
    expected_credit.
    """
    predictions = []
    for item in items:
        gold_idx = item.metadata["gold_position"]
        char_lens = [len(h) for h in item.hypotheses]
        max_len = max(char_lens)
        tied_set = [i for i, l in enumerate(char_lens) if l == max_len]
        expected_credit = (1.0 / len(tied_set)) if gold_idx in tied_set else 0.0

        predictions.append({
            "item_id": item.id,
            "regime": item.regime,
            "family": item.metadata.get("template", "unknown"),
            "gold_option_index": gold_idx,
            "per_option_char_lengths": char_lens,
            "tied_candidate_set": tied_set,
            "expected_credit": expected_credit,
            "baseline": "ac_longest_option",
        })
    return predictions


def answer_choice_fixed_position(items, fixed_pos):
    """Fixed-position baseline: always picks option at index `fixed_pos`.

    On a balanced 8-item/regime sample with 2/2/2/2 gold distribution,
    each fixed position scores exactly 0.25 = 2/8 in every regime.
    """
    predictions = []
    for item in items:
        gold_idx = item.metadata["gold_position"]
        predictions.append({
            "item_id": item.id,
            "regime": item.regime,
            "family": item.metadata.get("template", "unknown"),
            "gold_option_index": gold_idx,
            "predicted_option_index": fixed_pos,
            "correct": fixed_pos == gold_idx,
            "baseline": f"ac_position_{fixed_pos}",
        })
    return predictions


# ============================================================
# CORPUS-SCALE ANSWER-CHOICE TOLERANCE (ITEM 4)
# ============================================================
# For the eventual >=500/regime development corpus, the per-regime
# answer-choice accuracy threshold should be near the 4-choice
# chance level (0.25), NOT 0.50.
#
# This constant is a documented config value.  It is NOT used for
# the current 8-item/regime illustrative gates (those use exact
# deterministic checks instead).  It will be activated when the
# corpus grows to statistically-meaningful size.
#
# Value: one-sided 95% upper confidence bound at N=500,
# chance=0.25 ≈ 0.25 + 1.645 * sqrt(0.25*0.75/500) ≈ 0.282.
# We round up slightly to 0.30 for margin.
#
# 0.50 is explicitly NOT the eventual acceptance threshold — it
# was an interim placeholder that is too permissive for a 4-class
# task at chance=0.25.
CORPUS_SCALE_AC_TOLERANCE = 0.30


def run_answer_choice_baselines(items):
    """Run all answer-choice baselines across ALL items (all 4 regimes).

    Returns dict: {baseline_name: predictions_list}.

    Baselines:
      - ac_always_abstain: always picks the abstention option
      - ac_longest_option: tie-aware expected accuracy on longest option
      - ac_position_0..3: fixed-position picks for construction-balance check
    """
    results = {
        "ac_always_abstain": answer_choice_always_abstain(items),
        "ac_longest_option": answer_choice_longest_option(items),
    }
    for pos in range(4):
        results[f"ac_position_{pos}"] = answer_choice_fixed_position(items, pos)
    return results


def _prediction_credit(p):
    """Return the credit for a single prediction.

    Supports two schemas:
      - Deterministic (ac_always_abstain, ac_position_*): has "correct" bool
      - Tie-aware (ac_longest_option): has "expected_credit" float
    """
    if "expected_credit" in p:
        return p["expected_credit"]
    return 1.0 if p["correct"] else 0.0


def compute_answer_choice_accuracy(predictions, by_regime=False):
    """Compute overall and per-regime accuracy for answer-choice baselines.

    Handles both deterministic baselines (with "correct" field) and
    tie-aware baselines (with "expected_credit" field).

    Returns dict with 'overall' and per-regime keys.
    """
    if not predictions:
        return {"overall": 0.0}

    total_credit = sum(_prediction_credit(p) for p in predictions)
    result = {"overall": total_credit / len(predictions), "n_total": len(predictions)}

    if by_regime:
        regimes = sorted(set(p["regime"] for p in predictions))
        for regime in regimes:
            regime_preds = [p for p in predictions if p["regime"] == regime]
            regime_credit = sum(_prediction_credit(p) for p in regime_preds)
            n_total = len(regime_preds)
            result[regime] = {
                "accuracy": regime_credit / n_total if n_total > 0 else 0.0,
                "n_correct": regime_credit,
                "n_total": n_total,
            }
    return result


# ================================================================
# RUN ALL BASELINES
# ================================================================

def run_all_baselines(items, labels, families):
    """
    Run all baselines with genuine LOFO fitting.
    Returns (results_dict, all_fold_provenance).
    results_dict: {baseline_name: predictions_list}
    all_fold_provenance: list of per-fold provenance dicts
    """
    results = {}
    all_provenance = []

    # B1: Word unigram
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b1_features, "B1_word_unigram",
        use_counter_vectorize=True)
    results["B1_word_unigram"] = preds
    all_provenance.extend(prov)

    # B2: Char 3-5-gram
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b2_features, "B2_char_ngram",
        use_counter_vectorize=True)
    results["B2_char_ngram"] = preds
    all_provenance.extend(prov)

    # B3: Token/option-length (context-only)
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b3_features, "B3_token_length")
    results["B3_token_length"] = preds
    all_provenance.extend(prov)

    # B4: Evidence-length
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b4_features, "B4_evidence_length")
    results["B4_evidence_length"] = preds
    all_provenance.extend(prov)

    # B5: Name-frequency
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b5_features, "B5_name_frequency")
    results["B5_name_frequency"] = preds
    all_provenance.extend(prov)

    # B6: Polarity
    preds, prov = lofo_evaluate(
        items, labels, families, extract_b6_features, "B6_polarity")
    results["B6_polarity"] = preds
    all_provenance.extend(prov)

    # B7: Combined (B1+B2 counter + B3-B6 array)
    unique_families = sorted(set(families))
    b7_predictions = []
    for held_out_fam in unique_families:
        train_mask = [f != held_out_fam for f in families]
        test_mask = [f == held_out_fam for f in families]

        train_items = [it for it, m in zip(items, train_mask) if m]
        test_items = [it for it, m in zip(items, test_mask) if m]
        y_train = np.array([l for l, m in zip(labels, train_mask) if m], dtype=float)
        train_families_list = sorted(set(f for f, m in zip(families, train_mask) if m))
        train_ids = [it.id for it in train_items]
        test_ids = [it.id for it in test_items]

        if not train_items or not test_items:
            continue

        b1_train = [extract_b1_features(it) for it in train_items]
        b1_test = [extract_b1_features(it) for it in test_items]
        b1_tr_mat, b1_te_mat, b1_vs = vectorize_counters(b1_train, b1_test, max_features=200)

        b2_train = [extract_b2_features(it) for it in train_items]
        b2_test = [extract_b2_features(it) for it in test_items]
        b2_tr_mat, b2_te_mat, b2_vs = vectorize_counters(b2_train, b2_test, max_features=200)

        def _b7_arr(it):
            return np.concatenate([
                extract_b3_features(it), extract_b4_features(it),
                extract_b5_features(it), extract_b6_features(it)])

        arr_train = np.array([_b7_arr(it) for it in train_items])
        arr_test = np.array([_b7_arr(it) for it in test_items])

        X_train = np.hstack([b1_tr_mat, b2_tr_mat, arr_train])
        X_test = np.hstack([b1_te_mat, b2_te_mat, arr_test])

        preds, coef, n_feat = fit_and_predict_ridge(
            X_train, y_train, X_test, alpha=10.0)

        fold_prov = {
            "baseline": "B7_combined",
            "held_out_family": held_out_fam,
            "train_families": train_families_list,
            "train_item_ids": train_ids,
            "test_item_ids": test_ids,
            "n_train": len(train_items),
            "n_test": len(test_items),
            "feature_dim": X_train.shape[1],
            "n_features_fitted": n_feat,
            "model_type": "ridge_regression",
            "hyperparameters": {"alpha": 10.0, "b1_max_features": 200, "b2_max_features": 200},
            "coef_norm": float(np.linalg.norm(coef)) if len(coef) > 0 else 0.0,
            "coef_n_nonzero": int(np.sum(np.abs(coef) > 1e-10)) if len(coef) > 0 else 0,
            "prediction_variance": float(np.var(preds)) if len(preds) > 1 else 0.0,
        }
        all_provenance.append(fold_prov)

        y_test = np.array([l for l, m in zip(labels, test_mask) if m], dtype=float)
        for it, pred, true in zip(test_items, preds, y_test):
            b7_predictions.append({
                "item_id": it.id,
                "regime": it.regime,
                "family": it.metadata.get("template", "unknown"),
                "held_out_family": held_out_fam,
                "baseline": "B7_combined",
                "true_label": int(true),
                "predicted_prob": float(pred),
                "predicted_label": int(pred >= 0.5),
                "train_families": train_families_list,
                "train_item_ids": train_ids,
            })

    results["B7_combined"] = b7_predictions

    # Always-abstain
    preds, prov = always_abstain_evaluate(items, labels, families)
    results["always_abstain"] = preds
    all_provenance.extend(prov)

    # Option-length
    preds, prov = option_length_evaluate(items, labels, families)
    results["option_length"] = preds
    all_provenance.extend(prov)

    return results, all_provenance
