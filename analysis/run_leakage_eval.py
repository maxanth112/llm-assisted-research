#!/usr/bin/env python3
"""
Candidate-aware, permutation-equivariant leakage evaluator.

Phase A.1 rewrite.  All classifier baselines (6-11) now operate on ONE ROW
PER ITEM-CANDIDATE PAIR, with "target-vs-other" feature normalization.
Permutation equivariance is guaranteed: jointly permuting option order and
gold pointer produces the same selected candidate.

Fixes over Phase-A evaluator:
  * Classifier baselines predicted gold OPTION INDEX from narrative+evidence
    text that omitted hypothesis text entirely.  They tested text→position
    correlation, not whether surface cues identify the correct candidate.
  * "Polarity" baseline was mislabeled (actual polarity features were removed
    in Phase A but name was kept).  Renamed to "mention_evidence".
  * "Positional" baseline fed near-constant position indices {0,1,2,3}.
    Replaced with "first_mention_order" using genuine first-occurrence
    positions in text.

Preserved from Phase-A evaluator:
  * Heuristic baselines (1-5): already candidate-aware (argmax over
    per-candidate scores).
  * wilson_ci, chance_level_correct, _decompose, template_held_out_eval
    structure, final_audit_eval structure.
  * Feature hygiene: ONLY model-visible fields (narrative, question,
    hypotheses, evidence[].content).
  * Hard reconciliation assertion: sum(regime_correct) == agg_correct.
  * Chance = mean(1/n_options).
  * Per-regime verdicts by slicing (not refitting).

Allowed fields per baseline (documented in output):
  ALL baselines may read: narrative, question, hypotheses, evidence[*].content
  NO baseline may read: regime, metadata.*, gold_answer, gold_reasoning,
      evidence[*].supports, evidence[*].contradicts, evidence[*].diagnostic_value,
      source_precedence_rule
"""

import gc
import json, math, sys, warnings, time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')
import numpy as np

# ================================================================
# HELPERS
# ================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return (max(0.0, c - m), min(1.0, c + m))


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def chance_level_correct(items):
    """Chance = mean(1/n_options) across items.  Correct formulation."""
    if not items:
        return 0.0
    return float(np.mean([1.0 / len(it["hypotheses"]) for it in items]))


def extract_name(hyp: str) -> str:
    """Extract suspect name from hypothesis string (model-visible text)."""
    for pat in [" is responsible", " committed", " is the perpetrator", " is guilty"]:
        if pat in hyp:
            return hyp.split(pat)[0].strip()
    words = hyp.strip().split()
    return ' '.join(words[:2]) if len(words) >= 2 else hyp.strip()


def ev_content_text(item) -> str:
    """Concatenate evidence content fields (model-visible only)."""
    parts = []
    for ev in item.get("evidence", []):
        if isinstance(ev, dict):
            parts.append(ev.get("content", ""))
        elif isinstance(ev, str):
            parts.append(ev)
    return " ".join(parts)


def ev_content_list(item) -> List[str]:
    """Return list of individual evidence content strings."""
    parts = []
    for ev in item.get("evidence", []):
        if isinstance(ev, dict):
            parts.append(ev.get("content", ""))
        elif isinstance(ev, str):
            parts.append(ev)
    return parts


def gold_index(item) -> int:
    """Return index of gold_answer in hypotheses list, or -1."""
    try:
        return item["hypotheses"].index(item["gold_answer"])
    except ValueError:
        return -1


# ================================================================
# SURFACE-FORM SHORTCUT CHECKS (AMENDMENT-002 §2.5.2)
# ================================================================
# Deterministic structural checks run BEFORE classifier baselines.
# These catch the shortcut patterns (option count, abstention position,
# etc.) that caused v2 INSUFFICIENT failures.

ABSTENTION_PATTERNS = [
    "cannot be determined",
    "cannot be determined from available evidence",
    "insufficient evidence",
    "not enough information",
]

def _is_abstention_option(text: str) -> bool:
    """Check if a hypothesis option is an abstention/insufficiency option."""
    return any(pat in text.lower() for pat in ABSTENTION_PATTERNS)


def run_surface_form_checks(items: List[Dict], label: str = "corpus") -> Dict:
    """Run prohibited surface-form shortcut checks (AMENDMENT-002 §2.5.2).

    Returns dict with per-check results. Each check has:
      - passed: bool
      - details: str
    """
    from scipy import stats as sp_stats

    results = {}
    regimes = sorted(set(it.get("regime", "UNKNOWN") for it in items))

    # ---- S1: Option count (every item has exactly 4 options) ----
    non4 = [(i, len(it["hypotheses"])) for i, it in enumerate(items)
            if len(it["hypotheses"]) != 4]
    results["S1_option_count"] = {
        "passed": len(non4) == 0,
        "details": (f"All {len(items)} items have exactly 4 options"
                    if len(non4) == 0
                    else f"{len(non4)} items have != 4 options: "
                         f"first 5 = {non4[:5]}"),
    }

    # ---- S2: Abstention position uniformity per regime ----
    s2_results = {}
    s2_all_pass = True
    for regime in regimes:
        regime_items = [it for it in items if it.get("regime") == regime]
        positions = []
        for it in regime_items:
            for idx, hyp in enumerate(it["hypotheses"]):
                if _is_abstention_option(hyp):
                    positions.append(idx)
                    break
        if len(positions) >= 20:
            counts = [positions.count(p) for p in range(4)]
            chi2, pval = sp_stats.chisquare(counts)
            passed = pval > 0.05
            s2_results[regime] = {
                "counts": counts, "chi2": round(chi2, 3),
                "p": round(pval, 4), "passed": passed,
            }
            if not passed:
                s2_all_pass = False
        elif len(positions) > 0:
            s2_results[regime] = {
                "n": len(positions),
                "note": "too few items for chi-squared test",
                "passed": True,  # not enough data to fail
            }
    results["S2_abstention_position"] = {
        "passed": s2_all_pass,
        "per_regime": s2_results,
    }

    # ---- S3: Abstention presence (every item has abstention option) ----
    missing = []
    for i, it in enumerate(items):
        has_abs = any(_is_abstention_option(h) for h in it["hypotheses"])
        if not has_abs:
            missing.append(i)
    results["S3_abstention_presence"] = {
        "passed": len(missing) == 0,
        "details": (f"All {len(items)} items contain an abstention option"
                    if len(missing) == 0
                    else f"{len(missing)} items lack abstention option: "
                         f"first 5 indices = {missing[:5]}"),
    }

    # ---- S4: Evidence count distribution per regime pair ----
    s4_results = {}
    s4_all_pass = True
    ev_counts_by_regime = {}
    for regime in regimes:
        ev_counts_by_regime[regime] = [
            len(it.get("evidence", [])) for it in items
            if it.get("regime") == regime
        ]
    for i, r1 in enumerate(regimes):
        for r2 in regimes[i+1:]:
            if len(ev_counts_by_regime[r1]) >= 5 and len(ev_counts_by_regime[r2]) >= 5:
                stat, pval = sp_stats.ks_2samp(
                    ev_counts_by_regime[r1], ev_counts_by_regime[r2]
                )
                passed = pval > 0.05
                s4_results[f"{r1}_vs_{r2}"] = {
                    "ks_stat": round(stat, 4), "p": round(pval, 4),
                    "passed": passed,
                }
                if not passed:
                    s4_all_pass = False
    results["S4_evidence_count"] = {
        "passed": s4_all_pass,
        "regime_pairs": s4_results,
    }

    # ---- S5: Option text length distribution per regime pair ----
    s5_results = {}
    s5_all_pass = True
    opt_len_by_regime = {}
    for regime in regimes:
        regime_items = [it for it in items if it.get("regime") == regime]
        opt_len_by_regime[regime] = [
            float(np.mean([len(h) for h in it["hypotheses"]]))
            for it in regime_items
        ]
    for i, r1 in enumerate(regimes):
        for r2 in regimes[i+1:]:
            if len(opt_len_by_regime[r1]) >= 5 and len(opt_len_by_regime[r2]) >= 5:
                stat, pval = sp_stats.ks_2samp(
                    opt_len_by_regime[r1], opt_len_by_regime[r2]
                )
                passed = pval > 0.05
                s5_results[f"{r1}_vs_{r2}"] = {
                    "ks_stat": round(stat, 4), "p": round(pval, 4),
                    "passed": passed,
                }
                if not passed:
                    s5_all_pass = False
    results["S5_option_text_length"] = {
        "passed": s5_all_pass,
        "regime_pairs": s5_results,
    }

    # ---- S6: Gold-answer position uniformity per regime ----
    s6_results = {}
    s6_all_pass = True
    for regime in regimes:
        regime_items = [it for it in items if it.get("regime") == regime]
        gold_positions = [gold_index(it) for it in regime_items]
        gold_positions = [p for p in gold_positions if p >= 0]
        if len(gold_positions) >= 20:
            counts = [gold_positions.count(p) for p in range(4)]
            chi2, pval = sp_stats.chisquare(counts)
            passed = pval > 0.05
            s6_results[regime] = {
                "counts": counts, "chi2": round(chi2, 3),
                "p": round(pval, 4), "passed": passed,
            }
            if not passed:
                s6_all_pass = False
        elif len(gold_positions) > 0:
            s6_results[regime] = {
                "n": len(gold_positions),
                "note": "too few items for chi-squared test",
                "passed": True,
            }
    results["S6_gold_position"] = {
        "passed": s6_all_pass,
        "per_regime": s6_results,
    }

    return results


# ================================================================
# HEURISTIC BASELINES (per-item, no training)
# ================================================================
# Each returns a prediction array (one int per item = predicted hyp index).
# These are already candidate-aware: they compute per-candidate scores
# and pick argmax.

def pred_majority(items, majority_label: int) -> np.ndarray:
    """Predict the same majority_label for every item."""
    return np.full(len(items), majority_label, dtype=int)


def pred_position(items) -> np.ndarray:
    """Always predict index 0 (first hypothesis)."""
    return np.zeros(len(items), dtype=int)


def pred_mention_count(items) -> np.ndarray:
    """Predict the hypothesis whose extracted name appears most often
    in narrative + evidence content."""
    preds = []
    for it in items:
        text = (it.get("narrative", "") + " " + ev_content_text(it)).lower()
        hyps = it["hypotheses"]
        counts = []
        for h in hyps:
            n = extract_name(h).lower()
            counts.append(text.count(n))
        preds.append(int(np.argmax(counts)))
    return np.array(preds, dtype=int)


def pred_evidence_count(items) -> np.ndarray:
    """Predict hypothesis whose extracted name appears in the most
    evidence content strings."""
    preds = []
    for it in items:
        hyps = it["hypotheses"]
        ev_contents = ev_content_list(it)
        counts = []
        for h in hyps:
            n = extract_name(h).lower()
            counts.append(sum(1 for et in ev_contents if n in et.lower()))
        preds.append(int(np.argmax(counts)))
    return np.array(preds, dtype=int)


def pred_lexical_overlap(items) -> np.ndarray:
    """Predict hypothesis with highest word-level overlap with evidence."""
    preds = []
    for it in items:
        ew = set(ev_content_text(it).lower().split())
        hyps = it["hypotheses"]
        scores = []
        for h in hyps:
            hw = set(h.lower().split())
            scores.append(len(ew & hw))
        preds.append(int(np.argmax(scores)))
    return np.array(preds, dtype=int)


# ================================================================
# CANDIDATE-AWARE CLASSIFIER BASELINES (train/test split required)
# ================================================================
#
# Design: ONE ROW PER ITEM-CANDIDATE PAIR.
#
# For each item with K hypotheses, emit K rows.  Each row scores one
# candidate hypothesis against the item's context (narrative + evidence).
# Features use "target-vs-other" normalization: the target candidate's
# features, MINUS the mean of other candidates' features.
#
# Train a binary classifier: label=1 for the gold candidate, label=0
# for all others.  At test time, select the candidate with the highest
# predicted probability of label=1 (argmax within each item).
#
# PERMUTATION EQUIVARIANCE: because every candidate gets the same feature
# treatment (same function applied, no position index in features), and
# the final prediction is argmax of scores, permuting the option order
# and gold pointer jointly does NOT change which candidate is selected.
#
# Template-split discipline: all rows from the same item stay in the
# same fold.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse


def _candidate_texts(item) -> List[str]:
    """For each candidate in an item, produce a combined text string:
    candidate_hypothesis [SEP] narrative [SEP] evidence_content

    This ensures the classifier sees the hypothesis text alongside the
    context, making it candidate-aware."""
    narrative = item.get("narrative", "")
    ev_text = ev_content_text(item)
    texts = []
    for h in item["hypotheses"]:
        texts.append(f"{h} [SEP] {narrative} [SEP] {ev_text}")
    return texts


def _prepare_candidate_rows(items):
    """Expand items into candidate-level rows.

    Returns:
        texts: list of combined texts (one per candidate row)
        labels: binary array (1=gold candidate, 0=other)
        item_ids: array mapping each row to its source item index
        valid_mask: boolean array (True for items with valid gold_index)
    """
    texts = []
    labels = []
    item_ids = []
    valid_mask = np.ones(len(items), dtype=bool)

    for i, it in enumerate(items):
        gi = gold_index(it)
        if gi < 0:
            valid_mask[i] = False
        for j, h in enumerate(it["hypotheses"]):
            texts.append(f"{h} [SEP] {it.get('narrative', '')} [SEP] {ev_content_text(it)}")
            labels.append(1 if (gi >= 0 and j == gi) else 0)
            item_ids.append(i)

    return texts, np.array(labels), np.array(item_ids), valid_mask


def _candidate_predict(train_items, test_items, build_features_fn):
    """Generic candidate-aware train/predict pipeline.

    Args:
        train_items: training items
        test_items: test items
        build_features_fn: function(items) -> (feature_matrix, item_ids, valid_mask)
            Returns feature matrix at candidate-row level, item index per row,
            and per-item validity mask.

    Returns:
        np.ndarray of predicted hypothesis indices for each test item.
    """
    tr_X, tr_labels, tr_item_ids, tr_valid = build_features_fn(train_items, is_train=True)
    te_X, te_labels, te_item_ids, te_valid = build_features_fn(test_items, is_train=False)

    # Filter to valid training rows
    tr_row_valid = np.array([tr_valid[iid] for iid in tr_item_ids])
    tr_X_v = tr_X[tr_row_valid]
    tr_y_v = tr_labels[tr_row_valid]

    if len(set(tr_y_v.tolist())) < 2:
        # Degenerate: predict 0 for everything
        return np.zeros(len(test_items), dtype=int)

    clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
    clf.fit(tr_X_v, tr_y_v)

    # Free training data
    del tr_X, tr_X_v, tr_y_v, tr_labels
    gc.collect()

    # Predict probabilities for label=1 on test rows
    col_1 = clf.classes_.tolist().index(1) if 1 in clf.classes_ else -1
    if col_1 >= 0:
        probs = clf.predict_proba(te_X)[:, col_1]
    else:
        probs = np.zeros(te_X.shape[0])
    del clf
    gc.collect()

    # Argmax within each item
    full_preds = np.full(len(test_items), -1, dtype=int)
    for item_idx in range(len(test_items)):
        row_mask = te_item_ids == item_idx
        if not row_mask.any():
            continue
        item_probs = probs[row_mask]
        # Candidate index within this item's hypotheses
        full_preds[item_idx] = int(np.argmax(item_probs))

    return full_preds


class _TfidfFeatureBuilder:
    """Stateful feature builder for TF-IDF candidate-aware baselines.

    On training call: fits vectorizer, transforms training rows.
    On test call: transforms test rows using fitted vectorizer.
    """
    def __init__(self, analyzer='word', ngram_range=(1,2), max_features=200):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            analyzer=analyzer,
            ngram_range=ngram_range,
            stop_words='english' if analyzer == 'word' else None,
            dtype=np.float32,
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


def pred_tfidf_word(train_items, test_items) -> np.ndarray:
    builder = _TfidfFeatureBuilder(analyzer='word', ngram_range=(1,2), max_features=200)
    return _candidate_predict(train_items, test_items, builder)


def pred_tfidf_char(train_items, test_items) -> np.ndarray:
    builder = _TfidfFeatureBuilder(analyzer='char_wb', ngram_range=(2,4), max_features=200)
    return _candidate_predict(train_items, test_items, builder)


# ================================================================
# STRUCTURED CANDIDATE-AWARE FEATURES
# ================================================================
#
# For each item-candidate pair, compute features about the TARGET candidate
# and contrast features (target vs mean-of-others).
#
# Feature vector per candidate row:
#   [target_mention_count, target_evidence_count, target_length_sum,
#    target_first_mention_pos,
#    delta_mention_count, delta_evidence_count, delta_length_sum,
#    delta_first_mention_pos]
#
# Where delta_X = target_X - mean(other_candidates_X).
# This is permutation-equivariant: the same function is applied to each
# candidate identically, with no position index features.

def _compute_candidate_features(item, candidate_idx):
    """Compute structured features for one candidate in an item.

    Returns dict of raw feature values for this candidate.
    """
    hyps = item["hypotheses"]
    h = hyps[candidate_idx]
    name = extract_name(h).lower()

    narrative = item.get("narrative", "").lower()
    ev_contents = ev_content_list(item)
    full_text = narrative + " " + " ".join(ev_contents)
    full_text_lower = full_text.lower()

    # Mention count: how often this candidate's name appears in full text
    mention_count = full_text_lower.count(name) if name else 0

    # Evidence count: how many evidence items mention this candidate
    evidence_count = sum(1 for et in ev_contents if name in et.lower()) if name else 0

    # Length sum: total character length of evidence items mentioning this candidate
    length_sum = sum(len(et) for et in ev_contents if name in et.lower()) if name else 0

    # First mention position: character position of first occurrence in full text
    # (normalized by text length; 1.0 if not found)
    first_pos = full_text_lower.find(name) if name else -1
    text_len = max(len(full_text_lower), 1)
    first_mention_pos = first_pos / text_len if first_pos >= 0 else 1.0

    return {
        'mention_count': mention_count,
        'evidence_count': evidence_count,
        'length_sum': length_sum,
        'first_mention_pos': first_mention_pos,
    }


def _build_structured_candidate_rows(items, is_train=False):
    """Build structured features at candidate-row level with target-vs-other
    normalization.

    Returns: (X, labels, item_ids, valid_mask)
    """
    feat_names = ['mention_count', 'evidence_count', 'length_sum', 'first_mention_pos']
    n_raw = len(feat_names)
    n_features = n_raw * 2  # target + delta

    X_rows = []
    labels = []
    item_ids = []
    valid_mask = np.ones(len(items), dtype=bool)

    for i, it in enumerate(items):
        gi = gold_index(it)
        if gi < 0:
            valid_mask[i] = False

        hyps = it["hypotheses"]
        n_hyps = len(hyps)

        # Compute raw features for ALL candidates in this item
        all_feats = []
        for j in range(n_hyps):
            all_feats.append(_compute_candidate_features(it, j))

        # For each candidate, compute target features and delta (target - mean_others)
        for j in range(n_hyps):
            target = all_feats[j]
            others = [all_feats[k] for k in range(n_hyps) if k != j]

            row = []
            for fn in feat_names:
                t_val = target[fn]
                o_mean = np.mean([o[fn] for o in others]) if others else 0.0
                row.append(t_val)
                row.append(t_val - o_mean)

            X_rows.append(row)
            labels.append(1 if (gi >= 0 and j == gi) else 0)
            item_ids.append(i)

    X = np.array(X_rows) if X_rows else np.zeros((0, n_features))
    return X, np.array(labels), np.array(item_ids), valid_mask


def _structured_predict(train_items, test_items, col_selector=None):
    """Train/predict with structured candidate-aware features."""
    def build_fn(items, is_train=False):
        X, labels, item_ids, valid_mask = _build_structured_candidate_rows(items)
        if col_selector is not None:
            cols = [c for c in col_selector if c < X.shape[1]]
            X = X[:, cols]
        return X, labels, item_ids, valid_mask

    return _candidate_predict(train_items, test_items, build_fn)


def pred_length(train_items, test_items) -> np.ndarray:
    """Length-based features only: target_length_sum and delta_length_sum."""
    # Feature indices: length_sum=2 → target at 4, delta at 5
    cols = [4, 5]
    return _structured_predict(train_items, test_items, cols)


def pred_mention_evidence(train_items, test_items) -> np.ndarray:
    """Mention count + evidence count features (renamed from 'polarity').

    Honest name: these are mention-frequency and evidence-presence features,
    not sentiment/polarity features (those were removed in Phase A for
    feature hygiene).
    """
    # Feature indices: mention_count target=0,delta=1; evidence_count target=2,delta=3
    cols = [0, 1, 2, 3]
    return _structured_predict(train_items, test_items, cols)


def pred_first_mention_order(train_items, test_items) -> np.ndarray:
    """First-mention-position features (renamed from 'positional').

    Uses genuine first-occurrence position in text (normalized), not
    the near-constant hypothesis index that the old baseline used.
    """
    # Feature indices: first_mention_pos target=6, delta=7
    cols = [6, 7]
    return _structured_predict(train_items, test_items, cols)


def pred_combined(train_items, test_items) -> np.ndarray:
    """All structured features combined."""
    return _structured_predict(train_items, test_items, col_selector=None)


# ================================================================
# EVALUATION FRAMEWORK
# ================================================================

BASELINE_NAMES = [
    "1_majority_class",
    "2_label_position",
    "3_mention_count",
    "4_evidence_count",
    "5_lexical_overlap",
    "6_tfidf_word",
    "7_tfidf_char",
    "8_length_feature",
    "9_mention_evidence",
    "10_first_mention_order",
    "11_combined_shallow",
]

# Fields each baseline is allowed to read (for documentation)
BASELINE_ALLOWED_FIELDS = {
    "1_majority_class": ["hypotheses (count only)"],
    "2_label_position": ["hypotheses"],
    "3_mention_count": ["narrative", "hypotheses", "evidence[].content"],
    "4_evidence_count": ["hypotheses", "evidence[].content"],
    "5_lexical_overlap": ["hypotheses", "evidence[].content"],
    "6_tfidf_word": ["narrative", "hypotheses", "evidence[].content"],
    "7_tfidf_char": ["narrative", "hypotheses", "evidence[].content"],
    "8_length_feature": ["narrative", "hypotheses", "evidence[].content"],
    "9_mention_evidence": ["narrative", "hypotheses", "evidence[].content"],
    "10_first_mention_order": ["narrative", "hypotheses", "evidence[].content"],
    "11_combined_shallow": ["narrative", "hypotheses", "evidence[].content"],
}


def _compute_gold_array(items) -> np.ndarray:
    """Gold-index array for items."""
    return np.array([gold_index(it) for it in items])


def _decompose(preds: np.ndarray, golds: np.ndarray, items: list,
               chance_agg: float, alpha: float):
    """Compute aggregate + per-regime stats from one prediction array.
    Returns dict with aggregate and per_regime, with reconciliation enforced."""
    assert len(preds) == len(golds) == len(items)
    valid = golds >= 0
    correct = (preds == golds) & valid
    agg_k = int(correct.sum())
    agg_n = int(valid.sum())

    ci_lo, ci_hi = wilson_ci(agg_k, agg_n)
    threshold_agg = chance_agg + alpha

    result = {
        "n_correct": agg_k,
        "n_items": agg_n,
        "accuracy": round(agg_k / agg_n, 5) if agg_n else 0,
        "ci_lower": round(ci_lo, 5),
        "ci_upper": round(ci_hi, 5),
        "chance": round(chance_agg, 5),
        "threshold": round(threshold_agg, 5),
        "verdict": "FAIL" if ci_hi > threshold_agg else "PASS",
    }

    per_regime = {}
    regime_correct_sum = 0
    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
        mask = np.array([it["regime"] == regime for it in items])
        r_valid = valid & mask
        r_correct = correct & mask
        rk = int(r_correct.sum())
        rn = int(r_valid.sum())
        regime_correct_sum += rk

        if rn == 0:
            continue

        # Per-regime chance
        regime_items = [it for i, it in enumerate(items) if mask[i] and valid[i]]
        r_chance = chance_level_correct(regime_items)
        r_threshold = r_chance + alpha
        r_ci_lo, r_ci_hi = wilson_ci(rk, rn)

        per_regime[regime] = {
            "n_correct": rk,
            "n_items": rn,
            "accuracy": round(rk / rn, 5) if rn else 0,
            "ci_lower": round(r_ci_lo, 5),
            "ci_upper": round(r_ci_hi, 5),
            "chance": round(r_chance, 5),
            "threshold": round(r_threshold, 5),
            "verdict": "FAIL" if r_ci_hi > r_threshold else "PASS",
        }

    # HARD RECONCILIATION ASSERTION
    assert regime_correct_sum == agg_k, (
        f"Reconciliation failure: sum(regime_correct)={regime_correct_sum} "
        f"!= aggregate_correct={agg_k}"
    )

    result["per_regime"] = per_regime
    return result


def template_held_out_eval(items, alpha=0.05):
    """Leave-one-template-family-out evaluation.

    For EACH fold:
      - Heuristic baselines: compute per-item predictions on test items
      - Classifier baselines: train on train items, predict on test items
    Accumulate per-item predictions across all folds → ONE prediction array
    per baseline → decompose to aggregate + per-regime by SLICING.
    """
    by_template = defaultdict(list)
    for it in items:
        t = it.get("metadata", {}).get("template", "unknown")
        by_template[t].append(it)
    templates = sorted(by_template.keys())

    # Build item ordering: items grouped by template, track global indices
    ordered_items = []
    fold_boundaries = {}
    offset = 0
    for t in templates:
        fold_boundaries[t] = (offset, offset + len(by_template[t]))
        for it in by_template[t]:
            ordered_items.append(it)
            offset += 1

    n = len(ordered_items)
    golds = _compute_gold_array(ordered_items)
    chance_agg = chance_level_correct(ordered_items)

    # Determine majority label from full training set (mode of gold indices)
    valid_golds = [g for g in golds if g >= 0]
    majority_label = int(Counter(valid_golds).most_common(1)[0][0])

    # Allocate prediction arrays
    all_preds = {name: np.full(n, -1, dtype=int) for name in BASELINE_NAMES}

    for held_out_t in templates:
        lo, hi = fold_boundaries[held_out_t]
        test_items = ordered_items[lo:hi]
        train_items = [ordered_items[i] for i in range(n)
                       if i < lo or i >= hi]

        # --- Heuristic baselines ---
        all_preds["1_majority_class"][lo:hi] = pred_majority(test_items, majority_label)
        all_preds["2_label_position"][lo:hi] = pred_position(test_items)
        all_preds["3_mention_count"][lo:hi] = pred_mention_count(test_items)
        all_preds["4_evidence_count"][lo:hi] = pred_evidence_count(test_items)
        all_preds["5_lexical_overlap"][lo:hi] = pred_lexical_overlap(test_items)

        # --- Candidate-aware classifier baselines ---
        all_preds["6_tfidf_word"][lo:hi] = pred_tfidf_word(train_items, test_items)
        gc.collect()
        all_preds["7_tfidf_char"][lo:hi] = pred_tfidf_char(train_items, test_items)
        gc.collect()
        all_preds["8_length_feature"][lo:hi] = pred_length(train_items, test_items)
        all_preds["9_mention_evidence"][lo:hi] = pred_mention_evidence(train_items, test_items)
        all_preds["10_first_mention_order"][lo:hi] = pred_first_mention_order(train_items, test_items)
        all_preds["11_combined_shallow"][lo:hi] = pred_combined(train_items, test_items)
        gc.collect()

    # Decompose each baseline
    results = {}
    for name in BASELINE_NAMES:
        results[name] = _decompose(all_preds[name], golds, ordered_items,
                                   chance_agg, alpha)
    return results


def final_audit_eval(audit_items, train_items, alpha=0.05):
    """Evaluate all baselines on audit_items. Classifiers trained on train_items.
    Per-regime is decomposition of the SAME predictions (no refitting)."""
    golds = _compute_gold_array(audit_items)
    chance_agg = chance_level_correct(audit_items)

    # Majority label from train set
    train_golds = _compute_gold_array(train_items)
    valid_train = [g for g in train_golds if g >= 0]
    majority_label = int(Counter(valid_train).most_common(1)[0][0])

    all_preds = {}
    all_preds["1_majority_class"] = pred_majority(audit_items, majority_label)
    all_preds["2_label_position"] = pred_position(audit_items)
    all_preds["3_mention_count"] = pred_mention_count(audit_items)
    all_preds["4_evidence_count"] = pred_evidence_count(audit_items)
    all_preds["5_lexical_overlap"] = pred_lexical_overlap(audit_items)
    all_preds["6_tfidf_word"] = pred_tfidf_word(train_items, audit_items)
    all_preds["7_tfidf_char"] = pred_tfidf_char(train_items, audit_items)
    all_preds["8_length_feature"] = pred_length(train_items, audit_items)
    all_preds["9_mention_evidence"] = pred_mention_evidence(train_items, audit_items)
    all_preds["10_first_mention_order"] = pred_first_mention_order(train_items, audit_items)
    all_preds["11_combined_shallow"] = pred_combined(train_items, audit_items)

    results = {}
    for name in BASELINE_NAMES:
        results[name] = _decompose(all_preds[name], golds, audit_items,
                                   chance_agg, alpha)
    return results


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading datasets...", file=sys.stderr)
    train_items = load_jsonl("analysis/t2v2_train.jsonl")
    audit_items = load_jsonl("analysis/t2v2_final_audit.jsonl")
    print(f"  Train: {len(train_items)} items, Audit: {len(audit_items)} items",
          file=sys.stderr)

    alpha = 0.05
    train_chance = chance_level_correct(train_items)
    audit_chance = chance_level_correct(audit_items)
    print(f"  Train chance (mean 1/n_opts): {train_chance:.5f}, "
          f"threshold: {train_chance+alpha:.5f}", file=sys.stderr)
    print(f"  Audit chance (mean 1/n_opts): {audit_chance:.5f}, "
          f"threshold: {audit_chance+alpha:.5f}", file=sys.stderr)

    # ---- Surface-form shortcut checks (AMENDMENT-002 §2.5.2) ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running surface-form shortcut checks...",
          file=sys.stderr)
    sf_train = run_surface_form_checks(train_items, label="train")
    sf_audit = run_surface_form_checks(audit_items, label="audit")
    for split_label, sf_results in [("Train", sf_train), ("Audit", sf_audit)]:
        for check_name, check_result in sf_results.items():
            status = "PASS" if check_result["passed"] else "FAIL"
            print(f"  {split_label} {check_name}: {status}", file=sys.stderr)

    # ---- Template-held-out evaluation ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running template-held-out eval "
          f"on train set...", file=sys.stderr)
    held_out_results = template_held_out_eval(train_items, alpha=alpha)

    # ---- Final-audit evaluation (single read) ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running final-audit evaluation "
          f"(single read)...", file=sys.stderr)
    audit_results = final_audit_eval(audit_items, train_items, alpha=alpha)

    # ---- Overall verdict ----
    all_verdicts = ([r["verdict"] for r in held_out_results.values()] +
                    [r["verdict"] for r in audit_results.values()])
    overall = "FAIL" if "FAIL" in all_verdicts else "PASS"

    failed_ho = [n for n, r in held_out_results.items() if r["verdict"] == "FAIL"]
    failed_au = [n for n, r in audit_results.items() if r["verdict"] == "FAIL"]

    elapsed = time.time() - t0

    report = {
        "evaluator_version": "v2_a1_candidate_aware",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "design_changes": [
            "Classifier baselines (6-11) restructured to one-row-per-candidate-pair",
            "Target-vs-other feature normalization for permutation equivariance",
            "TF-IDF baselines include hypothesis text in feature representation",
            "Structured features: target + delta (target - mean_others)",
            "No hypothesis position index in any feature",
            "Baseline 9 renamed: polarity_feature -> mention_evidence",
            "Baseline 10 renamed: positional_feature -> first_mention_order",
        ],
        "preserved_from_phase_a": [
            "chance = mean(1/n_options)",
            "majority baseline: single canonical label from train",
            "per-regime = slice of aggregate predictions, not refit",
            "hard reconciliation assertion",
            "feature hygiene: no supports/contradicts/diagnostic_value",
        ],
        "surface_form_checks": {
            "train": sf_train,
            "audit": sf_audit,
        },
        "allowed_fields": BASELINE_ALLOWED_FIELDS,
        "train_corpus": {
            "path": "analysis/t2v2_train.jsonl",
            "n_items": len(train_items),
            "chance_level": round(train_chance, 5),
            "threshold": round(train_chance + alpha, 5),
            "alpha": alpha,
        },
        "final_audit_corpus": {
            "path": "analysis/t2v2_final_audit.jsonl",
            "n_items": len(audit_items),
            "chance_level": round(audit_chance, 5),
            "threshold": round(audit_chance + alpha, 5),
            "alpha": alpha,
        },
        "template_held_out_results": held_out_results,
        "final_audit_results": audit_results,
        "overall_verdict": overall,
        "failed_baselines_held_out": failed_ho,
        "failed_baselines_audit": failed_au,
        "elapsed_seconds": round(elapsed, 1),
    }

    # ---- Summary to stderr ----
    print(f"\n{'='*80}", file=sys.stderr)
    print("CANDIDATE-AWARE LEAKAGE EVALUATION RESULTS (Phase A.1)", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Train chance: {train_chance:.5f}  Threshold: "
          f"{train_chance+alpha:.5f}", file=sys.stderr)
    print(f"Audit chance: {audit_chance:.5f}  Threshold: "
          f"{audit_chance+alpha:.5f}", file=sys.stderr)

    for split_name, split_results in [("HELD-OUT", held_out_results),
                                       ("FINAL-AUDIT", audit_results)]:
        print(f"\n--- {split_name} Results ---", file=sys.stderr)
        for name in BASELINE_NAMES:
            r = split_results[name]
            print(f"  {name:30s}  acc={r['accuracy']:.4f}  "
                  f"CI=[{r['ci_lower']:.4f},{r['ci_upper']:.4f}]  "
                  f"{r['verdict']}", file=sys.stderr)
            for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
                if regime in r.get("per_regime", {}):
                    rr = r["per_regime"][regime]
                    print(f"    {regime:15s}  acc={rr['accuracy']:.4f}  "
                          f"CI=[{rr['ci_lower']:.4f},{rr['ci_upper']:.4f}]  "
                          f"ch={rr['chance']:.4f}  {rr['verdict']}",
                          file=sys.stderr)

    print(f"\nOVERALL VERDICT: {overall}", file=sys.stderr)
    if failed_ho:
        print(f"Failed on held-out: {failed_ho}", file=sys.stderr)
    if failed_au:
        print(f"Failed on audit: {failed_au}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # ---- Write JSON to new file (do NOT overwrite Phase-A results) ----
    outpath = "analysis/leakage_results_v2_a1.json"
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {outpath}", file=sys.stderr)

    return report


if __name__ == "__main__":
    main()
