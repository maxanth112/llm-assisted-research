#!/usr/bin/env python3
"""
Corrected 11-baseline leakage evaluator per AMENDMENT-001 §4 / AMENDMENT-002.

Fixes over v2 evaluator:
  (a) Per-regime AND aggregate verdicts on BOTH splits (held-out + audit).
  (b) Per-regime = decomposition by slicing one aggregate model's predictions,
      NOT refitting per regime.  Hard assertion: sum(regime_correct) == agg_correct.
  (c) Majority baseline: ONE canonical predicted label from training split,
      applied identically to every item.  Reconciliation guaranteed by slicing.
  (d) Chance = mean(1/n_options) per item, NOT 1/mean(n_options).
  (e) Feature hygiene: baselines use ONLY model-visible fields (narrative,
      question, hypotheses, evidence[].content).  NO use of supports, contradicts,
      diagnostic_value, regime, metadata, gold_answer, gold_reasoning, or any
      generator annotation.

Allowed fields per baseline (documented in output):
  ALL baselines may read: narrative, question, hypotheses, evidence[*].content
  NO baseline may read: regime, metadata.*, gold_answer, gold_reasoning,
      evidence[*].supports, evidence[*].contradicts, evidence[*].diagnostic_value,
      source_precedence_rule
"""

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


def gold_index(item) -> int:
    """Return index of gold_answer in hypotheses list, or -1."""
    try:
        return item["hypotheses"].index(item["gold_answer"])
    except ValueError:
        return -1


# ================================================================
# HEURISTIC BASELINES (per-item, no training)
# ================================================================
# Each returns a prediction array (one int per item = predicted hyp index).

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
        ev_contents = []
        for ev in it.get("evidence", []):
            if isinstance(ev, dict):
                ev_contents.append(ev.get("content", "").lower())
            elif isinstance(ev, str):
                ev_contents.append(ev.lower())
        counts = []
        for h in hyps:
            n = extract_name(h).lower()
            counts.append(sum(1 for et in ev_contents if n in et))
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
# CLASSIFIER BASELINES (train/test split required)
# ================================================================
# Each returns predictions on test_items (array of ints = predicted hyp index).
# Feature extraction uses ONLY model-visible fields.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _item_text(item) -> str:
    """Concatenation of model-visible text for an item."""
    return item.get("narrative", "") + " " + ev_content_text(item)


def _prepare_labels(items):
    """Return (valid_indices, labels) for items with valid gold_answer."""
    indices, labels = [], []
    for i, it in enumerate(items):
        gi = gold_index(it)
        if gi >= 0:
            indices.append(i)
            labels.append(gi)
    return indices, np.array(labels)


def _clf_fit_predict(train_X, train_y, test_X) -> np.ndarray:
    clf = LogisticRegression(max_iter=500, solver='saga', random_state=42)
    if len(set(train_y)) < 2:
        mc = Counter(train_y.tolist()).most_common(1)[0][0]
        return np.full(test_X.shape[0], mc)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)


def pred_tfidf_word(train_items, test_items) -> np.ndarray:
    tr_idx, tr_y = _prepare_labels(train_items)
    te_idx, te_y = _prepare_labels(test_items)
    tr_texts = [_item_text(train_items[i]) for i in tr_idx]
    te_texts = [_item_text(test_items[i]) for i in te_idx]
    vec = TfidfVectorizer(max_features=500, analyzer='word',
                          ngram_range=(1,2), stop_words='english')
    tr_X = vec.fit_transform(tr_texts)
    te_X = vec.transform(te_texts)
    fold_preds = _clf_fit_predict(tr_X, tr_y, te_X)
    # Map back to full test array (items with invalid gold get -1 prediction)
    full_preds = np.full(len(test_items), -1, dtype=int)
    for j, idx in enumerate(te_idx):
        full_preds[idx] = fold_preds[j]
    return full_preds


def pred_tfidf_char(train_items, test_items) -> np.ndarray:
    tr_idx, tr_y = _prepare_labels(train_items)
    te_idx, te_y = _prepare_labels(test_items)
    tr_texts = [_item_text(train_items[i]) for i in tr_idx]
    te_texts = [_item_text(test_items[i]) for i in te_idx]
    vec = TfidfVectorizer(max_features=500, analyzer='char_wb',
                          ngram_range=(2,4))
    tr_X = vec.fit_transform(tr_texts)
    te_X = vec.transform(te_texts)
    fold_preds = _clf_fit_predict(tr_X, tr_y, te_X)
    full_preds = np.full(len(test_items), -1, dtype=int)
    for j, idx in enumerate(te_idx):
        full_preds[idx] = fold_preds[j]
    return full_preds


def _build_clean_features(items, max_hyp=4):
    """Build structured features using ONLY model-visible fields.

    Per hypothesis slot:  mention_count, evidence_count, length_sum, position
    (4 features per slot, max_hyp slots = 16 features total)

    NO supports/contradicts/diagnostic_value (generator annotations).
    """
    X_list = []
    for it in items:
        hyps = it["hypotheses"]
        narrative_lower = it.get("narrative", "").lower()
        ev_contents = []
        for ev in it.get("evidence", []):
            if isinstance(ev, dict):
                ev_contents.append(ev.get("content", ""))
            elif isinstance(ev, str):
                ev_contents.append(ev)
        full_text = (narrative_lower + " " + " ".join(ev_contents)).lower()

        row = []
        for idx, h in enumerate(hyps):
            n = extract_name(h).lower()
            mention_count = full_text.count(n)
            evidence_count = sum(1 for et in ev_contents if n in et.lower())
            length_sum = sum(len(et) for et in ev_contents if n in et.lower())
            row.extend([mention_count, evidence_count, length_sum, idx])
        # Pad to max_hyp * 4
        while len(row) < max_hyp * 4:
            row.append(0)
        X_list.append(row[:max_hyp * 4])
    return np.array(X_list) if X_list else np.zeros((0, max_hyp * 4))


def _clf_structured(train_items, test_items, col_selector=None) -> np.ndarray:
    """Train/predict with structured features, optionally selecting columns."""
    tr_idx, tr_y = _prepare_labels(train_items)
    te_idx, te_y = _prepare_labels(test_items)
    tr_X = _build_clean_features([train_items[i] for i in tr_idx])
    te_X = _build_clean_features([test_items[i] for i in te_idx])
    if col_selector is not None:
        cols = [c for c in col_selector if c < tr_X.shape[1]]
        tr_X = tr_X[:, cols]
        te_X = te_X[:, cols]
    if len(tr_y) < 2 or len(te_y) < 1:
        return np.full(len(test_items), -1, dtype=int)
    fold_preds = _clf_fit_predict(tr_X, tr_y, te_X)
    full_preds = np.full(len(test_items), -1, dtype=int)
    for j, idx in enumerate(te_idx):
        full_preds[idx] = fold_preds[j]
    return full_preds


def pred_length(train_items, test_items) -> np.ndarray:
    # length_sum features: indices 2, 6, 10, 14 (every 4, offset 2)
    cols = [i * 4 + 2 for i in range(4)]
    return _clf_structured(train_items, test_items, cols)


def pred_polarity(train_items, test_items) -> np.ndarray:
    # After removing supports/contradicts, polarity = mention_count + evidence_count
    # as proxy (indices 0,1,4,5,8,9,12,13)
    cols = []
    for i in range(4):
        cols.extend([i * 4 + 0, i * 4 + 1])
    return _clf_structured(train_items, test_items, cols)


def pred_positional(train_items, test_items) -> np.ndarray:
    # position features: indices 3, 7, 11, 15 (every 4, offset 3)
    cols = [i * 4 + 3 for i in range(4)]
    return _clf_structured(train_items, test_items, cols)


def pred_combined(train_items, test_items) -> np.ndarray:
    return _clf_structured(train_items, test_items, col_selector=None)


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
    "9_polarity_feature",
    "10_positional_feature",
    "11_combined_shallow",
]

# Fields each baseline is allowed to read (for documentation)
BASELINE_ALLOWED_FIELDS = {
    "1_majority_class": ["hypotheses (count only)"],
    "2_label_position": ["hypotheses"],
    "3_mention_count": ["narrative", "hypotheses", "evidence[].content"],
    "4_evidence_count": ["hypotheses", "evidence[].content"],
    "5_lexical_overlap": ["hypotheses", "evidence[].content"],
    "6_tfidf_word": ["narrative", "evidence[].content", "hypotheses (for label index)"],
    "7_tfidf_char": ["narrative", "evidence[].content", "hypotheses (for label index)"],
    "8_length_feature": ["narrative", "hypotheses", "evidence[].content"],
    "9_polarity_feature": ["narrative", "hypotheses", "evidence[].content"],
    "10_positional_feature": ["narrative", "hypotheses", "evidence[].content"],
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
    item_to_fold = []
    fold_boundaries = {}
    offset = 0
    for t in templates:
        fold_boundaries[t] = (offset, offset + len(by_template[t]))
        for it in by_template[t]:
            ordered_items.append(it)
            item_to_fold.append(t)
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

        # --- Classifier baselines (train on train, predict on test) ---
        all_preds["6_tfidf_word"][lo:hi] = pred_tfidf_word(train_items, test_items)
        all_preds["7_tfidf_char"][lo:hi] = pred_tfidf_char(train_items, test_items)
        all_preds["8_length_feature"][lo:hi] = pred_length(train_items, test_items)
        all_preds["9_polarity_feature"][lo:hi] = pred_polarity(train_items, test_items)
        all_preds["10_positional_feature"][lo:hi] = pred_positional(train_items, test_items)
        all_preds["11_combined_shallow"][lo:hi] = pred_combined(train_items, test_items)

    # Decompose each baseline
    results = {}
    for name in BASELINE_NAMES:
        results[name] = _decompose(all_preds[name], golds, ordered_items,
                                   chance_agg, alpha)
    return results


def final_audit_eval(audit_items, train_items, alpha=0.05):
    """Evaluate all baselines on audit_items. Classifiers trained on train_items.
    Per-regime is decomposition of the SAME predictions (no refitting)."""
    n = len(audit_items)
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
    all_preds["9_polarity_feature"] = pred_polarity(train_items, audit_items)
    all_preds["10_positional_feature"] = pred_positional(train_items, audit_items)
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
        "evaluator_version": "v2_corrected",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "corrections_applied": [
            "chance = mean(1/n_options) instead of 1/mean(n_options)",
            "majority baseline: single canonical label from train, applied uniformly",
            "per-regime = slice of aggregate predictions, not refit",
            "hard reconciliation assertion: sum(regime_correct) == agg_correct",
            "feature hygiene: removed supports/contradicts/diagnostic_value from features",
            "per-regime verdicts with per-regime chance and threshold on both splits",
        ],
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
    print("CORRECTED LEAKAGE EVALUATION RESULTS", file=sys.stderr)
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

    # ---- Write JSON ----
    outpath = "analysis/leakage_results_v2_corrected.json"
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {outpath}", file=sys.stderr)

    return report


if __name__ == "__main__":
    main()
