#!/usr/bin/env python3
"""
Run the full 11-baseline leakage evaluation per AMENDMENT-001 §4.

- Template-held-out evaluation on t2v2_train.jsonl
- Single final-audit evaluation on t2v2_final_audit.jsonl
- Per-regime breakdown + aggregate + Wilson CIs
- Machine-readable JSON output
"""

import json, math, sys, os, warnings, time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Callable, Optional

warnings.filterwarnings('ignore')
import numpy as np

# --------------- helpers ---------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
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

def extract_name(hyp: str) -> str:
    for pat in [" is responsible", " committed", " is the perpetrator", " is guilty"]:
        if pat in hyp:
            return hyp.split(pat)[0].strip()
    words = hyp.strip().split()
    return ' '.join(words[:2]) if len(words) >= 2 else hyp.strip()

def ev_text(item):
    parts = []
    for ev in item.get("evidence", []):
        if isinstance(ev, dict):
            parts.append(ev.get("content", ev.get("text", "")))
        elif isinstance(ev, str):
            parts.append(ev)
    return " ".join(parts)

def chance_level(items):
    hyp_counts = [len(it.get("hypotheses",[])) for it in items]
    return 1.0 / (sum(hyp_counts)/len(hyp_counts)) if items else 0.0

# --------------- heuristic baselines (no training) ---------------

def bl_majority(items):
    golds = [it["gold_answer"] for it in items]
    c = Counter(golds)
    _, top = c.most_common(1)[0]
    k = sum(1 for g in golds if g == c.most_common(1)[0][0])
    return k, len(items)

def bl_position(items):
    k = sum(1 for it in items if it["hypotheses"][0] == it["gold_answer"])
    return k, len(items)

def bl_mention_count(items):
    k = 0
    for it in items:
        text = (it.get("narrative","") + " " + ev_text(it)).lower()
        names = [(h, extract_name(h)) for h in it["hypotheses"]]
        counts = {h: text.count(n.lower()) for h,n in names}
        mx = max(counts.values())
        pred = next(h for h in it["hypotheses"] if counts[h] == mx)
        if pred == it["gold_answer"]:
            k += 1
    return k, len(items)

def bl_evidence_count(items):
    k = 0
    for it in items:
        names = [(h, extract_name(h)) for h in it["hypotheses"]]
        counts = {}
        for h, n in names:
            c = 0
            for ev in it.get("evidence",[]):
                et = (ev.get("content","") if isinstance(ev,dict) else str(ev)).lower()
                if n.lower() in et:
                    c += 1
            counts[h] = c
        mx = max(counts.values())
        pred = next(h for h in it["hypotheses"] if counts[h] == mx)
        if pred == it["gold_answer"]:
            k += 1
    return k, len(items)

def bl_lexical_overlap(items):
    k = 0
    for it in items:
        ew = set(ev_text(it).lower().split())
        scores = {}
        for h in it["hypotheses"]:
            hw = set(h.lower().split())
            scores[h] = len(ew & hw)
        mx = max(scores.values())
        pred = next(h for h in it["hypotheses"] if scores[h] == mx)
        if pred == it["gold_answer"]:
            k += 1
    return k, len(items)

# --------------- classifier baselines (train/test) ---------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def _prepare_index_labels(items):
    """Return (texts, labels, valid_items) where labels are gold_answer index."""
    texts, labels, valid = [], [], []
    for it in items:
        ga = it["gold_answer"]
        try:
            idx = it["hypotheses"].index(ga)
        except ValueError:
            continue
        texts.append(it.get("narrative","") + " " + ev_text(it))
        labels.append(idx)
        valid.append(it)
    return texts, np.array(labels), valid

def _clf_train_predict(train_X, train_y, test_X):
    clf = LogisticRegression(max_iter=500, solver='saga', random_state=42)
    if len(set(train_y)) < 2:
        # predict majority from train
        mc = Counter(train_y).most_common(1)[0][0]
        return np.full(test_X.shape[0], mc)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)

def bl_tfidf_word(train_items, test_items):
    tr_texts, tr_y, _ = _prepare_index_labels(train_items)
    te_texts, te_y, _ = _prepare_index_labels(test_items)
    if len(tr_texts) < 2 or len(te_texts) < 1:
        return 0, len(te_texts)
    vec = TfidfVectorizer(max_features=500, analyzer='word', ngram_range=(1,2), stop_words='english')
    tr_X = vec.fit_transform(tr_texts)
    te_X = vec.transform(te_texts)
    preds = _clf_train_predict(tr_X, tr_y, te_X)
    return int((preds == te_y).sum()), len(te_y)

def bl_tfidf_char(train_items, test_items):
    tr_texts, tr_y, _ = _prepare_index_labels(train_items)
    te_texts, te_y, _ = _prepare_index_labels(test_items)
    if len(tr_texts) < 2 or len(te_texts) < 1:
        return 0, len(te_texts)
    vec = TfidfVectorizer(max_features=500, analyzer='char_wb', ngram_range=(2,4))
    tr_X = vec.fit_transform(tr_texts)
    te_X = vec.transform(te_texts)
    preds = _clf_train_predict(tr_X, tr_y, te_X)
    return int((preds == te_y).sum()), len(te_y)

def _build_structured_features(items, max_hyp=4):
    """Build per-candidate feature vectors: mention_count, ev_count, length, n_support, n_contradict, position, first_mention."""
    X_list, y_list = [], []
    for it in items:
        hyps = it["hypotheses"]
        ga = it["gold_answer"]
        try:
            yi = hyps.index(ga)
        except ValueError:
            continue
        names = [extract_name(h) for h in hyps]
        narrative_lower = it.get("narrative","").lower()
        evs = it.get("evidence",[])
        ev_texts_raw = []
        for ev in evs:
            if isinstance(ev,dict):
                ev_texts_raw.append(ev.get("content",ev.get("text","")))
            else:
                ev_texts_raw.append(str(ev))
        full_text = (narrative_lower + " " + " ".join(ev_texts_raw)).lower()
        combined_ev = " ".join(ev_texts_raw).lower()
        row = []
        for idx, (h, n) in enumerate(zip(hyps, names)):
            nl = n.lower()
            mention_count = full_text.count(nl)
            evidence_count = sum(1 for et in ev_texts_raw if nl in et.lower())
            length_feat = sum(len(et) for et in ev_texts_raw if nl in et.lower())
            n_sup, n_con = 0, 0
            for ev in evs:
                if not isinstance(ev, dict): continue
                for s in ev.get("supports",[]):
                    if isinstance(s,str) and (nl in s.lower() or h==s): n_sup += 1
                for c in ev.get("contradicts",[]):
                    if isinstance(c,str) and (nl in c.lower() or h==c): n_con += 1
            first_mention = combined_ev.find(nl)
            if first_mention == -1: first_mention = 999999
            row.extend([mention_count, evidence_count, length_feat, n_sup, n_con, idx, first_mention])
        while len(row) < max_hyp * 7:
            row.append(0)
        X_list.append(row[:max_hyp*7])
        y_list.append(yi)
    return np.array(X_list) if X_list else np.zeros((0,max_hyp*7)), np.array(y_list)

def bl_length(train_items, test_items):
    tr_X, tr_y = _build_structured_features(train_items)
    te_X, te_y = _build_structured_features(test_items)
    # length features are indices 2, 9, 16, 23 (every 7, offset 2)
    cols = [i*7+2 for i in range(4)]
    cols = [c for c in cols if c < tr_X.shape[1]]
    if len(tr_y) < 2 or len(te_y) < 1: return 0, len(te_y)
    preds = _clf_train_predict(tr_X[:,cols], tr_y, te_X[:,cols])
    return int((preds==te_y).sum()), len(te_y)

def bl_polarity(train_items, test_items):
    tr_X, tr_y = _build_structured_features(train_items)
    te_X, te_y = _build_structured_features(test_items)
    cols = []
    for i in range(4):
        cols.extend([i*7+3, i*7+4])  # n_sup, n_con
    cols = [c for c in cols if c < tr_X.shape[1]]
    if len(tr_y) < 2 or len(te_y) < 1: return 0, len(te_y)
    preds = _clf_train_predict(tr_X[:,cols], tr_y, te_X[:,cols])
    return int((preds==te_y).sum()), len(te_y)

def bl_positional(train_items, test_items):
    tr_X, tr_y = _build_structured_features(train_items)
    te_X, te_y = _build_structured_features(test_items)
    cols = []
    for i in range(4):
        cols.extend([i*7+5, i*7+6])  # position, first_mention
    cols = [c for c in cols if c < tr_X.shape[1]]
    if len(tr_y) < 2 or len(te_y) < 1: return 0, len(te_y)
    preds = _clf_train_predict(tr_X[:,cols], tr_y, te_X[:,cols])
    return int((preds==te_y).sum()), len(te_y)

def bl_combined(train_items, test_items):
    tr_X, tr_y = _build_structured_features(train_items)
    te_X, te_y = _build_structured_features(test_items)
    if len(tr_y) < 2 or len(te_y) < 1: return 0, len(te_y)
    preds = _clf_train_predict(tr_X, tr_y, te_X)
    return int((preds==te_y).sum()), len(te_y)


# --------------- evaluation framework ---------------

HEURISTIC_BASELINES = [
    ("1_majority_class", bl_majority),
    ("2_label_position", bl_position),
    ("3_mention_count", bl_mention_count),
    ("4_evidence_count", bl_evidence_count),
    ("5_lexical_overlap", bl_lexical_overlap),
]
CLASSIFIER_BASELINES = [
    ("6_tfidf_word", bl_tfidf_word),
    ("7_tfidf_char", bl_tfidf_char),
    ("8_length_feature", bl_length),
    ("9_polarity_feature", bl_polarity),
    ("10_positional_feature", bl_positional),
    ("11_combined_shallow", bl_combined),
]

def eval_on_split(items, train_items_for_clf=None, label="split"):
    """Evaluate all 11 baselines on `items`. For classifier baselines, train on `train_items_for_clf`."""
    results = {}
    for name, fn in HEURISTIC_BASELINES:
        k, n = fn(items)
        ci_lo, ci_hi = wilson_ci(k, n)
        results[name] = {"n_correct": k, "n_items": n, "accuracy": k/n if n else 0,
                         "ci_lower": round(ci_lo,5), "ci_upper": round(ci_hi,5)}
    train = train_items_for_clf if train_items_for_clf else items
    for name, fn in CLASSIFIER_BASELINES:
        k, n = fn(train, items)
        ci_lo, ci_hi = wilson_ci(k, n)
        results[name] = {"n_correct": k, "n_items": n, "accuracy": k/n if n else 0,
                         "ci_lower": round(ci_lo,5), "ci_upper": round(ci_hi,5)}
    return results

def per_regime_eval(items, train_for_clf=None, label=""):
    """Evaluate per-regime."""
    by_regime = defaultdict(list)
    for it in items:
        by_regime[it["regime"]].append(it)
    train_by_regime = defaultdict(list)
    if train_for_clf:
        for it in train_for_clf:
            train_by_regime[it["regime"]].append(it)
    result = {}
    for regime in ["CLEAN","DECOY","CONFLICT","INSUFFICIENT"]:
        if regime not in by_regime:
            continue
        regime_items = by_regime[regime]
        regime_train = train_by_regime.get(regime, regime_items)
        result[regime] = eval_on_split(regime_items, regime_train, label=f"{label}_{regime}")
    return result


def template_held_out_eval(items):
    """Leave-one-template-family-out for classifier training; heuristics just evaluated on test."""
    by_template = defaultdict(list)
    for it in items:
        t = it.get("metadata",{}).get("template","unknown")
        by_template[t].append(it)
    templates = sorted(by_template.keys())
    
    # For each template, hold it out and train on rest
    all_preds = {name: [] for name, _ in HEURISTIC_BASELINES + CLASSIFIER_BASELINES}
    all_golds = {name: [] for name, _ in HEURISTIC_BASELINES + CLASSIFIER_BASELINES}
    all_regimes = {name: [] for name, _ in HEURISTIC_BASELINES + CLASSIFIER_BASELINES}
    
    for held_out in templates:
        test_items = by_template[held_out]
        train_items = []
        for t in templates:
            if t != held_out:
                train_items.extend(by_template[t])
        
        # Heuristic baselines: just evaluate on test
        for name, fn in HEURISTIC_BASELINES:
            for it in test_items:
                k, n = fn([it])
                all_preds[name].append(1 if k > 0 else 0)
                all_golds[name].append(1)
                all_regimes[name].append(it["regime"])
        
        # Classifier baselines: train on train, predict on test
        for name, fn in CLASSIFIER_BASELINES:
            k_total, n_total = fn(train_items, test_items)
            # We need per-item preds for per-regime breakdown
            # Run per-item for heuristic-style count
            # Actually for proper aggregation, just accumulate totals
            pass
    
    # Simpler approach: just accumulate correct/total across folds
    fold_results = {}
    per_regime_results = defaultdict(lambda: defaultdict(lambda: {"k":0,"n":0}))
    
    for name, _ in HEURISTIC_BASELINES + CLASSIFIER_BASELINES:
        fold_results[name] = {"k": 0, "n": 0}
    
    for held_out in templates:
        test_items = by_template[held_out]
        train_items = []
        for t in templates:
            if t != held_out:
                train_items.extend(by_template[t])
        
        for name, fn in HEURISTIC_BASELINES:
            k, n = fn(test_items)
            fold_results[name]["k"] += k
            fold_results[name]["n"] += n
            # per-regime
            for regime in ["CLEAN","DECOY","CONFLICT","INSUFFICIENT"]:
                regime_items = [it for it in test_items if it["regime"] == regime]
                if regime_items:
                    rk, rn = fn(regime_items)
                    per_regime_results[name][regime]["k"] += rk
                    per_regime_results[name][regime]["n"] += rn
        
        for name, fn in CLASSIFIER_BASELINES:
            k, n = fn(train_items, test_items)
            fold_results[name]["k"] += k
            fold_results[name]["n"] += n
            # per-regime
            for regime in ["CLEAN","DECOY","CONFLICT","INSUFFICIENT"]:
                regime_test = [it for it in test_items if it["regime"] == regime]
                regime_train = [it for it in train_items if it["regime"] == regime]
                if regime_test:
                    rk, rn = fn(regime_train if regime_train else train_items, regime_test)
                    per_regime_results[name][regime]["k"] += rk
                    per_regime_results[name][regime]["n"] += rn
    
    # Compile results
    results = {}
    for name in [n for n,_ in HEURISTIC_BASELINES + CLASSIFIER_BASELINES]:
        k, n = fold_results[name]["k"], fold_results[name]["n"]
        ci_lo, ci_hi = wilson_ci(k, n)
        entry = {
            "n_correct": k, "n_items": n,
            "accuracy": round(k/n,5) if n else 0,
            "ci_lower": round(ci_lo,5), "ci_upper": round(ci_hi,5),
        }
        regime_breakdown = {}
        for regime in ["CLEAN","DECOY","CONFLICT","INSUFFICIENT"]:
            rk = per_regime_results[name][regime]["k"]
            rn = per_regime_results[name][regime]["n"]
            rci_lo, rci_hi = wilson_ci(rk, rn)
            regime_breakdown[regime] = {
                "n_correct": rk, "n_items": rn,
                "accuracy": round(rk/rn,5) if rn else 0,
                "ci_lower": round(rci_lo,5), "ci_upper": round(rci_hi,5),
            }
        entry["per_regime"] = regime_breakdown
        results[name] = entry
    return results

# --------------- main ---------------

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading datasets...", file=sys.stderr)
    train_items = load_jsonl("analysis/t2v2_train.jsonl")
    audit_items = load_jsonl("analysis/t2v2_final_audit.jsonl")
    print(f"  Train: {len(train_items)} items, Audit: {len(audit_items)} items", file=sys.stderr)
    
    # Compute chance levels
    train_chance = chance_level(train_items)
    audit_chance = chance_level(audit_items)
    alpha = 0.05
    train_threshold = train_chance + alpha
    audit_threshold = audit_chance + alpha
    
    print(f"  Train chance: {train_chance:.4f}, threshold: {train_threshold:.4f}", file=sys.stderr)
    print(f"  Audit chance: {audit_chance:.4f}, threshold: {audit_threshold:.4f}", file=sys.stderr)
    
    # Phase 1a: Template-held-out evaluation on train set
    print(f"\n[{time.strftime('%H:%M:%S')}] Running template-held-out evaluation on train set...", file=sys.stderr)
    held_out_results = template_held_out_eval(train_items)
    
    # Verdicts on held-out
    for name, res in held_out_results.items():
        res["verdict"] = "FAIL" if res["ci_upper"] > train_threshold else "PASS"
        for regime, rres in res.get("per_regime",{}).items():
            rr_chance = chance_level([it for it in train_items if it["regime"]==regime])
            rres["regime_chance"] = round(rr_chance, 5)
            rres["regime_threshold"] = round(rr_chance + alpha, 5)
            rres["verdict"] = "FAIL" if rres["ci_upper"] > rr_chance + alpha else "PASS"
    
    # Phase 1b: Final-audit evaluation (EXACTLY ONCE)
    print(f"\n[{time.strftime('%H:%M:%S')}] Running final-audit evaluation (single read)...", file=sys.stderr)
    audit_results = eval_on_split(audit_items, train_items_for_clf=train_items, label="final_audit")
    
    # Per-regime on audit
    audit_regime = per_regime_eval(audit_items, train_for_clf=train_items, label="audit")
    for name in audit_results:
        audit_results[name]["per_regime"] = {}
        for regime in ["CLEAN","DECOY","CONFLICT","INSUFFICIENT"]:
            if regime in audit_regime and name in audit_regime[regime]:
                audit_results[name]["per_regime"][regime] = audit_regime[regime][name]
        audit_results[name]["verdict"] = "FAIL" if audit_results[name]["ci_upper"] > audit_threshold else "PASS"
    
    # Overall verdict
    all_verdicts = [r["verdict"] for r in held_out_results.values()] + \
                   [r["verdict"] for r in audit_results.values()]
    overall = "FAIL" if "FAIL" in all_verdicts else "PASS"
    
    failed_baselines_held_out = [n for n,r in held_out_results.items() if r["verdict"]=="FAIL"]
    failed_baselines_audit = [n for n,r in audit_results.items() if r["verdict"]=="FAIL"]
    
    elapsed = time.time() - t0
    
    report = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "train_corpus": {
            "path": "analysis/t2v2_train.jsonl",
            "n_items": len(train_items),
            "chance_level": round(train_chance, 5),
            "threshold": round(train_threshold, 5),
            "alpha": alpha,
        },
        "final_audit_corpus": {
            "path": "analysis/t2v2_final_audit.jsonl",
            "n_items": len(audit_items),
            "chance_level": round(audit_chance, 5),
            "threshold": round(audit_threshold, 5),
            "alpha": alpha,
        },
        "template_held_out_results": held_out_results,
        "final_audit_results": audit_results,
        "overall_verdict": overall,
        "failed_baselines_held_out": failed_baselines_held_out,
        "failed_baselines_audit": failed_baselines_audit,
        "elapsed_seconds": round(elapsed, 1),
    }
    
    # Print summary
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"LEAKAGE EVALUATION RESULTS", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Train chance: {train_chance:.4f}  Threshold: {train_threshold:.4f}", file=sys.stderr)
    print(f"Audit chance: {audit_chance:.4f}  Threshold: {audit_threshold:.4f}", file=sys.stderr)
    
    print(f"\n--- Template-Held-Out Results (train) ---", file=sys.stderr)
    for name in sorted(held_out_results):
        r = held_out_results[name]
        print(f"  {name:30s}  acc={r['accuracy']:.4f}  CI=[{r['ci_lower']:.4f},{r['ci_upper']:.4f}]  {r['verdict']}", file=sys.stderr)
    
    print(f"\n--- Final-Audit Results ---", file=sys.stderr)
    for name in sorted(audit_results):
        r = audit_results[name]
        print(f"  {name:30s}  acc={r['accuracy']:.4f}  CI=[{r['ci_lower']:.4f},{r['ci_upper']:.4f}]  {r['verdict']}", file=sys.stderr)
    
    print(f"\nOVERALL VERDICT: {overall}", file=sys.stderr)
    if failed_baselines_held_out:
        print(f"Failed on held-out: {failed_baselines_held_out}", file=sys.stderr)
    if failed_baselines_audit:
        print(f"Failed on audit: {failed_baselines_audit}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)
    
    # Write JSON
    outpath = "analysis/leakage_results_v2.json"
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {outpath}", file=sys.stderr)
    
    return report

if __name__ == "__main__":
    main()
