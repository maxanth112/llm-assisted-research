#!/usr/bin/env python3
"""
v3.1 development-corpus leakage gate evaluation.

Generates a v3.1 development corpus (n_per_regime=500, 2000 items total),
runs ALL surface gates (S1-S6) and the FULL baseline battery (11 existing +
3 diagnostic), and reports aggregate/per-regime results with Wilson CIs.

This is NOT the 8,000-item final audit.

Usage:
    cd llm-assisted-research
    python3 analysis/run_v31_dev_gate.py
"""

import gc
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datasets.t2_generator.generator_v3_1 import (
    T2V31Generator,
    ABSTENTION_HYPOTHESIS,
    N_EVIDENCE_SLOTS,
)

import analysis.run_leakage_eval as _eval_mod

# ================================================================
# MONKEY-PATCH: adapt evaluator for v3.1 hypothesis format
# ================================================================
# v3.1 uses "Based on the available evidence, [Name] is uniquely
# identified as responsible." instead of "[Name] is responsible".
# The evaluator's extract_name and _is_abstention_option must be
# updated to handle both v2/v3 and v3.1 hypothesis formats.

_original_extract_name = _eval_mod.extract_name


def _v31_extract_name(hyp: str) -> str:
    """Extract suspect name from v3.1 or v2/v3 hypothesis string."""
    # v3.1 format: "Based on the available evidence, X is uniquely
    #               identified as responsible."
    v31_marker = " is uniquely identified as responsible"
    if v31_marker in hyp:
        prefix = "Based on the available evidence, "
        text = hyp
        if text.startswith(prefix):
            text = text[len(prefix):]
        return text.split(v31_marker)[0].strip()
    # Fall back to original for v2/v3 format
    return _original_extract_name(hyp)


_original_is_abstention = _eval_mod._is_abstention_option


def _v31_is_abstention_option(text: str) -> bool:
    """Check if hypothesis is abstention (v3.1 or v2/v3 format)."""
    # v3.1 abstention
    if "no listed suspect is uniquely identified" in text.lower():
        return True
    # Original v2/v3 patterns
    return _original_is_abstention(text)


# Apply patches to the module so all functions that call
# extract_name / _is_abstention_option use the v3.1-aware versions
_eval_mod.extract_name = _v31_extract_name
_eval_mod._is_abstention_option = _v31_is_abstention_option

# Also patch in diagnostic_baselines since it imports from run_leakage_eval
import analysis.diagnostic_baselines as _diag_mod
# diagnostic_baselines imports extract_name at module level; the import
# binds to the original function object, not the module attribute.
# We need to also patch the local reference if it exists.
if hasattr(_diag_mod, 'extract_name'):
    _diag_mod.extract_name = _v31_extract_name

# ================================================================

from analysis.run_leakage_eval import (
    load_jsonl,
    chance_level_correct,
    run_surface_form_checks,
    template_held_out_eval,
    wilson_ci,
    gold_index,
    BASELINE_NAMES,
)

from analysis.diagnostic_baselines import (
    pred_option_only,
    pred_full_candidate,
    evaluate_context_regime,
    DIAGNOSTIC_BASELINE_NAMES,
)


# ================================================================
# CONFIG
# ================================================================

N_PER_REGIME = 500
SEED = 42
CORPUS_PATH = "analysis/t2v31_dev_corpus.jsonl"
RESULTS_PATH = "analysis/leakage_results_v31_dev.json"
REPORT_PATH = "analysis/v31_devgate_report.md"


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()

    # ---- Generate corpus ----
    print(f"[{time.strftime('%H:%M:%S')}] Generating v3.1 dev corpus "
          f"(n_per_regime={N_PER_REGIME}, seed={SEED})...", file=sys.stderr)
    gen = T2V31Generator(seed=SEED)
    items_raw = gen.generate_dataset(n_per_regime=N_PER_REGIME, seed=SEED)
    print(f"  Total items: {len(items_raw)}", file=sys.stderr)

    # Export to JSONL
    lines = []
    for item in items_raw:
        lines.append(json.dumps(asdict(item), ensure_ascii=False))
    corpus_text = '\n'.join(lines) + '\n'
    corpus_hash = hashlib.sha256(corpus_text.encode('utf-8')).hexdigest()
    with open(CORPUS_PATH, 'w') as f:
        f.write(corpus_text)
    print(f"  Corpus SHA-256: {corpus_hash}", file=sys.stderr)

    # Reload as dicts
    items = load_jsonl(CORPUS_PATH)
    assert len(items) == len(items_raw)

    # Regime distribution
    regime_counts = Counter(it["regime"] for it in items)
    print(f"  Regimes: {dict(regime_counts)}", file=sys.stderr)

    # Chance level
    alpha = 0.05
    chance = chance_level_correct(items)
    threshold = chance + alpha
    print(f"  Chance: {chance:.5f}, Threshold: {threshold:.5f}",
          file=sys.stderr)

    # ---- Surface-form checks ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running surface-form checks...",
          file=sys.stderr)
    sf_results = run_surface_form_checks(items, label="v31_dev")
    sf_all_pass = True
    for check_name, check_result in sf_results.items():
        status = "PASS" if check_result["passed"] else "FAIL"
        print(f"  {check_name}: {status}", file=sys.stderr)
        if not check_result["passed"]:
            sf_all_pass = False
            print(f"    Detail: {json.dumps(check_result, indent=2, default=str)[:500]}",
                  file=sys.stderr)

    # ---- Template-held-out evaluation (11 baselines) ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running template-held-out eval "
          f"(11 baselines)...", file=sys.stderr)
    held_out_results = template_held_out_eval(items, alpha=alpha)

    # ---- Diagnostic baselines (D1, D2, D3) via template-held-out ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Running diagnostic baselines...",
          file=sys.stderr)

    # D1 and D3 use template-held-out, same as the 11 baselines
    by_template = defaultdict(list)
    for it in items:
        t = it.get("metadata", {}).get("template", "unknown")
        by_template[t].append(it)
    templates = sorted(by_template.keys())

    # Build ordered items and collect D1/D3 predictions per fold
    ordered_items = []
    d1_preds = []
    d3_preds = []

    for t in templates:
        test_items = by_template[t]
        train_items = []
        for ot in templates:
            if ot != t:
                train_items.extend(by_template[ot])

        p_d1 = pred_option_only(train_items, test_items)
        p_d3 = pred_full_candidate(train_items, test_items)

        for i, it in enumerate(test_items):
            ordered_items.append(it)
            d1_preds.append(int(p_d1[i]))
            d3_preds.append(int(p_d3[i]))

        gc.collect()

    d1_preds = np.array(d1_preds)
    d3_preds = np.array(d3_preds)

    # Compute D1, D3 results
    golds = np.array([gold_index(it) for it in ordered_items])
    chance_agg = chance_level_correct(ordered_items)

    diagnostic_results = {}
    for name, preds in [("D1_option_only", d1_preds),
                        ("D3_full_candidate", d3_preds)]:
        valid = golds >= 0
        n_valid = int(valid.sum())
        correct = (preds[valid] == golds[valid])
        k = int(correct.sum())
        acc = k / n_valid if n_valid > 0 else 0
        ci_lo, ci_hi = wilson_ci(k, n_valid)
        threshold_agg = chance_agg + alpha

        result = {
            "n_items": n_valid,
            "n_correct": k,
            "accuracy": round(acc, 5),
            "ci_lower": round(ci_lo, 5),
            "ci_upper": round(ci_hi, 5),
            "chance": round(chance_agg, 5),
            "threshold": round(threshold_agg, 5),
            "verdict": "FAIL" if ci_hi > threshold_agg else "PASS",
        }

        # Per-regime
        per_regime = {}
        for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
            mask = np.array([
                it.get("regime") == regime for it in ordered_items
            ])
            r_valid = valid & mask
            r_n = int(r_valid.sum())
            if r_n == 0:
                continue
            r_correct = (preds[r_valid] == golds[r_valid])
            r_k = int(r_correct.sum())
            r_acc = r_k / r_n if r_n > 0 else 0
            r_ci_lo, r_ci_hi = wilson_ci(r_k, r_n)
            r_chance = float(np.mean([
                1.0 / len(it["hypotheses"])
                for it, m in zip(ordered_items, mask) if m
            ]))
            r_threshold = r_chance + alpha
            per_regime[regime] = {
                "n_items": r_n,
                "n_correct": r_k,
                "accuracy": round(r_acc, 5),
                "ci_lower": round(r_ci_lo, 5),
                "ci_upper": round(r_ci_hi, 5),
                "chance": round(r_chance, 5),
                "threshold": round(r_threshold, 5),
                "verdict": "FAIL" if r_ci_hi > r_threshold else "PASS",
            }
        result["per_regime"] = per_regime
        diagnostic_results[name] = result

    # D2: context-regime classifier
    print(f"  Running D2 (context-regime classifier)...", file=sys.stderr)
    d2_result = evaluate_context_regime(items, alpha=alpha)
    diagnostic_results["D2_context_regime"] = d2_result

    # ---- Overall verdict ----
    all_verdicts = [r["verdict"] for r in held_out_results.values()]
    # D1 and D3 also gating (same 4-option argmax baselines)
    all_verdicts.extend([
        diagnostic_results["D1_option_only"]["verdict"],
        diagnostic_results["D3_full_candidate"]["verdict"],
    ])
    # D2 is informational (binary regime classifier, different scale)
    overall = "FAIL" if "FAIL" in all_verdicts else "PASS"

    failed_baselines = []
    for n, r in held_out_results.items():
        if r["verdict"] == "FAIL":
            failed_baselines.append(n)
    for n in ["D1_option_only", "D3_full_candidate"]:
        if diagnostic_results[n]["verdict"] == "FAIL":
            failed_baselines.append(n)

    elapsed = time.time() - t0

    # ---- Build report ----
    report = {
        "evaluator_version": "v3.1_dev_gate",
        "generator_version": "3.1.0",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                         time.gmtime()),
        "corpus_config": {
            "n_per_regime": N_PER_REGIME,
            "seed": SEED,
            "total_items": len(items),
            "regimes": dict(regime_counts),
            "n_hypotheses_per_item": 4,
            "n_evidence_per_item": N_EVIDENCE_SLOTS,
            "chance_level": round(chance, 5),
            "threshold": round(threshold, 5),
            "corpus_sha256": corpus_hash,
        },
        "surface_form_checks": sf_results,
        "template_held_out_results": held_out_results,
        "diagnostic_results": diagnostic_results,
        "overall_verdict": overall,
        "failed_baselines": failed_baselines,
        "elapsed_seconds": round(elapsed, 1),
    }

    # ---- Print summary ----
    print(f"\n{'='*80}", file=sys.stderr)
    print("V3.1 DEVELOPMENT CORPUS LEAKAGE GATE RESULTS", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Corpus: {len(items)} items, {N_PER_REGIME} per regime",
          file=sys.stderr)
    print(f"Chance: {chance:.5f}  Threshold: {threshold:.5f}",
          file=sys.stderr)
    print(f"Surface-form checks: {'ALL PASS' if sf_all_pass else 'SOME FAIL'}",
          file=sys.stderr)

    print(f"\n--- Template-Held-Out Results (11 baselines) ---",
          file=sys.stderr)
    for name in BASELINE_NAMES:
        r = held_out_results[name]
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

    print(f"\n--- Diagnostic Baselines ---", file=sys.stderr)
    for name in ["D1_option_only", "D3_full_candidate"]:
        r = diagnostic_results[name]
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

    d2 = diagnostic_results["D2_context_regime"]
    print(f"  D2_context_regime (binary)      acc={d2['accuracy']:.4f}  "
          f"CI=[{d2['ci_lower']:.4f},{d2['ci_upper']:.4f}]  "
          f"ch=0.5000  {d2['verdict']}", file=sys.stderr)
    for regime, rr in d2.get("per_regime", {}).items():
        print(f"    {regime:15s}  acc={rr['accuracy']:.4f}  "
              f"CI=[{rr['ci_lower']:.4f},{rr['ci_upper']:.4f}]",
              file=sys.stderr)

    print(f"\nOVERALL VERDICT: {overall}", file=sys.stderr)
    if failed_baselines:
        print(f"Failed baselines: {failed_baselines}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # ---- Save JSON ----
    with open(RESULTS_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}", file=sys.stderr)

    # ---- Generate markdown report ----
    _write_markdown_report(report, REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}", file=sys.stderr)

    return report


def _write_markdown_report(report, path):
    """Write a human-readable markdown report."""
    cfg = report["corpus_config"]
    sf = report["surface_form_checks"]
    ho = report["template_held_out_results"]
    diag = report["diagnostic_results"]

    lines = []
    lines.append("# v3.1 Development-Gate Leakage Evaluation Report\n")
    lines.append(f"Generated: {report['evaluation_date']}\n")

    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Generator | T2V31Generator v{report['generator_version']} |")
    lines.append(f"| n_per_regime | {cfg['n_per_regime']} |")
    lines.append(f"| Seed | {cfg['seed']} |")
    lines.append(f"| Total items | {cfg['total_items']} |")
    lines.append(f"| Evidence per item | {cfg['n_evidence_per_item']} |")
    lines.append(f"| Hypotheses per item | {cfg['n_hypotheses_per_item']} |")
    lines.append(f"| Chance level | {cfg['chance_level']:.5f} |")
    lines.append(f"| Threshold | {cfg['threshold']:.5f} |")
    lines.append(f"| Corpus SHA-256 | `{cfg['corpus_sha256']}` |")
    lines.append("")

    lines.append("## Reproduction Command\n")
    lines.append("```bash")
    lines.append("cd llm-assisted-research")
    lines.append("python3 analysis/run_v31_dev_gate.py")
    lines.append("```\n")

    lines.append("## Surface-Form Check Results\n")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    for name, result in sf.items():
        status = "PASS" if result["passed"] else "**FAIL**"
        lines.append(f"| {name} | {status} |")
    lines.append("")

    lines.append("## Template-Held-Out Baseline Results (Aggregate)\n")
    lines.append("| Baseline | Accuracy | Wilson 95% CI | Verdict |")
    lines.append("|---|---|---|---|")
    for name in BASELINE_NAMES:
        r = ho[name]
        v = r["verdict"]
        if v == "FAIL":
            v = "**FAIL**"
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | "
            f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | {v} |"
        )
    lines.append("")

    # Per-regime for key baselines
    for bname in ["6_tfidf_word", "11_combined_shallow"]:
        if bname in ho:
            lines.append(f"### Per-Regime: {bname}\n")
            lines.append("| Regime | Accuracy | Wilson 95% CI | Chance | Verdict |")
            lines.append("|---|---|---|---|---|")
            r = ho[bname]
            for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
                if regime in r.get("per_regime", {}):
                    rr = r["per_regime"][regime]
                    v = "**FAIL**" if rr["verdict"] == "FAIL" else "PASS"
                    lines.append(
                        f"| {regime} | {rr['accuracy']:.4f} | "
                        f"[{rr['ci_lower']:.4f}, {rr['ci_upper']:.4f}] | "
                        f"{rr['chance']:.4f} | {v} |"
                    )
            lines.append("")

    lines.append("## Diagnostic Baseline Results\n")
    lines.append("### D1: Option-Only Baseline\n")
    d1 = diag["D1_option_only"]
    lines.append(f"- **Aggregate**: acc={d1['accuracy']:.4f}, "
                 f"CI=[{d1['ci_lower']:.4f}, {d1['ci_upper']:.4f}], "
                 f"verdict={d1['verdict']}")
    lines.append(f"- Measures: whether hypothesis text alone leaks the answer")
    lines.append("")
    lines.append("| Regime | Accuracy | Wilson 95% CI | Verdict |")
    lines.append("|---|---|---|---|")
    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
        if regime in d1.get("per_regime", {}):
            rr = d1["per_regime"][regime]
            v = "**FAIL**" if rr["verdict"] == "FAIL" else "PASS"
            lines.append(
                f"| {regime} | {rr['accuracy']:.4f} | "
                f"[{rr['ci_lower']:.4f}, {rr['ci_upper']:.4f}] | {v} |"
            )
    lines.append("")

    lines.append("### D2: Context-Regime Classifier\n")
    d2 = diag["D2_context_regime"]
    lines.append(f"- **Aggregate**: acc={d2['accuracy']:.4f}, "
                 f"CI=[{d2['ci_lower']:.4f}, {d2['ci_upper']:.4f}], "
                 f"chance=0.5000, verdict={d2['verdict']}")
    lines.append(f"- Measures: whether narrative+evidence alone reveals the regime")
    lines.append("")

    lines.append("### D3: Full Candidate-Aware TF-IDF\n")
    d3 = diag["D3_full_candidate"]
    lines.append(f"- **Aggregate**: acc={d3['accuracy']:.4f}, "
                 f"CI=[{d3['ci_lower']:.4f}, {d3['ci_upper']:.4f}], "
                 f"verdict={d3['verdict']}")
    lines.append("")
    lines.append("| Regime | Accuracy | Wilson 95% CI | Verdict |")
    lines.append("|---|---|---|---|")
    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
        if regime in d3.get("per_regime", {}):
            rr = d3["per_regime"][regime]
            v = "**FAIL**" if rr["verdict"] == "FAIL" else "PASS"
            lines.append(
                f"| {regime} | {rr['accuracy']:.4f} | "
                f"[{rr['ci_lower']:.4f}, {rr['ci_upper']:.4f}] | {v} |"
            )
    lines.append("")

    lines.append(f"## Overall Verdict: {report['overall_verdict']}\n")
    if report["failed_baselines"]:
        lines.append(f"Failed baselines: {report['failed_baselines']}\n")
    else:
        lines.append("No baselines failed the leakage gate.\n")
    lines.append(f"Elapsed: {report['elapsed_seconds']:.1f}s\n")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    main()
