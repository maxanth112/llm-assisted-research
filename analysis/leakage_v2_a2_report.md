# Phase A.2 Leakage Evaluation Report

## Summary

Phase A.2 corrects two methodological issues in Phase A.1:

1. **TF-IDF baselines were not testing candidate-context leakage** (WI1):
   Phase A.1 concatenated `hypothesis [SEP] SAME narrative [SEP] SAME evidence`
   for every candidate row.  For a bag-of-words model, the shared context
   cancels during within-item argmax — the classifier could only learn
   candidate NAME priors.  Phase A.2 introduces TARGET normalization:
   replace each candidate's name with `TARGET`, others with `OTHER_1`,
   `OTHER_2`, etc.  Context now genuinely differs across candidates.

2. **Surface-form checks used chi-squared / KS non-rejection** (WI2):
   "Fails to reject non-uniformity" ≠ proof of balance.  Replaced with
   deterministic / constructive criteria.

## CRITICAL ASSESSMENT

**The Phase A.1 "CLEAN/DECOY/CONFLICT 11/11 PASS" result was PROVISIONAL**
because the TF-IDF baselines (6, 7) were not testing candidate-context
leakage — they could only learn candidate-name priors.

**Phase A.2 corrected baselines CONFIRM the prior conclusion:**
CLEAN/DECOY/CONFLICT remain 11/11 PASS on held-out.  The TARGET-normalized
TF-IDF baselines now genuinely test whether context implicates a specific
candidate via surface cues, and they still pass.  This is a **stronger**
confirmation than Phase A.1.

## Results Comparison: Phase A.1 vs Phase A.2

### Held-Out Evaluation (template-held-out cross-validation)

| Baseline | A.1 Agg | A.2 Agg | A.1 CLEAN | A.2 CLEAN | A.1 DECOY | A.2 DECOY | A.1 CONFLICT | A.2 CONFLICT | A.1 INSUF | A.2 INSUF |
|----------|---------|---------|-----------|-----------|-----------|-----------|--------------|--------------|-----------|-----------|
| 1_majority | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 2_position | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3_mention_count | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 4_evidence_count | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 5_lexical_overlap | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 6_tfidf_word | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |
| 7_tfidf_char | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |
| 8_length | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |
| 9_mention_ev | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |
| 10_first_mention | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |
| 11_combined | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |

**Held-out conclusion:** CLEAN/DECOY/CONFLICT = 11/11 PASS (unchanged).
INSUFFICIENT = same structural leak (100% for baselines 6-9, 11; ~47% for
baseline 10).  Overall FAIL driven entirely by INSUFFICIENT.

### What Changed

| Change | Phase A.1 | Phase A.2 |
|--------|-----------|-----------|
| TF-IDF text | `hyp [SEP] same_context` | `TARGET is responsible [SEP] TARGET_normalized_context` |
| Structured features | Count raw name in raw text | Count "TARGET" in normalized text |
| test_tfidf_detects_leakage | Called `pred_mention_count` (wrong) | Calls `pred_tfidf_word` and `pred_tfidf_char` (correct) |
| S2/S6 checks | chi-squared p>0.05 | counts differ by ≤1 |
| S4 check | KS test p>0.05 | identical sorted multiset |
| S5 check | KS test p>0.05 | mean within ±20% relative band |

### Injected-Leak Test Results

The test suite now verifies that the ACTUAL TF-IDF predictors detect injected leakage:

| Test | Predictor | Leaked Corpus Accuracy | Chance | Status |
|------|-----------|----------------------|--------|--------|
| test_tfidf_word_detects_leakage | pred_tfidf_word | >0.50 | 0.333 | PASS |
| test_tfidf_char_detects_leakage | pred_tfidf_char | >0.50 | 0.333 | PASS |

### Surface-Form Check Results (v2 corpus)

| Check | Train | Audit | Note |
|-------|-------|-------|------|
| S1 option_count | FAIL | FAIL | v2 has 3-option items (expected, needs v3) |
| S2 abstention_position | FAIL | FAIL | No abstention in CLEAN/DECOY/CONFLICT |
| S3 abstention_presence | FAIL | FAIL | Only INSUFFICIENT has abstention |
| S4 evidence_count | FAIL | FAIL | Regimes differ in evidence count |
| S5 option_text_length | PASS | PASS | Hypothesis lengths comparable |
| S6 gold_position | FAIL | FAIL | Only 3 positions in non-INSUFFICIENT |

These v2 corpus surface-form failures are expected and known — they reflect
v2's non-universal 4-option design.  The v3 generator (Phase B) must fix all
of S1-S6 by construction.

### Reconciliation

Hard assertion `sum(regime_correct) == aggregate_correct` passes for all
baselines on both splits.

## Overall Verdict

**FAIL** — driven entirely by INSUFFICIENT structural leak (4th option
identifiable by option count, abstention text presence).

**CLEAN/DECOY/CONFLICT: 11/11 PASS on held-out** — confirmed with corrected
TARGET-normalized TF-IDF baselines.  This is a stronger result than Phase A.1
because the TF-IDF classifiers now genuinely test candidate-context leakage.
