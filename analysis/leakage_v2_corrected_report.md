# Leakage Evaluation v2: Old vs Corrected Comparison Report

**Date:** 2026-08-27
**Evaluator:** `analysis/run_leakage_eval.py` (corrected version)
**Old results:** `analysis/leakage_results_v2.json` (uncorrected, preserved)
**Corrected results:** `analysis/leakage_results_v2_corrected.json`

---

## 1. Summary of Corrections Applied

| # | Defect | Fix |
|---|--------|-----|
| a | Per-regime verdicts missing on audit split; aggregate-only verdict on held-out | Full per-regime + aggregate verdicts on BOTH splits |
| b | Per-regime classifiers were REFIT per regime, causing agg/regime mismatch | ONE aggregate model, per-regime by SLICING predictions; hard reconciliation assertion |
| c | Majority baseline: recomputed per-regime giving 875 agg vs 1387 regime sum | ONE canonical predicted label from train set, applied identically to every item |
| d | Chance = 1/mean(n_options) = 0.31579 | Chance = mean(1/n_options) = 0.31944 |
| e | Structured features used `supports`, `contradicts`, `diagnostic_value` (generator-only metadata) | Features restricted to model-visible fields: narrative, hypotheses, evidence[].content |

## 2. Chance Level Change

| Method | Value | Threshold (chance + 0.05) |
|--------|-------|---------------------------|
| Old: 1/mean(n_options) | 0.31579 | 0.36579 |
| Corrected: mean(1/n_options) | 0.31944 | 0.36944 |

The difference is +0.00365. Both formulations give the same per-regime chance for
homogeneous-option regimes (0.3333 for 3-option, 0.2500 for 4-option).

## 3. Majority Baseline Fix

### Held-Out Split

| Metric | Old (v2) | Corrected |
|--------|----------|-----------|
| Aggregate n_correct | 875 | 1,472 |
| Aggregate accuracy | 0.167 | 0.280 |
| CLEAN n_correct | 201 | 581 |
| DECOY n_correct | 201 | 595 |
| CONFLICT n_correct | 110 | 296 |
| INSUFFICIENT n_correct | 875 | 0 |
| **Sum of regime correct** | **1,387** | **1,472** |
| **Aggregate correct** | **875** | **1,472** |
| **Reconciles?** | **NO (off by 512)** | **YES** |

**Root cause:** The old evaluator recomputed `bl_majority` independently per regime
slice, finding a different majority label per regime. The INSUFFICIENT regime's
majority was "Cannot be determined..." (all 875 items correct), while CLEAN/DECOY/CONFLICT
had different majority labels, each yielding ~11-13% accuracy. The aggregate
used the overall majority label (which happened to be "Cannot be determined...",
matching only INSUFFICIENT), giving 875. The regime sum was 1,387.

**Fix:** The corrected evaluator determines ONE majority label from the train set's
gold-index mode (index 0), and applies it identically to every item. The majority
label is a suspect hypothesis index, so it gets ~33% on 3-option regimes (CLEAN/DECOY/CONFLICT)
and 0% on INSUFFICIENT (where gold is always index 3). The sum reconciles exactly.

### Final-Audit Split

| Metric | Old (v2) | Corrected |
|--------|----------|-----------|
| Aggregate n_correct | 125 | 209 |
| Old regime sum | 198 | 209 |
| **Reconciles?** | **NO** | **YES** |

## 4. Aggregate Verdicts Comparison (Both Splits)

### Template-Held-Out (5,250 items)

| Baseline | Old Acc | Old Verdict | Corrected Acc | Corrected Verdict | Changed? |
|----------|---------|-------------|---------------|-------------------|----------|
| 1_majority_class | 0.167 | PASS | 0.280 | PASS | Acc changed (definition fix) |
| 2_label_position | 0.277 | PASS | 0.277 | PASS | No |
| 3_mention_count | 0.277 | PASS | 0.277 | PASS | No |
| 4_evidence_count | 0.277 | PASS | 0.277 | PASS | No |
| 5_lexical_overlap | 0.277 | PASS | 0.277 | PASS | No |
| 6_tfidf_word | 0.435 | FAIL | 0.435 | FAIL | No |
| 7_tfidf_char | 0.433 | FAIL | 0.433 | FAIL | No |
| 8_length_feature | 0.445 | FAIL | 0.445 | FAIL | No |
| 9_polarity_feature | 0.442 | FAIL | 0.442 | FAIL | No |
| 10_positional_feature | 0.448 | FAIL | 0.444 | FAIL | Acc changed (feature hygiene) |
| 11_combined_shallow | 0.447 | FAIL | 0.448 | FAIL | Acc changed (feature hygiene) |

### Final-Audit (750 items)

| Baseline | Old Acc | Old Verdict | Corrected Acc | Corrected Verdict | Changed? |
|----------|---------|-------------|---------------|-------------------|----------|
| 1_majority_class | 0.167 | PASS | 0.279 | PASS | Acc changed |
| 2_label_position | 0.276 | PASS | 0.276 | PASS | No |
| 3_mention_count | 0.276 | PASS | 0.276 | PASS | No |
| 4_evidence_count | 0.276 | PASS | 0.276 | PASS | No |
| 5_lexical_overlap | 0.276 | PASS | 0.276 | PASS | No |
| 6_tfidf_word | 0.448 | FAIL | 0.448 | FAIL | No |
| 7_tfidf_char | 0.447 | FAIL | 0.447 | FAIL | No |
| 8_length_feature | 0.461 | FAIL | 0.461 | FAIL | No |
| 9_polarity_feature | 0.451 | FAIL | 0.451 | FAIL | No |
| 10_positional_feature | 0.443 | FAIL | 0.445 | FAIL | Acc changed |
| 11_combined_shallow | 0.439 | FAIL | 0.469 | FAIL | Acc changed |

**Overall verdict unchanged: FAIL on both splits.**

## 5. Per-Regime Verdicts: Held-Out Split

### Corrected Per-Regime PASS Counts

| Regime | Old | Corrected | Change |
|--------|-----|-----------|--------|
| CLEAN | 11/11 | 11/11 | Same |
| DECOY | 11/11 | 11/11 | Same |
| CONFLICT | 9/11 | 11/11 | +2 (length, combined now PASS after feature hygiene fix) |
| INSUFFICIENT | 4/11 | 5/11 | +1 (majority now correctly predicts 0% on INSUFFICIENT) |

### Specific Verdict Changes (Held-Out)

| Baseline | Regime | Old Verdict | Corrected Verdict | Explanation |
|----------|--------|-------------|-------------------|-------------|
| 1_majority | INSUFFICIENT | FAIL (1.00) | PASS (0.00) | Old majority was "Cannot be determined" (matching all INSUFFICIENT); corrected majority is suspect index 0 (matching no INSUFFICIENT) |
| 8_length | CONFLICT | FAIL (0.359, CI upper 0.391) | PASS (0.343, CI upper 0.375) | Feature hygiene: removing generator-internal supports/contradicts changed feature values, lowering CONFLICT accuracy below threshold |
| 11_combined | CONFLICT | FAIL (0.359, CI upper 0.391) | PASS (0.351, CI upper 0.383) | Same: feature hygiene fix |

## 6. Per-Regime Verdicts: Final-Audit Split

**IMPORTANT: The old v2 evaluator did not compute per-regime verdicts on the
final-audit split. These are entirely new.**

### Corrected Final-Audit Per-Regime PASS Counts

| Regime | PASS count | FAIL count | Notes |
|--------|------------|------------|-------|
| CLEAN | 5/11 | 6/11 | CIs wide due to small N (250 items) |
| DECOY | 4/11 | 7/11 | CIs wide due to small N (250 items) |
| CONFLICT | 0/11 | 11/11 | All baselines have CI upper > threshold (N=125) |
| INSUFFICIENT | 5/11 | 6/11 | Classifiers still 100% (structural leak), heuristics near 0% |

### Detailed Final-Audit Per-Regime Verdicts

| Baseline | CLEAN | DECOY | CONFLICT | INSUFFICIENT |
|----------|-------|-------|----------|--------------|
| 1_majority | PASS | FAIL | FAIL | PASS |
| 2_label_position | FAIL | PASS | FAIL | PASS |
| 3_mention_count | FAIL | PASS | FAIL | PASS |
| 4_evidence_count | FAIL | PASS | FAIL | PASS |
| 5_lexical_overlap | FAIL | PASS | FAIL | PASS |
| 6_tfidf_word | FAIL | FAIL | FAIL | FAIL |
| 7_tfidf_char | PASS | FAIL | FAIL | FAIL |
| 8_length_feature | PASS | FAIL | FAIL | FAIL |
| 9_polarity_feature | FAIL | FAIL | FAIL | FAIL |
| 10_positional | PASS | FAIL | FAIL | FAIL |
| 11_combined | PASS | FAIL | FAIL | FAIL |

### Interpretation of Wide CIs on Final-Audit

The per-regime Ns on the final audit are small (CLEAN=250, DECOY=250,
CONFLICT=125, INSUFFICIENT=125). Wilson CIs are correspondingly wide. For
example, a heuristic baseline scoring 87/250 = 0.348 on CLEAN has CI upper
~0.409, which exceeds the threshold of 0.383 even though the point estimate
is below the 0.333 chance level. This is a statistical power issue (N too small
for equivalence testing at this margin), not evidence of true leakage.

**CONFLICT is particularly affected:** All 11 baselines fail on CONFLICT in the
audit because N=125 produces CIs too wide to demonstrate equivalence, even when
point estimates are at or below chance.

## 7. Retraction of Prior "CLEAN/DECOY Pass All 11" Claim

The AMENDMENT-001 POST-OUTCOME section (section 6.7) stated:

> "For CLEAN and DECOY regimes, T2 v2 passes all 11 baselines."

This claim is **retracted** for the final-audit split. The corrected per-regime
verdicts show:
- CLEAN: 5/11 PASS (6 fail due to wide CIs, not due to elevated accuracy)
- DECOY: 4/11 PASS (7 fail due to wide CIs)

The held-out split retains CLEAN 11/11 and DECOY 11/11 with the corrected
evaluator, so the claim holds for the larger held-out split but not for the
small final-audit split.

## 8. Reconciliation Verification

All 22 baseline/split combinations (11 baselines x 2 splits) pass the hard
reconciliation assertion: `sum(regime_correct) == aggregate_correct`.

The old v2 evaluator had widespread reconciliation failures:
- Majority held-out: 1,387 vs 875 (off by 512)
- Multiple classifier baselines off by 15-75 counts

## 9. Feature Hygiene Changes

The old v2 evaluator's `_build_structured_features` function used:
- `evidence[].supports` (generator annotation: which suspect each evidence supports)
- `evidence[].contradicts` (generator annotation: which suspect each evidence contradicts)
- `evidence[].diagnostic_value` (generator annotation)

These fields are invisible to the model being evaluated. The corrected evaluator
uses only model-visible fields: `narrative`, `hypotheses`, `evidence[].content`.

Impact: baselines 8-11 (structured features) changed slightly. The CONFLICT
marginal failures (8_length, 11_combined) on the held-out split flipped from
FAIL to PASS after removing the generator-internal polarity features.
