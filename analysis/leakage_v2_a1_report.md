# Leakage Evaluation v2 Phase A.1: Candidate-Aware Baseline Report

**Date:** 2026-08-27
**Evaluator:** `analysis/run_leakage_eval.py` (Phase A.1, candidate-aware rewrite)
**Results:** `analysis/leakage_results_v2_a1.json`
**Comparison baseline:** `analysis/leakage_results_v2_corrected.json` (Phase A)

---

## 1. Summary of Changes from Phase A

### 1.1 Candidate-Aware Classifier Baselines (6-11)

**Problem:** Phase-A classifier baselines (6-11) predicted the gold OPTION INDEX
from item text (narrative + evidence) that **omitted hypothesis text entirely**.
This tested whether text correlates with answer position, not whether surface
cues identify the correct candidate. The classifiers were not permutation-
equivariant: permuting the option order and gold pointer would change predictions.

**Fix:** All classifier baselines now use **one row per item-candidate pair** with
target-vs-other feature normalization:

1. For each item with K hypotheses, emit K rows.
2. Each row's features combine the target candidate's hypothesis text with the
   item context (narrative + evidence).
3. For TF-IDF baselines: text = `"{hypothesis} [SEP] {narrative} [SEP] {evidence}"`.
4. For structured baselines: features = `[target_X, target_X - mean(others_X)]`
   for each raw feature X.
5. Train a binary classifier (label=1 for gold candidate, 0 for others).
6. At test time, select the candidate with the highest P(label=1) per item.

**Permutation equivariance:** Every candidate receives identical feature treatment
(same function, no position index). Permuting option order and gold pointer jointly
produces the same selected candidate.

### 1.2 Renamed/Repaired Baselines

| # | Old Name | New Name | Change |
|---|----------|----------|--------|
| 9 | polarity_feature | mention_evidence | Honest rename: features are mention-count and evidence-count, not polarity (polarity features were removed in Phase A) |
| 10 | positional_feature | first_mention_order | Replaced near-constant position indices {0,1,2,3} with genuine first-occurrence position in text (normalized by text length) |

### 1.3 Preserved from Phase A

- Heuristic baselines (1-5): unchanged (already candidate-aware via argmax)
- Chance = mean(1/n_options)
- Majority baseline: single canonical label from train
- Per-regime = slice of aggregate predictions (not refit)
- Hard reconciliation assertion
- Feature hygiene: no supports/contradicts/diagnostic_value

## 2. Corpus Statistics

| Corpus | N items | Chance level | Threshold (chance + 0.05) |
|--------|---------|--------------|---------------------------|
| Template-held-out (train) | 5,250 | 0.31944 | 0.36944 |
| Final-audit | 750 | 0.31944 | 0.36944 |

Regime distribution (train): CLEAN=1,750, DECOY=1,750, CONFLICT=875, INSUFFICIENT=875.

## 3. Template-Held-Out Results (Aggregate)

| # | Baseline | Phase-A Acc | A.1 Acc | 95% Wilson CI | Phase-A Verdict | A.1 Verdict | Changed? |
|---|----------|-------------|---------|---------------|-----------------|-------------|----------|
| 1 | Majority class | 0.2804 | 0.2804 | [0.2684, 0.2927] | PASS | PASS | No |
| 2 | Label position | 0.2773 | 0.2773 | [0.2654, 0.2896] | PASS | PASS | No |
| 3 | Mention count | 0.2773 | 0.2773 | [0.2654, 0.2896] | PASS | PASS | No |
| 4 | Evidence count | 0.2773 | 0.2773 | [0.2654, 0.2896] | PASS | PASS | No |
| 5 | Lexical overlap | 0.2773 | 0.2773 | [0.2654, 0.2896] | PASS | PASS | No |
| 6 | TF-IDF word | 0.4350 | 0.4465 | [0.4331, 0.4600] | FAIL | FAIL | No |
| 7 | TF-IDF char | 0.4333 | 0.4469 | [0.4335, 0.4603] | FAIL | FAIL | No |
| 8 | Length feature | 0.4451 | 0.4453 | [0.4319, 0.4588] | FAIL | FAIL | No |
| 9 | Mention evidence (was: polarity) | 0.4421 | 0.4440 | [0.4306, 0.4575] | FAIL | FAIL | No |
| 10 | First mention order (was: positional) | 0.4438 | 0.3465 | [0.3337, 0.3595] | FAIL | **PASS** | **YES** |
| 11 | Combined shallow | 0.4480 | 0.4556 | [0.4422, 0.4691] | FAIL | FAIL | No |

**Verdict change:** Baseline 10 flipped from FAIL to PASS. The old "positional"
baseline used near-constant hypothesis index features, which incidentally learned
to predict the fixed answer position for INSUFFICIENT items. The new
"first_mention_order" baseline uses genuine text-ordering features and shows no
exploitable leakage at the aggregate level.

## 4. Per-Regime Results (Template-Held-Out)

### CLEAN (chance = 0.3333, threshold = 0.3833)

| # | Baseline | Accuracy | CI Upper | Verdict |
|---|----------|----------|----------|---------|
| 1 | Majority class | 0.3320 | 0.3544 | PASS |
| 2 | Label position | 0.3297 | 0.3521 | PASS |
| 3 | Mention count | 0.3297 | 0.3521 | PASS |
| 4 | Evidence count | 0.3297 | 0.3521 | PASS |
| 5 | Lexical overlap | 0.3297 | 0.3521 | PASS |
| 6 | TF-IDF word | 0.3463 | 0.3689 | PASS |
| 7 | TF-IDF char | 0.3331 | 0.3556 | PASS |
| 8 | Length feature | 0.3171 | 0.3393 | PASS |
| 9 | Mention evidence | 0.3297 | 0.3521 | PASS |
| 10 | First mention order | 0.3206 | 0.3428 | PASS |
| 11 | Combined shallow | 0.3434 | 0.3660 | PASS |

**All 11 baselines PASS.**

### DECOY (chance = 0.3333, threshold = 0.3833)

| # | Baseline | Accuracy | CI Upper | Verdict |
|---|----------|----------|----------|---------|
| 1 | Majority class | 0.3400 | 0.3625 | PASS |
| 2 | Label position | 0.3286 | 0.3509 | PASS |
| 3 | Mention count | 0.3286 | 0.3509 | PASS |
| 4 | Evidence count | 0.3286 | 0.3509 | PASS |
| 5 | Lexical overlap | 0.3286 | 0.3509 | PASS |
| 6 | TF-IDF word | 0.3223 | 0.3446 | PASS |
| 7 | TF-IDF char | 0.3429 | 0.3654 | PASS |
| 8 | Length feature | 0.3531 | 0.3758 | PASS |
| 9 | Mention evidence | 0.3286 | 0.3509 | PASS |
| 10 | First mention order | 0.3143 | 0.3364 | PASS |
| 11 | Combined shallow | 0.3543 | 0.3770 | PASS |

**All 11 baselines PASS.**

### CONFLICT (chance = 0.3333, threshold = 0.3833)

| # | Baseline | Accuracy | CI Upper | Verdict |
|---|----------|----------|----------|---------|
| 1 | Majority class | 0.3383 | 0.3703 | PASS |
| 2 | Label position | 0.3474 | 0.3796 | PASS |
| 3 | Mention count | 0.3474 | 0.3796 | PASS |
| 4 | Evidence count | 0.3474 | 0.3796 | PASS |
| 5 | Lexical overlap | 0.3474 | 0.3796 | PASS |
| 6 | TF-IDF word | 0.3417 | 0.3738 | PASS |
| 7 | TF-IDF char | 0.3291 | 0.3610 | PASS |
| 8 | Length feature | 0.3314 | 0.3633 | PASS |
| 9 | Mention evidence | 0.3474 | 0.3796 | PASS |
| 10 | First mention order | 0.3429 | 0.3749 | PASS |
| 11 | Combined shallow | 0.3383 | 0.3703 | PASS |

**All 11 baselines PASS.** Note: Phase A had 2 marginal CONFLICT failures
(8_length, 11_combined) before feature hygiene fix. With candidate-aware
baselines, all CONFLICT baselines comfortably PASS.

### INSUFFICIENT (chance = 0.2500, threshold = 0.3000)

| # | Baseline | Accuracy | CI Upper | Verdict |
|---|----------|----------|----------|---------|
| 1 | Majority class | 0.0000 | 0.0044 | PASS |
| 2 | Label position | 0.0000 | 0.0044 | PASS |
| 3 | Mention count | 0.0000 | 0.0044 | PASS |
| 4 | Evidence count | 0.0000 | 0.0044 | PASS |
| 5 | Lexical overlap | 0.0000 | 0.0044 | PASS |
| 6 | TF-IDF word | 1.0000 | 1.0000 | **FAIL** |
| 7 | TF-IDF char | 1.0000 | 1.0000 | **FAIL** |
| 8 | Length feature | 1.0000 | 1.0000 | **FAIL** |
| 9 | Mention evidence | 1.0000 | 1.0000 | **FAIL** |
| 10 | First mention order | 0.4663 | 0.4994 | **FAIL** |
| 11 | Combined shallow | 1.0000 | 1.0000 | **FAIL** |

**Structural leak persists.** The "Cannot be determined from available evidence"
option text is categorically different from suspect hypotheses, enabling any
text-aware classifier to identify it with 100% accuracy. Baseline 10
(first_mention_order) achieves 46.6% — the abstention option has no "name"
to find in text, so it gets a distinctive first_mention_pos=1.0, which the
classifier exploits.

**PASS count: 5/11** (heuristic baselines pass because they look for suspect
names and find no match for "Cannot be determined").

## 5. Final-Audit Results

### Aggregate

| # | Baseline | Accuracy | 95% Wilson CI | Verdict |
|---|----------|----------|---------------|---------|
| 1 | Majority class | 0.2787 | [0.2478, 0.3118] | PASS |
| 2 | Label position | 0.2760 | [0.2452, 0.3091] | PASS |
| 3 | Mention count | 0.2760 | [0.2452, 0.3091] | PASS |
| 4 | Evidence count | 0.2760 | [0.2452, 0.3091] | PASS |
| 5 | Lexical overlap | 0.2760 | [0.2452, 0.3091] | PASS |
| 6 | TF-IDF word | 0.4373 | [0.4022, 0.4731] | FAIL |
| 7 | TF-IDF char | 0.4307 | [0.3957, 0.4664] | FAIL |
| 8 | Length feature | 0.4453 | [0.4101, 0.4811] | FAIL |
| 9 | Mention evidence | 0.4427 | [0.4075, 0.4784] | FAIL |
| 10 | First mention order | 0.3880 | [0.3538, 0.4234] | FAIL |
| 11 | Combined shallow | 0.4387 | [0.4036, 0.4744] | FAIL |

### Per-Regime PASS Counts (Final-Audit)

| Regime | N items | PASS count | Notes |
|--------|---------|------------|-------|
| CLEAN | 250 | 2/11 | CIs wide at N=250 |
| DECOY | 250 | 3/11 | CIs wide at N=250 |
| CONFLICT | 125 | 1/11 | CIs wide at N=125; all failures from CI width |
| INSUFFICIENT | 125 | 5/11 | Structural leak for classifiers (100%) |

The audit per-regime failures are dominated by statistical power issues
(per-regime N too small for equivalence testing at chance+0.05 margin),
consistent with the Phase A findings. Point estimates for CLEAN/DECOY/CONFLICT
are at or below chance, but Wilson CIs at N=125-250 are too wide to demonstrate
equivalence.

## 6. Overall Verdict

**FAIL.** Same root cause as Phase A: INSUFFICIENT structural leak.

## 7. Diagnosis

### Confirmed: No Surface-Cue Leakage in CLEAN, DECOY, CONFLICT

On the template-held-out split (N=5,250), **all 11 baselines PASS for CLEAN,
DECOY, and CONFLICT** with the candidate-aware evaluator. This is a stronger
result than Phase A, where the non-candidate-aware evaluator also showed
all-PASS for these regimes — but the Phase-A result was potentially confounded
by the fact that classifiers were predicting position indices rather than
identifying candidates. The candidate-aware result confirms that no surface
cue (lexical, structural, or positional) identifies the correct candidate
beyond chance in the three non-INSUFFICIENT regimes.

### Confirmed: INSUFFICIENT Structural Leak Requires v3 Fix

The INSUFFICIENT structural leak is a format-level issue (unique option text
"Cannot be determined from available evidence" only appears in INSUFFICIENT
items) that cannot be fixed by feature engineering in the evaluator. It requires
the v3 generator design (universal 4-option format with abstention option in
every regime, per AMENDMENT-002 section 2.1).

### Baseline 10 Verdict Change Explained

The old "positional" baseline (Phase A) used hypothesis slot indices {0,1,2,3}
as features. Since INSUFFICIENT items always have 4 options while other regimes
have 3, the classifier learned that the 4th slot being non-zero correlates with
predicting index 3 (the "Cannot be determined" option). This was an artifact
of position-index features, not genuine positional leakage in the text.

The new "first_mention_order" baseline uses the actual character position of each
candidate's name in the text. Since "Cannot be determined" contains no suspect
name, its first_mention_pos defaults to 1.0 (not found), which is still
exploitable (INSUFFICIENT FAIL at 46.6%) but far less than 100%. On CLEAN/DECOY/
CONFLICT (where all candidates are suspect names), first_mention_order shows
no leakage (all PASS).

## 8. Reconciliation Verification

All 22 baseline/split combinations pass the hard reconciliation assertion:
`sum(regime_correct) == aggregate_correct`.

## 9. Relationship to Phase A Results

Phase A results (`analysis/leakage_results_v2_corrected.json`) are preserved
unchanged. This report presents the candidate-aware rewrite results
(`analysis/leakage_results_v2_a1.json`) alongside the Phase A results for
comparison. The key improvement is that the candidate-aware baselines correctly
test whether surface cues identify the correct candidate (not just whether
text correlates with answer position), providing a more rigorous leakage test
for CLEAN/DECOY/CONFLICT regimes.
