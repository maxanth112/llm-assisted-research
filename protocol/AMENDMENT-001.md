# Protocol Amendment 001: T2 Leakage Repair

**Amendment ID:** AMENDMENT-001
**Date filed:** 2026-08-23
**Amends:** T2 Diagnostic Generator (datasets/t2_generator/generator.py)
**Status:** PRE-OUTCOME (criteria specified before evaluating repaired generator)

---

## 1. Observed Leakage

The Stage 4A leakage battery revealed that shallow classifiers recover the
gold answer from surface cues at rates far exceeding chance (~0.235 for 4-hypothesis
items with 20% INSUFFICIENT items):

| Baseline | Accuracy | vs Chance |
|----------|----------|-----------|
| Lexical overlap (name mention-count) | ~0.750 | +0.515 |
| BoW TF-IDF Logistic Regression (CV) | ~0.812 | +0.577 |
| Majority class | ~0.250 | +0.015 (near chance) |

The lexical overlap heuristic succeeds because the guilty suspect's name appears
more frequently in incriminating evidence items (fingerprints, access logs, camera
footage) than innocent suspects' names. The TF-IDF classifier exploits both name
frequency and template-specific vocabulary patterns (e.g., evidence types
that only appear in guilty-pointing clauses).

## 2. Why This Invalidates T2 v1

The purpose of T2 is to test whether ACH scaffolding improves *relational reasoning*
— i.e., the ability to synthesize evidence across hypotheses and identify which
hypothesis is most (or least) consistent. If the answer is recoverable from
mention-frequency or lexical overlap alone:

1. **Any measured scaffold "gain" is confounded.** A model could improve not because
   the scaffold helps it reason about consistency/inconsistency, but because the
   scaffold causes it to attend more to high-mention-count names.
2. **The experiment cannot distinguish reasoning from bookkeeping.** The fundamental
   claim that ACH adds value over enumeration is untestable if the answer is trivially
   recoverable without ACH.
3. **Surface cues make the within-item design meaningless.** If the answer is the same
   under all conditions (because it's determined by name frequency, not reasoning),
   condition comparisons measure prompt-format effects on name-extraction, not on
   analytical depth.

**Verdict:** T2 v1 is invalidated. The generator must be rebuilt so the correct
answer is determined solely by relational structure among clues.

## 3. Generator Changes (T2 v2)

### 3.1 Core Design: Relational Reasoning via Structural Rules

In T2 v2, the correct answer is determined by a **relational rule** — a logical
pattern that requires combining multiple evidence items, not just counting mentions.

**Relational rule types used:**
- **Alibi-chain invalidation:** All suspects receive equally incriminating evidence,
  but only the guilty suspect's alibi is invalidated by a cross-reference between
  two other clues. Solving requires noticing the contradiction.
- **Temporal-sequence inconsistency:** Evidence mentions times; the guilty suspect is
  the one whose timeline is internally inconsistent (e.g., claimed to be in location X
  at time T, but another clue places them in location Y at time T).
- **Source-credibility cascade (CONFLICT regime):** Two sources disagree; a precedence
  rule (e.g., "forensic evidence > eyewitness testimony") determines which to trust.
  All suspects are mentioned equally; the relational rule picks the answer.

### 3.2 Counterbalancing Invariants

The v2 generator enforces the following for every generated item:

1. **Name-frequency equalization:** Each suspect name appears the same number of times
   (±1) across all evidence items and the narrative. Achieved by giving every suspect
   exactly one incriminating-looking evidence item and one exonerating-looking item.
2. **Evidence-count parity:** Each suspect is mentioned in exactly the same number of
   evidence items.
3. **Clue-polarity balance:** Each suspect receives the same number of incriminating
   and exonerating statements. The guilty determination comes from a *relational*
   inconsistency, not a polarity imbalance.
4. **Length matching:** Evidence items are padded/trimmed to a target length range so
   no suspect's evidence is distinguishable by character count.
5. **Positional uniformity:** The correct-answer position in the hypothesis list is
   uniformly distributed across items (verified over the full dataset).
6. **Entity-name randomization:** Names, locations, and dates are drawn randomly per
   item from pools, with no correlation to guilt.

### 3.3 Counterfactual Minimal Pairs

For each base item, the generator produces a **counterfactual twin** where:
- The same suspect names, evidence vocabulary, and template are used.
- A small relational change (e.g., swapping which alibi gets invalidated) flips
  the correct answer to a *different* suspect.
- Token inventories are matched: the pair shares >95% of tokens; only the relational
  structure differs.

This is the primary anti-leakage device. If tokens match but the answer differs,
no bag-of-words classifier can succeed.

### 3.4 Regime Definitions (preserved from v1)

- **CLEAN:** Relational rule uniquely identifies one suspect. All others have
  valid alibis that survive cross-referencing.
- **DECOY:** Same as CLEAN, plus additional salient-but-non-diagnostic decoy evidence
  for innocent suspects. Decoys are equally salient as the diagnostic evidence.
- **CONFLICT:** Two evidence sources disagree. A stated precedence rule (included in
  the narrative) determines the answer. Both conflicting suspects are mentioned equally.
- **INSUFFICIENT:** No relational rule resolves the evidence. All suspects are
  symmetrically positioned. Gold answer = "Cannot be determined."

### 3.5 Template Families

Items are generated from **8 template families** (theft, sabotage, data_breach,
contamination, each with 2 relational-rule variants). Template-held-out evaluation
ensures classifiers cannot memorize template-specific patterns.

## 4. Acceptance Criteria (PRE-SPECIFIED)

**These criteria are defined BEFORE evaluating the repaired generator output.**
They constitute the pass/fail gate for T2 v2.

### 4.1 Evaluation Corpus

- Generate N ≥ 800 items (≥200 per regime) for the leakage evaluation corpus.
- Use template-held-out splitting: train classifiers on items from template families
  A and test on items from held-out template family B. This prevents template
  memorization.
- Reserve one template family as a FINAL AUDIT set, evaluated exactly once.

### 4.2 Shallow Baselines Evaluated

All of the following must be evaluated:
1. Majority-class baseline
2. Label-position baseline (predict by answer position)
3. Candidate mention-count heuristic
4. Evidence-count heuristic (per-candidate)
5. Lexical-overlap heuristic (answer vs evidence overlap)
6. Word-level TF-IDF classifier
7. Character-level TF-IDF classifier
8. Length-feature classifier (evidence length per candidate)
9. Polarity-feature classifier (incriminating/exonerating count per candidate)
10. Positional-feature classifier (evidence order features)
11. Combined shallow-feature classifier (all features together)

### 4.3 Pass Criteria

**Definition of "no reliable, practically meaningful advantage over chance":**

For EACH baseline, on BOTH the template-held-out test set AND the final-audit set:

- The 95% Wilson confidence interval upper bound for accuracy must be **≤ chance + 0.05**
  (i.e., within 5 percentage points of chance level).
- Chance level = 1 / (average number of hypotheses per item in that split).

**Additionally:**
- Results must be reported broken down by regime (not only aggregate).
- If ANY baseline exceeds the threshold on the template-held-out test set, the
  specific leaking feature must be diagnosed and the generator fixed. Iteration
  is permitted ONLY against the held-out test set, NOT the final-audit set.
- The final-audit set is evaluated exactly ONCE. If it fails, the failure is
  reported honestly; no further tuning is permitted without filing AMENDMENT-002.

### 4.4 Counterbalancing Invariants (regression-tested)

The following must hold over any generated batch of ≥100 items:
- Name frequency per suspect: coefficient of variation (CV) ≤ 0.10
- Evidence count per suspect: exactly equal within each item
- Incriminating/exonerating polarity: balanced within each item (±1)
- Correct-answer position: chi-squared test p > 0.05 (uniform)
- Counterfactual pair token overlap: Jaccard similarity ≥ 0.85

## 5. Version Tracking

| Field | Old (T2 v1) | New (T2 v2) |
|-------|-------------|-------------|
| Generator file | datasets/t2_generator/generator.py | datasets/t2_generator/generator.py |
| Version | v1 | v2 |
| SHA-256 (v1, frozen) | d158fd585fb65b4ade316d26e9eafabd62176c61e2b0ab29f50a2984ac1b623b | (preserved in PROTOCOL.lock.json) |
| SHA-256 (v2) | f6cc03f405dd84794eb24314c01d696bbb9fa73dfac163558d0e1d1c5c12c62c | |

The old hash is **preserved** in PROTOCOL.lock.json (marked as superseded).
The new hash is **added** alongside it.

---

**Filed by:** Automated Stage 4B build
**Governance note:** This amendment documents a pre-outcome protocol change.
The acceptance criteria (§4) were specified before the repaired generator was
evaluated. If the final-audit set fails, the failure will be reported without
further tuning (per §4.3).
