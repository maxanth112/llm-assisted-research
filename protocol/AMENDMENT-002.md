# Protocol Amendment 002: T2 v3 Design and Evaluator Corrections

**Amendment ID:** AMENDMENT-002
**Date filed:** 2026-08-27
**Amends:** T2 Diagnostic Generator, Leakage Evaluator, Statistical Design
**Status:** PRE-OUTCOME (criteria specified before generating or evaluating T2 v3)
**Prerequisite:** AMENDMENT-001 (T2 v2 leakage repair, overall verdict: FAIL)

---

## 1. Motivation

### 1.1 Corrected v2 Leakage Results

The v2 leakage evaluator (`analysis/run_leakage_eval.py`, original version)
contained five defects that produced incorrect per-regime verdicts and
non-reconciling counts. The corrected evaluator (same file, updated version)
fixes all five defects. Full comparison: `analysis/leakage_v2_corrected_report.md`.

**Defects corrected:**

| # | Defect | Impact |
|---|--------|--------|
| a | Per-regime verdicts missing on audit split | Audit per-regime status was unknown |
| b | Per-regime classifiers refit per regime | Aggregate/regime counts didn't reconcile |
| c | Majority baseline recomputed per regime | 875 agg vs 1,387 regime sum (off by 512) |
| d | Chance = 1/mean(n_options) instead of mean(1/n_options) | Threshold off by 0.00365 |
| e | Features used generator-internal `supports`/`contradicts`/`diagnostic_value` | Classifier saw information invisible to evaluated model |

**Corrected v2 per-regime pass counts (held-out / final-audit):**

| Regime | Held-out PASS | Final-audit PASS |
|--------|---------------|------------------|
| CLEAN | 11/11 | 5/11 |
| DECOY | 11/11 | 4/11 |
| CONFLICT | 11/11 | 0/11 |
| INSUFFICIENT | 5/11 | 5/11 |

### 1.2 Retraction of Prior Claims

**Retracted claim (AMENDMENT-001 section 6.7):**

> "For CLEAN and DECOY regimes, T2 v2 passes all 11 baselines."

This claim was outcome-dependent and is retracted for the final-audit split.
The corrected evaluator shows:
- CLEAN final-audit: 5/11 PASS (6 fail due to wide Wilson CIs at N=250)
- DECOY final-audit: 4/11 PASS (7 fail due to wide Wilson CIs at N=250)
- CONFLICT final-audit: 0/11 PASS (all fail; N=125 too small for equivalence)

The held-out split (N=5,250) retains CLEAN 11/11, DECOY 11/11, CONFLICT 11/11
with the corrected evaluator, indicating that the audit failures are primarily a
statistical power issue (per-regime N too small for equivalence testing) rather
than evidence of true leakage.

**Retracted claim (AMENDMENT-001 section 6.3):**

> CONFLICT: "8_length_feature FAIL" and "11_combined_shallow FAIL"

After feature hygiene correction (removing generator-internal `supports`/`contradicts`
from structured features), both CONFLICT baselines now PASS on the held-out split
(8_length acc=0.343, CI upper=0.375 < threshold=0.383; 11_combined acc=0.351,
CI upper=0.383 <= threshold=0.383).

### 1.3 Outcome-Dependent Interpretation Relabeling

AMENDMENT-001 section 6.7 contained an outcome-dependent validity judgment:

> "since INSUFFICIENT items test whether models recognize epistemic limits
> [...] the structural identifiability is *not a confound for the primary
> research question*"

This interpretation was written after observing the results and is therefore
**post-hoc**. It is hereby relabeled as:

**POST-HOC INTERPRETATION (not pre-registered):** The above judgment about
whether the INSUFFICIENT structural leak constitutes a confound was formulated
after seeing that INSUFFICIENT items failed. A pre-registered study should not
present outcome-dependent interpretations as pre-specified. The appropriate
pre-specified response is to fix the structural identifiability in T2 v3 (below)
and evaluate the fix, not to argue that the failure doesn't matter.

### 1.4 Remaining v2 Failures Requiring T2 v3

1. **INSUFFICIENT structural abstention leak:** The "Cannot be determined"
   gold answer and 4th hypothesis option appear ONLY on INSUFFICIENT items
   in v2, making them trivially identifiable by any classifier (100% accuracy).
   The answer format is categorically different from other regimes.

2. **Audit per-regime power insufficiency:** Per-regime final-audit Ns
   (250/250/125/125) are too small for reliable equivalence testing at the
   chance+0.05 margin, producing wide CIs and spurious per-regime FAILs
   even when point estimates are at or below chance.

3. **Unequal regime sizes:** CONFLICT and INSUFFICIENT each have half the
   items of CLEAN and DECOY (875 vs 1,750 in train; 125 vs 250 in audit),
   reducing per-regime statistical power for the smaller regimes.

---

## 2. T2 v3 Generator Design (PRE-SPECIFIED)

**These design decisions are specified BEFORE any T2 v3 items are generated
or evaluated. They constitute the pre-specified generator requirements.**

### 2.1 Universal 4-Option Format

**EVERY** item in **EVERY** regime has exactly **4 answer options**:
- 3 suspect hypotheses (e.g., "Alice is responsible")
- 1 abstention option: "Cannot be determined from available evidence"

This eliminates the v2 structural leak where option count (3 vs 4) identified
the regime. Chance level is exactly 0.25 for all items and regimes.

### 2.2 Randomized Option Positions

All 4 option positions are **randomized independently on every item**,
INCLUDING the abstention option. The abstention option must NOT be pinned
to any fixed index (e.g., always index 3 as in v2).

**Invariant:** Within each regime, the correct-answer position must be
uniformly distributed over {0, 1, 2, 3}. Verified by chi-squared test
(p > 0.05) per regime over any generated batch of >= 100 items.

### 2.3 Equal Regime Sizes

All four regimes have equal N:
- Train/held-out: N_per_regime items each (total = 4 * N_per_regime)
- Final-audit: N_audit_per_regime items each (total = 4 * N_audit_per_regime)

N_per_regime and N_audit_per_regime are determined by the power simulation
(section 3.2) before generation.

### 2.4 CONFLICT Length Counterbalancing

Evidence item lengths are matched across suspects within each item with a
tighter tolerance than v2, to eliminate the residual length imbalance
observed in v2 CONFLICT items.

### 2.5 Template-Cue, Surface-Form, and Regime-Vocabulary Tests

Three categories of additional leakage tests are added:

#### 2.5.1 Template-Family Classifier (DIAGNOSTIC)

Can a classifier predict which template family generated an item from
model-visible text alone? If so, template cues leak structural information.

**Status:** Informational diagnostic (not gated). Reported for transparency
but does not block the audit gate.

#### 2.5.2 Prohibited Surface-Form Shortcuts (GATE)

*(Phase A.2 revision: all criteria are now deterministic / constructive.
Chi-squared and KS non-rejection tests have been removed — "fails to reject"
is not proof of balance, and KS becomes hypersensitive at large N.)*

The following structural invariants are hard-gated. ANY violation on ANY
item causes a FAIL:

| # | Shortcut | Gate criterion (deterministic) |
|---|----------|-------------------------------|
| S1 | Option count | Every item has exactly 4 options, regardless of regime |
| S2 | Abstention position | Per regime, abstention position counts differ by AT MOST 1 (exact balance by construction) |
| S3 | Abstention presence | Every item in every regime contains the abstention option |
| S4 | Evidence count | Per-regime evidence-count sorted multisets are identical (same regime size) or proportion differs by ≤ 0.05 (different regime sizes) |
| S5 | Option text length | Per-regime mean option text length is within ±20% relative of the grand mean across all regimes |
| S6 | Gold-answer position | Per regime, gold-answer position counts differ by AT MOST 1 (exact balance by construction) |

These checks are deterministic and run before any classifier-based evaluation.
They catch the structural leakage patterns that caused v2 INSUFFICIENT failures.

**Rationale for deterministic criteria:**
- S2/S6: The generator must produce items with exactly balanced position
  assignments.  A chi-squared test can fail to reject severe imbalance at
  small N, or reject trivial imbalance at large N.  The "max diff ≤ 1"
  criterion is a verifiable construction property.
- S4: Evidence-count distributions should match by design. The deterministic
  criterion removes the KS test's N-dependence.
- S5: A 20% relative band is a pre-specified practical equivalence margin.
  Hypothesis text uses the same "[Name] is responsible" pattern across
  regimes, so length should be comparable.

#### 2.5.3 Regime-Identifying Vocabulary Test (DIAGNOSTIC, demoted from GATE)

After removing suspect names, can a TF-IDF classifier predict the regime
from the remaining text? If regime is identifiable from vocabulary, the
generator's regime-specific language patterns may leak.

**Status:** Informational diagnostic (DEMOTED from gate). Rationale for
demotion:

1. Regimes differ in evidence structure by design (e.g., CONFLICT items have
   contradictory evidence, DECOY items have misleading evidence). Some
   vocabulary signal from these structural differences is unavoidable and
   does not constitute a confound for the primary research question.
2. The real danger is structural shortcuts (option count, abstention
   presence/position) that allow trivial identification without reading
   content. These are caught by the surface-form gates (§2.5.2).
3. A hard gate on vocabulary classification risks spurious failures from
   regime-appropriate language that the evaluated model cannot exploit as
   a shortcut to the correct answer.

**Reporting requirement:** The regime-vocabulary classifier accuracy and CI
are reported in the leakage report. If CI upper > 0.30, the report flags
this with a note explaining whether the signal comes from (a) structural
shortcuts (which should have been caught by §2.5.2) or (b) content-level
vocabulary differences inherent to the regime design.

#### 2.5.4 Cross-Regime Counterfactual Pairs (DIAGNOSTIC)

The generator produces counterfactual minimal pairs: items sharing the same
template and suspect set but assigned to different regimes. For each pair,
the only differences should be in evidence content (polarity, diagnosticity,
sufficiency), not in surface structure.

**Verification:** For each counterfactual pair (item_A in regime R1,
item_B in regime R2):
- Same number of suspects, same suspect names
- Same number of evidence items (±1 allowed)
- Jaccard similarity of non-evidence text >= 0.85
- Different gold answer only when regime semantics require it

This is an informational diagnostic that helps interpret regime-vocabulary
classifier results: if cross-regime counterfactual pairs have high surface
similarity but the vocabulary classifier still discriminates, the signal
is likely from evidence content (acceptable) rather than template structure
(concerning).

### 2.6 Preserved v2 Counterbalancing Invariants

All v2 counterbalancing invariants are preserved in v3:
- Name-frequency equalization (CV <= 0.10)
- Evidence-count parity per suspect
- Clue-polarity balance (within ±1)
- Counterfactual minimal pairs (Jaccard >= 0.85)

**Positional balance — v2 vs v3 distinction:**

The v2 generator provides a WEAKER positional guarantee than v3:
- **v2 (historical):** No single gold-answer position holds more than 50% of
  items. This is tested by `TestV2LegacyPositionBalance` in the v2 test suite.
  The v2 generator uses a simple shuffle that avoids gross clustering but does
  NOT enforce exact balance by construction.
- **v3 (new, §2.5.2 S6):** Per-regime gold-answer position counts differ by
  AT MOST 1 (exact balance by construction). This is a strictly stronger
  requirement enforced by the v3 generator and verified by its own test suite.

The v2 positional test is retained as a LEGACY, NON-GATING regression test.
It is NOT represented as satisfying the v3 max-diff-≤-1 invariant.

---

## 3. Gate and Audit Procedure (PRE-SPECIFIED)

### 3.1 Per-Regime Equivalence Gate

For EACH baseline, on BOTH the template-held-out split AND the final-audit split,
for EACH regime individually AND in aggregate:

**PASS criterion:** Wilson 95% CI upper bound for accuracy <= chance + 0.05

Where chance = mean(1/n_options) per item = 0.25 under universal 4-option design.

Gate semantics: every cell in the {baseline} x {regime} x {split} matrix must
PASS. A single FAIL means the overall verdict is FAIL.

### 3.2 Audit-N Power Simulation (JOINT GATE)

**Before generating the final-audit set**, run an equivalence-gate power
simulation to determine the minimum per-regime N such that:

P(ALL baselines x regimes PASS simultaneously | true accuracy = chance) >= target_power

This is a **joint gate** requirement, not per-baseline. The overall gate requires
every cell in the {baseline x regime} matrix to PASS. Under independence, the
joint pass probability is the product of marginal pass probabilities, which
can be far below the individual marginal even when each cell individually
achieves high pass probability.

**Simulation:** `analysis/leakage_audit_power.py`
**Results:** `analysis/leakage_audit_power_results.json`

**Key parameters:**
- K = 11 baselines x 4 regimes = 44 cells on the audit split
- Chance = 0.25 for all regimes (universal 4-option design)
- Gate: Wilson 95% CI upper <= 0.25 + 0.05 = 0.30

**Power table (selected rows; P(joint gate passes | true accuracy = chance),
rho=0 exact binomial, rho>0 MC with block correlation):**

| N/regime | Total N | Marginal P(PASS) | Joint (rho=0) | Joint (rho=0.3) | Joint (rho=0.6) |
|----------|---------|------------------|---------------|-----------------|-----------------|
| 500 | 2,000 | 0.681 | 0.000 | 0.000 | 0.005 |
| 1,000 | 4,000 | 0.941 | 0.068 | 0.159 | 0.307 |
| 1,500 | 6,000 | 0.992 | 0.691 | 0.734 | 0.803 |
| 2,000 | 8,000 | 0.999 | **0.949** | 0.952 | 0.963 |

Full fine-grid table: `analysis/leakage_audit_power_results_report.md`

**First tested N achieving target P(gate passes):**

*(These are "first tested N" values in the grid, not mathematical minima.
The true minimum may lie between grid points.)*

| Target | rho=0.0 | rho=0.3 | rho=0.6 |
|--------|---------|---------|---------|
| >=0.80 | 1,500-2,000/regime | 1,500-2,000/regime | 1,500/regime |
| >=0.90 | 2,000/regime (8,000 total) | 2,000/regime (8,000 total) | 2,000/regime (8,000 total) |

**Correlation model:** Block correlation — baselines WITHIN the same regime
share rho, baselines ACROSS different regimes are independent. rho=0 uses
exact binomial (closed-form, no Monte Carlo). rho>0 uses Monte Carlo with
200k simulations per configuration.

**Design target:** P(joint gate passes | true accuracy = chance) >= 0.90.

**FROZEN audit-size decision:** **2,000 items per regime (8,000 total).**

- Design basis: rho=0 (independence) as conservative default.
  Independence is conservative because positive within-regime correlation
  increases the joint pass probability (failures cluster rather than spread).
- At N=2,000/regime, the exact independence result is ~0.949 joint pass
  probability, CONFIRMED by the reproducible fine-grid table in
  `analysis/leakage_audit_power_results.json`.
- This meets the >=0.90 target but does NOT claim 0.95 (the 0.95 threshold
  would require ~2,500/regime under independence).
- Structured within-regime correlation (rho>0) is reported as sensitivity
  analysis only and is not the design basis.

The per-regime audit N is FROZEN at 2,000 before generating the fresh audit set.

### 3.3 Fresh Final-Audit Set

- A genuinely fresh, untouched reserved template family is created for the
  v3 final audit.
- The v2 final audit (contamination_timeline, 750 items) is SPENT and will
  NOT be reused for tuning or evaluation.
- The v3 audit template family is generated, sealed, and not opened until
  the single audit evaluation run.

### 3.4 Reconciliation Requirement

The evaluator must enforce, for every baseline on both splits:

```
assert sum(per_regime_correct) == aggregate_correct
```

Any reconciliation failure causes the run to abort with an error.

---

## 4. Statistical Design Corrections (PRE-SPECIFIED)

### 4.1 Estimable Contrasts

The original protocol specified unconditional E, T, D main effects and a
three-way E×T×D interaction. This is incorrect: E=0 cells with T=1 or D=1
are incoherent (one cannot tabulate or disconfirm without first enumerating
hypotheses). Only 5 of the 8 E/T/D cells are coherent: 000, 100, 110, 101, 111.

**Corrected estimands:**

| # | Contrast | Operationalization |
|---|----------|--------------------|
| 1 | Enumeration | 100 vs 000 |
| 2 | T\|E=1 | mean(110, 111) vs mean(100, 101) |
| 3 | D\|E=1 | mean(101, 111) vs mean(100, 110) |
| 4 | T×D\|E=1 | (111 − 110) − (101 − 100) |

**Removed:** Unconditional E, T, D main effects; E×T×D three-way interaction.

### 4.2 Model as Fixed Effect, Primary Estimator

With only 2-3 model families, model is treated as a fixed effect. The former
`(1|model)` random effect specification is withdrawn. Model-specific estimates
are reported; no generalization claim is made beyond the tested models.

**Primary estimator:** Paired marginal contrast for the D effect, conditional
on E=1, in adversarial regimes (DECOY+CONFLICT), averaged over T:

```
contrast_i = 0.5 * [(Y_101_i - Y_100_i) + (Y_111_i - Y_110_i)]
```

where i indexes items. The estimand is E[contrast_i].

**Primary test:** One-sample paired t-test on the item-level contrast values.
Reports paired bootstrap 95% CI on the mean contrast.

**Robustness (secondary, not gated):** Effect-coded binomial GEE with
item-clustered robust SE. Reported for comparison but does not replace
the paired t-test as the primary analysis.

**Secondary tests (reported, not gated):**
- McNemar tests: 101 vs 100, 111 vs 110 (paired binary)
- Enumeration contrast: 100 vs 000
- T|E=1: mean(110,111) vs mean(100,101)
- T×D|E=1 interaction

### 4.3 Call-Matching Relabel

The 111 vs prism_full comparison was labeled "call-matched" but 111 uses 1
API call while prism_full uses 4. This is relabeled as "decomposition vs
scaffolding" — a comparison of reasoning strategies, not a cost-matched control.

---

## 5. Version Tracking

### 5.1 Superseded v2 Hashes (from AMENDMENT-001)

| File | SHA-256 (v2, superseded) |
|------|--------------------------|
| datasets/t2_generator/generator.py | f6cc03f405dd84794eb24314c01d696bbb9fa73dfac163558d0e1d1c5c12c62c |
| analysis/leakage_check.py | 1c16152e390f36ee0091e12d1f18f9b92643b5ee075c6c56c065f59751e23fe4 |

### 5.2 Corrected Evaluator Hash (Phase A)

| File | SHA-256 |
|------|---------|
| analysis/run_leakage_eval.py (corrected) | [to be computed at freeze time] |

### 5.3 v3 Hashes (to be added in Phase B)

Reserved for:
- datasets/t2_generator/generator.py (v3)
- analysis/run_leakage_eval.py (v3-compatible)
- T2 v3 train corpus
- T2 v3 final-audit corpus

These hashes will be computed and recorded when the v3 generator is frozen,
BEFORE generating the final-audit set.

---

## 6. Governance

### 6.1 Pre-Outcome Status

This amendment is filed BEFORE:
- Any T2 v3 items are generated
- Any T2 v3 evaluation is run
- The v3 final-audit set exists

All design decisions in sections 2-4 are pre-specified. If the v3 evaluation
fails, the failure will be reported honestly without further tuning, per the
same governance rules as AMENDMENT-001 section 4.3.

### 6.2 Relationship to AMENDMENT-001

AMENDMENT-001 remains the authoritative record of:
- The T2 v1 leakage findings (section 1)
- The T2 v2 generator design (section 3)
- The pre-specified v2 acceptance criteria (section 4)
- The v2 evaluation outcomes (section 6)

AMENDMENT-002 does NOT modify or delete any AMENDMENT-001 text. It:
- Corrects the evaluator defects (section 1.1)
- Retracts specific outcome-dependent claims (section 1.2)
- Relabels post-hoc interpretation as post-hoc (section 1.3)
- Specifies the v3 design pre-outcome (sections 2-4)

### 6.3 PROTOCOL.lock.json

The PROTOCOL.lock.json file is a frozen hash-locked artifact. Per the
append-only amendment pattern established by AMENDMENT-001:
- Existing v2 hashes in `file_hashes` and `superseded_hashes` are PRESERVED
- New v3 hashes will be ADDED alongside (not replacing) when v3 is frozen
- The lock file's `notes` and `amendment` fields will be updated to reference
  AMENDMENT-002

This update will occur in the Phase B commit when the v3 generator is frozen,
not in this Phase A commit.

---

## 7. v3.2 Pre-Registered Design and Gate Rules (FROZEN BEFORE v3.2 GENERATION)

**Date frozen:** 2026-09-04 (original §7); **revised** 2026-09-04
**Status:** PRE-OUTCOME — committed BEFORE any v3.2 corpus is generated or
any v3.2 classifier is run.

**Revision note:** This section supersedes the earlier §7 committed at
`e19b452`. The earlier version specified a "structured classifier" gate and
an "absent from / present at" substitution-based construction. That approach
is REJECTED because it placed the answerability distinction in polarity
tokens, enabling keyword-based shortcutting rather than requiring relational
reasoning. The revised §7 below specifies a permutation-based construction
with exact unigram-multiset preservation and a revised classifier rule.

### 7.1 Counterfactual-Pair Construction Decision (PERMUTATION / ENDPOINT-REWIRING)

v3.2 constructs answerable/insufficient counterfactual pairs by **exact
token-multiset-preserving permutation**: entities, predicates, evidence-slot
templates, and the complete item-level unigram token multiset are held fixed.
Entity assignments or relation endpoints are permuted across TWO OR MORE
evidence slots so that the item-level unigram multiset is byte-for-byte
identical between the answerable and insufficient pair members.

**Intended mechanism:** The answerable member connects both necessary facts
(e.g., access evidence + alibi invalidation) to the SAME suspect, yielding
a unique conclusion. The insufficient member distributes those same facts
across DIFFERENT suspects, leaving no uniquely-supported suspect. Same
tokens, different relational graph.

**Label assignment:** Labels (answerable / insufficient) are assigned by a
deterministic symbolic rule engine that inspects the relational graph
(uniqueness of support in the answerable member; no uniquely-supported
suspect in the insufficient member). Labels are NEVER assigned by surface
text inspection.

### 7.2 Frozen Construction Invariants

For every counterfactual pair, the following invariants MUST hold:

| # | Invariant | Verification method |
|---|-----------|-------------------|
| C1 | Exact item-level unigram-multiset equality | `Counter(re.findall(r'\b\w+\b', text.lower()))` over narrative + all evidence content is identical between pair members |
| C2 | Identical evidence counts | Both members have exactly `N_EVIDENCE_SLOTS` evidence items |
| C3 | Identical option text | Hypothesis texts (and their templates) are identical between pair members |
| C4 | Identical global entity/predicate frequencies | Each entity name appears the same number of times in both members |
| C5 | Label difference caused SOLELY by the relational graph | The symbolic rule engine derives different labels from identical token inventories by inspecting which entity-slot connections differ |
| C6 | Pair- and template-family-grouped eval splits | Counterpart items (answerable + insufficient twin) AND items from the same template family must never cross train/test split boundary |

### 7.3 Positive Control: Deterministic Symbolic Oracle

A deterministic relation-aware symbolic oracle/solver MUST achieve 100%
accuracy on BOTH members of every pair. This oracle inspects the relational
graph (entity-slot bindings) and applies the same logical rule the generator
uses to assign labels. If the oracle fails on any item, the item is
malformed and must not enter the corpus.

### 7.4 Binary Non-Relational Baseline Gate (DECISIVE)

**Gate definition:** For each NON-RELATIONAL baseline in the battery, the
one-sided 95% upper confidence bound on balanced accuracy must be ≤ 0.55
(chance balanced accuracy = 0.50), evaluated with pair-grouped AND
template-family-grouped train/test splits (counterparts must never cross the
split).

**Non-relational baseline battery** (minimum required):

| # | Baseline | Features received | Relational access |
|---|----------|------------------|-------------------|
| B1 | Word unigram TF-IDF | Bag of unigram tokens from narrative + evidence | None |
| B2 | Character n-gram TF-IDF | Character 3-5 grams from narrative + evidence | None |
| B3 | Evidence/token counts | Number of evidence slots, total token count, per-slot token counts | None |
| B4 | Item and slot lengths | Character lengths of narrative, each evidence slot, total item | None |
| B5 | Position features | Gold-answer position, abstention position | None |
| B6 | Global polarity-token totals | Counts of words like "confirmed", "denied", "absent", "present", etc. across the full item | None |
| B7 | Combined shallow | Union of B1-B6 features | None |

For every baseline, the feature description is recorded so a reader can
confirm it has no access to relational structure (entity-slot bindings,
cross-reference targets, alibi-invalidation chains).

**This rule is frozen prior to observing results.**

### 7.5 Revised Classifier Rule (SUPERSEDES EARLIER "STRUCTURED CLASSIFIER" RULE)

Success by a classifier that ENCODES RELATION ENDPOINTS or performs the
intended logical operation (i.e., inspects which suspect's alibi is
invalidated by which cross-reference) is NOT leakage — it is a positive
control that SHOULD succeed. Only SHALLOW NON-RELATIONAL features (as
enumerated in §7.4) must remain near chance (≤ 0.55 upper bound).

Rationale: The entire point of the permutation-based construction is that
the label difference is carried solely by the relational graph. A classifier
with access to relational structure SHOULD distinguish the pair members;
failure to do so would indicate the construction is broken. Conversely, a
classifier limited to surface features SHOULD NOT distinguish them; success
would indicate surface leakage.

### 7.6 Four-Way D2 Regime Classifier (REPORTED DIAGNOSTIC, NOT A GATE)

The four-way D2 regime classifier (predicting CLEAN/DECOY/CONFLICT/INSUFFICIENT
from narrative + evidence context only) remains a REPORTED DIAGNOSTIC with
feature importances. It is NOT a decisive gate.

Rationale: The four regimes intentionally differ in evidence structure —
CLEAN has confirmed alibis with cross-reference invalidation, DECOY adds
motive decoys, CONFLICT introduces source-precedence disputes, and
INSUFFICIENT has symmetric evidence with no uniquely-supported suspect.
These structural differences are inherent to the diagnostic design and
SHOULD produce some vocabulary signal. The D2 classifier accuracy quantifies
this signal for transparency, but gating on it would penalize the generator
for correctly implementing distinct regimes.

Feature importances from the four-way D2 classifier are reported to
identify which specific vocabulary patterns drive regime separation,
informing future generator refinements.

---

**Filed by:** Automated Phase A.2 build
**Governance note:** This amendment documents pre-specified design decisions
for T2 v3. The acceptance criteria (sections 2-3) are specified before any
v3 items exist. The evaluator corrections (section 1) are applied to the
existing v2 data to establish accurate baselines. Section 7 records the v3.2
pre-registered construction decision and gate rules, revised to specify
permutation-based exact-unigram-multiset construction (superseding the
earlier substitution-based approach committed at `e19b452`). All rules are
frozen before any v3.2 generation or evaluation.
