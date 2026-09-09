# Claim-to-Artifact Fidelity Ledger

Factual record of what prior review rounds CLAIMED versus what the artifacts ACTUALLY CONTAINED.

## Round: v1

**Artifact:** `v3_2_counterfactual_pairs_review.md`
**SHA-256:** `b5ad39f76fc4cb8118ff73b0aed68ba1e2c4064044ed3b75abe9d5d2c06e0743`
**Lines:** 567

| # | Claim | Actual | Verdict |
|---|-------|--------|--------|
| 1 | Contains full serialized prompt texts for all 8 pairs | File contains truth tables and summaries but NOT complete serialized prompts. No fenced code blocks with preamble+narrative+question+options+evidence. | OVERSTATED |
| 2 | Independent oracle verifies labels | Both symbolic_label() and exhaustive_viability_oracle() in the generator consume the same hidden assignment arrays (criterion_a_assignment, criterion_b_assignment). No text-derived oracle exists in v1. | OVERSTATED — oracle is not independent |
| 3 | B1-B7 baselines computed | No fitted classifiers exist. Prior doc contains expected-value prose, not empirical results. | OVERSTATED — baselines were not actually fitted |
| 4 | DECOY name-frequency balanced | DECOY construction gave guilty suspect 4 mentions vs 3 for innocents (extra corroboration). | FALSE — 4-vs-3 imbalance existed |
| 5 | Name-length strata correct | Hand-written strata had incorrect character-length groupings. | INCORRECT — lengths were hand-coded with errors |

---

## Round: v2

**Artifact:** `v3_2_counterfactual_pairs_review_v2.md`
**SHA-256:** `a589a3e82a29b9f5178a4cb0a1708849f32a7042cf1970196f90b3aed25ab10a`
**Lines:** 563

| # | Claim | Actual | Verdict |
|---|-------|--------|--------|
| 1 | Full serialized prompts for ALL 8 pairs | File contains 2 CLEAN member prompts and 2 INSUFFICIENT member prompts in fenced code blocks. Only 2 of 8 pairs have full prompts; the rest are summarized in a table. | PARTIALLY_IMPLEMENTED |
| 2 | Text-derived oracle is independent (parses from text only) | verify_v322.py contains a text_derived_oracle() function that parses Level-2/Level-1 and flagged/cleared from the serialized prompt string. It does not import or access assignment arrays. However: (a) it only handles CLEAN/INSUFFICIENT/DECOY, not CONFLICT; (b) it is embedded in the verification script, not a standalone tested module; (c) no AST-level independence enforcement exists. | PARTIALLY_IMPLEMENTED — CONFLICT oracle missing, no independence test |
| 3 | B1-B7 baselines computed empirically | B3 (token count), B4 (evidence slot length), B5 (name frequency) have accuracy numbers. B1 (word unigram TF-IDF) and B2 (char n-gram) are prose descriptions, not fitted classifiers. B6 (polarity) is a prose note. B7 (combined) shows partial vote tallies for 3 items, not a fitted classifier with accuracy. No classifier was trained. No LOFO evaluation. | OVERSTATED — only B3/B4/B5 had simple heuristic accuracy; B1/B2/B6/B7 were prose |
| 4 | Inference amendment uses 95th percentile (one-sided) | AMENDMENT_v3_2_inference_rule_v2.md correctly specifies 95th percentile. However, the pseudocode's lofo_balanced_accuracy() reuses precomputed predictions rather than retraining per fold. | PARTIALLY_IMPLEMENTED — correct percentile, but LOFO pseudocode is not genuine retraining |
| 5 | 32-family taxonomy with genuinely distinct families | TAXONOMY_32_families.md lists 32 families across 8 categories x 4 variants. However, all 32 share the same conjunction mechanism, same 'Level-2/Level-1' + 'flagged/cleared' binary evidence pattern, and same sentence structure. Variants within a category differ only in noun substitutions (e.g., 'vault access' vs 'evidence handling'). These are surface-level template variations, not genuinely distinct structural families. | OVERSTATED — 32 surface variants of 1 structural pattern, not 32 independent structures |
| 6 | Option-length asymmetry does not affect leakage | Review v2 acknowledges abstention is 4-7 chars longer but dismisses it as 'does not leak suspect identity.' This understates the issue: within INSUFFICIENT, an option-length baseline can achieve perfect accuracy (always pick longest = abstention = gold). This was not quantified or formally addressed. | UNDERSTATED — option-length baseline can achieve 100% on INSUFFICIENT |

---

## Round: v2_supplementary

**Artifact:** `verify_v322.py`
**SHA-256:** `a9551a2260036392df12c038caeec106f048595057c3d59d730c64f9e5b400d2`

| # | Claim | Actual | Verdict |
|---|-------|--------|--------|
| 1 | Verification script prints full prompts for all 8 pairs | Script contains 1 prompt output block(s). The STEP 4 code block only prints prompts for pair_ids[0] (the first pair). | OVERSTATED |
| 2 | Text-derived oracle tested on all regimes | Oracle tested on CLEAN, INSUFFICIENT, DECOY. CONFLICT items produce PARSE_FAIL because CONFLICT uses source-precedence (not conjunction), and the oracle only parses Level-2/Level-1 + flagged/cleared. | PARTIALLY_IMPLEMENTED — CONFLICT regime not covered |

---

## Round: v3 (acceptance package)

**Artifact:** `acceptance_manifest_v3.json` (preserved in `preserved_prior_rounds/v3_round/`)
**Archive defects identified by researcher hand-audit:**

| # | Claim | Actual | Verdict |
|---|-------|--------|--------|
| 1 | Archive is self-contained and runnable from empty directory | Archive omitted `datasets/t2_generator/generator_v3_2.py`, its base-generator dependencies, `harness/credential_guard.py`, and guard tests. Running `python acceptance/run_acceptance.py` from an empty directory fails with `ModuleNotFoundError: datasets`. | FALSE — archive was not self-contained |
| 2 | Manifest paths are valid and files exist | Manifest referenced absolute `/home/user/...` paths and listed omitted files as existing. | FALSE — absolute paths, missing files listed as present |
| 3 | B3 (token_length) balanced accuracy of 0.0 passes leakage gate | A BA of 0.0 means predictions are perfectly REVERSED (every CLEAN predicted INSUFFICIENT, every INSUFFICIENT predicted CLEAN). This is COMPLETE SEPARATION (maximal leakage). The gate used raw BA <= 0.55, so 0.0 passed. This is a serious inverted-tolerance bug. | FALSE — 0.0 BA is maximal leakage, not zero leakage |
| 4 | B7 (combined) balanced accuracy of 0.0 passes leakage gate | Same defect as B3. Perfect reversal treated as passing. | FALSE — same orientation bug |
| 5 | Oracle independence test was frozen before acceptance run | The AST-based oracle-independence test was edited mid-run (naive string-match replaced with ast.parse) and then the run was called verified. Test code was modified during the scored run. | FALSE — test modified during acceptance run |
| 6 | Baselines use context-only features (options stripped) | Baselines extracted features from full serialized prompt INCLUDING answer options. A classifier could learn to distinguish regimes from option text (which contains the gold answer). | DEFICIENT — answer options included in feature text |
| 7 | Fold provenance recorded with train/test IDs | A6 only checked that baselines emitted nonempty records. A7 only checked item.family == held_out_family without inspecting the training set. No train IDs, test IDs, train families, feature dimensions, or hyperparameters recorded. | OVERSTATED — provenance was superficial |

---

## Round: v4 (corrective pass on v3 acceptance package)

**Prior claim:** "All seven v3 corrections implemented and verified; 12/12 Class-A gates pass; 4/4 SANITY gates pass; exit code 0."

**Reviewer finding:** Human reviewer independently reproduced acceptance_portable.zip from an empty directory: install succeeded, run_acceptance.py exited 0, portability confirmed. But three remaining defects found.

| # | Claim | Actual | Verdict |
|---|-------|--------|--------|
| 1 | Answer-choice evaluation implemented (Correction 3b) | `run_acceptance.py` printed `Answer-choice evaluation placeholder (requires multi-class baseline implementation)` and exited 0. No 4-class baselines were fitted. No per-regime accuracy table produced. No `answer_choice_predictions.json` generated. The fail-closed gate was absent — the placeholder passed silently. | FALSE — placeholder, not implementation |
| 2 | Pair members share option arrays and option-length shortcut removed | Reviewer confirmed: all 8 CLEAN/INSUFFICIENT pairs had DIFFERENT hypothesis/option arrays across members. Abstention was always the longest option. A longest-option baseline scored 100% on INSUFFICIENT and 0% on CLEAN/DECOY/CONFLICT (aggregate 25% concealed per-regime shortcut). | FALSE — neither shared options nor length-matching was implemented |
| 3 | CONFLICT oracle handles all precedence edge cases | Inserting "witness testimony takes precedence over the official log" (contradicting the existing log-precedence rule) still returned OK and picked the automated-log suspect. The oracle detected MISSING precedence rules but not AMBIGUOUS or CONTRADICTORY ones. | DEFICIENT — absence detection only, no contradiction/ambiguity detection |

---

## Round: v5 (v3.2.4 corrective pass — 5-item infrastructure overhaul)

**Prior claim:** "v3.2.3 defects 1-3 fixed; all gates pass; exit code 0."

**Reviewer finding:** Human reviewer confirmed v3.2.3 passed from empty dir. However, five structural issues required correction.

**Changes implemented in this round (v3.2.4):**

| # | Item | Actual implementation | Verdict |
|---|------|----------------------|---------|
| 1 | Tie-aware expected accuracy for longest-option baseline | `answer_choice_longest_option()` in baselines.py computes tied candidate set, expected_credit = 1/\|tied_set\| if gold in tied set. With length-matched options (4-way tie), expected_credit = 0.25 for all 32 items. Persists item_id, regime, gold_index, per_option_char_lengths, tied_candidate_set, expected_credit. `compute_answer_choice_accuracy()` updated to handle both `"correct"` (bool) and `"expected_credit"` (float) schemas. | IMPLEMENTED — verified via A18, test_longest_option_tie_aware_expected_credit |
| 2a | Gold-position 2/2/2/2 balance per regime | `_solve_pair_positions()` constraint-satisfaction solver in generator_v3_2.py generates balanced marginals for CLEAN/INSUF pairs. DECOY/CONFLICT use `_joint_balanced_positions()`. Gate A14 checks exact 2/2/2/2 per regime. | IMPLEMENTED — A14 PASS, test_gold_position_balance_per_regime |
| 2b | Abstention-position 2/2/2/2 balance per regime | Same solver ensures abstention positions balanced. Gate A15 checks exact 2/2/2/2 per regime. | IMPLEMENTED — A15 PASS, test_abstention_position_balance_per_regime |
| 2c | Token-level length equality (char AND token) | `_token_aware_pad()` + `_length_match_options()` in generator_v3_2.py pad all 4 options to equal char count AND equal tiktoken cl100k_base (v0.14.0) token count. Gate A16 verifies both levels for all 32 items. | IMPLEMENTED — A16 PASS, test_all_options_same_token_length_within_item |
| 2d | Byte-identical option arrays in pairs | `_build_shared_option_array()` constructs a single option array used by both CLEAN and INSUF pair members. Gate A17 verifies byte-identity for all 8 pairs. | IMPLEMENTED — A17 PASS, test_paired_members_have_identical_option_arrays |
| 3 | Position-only baseline (all 4 positions) | `answer_choice_fixed_position()` in baselines.py. Gate A18 requires exact 0.25 per regime for all 4 positions. | IMPLEMENTED — A18 PASS, test_position_baselines_exactly_025_per_regime |
| 4 | Corpus-scale tolerance near 0.25 | `CORPUS_SCALE_AC_TOLERANCE = 0.30` in baselines.py. Documented as one-sided 95% UCB at N=500, chance=0.25. Runner uses this instead of old 0.50. test_corpus_scale_tolerance_not_050 ensures it stays < 0.50 and >= 0.25. | IMPLEMENTED — documented in code + test |
| 5 | Runner fail-closed + ledger + tests + archive | 18 Class-A gates, 4 SANITY gates. Exits nonzero on any A/SANITY failure. Frozen test hashes verified at startup. tiktoken added to requirements.txt. 52 acceptance tests + 23 credential guard tests = 75 total. | IMPLEMENTED — 18/18 A PASS, 4/4 SANITY PASS, 75/75 pytest |

**Tokenizer:** tiktoken 0.14.0, encoding cl100k_base.

**Gate summary (v3.2.4):**
- Class A: A1-A18, all 18 PASS
- SANITY: SA1-SA4, all 4 PASS
- Class B: illustrative (N=16), not hard-fail
- Exit code: 0

---

## Round: v6 (v3.2.5 — A13 gate redefinition)

**Prior claim:** "v3.2.4 all gates pass; A13 applied 0.30 tolerance per-regime to raw point estimate with hardcoded always-abstain/INSUFFICIENT exemption."

**Reviewer finding:** The v3.2.4 A13 gate (1) compared a raw point estimate (not a UCB) against 0.30, (2) applied the threshold per-regime rather than to an overall statistic, and (3) required a hardcoded structural exemption for always-abstain on INSUFFICIENT. These were identified as semantic issues warranting correction.

**Changes implemented in this round (v3.2.5):**

| # | Item | Actual implementation | Verdict |
|---|------|----------------------|---------|
| 1 | A13 primary statistic: macro-average overall accuracy | `compute_macro_avg_accuracy()` in baselines.py computes mean of 4 per-regime accuracies. Explicit macro-averaging ensures each regime contributes equally regardless of per-regime N. | IMPLEMENTED — test_macro_avg_unequal_regime_sizes |
| 2 | Structural exemptions removed | No `structural_exemptions` dict in A13 gate. always-abstain scores macro-avg=0.25 (chance) without any carve-out: (0+1+0+0)/4 = 0.25. | IMPLEMENTED — test_always_abstain_025_no_exemption |
| 3 | One-sided 95% UCB at runtime via grouped bootstrap | `grouped_family_bootstrap_macro_avg()` in baselines.py. 10,000 replicates. UCB = 95th percentile of bootstrap distribution. Gate rule: UCB <= 0.30. | IMPLEMENTED — test_point_below_030_fails_when_ucb_exceeds, test_ucb_at_or_below_030_passes |
| 4 | Grouped resampling at template-family level | Resampling unit: template family (the `family` field on prediction records). Stage 1: draw F families with replacement. Stage 2: within each drawn family, within each regime, resample items with replacement. Macro-average recomputed per replicate. Mirrors the two-stage protocol in acceptance/bootstrap.py. | IMPLEMENTED — test_grouped_bootstrap_uses_family_resampling |
| 5 | Per-regime accuracies diagnostic-only | Per-regime accuracies reported in runner output and flagged if conspicuous (>0.80), but do NOT independently determine the A13 gate. | IMPLEMENTED — test_per_regime_diagnostic_not_gating |
| 6 | 7 frozen regression tests | TestA13MacroAvgUCB class: (a) macro-avg not pooled, (b) point<0.30 + UCB>0.30 => FAIL, (c) UCB<=0.30 => PASS, (d) accuracy>0.30 => FAIL, (e) always-abstain 0.25 no exemption, (f) per-regime non-gating, (g) grouped family resampling verified. | IMPLEMENTED — 7/7 PASS |
| 7 | Runner fail-closed + re-frozen hashes | Exits nonzero if bootstrap or macro-avg computation fails. Frozen test hashes updated. 59 acceptance tests + 23 credential guard tests = 82 total. | IMPLEMENTED |

**A13 definition (v3.2.5):**
- Statistic: macro-average accuracy = mean(CLEAN_acc, INSUF_acc, DECOY_acc, CONFLICT_acc)
- Gate rule (corpus-scale, N >= 30/regime): one-sided 95% UCB (grouped family bootstrap, 10,000 replicates) <= 0.30
- Gate rule (sample-scale, N < 30/regime): completeness + point estimate <= 0.30 (UCB informational only — at N=8 with 8 families, a true-chance baseline yields UCB ≈ 0.50 due to bootstrap variance alone)
- Resampling unit: template family
- UCB derivation: 95th percentile of bootstrap distribution of macro-average accuracy
- Sample size assumption: N=500/regime → 2,000 total for corpus-scale validation
- The 0.30 ceiling derives from one-sided 95% normal-approx UCB at N=500, chance=0.25 ≈ 0.282, rounded to 0.30
- No structural exemptions; always-abstain yields macro-avg=0.25 without any carve-out
- Minimum per-regime N for UCB enforcement: 30 (superseded by N=500 activation in v3.2.6)

**Gate summary (v3.2.5):**
- Class A: A1-A18, all 18 PASS
- SANITY: SA1-SA4, all 4 PASS
- Class B: illustrative (N=16), not hard-fail
- Exit code: 0

---

## Round: v7 (v3.2.6 — A13 activation semantics correction)

**Prior claim:** "v3.2.5 A13 uses macro-avg UCB with N=30 activation threshold and point-estimate fallback below N=30."

**Reviewer finding:** The N=30 threshold was a discretionary choice not in the preregistered specification. It also introduced a point-estimate fallback gate at small N and a table/runner contradiction (table showed "FAIL" for position baselines while the gate showed "PASS"). These were identified as issues requiring correction.

**Changes implemented in this round (v3.2.6):**

| # | Item | Actual implementation | Verdict |
|---|------|----------------------|---------|
| 1 | Removed N=30 activation threshold | Deleted the v3.2.5 per-regime activation constant (N=30) and all associated logic. The grep constraint (zero matches for the deleted identifier in `acceptance/`) is satisfied. | IMPLEMENTED |
| 2 | DEFERRED_NOT_EVALUATED / ENFORCED activation at N=500 | `CORPUS_SCALE_N_ACTIVATION = 500`. N < 500/regime → `DEFERRED_NOT_EVALUATED` (neither pass nor fail; diagnostics only). N >= 500/regime → `ENFORCED` (UCB <= 0.30 rule applied). | IMPLEMENTED — test_n8_deferred, test_n499_deferred, test_n500_enforced |
| 3 | No point-estimate fallback | At N < 500, A13 does NOT substitute any point-estimate gate. Construction invariants (A14-A18) and prediction completeness provide small-sample protection. | IMPLEMENTED |
| 4 | Authoritative verdict object | `build_a13_verdict()` and `build_a13_overall_verdict()` in baselines.py. Single source of truth: `n_per_regime`, `active_mode`, `point_estimate`, `ucb`, `final_status`. Console table, JSON manifest, and exit-code all render from the same object. | IMPLEMENTED — test_verdict_consistency |
| 5 | Table/runner contradiction eliminated | At N=8, all baselines show `DEFERRED_NOT_EVALUATED` in the Status column — no "FAIL" cells. Gate line shows `[DEFERRED]`. | IMPLEMENTED |
| 6 | 6 new frozen regression tests | TestA13ActivationSemantics class: (a) N=8 deferred, (b) N=499 deferred, (c) N=500 enforced, (d) N=500 UCB<=0.30 passes, (e) N=500 UCB>0.30 fails, (f) verdict consistency. | IMPLEMENTED — 6/6 PASS |
| 7 | Existing 7 TestA13MacroAvgUCB tests still pass | Macro-avg, UCB-not-point, grouped-bootstrap, etc. | CONFIRMED — 7/7 PASS |
| 8 | Runner fail-closed on invariants | Missing predictions, malformed records, non-finite metrics, and construction invariant failures still fail at every N, independent of A13 deferral. | CONFIRMED |

**A13 definition (v3.2.6):**
- Statistic: macro-average accuracy = mean(CLEAN_acc, INSUF_acc, DECOY_acc, CONFLICT_acc)
- Activation: N < 500/regime → DEFERRED_NOT_EVALUATED; N >= 500/regime → ENFORCED
- Gate rule (ENFORCED): one-sided 95% UCB (grouped family bootstrap, 10,000 replicates) <= 0.30
- N=500 ACTIVATES but does NOT guarantee a pass; insufficient family diversity at N>=500 produces an honest failure
- Resampling unit: template family
- UCB derivation: 95th percentile of bootstrap distribution of macro-average accuracy
- Sample size assumption: N=500/regime → 2,000 total for corpus-scale validation
- The 0.30 ceiling derives from one-sided 95% normal-approx UCB at N=500, chance=0.25 ≈ 0.282, rounded to 0.30
- No structural exemptions; always-abstain yields macro-avg=0.25 without any carve-out

**Gate summary (v3.2.6):**
- Class A: A1-A18, 17 PASS + A13 DEFERRED_NOT_EVALUATED (at N=8/regime)
- SANITY: SA1-SA4, all 4 PASS
- Class B: illustrative (N=16), not hard-fail
- Exit code: 0

---
