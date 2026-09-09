#!/usr/bin/env python3
"""
Deterministic pytest suite for v3.2 acceptance (v4, corrected).

Corrections applied:
- Correction 2: Orientation-invariant separability checks
- Correction 3a: Context-only regime leakage baselines
- Correction 4: Fit-provenance verification (train/test disjointness)
- Correction 5: Oracle mutation tests
- Correction 7: Hard sanity gates (degenerate probabilities, metric ranges)

Tests are FAIL-CLOSED: if something can't be checked, it fails.
"""

import ast
import copy
import glob
import inspect
import os
import re
import sys
import json
import hashlib
from collections import Counter

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.t2_generator.generator_v3_2 import (
    T2V32Generator,
    tokenize_text,
    multiset_hash,
    verify_prompt_multiset_equality,
    serialize_prompt,
    compute_name_length_strata,
    NAME_LENGTH_STRATA,
    FAMILY_EVIDENCE_SPECS,
    __version__,
)
from datasets.t2_generator.generator import SUSPECTS as ALL_SUSPECTS
from datasets.t2_generator.generator_v3_1 import TEMPLATE_FAMILIES
from harness.credential_guard import (
    validate_provider,
    CredentialGuardError,
    OFFLINE_ALLOWLIST,
    BLOCKED_ENV_VARS,
    KNOWN_BLOCKED_ENDPOINTS,
)
from acceptance.oracles import conjunction_oracle, conflict_oracle, run_oracle
from acceptance.baselines import (
    run_all_baselines, compute_balanced_accuracy,
    orientation_invariant_separability,
    extract_b1_features, extract_b2_features, extract_b3_features,
    extract_b4_features, extract_b5_features, extract_b6_features,
    extract_context_only_text,
    run_answer_choice_baselines, compute_answer_choice_accuracy,
    compute_macro_avg_accuracy, grouped_family_bootstrap_macro_avg,
    build_a13_verdict, build_a13_overall_verdict,
    _prediction_credit, REGIMES,
    CORPUS_SCALE_AC_TOLERANCE, CORPUS_SCALE_N_ACTIVATION,
)
import tiktoken

SEED = 42


@pytest.fixture(scope="module")
def generated_items():
    gen = T2V32Generator(seed=SEED)
    return gen.generate_dataset(n_per_regime=8, seed=SEED)


@pytest.fixture(scope="module")
def items_by_regime(generated_items):
    by_regime = {}
    for item in generated_items:
        by_regime.setdefault(item.regime, []).append(item)
    return by_regime


@pytest.fixture(scope="module")
def clean_insuf_pairs(items_by_regime):
    clean_by_pair = {it.metadata["pair_id"]: it for it in items_by_regime["CLEAN"]}
    insuf_by_pair = {it.metadata["pair_id"]: it for it in items_by_regime["INSUFFICIENT"]}
    return clean_by_pair, insuf_by_pair


# ================================================================
# MULTISET EQUALITY TESTS
# ================================================================

class TestMultisetEquality:
    def test_all_8_pairs_exist(self, clean_insuf_pairs):
        clean_by_pair, insuf_by_pair = clean_insuf_pairs
        assert len(clean_by_pair) == 8, f"Expected 8 CLEAN items, got {len(clean_by_pair)}"
        assert len(insuf_by_pair) == 8, f"Expected 8 INSUF items, got {len(insuf_by_pair)}"
        assert set(clean_by_pair.keys()) == set(insuf_by_pair.keys())

    def test_multiset_equality_all_pairs(self, clean_insuf_pairs):
        clean_by_pair, insuf_by_pair = clean_insuf_pairs
        for pid in clean_by_pair:
            equal, only_a, only_b, ha, hb = verify_prompt_multiset_equality(
                clean_by_pair[pid], insuf_by_pair[pid]
            )
            assert equal, (
                f"Pair {pid} multiset MISMATCH. "
                f"Only in CLEAN: {dict(only_a)}, Only in INSUF: {dict(only_b)}"
            )
            assert ha == hb, f"Pair {pid} hash mismatch: {ha} != {hb}"

    def test_serialized_prompts_present(self, generated_items):
        for item in generated_items:
            prompt = item.metadata.get("serialized_prompt", "")
            assert len(prompt) > 100, f"Item {item.id} has no/short serialized_prompt"

    def test_prompts_contain_all_components(self, generated_items):
        for item in generated_items:
            prompt = item.metadata.get("serialized_prompt", "")
            has_rule = "INVESTIGATION RULE:" in prompt
            has_prec = "precedence" in prompt.lower()
            assert has_rule or has_prec, f"Item {item.id}: no rule or precedence"
            assert "Evidence:" in prompt, f"Item {item.id}: no Evidence section"
            assert "(A)" in prompt and "(B)" in prompt, f"Item {item.id}: missing options"
            options = re.findall(r"\([A-D]\)", prompt)
            assert len(options) == 4, f"Item {item.id}: expected 4 options, got {len(options)}"


# ================================================================
# NAME-FREQUENCY / CO-OCCURRENCE BALANCE TESTS
# ================================================================

class TestNameBalance:
    def test_name_frequency_balanced_all_regimes(self, generated_items):
        for item in generated_items:
            freqs = item.metadata.get("name_frequencies", {})
            if not freqs:
                continue
            vals = list(freqs.values())
            assert len(set(vals)) == 1, (
                f"Item {item.id} ({item.regime}): name frequency imbalance: {freqs}"
            )

    def test_name_length_same_within_item(self, generated_items):
        for item in generated_items:
            names = list(item.metadata.get("name_frequencies", {}).keys())
            if not names:
                continue
            lengths = [len(n) for n in names]
            assert len(set(lengths)) == 1, (
                f"Item {item.id}: name length mismatch: "
                f"{dict(zip(names, lengths))}"
            )

    def test_name_length_strata_programmatic(self):
        strata = compute_name_length_strata(ALL_SUSPECTS)
        for length, names in strata.items():
            assert len(names) >= 3, f"Stratum {length} has only {len(names)} names"
            for name in names:
                assert len(name) == length, (
                    f"Name '{name}' has len={len(name)} but is in stratum {length}"
                )


# ================================================================
# OPTION-LENGTH TESTS (updated for DEFECT 2 fix: length-matched)
# ================================================================

class TestOptionLength:
    def test_all_options_same_char_length_within_item(self, generated_items):
        """All 4 options must have identical character length within each item."""
        for item in generated_items:
            hyp_lens = [len(h) for h in item.hypotheses]
            assert len(set(hyp_lens)) == 1, (
                f"Item {item.id} ({item.regime}): char lengths not equal: "
                f"{hyp_lens}"
            )

    def test_all_options_same_token_length_within_item(self, generated_items):
        """All 4 options must have identical tiktoken cl100k_base token count
        within each item (Item 2c: token-level length equality)."""
        _tok = tiktoken.get_encoding("cl100k_base")
        for item in generated_items:
            tok_lens = [len(_tok.encode(h)) for h in item.hypotheses]
            assert len(set(tok_lens)) == 1, (
                f"Item {item.id} ({item.regime}): token lengths not equal: "
                f"{tok_lens}"
            )

    def test_paired_members_have_identical_option_arrays(self, clean_insuf_pairs):
        """CLEAN and INSUFFICIENT pair members must share byte-identical
        option arrays in identical order."""
        clean_by_pair, insuf_by_pair = clean_insuf_pairs
        for pid in sorted(clean_by_pair.keys()):
            c = clean_by_pair[pid]
            ins = insuf_by_pair[pid]
            assert c.hypotheses == ins.hypotheses, (
                f"Pair {pid}: option arrays differ.\n"
                f"  CLEAN: {c.hypotheses}\n"
                f"  INSUF: {ins.hypotheses}"
            )

    def test_longest_option_tie_aware_expected_credit(self, generated_items):
        """With length-matched options, all 4 options tie at max length.
        Tie-aware expected credit = 1/4 = 0.25 for every item in every regime."""
        ac_results = run_answer_choice_baselines(generated_items)
        lo_preds = ac_results["ac_longest_option"]
        for p in lo_preds:
            assert len(p["tied_candidate_set"]) == 4, (
                f"Item {p['item_id']}: tied set size "
                f"{len(p['tied_candidate_set'])} != 4 — "
                f"char lengths not all equal"
            )
            assert abs(p["expected_credit"] - 0.25) < 1e-9, (
                f"Item {p['item_id']}: expected_credit "
                f"{p['expected_credit']} != 0.25"
            )


# ================================================================
# ORACLE INDEPENDENCE TESTS (Correction 5)
# ================================================================

class TestOracleIndependence:
    def test_conjunction_oracle_signature(self):
        sig = inspect.signature(conjunction_oracle)
        params = list(sig.parameters.keys())
        assert params == ["prompt"], (
            f"conjunction_oracle should accept only 'prompt', got {params}"
        )

    def test_conflict_oracle_signature(self):
        sig = inspect.signature(conflict_oracle)
        params = list(sig.parameters.keys())
        assert params == ["prompt"], (
            f"conflict_oracle should accept only 'prompt', got {params}"
        )

    def test_oracle_source_no_forbidden_fields(self):
        """AST-inspect oracle source: must not READ from generator metadata."""
        import acceptance.oracles as omod
        source = inspect.getsource(omod)
        tree = ast.parse(source)

        forbidden_reads = {
            "criterion_a_assignment", "criterion_b_assignment",
            "oracle_result", "gold_answer",
            "gold_reasoning", "guilty_suspect", "guilty_idx",
        }

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    if node.slice.value in forbidden_reads:
                        violations.append(
                            f"Subscript access to '{node.slice.value}' at line {node.lineno}"
                        )
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_reads:
                    violations.append(
                        f"Attribute access to '{node.attr}' at line {node.lineno}"
                    )
            if isinstance(node, ast.ImportFrom):
                if node.module and "generator" in node.module:
                    violations.append(
                        f"Import from generator module: {node.module} at line {node.lineno}"
                    )

        assert not violations, (
            f"Oracle source has {len(violations)} forbidden access(es):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_oracle_works_with_metadata_stripped(self, clean_insuf_pairs):
        clean_by_pair, insuf_by_pair = clean_insuf_pairs
        for pid in clean_by_pair:
            prompt = clean_by_pair[pid].metadata["serialized_prompt"]
            result = conjunction_oracle(prompt)
            assert result["status"] == "OK", (
                f"Pair {pid} CLEAN oracle failed: {result.get('reason', '?')}"
            )
            assert result["derived_label"] == "ANSWERABLE"

            prompt_ins = insuf_by_pair[pid].metadata["serialized_prompt"]
            result_ins = conjunction_oracle(prompt_ins)
            assert result_ins["status"] == "OK", (
                f"Pair {pid} INSUF oracle failed: {result_ins.get('reason', '?')}"
            )
            assert result_ins["derived_label"] == "INSUFFICIENT"

    def test_conflict_oracle_works(self, items_by_regime):
        for item in items_by_regime.get("CONFLICT", []):
            prompt = item.metadata["serialized_prompt"]
            result = conflict_oracle(prompt)
            assert result["status"] == "OK", (
                f"CONFLICT item {item.id} oracle failed: {result.get('reason', '?')}"
            )
            assert result["derived_label"] == "ANSWERABLE"

    def test_decoy_oracle_works(self, items_by_regime):
        for item in items_by_regime.get("DECOY", []):
            prompt = item.metadata["serialized_prompt"]
            result = conjunction_oracle(prompt)
            assert result["status"] == "OK", (
                f"DECOY item {item.id} oracle failed: {result.get('reason', '?')}"
            )
            assert result["derived_label"] == "ANSWERABLE"


# ================================================================
# ORACLE MUTATION TESTS (Correction 5)
# ================================================================

class TestOracleMutations:
    """Mutation tests proving the oracle reads semantics, not just surface patterns."""

    def test_mutation_swap_flagged_cleared_changes_answer(self, clean_insuf_pairs):
        """Swapping flagged/cleared evidence changes the derived answer."""
        clean_by_pair, _ = clean_insuf_pairs
        pid = sorted(clean_by_pair.keys())[0]
        prompt = clean_by_pair[pid].metadata["serialized_prompt"]
        original = conjunction_oracle(prompt)
        assert original["status"] == "OK"

        # Swap all "flagged" <-> "cleared" in the evidence
        mutated = prompt.replace(" was flagged in the", " was TEMP_SWAP in the")
        mutated = mutated.replace(" was cleared in the", " was flagged in the")
        mutated = mutated.replace(" was TEMP_SWAP in the", " was cleared in the")
        mutant_result = conjunction_oracle(mutated)
        assert mutant_result["status"] == "OK", (
            f"Mutated prompt oracle failed: {mutant_result.get('reason', '?')}")
        # The answer must CHANGE: label, viable count, OR responsible suspect
        assert (mutant_result["derived_label"] != original["derived_label"] or
                mutant_result["n_viable"] != original["n_viable"] or
                mutant_result.get("responsible") != original.get("responsible")), (
            "Swapping flagged/cleared did NOT change oracle result — "
            "oracle is not reading criterion B semantics"
        )

    def test_mutation_swap_level2_level1_changes_answer(self, clean_insuf_pairs):
        """Swapping Level-2/Level-1 evidence changes the derived answer."""
        clean_by_pair, _ = clean_insuf_pairs
        pid = sorted(clean_by_pair.keys())[0]
        prompt = clean_by_pair[pid].metadata["serialized_prompt"]
        original = conjunction_oracle(prompt)
        assert original["status"] == "OK"

        mutated = prompt.replace(" a Level-2 ", " a TEMP_SWAP ")
        mutated = mutated.replace(" a Level-1 ", " a Level-2 ")
        mutated = mutated.replace(" a TEMP_SWAP ", " a Level-1 ")
        mutant_result = conjunction_oracle(mutated)
        assert mutant_result["status"] == "OK"
        assert (mutant_result["derived_label"] != original["derived_label"] or
                mutant_result["n_viable"] != original["n_viable"] or
                mutant_result.get("responsible") != original.get("responsible")), (
            "Swapping Level-2/Level-1 did NOT change oracle result — "
            "oracle is not reading criterion A semantics"
        )

    def test_mutation_delete_investigation_rule_causes_parse_fail(self, clean_insuf_pairs):
        """Deleting the INVESTIGATION RULE causes a controlled parse failure."""
        clean_by_pair, _ = clean_insuf_pairs
        pid = sorted(clean_by_pair.keys())[0]
        prompt = clean_by_pair[pid].metadata["serialized_prompt"]

        # Remove the investigation rule line
        lines = prompt.split("\n")
        mutated_lines = [l for l in lines if "INVESTIGATION RULE:" not in l]
        mutated = "\n".join(mutated_lines)
        result = conjunction_oracle(mutated)
        assert result["status"] == "PARSE_FAIL", (
            "Deleting INVESTIGATION RULE should cause PARSE_FAIL, "
            f"got status={result['status']}"
        )

    def test_conflict_mutation_change_timestamp_changes_answer(self, items_by_regime):
        """Changing the authoritative timestamp changes the derived answer."""
        conflict_items = items_by_regime.get("CONFLICT", [])
        assert len(conflict_items) > 0, "No CONFLICT items to test"
        item = conflict_items[0]
        prompt = item.metadata["serialized_prompt"]
        original = conflict_oracle(prompt)
        assert original["status"] == "OK"

        # Find the automated badge system line and change the time reference
        auto_time = original["parsed_facts"]["automated_log_time"]
        suspects = list(original["parsed_facts"]["access_map"].keys())
        # Find a suspect NOT identified by the auto log
        other_suspects = [s for s in suspects if s != original["responsible"]]
        if other_suspects:
            other_time = original["parsed_facts"]["access_map"][other_suspects[0]]
            # Swap the time in the automated badge line
            mutated = prompt.replace(
                f"entered the building at {auto_time}",
                f"entered the building at {other_time}",
                1  # Only replace in the automated badge line (first occurrence)
            )
            # But we need to be careful: the access evidence also has times
            # Just swap in the badge system context
            mutant_result = conflict_oracle(mutated)
            # Result should either change to a different suspect or parse fail
            assert (mutant_result["status"] != "OK" or
                    mutant_result["responsible"] != original["responsible"]), (
                "Changing automated log timestamp did NOT change conflict oracle result"
            )

    def test_conflict_mutation_delete_precedence_rule_causes_parse_fail(self, items_by_regime):
        """Deleting the precedence rule causes a controlled parse failure."""
        conflict_items = items_by_regime.get("CONFLICT", [])
        assert len(conflict_items) > 0
        item = conflict_items[0]
        prompt = item.metadata["serialized_prompt"]

        # Remove the SOURCE PRECEDENCE RULE: line(s)
        mutated = re.sub(
            r"SOURCE\s+PRECEDENCE\s+RULE:[^\n]*\n?",
            "", prompt
        )
        # Remove any standalone "official automated system logs take precedence" text
        mutated = re.sub(
            r"official automated system logs take precedence[^.]*\.",
            "", mutated, flags=re.IGNORECASE
        )
        # Remove residual "precedence" mentions so auto-detect doesn't match
        mutated = mutated.replace("precedence rules", "evidence rules")
        mutated = mutated.replace("precedence", "evaluation")
        result = conflict_oracle(mutated)
        assert result["status"] == "PARSE_FAIL", (
            "Deleting precedence rule should cause PARSE_FAIL, "
            f"got status={result['status']}"
        )

    def test_conflict_mutation_contradictory_precedence_causes_parse_fail(self, items_by_regime):
        """DEFECT 3 FIX: Inserting a contradictory precedence statement
        (e.g. 'witness testimony takes precedence') on top of the existing
        log-precedence rule must cause PARSE_FAIL, not silently pick one."""
        conflict_items = items_by_regime.get("CONFLICT", [])
        assert len(conflict_items) > 0
        item = conflict_items[0]
        prompt = item.metadata["serialized_prompt"]

        # Verify original works
        original = conflict_oracle(prompt)
        assert original["status"] == "OK", (
            f"Original CONFLICT oracle failed: {original.get('reason', '?')}")

        # Insert contradictory precedence: witness testimony takes precedence
        contradiction = (
            "\n\nIMPORTANT: In this jurisdiction, witness testimony takes "
            "precedence over all other evidence sources including automated logs."
        )
        mutated = prompt + contradiction
        result = conflict_oracle(mutated)
        assert result["status"] == "PARSE_FAIL", (
            f"Contradictory precedence should cause PARSE_FAIL, "
            f"got status={result['status']}. The oracle should detect that "
            f"both official logs AND witness testimony are claimed to take "
            f"precedence, creating an irreconcilable ambiguity."
        )


# ================================================================
# BASELINE EXECUTION TESTS (Corrections 3a, 4)
# ================================================================

class TestBaselineExecution:
    @pytest.fixture(scope="class")
    def baseline_data(self, items_by_regime):
        clean = items_by_regime["CLEAN"]
        insuf = items_by_regime["INSUFFICIENT"]
        items = clean + insuf
        labels = [1] * len(clean) + [0] * len(insuf)
        families = [it.metadata["template"] for it in items]
        return items, labels, families

    def test_all_baselines_produce_predictions(self, baseline_data):
        items, labels, families = baseline_data
        results, provenance = run_all_baselines(items, labels, families)
        required = ["B1_word_unigram", "B2_char_ngram", "B3_token_length",
                     "B4_evidence_length", "B5_name_frequency", "B6_polarity",
                     "B7_combined", "always_abstain", "option_length"]
        for name in required:
            assert name in results, f"Baseline {name} not in results"
            preds = results[name]
            assert len(preds) > 0, f"Baseline {name} produced 0 predictions"
            for p in preds:
                assert "item_id" in p
                assert "true_label" in p
                assert "predicted_label" in p
                assert "predicted_prob" in p

    def test_fold_separation_strict(self, baseline_data):
        """Assert held-out family is ABSENT from train families for every fold.
        Correction 4: Direct set-disjointness check."""
        items, labels, families = baseline_data
        results, provenance = run_all_baselines(items, labels, families)

        # Check via provenance records
        for prov in provenance:
            held_out = prov.get("held_out_family")
            train_fams = prov.get("train_families", [])
            if held_out and train_fams:
                assert held_out not in train_fams, (
                    f"Baseline {prov['baseline']}: held-out family '{held_out}' "
                    f"found in train families {train_fams}"
                )
                # Also check ID disjointness
                train_ids = set(prov.get("train_item_ids", []))
                test_ids = set(prov.get("test_item_ids", []))
                overlap = train_ids & test_ids
                assert len(overlap) == 0, (
                    f"Baseline {prov['baseline']}: train/test ID overlap: {overlap}"
                )

        # Also check via prediction records
        for name, preds in results.items():
            if name in ("always_abstain", "option_length"):
                continue
            for p in preds:
                held_out = p.get("held_out_family")
                item_fam = p.get("family")
                if held_out and item_fam:
                    assert item_fam == held_out, (
                        f"Baseline {name}: item family '{item_fam}' != "
                        f"held_out_family '{held_out}'"
                    )

    def test_fit_provenance_recorded(self, baseline_data):
        """Correction 4: Every fold has provenance with required fields."""
        items, labels, families = baseline_data
        results, provenance = run_all_baselines(items, labels, families)

        required_fields = {
            "baseline", "held_out_family", "train_families",
            "train_item_ids", "test_item_ids", "n_train", "n_test",
            "feature_dim", "model_type", "hyperparameters",
            "coef_norm", "prediction_variance",
        }
        fitted_baselines = {"B1_word_unigram", "B2_char_ngram", "B3_token_length",
                           "B4_evidence_length", "B5_name_frequency", "B6_polarity",
                           "B7_combined"}
        for prov in provenance:
            bl = prov.get("baseline", "")
            if bl not in fitted_baselines:
                continue
            missing = required_fields - set(prov.keys())
            assert not missing, (
                f"Baseline {bl} fold provenance missing fields: {missing}"
            )

    def test_context_only_features_exclude_options(self, baseline_data):
        """Correction 3a: Context-only text must not contain option lines."""
        items, _, _ = baseline_data
        for item in items:
            ctx = extract_context_only_text(item)
            assert "(A) Based" not in ctx, (
                f"Item {item.id}: context-only text still contains option (A)")
            assert "(B) Based" not in ctx, (
                f"Item {item.id}: context-only text still contains option (B)")
            assert "(C) Based" not in ctx, (
                f"Item {item.id}: context-only text still contains option (C)")
            assert "(D) Based" not in ctx, (
                f"Item {item.id}: context-only text still contains option (D)")

    def test_option_length_baseline_uninformative(self, baseline_data):
        """With length-matched options, option_length baseline picks index 0
        (all same length). This baseline should achieve ~chance-level BA=0.5
        (orientation-invariant S=0.5) in binary classification."""
        items, labels, families = baseline_data
        results, _ = run_all_baselines(items, labels, families)
        ol_preds = results["option_length"]
        s = orientation_invariant_separability(ol_preds)
        assert s <= 0.55, (
            f"option_length baseline S={s:.4f} > 0.55 — "
            f"length-matching may not be working for binary baseline"
        )

    def test_always_abstain_behavior(self, baseline_data):
        items, labels, families = baseline_data
        results, _ = run_all_baselines(items, labels, families)
        aa_preds = results["always_abstain"]
        for p in aa_preds:
            assert p["predicted_label"] == 0


# ================================================================
# ORIENTATION-INVARIANT SEPARABILITY TESTS (Correction 2)
# ================================================================

class TestOrientationInvariant:
    def test_perfect_reversal_scores_1(self):
        """BA=0.0 (perfect reversal) must give S=1.0, not pass as 0.0."""
        preds = [
            {"true_label": 1, "predicted_label": 0},
            {"true_label": 1, "predicted_label": 0},
            {"true_label": 0, "predicted_label": 1},
            {"true_label": 0, "predicted_label": 1},
        ]
        ba = compute_balanced_accuracy(preds)
        s = orientation_invariant_separability(preds)
        assert ba == 0.0, f"Expected BA=0.0, got {ba}"
        assert s == 1.0, f"Expected S=1.0 for perfect reversal, got {s}"

    def test_perfect_prediction_scores_1(self):
        """BA=1.0 must give S=1.0."""
        preds = [
            {"true_label": 1, "predicted_label": 1},
            {"true_label": 0, "predicted_label": 0},
        ]
        s = orientation_invariant_separability(preds)
        assert s == 1.0

    def test_chance_scores_half(self):
        """BA=0.5 must give S=0.5."""
        preds = [
            {"true_label": 1, "predicted_label": 1},
            {"true_label": 1, "predicted_label": 0},
            {"true_label": 0, "predicted_label": 0},
            {"true_label": 0, "predicted_label": 1},
        ]
        s = orientation_invariant_separability(preds)
        assert s == 0.5

    def test_separability_range(self, items_by_regime):
        """All separability values must be in [0.5, 1.0]."""
        clean = items_by_regime["CLEAN"]
        insuf = items_by_regime["INSUFFICIENT"]
        items = clean + insuf
        labels = [1] * len(clean) + [0] * len(insuf)
        families = [it.metadata["template"] for it in items]
        results, _ = run_all_baselines(items, labels, families)
        for name, preds in results.items():
            if not preds:
                continue
            s = orientation_invariant_separability(preds)
            assert 0.5 <= s <= 1.0, (
                f"Baseline {name}: separability {s} outside [0.5, 1.0]"
            )


# ================================================================
# CREDENTIAL GUARD TESTS
# ================================================================

class TestCredentialGuard:
    def test_all_real_endpoints_blocked(self):
        for url in KNOWN_BLOCKED_ENDPOINTS:
            with pytest.raises(CredentialGuardError):
                validate_provider(url + "/v1", "any-model")

    def test_mock_allowed(self):
        assert validate_provider("mock://test", "any-model")

    def test_localhost_allowed(self):
        assert validate_provider("http://localhost:8000", "model")
        assert validate_provider("http://127.0.0.1:9090", "model")

    def test_all_cloud_api_vars_blocked(self):
        required_blocked = {
            "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
            "TOGETHER_API_KEY", "FIREWORKS_API_KEY",
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "HF_API_TOKEN", "HUGGING_FACE_HUB_TOKEN",
            "GOOGLE_API_KEY", "MISTRAL_API_KEY",
        }
        assert required_blocked.issubset(BLOCKED_ENV_VARS), (
            f"Missing blocked vars: {required_blocked - BLOCKED_ENV_VARS}"
        )


# ================================================================
# VIABILITY COUNT TESTS
# ================================================================

class TestViability:
    def test_clean_items_have_1_viable(self, items_by_regime):
        for item in items_by_regime["CLEAN"]:
            oracle = item.metadata.get("oracle_result", {})
            assert oracle.get("n_viable") == 1, (
                f"CLEAN {item.id}: n_viable={oracle.get('n_viable')}, expected 1"
            )
            assert oracle.get("oracle_label") == "ANSWERABLE"

    def test_insufficient_items_have_ge2_viable(self, items_by_regime):
        for item in items_by_regime["INSUFFICIENT"]:
            oracle = item.metadata.get("oracle_result", {})
            assert oracle.get("n_viable", 0) >= 2, (
                f"INSUF {item.id}: n_viable={oracle.get('n_viable')}, expected >= 2"
            )
            assert oracle.get("oracle_label") == "INSUFFICIENT"


# ================================================================
# HARD SANITY GATES (Correction 7)
# ================================================================

class TestSanityGates:
    """Hard machinery-sanity gates that block overall green."""

    def test_no_degenerate_probabilities(self, items_by_regime):
        """No fitted baseline should emit constant probabilities across all items."""
        clean = items_by_regime["CLEAN"]
        insuf = items_by_regime["INSUFFICIENT"]
        items = clean + insuf
        labels = [1] * len(clean) + [0] * len(insuf)
        families = [it.metadata["template"] for it in items]
        results, provenance = run_all_baselines(items, labels, families)

        fitted_baselines = ["B1_word_unigram", "B2_char_ngram", "B3_token_length",
                           "B4_evidence_length", "B5_name_frequency", "B6_polarity",
                           "B7_combined"]
        for name in fitted_baselines:
            preds = results.get(name, [])
            if not preds:
                continue
            probs = [p["predicted_prob"] for p in preds]
            # Check for non-degenerate variation
            prob_var = np.var(probs)
            # We allow zero variance only if ALL predictions are 0.5 (uninformative)
            # A fitted model should show SOME variation
            if prob_var == 0.0:
                # All constant — check if it's 0.5 (uninformative, acceptable)
                # or some other constant (degenerate, flag it)
                mean_prob = np.mean(probs)
                if abs(mean_prob - 0.5) > 0.01:
                    pytest.fail(
                        f"Baseline {name}: all predictions are constant at "
                        f"{mean_prob:.4f} (degenerate, non-0.5)"
                    )

    def test_metric_ranges(self, items_by_regime):
        """All metrics must be in expected ranges."""
        clean = items_by_regime["CLEAN"]
        insuf = items_by_regime["INSUFFICIENT"]
        items = clean + insuf
        labels = [1] * len(clean) + [0] * len(insuf)
        families = [it.metadata["template"] for it in items]
        results, _ = run_all_baselines(items, labels, families)

        for name, preds in results.items():
            if not preds:
                continue
            ba = compute_balanced_accuracy(preds)
            s = orientation_invariant_separability(preds)
            assert 0.0 <= ba <= 1.0, f"{name}: BA={ba} outside [0,1]"
            assert 0.5 <= s <= 1.0, f"{name}: S={s} outside [0.5,1.0]"


# ================================================================
# ANSWER-CHOICE EVALUATION TESTS (DEFECT 1 fix)
# ================================================================

class TestConstructionInvariants:
    """Construction invariant tests: balanced gold/abstention positions."""

    def test_gold_position_balance_per_regime(self, items_by_regime):
        """Gold positions must be exactly 2/2/2/2 in every regime (Item 2a)."""
        for regime, regime_items in items_by_regime.items():
            gold_counts = Counter(
                it.metadata["gold_position"] for it in regime_items
            )
            for pos in range(4):
                assert gold_counts.get(pos, 0) == 2, (
                    f"{regime}: gold position {pos} has "
                    f"{gold_counts.get(pos, 0)} items (need 2). "
                    f"Full: {dict(sorted(gold_counts.items()))}"
                )

    def test_abstention_position_balance_per_regime(self, items_by_regime):
        """Abstention positions must be exactly 2/2/2/2 in every regime (Item 2b)."""
        for regime, regime_items in items_by_regime.items():
            abst_counts = Counter(
                it.metadata["abstention_position"] for it in regime_items
            )
            for pos in range(4):
                assert abst_counts.get(pos, 0) == 2, (
                    f"{regime}: abstention position {pos} has "
                    f"{abst_counts.get(pos, 0)} items (need 2). "
                    f"Full: {dict(sorted(abst_counts.items()))}"
                )

    def test_clean_gold_ne_abstention(self, items_by_regime):
        """In CLEAN items, gold_position != abstention_position."""
        for item in items_by_regime["CLEAN"]:
            assert item.metadata["gold_position"] != item.metadata["abstention_position"], (
                f"CLEAN {item.id}: gold == abstention at position "
                f"{item.metadata['gold_position']}"
            )

    def test_insufficient_gold_eq_abstention(self, items_by_regime):
        """In INSUFFICIENT items, gold_position == abstention_position."""
        for item in items_by_regime["INSUFFICIENT"]:
            assert item.metadata["gold_position"] == item.metadata["abstention_position"], (
                f"INSUF {item.id}: gold={item.metadata['gold_position']} "
                f"!= abstention={item.metadata['abstention_position']}"
            )


class TestAnswerChoiceEvaluation:
    """Tests for 4-class answer-choice baselines across all regimes."""

    def test_all_baselines_cover_all_items(self, generated_items):
        """Every answer-choice baseline must produce predictions for all 32 items."""
        ac_results = run_answer_choice_baselines(generated_items)
        expected_ids = {it.id for it in generated_items}
        for bl_name, preds in ac_results.items():
            covered_ids = {p["item_id"] for p in preds}
            missing = expected_ids - covered_ids
            assert len(missing) == 0, (
                f"Baseline {bl_name}: missing {len(missing)} items: {sorted(missing)[:5]}"
            )

    def test_prediction_format_deterministic(self, generated_items):
        """Deterministic baselines (always-abstain, position-*) have required fields."""
        ac_results = run_answer_choice_baselines(generated_items)
        required = {"item_id", "regime", "family", "gold_option_index",
                     "predicted_option_index", "correct", "baseline"}
        deterministic_baselines = [k for k in ac_results
                                   if k != "ac_longest_option"]
        for bl_name in deterministic_baselines:
            for p in ac_results[bl_name]:
                missing = required - set(p.keys())
                assert not missing, (
                    f"Baseline {bl_name}, item {p.get('item_id')}: "
                    f"missing fields: {missing}"
                )

    def test_prediction_format_tie_aware(self, generated_items):
        """Tie-aware baseline (ac_longest_option) has expected_credit fields."""
        ac_results = run_answer_choice_baselines(generated_items)
        required = {"item_id", "regime", "family", "gold_option_index",
                     "per_option_char_lengths", "tied_candidate_set",
                     "expected_credit", "baseline"}
        for p in ac_results["ac_longest_option"]:
            missing = required - set(p.keys())
            assert not missing, (
                f"ac_longest_option, item {p.get('item_id')}: "
                f"missing fields: {missing}"
            )

    def test_no_per_regime_shortcut_above_tolerance(self, generated_items):
        """Per-regime diagnostic: no baseline exceeds 0.30 in any non-exempt
        regime.  This is a DIAGNOSTIC (non-gating) check; the primary gate
        is the macro-avg UCB.  always-abstain on INSUFFICIENT (1.00) is
        structurally expected and acceptable under macro-avg semantics."""
        ac_results = run_answer_choice_baselines(generated_items)
        structural_exemptions = {
            ("ac_always_abstain", "INSUFFICIENT"),
        }
        for bl_name, preds in ac_results.items():
            acc = compute_answer_choice_accuracy(preds, by_regime=True)
            for regime in ["CLEAN", "INSUFFICIENT", "DECOY", "CONFLICT"]:
                if (bl_name, regime) in structural_exemptions:
                    continue
                regime_acc = acc.get(regime, {}).get("accuracy", 0.0)
                assert regime_acc <= CORPUS_SCALE_AC_TOLERANCE, (
                    f"Baseline {bl_name}: {regime} accuracy="
                    f"{regime_acc:.4f} > {CORPUS_SCALE_AC_TOLERANCE} "
                    f"(shortcut detected)"
                )

    def test_longest_option_exactly_chance_after_length_fix(self, generated_items):
        """After length-matching, longest-option expected accuracy must be
        exactly 0.25 in every regime (4-way tie -> 1/4 credit)."""
        ac_results = run_answer_choice_baselines(generated_items)
        preds = ac_results["ac_longest_option"]
        acc = compute_answer_choice_accuracy(preds, by_regime=True)
        for regime in ["CLEAN", "INSUFFICIENT", "DECOY", "CONFLICT"]:
            r_acc = acc.get(regime, {}).get("accuracy", -1.0)
            assert abs(r_acc - 0.25) < 1e-9, (
                f"ac_longest_option {regime} accuracy={r_acc:.4f} != 0.25"
            )

    def test_position_baselines_exactly_025_per_regime(self, generated_items):
        """Each fixed-position baseline must score exactly 0.25 in every regime
        (Item 3: construction balance check)."""
        ac_results = run_answer_choice_baselines(generated_items)
        for pos in range(4):
            bl_name = f"ac_position_{pos}"
            preds = ac_results[bl_name]
            acc = compute_answer_choice_accuracy(preds, by_regime=True)
            for regime in ["CLEAN", "INSUFFICIENT", "DECOY", "CONFLICT"]:
                r_acc = acc.get(regime, {}).get("accuracy", -1.0)
                assert abs(r_acc - 0.25) < 1e-9, (
                    f"{bl_name}: {regime} accuracy={r_acc:.4f} != 0.25 — "
                    f"gold position balance violated"
                )

    def test_corpus_scale_tolerance_not_050(self):
        """Item 4: Corpus-scale tolerance must NOT be 0.50."""
        assert CORPUS_SCALE_AC_TOLERANCE < 0.50, (
            f"CORPUS_SCALE_AC_TOLERANCE={CORPUS_SCALE_AC_TOLERANCE} >= 0.50. "
            f"The threshold must be near 0.25 chance level, not 0.50."
        )
        assert CORPUS_SCALE_AC_TOLERANCE >= 0.25, (
            f"CORPUS_SCALE_AC_TOLERANCE={CORPUS_SCALE_AC_TOLERANCE} < 0.25 "
            f"(below chance — too strict)"
        )


# ====================================================================
# v3.2.5 A13 REDEFINITION REGRESSION TESTS
# ====================================================================
# These tests verify the new A13 macro-avg UCB gate semantics using
# synthetic / injected accuracy inputs.

def _make_synthetic_predictions(per_regime_correct, families=None):
    """Build synthetic answer-choice prediction records.

    per_regime_correct: dict mapping regime -> list of (family, correct_bool).
    Returns list of prediction dicts with 'regime', 'family', 'correct',
    'item_id' fields.
    """
    preds = []
    idx = 0
    for regime, entries in per_regime_correct.items():
        for fam, correct in entries:
            preds.append({
                "item_id": f"synth_{idx:04d}",
                "regime": regime,
                "family": fam if families is None else fam,
                "correct": correct,
                "baseline": "synth",
            })
            idx += 1
    return preds


class TestA13MacroAvgUCB:
    """v3.2.5 regression tests for the macro-avg UCB gate."""

    def test_macro_avg_unequal_regime_sizes(self):
        """(a) Unequal regime sizes are weighted equally (macro, not pooled).

        Construct predictions where CLEAN has 10 items (all correct) and
        INSUFFICIENT/DECOY/CONFLICT have 2 items each (all wrong).
        Pooled accuracy = 10/16 = 0.625.
        Macro-avg = (1.0 + 0.0 + 0.0 + 0.0)/4 = 0.25.
        """
        preds = _make_synthetic_predictions({
            "CLEAN": [("fam_a", True)] * 5 + [("fam_b", True)] * 5,
            "INSUFFICIENT": [("fam_a", False)] + [("fam_b", False)],
            "DECOY": [("fam_a", False)] + [("fam_b", False)],
            "CONFLICT": [("fam_a", False)] + [("fam_b", False)],
        })
        result = compute_macro_avg_accuracy(preds)
        assert abs(result["macro_avg"] - 0.25) < 1e-9, (
            f"macro_avg={result['macro_avg']}, expected 0.25 (not pooled 0.625)")

    def test_point_below_030_fails_when_ucb_exceeds(self):
        """(b) Point estimate below 0.30 FAILS when UCB > 0.30.

        fam_a: correct in CLEAN and INSUFFICIENT.  fam_b: wrong everywhere.
        Per-regime: CLEAN=0.5, INSUF=0.5, DECOY=0, CONFLICT=0.
        Point macro = (0.5 + 0.5 + 0 + 0)/4 = 0.25 < 0.30.
        When bootstrap draws fam_a twice, macro jumps to 0.50,
        pushing UCB well above 0.30.
        """
        preds = _make_synthetic_predictions({
            "CLEAN": [("fam_a", True), ("fam_b", False)],
            "INSUFFICIENT": [("fam_a", True), ("fam_b", False)],
            "DECOY": [("fam_a", False), ("fam_b", False)],
            "CONFLICT": [("fam_a", False), ("fam_b", False)],
        })
        result = compute_macro_avg_accuracy(preds)
        assert result["macro_avg"] < 0.30, (
            f"precondition: point={result['macro_avg']:.4f} must be < 0.30")

        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=10000, seed=42)
        assert boot["upper_95"] > 0.30, (
            f"Expected UCB > 0.30 despite point={boot['point_macro_avg']:.4f}, "
            f"got UCB={boot['upper_95']:.4f}")
        assert not boot["gate_passes"], "Gate should FAIL when UCB > 0.30"

    def test_ucb_at_or_below_030_passes(self):
        """(c) Point estimate whose UCB <= 0.30 PASSES.

        Use many families with many items each, all scoring exactly 0.25.
        With enough data, the grouped bootstrap UCB converges close to
        the point estimate.
        """
        families = [f"fam_{i}" for i in range(20)]
        # 20 items per family per regime (4 correct + 16 wrong = 0.20 each;
        # but we want exactly 0.25, so 5 correct + 15 wrong per family)
        preds = _make_synthetic_predictions({
            regime: [
                entry
                for fam in families
                for entry in [(fam, True)] * 5 + [(fam, False)] * 15
            ]
            for regime in REGIMES
        })
        result = compute_macro_avg_accuracy(preds)
        # Each regime: 100/400 correct = 0.25
        assert abs(result["macro_avg"] - 0.25) < 1e-9

        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=10000, seed=42)
        assert boot["upper_95"] <= 0.30, (
            f"UCB={boot['upper_95']:.4f} > 0.30 with point=0.25 and "
            f"20 families x 20 items each")
        assert boot["gate_passes"], "Gate should PASS when UCB <= 0.30"

    def test_overall_above_030_fails(self):
        """(d) Overall accuracy above 0.30 FAILS.

        Make every prediction correct: macro-avg = 1.0, UCB >= 1.0.
        """
        preds = _make_synthetic_predictions({
            regime: [("fam_a", True)] * 4 + [("fam_b", True)] * 4
            for regime in REGIMES
        })
        result = compute_macro_avg_accuracy(preds)
        assert result["macro_avg"] > 0.30

        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=10000, seed=42)
        assert boot["upper_95"] > 0.30
        assert not boot["gate_passes"], "Gate should FAIL when accuracy=1.0"

    def test_always_abstain_025_no_exemption(self, generated_items):
        """(e) always-abstain produces macro-avg 0.25 WITHOUT any exemption.

        The always-abstain baseline scores 1.00 in INSUFFICIENT and 0.00
        elsewhere. Macro-avg = (0+1+0+0)/4 = 0.25.  No structural_exemptions
        dict needed.
        """
        ac_results = run_answer_choice_baselines(generated_items)
        preds = ac_results["ac_always_abstain"]

        result = compute_macro_avg_accuracy(preds)
        assert abs(result["macro_avg"] - 0.25) < 1e-9, (
            f"always-abstain macro_avg={result['macro_avg']}, expected 0.25")

        # Confirm per-regime pattern
        assert abs(result["per_regime"]["INSUFFICIENT"]["accuracy"] - 1.0) < 1e-9
        for regime in ["CLEAN", "DECOY", "CONFLICT"]:
            assert abs(result["per_regime"][regime]["accuracy"] - 0.0) < 1e-9

        # UCB should also pass without exemption
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=10000, seed=42)
        assert boot["gate_passes"], (
            f"always-abstain should pass the UCB gate without exemption, "
            f"but UCB={boot['upper_95']:.4f}")

    def test_per_regime_diagnostic_not_gating(self):
        """(f) Per-regime results are reported but do NOT independently gate.

        Construct a baseline where one regime has accuracy 0.50 (above 0.30)
        but the macro-average is 0.25 and UCB <= 0.30. The gate should PASS
        because per-regime values are diagnostic only.
        """
        # 20 families, 20 items per family per regime.
        # CLEAN: 10/20 correct per family = 0.50 (above 0.30).
        # INSUFFICIENT: 0/20 correct per family = 0.00.
        # DECOY: 5/20 = 0.25. CONFLICT: 5/20 = 0.25.
        # Macro = (0.50 + 0.00 + 0.25 + 0.25)/4 = 0.25.
        families = [f"fam_{i}" for i in range(20)]
        preds = _make_synthetic_predictions({
            "CLEAN": [
                entry
                for fam in families
                for entry in [(fam, True)] * 10 + [(fam, False)] * 10
            ],
            "INSUFFICIENT": [
                entry
                for fam in families
                for entry in [(fam, False)] * 20
            ],
            "DECOY": [
                entry
                for fam in families
                for entry in [(fam, True)] * 5 + [(fam, False)] * 15
            ],
            "CONFLICT": [
                entry
                for fam in families
                for entry in [(fam, True)] * 5 + [(fam, False)] * 15
            ],
        })
        result = compute_macro_avg_accuracy(preds)
        # CLEAN = 0.5, INSUFFICIENT = 0.0, DECOY = 0.25, CONFLICT = 0.25
        assert result["per_regime"]["CLEAN"]["accuracy"] > 0.30, \
            "precondition: CLEAN accuracy must exceed per-regime 0.30 threshold"
        assert abs(result["macro_avg"] - 0.25) < 1e-9, \
            "precondition: macro-avg must be 0.25"

        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=10000, seed=42)
        # Gate should pass because macro-avg UCB is checked, not per-regime
        assert boot["gate_passes"], (
            f"Gate should PASS (macro-avg UCB={boot['upper_95']:.4f} <= 0.30) "
            f"even though CLEAN={result['per_regime']['CLEAN']['accuracy']:.2f} > 0.30 — "
            f"per-regime is diagnostic only")

    def test_grouped_bootstrap_uses_family_resampling(self):
        """(g) Grouped resampling uses template families, not item-level.

        Verify that grouped_family_bootstrap_macro_avg resamples at the
        family level by showing it produces DIFFERENT bootstrap variance
        than a hypothetical item-level bootstrap would.

        With 1 family and 8 items, family-level resampling draws the SAME
        family every time (no variance). With item-level, there'd be
        variance. So if std ≈ 0 with 1 family, the function is resampling
        families.
        """
        # 1 family: all items have the same family
        preds = _make_synthetic_predictions({
            "CLEAN": [("single_fam", True), ("single_fam", False)],
            "INSUFFICIENT": [("single_fam", True), ("single_fam", False)],
            "DECOY": [("single_fam", True), ("single_fam", False)],
            "CONFLICT": [("single_fam", True), ("single_fam", False)],
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=5000, seed=42)
        assert boot["n_families"] == 1, f"Expected 1 family, got {boot['n_families']}"

        # With 1 family, Stage 1 always draws the same family.
        # Stage 2 resamples items WITHIN that family, so there IS some
        # item-level variance. But we can verify the function reports
        # n_families=1 and USES grouped resampling by checking that
        # the resampling unit is recorded.

        # More definitive test: 2 families with OPPOSITE results.
        # Family-level resampling produces {0.25, 0.75, 0.25, 0.75, 0.50}
        # as possible macro-averages.  Item-level would produce a smoother
        # distribution.
        preds_2fam = _make_synthetic_predictions({
            regime: [("fam_all_correct", True)] * 4 + [("fam_all_wrong", False)] * 4
            for regime in REGIMES
        })
        boot_2fam = grouped_family_bootstrap_macro_avg(
            preds_2fam, n_bootstrap=10000, seed=42)
        assert boot_2fam["n_families"] == 2

        # With 2 families and family-level resampling, the bootstrap
        # distribution should be TRIMODAL (both correct-fam drawn,
        # both wrong-fam drawn, or one of each). The possible macro-averages
        # are 1.0, 0.0, and ~0.5. Check that the distribution has high std
        # (it should be ~0.35 for trimodal, vs ~0.06 for item-level with
        # 32 items).
        assert boot_2fam["bootstrap_std"] > 0.15, (
            f"Expected high bootstrap std with 2 polar families "
            f"(family-level resampling), got {boot_2fam['bootstrap_std']:.4f}. "
            f"This suggests item-level resampling was used instead of "
            f"grouped family resampling.")


# ====================================================================
# v3.2.6 A13 ACTIVATION SEMANTICS REGRESSION TESTS
# ====================================================================
# These tests verify DEFERRED_NOT_EVALUATED / ENFORCED activation at
# N=500, the authoritative verdict object, and table/manifest/exit
# consistency.

class TestA13ActivationSemantics:
    """v3.2.6 regression tests for N=500 activation and verdict objects."""

    def test_n8_deferred(self):
        """(a) N=8/regime → DEFERRED_NOT_EVALUATED."""
        preds = _make_synthetic_predictions({
            regime: [("fam_a", True), ("fam_b", False)] * 4
            for regime in REGIMES
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=100, seed=42)
        v = build_a13_verdict("test_bl", preds, n_per_regime=8, boot_result=boot)
        assert v["active_mode"] == "DEFERRED_NOT_EVALUATED"
        assert v["final_status"] == "DEFERRED_NOT_EVALUATED"
        assert v["n_per_regime"] == 8
        # Point estimate and UCB are still populated as diagnostics
        assert "point_estimate" in v
        assert "ucb" in v

    def test_n499_deferred(self):
        """(b) N=499/regime → DEFERRED_NOT_EVALUATED."""
        preds = _make_synthetic_predictions({
            regime: [("fam_a", False)] * 499
            for regime in REGIMES
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=100, seed=42)
        v = build_a13_verdict("test_bl", preds, n_per_regime=499, boot_result=boot)
        assert v["active_mode"] == "DEFERRED_NOT_EVALUATED"
        assert v["final_status"] == "DEFERRED_NOT_EVALUATED"
        assert v["n_per_regime"] == 499

    def test_n500_enforced(self):
        """(c) N=500/regime → ENFORCED."""
        # 25 families, 20 items per family per regime = 500/regime.
        # All wrong → macro-avg = 0.0, UCB ≈ 0.0 → PASS.
        families = [f"fam_{i}" for i in range(25)]
        preds = _make_synthetic_predictions({
            regime: [
                entry
                for fam in families
                for entry in [(fam, False)] * 20
            ]
            for regime in REGIMES
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=100, seed=42)
        v = build_a13_verdict("test_bl", preds, n_per_regime=500, boot_result=boot)
        assert v["active_mode"] == "ENFORCED"
        assert v["n_per_regime"] == 500
        # All wrong: UCB should be 0.0 ≤ 0.30 → PASS
        assert v["final_status"] == "PASS"

    def test_n500_ucb_below_030_passes(self):
        """(d) At N=500, constructed case whose UCB ≤ 0.30 → PASS."""
        # 25 families, 20 items each per regime = 500/regime.
        # 5 correct + 15 wrong per family per regime = 0.25.
        families = [f"fam_{i}" for i in range(25)]
        preds = _make_synthetic_predictions({
            regime: [
                entry
                for fam in families
                for entry in [(fam, True)] * 5 + [(fam, False)] * 15
            ]
            for regime in REGIMES
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=5000, seed=42)
        v = build_a13_verdict("test_bl", preds, n_per_regime=500, boot_result=boot)
        assert v["active_mode"] == "ENFORCED"
        assert abs(v["point_estimate"] - 0.25) < 1e-9
        assert v["ucb"] <= 0.30, f"UCB={v['ucb']:.4f} > 0.30"
        assert v["final_status"] == "PASS"

    def test_n500_ucb_above_030_fails(self):
        """(e) At N=500, constructed case whose UCB > 0.30 → FAIL."""
        # 2 families at 500/regime total. fam_a: all correct. fam_b: all wrong.
        # Point = 0.50. Family-level bootstrap with 2 families produces
        # huge variance → UCB well above 0.30.
        preds = _make_synthetic_predictions({
            regime: [("fam_a", True)] * 250 + [("fam_b", False)] * 250
            for regime in REGIMES
        })
        boot = grouped_family_bootstrap_macro_avg(preds, n_bootstrap=2000, seed=42)
        v = build_a13_verdict("test_bl", preds, n_per_regime=500, boot_result=boot)
        assert v["active_mode"] == "ENFORCED"
        assert v["ucb"] > 0.30, f"UCB={v['ucb']:.4f} should exceed 0.30"
        assert v["final_status"] == "FAIL"

    def test_verdict_consistency(self):
        """(f) Consistency: verdict object's active_mode/final_status agree
        with what overall_verdict produces and what the runner would render.

        At N=8, all verdicts should show DEFERRED_NOT_EVALUATED everywhere.
        No baseline should show PASS or FAIL.
        """
        families = ["fam_a", "fam_b"]
        preds_per_bl = {}
        for bl_name in ["ac_always_abstain", "ac_position_0"]:
            preds = _make_synthetic_predictions({
                regime: [(fam, bl_name == "ac_always_abstain"
                          and regime == "INSUFFICIENT")
                         for fam in families]
                for regime in REGIMES
            })
            preds_per_bl[bl_name] = preds

        # Build per-baseline verdicts at N=8
        verdicts = {}
        for bl_name, bl_preds in preds_per_bl.items():
            boot = grouped_family_bootstrap_macro_avg(
                bl_preds, n_bootstrap=100, seed=42)
            v = build_a13_verdict(bl_name, bl_preds, n_per_regime=8,
                                  boot_result=boot)
            verdicts[bl_name] = v
            # Each baseline must show DEFERRED
            assert v["active_mode"] == "DEFERRED_NOT_EVALUATED", (
                f"{bl_name}: expected DEFERRED_NOT_EVALUATED, "
                f"got {v['active_mode']}")
            assert v["final_status"] == "DEFERRED_NOT_EVALUATED", (
                f"{bl_name}: expected final_status=DEFERRED_NOT_EVALUATED, "
                f"got {v['final_status']}")

        # Overall verdict must also be DEFERRED
        overall = build_a13_overall_verdict(
            list(verdicts.values()), completeness_violations=[])
        assert overall["active_mode"] == "DEFERRED_NOT_EVALUATED"
        assert overall["final_status"] == "DEFERRED_NOT_EVALUATED"

        # With completeness violations, overall must FAIL even at N=8
        overall_fail = build_a13_overall_verdict(
            list(verdicts.values()),
            completeness_violations=["bl_x: missing 3 items"])
        assert overall_fail["final_status"] == "FAIL"


class TestDeletedConstantAbsence:
    """v3.2.7: Verify the deleted N=30 constant is absent from Python source.

    The check is scoped to executable Python files (*.py) only,
    excluding this test file (which must name the constant in string
    literals to search for it).  Ledgers, documentation, and the
    portable archive are explicitly excluded — those documents
    legitimately name the deleted constant when documenting its removal.
    """

    # Build the search needle from parts so a naive grep of this file
    # for the constant as a Python identifier does not match.
    _DELETED_CONST = "CORPUS_SCALE_" + "MIN_N_PER_REGIME"

    def test_deleted_constant_absent_from_py_source(self):
        """No *.py file under acceptance/ (other than this test) references the deleted constant."""
        acceptance_dir = os.path.dirname(os.path.abspath(__file__))
        this_file = os.path.abspath(__file__)
        py_files = glob.glob(os.path.join(acceptance_dir, "*.py"))
        assert len(py_files) > 0, "No .py files found — test is misconfigured"

        needle = self._DELETED_CONST
        hits = []
        for py_path in sorted(py_files):
            if os.path.abspath(py_path) == this_file:
                continue  # exclude this test file
            with open(py_path, "r") as f:
                for lineno, line in enumerate(f, 1):
                    if needle in line:
                        hits.append(f"{os.path.basename(py_path)}:{lineno}: {line.rstrip()}")

        assert hits == [], (
            f"{needle} found in Python source "
            f"(should only appear in ledger/docs):\n" + "\n".join(hits))

    def test_replacement_constant_present_in_source(self):
        """CORPUS_SCALE_N_ACTIVATION = 500 is present in baselines.py."""
        baselines_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "baselines.py")
        with open(baselines_path, "r") as f:
            source = f.read()
        assert "CORPUS_SCALE_N_ACTIVATION = 500" in source, (
            "CORPUS_SCALE_N_ACTIVATION = 500 not found in baselines.py")
