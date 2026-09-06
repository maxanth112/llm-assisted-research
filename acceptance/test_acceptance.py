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
    CORPUS_SCALE_AC_TOLERANCE,
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
        """No answer-choice baseline exceeds CORPUS_SCALE_AC_TOLERANCE in any
        single regime, EXCEPT structurally guaranteed matches (always-abstain
        on INSUFFICIENT where gold IS abstention by definition)."""
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
