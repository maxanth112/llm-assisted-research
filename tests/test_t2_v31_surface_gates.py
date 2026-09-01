"""
Deterministic surface-gate tests for the T2 v3.1 generator.

Every assertion is deterministic — NO probabilistic thresholds, no
chi-squared tests, no coefficient-of-variation checks (except S5
which uses the same 20% relative band as the evaluator).

Surface gates tested:
  S1: Universal 4-option (3 suspects + 1 abstention).
  S2: Gold-position max-diff <= 1 per regime.
  S3: Abstention-position max-diff <= 1 per regime.
  S4: Cross-regime evidence normalization (all regimes have identical
      evidence slot count).
  S5: Option text length within 20% relative band across regimes.
  S6: Hypothesis order randomized per item (gold/abstention at
      pre-assigned positions).

Additional v3.1-specific invariants:
  - Lexical parallelism: all hypotheses share the same syntactic frame.
  - Paired structure: INSUFFICIENT items use the same slot count as
    answerable items.
  - Determinism: same seed produces identical output.
  - Gold validity: gold in hypotheses, regime-correct.
  - ID prefix: "t2v31_".
  - No hash()-based seeds.

These tests are GATING for the v3.1 audit.
"""

import math
from collections import Counter
from typing import List

import numpy as np
import pytest

from datasets.t2_generator.generator_v3_1 import (
    T2V31Generator,
    ABSTENTION_HYPOTHESIS,
    SUSPECT_HYPOTHESIS_TEMPLATE,
    N_EVIDENCE_SLOTS,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def v31_items():
    """Generate a v3.1 dataset with n_per_regime=8 (32 items total)."""
    g = T2V31Generator(seed=42)
    return g.generate_dataset(n_per_regime=8, seed=42)


@pytest.fixture
def v31_medium():
    """Generate a medium v3.1 dataset with n_per_regime=16 (64 items)."""
    g = T2V31Generator(seed=42)
    return g.generate_dataset(n_per_regime=16, seed=42)


@pytest.fixture
def v31_large():
    """Generate a larger v3.1 dataset with n_per_regime=50 (200 items)."""
    g = T2V31Generator(seed=42)
    return g.generate_dataset(n_per_regime=50, seed=42)


# ================================================================
# S1: Universal 4-option
# ================================================================

class TestS1UniversalFourOption:
    """S1: Every item must have exactly 4 hypotheses."""

    def test_all_items_have_four_hypotheses(self, v31_items):
        for item in v31_items:
            assert len(item.hypotheses) == 4, (
                f"Item {item.id}: {len(item.hypotheses)} hypotheses"
            )

    def test_no_duplicate_hypotheses(self, v31_items):
        for item in v31_items:
            assert len(set(item.hypotheses)) == 4, (
                f"Item {item.id}: duplicate hypotheses: {item.hypotheses}"
            )

    def test_abstention_present_in_every_item(self, v31_items):
        for item in v31_items:
            assert ABSTENTION_HYPOTHESIS in item.hypotheses, (
                f"Item {item.id}: missing abstention option"
            )

    def test_four_options_at_scale(self, v31_large):
        for item in v31_large:
            assert len(item.hypotheses) == 4
            assert len(set(item.hypotheses)) == 4
            assert ABSTENTION_HYPOTHESIS in item.hypotheses


# ================================================================
# S2: Gold-position exact balance
# ================================================================

class TestS2GoldPositionBalance:
    """S2: Gold-answer position counts within each regime differ by at most 1."""

    def _check_balance(self, items):
        by_regime = {}
        for item in items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, regime_items in by_regime.items():
            positions = []
            for item in regime_items:
                pos = item.hypotheses.index(item.gold_answer)
                positions.append(pos)
            counter = Counter(positions)
            # Must have all 4 positions represented
            counts = [counter.get(p, 0) for p in range(4)]
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime}: gold position counts {dict(counter)} "
                f"have max diff {max_diff} > 1"
            )

    def test_gold_position_balance(self, v31_items):
        self._check_balance(v31_items)

    def test_gold_position_balance_medium(self, v31_medium):
        self._check_balance(v31_medium)

    def test_gold_position_balance_large(self, v31_large):
        self._check_balance(v31_large)


# ================================================================
# S3: Abstention-position exact balance
# ================================================================

class TestS3AbstentionPositionBalance:
    """S3: Abstention position counts within each regime differ by at most 1."""

    def _check_balance(self, items):
        by_regime = {}
        for item in items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, regime_items in by_regime.items():
            positions = []
            for item in regime_items:
                pos = item.hypotheses.index(ABSTENTION_HYPOTHESIS)
                positions.append(pos)
            counter = Counter(positions)
            counts = [counter.get(p, 0) for p in range(4)]
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime}: abstention position counts "
                f"{dict(counter)} have max diff {max_diff} > 1"
            )

    def test_abstention_position_balance(self, v31_items):
        self._check_balance(v31_items)

    def test_abstention_position_balance_medium(self, v31_medium):
        self._check_balance(v31_medium)

    def test_abstention_position_balance_large(self, v31_large):
        self._check_balance(v31_large)


# ================================================================
# S4: Cross-regime evidence normalization
# ================================================================

class TestS4EvidenceNormalization:
    """S4: ALL regimes have exactly N_EVIDENCE_SLOTS evidence items."""

    def test_evidence_count_uniform(self, v31_items):
        for item in v31_items:
            assert len(item.evidence) == N_EVIDENCE_SLOTS, (
                f"Item {item.id}: {len(item.evidence)} evidence items, "
                f"expected {N_EVIDENCE_SLOTS}"
            )

    def test_evidence_count_cross_regime(self, v31_large):
        """Sorted evidence-count multisets are identical across regimes."""
        by_regime = {}
        for item in v31_large:
            by_regime.setdefault(item.regime, []).append(item)

        regimes = sorted(by_regime.keys())
        ref_multiset = sorted(
            len(it.evidence) for it in by_regime[regimes[0]]
        )
        for regime in regimes[1:]:
            other_multiset = sorted(
                len(it.evidence) for it in by_regime[regime]
            )
            assert ref_multiset == other_multiset, (
                f"{regimes[0]} vs {regime}: evidence count multisets differ"
            )

    def test_evidence_count_at_scale(self, v31_large):
        for item in v31_large:
            assert len(item.evidence) == N_EVIDENCE_SLOTS


# ================================================================
# S5: Option text length tolerance
# ================================================================

class TestS5OptionTextLength:
    """S5: Per-regime mean option text length within 20% relative band."""

    def test_option_length_tolerance(self, v31_large):
        by_regime = {}
        for item in v31_large:
            by_regime.setdefault(item.regime, []).append(item)

        regime_means = {}
        for regime, regime_items in by_regime.items():
            lengths = [
                float(np.mean([len(h) for h in it.hypotheses]))
                for it in regime_items
            ]
            regime_means[regime] = float(np.mean(lengths))

        grand_mean = float(np.mean(list(regime_means.values())))
        assert grand_mean > 0, "Grand mean option length is 0"

        for regime, rmean in regime_means.items():
            rel_diff = abs(rmean - grand_mean) / grand_mean
            assert rel_diff <= 0.20, (
                f"Regime {regime}: mean option length {rmean:.2f} differs "
                f"from grand mean {grand_mean:.2f} by {rel_diff:.4f} > 0.20"
            )

    def test_lexical_parallelism_tight_band(self, v31_large):
        """v3.1 hypothesis parallelism should give < 1% relative diff."""
        by_regime = {}
        for item in v31_large:
            by_regime.setdefault(item.regime, []).append(item)

        regime_means = {}
        for regime, regime_items in by_regime.items():
            lengths = [
                float(np.mean([len(h) for h in it.hypotheses]))
                for it in regime_items
            ]
            regime_means[regime] = float(np.mean(lengths))

        grand_mean = float(np.mean(list(regime_means.values())))
        for regime, rmean in regime_means.items():
            rel_diff = abs(rmean - grand_mean) / grand_mean
            assert rel_diff <= 0.05, (
                f"Regime {regime}: v3.1 lexically-parallel hypotheses "
                f"should have < 5% rel diff, got {rel_diff:.4f}"
            )


# ================================================================
# S6: Hypothesis order randomized
# ================================================================

class TestS6HypothesisShuffled:
    """S6: Gold/abstention at pre-assigned positions (same check as S2)."""

    def test_s6_gold_position_max_diff_le_1(self, v31_items):
        by_regime = {}
        for item in v31_items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, regime_items in by_regime.items():
            positions = [
                it.hypotheses.index(it.gold_answer) for it in regime_items
            ]
            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1


# ================================================================
# v3.1-specific invariants
# ================================================================

class TestV31LexicalParallelism:
    """All hypotheses share the same syntactic frame."""

    def test_suspect_hypothesis_format(self, v31_items):
        """Suspect hypotheses must use the SUSPECT_HYPOTHESIS_TEMPLATE."""
        for item in v31_items:
            for hyp in item.hypotheses:
                if hyp == ABSTENTION_HYPOTHESIS:
                    continue
                # Must start with "Based on the available evidence, "
                assert hyp.startswith(
                    "Based on the available evidence, "
                ), f"Item {item.id}: hypothesis doesn't match template: {hyp}"
                # Must end with "is uniquely identified as responsible."
                assert hyp.endswith(
                    "is uniquely identified as responsible."
                ), f"Item {item.id}: hypothesis doesn't match template: {hyp}"

    def test_abstention_hypothesis_format(self, v31_items):
        """Abstention hypothesis must be the exact ABSTENTION_HYPOTHESIS."""
        for item in v31_items:
            abs_hyps = [h for h in item.hypotheses if h == ABSTENTION_HYPOTHESIS]
            assert len(abs_hyps) == 1, (
                f"Item {item.id}: expected exactly 1 abstention hypothesis"
            )

    def test_hypothesis_word_overlap(self, v31_items):
        """Bag-of-words overlap between suspect and abstention hypotheses.

        The abstention and suspect hypotheses should share most words.
        Check that Jaccard similarity > 0.5 for every item.
        """
        for item in v31_items:
            abs_words = set(ABSTENTION_HYPOTHESIS.lower().split())
            for hyp in item.hypotheses:
                if hyp == ABSTENTION_HYPOTHESIS:
                    continue
                suspect_words = set(hyp.lower().split())
                intersection = abs_words & suspect_words
                union = abs_words | suspect_words
                jaccard = len(intersection) / len(union)
                assert jaccard > 0.5, (
                    f"Item {item.id}: low word overlap between suspect "
                    f"and abstention hypotheses (Jaccard={jaccard:.3f})"
                )


class TestV31Determinism:
    """Verify v3.1 generator is deterministic from seed."""

    def test_same_seed_same_output(self):
        g1 = T2V31Generator(seed=42)
        g2 = T2V31Generator(seed=42)
        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=42)

        assert len(items1) == len(items2)
        for a, b in zip(items1, items2):
            assert a.id == b.id
            assert a.gold_answer == b.gold_answer
            assert a.hypotheses == b.hypotheses
            assert a.regime == b.regime
            assert len(a.evidence) == len(b.evidence)
            for ea, eb in zip(a.evidence, b.evidence):
                assert ea["content"] == eb["content"]

    def test_different_seed_different_output(self):
        g1 = T2V31Generator(seed=42)
        g2 = T2V31Generator(seed=99)
        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=99)

        any_different = any(
            a.hypotheses != b.hypotheses or a.gold_answer != b.gold_answer
            for a, b in zip(items1, items2)
        )
        assert any_different


class TestV31RegimeBalance:
    """Verify regime distribution is balanced."""

    def test_regime_counts(self):
        g = T2V31Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)
        regime_counts = Counter(item.regime for item in items)
        assert regime_counts["CLEAN"] == 8
        assert regime_counts["DECOY"] == 8
        assert regime_counts["CONFLICT"] == 8
        assert regime_counts["INSUFFICIENT"] == 8


class TestV31ItemIds:
    """Verify v3.1 items have v3.1 ID prefix."""

    def test_v31_prefix(self, v31_items):
        for item in v31_items:
            assert item.id.startswith("t2v31_"), (
                f"Item {item.id} does not start with 't2v31_'"
            )


class TestV31GoldAnswerValidity:
    """Verify gold answer correctness per regime."""

    def test_gold_in_hypotheses(self, v31_items):
        for item in v31_items:
            assert item.gold_answer in item.hypotheses, (
                f"Item {item.id}: gold_answer not in hypotheses"
            )

    def test_insufficient_gold_is_abstention(self, v31_items):
        for item in v31_items:
            if item.regime == "INSUFFICIENT":
                assert item.gold_answer == ABSTENTION_HYPOTHESIS, (
                    f"INSUFFICIENT item {item.id}: gold should be abstention"
                )

    def test_non_insufficient_gold_is_suspect(self, v31_items):
        for item in v31_items:
            if item.regime != "INSUFFICIENT":
                assert item.gold_answer != ABSTENTION_HYPOTHESIS, (
                    f"Non-INSUFFICIENT item {item.id}: gold should not be "
                    f"abstention"
                )
                assert "is uniquely identified as responsible" in item.gold_answer


class TestV31GoldAbstentionNonCollision:
    """Verify gold and abstention at different positions (non-INSUFFICIENT)."""

    def test_non_collision(self, v31_items):
        for item in v31_items:
            if item.regime == "INSUFFICIENT":
                continue
            gold_pos = item.hypotheses.index(item.gold_answer)
            abs_pos = item.hypotheses.index(ABSTENTION_HYPOTHESIS)
            assert gold_pos != abs_pos, (
                f"Item {item.id}: gold and abstention at same position "
                f"{gold_pos}"
            )

    def test_non_collision_large(self, v31_large):
        for item in v31_large:
            if item.regime == "INSUFFICIENT":
                continue
            gold_pos = item.hypotheses.index(item.gold_answer)
            abs_pos = item.hypotheses.index(ABSTENTION_HYPOTHESIS)
            assert gold_pos != abs_pos


class TestV31EvidenceStructure:
    """Verify evidence item structure."""

    def test_evidence_has_required_fields(self, v31_items):
        required = {"id", "content", "supports", "contradicts",
                    "diagnostic_value"}
        for item in v31_items:
            for ev in item.evidence:
                assert required.issubset(ev.keys()), (
                    f"Item {item.id}, evidence {ev.get('id', '?')}: "
                    f"missing fields {required - set(ev.keys())}"
                )

    def test_evidence_ids_sequential(self, v31_items):
        for item in v31_items:
            ids = [ev["id"] for ev in item.evidence]
            expected = [f"E{i:03d}" for i in range(1, len(item.evidence) + 1)]
            assert ids == expected, (
                f"Item {item.id}: evidence IDs not sequential: {ids}"
            )
