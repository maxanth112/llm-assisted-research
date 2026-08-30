"""
Exact-balance tests for the T2 v3 generator (AMENDMENT-002 §2.5.2).

These tests verify that the v3 generator enforces ALL balance invariants
BY CONSTRUCTION.  Every assertion uses DETERMINISTIC criteria — no
probabilistic thresholds, no chi-squared tests, no coefficient-of-variation
tests.

Invariants tested:
  S1: Every item has exactly 4 hypotheses.
  S2: Within each regime, gold-answer position counts differ by at most 1.
  S3: The abstention option is present in every item AND its position
      counts within each regime differ by at most 1.
  S4: Evidence-count parity per suspect (±1).
  S5: Evidence-length matching (inherited from v2, CV <= 0.30).
  S6: Per-regime gold-position counts differ by at most 1 (same as S2).

These are GATING tests for the v3 audit.
"""

import math
from collections import Counter
from typing import List

import pytest

from datasets.t2_generator.generator_v3 import T2V3Generator, ABSTENTION_TEXT


@pytest.fixture
def v3_items():
    """Generate a v3 dataset with n_per_regime=8 (32 items total)."""
    g = T2V3Generator(seed=42)
    return g.generate_dataset(n_per_regime=8, seed=42)


@pytest.fixture
def v3_large():
    """Generate a larger v3 dataset for tighter balance checks."""
    g = T2V3Generator(seed=42)
    return g.generate_dataset(n_per_regime=16, seed=42)


class TestS1UniversalFourOption:
    """S1: Every item must have exactly 4 hypotheses."""

    def test_all_items_have_four_hypotheses(self, v3_items):
        for item in v3_items:
            assert len(item.hypotheses) == 4, (
                f"Item {item.id}: has {len(item.hypotheses)} hypotheses, expected 4"
            )

    def test_no_duplicate_hypotheses(self, v3_items):
        for item in v3_items:
            assert len(set(item.hypotheses)) == 4, (
                f"Item {item.id}: has duplicate hypotheses: {item.hypotheses}"
            )

    def test_abstention_present_in_every_item(self, v3_items):
        for item in v3_items:
            assert ABSTENTION_TEXT in item.hypotheses, (
                f"Item {item.id}: missing abstention option. "
                f"Hypotheses: {item.hypotheses}"
            )


class TestS2GoldPositionExactBalance:
    """S2: Gold-answer position counts within each regime differ by at most 1."""

    def test_gold_position_max_diff_le_1(self, v3_items):
        by_regime = {}
        for item in v3_items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, items in by_regime.items():
            positions = []
            for item in items:
                try:
                    pos = item.hypotheses.index(item.gold_answer)
                    positions.append(pos)
                except ValueError:
                    pytest.fail(
                        f"Item {item.id}: gold_answer '{item.gold_answer}' "
                        f"not found in hypotheses {item.hypotheses}"
                    )

            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime}: gold position counts {dict(counter)} "
                f"have max diff {max_diff} > 1"
            )

    def test_gold_position_max_diff_le_1_large(self, v3_large):
        """Same check on larger batch for additional confidence."""
        by_regime = {}
        for item in v3_large:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, items in by_regime.items():
            positions = []
            for item in items:
                pos = item.hypotheses.index(item.gold_answer)
                positions.append(pos)

            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime} (n={len(items)}): gold position counts "
                f"{dict(counter)} have max diff {max_diff} > 1"
            )


class TestS3AbstentionPositionExactBalance:
    """S3: Abstention position counts within each regime differ by at most 1."""

    def test_abstention_position_max_diff_le_1(self, v3_items):
        by_regime = {}
        for item in v3_items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, items in by_regime.items():
            positions = []
            for item in items:
                try:
                    pos = item.hypotheses.index(ABSTENTION_TEXT)
                    positions.append(pos)
                except ValueError:
                    pytest.fail(
                        f"Item {item.id}: abstention not found in hypotheses"
                    )

            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime}: abstention position counts {dict(counter)} "
                f"have max diff {max_diff} > 1"
            )

    def test_abstention_position_max_diff_le_1_large(self, v3_large):
        by_regime = {}
        for item in v3_large:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, items in by_regime.items():
            positions = []
            for item in items:
                pos = item.hypotheses.index(ABSTENTION_TEXT)
                positions.append(pos)

            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime} (n={len(items)}): abstention position counts "
                f"{dict(counter)} have max diff {max_diff} > 1"
            )


class TestS4EvidenceCountParity:
    """S4: Evidence-count parity per suspect (±1)."""

    def test_evidence_count_per_suspect(self, v3_items):
        for item in v3_items:
            suspects = [
                h for h in item.hypotheses if h != ABSTENTION_TEXT
            ]
            suspect_names = [
                h.replace(" is responsible", "") for h in suspects
            ]

            if not suspect_names:
                continue

            ev_counts = {s: 0 for s in suspect_names}
            for ev in item.evidence:
                for s in suspect_names:
                    if s in ev["content"]:
                        ev_counts[s] += 1

            values = list(ev_counts.values())
            if not values:
                continue

            assert max(values) - min(values) <= 1, (
                f"Item {item.id}: evidence counts unbalanced. {ev_counts}"
            )


class TestS5EvidenceLengthMatching:
    """S5: Evidence items should have similar lengths within each item."""

    def test_evidence_length_uniformity(self, v3_items):
        for item in v3_items:
            if len(item.evidence) < 2:
                continue

            lengths = [len(ev["content"]) for ev in item.evidence]
            mean_len = sum(lengths) / len(lengths)

            if mean_len == 0:
                continue

            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            cv = math.sqrt(variance) / mean_len

            assert cv <= 0.30, (
                f"Item {item.id}: evidence length CV={cv:.3f} > 0.30. "
                f"Lengths: {lengths}"
            )


class TestS6PositionBalance:
    """S6: Per-regime gold-position counts differ by at most 1.

    This is the same invariant as S2, stated separately because the
    amendment lists it as a distinct surface-form check.
    """

    def test_s6_identical_to_s2(self, v3_items):
        """S6 is the same check as S2 — verified here for clarity."""
        by_regime = {}
        for item in v3_items:
            by_regime.setdefault(item.regime, []).append(item)

        for regime, items in by_regime.items():
            positions = [
                item.hypotheses.index(item.gold_answer) for item in items
            ]
            counter = Counter(positions)
            counts = list(counter.values())
            max_diff = max(counts) - min(counts)
            assert max_diff <= 1, (
                f"Regime {regime}: gold position counts {dict(counter)} "
                f"have max diff {max_diff} > 1"
            )


class TestV3Determinism:
    """Verify v3 generator is deterministic from seed."""

    def test_same_seed_same_output(self):
        g1 = T2V3Generator(seed=42)
        g2 = T2V3Generator(seed=42)

        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=42)

        assert len(items1) == len(items2)
        for a, b in zip(items1, items2):
            assert a.id == b.id
            assert a.gold_answer == b.gold_answer
            assert a.hypotheses == b.hypotheses

    def test_different_seed_different_output(self):
        g1 = T2V3Generator(seed=42)
        g2 = T2V3Generator(seed=99)

        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=99)

        any_different = any(
            a.hypotheses != b.hypotheses or a.gold_answer != b.gold_answer
            for a, b in zip(items1, items2)
        )
        assert any_different


class TestV3RegimeBalance:
    """Verify regime distribution is balanced."""

    def test_regime_counts(self):
        g = T2V3Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)

        regime_counts = Counter(item.regime for item in items)
        assert regime_counts["CLEAN"] == 8
        assert regime_counts["DECOY"] == 8
        assert regime_counts["CONFLICT"] == 8
        assert regime_counts["INSUFFICIENT"] == 8


class TestV3ItemIds:
    """Verify v3 items have v3 ID prefix."""

    def test_v3_prefix(self, v3_items):
        for item in v3_items:
            assert item.id.startswith("t2v3_"), (
                f"Item {item.id} does not start with 't2v3_'"
            )


class TestV3GoldAnswerValidity:
    """Verify every gold answer appears in its item's hypotheses."""

    def test_gold_in_hypotheses(self, v3_items):
        for item in v3_items:
            assert item.gold_answer in item.hypotheses, (
                f"Item {item.id}: gold_answer '{item.gold_answer}' "
                f"not in hypotheses {item.hypotheses}"
            )

    def test_insufficient_gold_is_abstention(self, v3_items):
        for item in v3_items:
            if item.regime == "INSUFFICIENT":
                assert item.gold_answer == ABSTENTION_TEXT, (
                    f"INSUFFICIENT item {item.id}: gold_answer should be "
                    f"abstention but is '{item.gold_answer}'"
                )

    def test_non_insufficient_gold_is_suspect(self, v3_items):
        for item in v3_items:
            if item.regime != "INSUFFICIENT":
                assert item.gold_answer != ABSTENTION_TEXT, (
                    f"Non-INSUFFICIENT item {item.id}: gold_answer should be "
                    f"a suspect but is abstention"
                )
                assert "is responsible" in item.gold_answer, (
                    f"Item {item.id}: gold_answer '{item.gold_answer}' "
                    f"doesn't match expected format"
                )


class TestV3GoldAbstentionNonCollision:
    """Verify gold and abstention don't occupy the same position."""

    def test_gold_abstention_different_positions(self, v3_items):
        for item in v3_items:
            if item.regime == "INSUFFICIENT":
                continue  # For INSUFFICIENT, gold IS abstention
            gold_pos = item.hypotheses.index(item.gold_answer)
            abs_pos = item.hypotheses.index(ABSTENTION_TEXT)
            assert gold_pos != abs_pos, (
                f"Item {item.id}: gold and abstention at same position {gold_pos}"
            )
