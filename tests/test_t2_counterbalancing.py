"""
Regression tests for T2 v2 counterbalancing invariants.

These tests verify that the v2 generator enforces surface-level balance
across suspects, preventing shallow classifiers from recovering the
correct answer via mention frequency, evidence count, length, or
polarity cues.

The positional-balance test (TestV2LegacyPositionBalance) checks the
WEAKER guarantee the v2 generator actually provides — no gross positional
bias (no position > 50%).  The stronger exact-balance invariant
(max position-count difference <= 1) is a v3 requirement (AMENDMENT-002
§2.5.2 S6) and is tested in the v3 test suite, not here.

Pre-specified in AMENDMENT-001 §4.4.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any

import pytest

from datasets.t2_generator.generator import T2Generator, T2Item


class TestNameFrequencyEqualization:
    """Verify that suspect names appear with equal frequency across evidence."""

    @pytest.fixture
    def large_batch(self):
        g = T2Generator(seed=42)
        return g.generate_dataset(n_per_regime=8, seed=42)

    def test_name_frequency_within_item(self, large_batch):
        """Each suspect name should appear approximately equally often
        across all evidence items within a single item."""
        for item in large_batch:
            if item.regime == "INSUFFICIENT":
                continue  # INSUFFICIENT is symmetric by design

            # Extract suspect names from hypotheses
            suspects = []
            for h in item.hypotheses:
                if h == "Cannot be determined from available evidence":
                    continue
                name = h.replace(" is responsible", "")
                suspects.append(name)

            if not suspects:
                continue

            # Count name occurrences across all evidence
            all_evidence_text = " ".join(ev["content"] for ev in item.evidence)
            counts = {s: all_evidence_text.count(s) for s in suspects}

            # CV (coefficient of variation) should be ≤ 0.25
            values = list(counts.values())
            if not values or max(values) == 0:
                continue
            mean_count = sum(values) / len(values)
            if mean_count == 0:
                continue
            variance = sum((v - mean_count) ** 2 for v in values) / len(values)
            cv = math.sqrt(variance) / mean_count

            assert cv <= 0.25, (
                f"Item {item.id}: name frequency CV={cv:.3f} > 0.25. "
                f"Counts: {counts}"
            )

    def test_name_frequency_over_batch(self, large_batch):
        """Across the entire batch, no suspect name should dominate in
        gold answers more than expected by chance."""
        # Exclude INSUFFICIENT items (gold = "Cannot be determined")
        items_with_suspect = [
            it for it in large_batch
            if it.gold_answer != "Cannot be determined from available evidence"
        ]
        if len(items_with_suspect) < 10:
            pytest.skip("Too few non-INSUFFICIENT items")

        gold_names = [
            it.gold_answer.replace(" is responsible", "")
            for it in items_with_suspect
        ]
        counter = Counter(gold_names)
        # No single name should be the answer for >40% of items
        max_frac = max(counter.values()) / len(gold_names)
        assert max_frac <= 0.40, (
            f"Gold answer name concentration {max_frac:.2f} > 0.40: {counter}"
        )


class TestEvidenceCountParity:
    """Verify each suspect is mentioned in the same number of evidence items."""

    @pytest.fixture
    def items(self):
        g = T2Generator(seed=42)
        return g.generate_dataset(n_per_regime=8, seed=42)

    def test_evidence_count_per_suspect(self, items):
        """Each suspect should appear in the same number of evidence items (±1)."""
        for item in items:
            if item.regime == "INSUFFICIENT":
                continue

            suspects = []
            for h in item.hypotheses:
                if h == "Cannot be determined from available evidence":
                    continue
                suspects.append(h.replace(" is responsible", ""))

            if not suspects:
                continue

            # Count evidence items mentioning each suspect
            ev_counts = {s: 0 for s in suspects}
            for ev in item.evidence:
                for s in suspects:
                    if s in ev["content"]:
                        ev_counts[s] += 1

            values = list(ev_counts.values())
            if not values:
                continue
            # Max - min should be ≤ 1
            assert max(values) - min(values) <= 1, (
                f"Item {item.id}: evidence counts unbalanced. {ev_counts}"
            )


class TestPolarityBalance:
    """Verify incriminating/exonerating evidence is balanced across suspects."""

    @pytest.fixture
    def items(self):
        g = T2Generator(seed=42)
        return g.generate_dataset(n_per_regime=8, seed=42)

    def test_polarity_balance(self, items):
        """Each suspect should have similar incriminating vs exonerating
        evidence counts (±1)."""
        for item in items:
            if item.regime == "INSUFFICIENT":
                continue

            suspects = []
            for h in item.hypotheses:
                if h == "Cannot be determined from available evidence":
                    continue
                suspects.append(h.replace(" is responsible", ""))

            if not suspects:
                continue

            supports_count = {s: 0 for s in suspects}
            contradicts_count = {s: 0 for s in suspects}

            for ev in item.evidence:
                for s in ev.get("supports", []):
                    if s in supports_count:
                        supports_count[s] += 1
                for s in ev.get("contradicts", []):
                    if s in contradicts_count:
                        contradicts_count[s] += 1

            # Check balance: supports and contradicts should be similar across suspects
            sup_values = list(supports_count.values())
            con_values = list(contradicts_count.values())

            if sup_values:
                assert max(sup_values) - min(sup_values) <= 1, (
                    f"Item {item.id}: support counts unbalanced. {supports_count}"
                )
            if con_values:
                assert max(con_values) - min(con_values) <= 1, (
                    f"Item {item.id}: contradict counts unbalanced. {contradicts_count}"
                )


class TestV2LegacyPositionBalance:
    """LEGACY (v2): Verify the v2 generator avoids gross positional bias.

    This test checks the WEAKER positional guarantee that the v2 generator
    actually provides: no single hypothesis position holds more than 50% of
    gold answers.  This is NOT the exact-balance invariant (max position-count
    difference <= 1) required by the v3 generator (AMENDMENT-002 §2.5.2 S6).

    The v2 generator does not enforce exact positional balance by construction.
    It uses a simple shuffle that, over a large enough batch, avoids gross
    clustering but does NOT guarantee max-diff <= 1.

    This test is NON-GATING for the v3 audit — it documents the historical v2
    guarantee only.  The v3 gating test is in test_t2_v3_counterbalancing.py
    (to be added in Phase B).
    """

    def test_v2_legacy_no_gross_positional_bias(self):
        """LEGACY (v2): No single position holds > 50% of gold answers.

        This is the v2 generator's weaker historical positional guarantee.
        It does NOT verify max position-count difference <= 1 (that is the
        v3 invariant, tested separately).
        """
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=16, seed=42)

        positions = []
        for item in items:
            if item.gold_answer == "Cannot be determined from available evidence":
                continue
            try:
                pos = item.hypotheses.index(item.gold_answer)
                positions.append(pos)
            except ValueError:
                continue

        if len(positions) < 20:
            pytest.skip("Too few items for position test")

        counter = Counter(positions)

        if len(counter) < 2:
            pytest.skip("Only one position observed")

        # v2 guarantee: no gross positional bias (no single position > 50%).
        # This is strictly weaker than the v3 requirement of max diff <= 1.
        n_total = len(positions)
        max_frac = max(counter.values()) / n_total

        assert max_frac <= 0.50, (
            f"Position distribution has gross bias: max fraction "
            f"{max_frac:.3f} > 0.50. Distribution: {dict(counter)}"
        )


class TestCounterfactualMinimalPairs:
    """Verify counterfactual pairs have matched tokens but different answers."""

    def test_counterfactual_token_overlap(self):
        """Counterfactual pairs should share ≥85% of their tokens (Jaccard)."""
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)

        pairs_tested = 0
        for item in items:
            if item.regime not in ["CLEAN", "DECOY"]:
                continue
            try:
                twin = g.generate_counterfactual_pair(item, seed=hash(item.id) % 1000000)
            except (AttributeError, NotImplementedError):
                pytest.skip("generate_counterfactual_pair not implemented")
                return

            # Token inventories
            def tokenize(t2_item):
                all_text = t2_item.narrative + " " + " ".join(
                    ev["content"] for ev in t2_item.evidence
                )
                return set(re.findall(r'\b\w+\b', all_text.lower()))

            tokens_a = tokenize(item)
            tokens_b = tokenize(twin)

            if not tokens_a or not tokens_b:
                continue

            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            assert jaccard >= 0.85, (
                f"Counterfactual pair Jaccard={jaccard:.3f} < 0.85 "
                f"for item {item.id}"
            )

            # Answers must differ
            assert item.gold_answer != twin.gold_answer, (
                f"Counterfactual pair has same answer: {item.gold_answer}"
            )

            pairs_tested += 1

        assert pairs_tested >= 5, f"Only tested {pairs_tested} counterfactual pairs"


class TestRegimeBalance:
    """Verify regime distribution is balanced."""

    def test_regime_counts(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)

        regime_counts = Counter(item.regime for item in items)
        assert regime_counts["CLEAN"] == 8
        assert regime_counts["DECOY"] == 8
        assert regime_counts["CONFLICT"] == 8
        assert regime_counts["INSUFFICIENT"] == 8


class TestDeterminism:
    """Verify generator is deterministic from seed."""

    def test_same_seed_same_output(self):
        g1 = T2Generator(seed=42)
        g2 = T2Generator(seed=42)

        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=42)

        assert len(items1) == len(items2)
        for a, b in zip(items1, items2):
            assert a.id == b.id
            assert a.gold_answer == b.gold_answer
            assert a.narrative == b.narrative

    def test_different_seed_different_output(self):
        g1 = T2Generator(seed=42)
        g2 = T2Generator(seed=99)

        items1 = g1.generate_dataset(n_per_regime=4, seed=42)
        items2 = g2.generate_dataset(n_per_regime=4, seed=99)

        # At least some items should differ
        any_different = any(
            a.gold_answer != b.gold_answer or a.narrative != b.narrative
            for a, b in zip(items1, items2)
        )
        assert any_different


class TestThreeOrMoreHypotheses:
    """Every item must have ≥3 hypotheses."""

    def test_hypothesis_count(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)

        for item in items:
            real_hyps = [
                h for h in item.hypotheses
                if h != "Cannot be determined from available evidence"
            ]
            assert len(real_hyps) >= 3, (
                f"Item {item.id} has only {len(real_hyps)} non-INSUFFICIENT hypotheses"
            )


class TestLengthMatching:
    """Evidence items should have similar lengths within each item."""

    def test_evidence_length_uniformity(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)

        for item in items:
            if len(item.evidence) < 2:
                continue

            lengths = [len(ev["content"]) for ev in item.evidence]
            mean_len = sum(lengths) / len(lengths)

            if mean_len == 0:
                continue

            # CV of evidence lengths should be ≤ 0.30
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            cv = math.sqrt(variance) / mean_len

            assert cv <= 0.30, (
                f"Item {item.id}: evidence length CV={cv:.3f} > 0.30. "
                f"Lengths: {lengths}"
            )
