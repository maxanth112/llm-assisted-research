"""Tests for T2 v2 deterministic diagnostic generator."""

import pytest

from datasets.t2_generator.generator import T2Generator, T2Item, export_jsonl


# V2 template families
TEMPLATE_KEYS = [
    "theft_alibi", "theft_timeline",
    "sabotage_alibi", "sabotage_timeline",
    "data_breach_alibi", "data_breach_timeline",
    "contamination_alibi", "contamination_timeline"
]


class TestT2GeneratorDeterminism:
    """Verify generator is deterministic from seed."""

    def test_same_seed_same_output(self):
        g1 = T2Generator(seed=42)
        g2 = T2Generator(seed=42)

        item1 = g1.generate_clean_item("theft_alibi", seed=1, item_id="test1")
        item2 = g2.generate_clean_item("theft_alibi", seed=1, item_id="test1")

        assert item1.narrative == item2.narrative
        assert item1.gold_answer == item2.gold_answer
        assert len(item1.evidence) == len(item2.evidence)

    def test_different_seed_different_output(self):
        g1 = T2Generator(seed=42)
        g2 = T2Generator(seed=99)

        item1 = g1.generate_clean_item("theft_alibi", seed=1, item_id="test1")
        item2 = g2.generate_clean_item("theft_alibi", seed=1, item_id="test1")

        # Different master seeds should produce different outputs
        # (different entity selections)
        assert item1.narrative != item2.narrative


class TestT2Regimes:
    """Test all four regimes produce valid items."""

    @pytest.fixture
    def generator(self):
        return T2Generator(seed=42)

    @pytest.mark.parametrize("template_key", TEMPLATE_KEYS)
    def test_clean_regime(self, generator, template_key):
        item = generator.generate_clean_item(template_key, seed=1, item_id=f"clean_{template_key}")
        assert item.regime == "CLEAN"
        assert len(item.hypotheses) >= 3
        assert len(item.evidence) >= 3
        assert item.gold_answer != "Cannot be determined from available evidence"
        assert item.source_precedence_rule is None

    @pytest.mark.parametrize("template_key", TEMPLATE_KEYS)
    def test_decoy_regime(self, generator, template_key):
        item = generator.generate_decoy_item(template_key, seed=1, item_id=f"decoy_{template_key}")
        assert item.regime == "DECOY"
        assert len(item.hypotheses) >= 3
        # Should have more evidence than CLEAN due to decoys
        assert len(item.evidence) >= 4
        assert item.gold_answer != "Cannot be determined from available evidence"

    @pytest.mark.parametrize("template_key", TEMPLATE_KEYS)
    def test_conflict_regime(self, generator, template_key):
        item = generator.generate_conflict_item(template_key, seed=1, item_id=f"conflict_{template_key}")
        assert item.regime == "CONFLICT"
        assert len(item.hypotheses) >= 3
        assert item.source_precedence_rule is not None
        assert item.gold_answer != "Cannot be determined from available evidence"

    @pytest.mark.parametrize("template_key", TEMPLATE_KEYS)
    def test_insufficient_regime(self, generator, template_key):
        item = generator.generate_insufficient_item(template_key, seed=1, item_id=f"insuff_{template_key}")
        assert item.regime == "INSUFFICIENT"
        assert len(item.hypotheses) >= 4  # 3 suspects + "Cannot determine"
        assert item.gold_answer == "Cannot be determined from available evidence"


class TestT2DatasetGeneration:
    """Test balanced dataset generation."""

    def test_dataset_balance(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=4, seed=42)

        # Should have 4 per regime = 16 total
        assert len(items) == 16

        regime_counts = {}
        for item in items:
            regime_counts[item.regime] = regime_counts.get(item.regime, 0) + 1

        assert regime_counts["CLEAN"] == 4
        assert regime_counts["DECOY"] == 4
        assert regime_counts["CONFLICT"] == 4
        assert regime_counts["INSUFFICIENT"] == 4

    def test_unique_ids(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=4, seed=42)
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids)), "All item IDs must be unique"

    def test_all_templates_used(self):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=8, seed=42)
        templates = {item.metadata.get("template") for item in items}
        assert templates == set(TEMPLATE_KEYS)


class TestT2CounterfactualPairs:
    """Test counterfactual pair generation."""

    def test_counterfactual_different_answer(self):
        g = T2Generator(seed=42)
        item = g.generate_clean_item("theft_alibi", seed=1, item_id="base")
        twin = g.generate_counterfactual_pair(item, seed=100)

        # Answers must differ
        assert item.gold_answer != twin.gold_answer
        assert "is responsible" in twin.gold_answer
        assert twin.regime == item.regime

    def test_counterfactual_preserves_narrative(self):
        g = T2Generator(seed=42)
        item = g.generate_clean_item("theft_alibi", seed=1, item_id="base")
        twin = g.generate_counterfactual_pair(item, seed=100)

        # Narrative should be identical
        assert item.narrative == twin.narrative

    def test_counterfactual_id(self):
        g = T2Generator(seed=42)
        item = g.generate_clean_item("theft_alibi", seed=1, item_id="base")
        twin = g.generate_counterfactual_pair(item, seed=100)

        assert "counterfactual" in twin.id

    def test_counterfactual_rejects_insufficient(self):
        g = T2Generator(seed=42)
        item = g.generate_insufficient_item("theft_alibi", seed=1, item_id="insuff")
        with pytest.raises(ValueError):
            g.generate_counterfactual_pair(item, seed=100)


class TestT2ItemSerialization:
    def test_to_dict_roundtrip(self):
        g = T2Generator(seed=42)
        item = g.generate_clean_item("theft_alibi", seed=1, item_id="test_serial")
        d = item.to_dict()
        assert d["id"] == "test_serial"
        assert d["regime"] == "CLEAN"
        assert isinstance(d["evidence"], list)

    def test_export_jsonl(self, tmp_path):
        g = T2Generator(seed=42)
        items = g.generate_dataset(n_per_regime=2, seed=42)
        path = str(tmp_path / "test.jsonl")
        export_jsonl(items, path)

        # Read back
        import json
        with open(path) as f:
            loaded = [json.loads(line) for line in f if line.strip()]
        assert len(loaded) == len(items)
