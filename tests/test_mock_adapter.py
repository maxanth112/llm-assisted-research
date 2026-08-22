"""Tests for MockAdapter across all modes and conditions."""

import json
import pytest

from harness.providers.mock_adapter import MockAdapter
from harness.parsers import parse_condition_output


class TestMockAdapterCompliant:
    """Compliant mode produces valid JSON for every condition."""

    @pytest.fixture
    def adapter(self):
        return MockAdapter(mode="compliant", seed=42)

    @pytest.mark.parametrize("condition_id", [
        "000", "100", "110", "101", "111", "filter_only", "prism_full", "free_cot"
    ])
    def test_compliant_output_parses(self, adapter, condition_id):
        """MockAdapter compliant mode should produce output that parses for every condition."""
        messages = [{"role": "user", "content": "test prompt"}]
        params = {"item_id": "test_item", "condition_id": condition_id}
        completion = adapter.complete(messages, params)

        result = parse_condition_output(completion.text, condition_id)
        assert result.success, (
            f"Condition {condition_id} failed to parse compliant output.\n"
            f"Text: {completion.text[:200]}\n"
            f"Log: {result.parse_attempt_log}"
        )

    def test_deterministic_output(self):
        """Same seed should produce identical output."""
        adapter1 = MockAdapter(mode="compliant", seed=42)
        adapter2 = MockAdapter(mode="compliant", seed=42)

        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "item1", "condition_id": "000"}

        out1 = adapter1.complete(messages, params)
        out2 = adapter2.complete(messages, params)
        assert out1.text == out2.text

    def test_completion_fields(self):
        adapter = MockAdapter(mode="compliant", seed=42)
        messages = [{"role": "user", "content": "test prompt"}]
        params = {"item_id": "item1", "condition_id": "000"}
        completion = adapter.complete(messages, params)

        assert completion.model == "mock-model"
        assert completion.model_version == "1.0"
        assert completion.prompt_tokens > 0
        assert completion.completion_tokens > 0
        assert completion.latency_ms >= 0
        assert not completion.cached


class TestMockAdapterMalformed:
    """Malformed mode produces outputs that may fail parsing."""

    def test_malformed_output_is_different(self):
        compliant = MockAdapter(mode="compliant", seed=42)
        malformed = MockAdapter(mode="malformed", seed=42)

        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "item1", "condition_id": "000"}

        out_c = compliant.complete(messages, params)
        out_m = malformed.complete(messages, params)
        # Malformed should differ from compliant
        assert out_c.text != out_m.text


class TestMockAdapterIncomplete:
    """Incomplete mode produces structurally valid JSON missing required fields."""

    def test_incomplete_fails_validation(self):
        adapter = MockAdapter(mode="incomplete", seed=42)
        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "item1", "condition_id": "000"}

        completion = adapter.complete(messages, params)
        result = parse_condition_output(completion.text, "000")
        # Should fail because required fields are missing
        assert not result.success


class TestMockAdapterContradictory:
    """Contradictory mode produces valid JSON with contradictory content."""

    def test_contradictory_parses_structurally(self):
        adapter = MockAdapter(mode="contradictory", seed=42)
        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "item1", "condition_id": "000"}

        completion = adapter.complete(messages, params)
        result = parse_condition_output(completion.text, "000")
        assert result.success  # Structurally valid, just contradictory content


class TestMockAdapterRuleBased:
    """Rule-based mode uses gold answers with configurable accuracy."""

    def test_perfect_accuracy(self):
        gold_answers = {"item1": "A", "item2": "B", "item3": "C"}
        adapter = MockAdapter(
            mode="rule_based",
            gold_answers=gold_answers,
            accuracy=1.0,
            seed=42
        )

        for item_id, gold in gold_answers.items():
            messages = [{"role": "user", "content": "test"}]
            params = {"item_id": item_id, "condition_id": "000"}
            completion = adapter.complete(messages, params)
            result = parse_condition_output(completion.text, "000")
            assert result.success
            assert result.data["answer"] == gold

    def test_zero_accuracy(self):
        gold_answers = {"item1": "A"}
        adapter = MockAdapter(
            mode="rule_based",
            gold_answers=gold_answers,
            accuracy=0.0,
            seed=42
        )

        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "item1", "condition_id": "000"}
        completion = adapter.complete(messages, params)
        result = parse_condition_output(completion.text, "000")
        assert result.success
        assert result.data["answer"] != "A"

    def test_rule_based_all_conditions(self):
        """Rule-based mode should produce parseable output for all conditions."""
        gold = {"item1": "A"}
        adapter = MockAdapter(mode="rule_based", gold_answers=gold, accuracy=1.0, seed=42)

        for cid in ["000", "100", "110", "101", "111", "prism_full", "free_cot"]:
            messages = [{"role": "user", "content": "test"}]
            params = {"item_id": "item1", "condition_id": cid}
            completion = adapter.complete(messages, params)
            result = parse_condition_output(completion.text, cid)
            assert result.success, f"Rule-based mode failed for {cid}: {result.parse_attempt_log}"

    def test_missing_gold_falls_back(self):
        """When item has no gold answer, falls back to compliant mode."""
        adapter = MockAdapter(mode="rule_based", gold_answers={}, accuracy=1.0, seed=42)
        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "unknown_item", "condition_id": "000"}
        completion = adapter.complete(messages, params)
        result = parse_condition_output(completion.text, "000")
        assert result.success


class TestMockAdapterProviderInterface:
    """Test ModelProvider interface compliance."""

    def test_provider_name(self):
        adapter = MockAdapter()
        assert adapter.provider_name() == "mock"

    def test_estimate_cost_zero(self):
        adapter = MockAdapter()
        assert adapter.estimate_cost(1000, 500) == 0.0

    def test_custom_response_map(self):
        custom = '```json\n{"answer": "Z", "confidence": 99, "reasoning": "custom"}\n```'
        adapter = MockAdapter(response_map={"special_item": custom})
        messages = [{"role": "user", "content": "test"}]
        params = {"item_id": "special_item", "condition_id": "000"}
        completion = adapter.complete(messages, params)
        assert "Z" in completion.text
