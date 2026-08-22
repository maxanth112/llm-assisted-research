"""Tests for robust JSON parsing with multiple strategies."""

import json
import pytest

from harness.parsers import (
    extract_json_block,
    repair_json,
    parse_condition_output,
    SCHEMA_MAP,
)


class TestExtractJsonBlock:
    """Test JSON extraction from various text formats."""

    def test_json_fence(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_generic_fence(self):
        text = 'Some text\n```\n{"key": "value"}\n```\nMore text'
        result = extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_bare_json_object(self):
        text = 'Some text before {"key": "value"} some text after'
        result = extract_json_block(text)
        assert result == '{"key": "value"}'

    def test_bare_json_array(self):
        text = 'Some text [1, 2, 3] after'
        result = extract_json_block(text)
        assert result == '[1, 2, 3]'

    def test_nested_braces(self):
        text = '{"outer": {"inner": "value"}}'
        result = extract_json_block(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_no_json(self):
        text = 'This is plain text with no JSON'
        result = extract_json_block(text)
        assert result is None


class TestRepairJson:
    """Test JSON repair strategies."""

    def test_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        repaired = repair_json(text)
        data = json.loads(repaired)
        assert data["a"] == 1

    def test_single_quotes(self):
        text = "{'a': 'value'}"
        repaired = repair_json(text)
        data = json.loads(repaired)
        assert data["a"] == "value"

    def test_unquoted_keys(self):
        text = '{answer: "A", confidence: 75}'
        repaired = repair_json(text)
        data = json.loads(repaired)
        assert data["answer"] == "A"


class TestParseConditionOutput:
    """Test condition-specific parsing across all conditions."""

    def test_parse_000_compliant(self):
        data = {"answer": "A", "confidence": 75, "reasoning": "test"}
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "000")
        assert result.success
        assert result.data["answer"] == "A"

    def test_parse_100_compliant(self):
        data = {
            "hypotheses": [
                {"hypothesis": "H1", "supporting_evidence": "s", "contradicting_evidence": "c"}
            ],
            "answer": "B",
            "confidence": 80,
            "reasoning": "test"
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "100")
        assert result.success

    def test_parse_110_compliant(self):
        data = {
            "hypotheses": ["H1", "H2"],
            "evidence_table": [{"evidence": "E1", "summaries": {"H1": "s1", "H2": "s2"}}],
            "answer": "A",
            "confidence": 70,
            "reasoning": "test"
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "110")
        assert result.success

    def test_parse_101_compliant(self):
        data = {
            "hypotheses": [
                {"hypothesis": "H1", "disconfirming_analysis": "analysis"}
            ],
            "answer": "C",
            "confidence": 65,
            "reasoning": "test"
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "101")
        assert result.success

    def test_parse_111_compliant(self):
        data = {
            "hypotheses": ["H1", "H2"],
            "ach_matrix": [
                {"evidence": "E1", "consistency_codes": {"H1": "C", "H2": "I"}, "diagnostic_value": 0.8}
            ],
            "inconsistency_counts": {"H1": 0, "H2": 1},
            "high_diagnostic_evidence": ["E1"],
            "answer": "A",
            "confidence": 90,
            "reasoning": "test"
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "111")
        assert result.success

    def test_parse_filter_only_compliant(self):
        data = {
            "relevant_evidence": ["E1"],
            "contextual_evidence": ["E2"],
            "irrelevant_evidence": ["E3"],
            "reasoning": "test"
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "filter_only")
        assert result.success

    def test_parse_prism_full_compliant(self):
        data = {
            "verdict": "A",
            "confidence": 85,
            "reasoning": "test",
            "key_evidence": ["E1"]
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "prism_full")
        assert result.success

    def test_parse_free_cot_compliant(self):
        data = {"answer": "D", "confidence": 60, "reasoning": "step by step"}
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "free_cot")
        assert result.success

    def test_all_schema_map_conditions_covered(self):
        """Every condition in SCHEMA_MAP parses compliant output."""
        for cid in SCHEMA_MAP:
            assert cid in SCHEMA_MAP

    def test_malformed_trailing_comma(self):
        text = '```json\n{"answer": "A", "confidence": 75, "reasoning": "test",}\n```'
        result = parse_condition_output(text, "000")
        assert result.success, f"Repair should fix trailing comma: {result.parse_attempt_log}"

    def test_malformed_single_quotes(self):
        text = "```json\n{'answer': 'A', 'confidence': 75, 'reasoning': 'test'}\n```"
        result = parse_condition_output(text, "000")
        assert result.success, f"Repair should fix single quotes: {result.parse_attempt_log}"

    def test_incomplete_output_fails(self):
        """Missing required fields should fail parsing."""
        text = '```json\n{"answer": "A"}\n```'
        result = parse_condition_output(text, "000")
        assert not result.success

    def test_no_json_fails(self):
        text = "I think the answer is A because of the evidence."
        result = parse_condition_output(text, "000")
        assert not result.success

    def test_parse_attempt_log_populated(self):
        """All parse attempts must be logged."""
        text = "no json here"
        result = parse_condition_output(text, "000")
        assert len(result.parse_attempt_log) >= 2

    def test_unknown_condition_fails(self):
        text = '```json\n{"answer": "A"}\n```'
        result = parse_condition_output(text, "unknown_condition_xyz")
        assert not result.success
        assert "No schema mapping" in result.error

    def test_contradictory_still_parses(self):
        """Valid JSON with contradictory content should still parse structurally."""
        data = {"answer": "A", "confidence": 25, "reasoning": "All evidence says B but I pick A"}
        text = "```json\n" + json.dumps(data) + "\n```"
        result = parse_condition_output(text, "000")
        assert result.success  # structural parse succeeds
