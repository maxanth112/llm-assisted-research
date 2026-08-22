"""Tests for Pydantic v2 output schemas."""

import pytest
from pydantic import ValidationError

from harness.schemas import (
    DirectAnswer,
    EnumerateOutput,
    TablePlaceboOutput,
    ProseDisconfirmOutput,
    FullACHOutput,
    FilterOutput,
    PRISMVerdictOutput,
    ParseResult,
    TrialRecord,
)


class TestDirectAnswer:
    def test_valid(self):
        da = DirectAnswer(answer="A", confidence=75, reasoning="test")
        assert da.answer == "A"
        assert da.confidence == 75

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            DirectAnswer(answer="A", confidence=101, reasoning="test")
        with pytest.raises(ValidationError):
            DirectAnswer(answer="A", confidence=-1, reasoning="test")

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            DirectAnswer(answer="A", confidence=50)


class TestEnumerateOutput:
    def test_valid(self):
        eo = EnumerateOutput(
            hypotheses=[
                EnumerateOutput.Hypothesis(
                    hypothesis="H1",
                    supporting_evidence="sup",
                    contradicting_evidence="contra"
                )
            ],
            answer="B",
            confidence=80,
            reasoning="test"
        )
        assert len(eo.hypotheses) == 1

    def test_model_dump(self):
        eo = EnumerateOutput(
            hypotheses=[
                EnumerateOutput.Hypothesis(
                    hypothesis="H1",
                    supporting_evidence="sup",
                    contradicting_evidence="contra"
                )
            ],
            answer="A",
            confidence=50,
            reasoning="r"
        )
        d = eo.model_dump()
        assert "hypotheses" in d
        assert d["hypotheses"][0]["hypothesis"] == "H1"


class TestFullACHOutput:
    def test_valid(self):
        out = FullACHOutput(
            hypotheses=["H1", "H2"],
            ach_matrix=[
                {"evidence": "E1", "consistency_codes": {"H1": "C", "H2": "I"}, "diagnostic_value": 0.8}
            ],
            inconsistency_counts={"H1": 0, "H2": 1},
            high_diagnostic_evidence=["E1"],
            answer="A",
            confidence=90,
            reasoning="test"
        )
        assert out.inconsistency_counts["H2"] == 1


class TestFilterOutput:
    def test_valid(self):
        fo = FilterOutput(
            relevant_evidence=["E1"],
            contextual_evidence=["E2"],
            irrelevant_evidence=["E3"],
            reasoning="test"
        )
        assert len(fo.relevant_evidence) == 1


class TestPRISMVerdictOutput:
    def test_valid(self):
        pv = PRISMVerdictOutput(
            verdict="A",
            confidence=85,
            reasoning="test",
            key_evidence=["E1"]
        )
        assert pv.verdict == "A"


class TestParseResult:
    def test_successful_parse(self):
        pr = ParseResult(
            success=True,
            data={"answer": "A"},
            raw_text="raw",
            condition_id="000"
        )
        assert pr.success
        assert pr.error is None

    def test_failed_parse(self):
        pr = ParseResult(
            success=False,
            raw_text="raw",
            error="parse failed",
            condition_id="000",
            parse_attempt_log=["Attempt 1: failed"]
        )
        assert not pr.success
        assert len(pr.parse_attempt_log) == 1


class TestTrialRecord:
    def test_valid(self):
        tr = TrialRecord(
            item_id="item_001",
            condition_id="000",
            run_index=0,
            seed=12345,
            model_id="mock",
            model_version="1.0",
            params={"temperature": 0.7},
            prompt_hash="abc123",
            raw_output="raw",
            token_counts={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=10.5,
            estimated_cost_usd=0.001,
            timestamp=1000000.0,
        )
        assert tr.item_id == "item_001"
        assert tr.errors == []
