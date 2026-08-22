"""Pydantic v2 schemas for structured outputs from each experimental condition."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class DirectAnswer(BaseModel):
    """Baseline condition output: direct answer without enumeration."""
    answer: str = Field(..., description="The selected answer")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Brief reasoning for the answer")


class EnumerateOutput(BaseModel):
    """E-only condition: enumerate hypotheses before answering."""

    class Hypothesis(BaseModel):
        hypothesis: str
        supporting_evidence: str
        contradicting_evidence: str

    hypotheses: list[Hypothesis] = Field(..., description="List of enumerated hypotheses")
    answer: str = Field(..., description="The selected answer")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Final reasoning for the answer")


class TablePlaceboOutput(BaseModel):
    """E+Table condition: hypotheses with evidence table (no diagnostic coding)."""

    hypotheses: list[str] = Field(..., description="List of hypothesis strings")
    evidence_table: list[dict[str, Any]] = Field(
        ...,
        description="List of evidence items with summaries dict mapping hypothesis->summary"
    )
    answer: str = Field(..., description="The selected answer")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Final reasoning for the answer")


class ProseDisconfirmOutput(BaseModel):
    """E+Prose-Disconfirm condition: hypotheses with disconfirming analysis."""

    class HypothesisAnalysis(BaseModel):
        hypothesis: str
        disconfirming_analysis: str

    hypotheses: list[HypothesisAnalysis] = Field(
        ...,
        description="Hypotheses with disconfirming analysis"
    )
    answer: str = Field(..., description="The selected answer")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Final reasoning for the answer")


class FullACHOutput(BaseModel):
    """Full ACH condition: complete Analysis of Competing Hypotheses."""

    hypotheses: list[str] = Field(..., description="List of hypothesis strings")
    ach_matrix: list[dict[str, Any]] = Field(
        ...,
        description="ACH matrix rows with evidence, consistency_codes dict (hyp->C/I/N), diagnostic_value"
    )
    inconsistency_counts: dict[str, int] = Field(
        ...,
        description="Count of inconsistencies per hypothesis"
    )
    high_diagnostic_evidence: list[str] = Field(
        ...,
        description="Evidence items with highest diagnostic value"
    )
    answer: str = Field(..., description="The selected answer")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Final reasoning for the answer")


class FilterOutput(BaseModel):
    """Output from evidence filtering step."""

    relevant_evidence: list[str] = Field(..., description="Directly relevant evidence")
    contextual_evidence: list[str] = Field(..., description="Background/contextual evidence")
    irrelevant_evidence: list[str] = Field(..., description="Irrelevant evidence")
    reasoning: str = Field(..., description="Reasoning for categorization")


class PRISMVerdictOutput(BaseModel):
    """Final verdict from PRISM 4-agent pipeline."""

    verdict: str = Field(..., description="Final answer verdict")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(..., description="Final reasoning")
    key_evidence: list[str] = Field(..., description="Key evidence supporting verdict")


class ParseResult(BaseModel):
    """Result of parsing model output into structured format."""

    success: bool = Field(..., description="Whether parsing succeeded")
    data: Optional[Any] = Field(None, description="Parsed structured data")
    raw_text: str = Field(..., description="Raw model output")
    error: Optional[str] = Field(None, description="Error message if parsing failed")
    condition_id: str = Field(..., description="Condition ID for this parse")
    parse_attempt_log: list[str] = Field(
        default_factory=list,
        description="Log of all parse attempts and strategies tried"
    )


class TrialRecord(BaseModel):
    """Complete record of a single experimental trial."""

    item_id: str = Field(..., description="ID of the test item")
    condition_id: str = Field(..., description="Experimental condition ID")
    run_index: int = Field(..., description="Run number (0-indexed)")
    seed: int = Field(..., description="Random seed for this trial")
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version string")
    params: dict[str, Any] = Field(..., description="Model generation parameters")
    prompt_hash: str = Field(..., description="SHA-256 hash of the prompt")
    raw_output: str = Field(..., description="Raw model output")
    parsed_result: Optional[dict[str, Any]] = Field(
        None,
        description="Parsed structured output"
    )
    token_counts: dict[str, int] = Field(
        ...,
        description="Token counts (prompt_tokens, completion_tokens)"
    )
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    timestamp: float = Field(..., description="Unix timestamp of trial completion")
