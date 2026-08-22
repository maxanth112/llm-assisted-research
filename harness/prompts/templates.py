"""Jinja2-based prompt template system for LLM evaluation harness.

This module provides structured prompt templates for various reasoning strategies
including direct answering, hypothesis enumeration, ACH (Analysis of Competing Hypotheses),
and the PRISM multi-agent system.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Any
from jinja2 import Template


@dataclass
class PromptTemplate:
    """A versioned prompt template with content hash for reproducibility."""
    name: str
    version: str
    template_text: str
    content_hash: str

    def __post_init__(self):
        """Compute SHA-256 hash of template text."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.template_text.encode('utf-8')
            ).hexdigest()


# Template definitions

TEMPLATE_000_DIRECT = PromptTemplate(
    name="TEMPLATE_000_DIRECT",
    version="1.0.0",
    template_text="""Answer the question based on the narrative.

{{ narrative }}

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Provide answer in JSON format with the following structure:
{
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation of your reasoning"
}""",
    content_hash=""
)
TEMPLATE_000_DIRECT.__post_init__()


TEMPLATE_100_ENUMERATE = PromptTemplate(
    name="TEMPLATE_100_ENUMERATE",
    version="1.0.0",
    template_text="""Answer by first enumerating all candidate hypotheses.

{{ narrative }}

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Instructions:
1. List each hypothesis
2. For each hypothesis, identify supporting and contradicting evidence
3. Select the most likely hypothesis

Provide answer in JSON format:
{
  "hypotheses": [
    {
      "hypothesis": "hypothesis description",
      "supporting_evidence": ["evidence 1", "evidence 2", ...],
      "contradicting_evidence": ["evidence 1", "evidence 2", ...]
    },
    ...
  ],
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation of your selection"
}""",
    content_hash=""
)
TEMPLATE_100_ENUMERATE.__post_init__()


TEMPLATE_110_TABLE_PLACEBO = PromptTemplate(
    name="TEMPLATE_110_TABLE_PLACEBO",
    version="1.0.0",
    template_text="""Organize evidence in a structured table.

{{ narrative }}

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Instructions:
1. List all hypotheses
2. Create an evidence×hypothesis table
3. For each cell, write a NEUTRAL summary only (DO NOT use C/I/N codes or consistency labels)
4. Select the most likely answer

Provide answer in JSON format:
{
  "hypotheses": ["hypothesis 1", "hypothesis 2", ...],
  "evidence_table": [
    {
      "evidence": "evidence description",
      "summaries": {
        "hypothesis 1": "neutral summary",
        "hypothesis 2": "neutral summary",
        ...
      }
    },
    ...
  ],
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation of your selection"
}""",
    content_hash=""
)
TEMPLATE_110_TABLE_PLACEBO.__post_init__()


TEMPLATE_101_PROSE_DISCONFIRM = PromptTemplate(
    name="TEMPLATE_101_PROSE_DISCONFIRM",
    version="1.0.0",
    template_text="""Analyze what evidence CONTRADICTS each hypothesis.

{{ narrative }}

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Instructions:
1. List all hypotheses
2. For each hypothesis, write a prose analysis about evidence that contradicts it
3. Select the hypothesis that is LEAST contradicted
4. Do NOT use a table format

Provide answer in JSON format:
{
  "hypotheses": [
    {
      "hypothesis": "hypothesis description",
      "disconfirming_analysis": "prose analysis of contradicting evidence"
    },
    ...
  ],
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation of why this hypothesis is least contradicted"
}""",
    content_hash=""
)
TEMPLATE_101_PROSE_DISCONFIRM.__post_init__()


TEMPLATE_111_FULL_ACH = PromptTemplate(
    name="TEMPLATE_111_FULL_ACH",
    version="1.0.0",
    template_text="""You are an intelligence analyst trained in ACH (Analysis of Competing Hypotheses).

{{ narrative }}

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Instructions:
1. Enumerate all hypotheses
2. Build an evidence×hypothesis matrix
3. Score each cell as C (consistent), I (inconsistent), or N (neutral)
4. Count the number of inconsistencies for each hypothesis
5. Identify high-diagnostic evidence (evidence that discriminates between hypotheses)
6. Select the hypothesis with the FEWEST inconsistencies

Provide answer in JSON format:
{
  "hypotheses": ["hypothesis 1", "hypothesis 2", ...],
  "ach_matrix": [
    {
      "evidence": "evidence description",
      "consistency_codes": {
        "hypothesis 1": "C/I/N",
        "hypothesis 2": "C/I/N",
        ...
      },
      "diagnostic_value": "HIGH/MEDIUM/LOW"
    },
    ...
  ],
  "inconsistency_counts": {
    "hypothesis 1": <count>,
    "hypothesis 2": <count>,
    ...
  },
  "high_diagnostic_evidence": ["evidence 1", "evidence 2", ...],
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation based on ACH analysis"
}""",
    content_hash=""
)
TEMPLATE_111_FULL_ACH.__post_init__()


TEMPLATE_FILTER_CALL1 = PromptTemplate(
    name="TEMPLATE_FILTER_CALL1",
    version="1.0.0",
    template_text="""You are a forensic analyst with 20 years of experience. Review the case and classify evidence as RELEVANT, CONTEXTUAL, or IRRELEVANT.

{{ narrative }}

Question: {{ question }}

Provide your classification in JSON format:
{
  "relevant_evidence": ["evidence directly related to answering the question", ...],
  "contextual_evidence": ["background information that provides context", ...],
  "irrelevant_evidence": ["information not useful for this question", ...],
  "reasoning": "explanation of your classification approach"
}""",
    content_hash=""
)
TEMPLATE_FILTER_CALL1.__post_init__()


TEMPLATE_FILTER_CALL2 = PromptTemplate(
    name="TEMPLATE_FILTER_CALL2",
    version="1.0.0",
    template_text="""Answer based on the filtered evidence.

Question: {{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Relevant Evidence:
{% for evidence in relevant_evidence %}
- {{ evidence }}
{% endfor %}

Contextual Evidence:
{% for evidence in context_evidence %}
- {{ evidence }}
{% endfor %}

Provide answer in JSON format:
{
  "answer": "your answer",
  "confidence": <0-100>,
  "reasoning": "explanation based on the filtered evidence"
}""",
    content_hash=""
)
TEMPLATE_FILTER_CALL2.__post_init__()


# PRISM templates - faithful to PRISM ACL 2026 Appendix C

TEMPLATE_PRISM_A1 = PromptTemplate(
    name="TEMPLATE_PRISM_A1",
    version="1.0.0",
    template_text="""You are a forensic analyst with 20 years of experience specializing in evidence evaluation.

Your task is to analyze evidence for relevance, reliability, and diagnostic value. Categorize each piece of evidence as HIGH_VALUE, MEDIUM_VALUE, LOW_VALUE, or IRRELEVANT.

{{ narrative }}

{{ question }}

Provide your analysis in JSON format:
{
  "evidence_analysis": [
    {
      "evidence": "evidence description",
      "relevance": "how relevant this evidence is to the question",
      "reliability": "assessment of evidence reliability",
      "diagnostic_value": "how much this evidence discriminates between hypotheses",
      "category": "HIGH_VALUE/MEDIUM_VALUE/LOW_VALUE/IRRELEVANT",
      "notes": "additional observations"
    },
    ...
  ],
  "summary": "overall assessment of the evidence landscape"
}""",
    content_hash=""
)
TEMPLATE_PRISM_A1.__post_init__()


TEMPLATE_PRISM_A2 = PromptTemplate(
    name="TEMPLATE_PRISM_A2",
    version="1.0.0",
    template_text="""You are a criminal investigator using M.O.M.A (Motive, Opportunity, Means, Alibi) framework.

{{ narrative }}

{{ question }}

Suspects to analyze:
{% for hypothesis in hypotheses %}
- {{ hypothesis }}
{% endfor %}

For each suspect, rate each MOMA factor as STRONG, MODERATE, WEAK, or ABSENT.

Provide your analysis in JSON format:
{
  "moma_analysis": [
    {
      "suspect": "suspect name",
      "motive": {
        "rating": "STRONG/MODERATE/WEAK/ABSENT",
        "evidence": "supporting evidence"
      },
      "opportunity": {
        "rating": "STRONG/MODERATE/WEAK/ABSENT",
        "evidence": "supporting evidence"
      },
      "means": {
        "rating": "STRONG/MODERATE/WEAK/ABSENT",
        "evidence": "supporting evidence"
      },
      "alibi": {
        "rating": "STRONG/MODERATE/WEAK/ABSENT",
        "evidence": "supporting evidence"
      },
      "overall": "overall assessment"
    },
    ...
  ],
  "answer": "most likely suspect",
  "confidence": <0-100>,
  "reasoning": "explanation based on MOMA analysis"
}""",
    content_hash=""
)
TEMPLATE_PRISM_A2.__post_init__()


TEMPLATE_PRISM_A3 = PromptTemplate(
    name="TEMPLATE_PRISM_A3",
    version="1.0.0",
    template_text="""You are an intelligence analyst trained in ACH (Analysis of Competing Hypotheses).

{{ narrative }}

{{ question }}

Hypotheses to evaluate:
{% for hypothesis in hypotheses %}
- {{ hypothesis }}
{% endfor %}

Instructions:
1. Build a consistency matrix with C (consistent), I (inconsistent), N (neutral) codes
2. Count inconsistencies for each hypothesis
3. Identify diagnostic evidence that discriminates between hypotheses
4. Analyze sensitivity to key assumptions

Provide your analysis in JSON format:
{
  "hypotheses": ["hypothesis 1", "hypothesis 2", ...],
  "evidence_items": ["evidence 1", "evidence 2", ...],
  "ach_matrix": [
    {
      "evidence": "evidence description",
      "consistency": {
        "hypothesis 1": "C/I/N",
        "hypothesis 2": "C/I/N",
        ...
      }
    },
    ...
  ],
  "inconsistency_counts": {
    "hypothesis 1": <count>,
    "hypothesis 2": <count>,
    ...
  },
  "diagnostic_evidence": [
    {
      "evidence": "evidence description",
      "diagnosticity": "HIGH/MEDIUM/LOW",
      "reasoning": "why this is diagnostic"
    },
    ...
  ],
  "sensitivity_analysis": "analysis of how robust conclusions are to key assumptions",
  "answer": "most likely hypothesis",
  "confidence": <0-100>,
  "reasoning": "explanation based on ACH analysis",
  "caveats": ["caveat 1", "caveat 2", ...]
}""",
    content_hash=""
)
TEMPLATE_PRISM_A3.__post_init__()


TEMPLATE_PRISM_A4 = PromptTemplate(
    name="TEMPLATE_PRISM_A4",
    version="1.0.0",
    template_text="""You are a judge presiding over a murder case.

{{ narrative }}

{{ question }}

Evidence Summary:
{{ evidence_summary }}

ACH Analysis:
{{ ach_matrix }}

Based on the evidence and analysis presented, deliver your verdict.

Provide your verdict in JSON format:
{
  "verdict": "your verdict",
  "confidence": <0-100>,
  "reasoning": "detailed explanation of your verdict",
  "key_evidence": ["most important evidence 1", "most important evidence 2", ...],
  "reasonable_doubt_analysis": "analysis of whether there is reasonable doubt",
  "alternative_hypotheses_considered": [
    {
      "hypothesis": "alternative hypothesis",
      "why_rejected": "explanation of why this was rejected"
    },
    ...
  ]
}""",
    content_hash=""
)
TEMPLATE_PRISM_A4.__post_init__()


TEMPLATE_FREE_COT = PromptTemplate(
    name="TEMPLATE_FREE_COT",
    version="1.0.0",
    template_text="""Think step by step. Write approximately {{ target_tokens }} tokens of reasoning.

{{ narrative }}

{{ question }}

Choices:
{% for choice in choices %}
{{ loop.index }}. {{ choice }}
{% endfor %}

Provide your answer in JSON format:
{
  "reasoning": "your step-by-step reasoning (approximately {{ target_tokens }} tokens)",
  "answer": "your answer",
  "confidence": <0-100>
}""",
    content_hash=""
)
TEMPLATE_FREE_COT.__post_init__()


# Template registry
ALL_TEMPLATES = {
    "TEMPLATE_000_DIRECT": TEMPLATE_000_DIRECT,
    "TEMPLATE_100_ENUMERATE": TEMPLATE_100_ENUMERATE,
    "TEMPLATE_110_TABLE_PLACEBO": TEMPLATE_110_TABLE_PLACEBO,
    "TEMPLATE_101_PROSE_DISCONFIRM": TEMPLATE_101_PROSE_DISCONFIRM,
    "TEMPLATE_111_FULL_ACH": TEMPLATE_111_FULL_ACH,
    "TEMPLATE_FILTER_CALL1": TEMPLATE_FILTER_CALL1,
    "TEMPLATE_FILTER_CALL2": TEMPLATE_FILTER_CALL2,
    "TEMPLATE_PRISM_A1": TEMPLATE_PRISM_A1,
    "TEMPLATE_PRISM_A2": TEMPLATE_PRISM_A2,
    "TEMPLATE_PRISM_A3": TEMPLATE_PRISM_A3,
    "TEMPLATE_PRISM_A4": TEMPLATE_PRISM_A4,
    "TEMPLATE_FREE_COT": TEMPLATE_FREE_COT,
}


def render_prompt(template: PromptTemplate, **kwargs: Any) -> str:
    """Render a prompt template with the given variables.

    Args:
        template: The PromptTemplate to render
        **kwargs: Variables to pass to the Jinja2 template

    Returns:
        Rendered prompt string
    """
    jinja_template = Template(template.template_text)
    return jinja_template.render(**kwargs)


def get_all_templates() -> Dict[str, PromptTemplate]:
    """Get all available prompt templates.

    Returns:
        Dictionary mapping template names to PromptTemplate objects
    """
    return ALL_TEMPLATES.copy()
