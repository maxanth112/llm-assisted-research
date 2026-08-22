"""Experimental condition definitions for the ACH factorial experiment."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Condition:
    """Defines an experimental condition."""

    id: str
    name: str
    E: bool  # Enumeration factor
    T: bool  # Tabular representation factor
    D: bool  # Disconfirmation factor
    num_calls: int
    description: str
    prompt_template_name: str


# Factorial conditions (2x2x2 design)
CONDITION_000 = Condition(
    id="000",
    name="Baseline",
    E=False,
    T=False,
    D=False,
    num_calls=1,
    description="Direct answer without enumeration, table, or disconfirmation",
    prompt_template_name="direct_answer"
)

CONDITION_100 = Condition(
    id="100",
    name="E-only",
    E=True,
    T=False,
    D=False,
    num_calls=1,
    description="Enumerate hypotheses only",
    prompt_template_name="enumerate"
)

CONDITION_110 = Condition(
    id="110",
    name="E+Table",
    E=True,
    T=True,
    D=False,
    num_calls=1,
    description="Enumerate hypotheses with evidence table (no diagnostic coding)",
    prompt_template_name="table_placebo"
)

CONDITION_101 = Condition(
    id="101",
    name="E+Prose-Disconfirm",
    E=True,
    T=False,
    D=True,
    num_calls=1,
    description="Enumerate hypotheses with prose disconfirmation analysis",
    prompt_template_name="prose_disconfirm"
)

CONDITION_111 = Condition(
    id="111",
    name="Full ACH",
    E=True,
    T=True,
    D=True,
    num_calls=1,
    description="Full ACH matrix with diagnostic coding",
    prompt_template_name="full_ach"
)

# Reference/control conditions
CONDITION_FILTER_ONLY = Condition(
    id="filter_only",
    name="A1-filter control",
    E=False,
    T=False,
    D=False,
    num_calls=2,
    description="Two-call pipeline: evidence filtering then answer",
    prompt_template_name="filter_only"
)

CONDITION_PRISM_FULL = Condition(
    id="prism_full",
    name="Full PRISM 4-agent",
    E=True,
    T=True,
    D=True,
    num_calls=4,
    description="Full 4-agent PRISM pipeline (A1->A2->A3->A4)",
    prompt_template_name="prism_full"
)

CONDITION_FREE_COT = Condition(
    id="free_cot",
    name="Free CoT placebo",
    E=False,
    T=False,
    D=False,
    num_calls=1,
    description="Free-form chain-of-thought without structure",
    prompt_template_name="free_cot"
)

# All conditions registry
ALL_CONDITIONS = {
    "000": CONDITION_000,
    "100": CONDITION_100,
    "110": CONDITION_110,
    "101": CONDITION_101,
    "111": CONDITION_111,
    "filter_only": CONDITION_FILTER_ONLY,
    "prism_full": CONDITION_PRISM_FULL,
    "free_cot": CONDITION_FREE_COT,
}

# Factorial conditions (2^3 design)
FACTORIAL_CONDITIONS = ["000", "100", "110", "101", "111"]

# Reference/control conditions
REFERENCE_CONDITIONS = ["filter_only", "prism_full", "free_cot"]

# Factor mapping for factorial conditions
CONDITION_FACTOR_MAP = {
    "000": {"E": 0, "T": 0, "D": 0},
    "100": {"E": 1, "T": 0, "D": 0},
    "110": {"E": 1, "T": 1, "D": 0},
    "101": {"E": 1, "T": 0, "D": 1},
    "111": {"E": 1, "T": 1, "D": 1},
}


def get_condition(condition_id: str) -> Optional[Condition]:
    """Get a condition by ID."""
    return ALL_CONDITIONS.get(condition_id)


def get_factorial_conditions() -> list[str]:
    """Get list of factorial condition IDs (000, 100, 110, 101, 111)."""
    return FACTORIAL_CONDITIONS.copy()


def get_reference_conditions() -> list[str]:
    """Get list of reference/control condition IDs (filter_only, prism_full, free_cot)."""
    return REFERENCE_CONDITIONS.copy()


def get_all_conditions() -> list[str]:
    """Get all condition IDs."""
    return list(ALL_CONDITIONS.keys())
