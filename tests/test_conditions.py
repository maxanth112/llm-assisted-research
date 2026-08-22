"""Tests for experimental condition definitions."""

from harness.conditions import (
    ALL_CONDITIONS,
    FACTORIAL_CONDITIONS,
    REFERENCE_CONDITIONS,
    CONDITION_FACTOR_MAP,
    get_condition,
    get_factorial_conditions,
    get_reference_conditions,
    get_all_conditions,
)


class TestConditionDefinitions:
    """Verify all 8 conditions are defined correctly."""

    def test_factorial_condition_count(self):
        assert len(FACTORIAL_CONDITIONS) == 5

    def test_reference_condition_count(self):
        assert len(REFERENCE_CONDITIONS) == 3

    def test_all_conditions_count(self):
        assert len(ALL_CONDITIONS) == 8

    def test_factorial_ids(self):
        assert set(FACTORIAL_CONDITIONS) == {"000", "100", "110", "101", "111"}

    def test_reference_ids(self):
        assert set(REFERENCE_CONDITIONS) == {"filter_only", "prism_full", "free_cot"}

    def test_factor_map_matches_conditions(self):
        for cond_id, factors in CONDITION_FACTOR_MAP.items():
            cond = get_condition(cond_id)
            assert cond is not None, f"Condition {cond_id} not found"
            assert int(cond.E) == factors["E"]
            assert int(cond.T) == factors["T"]
            assert int(cond.D) == factors["D"]

    def test_baseline_has_no_factors(self):
        baseline = get_condition("000")
        assert not baseline.E
        assert not baseline.T
        assert not baseline.D

    def test_full_ach_has_all_factors(self):
        full_ach = get_condition("111")
        assert full_ach.E
        assert full_ach.T
        assert full_ach.D

    def test_prism_has_4_calls(self):
        prism = get_condition("prism_full")
        assert prism.num_calls == 4

    def test_filter_only_has_2_calls(self):
        filt = get_condition("filter_only")
        assert filt.num_calls == 2

    def test_single_call_conditions(self):
        for cid in ["000", "100", "110", "101", "111", "free_cot"]:
            cond = get_condition(cid)
            assert cond.num_calls == 1, f"{cid} should have 1 call"

    def test_get_condition_unknown_returns_none(self):
        assert get_condition("nonexistent") is None

    def test_get_factorial_returns_copy(self):
        a = get_factorial_conditions()
        b = get_factorial_conditions()
        assert a == b
        a.append("extra")
        assert len(get_factorial_conditions()) == 5

    def test_get_all_conditions(self):
        all_conds = get_all_conditions()
        assert len(all_conds) == 8
        for cid in FACTORIAL_CONDITIONS + REFERENCE_CONDITIONS:
            assert cid in all_conds


class TestConditionAttributes:
    """Verify each condition has required attributes."""

    def test_all_have_prompt_template_name(self):
        for cid, cond in ALL_CONDITIONS.items():
            assert cond.prompt_template_name, f"{cid} missing prompt_template_name"

    def test_all_have_description(self):
        for cid, cond in ALL_CONDITIONS.items():
            assert cond.description, f"{cid} missing description"

    def test_all_have_name(self):
        for cid, cond in ALL_CONDITIONS.items():
            assert cond.name, f"{cid} missing name"
