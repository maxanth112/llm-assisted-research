"""Tests for scoring and evaluation metrics."""

import math
import pytest

from harness.scoring import (
    normalize_answer,
    exact_match,
    accuracy,
    abstention_correctness,
    flip_rate,
    self_consistency,
    matrix_faithfulness,
    calibration_ece,
    brier_score,
    wilson_ci,
    aggregate_by_condition,
)


class TestNormalizeAnswer:
    def test_uppercase(self):
        assert normalize_answer("a") == "A"

    def test_strip_whitespace(self):
        assert normalize_answer("  A  ") == "A"

    def test_extract_choice_letter(self):
        assert normalize_answer("B is correct") == "B"

    def test_non_choice_text(self):
        result = normalize_answer("42")
        assert result == "42"


class TestExactMatch:
    def test_match(self):
        assert exact_match("A", "A")

    def test_case_insensitive(self):
        assert exact_match("a", "A")

    def test_mismatch(self):
        assert not exact_match("A", "B")


class TestAccuracy:
    def test_perfect(self):
        result = accuracy(["A", "B", "C"], ["A", "B", "C"])
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == 3

    def test_zero(self):
        result = accuracy(["A", "B", "C"], ["D", "D", "D"])
        assert result["accuracy"] == 0.0

    def test_partial(self):
        result = accuracy(["A", "B", "C", "D"], ["A", "B", "D", "D"])
        assert result["accuracy"] == 0.75

    def test_empty(self):
        result = accuracy([], [])
        assert result["accuracy"] == 0.0
        assert result["n"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy(["A"], ["A", "B"])


class TestAbstentionCorrectness:
    def test_basic(self):
        preds = ["A", "B", "ABSTAIN"]
        golds = ["A", "C", "A"]
        abstain = [False, False, True]
        regimes = ["normal", "normal", "hard"]

        result = abstention_correctness(preds, golds, abstain, regimes)
        assert result["normal"]["n"] == 2
        assert result["normal"]["n_correct"] == 1
        assert result["hard"]["n_abstain"] == 1

    def test_overall(self):
        preds = ["A", "B"]
        golds = ["A", "B"]
        abstain = [False, False]
        regimes = ["r1", "r2"]
        result = abstention_correctness(preds, golds, abstain, regimes)
        assert result["overall"]["accuracy"] == 1.0


class TestFlipRate:
    def test_no_flips(self):
        result = flip_rate(["A", "B", "C"], ["A", "B", "C"])
        assert result["flip_rate"] == 0.0

    def test_all_flips(self):
        result = flip_rate(["A", "B", "C"], ["B", "C", "A"])
        assert result["flip_rate"] == 1.0

    def test_partial_flips(self):
        result = flip_rate(["A", "B"], ["A", "C"])
        assert result["flip_rate"] == 0.5


class TestSelfConsistency:
    def test_perfect_agreement(self):
        runs = [["A", "B", "C"], ["A", "B", "C"], ["A", "B", "C"]]
        result = self_consistency(runs)
        assert result["mean_agreement"] == 1.0

    def test_no_agreement(self):
        runs = [["A", "B"], ["B", "A"]]
        result = self_consistency(runs)
        assert result["mean_agreement"] == 0.5

    def test_empty(self):
        result = self_consistency([])
        assert result["mean_agreement"] == 0.0


class TestMatrixFaithfulness:
    def test_faithful(self):
        matrices = [{"inconsistency_counts": {"Answer is A": 0, "Answer is B": 3}}]
        answers = ["A"]
        result = matrix_faithfulness(matrices, answers)
        assert result["n_faithful"] == 1

    def test_unfaithful(self):
        matrices = [{"inconsistency_counts": {"Answer is A": 0, "Answer is B": 3}}]
        answers = ["B"]
        result = matrix_faithfulness(matrices, answers)
        assert result["n_faithful"] == 0


class TestCalibrationECE:
    def test_perfect_calibration(self):
        # All predictions at 100% confidence and all correct
        confs = [100.0] * 10
        correct = [True] * 10
        result = calibration_ece(confs, correct)
        assert result["ece"] < 0.01

    def test_ece_range(self):
        confs = [50.0, 80.0, 30.0, 90.0]
        correct = [True, False, True, True]
        result = calibration_ece(confs, correct)
        assert 0.0 <= result["ece"] <= 1.0


class TestBrierScore:
    def test_perfect(self):
        result = brier_score([100.0, 100.0], [True, True])
        assert result["brier_score"] == 0.0

    def test_worst(self):
        result = brier_score([100.0, 100.0], [False, False])
        assert result["brier_score"] == 1.0

    def test_empty(self):
        result = brier_score([], [])
        assert result["brier_score"] == 0.0


class TestWilsonCI:
    def test_basic(self):
        lower, upper = wilson_ci(50, 100)
        assert lower < 0.5 < upper

    def test_zero(self):
        lower, upper = wilson_ci(0, 100)
        assert lower >= 0.0
        assert upper > 0.0

    def test_perfect(self):
        lower, upper = wilson_ci(100, 100)
        assert lower > 0.9
        assert upper <= 1.0

    def test_empty(self):
        lower, upper = wilson_ci(0, 0)
        assert lower == 0.0
        assert upper == 0.0


class TestAggregateByCondition:
    def test_basic_aggregation(self):
        # aggregate_by_condition reads answer from parsed_result.answer (flat)
        trials = [
            {"condition_id": "000", "gold_answer": "A",
             "parsed_result": {"success": True, "answer": "A", "data": {"confidence": 80}}},
            {"condition_id": "000", "gold_answer": "B",
             "parsed_result": {"success": True, "answer": "B", "data": {"confidence": 70}}},
            {"condition_id": "100", "gold_answer": "A",
             "parsed_result": {"success": True, "answer": "C", "data": {"confidence": 60}}},
        ]
        result = aggregate_by_condition(trials)
        assert "000" in result
        assert "100" in result
        assert result["000"]["accuracy"] == 1.0
        assert result["100"]["accuracy"] == 0.0

    def test_parse_failure_rate(self):
        trials = [
            {"condition_id": "000", "gold_answer": "A",
             "parsed_result": {"success": False}},
            {"condition_id": "000", "gold_answer": "B",
             "parsed_result": {"success": True, "data": {"answer": "B"}}},
        ]
        result = aggregate_by_condition(trials)
        assert result["000"]["parse_fail_rate"] == 0.5
