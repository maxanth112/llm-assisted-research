"""Tests for experiment randomization and reproducibility."""

import pytest

from harness.randomization import ExperimentRandomizer


class TestSeedDerivation:
    """Verify SHA-256 seed derivation is deterministic and unique."""

    def test_same_input_same_seed(self):
        r1 = ExperimentRandomizer(master_seed=42)
        r2 = ExperimentRandomizer(master_seed=42)
        assert r1.derive_seed("item1", "000") == r2.derive_seed("item1", "000")

    def test_different_item_different_seed(self):
        r = ExperimentRandomizer(master_seed=42)
        s1 = r.derive_seed("item1", "000")
        s2 = r.derive_seed("item2", "000")
        assert s1 != s2

    def test_different_condition_different_seed(self):
        r = ExperimentRandomizer(master_seed=42)
        s1 = r.derive_seed("item1", "000")
        s2 = r.derive_seed("item1", "100")
        assert s1 != s2

    def test_different_master_seed(self):
        r1 = ExperimentRandomizer(master_seed=42)
        r2 = ExperimentRandomizer(master_seed=99)
        assert r1.derive_seed("item1", "000") != r2.derive_seed("item1", "000")

    def test_seed_is_positive_integer(self):
        r = ExperimentRandomizer(master_seed=42)
        seed = r.derive_seed("item1", "000", 0)
        assert isinstance(seed, int)
        assert seed >= 0


class TestShuffleChoices:
    """Verify choice shuffling is reproducible and correct."""

    def test_shuffle_is_permutation(self):
        r = ExperimentRandomizer(master_seed=42)
        choices = ["A", "B", "C", "D"]
        shuffled, perm = r.shuffle_choices(choices, "item1", "000")
        assert sorted(shuffled) == sorted(choices)

    def test_shuffle_reproducible(self):
        r1 = ExperimentRandomizer(master_seed=42)
        r2 = ExperimentRandomizer(master_seed=42)
        choices = ["A", "B", "C", "D"]
        s1, p1 = r1.shuffle_choices(choices, "item1", "000")
        s2, p2 = r2.shuffle_choices(choices, "item1", "000")
        assert s1 == s2
        assert p1 == p2

    def test_permutation_indices_valid(self):
        r = ExperimentRandomizer(master_seed=42)
        choices = ["W", "X", "Y", "Z"]
        shuffled, perm = r.shuffle_choices(choices, "item1", "000")
        # perm[shuffled_idx] = original_idx
        for shuffled_idx, original_idx in enumerate(perm):
            assert shuffled[shuffled_idx] == choices[original_idx]

    def test_different_items_different_shuffles(self):
        r = ExperimentRandomizer(master_seed=42)
        choices = ["A", "B", "C", "D"]
        s1, _ = r.shuffle_choices(choices, "item1", "000")
        s2, _ = r.shuffle_choices(choices, "item2", "000")
        # Very unlikely to be the same for different items
        # (but technically possible - we just check they're valid permutations)
        assert sorted(s1) == sorted(s2) == sorted(choices)


class TestUnshuffleAnswer:
    """Verify answer unshuffling maps back correctly."""

    def test_unshuffle_roundtrip(self):
        r = ExperimentRandomizer(master_seed=42)
        choices = ["Apple", "Banana", "Cherry", "Date"]

        shuffled, perm = r.shuffle_choices(choices, "item1", "000")

        # Suppose model picks "B" (second position in shuffled order)
        # That should map back to the original position of what's now at position 1
        labels = ["A", "B", "C", "D"]
        for shuffled_idx in range(len(choices)):
            shuffled_label = labels[shuffled_idx]
            original_label = r.unshuffle_answer(shuffled_label, choices, "item1", "000")
            # The original label should correspond to perm[shuffled_idx]
            expected_label = labels[perm[shuffled_idx]]
            assert original_label == expected_label

    def test_unshuffle_unknown_answer(self):
        """Non-standard answers should be returned as-is."""
        r = ExperimentRandomizer(master_seed=42)
        choices = ["A", "B", "C", "D"]
        result = r.unshuffle_answer("Cannot determine", choices, "item1", "000")
        assert result == "Cannot determine"

    def test_unshuffle_out_of_range(self):
        """Answer index beyond choices length should be returned as-is."""
        r = ExperimentRandomizer(master_seed=42)
        choices = ["A", "B"]  # only 2 choices
        result = r.unshuffle_answer("D", choices, "item1", "000")
        assert result == "D"


class TestTrialSchedule:
    """Verify trial schedule generation."""

    def test_schedule_length(self):
        r = ExperimentRandomizer(master_seed=42)
        schedule = r.generate_trial_schedule(
            item_ids=["i1", "i2"],
            condition_ids=["000", "100"],
            k_runs=3
        )
        assert len(schedule) == 2 * 2 * 3

    def test_schedule_contains_all_combinations(self):
        r = ExperimentRandomizer(master_seed=42)
        items = ["i1", "i2"]
        conditions = ["000", "100"]
        schedule = r.generate_trial_schedule(items, conditions, k_runs=2)

        combos = {(t["item_id"], t["condition_id"], t["run_index"]) for t in schedule}
        for item in items:
            for cond in conditions:
                for run in range(2):
                    assert (item, cond, run) in combos

    def test_schedule_seeds_unique(self):
        r = ExperimentRandomizer(master_seed=42)
        schedule = r.generate_trial_schedule(
            item_ids=["i1", "i2", "i3"],
            condition_ids=["000", "100", "111"],
            k_runs=3
        )
        seeds = [t["seed"] for t in schedule]
        assert len(seeds) == len(set(seeds)), "All seeds should be unique"

    def test_schedule_reproducible(self):
        r1 = ExperimentRandomizer(master_seed=42)
        r2 = ExperimentRandomizer(master_seed=42)
        s1 = r1.generate_trial_schedule(["i1"], ["000"], 2)
        s2 = r2.generate_trial_schedule(["i1"], ["000"], 2)
        assert s1 == s2
