"""Experiment randomization and reproducibility utilities."""

import hashlib
import random
from typing import Any


class ExperimentRandomizer:
    """
    Handles all randomization for the experiment with reproducible seeding.

    Uses SHA-256 based seed derivation for perfect reproducibility.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize randomizer with master seed.

        Args:
            master_seed: Master seed for all derived randomization
        """
        self.master_seed = master_seed

    def derive_seed(self, *components: Any) -> int:
        """
        Derive a deterministic seed from multiple components using SHA-256.

        Args:
            *components: Components to hash (converted to strings)

        Returns:
            Integer seed derived from components
        """
        # Create deterministic string from all components
        seed_string = ":".join(str(c) for c in [self.master_seed] + list(components))

        # Hash to get deterministic seed
        hash_digest = hashlib.sha256(seed_string.encode("utf-8")).digest()

        # Convert first 8 bytes to int
        seed = int.from_bytes(hash_digest[:8], byteorder="big")

        return seed

    def shuffle_choices(
        self,
        choices: list[str],
        item_id: str,
        condition_id: str
    ) -> tuple[list[str], list[int]]:
        """
        Shuffle answer choices deterministically based on item and condition.

        Args:
            choices: Original list of choices
            item_id: Item identifier
            condition_id: Condition identifier

        Returns:
            Tuple of (shuffled_choices, permutation_indices)
            permutation_indices maps shuffled position -> original position
        """
        # Derive seed for this specific shuffle
        seed = self.derive_seed(item_id, condition_id, "shuffle")

        # Create indexed list
        indexed_choices = list(enumerate(choices))

        # Shuffle with derived seed
        rng = random.Random(seed)
        rng.shuffle(indexed_choices)

        # Extract shuffled choices and permutation
        shuffled_choices = [choice for _, choice in indexed_choices]
        permutation_indices = [idx for idx, _ in indexed_choices]

        return shuffled_choices, permutation_indices

    def unshuffle_answer(
        self,
        answer: str,
        choices: list[str],
        item_id: str,
        condition_id: str
    ) -> str:
        """
        Map an answer from shuffled choices back to original choice labels.

        Args:
            answer: Answer in shuffled space (e.g., "B" when choices were shuffled)
            choices: Original unshuffled choices
            item_id: Item identifier
            condition_id: Condition identifier

        Returns:
            Answer mapped back to original choice labels
        """
        # Get the shuffle permutation
        shuffled_choices, permutation_indices = self.shuffle_choices(
            choices, item_id, condition_id
        )

        # Standard choice labels
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

        # Find which shuffled position the answer corresponds to
        try:
            shuffled_idx = labels.index(answer)
        except (ValueError, IndexError):
            # Answer not in standard format or out of range
            return answer

        if shuffled_idx >= len(permutation_indices):
            # Answer index exceeds number of choices
            return answer

        # Map back to original position
        original_idx = permutation_indices[shuffled_idx]

        # Return original label
        return labels[original_idx]

    def generate_trial_schedule(
        self,
        item_ids: list[str],
        condition_ids: list[str],
        k_runs: int
    ) -> list[dict[str, Any]]:
        """
        Generate complete trial schedule with seeds for all combinations.

        Args:
            item_ids: List of item identifiers
            condition_ids: List of condition identifiers
            k_runs: Number of runs per item-condition pair

        Returns:
            List of trial dicts with keys: item_id, condition_id, run_index, seed
        """
        schedule = []

        for item_id in item_ids:
            for condition_id in condition_ids:
                for run_index in range(k_runs):
                    # Derive unique seed for this trial
                    seed = self.derive_seed(item_id, condition_id, run_index)

                    schedule.append({
                        "item_id": item_id,
                        "condition_id": condition_id,
                        "run_index": run_index,
                        "seed": seed
                    })

        return schedule
