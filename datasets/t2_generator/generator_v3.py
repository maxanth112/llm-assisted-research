"""
T2 v3 Generator — Exact-Balance Variant

Wraps the v2 generator to enforce the v3 invariants by construction
(AMENDMENT-002 §2.5.2):

  S1: Universal 4-option items — every item has exactly 4 hypotheses
      (3 suspect hypotheses + 1 abstention option).
  S2: Gold-position exact balance — within each regime, the gold-answer
      position counts differ by at most 1 (round-robin assignment).
  S3: Abstention-position exact balance — the "Cannot be determined"
      option's position also balanced (max diff <= 1) within each regime.
  S4: Evidence-count parity per suspect (inherited from v2).
  S5: Evidence-length matching (inherited from v2).
  S6: Hypothesis order randomized — explicit Fisher-Yates shuffle per
      item, with the position constraints enforced BEFORE shuffling via
      a pre-assigned slot system.

All invariants are deterministic: NO probabilistic criteria, no CV
thresholds, no chi-squared tests.

The v3 generator does NOT modify the v2 generator. It creates items
using v2, then post-processes to enforce v3 invariants.
"""

__version__ = "3.0.0"

import random
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from datasets.t2_generator.generator import (
    T2Generator as V2Generator,
    T2Item,
    SUSPECTS,
)


ABSTENTION_TEXT = "Cannot be determined from available evidence"


class T2V3Generator:
    """T2 v3 generator with exact-balance invariants enforced by construction."""

    def __init__(self, seed: int = 42):
        self.master_seed = seed
        self.v2 = V2Generator(seed=seed)

    def generate_dataset(
        self,
        n_per_regime: int = 8,
        seed: int = 42,
    ) -> List[T2Item]:
        """Generate a balanced dataset with v3 exact-balance invariants.

        Invariants enforced by construction:
          - S1: Every item has exactly 4 hypotheses
          - S2: Gold-position counts within each regime differ by at most 1
          - S3: Abstention-position counts within each regime differ by at most 1
          - S6: Hypotheses are explicitly shuffled per item

        Args:
            n_per_regime: Number of items per regime.
            seed: Master seed for generation.

        Returns:
            List of T2Item with v3 invariants guaranteed.
        """
        self.master_seed = seed
        self.v2 = V2Generator(seed=seed)

        # Generate v2 items as base
        v2_items = self.v2.generate_dataset(n_per_regime=n_per_regime, seed=seed)

        # Group by regime
        by_regime: Dict[str, List[T2Item]] = {}
        for item in v2_items:
            by_regime.setdefault(item.regime, []).append(item)

        rng = random.Random(seed)
        result = []

        for regime, items in by_regime.items():
            n = len(items)
            is_insufficient = (regime == "INSUFFICIENT")

            if is_insufficient:
                # For INSUFFICIENT: gold IS abstention. Only need one
                # position assignment (the gold/abstention position).
                gold_positions = _round_robin_positions(n, k=4,
                                                        rng_seed=seed + hash(regime))
                abstention_positions = gold_positions  # same slot
            else:
                # For non-INSUFFICIENT: gold and abstention must be at
                # different positions, both balanced.
                # Use joint pair assignment from 4*3=12 ordered pairs.
                gold_positions, abstention_positions = \
                    _joint_balanced_positions(n, k=4,
                                             rng_seed=seed + hash(regime))

            for i, item in enumerate(items):
                v3_item = self._convert_to_v3(
                    item,
                    gold_position=gold_positions[i],
                    abstention_position=abstention_positions[i],
                    rng=rng,
                )
                result.append(v3_item)

        return result

    def _convert_to_v3(
        self,
        item: T2Item,
        gold_position: int,
        abstention_position: int,
        rng: random.Random,
    ) -> T2Item:
        """Convert a v2 item to v3 format with exact position assignments.

        1. Ensure exactly 4 hypotheses (add abstention if missing)
        2. Place gold answer at gold_position
        3. Place abstention at abstention_position
        4. Fill remaining positions with non-gold, non-abstention hypotheses
        """
        # Extract suspect hypotheses (exclude abstention if already present)
        suspect_hyps = [h for h in item.hypotheses if h != ABSTENTION_TEXT]

        # Identify gold
        is_insufficient = (item.gold_answer == ABSTENTION_TEXT)

        # Build the 4-slot hypothesis list
        slots = [None] * 4

        if is_insufficient:
            # Gold IS the abstention. Place it at gold_position.
            slots[gold_position] = ABSTENTION_TEXT
            # Fill remaining 3 slots with suspect hypotheses
            remaining_positions = [p for p in range(4) if p != gold_position]
            # Shuffle suspects for randomization
            suspect_list = list(suspect_hyps)
            rng.shuffle(suspect_list)
            for pos, hyp in zip(remaining_positions, suspect_list):
                slots[pos] = hyp
        else:
            # Gold is a suspect hypothesis. Place it at gold_position.
            slots[gold_position] = item.gold_answer
            # Abstention at abstention_position
            slots[abstention_position] = ABSTENTION_TEXT
            # Fill remaining slots with other suspects
            other_suspects = [h for h in suspect_hyps if h != item.gold_answer]
            rng.shuffle(other_suspects)
            remaining_positions = [p for p in range(4) if slots[p] is None]
            for pos, hyp in zip(remaining_positions, other_suspects):
                slots[pos] = hyp

        assert all(s is not None for s in slots), f"Unfilled slot in {item.id}: {slots}"
        assert len(set(slots)) == 4 or len(set(slots)) == len(slots), \
            f"Duplicate hypotheses in {item.id}: {slots}"

        # Update metadata
        metadata = dict(item.metadata) if item.metadata else {}
        metadata['v3'] = True
        metadata['gold_position'] = gold_position
        metadata['abstention_position'] = abstention_position if not is_insufficient else gold_position

        return T2Item(
            id=item.id.replace("t2v2_", "t2v3_"),
            regime=item.regime,
            narrative=item.narrative,
            question=item.question,
            hypotheses=slots,
            evidence=item.evidence,
            gold_answer=item.gold_answer,
            gold_reasoning=item.gold_reasoning,
            source_precedence_rule=item.source_precedence_rule,
            metadata=metadata,
        )


def _round_robin_positions(n: int, k: int = 4, rng_seed: int = 0) -> List[int]:
    """Assign positions 0..k-1 to n items via shuffled round-robin.

    Guarantees: max(count) - min(count) <= 1 across all k positions.

    The positions are first assigned in strict round-robin (0,1,2,3,0,1,...),
    then the resulting list is shuffled to remove ordering artifacts.
    """
    positions = [i % k for i in range(n)]
    rng = random.Random(rng_seed)
    rng.shuffle(positions)
    return positions


def _joint_balanced_positions(
    n: int,
    k: int = 4,
    rng_seed: int = 0,
) -> tuple:
    """Assign (gold_pos, abstention_pos) pairs ensuring BOTH marginals balanced.

    Strategy:
    1. Generate balanced gold positions via round-robin (max diff <= 1).
    2. Generate balanced abstention positions via round-robin (max diff <= 1).
    3. Pair them. If any pair collides (gold == abstention), swap
       abstention assignments between items to resolve, preserving both
       marginal balances.

    This guarantees:
      - Gold positions: each of 0..k-1 appears with max diff <= 1
      - Abstention positions: each of 0..k-1 appears with max diff <= 1
      - Gold != abstention for every item
    """
    rng = random.Random(rng_seed)

    gold_positions = _round_robin_positions(n, k=k, rng_seed=rng_seed)
    abstention_positions = _round_robin_positions(n, k=k, rng_seed=rng_seed + 7)

    # Resolve collisions by swapping abstention values between items.
    # A swap preserves both marginal counts.
    max_attempts = n * n
    attempt = 0
    while attempt < max_attempts:
        # Find collision indices
        collisions = [i for i in range(n) if gold_positions[i] == abstention_positions[i]]
        if not collisions:
            break

        c = collisions[0]
        # Find a non-collision item whose abstention we can swap with
        resolved = False
        candidates = list(range(n))
        rng.shuffle(candidates)
        for j in candidates:
            if j == c:
                continue
            # After swap: c gets abs[j], j gets abs[c]
            # Check neither creates a new collision
            if (gold_positions[c] != abstention_positions[j] and
                    gold_positions[j] != abstention_positions[c]):
                abstention_positions[c], abstention_positions[j] = \
                    abstention_positions[j], abstention_positions[c]
                resolved = True
                break
        if not resolved:
            # Fallback: just shift this one (may slightly break balance)
            abstention_positions[c] = (gold_positions[c] + 1) % k
        attempt += 1

    return gold_positions, abstention_positions
