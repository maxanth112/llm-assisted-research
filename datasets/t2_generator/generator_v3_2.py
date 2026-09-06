"""
T2 v3.2 Generator --- Permutation-Based Counterfactual Pairs (v3.2.2)

MECHANISM: Conjunction-based relational reasoning with exact token-multiset
preservation between counterfactual pair members.

KEY DESIGN (v3.2.2 fixes over v3.2.1):
  - Explicit INVESTIGATION RULE stated in every item preamble (defect 3)
  - Unambiguous evidence semantics: each criterion established by specific
    keyword ("Level-2" vs "Level-1", "flagged" vs "cleared") (defect 3)
  - Cross-regime name-frequency balance: every suspect mentioned exactly
    the same number of times in every regime, including DECOY (defect 6)
  - Programmatic name-length control: suspects drawn from length-matched
    strata so character lengths cannot correlate with role/label (defect 7)

Each item has 3 suspects, each mentioned in exactly 3 evidence dimensions
plus one neutral slot:
  D1: Access evidence (all suspects equal, non-diagnostic)
  D2: Criterion A (clearance level: Level-2 = positive, Level-1 = negative)
  D3: Criterion B (anomaly flag: flagged = positive, cleared = negative)
  D4: Neutral context (no suspect names)

Decision rule (stated explicitly in item text):
  A suspect is VIABLE iff they meet BOTH criteria (Level-2 AND flagged).
  - 1 viable  -> ANSWERABLE
  - >=2 viable -> INSUFFICIENT
  - 0 viable  -> INVALID (construction error)

ANSWERABLE:  crit_a=[1,1,0], crit_b=[1,0,1] -> only S0 has both -> 1 viable
INSUFFICIENT: crit_a=[1,1,0], crit_b=[1,1,0] -> S0 and S1 have both -> 2 viable

Permutation: single transposition in criterion B (swap S1<->S2).
Token multiset preserved because cap assignment identical + cor uses same
2x positive + 1x negative features just bound to different suspects.
"""

__version__ = "3.2.4"

import re
import hashlib
import random
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import asdict

import tiktoken

from datasets.t2_generator.generator import (
    T2Item,
    SUSPECTS as ALL_SUSPECTS,
    WITNESSES,
    LOCATIONS,
    COMPANIES,
)

from datasets.t2_generator.generator_v3_1 import (
    SUSPECT_HYPOTHESIS_TEMPLATE,
    ABSTENTION_HYPOTHESIS,
    N_EVIDENCE_SLOTS,
    TEMPLATE_FAMILIES,
    _round_robin_positions,
    _joint_balanced_positions,
)


# Tokenizer for token-level length matching (Item 2c).
# tiktoken cl100k_base is the OpenAI tokenizer used by GPT-4/3.5-turbo.
# Loaded once at module level; no network call needed after first cache.
TOKENIZER_NAME = "cl100k_base"
TOKENIZER_VERSION = "tiktoken 0.14.0"
_CL100K = tiktoken.get_encoding(TOKENIZER_NAME)


def _length_match_options(options: List[str]) -> List[str]:
    """Pad all options to equal character length AND equal token count.

    Uses tiktoken cl100k_base for token counting.  Trailing padding uses
    only '.' and ' ' characters (non-word chars) so \b\w+\b tokenization
    is unaffected.

    Algorithm:
      1. Find target_chars = max(len(o) for o in options) + 1
         (the +1 gives room for the naturally-longest option to also get
          a padding token, ensuring all can reach the same token count).
      2. Find target_tokens = max token count achievable by all options
         at target_chars.
      3. For each option, find a suffix of '.' and ' ' characters of the
         right length that produces exactly target_tokens tokens.
    """
    max_char = max(len(o) for o in options)
    target_chars = max_char + 1  # +1 gives room for the longest to get padding

    # Determine feasible target token count: pad each to target_chars, find max
    candidate_tokens = []
    for o in options:
        pad = target_chars - len(o)
        # Max tokens from padding = pad (each char a separate token)
        # Min tokens = 1 (all dots merge)
        base_tok = len(_CL100K.encode(o))
        # With 1 space + dots: adds ~2 tokens reliably
        candidate_tokens.append(base_tok + pad)  # upper bound

    # The target is the max base_tokens + 1 (for the trailing space padding)
    base_tokens = [len(_CL100K.encode(o)) for o in options]
    target_tokens = max(base_tokens) + 1

    result = []
    for o in options:
        padded = _token_aware_pad(o, target_chars, target_tokens)
        if padded is None:
            raise ValueError(
                f"Cannot pad option to {target_chars} chars / {target_tokens} tokens: "
                f"{repr(o[:60])}... (base: {len(o)} chars, "
                f"{len(_CL100K.encode(o))} tokens)"
            )
        result.append(padded)
    return result


def _token_aware_pad(text: str, target_chars: int, target_tokens: int) -> Optional[str]:
    """Pad text to exact target_chars and target_tokens using '.' and ' ' only.

    Uses brute-force search over short suffixes (max ~8 chars).  Returns None
    if no valid padding exists.
    """
    pad_len = target_chars - len(text)
    if pad_len < 0:
        return None
    if pad_len == 0:
        return text if len(_CL100K.encode(text)) == target_tokens else None

    base_tok = len(_CL100K.encode(text))
    need_extra = target_tokens - base_tok
    if need_extra < 0:
        return None

    # For short pad_len (≤10), brute-force 2^pad_len combinations of '.' and ' '
    if pad_len <= 12:
        from itertools import product
        for combo in product(". ", repeat=pad_len):
            suffix = "".join(combo)
            candidate = text + suffix
            if len(_CL100K.encode(candidate)) == target_tokens:
                return candidate
        return None

    # Longer padding: heuristic — space-separated dot groups
    suffix = " " + "." * (pad_len - 1)
    candidate = text + suffix
    if len(_CL100K.encode(candidate)) == target_tokens:
        return candidate
    return None


def _build_shared_option_array(
    suspects: List[str],
    abstention_pos: int,
    clean_gold_pos: int,
) -> List[str]:
    """Build a SHARED, length-matched option array with controlled positions.

    The abstention option is placed at abstention_pos, the guilty suspect's
    hypothesis at clean_gold_pos, and the remaining suspects fill the rest
    deterministically (sorted by name for reproducibility).

    Both pair members use this EXACT array.  Only the gold_position metadata
    differs (CLEAN gold = clean_gold_pos, INSUFFICIENT gold = abstention_pos).

    Returns 4 options, all padded to equal char length AND equal token count.
    """
    guilty = suspects[0]  # S0 is always the guilty suspect in CLEAN
    guilty_hyp = SUSPECT_HYPOTHESIS_TEMPLATE.format(name=guilty)
    other_hyps = sorted(
        [SUSPECT_HYPOTHESIS_TEMPLATE.format(name=s) for s in suspects[1:]]
    )

    slots = [None] * 4
    slots[abstention_pos] = ABSTENTION_HYPOTHESIS
    slots[clean_gold_pos] = guilty_hyp

    remaining_positions = [i for i in range(4) if slots[i] is None]
    for pos, hyp in zip(remaining_positions, other_hyps):
        slots[pos] = hyp

    assert all(s is not None for s in slots)
    return _length_match_options(slots)


# ================================================================
# INVESTIGATION RULE PREAMBLE (stated in every item)
# ================================================================

FAMILY_EVIDENCE_SPECS = {
    "theft_alibi": {
        "criterion_a_name": "Level-2 security clearance for the restricted zone",
        "criterion_a_pos": "{suspect} holds a Level-2 security clearance for the restricted zone",
        "criterion_a_neg": "{suspect} holds a Level-1 security clearance for the restricted zone",
        "criterion_b_name": "flagged status in the anomaly detection log for the incident period",
        "criterion_b_pos": "{suspect} was flagged in the anomaly detection log for the incident period",
        "criterion_b_neg": "{suspect} was cleared in the anomaly detection log for the incident period",
    },
    "theft_timeline": {
        "criterion_a_name": "Level-2 access authorization for the secured vault",
        "criterion_a_pos": "{suspect} holds a Level-2 access authorization for the secured vault",
        "criterion_a_neg": "{suspect} holds a Level-1 access authorization for the secured vault",
        "criterion_b_name": "flagged status in the inventory discrepancy report for the audit window",
        "criterion_b_pos": "{suspect} was flagged in the inventory discrepancy report for the audit window",
        "criterion_b_neg": "{suspect} was cleared in the inventory discrepancy report for the audit window",
    },
    "sabotage_alibi": {
        "criterion_a_name": "Level-2 technical certification for the critical system",
        "criterion_a_pos": "{suspect} holds a Level-2 technical certification for the critical system",
        "criterion_a_neg": "{suspect} holds a Level-1 technical certification for the critical system",
        "criterion_b_name": "flagged status in the system integrity audit for the maintenance window",
        "criterion_b_pos": "{suspect} was flagged in the system integrity audit for the maintenance window",
        "criterion_b_neg": "{suspect} was cleared in the system integrity audit for the maintenance window",
    },
    "sabotage_timeline": {
        "criterion_a_name": "Level-2 override authorization for the control interface",
        "criterion_a_pos": "{suspect} holds a Level-2 override authorization for the control interface",
        "criterion_a_neg": "{suspect} holds a Level-1 override authorization for the control interface",
        "criterion_b_name": "flagged status in the operational deviation log for the shutdown period",
        "criterion_b_pos": "{suspect} was flagged in the operational deviation log for the shutdown period",
        "criterion_b_neg": "{suspect} was cleared in the operational deviation log for the shutdown period",
    },
    "data_breach_alibi": {
        "criterion_a_name": "Level-2 network privilege for the classified database",
        "criterion_a_pos": "{suspect} holds a Level-2 network privilege for the classified database",
        "criterion_a_neg": "{suspect} holds a Level-1 network privilege for the classified database",
        "criterion_b_name": "flagged status in the data exfiltration scan for the breach window",
        "criterion_b_pos": "{suspect} was flagged in the data exfiltration scan for the breach window",
        "criterion_b_neg": "{suspect} was cleared in the data exfiltration scan for the breach window",
    },
    "data_breach_timeline": {
        "criterion_a_name": "Level-2 system credential for the encrypted archive",
        "criterion_a_pos": "{suspect} holds a Level-2 system credential for the encrypted archive",
        "criterion_a_neg": "{suspect} holds a Level-1 system credential for the encrypted archive",
        "criterion_b_name": "flagged status in the access anomaly report for the transfer window",
        "criterion_b_pos": "{suspect} was flagged in the access anomaly report for the transfer window",
        "criterion_b_neg": "{suspect} was cleared in the access anomaly report for the transfer window",
    },
    "contamination_alibi": {
        "criterion_a_name": "Level-2 handling permit for the controlled substance",
        "criterion_a_pos": "{suspect} holds a Level-2 handling permit for the controlled substance",
        "criterion_a_neg": "{suspect} holds a Level-1 handling permit for the controlled substance",
        "criterion_b_name": "flagged status in the contamination trace analysis for the production run",
        "criterion_b_pos": "{suspect} was flagged in the contamination trace analysis for the production run",
        "criterion_b_neg": "{suspect} was cleared in the contamination trace analysis for the production run",
    },
    "contamination_timeline": {
        "criterion_a_name": "Level-2 process authorization for the sterile facility",
        "criterion_a_pos": "{suspect} holds a Level-2 process authorization for the sterile facility",
        "criterion_a_neg": "{suspect} holds a Level-1 process authorization for the sterile facility",
        "criterion_b_name": "flagged status in the quality deviation log for the batch period",
        "criterion_b_pos": "{suspect} was flagged in the quality deviation log for the batch period",
        "criterion_b_neg": "{suspect} was cleared in the quality deviation log for the batch period",
    },
}


def make_investigation_rule(criterion_a_name: str, criterion_b_name: str) -> str:
    """Generate the explicit investigation rule preamble."""
    return (
        "INVESTIGATION RULE: A suspect is considered responsible if and only if "
        "BOTH of the following conditions are met: "
        f"(1) The suspect holds a {criterion_a_name}. "
        f"(2) The suspect has a {criterion_b_name}. "
        "If exactly one suspect meets both conditions, that suspect is responsible. "
        "If multiple suspects meet both conditions, or if no suspect meets both "
        "conditions, then responsibility cannot be uniquely determined."
    )


ACCESS_TEMPLATE = (
    "Facility entry records confirm {suspect} was present in the "
    "building during the incident period at {time}"
)

NEUTRAL_POOL = [
    ("Internal review of monitoring systems found all recording "
     "devices were functioning normally during the incident period"),
    ("Routine compliance audit completed prior to the incident "
     "identified no procedural deviations in the affected area"),
    ("Facility scheduling records confirm all listed employees "
     "were on active duty assignments during the relevant dates"),
    ("Environmental monitoring data from the incident period has been "
     "verified and shows no anomalous conditions in the building"),
]


# ================================================================
# PROGRAMMATIC NAME-LENGTH CONTROL (defect 7)
# ================================================================

def compute_name_length_strata(
    suspects: List[str],
    min_per_stratum: int = 3,
) -> Dict[int, List[str]]:
    """Group suspect names by character length.
    Returns dict mapping char_length -> sorted list of names.
    Only includes strata with >= min_per_stratum names.
    """
    strata: Dict[int, List[str]] = {}
    for name in suspects:
        length = len(name)
        strata.setdefault(length, []).append(name)
    return {k: sorted(v) for k, v in sorted(strata.items())
            if len(v) >= min_per_stratum}


NAME_LENGTH_STRATA = compute_name_length_strata(ALL_SUSPECTS)


def draw_length_matched_suspects(
    rng: random.Random,
    n: int = 3,
    strata: Optional[Dict[int, List[str]]] = None,
) -> List[str]:
    """Draw n suspects from the same length stratum."""
    if strata is None:
        strata = NAME_LENGTH_STRATA
    valid_lengths = [k for k, v in strata.items() if len(v) >= n]
    if not valid_lengths:
        raise ValueError(
            f"No length stratum has >= {n} names. "
            f"Strata: {dict((k, len(v)) for k, v in strata.items())}"
        )
    chosen_length = rng.choice(valid_lengths)
    return rng.sample(strata[chosen_length], n)


# ================================================================
# SYMBOLIC RULE ENGINE
# ================================================================

def symbolic_label(
    criterion_a_assignment: List[int],
    criterion_b_assignment: List[int],
    n_suspects: int = 3,
) -> Tuple[Optional[int], str, List[int]]:
    """Determine label from relational graph (conjunction rule).
    Returns (identified_idx, label, viable_indices).
    """
    viable = [i for i in range(n_suspects)
              if criterion_a_assignment[i] == 1
              and criterion_b_assignment[i] == 1]
    if len(viable) == 1:
        return viable[0], "ANSWERABLE", viable
    elif len(viable) >= 2:
        return None, "INSUFFICIENT", viable
    else:
        return None, "INVALID", viable


def exhaustive_viability_oracle(
    criterion_a_assignment: List[int],
    criterion_b_assignment: List[int],
    n_suspects: int = 3,
) -> Dict:
    """Assignment-based oracle. NOT the text-derived oracle (see text_oracle.py)."""
    truth_table = {}
    viable_set = []
    for i in range(n_suspects):
        has_a = criterion_a_assignment[i] == 1
        has_b = criterion_b_assignment[i] == 1
        has_both = has_a and has_b
        truth_table[f"S{i}"] = {
            "meets_criterion_a": has_a,
            "meets_criterion_b": has_b,
            "satisfies_conjunction": has_both,
        }
        if has_both:
            viable_set.append(i)
    n_viable = len(viable_set)
    if n_viable == 1:
        oracle_label = "ANSWERABLE"
    elif n_viable >= 2:
        oracle_label = "INSUFFICIENT"
    else:
        oracle_label = "INVALID"
    return {
        "truth_table": truth_table,
        "viable_indices": viable_set,
        "n_viable": n_viable,
        "oracle_label": oracle_label,
    }


# ================================================================
# PROMPT SERIALIZATION
# ================================================================

def serialize_prompt(
    investigation_rule: str,
    narrative: str,
    question: str,
    hypotheses: List[str],
    evidence: List[Dict],
) -> str:
    """Serialize an item into the exact prompt string a model receives."""
    lines = []
    lines.append(investigation_rule)
    lines.append("")
    lines.append(narrative)
    lines.append("")
    lines.append("Evidence:")
    for ev in evidence:
        content = ev["content"] if isinstance(ev, dict) else ev
        lines.append(f"- {content}")
    lines.append("")
    lines.append(question)
    lines.append("")
    for i, hyp in enumerate(hypotheses):
        lines.append(f"({chr(65+i)}) {hyp}")
    return "\n".join(lines)


# ================================================================
# TOKENIZATION AND MULTISET UTILITIES
# ================================================================

def tokenize_text(text: str) -> Counter:
    """Tokenize text into word-level unigrams. Uses \\b\\w+\\b on lowercased text."""
    return Counter(re.findall(r'\b\w+\b', text.lower()))


def multiset_hash(counter: Counter) -> str:
    """SHA-256 hash of a sorted token multiset."""
    items = sorted(counter.items())
    canonical = "|".join(f"{tok}:{cnt}" for tok, cnt in items)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _solve_pair_positions(
    n_pairs: int,
    k: int = 4,
    rng_seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Solve the joint constraint-satisfaction problem for pair positions.

    Returns (abstention_positions, clean_gold_positions) each of length
    n_pairs, satisfying:
      (a) clean_gold_positions has exactly n_pairs/k items per position
      (b) abstention_positions has exactly n_pairs/k items per position
      (c) For each pair i: clean_gold_positions[i] != abstention_positions[i]

    Constraint (d) — INSUFFICIENT gold == abstention — is enforced by
    the caller, not here.
    """
    assert n_pairs % k == 0, f"n_pairs={n_pairs} must be divisible by k={k}"
    per_pos = n_pairs // k

    rng = random.Random(rng_seed)

    # Generate balanced marginals
    abst = []
    gold = []
    for p in range(k):
        abst.extend([p] * per_pos)
        gold.extend([p] * per_pos)
    rng.shuffle(abst)
    rng.shuffle(gold)

    # Resolve collisions (where gold[i] == abst[i]) by swapping
    max_attempts = n_pairs * n_pairs * 10
    for _ in range(max_attempts):
        collisions = [i for i in range(n_pairs) if gold[i] == abst[i]]
        if not collisions:
            break
        c = collisions[0]
        candidates = list(range(n_pairs))
        rng.shuffle(candidates)
        for j in candidates:
            if j == c:
                continue
            # Swap gold[c] <-> gold[j]: preserves gold marginal counts
            if (gold[j] != abst[c] and gold[c] != abst[j]):
                gold[c], gold[j] = gold[j], gold[c]
                break
    else:
        raise RuntimeError(
            "Could not resolve gold/abstention collisions after "
            f"{max_attempts} attempts"
        )

    # Final verification
    for i in range(n_pairs):
        assert gold[i] != abst[i], f"Collision at pair {i}"
    from collections import Counter
    gold_counts = Counter(gold)
    abst_counts = Counter(abst)
    for p in range(k):
        assert gold_counts[p] == per_pos, f"Gold imbalance at pos {p}: {gold_counts}"
        assert abst_counts[p] == per_pos, f"Abst imbalance at pos {p}: {abst_counts}"

    return abst, gold


# ================================================================
# GENERATOR
# ================================================================

class T2V32Generator:
    """T2 v3.2.4 generator — balanced construction."""

    def __init__(self, seed: int = 42):
        self.master_seed = seed

    def generate_dataset(
        self,
        n_per_regime: int = 8,
        seed: int = 42,
    ) -> List[T2Item]:
        self.master_seed = seed
        rng = random.Random(seed)
        all_items = []

        clean_items_raw = []
        insufficient_items_raw = []
        items_per_template = n_per_regime // len(TEMPLATE_FAMILIES)
        remainder = n_per_regime % len(TEMPLATE_FAMILIES)
        pair_counter = 0

        for t_idx, template in enumerate(TEMPLATE_FAMILIES):
            count = items_per_template + (1 if t_idx < remainder else 0)
            for _ in range(count):
                pair_seed = rng.randint(0, 999999)
                pair_id = f"t2v32_pair_{pair_counter:04d}"
                clean_raw, insuf_raw = self._generate_counterfactual_pair(
                    template, pair_seed, pair_id
                )
                clean_raw["pair_id"] = pair_id
                insuf_raw["pair_id"] = pair_id
                clean_items_raw.append(clean_raw)
                insufficient_items_raw.append(insuf_raw)
                pair_counter += 1

        clean_items_raw = clean_items_raw[:n_per_regime]
        insufficient_items_raw = insufficient_items_raw[:n_per_regime]

        # ---- ITEM 2: Balanced position assignment for pairs ----
        # Solve the joint constraint-satisfaction problem:
        #   (a) CLEAN gold positions = exactly 2 per position {0,1,2,3}
        #   (b) Abstention positions = exactly 2 per position {0,1,2,3}
        #   (c) Within each pair, CLEAN gold != abstention (they share an array)
        #   (d) INSUFFICIENT gold == abstention (by definition)
        #
        # Assignment: explicit layout over 8 pairs.
        n_pairs = n_per_regime
        assert n_pairs == 8, (
            f"Balanced construction requires exactly 8 pairs, got {n_pairs}"
        )
        pair_abst_pos, pair_clean_gold_pos = _solve_pair_positions(
            n_pairs, k=4, rng_seed=seed + 777
        )

        # Build CLEAN and INSUFFICIENT items with controlled positions.
        # Each pair gets a shared option array built with these positions.
        for i, (c_raw, i_raw) in enumerate(
            zip(clean_items_raw, insufficient_items_raw)
        ):
            abst_pos = pair_abst_pos[i]
            clean_gold = pair_clean_gold_pos[i]

            # Build shared option array
            shared = _build_shared_option_array(
                c_raw["suspects"], abst_pos, clean_gold
            )
            c_raw["shared_options"] = shared
            c_raw["_assigned_gold_pos"] = clean_gold
            c_raw["_assigned_abst_pos"] = abst_pos
            i_raw["shared_options"] = shared
            i_raw["_assigned_gold_pos"] = abst_pos  # INSUF gold = abstention
            i_raw["_assigned_abst_pos"] = abst_pos

        # Assemble CLEAN items
        for i, raw in enumerate(clean_items_raw):
            item = self._assemble_item(
                raw,
                gold_position=raw["_assigned_gold_pos"],
                abstention_position=raw["_assigned_abst_pos"],
                rng=rng,
            )
            all_items.append(item)

        # Assemble INSUFFICIENT items
        for i, raw in enumerate(insufficient_items_raw):
            item = self._assemble_item(
                raw,
                gold_position=raw["_assigned_gold_pos"],
                abstention_position=raw["_assigned_abst_pos"],
                rng=rng,
            )
            all_items.append(item)

        # ---- DECOY and CONFLICT: independent balanced positions ----
        decoy_items_raw = self._generate_standalone_items(
            "DECOY", n_per_regime, rng, len(clean_items_raw)
        )
        conflict_items_raw = self._generate_standalone_items(
            "CONFLICT", n_per_regime, rng,
            len(clean_items_raw) + len(decoy_items_raw)
        )

        for regime_idx, (regime, raws) in enumerate([
            ("DECOY", decoy_items_raw),
            ("CONFLICT", conflict_items_raw),
        ]):
            n = len(raws)
            regime_seed_offset = (regime_idx + 2) * 1000  # offset past CLEAN/INSUF
            gold_positions, abstention_positions = \
                _joint_balanced_positions(
                    n, k=4, rng_seed=seed + regime_seed_offset + 1
                )
            for i, raw in enumerate(raws):
                item = self._assemble_item(
                    raw,
                    gold_position=gold_positions[i],
                    abstention_position=abstention_positions[i],
                    rng=rng,
                )
                all_items.append(item)

        return all_items

    def _generate_counterfactual_pair(
        self, template_key: str, seed: int, pair_id: str,
    ) -> Tuple[Dict, Dict]:
        rng = random.Random(seed)
        spec = FAMILY_EVIDENCE_SPECS[template_key]
        suspects = draw_length_matched_suspects(rng, 3)
        company = rng.choice(COMPANIES)
        location = rng.choice(LOCATIONS)
        incident_date = rng.choice([
            "March 15, 2024", "April 22, 2024", "May 8, 2024"
        ])
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)
        investigation_rule = make_investigation_rule(
            spec["criterion_a_name"], spec["criterion_b_name"]
        )
        narrative = self._make_narrative(
            template_key, company, location, incident_date
        )
        question = self._make_question()

        # Shared option array built later by _assemble_item with explicit
        # position assignments (Item 2 balanced construction).

        # ANSWERABLE
        crit_a_ans = [1, 1, 0]
        crit_b_ans = [1, 0, 1]
        evidence_ans = self._build_evidence(
            suspects, access_times, spec, crit_a_ans, crit_b_ans,
            rng=random.Random(seed + 1),
        )
        g_idx, label, viable = symbolic_label(crit_a_ans, crit_b_ans)
        assert label == "ANSWERABLE" and g_idx == 0 and len(viable) == 1
        oracle = exhaustive_viability_oracle(crit_a_ans, crit_b_ans)
        assert oracle["oracle_label"] == "ANSWERABLE"

        clean_raw = {
            "item_id": f"{pair_id}_clean",
            "regime": "CLEAN",
            "template": template_key,
            "suspects": list(suspects),
            "guilty": suspects[0],
            "guilty_idx": 0,
            "narrative": narrative,
            "question": question,
            "investigation_rule": investigation_rule,
            "evidence": evidence_ans,
            "gold_reasoning": (
                f"By the investigation rule: {suspects[0]} meets both criteria "
                f"(criterion A and criterion B). "
                f"{suspects[1]} meets only criterion A. "
                f"{suspects[2]} meets only criterion B. "
                f"Exactly one suspect meets both conditions."
            ),
            "source_precedence_rule": None,
            "mechanism": "conjunction_answerable",
            "criterion_a_assignment": crit_a_ans,
            "criterion_b_assignment": crit_b_ans,
            "viable_suspects": viable,
            "oracle_result": oracle,
        }

        # INSUFFICIENT
        crit_a_ins = [1, 1, 0]
        crit_b_ins = [1, 1, 0]
        evidence_ins = self._build_evidence(
            suspects, access_times, spec, crit_a_ins, crit_b_ins,
            rng=random.Random(seed + 1),
        )
        g_idx_ins, label_ins, viable_ins = symbolic_label(crit_a_ins, crit_b_ins)
        assert label_ins == "INSUFFICIENT" and len(viable_ins) >= 2
        oracle_ins = exhaustive_viability_oracle(crit_a_ins, crit_b_ins)
        assert oracle_ins["oracle_label"] == "INSUFFICIENT"

        insuf_raw = {
            "item_id": f"{pair_id}_insuf",
            "regime": "INSUFFICIENT",
            "template": template_key,
            "suspects": list(suspects),
            "guilty": None,
            "guilty_idx": None,
            "narrative": narrative,
            "question": question,
            "investigation_rule": investigation_rule,
            "evidence": evidence_ins,
            "gold_reasoning": (
                f"By the investigation rule: both {suspects[0]} and "
                f"{suspects[1]} meet both criteria. {suspects[2]} meets "
                f"neither criterion. Multiple suspects satisfy both "
                f"conditions, so responsibility cannot be uniquely determined."
            ),
            "source_precedence_rule": None,
            "mechanism": "conjunction_insufficient",
            "criterion_a_assignment": crit_a_ins,
            "criterion_b_assignment": crit_b_ins,
            "viable_suspects": viable_ins,
            "oracle_result": oracle_ins,
        }
        return clean_raw, insuf_raw

    def _build_evidence(
        self, suspects, access_times, spec, crit_a_assignment,
        crit_b_assignment, rng,
    ) -> List[Dict]:
        evidence = []
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": ACCESS_TEMPLATE.format(
                    suspect=suspect, time=access_times[i]),
                "supports": [suspect], "contradicts": [],
                "diagnostic_value": "medium",
            })
        for i, suspect in enumerate(suspects):
            tmpl = spec["criterion_a_pos"] if crit_a_assignment[i] == 1 \
                else spec["criterion_a_neg"]
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": tmpl.format(suspect=suspect),
                "supports": [], "contradicts": [],
                "diagnostic_value": "high",
            })
        for i, suspect in enumerate(suspects):
            tmpl = spec["criterion_b_pos"] if crit_b_assignment[i] == 1 \
                else spec["criterion_b_neg"]
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": tmpl.format(suspect=suspect),
                "supports": [], "contradicts": [],
                "diagnostic_value": "high",
            })
        neutral_idx = rng.randint(0, len(NEUTRAL_POOL) - 1)
        evidence.append({
            "id": f"E{len(evidence)+1:03d}",
            "content": NEUTRAL_POOL[neutral_idx],
            "supports": [], "contradicts": [],
            "diagnostic_value": "none",
        })
        assert len(evidence) == N_EVIDENCE_SLOTS
        rng_shuffle = random.Random(rng.randint(0, 999999))
        rng_shuffle.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"
        return evidence

    def _generate_standalone_items(self, regime, n_items, rng, id_offset):
        items_raw = []
        items_per_template = n_items // len(TEMPLATE_FAMILIES)
        remainder = n_items % len(TEMPLATE_FAMILIES)
        item_counter = id_offset
        for t_idx, template in enumerate(TEMPLATE_FAMILIES):
            count = items_per_template + (1 if t_idx < remainder else 0)
            for _ in range(count):
                item_seed = rng.randint(0, 999999)
                item_id = f"t2v32_{regime.lower()}_{template}_{item_counter:04d}"
                if regime == "DECOY":
                    raw = self._generate_decoy_item(template, item_seed, item_id)
                else:
                    raw = self._generate_conflict_item(template, item_seed, item_id)
                items_raw.append(raw)
                item_counter += 1
        return items_raw

    def _generate_decoy_item(self, template_key, seed, item_id):
        """DECOY: conjunction mechanism + distracting motive in access slots.
        NAME-FREQUENCY FIX (defect 6): Each suspect appears exactly 3 times
        in evidence (access + criterion A + criterion B). The decoy motive
        is embedded in the access slot text, not as an extra mention.
        """
        rng = random.Random(seed)
        spec = FAMILY_EVIDENCE_SPECS[template_key]
        suspects = draw_length_matched_suspects(rng, 3)
        company = rng.choice(COMPANIES)
        location = rng.choice(LOCATIONS)
        incident_date = rng.choice([
            "March 15, 2024", "April 22, 2024", "May 8, 2024"
        ])
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)
        investigation_rule = make_investigation_rule(
            spec["criterion_a_name"], spec["criterion_b_name"]
        )
        guilty_idx = rng.randint(0, 2)
        guilty = suspects[guilty_idx]
        narrative = self._make_narrative(
            template_key, company, location, incident_date
        )
        question = self._make_question()

        # Same 2-pos/1-neg structure as CLEAN but with guilty at a variable index
        others = [i for i in range(3) if i != guilty_idx]
        crit_a = [0, 0, 0]
        crit_a[guilty_idx] = 1
        crit_a[others[0]] = 1  # guilty + one other get positive criterion A
        crit_b = [0, 0, 0]
        crit_b[guilty_idx] = 1
        crit_b[others[1]] = 1  # guilty + the other other get positive criterion B
        # Result: guilty has both, others[0] has only A, others[1] has only B

        evidence = []
        # Access with embedded decoy motive (same for all suspects)
        decoy_details = [
            "who had recently expressed concerns about workplace conditions",
            "who was observed reviewing restricted area procedures last week",
            "who had submitted a transfer request citing professional disagreements",
        ]
        rng.shuffle(decoy_details)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": (
                    f"Facility entry records confirm {suspect}, "
                    f"{decoy_details[i]}, was present in the building "
                    f"during the incident period at {access_times[i]}"
                ),
                "supports": [suspect], "contradicts": [],
                "diagnostic_value": "medium",
            })
        # Criterion A
        for i, suspect in enumerate(suspects):
            tmpl = spec["criterion_a_pos"] if crit_a[i] == 1 \
                else spec["criterion_a_neg"]
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": tmpl.format(suspect=suspect),
                "supports": [], "contradicts": [],
                "diagnostic_value": "high",
            })
        # Criterion B
        for i, suspect in enumerate(suspects):
            tmpl = spec["criterion_b_pos"] if crit_b[i] == 1 \
                else spec["criterion_b_neg"]
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": tmpl.format(suspect=suspect),
                "supports": [], "contradicts": [],
                "diagnostic_value": "high",
            })
        neutral_idx = rng.randint(0, len(NEUTRAL_POOL) - 1)
        evidence.append({
            "id": f"E{len(evidence)+1:03d}",
            "content": NEUTRAL_POOL[neutral_idx],
            "supports": [], "contradicts": [],
            "diagnostic_value": "none",
        })
        assert len(evidence) == N_EVIDENCE_SLOTS
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        return {
            "item_id": item_id, "regime": "DECOY",
            "template": template_key, "suspects": suspects,
            "guilty": guilty, "guilty_idx": guilty_idx,
            "narrative": narrative, "question": question,
            "investigation_rule": investigation_rule,
            "evidence": evidence,
            "gold_reasoning": (
                f"Despite decoy motive details for all suspects, the "
                f"investigation rule requires both criteria. Only {guilty} "
                f"meets both conditions."
            ),
            "source_precedence_rule": None,
            "mechanism": "conjunction_with_decoys",
            "criterion_a_assignment": crit_a,
            "criterion_b_assignment": crit_b,
        }

    def _generate_conflict_item(self, template_key, seed, item_id):
        """CONFLICT item with source-precedence resolution."""
        rng = random.Random(seed)
        spec = FAMILY_EVIDENCE_SPECS[template_key]
        suspects = draw_length_matched_suspects(rng, 3)
        company = rng.choice(COMPANIES)
        location = rng.choice(LOCATIONS)
        incident_date = rng.choice([
            "March 15, 2024", "April 22, 2024", "May 8, 2024"
        ])
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)
        investigation_rule = make_investigation_rule(
            spec["criterion_a_name"], spec["criterion_b_name"]
        )
        guilty_idx = rng.randint(0, 2)
        guilty = suspects[guilty_idx]
        innocents = [s for s in suspects if s != guilty]
        conflict_innocent = rng.choice(innocents)
        narrative = self._make_narrative(
            template_key, company, location, incident_date
        )
        question = self._make_question(conflict=True)
        witnesses = rng.sample(WITNESSES, 3)
        evidence = []
        # 3 access
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": ACCESS_TEMPLATE.format(
                    suspect=suspect, time=access_times[i]),
                "supports": [suspect], "contradicts": [],
                "diagnostic_value": "medium",
            })
        # 3 exonerating testimony (one per suspect - balanced mentions)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": (
                    f"{witnesses[i]} testified they saw {suspect} "
                    f"leaving the building before the incident window"
                ),
                "supports": [], "contradicts": [suspect],
                "diagnostic_value": "medium",
            })
        # 1 official record -> guilty (no suspect name, uses time reference)
        evidence.append({
            "id": f"E{len(evidence)+1:03d}",
            "content": (
                f"Automated badge system recorded entry to the restricted "
                f"area during the incident window on {incident_date} by "
                f"the employee who entered the building at "
                f"{access_times[guilty_idx]}"
            ),
            "supports": [], "contradicts": [],
            "diagnostic_value": "high",
        })
        # 1 unverified report -> innocent
        ci_idx = suspects.index(conflict_innocent)
        evidence.append({
            "id": f"E{len(evidence)+1:03d}",
            "content": (
                f"An unverified witness account claims the person seen "
                f"in the restricted area matched someone who entered "
                f"the building at {access_times[ci_idx]}"
            ),
            "supports": [], "contradicts": [],
            "diagnostic_value": "high",
        })
        # 2 neutral
        neutral_indices = rng.sample(range(len(NEUTRAL_POOL)), 2)
        for ni in neutral_indices:
            evidence.append({
                "id": f"E{len(evidence)+1:03d}",
                "content": NEUTRAL_POOL[ni],
                "supports": [], "contradicts": [],
                "diagnostic_value": "none",
            })
        assert len(evidence) == N_EVIDENCE_SLOTS
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"
        source_precedence_rule = (
            "When evidence sources conflict, official automated system "
            "logs take precedence over witness testimony and "
            "unverified accounts."
        )
        return {
            "item_id": item_id, "regime": "CONFLICT",
            "template": template_key, "suspects": suspects,
            "guilty": guilty, "guilty_idx": guilty_idx,
            "narrative": narrative, "question": question,
            "investigation_rule": investigation_rule,
            "evidence": evidence,
            "gold_reasoning": (
                f"Conflicting evidence: automated log points to {guilty}, "
                f"unverified report points to {conflict_innocent}. "
                f"Applying precedence rule, {guilty} is responsible."
            ),
            "source_precedence_rule": source_precedence_rule,
            "mechanism": "source_precedence_conflict",
        }

    def _make_narrative(self, template_key, company, location, incident_date):
        if "theft" in template_key:
            return (
                f"At {company}, a valuable item was reported missing from "
                f"{location} on {incident_date} between 8:00 PM and "
                f"9:00 PM. Investigation has identified 3 employees with "
                f"potential involvement."
            )
        elif "sabotage" in template_key:
            return (
                f"At {company}, critical equipment at {location} was found "
                f"disabled on {incident_date} between 8:00 PM and 9:00 PM. "
                f"3 employees had the access and knowledge required."
            )
        elif "data_breach" in template_key:
            return (
                f"At {company}, unauthorized access to sensitive data was "
                f"detected on {incident_date} between 8:00 PM and 9:00 PM. "
                f"3 employees had sufficient system access."
            )
        else:
            return (
                f"At {company}, a production batch at {location} was found "
                f"contaminated on {incident_date} between 8:00 PM and "
                f"9:00 PM. 3 staff members had access during production."
            )

    def _make_question(self, conflict=False):
        if conflict:
            return ("Based on the investigation rule, the evidence, and the "
                    "stated precedence rules, who is responsible?")
        return ("Based on the investigation rule and the available evidence, "
                "who is responsible?")

    def _assemble_item(self, raw, gold_position, abstention_position, rng):
        suspects = raw["suspects"]
        guilty = raw["guilty"]
        regime = raw["regime"]
        is_insufficient = (regime == "INSUFFICIENT")

        # DEFECT 2 FIX: Use shared, length-matched option array for
        # CLEAN/INSUFFICIENT pair members.
        if "shared_options" in raw:
            # Pair member: use the pre-built shared array verbatim.
            # Both members get IDENTICAL option text in IDENTICAL order.
            slots = list(raw["shared_options"])
            assert len(slots) == 4

            # Determine gold_answer from the shared array
            if is_insufficient:
                # Gold = the abstention option (contains "no listed suspect")
                gold_answer = next(
                    s for s in slots if "no listed suspect" in s)
                # Override gold_position to the actual position in shared array
                gold_position = slots.index(gold_answer)
            else:
                # Gold = the guilty suspect's hypothesis
                guilty_fragment = f"{guilty} is uniquely identified"
                gold_answer = next(
                    s for s in slots if guilty_fragment in s)
                gold_position = slots.index(gold_answer)
                # abstention_position: find abstention in shared array
                abstention_position = next(
                    i for i, s in enumerate(slots)
                    if "no listed suspect" in s)
        else:
            # Standalone item (DECOY, CONFLICT): build options normally
            suspect_hyps = [
                SUSPECT_HYPOTHESIS_TEMPLATE.format(name=s) for s in suspects
            ]
            # Length-match standalone items too
            all_opts = suspect_hyps + [ABSTENTION_HYPOTHESIS]
            all_opts = _length_match_options(all_opts)
            suspect_hyps_lm = [o for o in all_opts if "no listed suspect" not in o]
            abstention_lm = next(o for o in all_opts if "no listed suspect" in o)

            slots = [None] * 4
            if is_insufficient:
                gold_answer = abstention_lm
                slots[gold_position] = abstention_lm
                remaining = [p for p in range(4) if p != gold_position]
                hyp_list = list(suspect_hyps_lm)
                rng.shuffle(hyp_list)
                for pos, hyp in zip(remaining, hyp_list):
                    slots[pos] = hyp
            else:
                guilty_fragment = f"{guilty} is uniquely identified"
                gold_answer = next(h for h in suspect_hyps_lm if guilty_fragment in h)
                slots[gold_position] = gold_answer
                slots[abstention_position] = abstention_lm
                other_hyps = [h for h in suspect_hyps_lm if h != gold_answer]
                rng.shuffle(other_hyps)
                remaining = [p for p in range(4) if slots[p] is None]
                for pos, hyp in zip(remaining, other_hyps):
                    slots[pos] = hyp
        assert all(s is not None for s in slots)

        name_freqs = {
            s: sum(1 for e in raw["evidence"] if s in e["content"])
            for s in suspects
        }

        # For CONFLICT items, include the source-precedence rule in the prompt
        # so the oracle can parse it from the rendered text alone.
        inv_rule = raw["investigation_rule"]
        if raw.get("source_precedence_rule"):
            inv_rule = inv_rule + "\n\nSOURCE PRECEDENCE RULE: " + raw["source_precedence_rule"]

        prompt_text = serialize_prompt(
            inv_rule, raw["narrative"],
            raw["question"], slots, raw["evidence"],
        )

        metadata = {
            "template": raw["template"],
            "n_suspects": len(suspects),
            "n_evidence": len(raw["evidence"]),
            "mechanism": raw["mechanism"],
            "v32": True, "v32_version": __version__,
            "gold_position": gold_position,
            "abstention_position": (
                gold_position if is_insufficient else abstention_position
            ),
            "name_frequencies": name_freqs,
            "investigation_rule": raw["investigation_rule"],
            "serialized_prompt": prompt_text,
        }
        if guilty is not None:
            metadata["guilty_suspect"] = guilty
            metadata["guilty_position"] = raw["guilty_idx"]
        for key in ["pair_id", "criterion_a_assignment",
                     "criterion_b_assignment", "viable_suspects",
                     "oracle_result"]:
            if key in raw:
                metadata[key] = raw[key]

        return T2Item(
            id=raw["item_id"], regime=regime,
            narrative=raw["narrative"], question=raw["question"],
            hypotheses=slots, evidence=raw["evidence"],
            gold_answer=gold_answer,
            gold_reasoning=raw["gold_reasoning"],
            source_precedence_rule=raw.get("source_precedence_rule"),
            metadata=metadata,
        )


# ================================================================
# VERIFICATION UTILITIES
# ================================================================

def tokenize_item_prompt(item: T2Item) -> Counter:
    """Tokenize the full serialized prompt."""
    return tokenize_text(item.metadata.get("serialized_prompt", ""))


def verify_prompt_multiset_equality(
    item_a: T2Item, item_b: T2Item
) -> Tuple[bool, Counter, Counter, str, str]:
    """Verify exact multiset equality over full serialized prompts.
    Returns (equal, only_in_a, only_in_b, hash_a, hash_b).
    """
    ca = tokenize_item_prompt(item_a)
    cb = tokenize_item_prompt(item_b)
    ha = multiset_hash(ca)
    hb = multiset_hash(cb)
    return ca == cb, ca - cb, cb - ca, ha, hb
