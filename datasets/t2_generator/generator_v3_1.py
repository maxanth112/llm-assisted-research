"""
T2 v3.1 Generator — Evidence-Normalized, Lexically-Parallel Variant

NEW variant alongside v3 (does NOT replace v3 or v2).

Addresses the two root causes of the v3 dev-gate FAIL:

  1. Abstention text asymmetry: v3 used "Cannot be determined from available
     evidence" for abstention vs "[Name] is responsible" for suspects.  A
     TF-IDF classifier trivially distinguished these surface forms, achieving
     100% accuracy on INSUFFICIENT items.

     v3.1 FIX: All hypotheses use the same syntactic frame:
       - Suspect: "Based on the available evidence, [Name] is uniquely
                   identified as responsible."
       - Abstention: "Based on the available evidence, no listed suspect
                      is uniquely identified as responsible."
     Bag-of-words overlap is now near-maximal between suspect and abstention.

  2. Evidence-count regime leak: v2 generated different numbers of evidence
     items per regime (CLEAN=7, CONFLICT=8, DECOY=10, INSUFFICIENT=7).
     v3 inherited this since it wraps v2 without modifying evidence.

     v3.1 FIX: ALL regimes produce exactly N_EVIDENCE_SLOTS evidence items.
     The slot PURPOSE varies by regime (incriminating / exonerating / decoy /
     conflicting / ambiguous), but the slot count, surface template, and
     approximate length are matched.  Only the logical RELATIONS among
     evidence items differ across regimes.

  3. Paired answerable/insufficient items: v3.1 generates paired items
     where an answerable (CLEAN-type) item and an insufficient item share
     the same surface structure, differing ONLY in the evidentiary relations
     that determine whether a unique conclusion exists.

Surface gates enforced BY CONSTRUCTION (deterministic, no probabilistic
criteria):
  S1: Universal 4-option (3 suspects + 1 abstention)
  S2: Gold-position max-diff <= 1 per regime (round-robin)
  S3: Abstention-position max-diff <= 1 per regime (round-robin)
  S4: Cross-regime evidence normalization (identical slot count)
  S5: Option text length within 20% relative band
  S6: Hypothesis order randomized per item after position assignment

All invariants are deterministic. NO probabilistic criteria.
"""

__version__ = "3.1.0"

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field

from datasets.t2_generator.generator import (
    T2Generator as V2Generator,
    T2Item,
    SUSPECTS,
    WITNESSES,
    LOCATIONS,
    COMPANIES,
    FILLER_PHRASES,
)


# ================================================================
# CONSTANTS
# ================================================================

# Lexically-parallel hypothesis templates.
# The critical design: abstention and suspect options share the same
# syntactic frame so bag-of-words cannot distinguish them.
SUSPECT_HYPOTHESIS_TEMPLATE = (
    "Based on the available evidence, {name} is uniquely identified "
    "as responsible."
)
ABSTENTION_HYPOTHESIS = (
    "Based on the available evidence, no listed suspect is uniquely "
    "identified as responsible."
)

# Fixed evidence slot count for ALL regimes.
N_EVIDENCE_SLOTS = 10

# Templates for evidence-slot content (regime-agnostic surface form).
# Each slot has a "purpose" that varies by regime, but the surface
# template is chosen from the same pool across regimes.
EVIDENCE_TEMPLATES_ACCESS = [
    "Access logs show {suspect} was present near the scene at {time}",
    "Badge records indicate {suspect} entered the area at {time}",
]

EVIDENCE_TEMPLATES_ALIBI_CONFIRMED = [
    "{suspect} claims to have been at {location} during the incident, "
    "confirmed by {witness}",
    "{suspect} reported being at {location} at the time, verified by "
    "{witness}",
]

EVIDENCE_TEMPLATES_ALIBI_UNVERIFIED = [
    "{suspect} claims to have been at {location} during the incident, "
    "but {witness} cannot confirm or deny this",
    "{suspect} reported being at {location} at the time, but {witness} "
    "is unable to verify this claim",
]

EVIDENCE_TEMPLATES_MOTIVE = [
    "{suspect} had recently expressed frustration about work "
    "assignments in private conversations",
    "{suspect} was observed displaying unusual behavior during the "
    "week prior to the incident",
    "{suspect} had financial pressures that could theoretically "
    "provide motivation for misconduct",
]

EVIDENCE_TEMPLATES_CROSS_REF = [
    "{witness} was actually on approved leave and not present at work "
    "on {date}, according to HR records",
    "{location}'s security logs show no entries during the incident "
    "window on {date}, per the automated access system",
]

EVIDENCE_TEMPLATES_CONFLICT_OFFICIAL = [
    "Automated security system recorded badge access in the restricted "
    "area during the incident window on {date}. The badge belongs to "
    "an employee who was also logged entering at {time}",
]

EVIDENCE_TEMPLATES_CONFLICT_WITNESS = [
    "An unverified witness report claims that the person seen at the "
    "restricted area matched the description of someone who was also "
    "logged entering at {time}",
]

EVIDENCE_TEMPLATES_AMBIGUITY = [
    "Critical security footage from the incident window is corrupted "
    "and cannot be recovered for analysis",
    "The electronic access log system experienced intermittent failures "
    "during the incident window, making records unreliable",
    "Environmental monitoring data from the incident period has been "
    "flagged as potentially compromised during routine audit",
]

# The 8 template families from v2
TEMPLATE_FAMILIES = [
    "theft_alibi", "theft_timeline",
    "sabotage_alibi", "sabotage_timeline",
    "data_breach_alibi", "data_breach_timeline",
    "contamination_alibi", "contamination_timeline",
]


# ================================================================
# GENERATOR
# ================================================================

class T2V31Generator:
    """T2 v3.1 generator with evidence normalization and lexical parallelism.

    This generator does NOT wrap v2 or v3. It generates items directly
    to ensure evidence-slot normalization across regimes.
    """

    def __init__(self, seed: int = 42):
        self.master_seed = seed

    def generate_dataset(
        self,
        n_per_regime: int = 8,
        seed: int = 42,
    ) -> List[T2Item]:
        """Generate a balanced dataset with v3.1 invariants.

        Invariants enforced by construction:
          S1: Every item has exactly 4 hypotheses
          S2: Gold-position max-diff <= 1 per regime
          S3: Abstention-position max-diff <= 1 per regime
          S4: All regimes have exactly N_EVIDENCE_SLOTS evidence items
          S5: Option text length within tolerance (same template for all)
          S6: Hypotheses shuffled per item after position assignment

        Args:
            n_per_regime: Items per regime. Must be >= 1.
            seed: Master seed.

        Returns:
            List of T2Item with v3.1 invariants guaranteed.
        """
        self.master_seed = seed
        rng = random.Random(seed)

        regimes = ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]
        all_items = []

        for regime in regimes:
            # Distribute items across template families
            items_per_template = n_per_regime // len(TEMPLATE_FAMILIES)
            remainder = n_per_regime % len(TEMPLATE_FAMILIES)

            regime_items_raw = []
            item_counter = len(all_items)

            for t_idx, template in enumerate(TEMPLATE_FAMILIES):
                count = items_per_template + (1 if t_idx < remainder else 0)
                for i in range(count):
                    item_seed = rng.randint(0, 999999)
                    item_id = (f"t2v31_{regime.lower()}_{template}_"
                               f"{item_counter:04d}")
                    raw = self._generate_raw_item(
                        regime, template, item_seed, item_id
                    )
                    regime_items_raw.append(raw)
                    item_counter += 1

            # Now assign positions with exact balance
            n = len(regime_items_raw)
            is_insufficient = (regime == "INSUFFICIENT")

            # Use integer seeds derived from master seed, NOT hash()
            regime_seed_offset = regimes.index(regime) * 1000
            if is_insufficient:
                gold_positions = _round_robin_positions(
                    n, k=4, rng_seed=seed + regime_seed_offset + 1
                )
                abstention_positions = gold_positions  # same slot
            else:
                gold_positions, abstention_positions = \
                    _joint_balanced_positions(
                        n, k=4, rng_seed=seed + regime_seed_offset + 1
                    )

            for i, raw in enumerate(regime_items_raw):
                item = self._assemble_v31_item(
                    raw,
                    gold_position=gold_positions[i],
                    abstention_position=abstention_positions[i],
                    rng=rng,
                )
                all_items.append(item)

        return all_items

    def _generate_raw_item(
        self,
        regime: str,
        template_key: str,
        seed: int,
        item_id: str,
    ) -> Dict:
        """Generate raw item data (not yet assembled into T2Item).

        Returns a dict with: suspects, guilty (or None for INSUFFICIENT),
        narrative, question, evidence (list of dicts), gold_reasoning,
        source_precedence_rule, template, mechanism, metadata.

        ALL regimes produce exactly N_EVIDENCE_SLOTS evidence items.
        """
        rng = random.Random(seed)

        suspects = rng.sample(SUSPECTS, 3)
        location = rng.choice(LOCATIONS)
        company = rng.choice(COMPANIES)
        incident_date = rng.choice([
            "March 15, 2024", "April 22, 2024", "May 8, 2024"
        ])
        time_window = rng.choice([
            ("8:00 PM", "9:00 PM"),
            ("6:30 PM", "7:30 PM"),
            ("7:00 PM", "8:00 PM"),
        ])

        # Common entities
        alibi_witnesses = rng.sample(WITNESSES, 3)
        alibi_locations = rng.sample(
            [loc for loc in LOCATIONS if loc != location], 3
        )
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)

        # Narrative (same across all regimes for this template)
        narrative = self._make_narrative(
            template_key, company, location, incident_date,
            time_window, len(suspects), regime
        )
        question = self._make_question(template_key, regime)

        # Build evidence — exactly N_EVIDENCE_SLOTS items for every regime
        if regime == "CLEAN":
            guilty_idx = rng.randint(0, 2)
            guilty = suspects[guilty_idx]
            evidence, mechanism = self._build_clean_evidence(
                suspects, guilty, guilty_idx, template_key,
                alibi_witnesses, alibi_locations, access_times,
                incident_date, rng
            )
            gold_reasoning = (
                f"Relational reasoning: {guilty}'s alibi is invalidated "
                f"by cross-reference evidence. Innocent suspects' alibis "
                f"remain valid. Name frequency and individual evidence "
                f"strength are uninformative (all suspects have identical "
                f"evidence profiles)."
            )
            source_precedence_rule = None

        elif regime == "DECOY":
            guilty_idx = rng.randint(0, 2)
            guilty = suspects[guilty_idx]
            evidence, mechanism = self._build_decoy_evidence(
                suspects, guilty, guilty_idx, template_key,
                alibi_witnesses, alibi_locations, access_times,
                incident_date, rng
            )
            gold_reasoning = (
                f"Despite equally salient decoy evidence for all suspects, "
                f"relational reasoning reveals {guilty}'s alibi is "
                f"invalidated by cross-reference. All suspects have "
                f"identical evidence profiles."
            )
            source_precedence_rule = None

        elif regime == "CONFLICT":
            guilty_idx = rng.randint(0, 2)
            guilty = suspects[guilty_idx]
            innocent = [s for s in suspects if s != guilty]
            conflict_innocent = rng.choice(innocent)
            evidence, mechanism = self._build_conflict_evidence(
                suspects, guilty, guilty_idx, conflict_innocent,
                template_key, alibi_witnesses, alibi_locations,
                access_times, incident_date, rng
            )
            source_precedence_rule = (
                "When evidence sources conflict, official system logs "
                "and automated records take precedence over witness "
                "testimony and observational reports."
            )
            gold_reasoning = (
                f"Conflicting evidence: official automated log points to "
                f"{guilty}, unverified witness report points to "
                f"{conflict_innocent}. Applying precedence rule (official "
                f"records > witness testimony), {guilty} is responsible."
            )

        else:  # INSUFFICIENT
            guilty_idx = None
            guilty = None
            evidence, mechanism = self._build_insufficient_evidence(
                suspects, template_key,
                alibi_witnesses, alibi_locations, access_times,
                incident_date, rng
            )
            gold_reasoning = (
                "Evidence is perfectly symmetric across all suspects. "
                "Each has identical evidence profiles: access, "
                "unverifiable alibis, and no distinguishing factors. "
                "No alibi invalidation, no precedence rule applies, "
                "no relational inconsistency exists."
            )
            source_precedence_rule = None

        # Normalize evidence lengths
        self._normalize_evidence_lengths(evidence, rng)

        # Shuffle evidence order
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        return {
            "item_id": item_id,
            "regime": regime,
            "template": template_key,
            "suspects": suspects,
            "guilty": guilty,
            "guilty_idx": guilty_idx,
            "narrative": narrative,
            "question": question,
            "evidence": evidence,
            "gold_reasoning": gold_reasoning,
            "source_precedence_rule": source_precedence_rule,
            "mechanism": mechanism,
        }

    # ---- Evidence builders: each returns exactly N_EVIDENCE_SLOTS items ----

    def _build_clean_evidence(
        self, suspects, guilty, guilty_idx, template_key,
        alibi_witnesses, alibi_locations, access_times,
        incident_date, rng
    ) -> Tuple[List[Dict], str]:
        """CLEAN: 3 access + 3 confirmed alibis + 1 cross-ref + 3 neutral context.

        Total: 10 slots. The 3 neutral context slots carry information
        about the workplace/investigation that is symmetric across suspects
        and does not affect the answer.
        """
        evidence = []
        eid = 1

        # Slots 1-3: Access evidence (incriminating, one per suspect)
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ACCESS)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(suspect=suspect, time=access_times[i]),
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium",
            })
            eid += 1

        # Slots 4-6: Confirmed alibis (exonerating, one per suspect)
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ALIBI_CONFIRMED)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(
                    suspect=suspect,
                    location=alibi_locations[i],
                    witness=alibi_witnesses[i],
                ),
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high",
            })
            eid += 1

        # Slot 7: Cross-reference invalidating guilty suspect's alibi
        gw = alibi_witnesses[guilty_idx]
        gl = alibi_locations[guilty_idx]
        if "alibi" in template_key:
            cross_ref = EVIDENCE_TEMPLATES_CROSS_REF[0].format(
                witness=gw, date=incident_date
            )
        else:
            cross_ref = EVIDENCE_TEMPLATES_CROSS_REF[1].format(
                location=gl, date=incident_date
            )
        evidence.append({
            "id": f"E{eid:03d}",
            "content": cross_ref,
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high",
        })
        eid += 1

        # Slots 8-10: Neutral context (symmetric, no suspect names)
        neutral_pool = [
            "Internal review of security protocols found all access "
            "points were functioning normally during the incident period",
            "Workplace scheduling records confirm all listed employees "
            "were on active duty assignments during the relevant dates",
            "Routine compliance audit completed prior to the incident "
            "identified no procedural deviations in the affected area",
        ]
        rng.shuffle(neutral_pool)
        for text in neutral_pool[:3]:
            evidence.append({
                "id": f"E{eid:03d}",
                "content": text,
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none",
            })
            eid += 1

        assert len(evidence) == N_EVIDENCE_SLOTS
        return evidence, "alibi_chain_invalidation"

    def _build_decoy_evidence(
        self, suspects, guilty, guilty_idx, template_key,
        alibi_witnesses, alibi_locations, access_times,
        incident_date, rng
    ) -> Tuple[List[Dict], str]:
        """DECOY: 3 access + 3 confirmed alibis + 3 decoys + 1 cross-ref.

        Total: 10 slots. This is the natural count for DECOY.
        """
        evidence = []
        eid = 1

        # Slots 1-3: Access evidence
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ACCESS)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(suspect=suspect, time=access_times[i]),
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium",
            })
            eid += 1

        # Slots 4-6: Confirmed alibis
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ALIBI_CONFIRMED)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(
                    suspect=suspect,
                    location=alibi_locations[i],
                    witness=alibi_witnesses[i],
                ),
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high",
            })
            eid += 1

        # Slots 7-9: Decoy motive/behavior (one per suspect, balanced)
        motive_templates = list(EVIDENCE_TEMPLATES_MOTIVE)
        rng.shuffle(motive_templates)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{eid:03d}",
                "content": motive_templates[i].format(suspect=suspect),
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "low",
            })
            eid += 1

        # Slot 10: Cross-reference invalidating guilty suspect's alibi
        gw = alibi_witnesses[guilty_idx]
        gl = alibi_locations[guilty_idx]
        if "alibi" in template_key:
            cross_ref = EVIDENCE_TEMPLATES_CROSS_REF[0].format(
                witness=gw, date=incident_date
            )
        else:
            cross_ref = EVIDENCE_TEMPLATES_CROSS_REF[1].format(
                location=gl, date=incident_date
            )
        evidence.append({
            "id": f"E{eid:03d}",
            "content": cross_ref,
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high",
        })
        eid += 1

        assert len(evidence) == N_EVIDENCE_SLOTS
        return evidence, "alibi_chain_invalidation_with_decoys"

    def _build_conflict_evidence(
        self, suspects, guilty, guilty_idx, conflict_innocent,
        template_key, alibi_witnesses, alibi_locations, access_times,
        incident_date, rng
    ) -> Tuple[List[Dict], str]:
        """CONFLICT: 3 access + 3 exonerating + 2 conflicting + 2 neutral.

        Total: 10 slots. Added 2 neutral context slots to match.
        """
        evidence = []
        eid = 1

        witnesses = alibi_witnesses  # reuse for testimony

        # Slots 1-3: Access evidence (incriminating)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{eid:03d}",
                "content": f"Access logs show {suspect} was present near "
                           f"the scene at {access_times[i]}",
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium",
            })
            eid += 1

        # Slots 4-6: Exonerating testimony (one per suspect)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{eid:03d}",
                "content": f"{witnesses[i]} testified they saw {suspect} "
                           f"leaving the building before the incident "
                           f"window",
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "medium",
            })
            eid += 1

        # Slot 7: Official system log (HIGH precedence) -> guilty
        evidence.append({
            "id": f"E{eid:03d}",
            "content": EVIDENCE_TEMPLATES_CONFLICT_OFFICIAL[0].format(
                date=incident_date,
                time=access_times[guilty_idx],
            ),
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high",
        })
        eid += 1

        # Slot 8: Witness report (LOW precedence) -> innocent
        ci_idx = suspects.index(conflict_innocent)
        evidence.append({
            "id": f"E{eid:03d}",
            "content": EVIDENCE_TEMPLATES_CONFLICT_WITNESS[0].format(
                time=access_times[ci_idx],
            ),
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high",
        })
        eid += 1

        # Slots 9-10: Neutral context (symmetric)
        neutral_pool = [
            "Internal review of security protocols found all access "
            "points were functioning normally during the incident period",
            "Workplace scheduling records confirm all listed employees "
            "were on active duty assignments during the relevant dates",
            "Routine compliance audit completed prior to the incident "
            "identified no procedural deviations in the affected area",
        ]
        rng.shuffle(neutral_pool)
        for text in neutral_pool[:2]:
            evidence.append({
                "id": f"E{eid:03d}",
                "content": text,
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none",
            })
            eid += 1

        assert len(evidence) == N_EVIDENCE_SLOTS
        return evidence, "source_precedence_conflict"

    def _build_insufficient_evidence(
        self, suspects, template_key,
        alibi_witnesses, alibi_locations, access_times,
        incident_date, rng
    ) -> Tuple[List[Dict], str]:
        """INSUFFICIENT: 3 access + 3 unverified alibis + 3 neutral + 1 ambiguity.

        Total: 10 slots. Perfectly symmetric across suspects.
        The key difference from CLEAN: alibis are UNVERIFIABLE (not confirmed),
        and there is no cross-reference invalidation. The neutral context slots
        use the same pool as CLEAN/CONFLICT.
        """
        evidence = []
        eid = 1

        # Slots 1-3: Access evidence (same as other regimes)
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ACCESS)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(suspect=suspect, time=access_times[i]),
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "low",
            })
            eid += 1

        # Slots 4-6: Unverified alibis (key difference from CLEAN)
        for i, suspect in enumerate(suspects):
            tmpl = rng.choice(EVIDENCE_TEMPLATES_ALIBI_UNVERIFIED)
            evidence.append({
                "id": f"E{eid:03d}",
                "content": tmpl.format(
                    suspect=suspect,
                    location=alibi_locations[i],
                    witness=alibi_witnesses[i],
                ),
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none",
            })
            eid += 1

        # Slot 7: Ambiguity statement (no names, symmetric)
        ambiguity = rng.choice(EVIDENCE_TEMPLATES_AMBIGUITY)
        evidence.append({
            "id": f"E{eid:03d}",
            "content": ambiguity,
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "none",
        })
        eid += 1

        # Slots 8-10: Neutral context (same pool as CLEAN/CONFLICT)
        neutral_pool = [
            "Internal review of security protocols found all access "
            "points were functioning normally during the incident period",
            "Workplace scheduling records confirm all listed employees "
            "were on active duty assignments during the relevant dates",
            "Routine compliance audit completed prior to the incident "
            "identified no procedural deviations in the affected area",
        ]
        rng.shuffle(neutral_pool)
        for text in neutral_pool[:3]:
            evidence.append({
                "id": f"E{eid:03d}",
                "content": text,
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none",
            })
            eid += 1

        assert len(evidence) == N_EVIDENCE_SLOTS
        return evidence, "perfect_symmetry"

    # ---- Narrative / question builders ----

    def _make_narrative(self, template_key, company, location,
                        incident_date, time_window, n_suspects, regime):
        """Generate narrative text. Same surface form across regimes."""
        tw0, tw1 = time_window
        if "theft" in template_key:
            return (
                f"At {company}, a valuable item was reported missing from "
                f"{location} on {incident_date} between {tw0} and {tw1}. "
                f"Investigation has identified {n_suspects} employees with "
                f"potential access to the area."
            )
        elif "sabotage" in template_key:
            return (
                f"At {company}'s {location}, critical equipment was found "
                f"disabled on {incident_date} between {tw0} and {tw1}. "
                f"{n_suspects} employees had the technical knowledge and "
                f"access required."
            )
        elif "data_breach" in template_key:
            return (
                f"At {company}, unauthorized access to sensitive data was "
                f"detected on {incident_date} between {tw0} and {tw1}. "
                f"{n_suspects} employees had sufficient system privileges."
            )
        else:  # contamination
            return (
                f"At {company}'s {location}, a production batch was found "
                f"contaminated on {incident_date} between {tw0} and {tw1}. "
                f"{n_suspects} staff members had access during production."
            )

    def _make_question(self, template_key, regime):
        """Generate question text. Same across regimes to avoid leaking."""
        if regime == "CONFLICT":
            prefix = ("Based on the available evidence and stated "
                      "precedence rules, ")
        else:
            prefix = "Based on the available evidence, "

        if "theft" in template_key:
            return prefix + "who is most likely responsible?"
        elif "sabotage" in template_key:
            return prefix + "who is most likely responsible?"
        elif "data_breach" in template_key:
            return prefix + "who is most likely responsible?"
        else:
            return prefix + "who is most likely responsible?"

    # ---- Length normalization ----

    def _normalize_evidence_lengths(self, evidence, rng):
        """Pad shorter evidence items to match the longest (±15 chars)."""
        if not evidence:
            return
        lengths = [len(e["content"]) for e in evidence]
        max_len = max(lengths)
        for ev in evidence:
            if len(ev["content"]) < max_len - 15:
                ev["content"] = self._pad_to_length(
                    ev["content"], max_len - 10, rng
                )

    def _pad_to_length(self, text, target, rng):
        if len(text) >= target:
            return text
        filler = rng.choice(FILLER_PHRASES)
        padded = f"{text}, {filler}"
        if len(padded) < target - 15:
            return self._pad_to_length(padded, target, rng)
        return padded

    # ---- Final assembly ----

    def _assemble_v31_item(
        self,
        raw: Dict,
        gold_position: int,
        abstention_position: int,
        rng: random.Random,
    ) -> T2Item:
        """Assemble raw item data into a T2Item with v3.1 hypotheses."""
        suspects = raw["suspects"]
        guilty = raw["guilty"]
        regime = raw["regime"]
        is_insufficient = (regime == "INSUFFICIENT")

        # Build hypothesis texts using lexically-parallel template
        suspect_hyps = [
            SUSPECT_HYPOTHESIS_TEMPLATE.format(name=s) for s in suspects
        ]

        # Build 4-slot hypothesis list
        slots = [None] * 4

        if is_insufficient:
            # Gold IS abstention
            gold_answer = ABSTENTION_HYPOTHESIS
            slots[gold_position] = ABSTENTION_HYPOTHESIS
            remaining = [p for p in range(4) if p != gold_position]
            hyp_list = list(suspect_hyps)
            rng.shuffle(hyp_list)
            for pos, hyp in zip(remaining, hyp_list):
                slots[pos] = hyp
        else:
            # Gold is a suspect
            gold_answer = SUSPECT_HYPOTHESIS_TEMPLATE.format(name=guilty)
            slots[gold_position] = gold_answer
            slots[abstention_position] = ABSTENTION_HYPOTHESIS
            other_hyps = [h for h in suspect_hyps if h != gold_answer]
            rng.shuffle(other_hyps)
            remaining = [p for p in range(4) if slots[p] is None]
            for pos, hyp in zip(remaining, other_hyps):
                slots[pos] = hyp

        assert all(s is not None for s in slots), (
            f"Unfilled slot in {raw['item_id']}: {slots}"
        )

        metadata = {
            "template": raw["template"],
            "n_suspects": len(suspects),
            "n_evidence": len(raw["evidence"]),
            "mechanism": raw["mechanism"],
            "v31": True,
            "gold_position": gold_position,
            "abstention_position": (
                gold_position if is_insufficient else abstention_position
            ),
            "name_frequencies": {
                s: sum(1 for e in raw["evidence"] if s in e["content"])
                for s in suspects
            },
        }
        if guilty is not None:
            metadata["guilty_suspect"] = guilty
            metadata["guilty_position"] = raw["guilty_idx"]

        return T2Item(
            id=raw["item_id"],
            regime=regime,
            narrative=raw["narrative"],
            question=raw["question"],
            hypotheses=slots,
            evidence=raw["evidence"],
            gold_answer=gold_answer,
            gold_reasoning=raw["gold_reasoning"],
            source_precedence_rule=raw["source_precedence_rule"],
            metadata=metadata,
        )


# ================================================================
# POSITION ASSIGNMENT (same algorithms as v3)
# ================================================================

def _round_robin_positions(n: int, k: int = 4, rng_seed: int = 0) -> List[int]:
    """Assign positions 0..k-1 via shuffled round-robin. Max-diff <= 1."""
    positions = [i % k for i in range(n)]
    rng = random.Random(rng_seed)
    rng.shuffle(positions)
    return positions


def _joint_balanced_positions(
    n: int,
    k: int = 4,
    rng_seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Assign (gold, abstention) pairs with both marginals balanced.

    Strategy: two independent round-robins + collision resolution via swaps.
    Swaps preserve both marginal counts.
    """
    rng = random.Random(rng_seed)

    gold_positions = _round_robin_positions(n, k=k, rng_seed=rng_seed)
    abstention_positions = _round_robin_positions(
        n, k=k, rng_seed=rng_seed + 7
    )

    max_attempts = n * n
    for _ in range(max_attempts):
        collisions = [
            i for i in range(n)
            if gold_positions[i] == abstention_positions[i]
        ]
        if not collisions:
            break

        c = collisions[0]
        candidates = list(range(n))
        rng.shuffle(candidates)
        resolved = False
        for j in candidates:
            if j == c:
                continue
            if (gold_positions[c] != abstention_positions[j] and
                    gold_positions[j] != abstention_positions[c]):
                abstention_positions[c], abstention_positions[j] = \
                    abstention_positions[j], abstention_positions[c]
                resolved = True
                break
        if not resolved:
            abstention_positions[c] = (gold_positions[c] + 1) % k

    return gold_positions, abstention_positions
