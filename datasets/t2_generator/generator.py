"""
T2 v2 Generator - Relational Reasoning Diagnostic

CRITICAL DESIGN PRINCIPLE:
The correct answer is determined ONLY by relational reasoning - the logical
relationships BETWEEN evidence items - NOT by name frequency, evidence count,
or incriminating-sounding text.

V2 Improvements over V1:
- Name-frequency equalization: Every suspect mentioned exactly the same number of times
- Evidence-count parity: Each suspect gets exactly 2 evidence items (1 incriminating, 1 exonerating)
- Polarity balance: Guilt determined by relational inconsistency, not individual evidence
- Length matching: All evidence items have similar character lengths
- Positional uniformity: Guilty suspect position cycles through 0,1,2
- Counterfactual minimal pairs: Twin items with same tokens but different correct answers
"""

__version__ = "2.0.0"

import random
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class T2Item:
    """A single T2 diagnostic test item."""
    id: str
    regime: str  # CLEAN, DECOY, CONFLICT, INSUFFICIENT
    narrative: str
    question: str
    hypotheses: List[str]  # Always >= 3
    evidence: List[Dict[str, Any]]  # id, content, supports, contradicts, diagnostic_value
    gold_answer: str  # Hypothesis text or "CANNOT_DETERMINE"
    gold_reasoning: str
    source_precedence_rule: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# Entity pools (expanded for more diversity)
SUSPECTS = [
    "Alex Chen", "Blake Rivera", "Casey Morgan", "Dana Patel",
    "Ellis Kim", "Finley Brooks", "Grey Martinez", "Harper Quinn",
    "Indigo Taylor", "Jordan Lee", "Kelly Zhao", "Morgan Walsh"
]

LOCATIONS = [
    "Building A - East Wing", "Building B - Research Lab", "Building C - Server Room",
    "Building D - Storage Facility", "Main Office - 3rd Floor", "Warehouse District",
    "Downtown Branch Office", "Research Campus - North Wing", "Production Floor - Zone 4",
    "Security Office Complex", "Data Center - Level 2", "Testing Laboratory"
]

WITNESSES = [
    "Security Officer Sarah Johnson", "Custodian Mike Thompson",
    "IT Administrator Pat Williams", "Reception Staff Chris Anderson",
    "Maintenance Lead Sam Rodriguez", "Lab Supervisor Dr. Taylor Kim",
    "Shift Manager Alex Park", "Quality Inspector Jamie Foster",
    "HR Director Robin Hayes", "Operations Manager Drew Bennett",
    "Compliance Officer Jordan Gray", "Training Coordinator Casey Moore"
]

COMPANIES = [
    "TechCorp Industries", "DataVault Systems", "BioResearch Labs",
    "Manufacturing Solutions Inc", "Global Logistics Co", "Precision Engineering Ltd",
    "Advanced Materials Corp", "Integrated Systems Group"
]

# Neutral filler phrases for length padding
FILLER_PHRASES = [
    "as documented in the official records",
    "according to standard procedures",
    "following routine protocols",
    "in compliance with regulations",
    "as per company policy",
    "consistent with established guidelines",
    "in accordance with protocol",
    "per standard operating procedure"
]


class T2Generator:
    """Generator for T2 v2 diagnostic test items with relational reasoning."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed."""
        self.master_seed = seed
        self.rng = random.Random(seed)

    def _get_seeded_rng(self, *args) -> random.Random:
        """Create a seeded RNG from master seed and additional args."""
        seed_str = f"{self.master_seed}_{'_'.join(map(str, args))}"
        seed_val = hash(seed_str) % (2**32)
        return random.Random(seed_val)

    def _pad_to_length(self, text: str, target_length: int, rng: random.Random) -> str:
        """Pad text to target length with neutral filler phrases."""
        if len(text) >= target_length:
            return text

        filler = rng.choice(FILLER_PHRASES)
        padded = f"{text}, {filler}"

        # Recursively add more filler if needed
        if len(padded) < target_length - 15:
            return self._pad_to_length(padded, target_length, rng)
        return padded

    def _normalize_evidence_lengths(self, evidence_list: List[Dict], rng: random.Random):
        """Normalize all evidence items to similar lengths (±15 chars)."""
        if not evidence_list:
            return

        lengths = [len(e["content"]) for e in evidence_list]
        max_length = max(lengths)

        for ev in evidence_list:
            if len(ev["content"]) < max_length - 15:
                ev["content"] = self._pad_to_length(ev["content"], max_length - 10, rng)

    def generate_clean_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a CLEAN regime item using alibi-chain invalidation.

        Mechanism:
        - Each suspect gets 1 incriminating evidence (access to scene)
        - Each suspect gets 1 exonerating evidence (alibi)
        - For guilty suspect ONLY, a cross-reference invalidates their alibi WITHOUT naming them
        - Name frequency is perfectly balanced
        """
        rng = self._get_seeded_rng("clean", template_key, seed)

        # Select 3 suspects and cycle guilty position uniformly
        suspects = rng.sample(SUSPECTS, 3)
        guilty_idx = rng.randint(0, 2)  # Uniform random over 0,1,2
        guilty = suspects[guilty_idx]
        innocent = [s for s in suspects if s != guilty]

        # Select entities
        location = rng.choice(LOCATIONS)
        company = rng.choice(COMPANIES)

        # Generate narrative
        incident_date = rng.choice(["March 15, 2024", "April 22, 2024", "May 8, 2024"])
        time_window = rng.choice([("8:00 PM", "9:00 PM"), ("6:30 PM", "7:30 PM"), ("7:00 PM", "8:00 PM")])

        if template_key == "theft_alibi":
            item_type = rng.choice(["prototype device", "confidential document set", "access key"])
            narrative = f"At {company}, a valuable {item_type} was stolen from {location} on {incident_date} between {time_window[0]} and {time_window[1]}. Investigation has identified {len(suspects)} employees with potential access."
            question = "Based on the available evidence, who is most likely responsible for the theft?"
        elif template_key == "theft_timeline":
            item_type = rng.choice(["research sample", "data backup drive", "security badge"])
            narrative = f"At {company}, a {item_type} was removed from {location} on {incident_date}. The theft occurred during a specific time window between {time_window[0]} and {time_window[1]}. {len(suspects)} employees are under investigation."
            question = "Based on the available evidence, who is most likely responsible for the theft?"
        elif template_key == "sabotage_alibi":
            narrative = f"At {company}'s {location}, critical equipment was sabotaged on {incident_date}, causing a production shutdown. The sabotage occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had the technical knowledge and access."
            question = "Based on the available evidence, who is most likely responsible for the sabotage?"
        elif template_key == "sabotage_timeline":
            narrative = f"Critical systems at {company}'s {location} were deliberately disabled on {incident_date}. System logs indicate the sabotage occurred during a narrow window between {time_window[0]} and {time_window[1]}. {len(suspects)} individuals had access."
            question = "Based on the available evidence, who is most likely responsible for the sabotage?"
        elif template_key == "data_breach_alibi":
            narrative = f"At {company}, unauthorized access to sensitive data was detected on {incident_date}. The breach occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had sufficient system privileges to access the compromised database."
            question = "Based on the available evidence, who is most likely responsible for the data breach?"
        elif template_key == "data_breach_timeline":
            narrative = f"At {company}, customer data was exfiltrated on {incident_date}. Network logs show the breach occurred in a specific timeframe between {time_window[0]} and {time_window[1]}. Investigation focuses on {len(suspects)} employees."
            question = "Based on the available evidence, who is most likely responsible for the data breach?"
        elif template_key == "contamination_alibi":
            narrative = f"At {company}'s {location}, a production batch was contaminated on {incident_date}, requiring a costly recall. The contamination occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} staff members had access during production."
            question = "Based on the available evidence, who is most likely responsible for the contamination?"
        else:  # contamination_timeline
            narrative = f"A contamination incident at {company}'s {location} occurred on {incident_date}. Quality control traced the contamination to a specific production window between {time_window[0]} and {time_window[1]}. {len(suspects)} operators are under investigation."
            question = "Based on the available evidence, who is most likely responsible for the contamination?"

        # Build evidence using alibi-chain invalidation
        evidence = []
        evidence_id = 1

        # Assign witnesses and locations for alibis
        alibi_witnesses = rng.sample(WITNESSES, 3)
        alibi_locations = rng.sample([loc for loc in LOCATIONS if loc != location], 3)
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)

        # Phase 1: Each suspect gets incriminating evidence (access to scene)
        for i, suspect in enumerate(suspects):
            if "alibi" in template_key:
                content = f"Access logs show {suspect} was present near the scene at {access_times[i]}"
            else:  # timeline variant
                content = f"Badge records indicate {suspect} entered the area at {access_times[i]}"

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": content,
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium"
            })
            evidence_id += 1

        # Phase 2: Each suspect gets exonerating evidence (alibi)
        for i, suspect in enumerate(suspects):
            if "alibi" in template_key:
                content = f"{suspect} claims to have been at {alibi_locations[i]} during the incident, confirmed by {alibi_witnesses[i]}"
            else:  # timeline variant
                content = f"{suspect} reported being at {alibi_locations[i]} at the time, verified by {alibi_witnesses[i]}"

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": content,
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        # Phase 3: Cross-reference that invalidates ONLY guilty suspect's alibi (WITHOUT naming them)
        guilty_witness_idx = suspects.index(guilty)
        guilty_witness = alibi_witnesses[guilty_witness_idx]
        guilty_location = alibi_locations[guilty_witness_idx]

        if "alibi" in template_key:
            # Invalidate via witness unavailability
            cross_ref = f"{guilty_witness} was actually on approved leave and not present at work on {incident_date}, according to HR records"
        else:  # timeline variant
            # Invalidate via location access logs
            cross_ref = f"{guilty_location}'s security logs show no entries during the incident window on {incident_date}, per the automated access system"

        evidence.append({
            "id": f"E{evidence_id:03d}",
            "content": cross_ref,
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high"
        })
        evidence_id += 1

        # Normalize evidence lengths
        self._normalize_evidence_lengths(evidence, rng)

        # Shuffle evidence to avoid position bias
        rng.shuffle(evidence)

        # Reassign evidence IDs after shuffling
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        gold_reasoning = f"The evidence requires relational reasoning: {guilty}'s alibi is invalidated by the cross-reference showing that {guilty_witness if 'alibi' in template_key else guilty_location} could not have confirmed their whereabouts. The innocent suspects' alibis remain valid. This cannot be determined by name frequency (all suspects mentioned equally) or individual evidence strength (all have both incriminating and exonerating evidence)."

        return T2Item(
            id=item_id,
            regime="CLEAN",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{guilty} is responsible",
            gold_reasoning=gold_reasoning,
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "guilty_suspect": guilty,
                "guilty_position": guilty_idx,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "mechanism": "alibi_chain_invalidation",
                "name_frequencies": {s: sum(1 for e in evidence if s in e["content"]) for s in suspects}
            }
        )

    def generate_decoy_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a DECOY regime item.

        Same as CLEAN but adds equally salient decoy evidence for ALL suspects.
        Each suspect gets +1 decoy evidence item (motive/behavior).
        Name frequency remains balanced.
        """
        rng = self._get_seeded_rng("decoy", template_key, seed)

        # Select 3 suspects and cycle guilty position uniformly
        suspects = rng.sample(SUSPECTS, 3)
        guilty_idx = rng.randint(0, 2)  # Uniform random over 0,1,2
        guilty = suspects[guilty_idx]

        # Select entities
        location = rng.choice(LOCATIONS)
        company = rng.choice(COMPANIES)

        # Generate narrative (same structure as CLEAN)
        incident_date = rng.choice(["March 15, 2024", "April 22, 2024", "May 8, 2024"])
        time_window = rng.choice([("8:00 PM", "9:00 PM"), ("6:30 PM", "7:30 PM"), ("7:00 PM", "8:00 PM")])

        if template_key == "theft_alibi":
            item_type = rng.choice(["prototype device", "confidential document set", "access key"])
            narrative = f"At {company}, a valuable {item_type} was stolen from {location} on {incident_date} between {time_window[0]} and {time_window[1]}. Investigation has identified {len(suspects)} employees with potential access."
            question = "Based on the available evidence, who is most likely responsible for the theft?"
        elif template_key == "theft_timeline":
            item_type = rng.choice(["research sample", "data backup drive", "security badge"])
            narrative = f"At {company}, a {item_type} was removed from {location} on {incident_date}. The theft occurred during a specific time window between {time_window[0]} and {time_window[1]}. {len(suspects)} employees are under investigation."
            question = "Based on the available evidence, who is most likely responsible for the theft?"
        elif template_key == "sabotage_alibi":
            narrative = f"At {company}'s {location}, critical equipment was sabotaged on {incident_date}, causing a production shutdown. The sabotage occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had the technical knowledge and access."
            question = "Based on the available evidence, who is most likely responsible for the sabotage?"
        elif template_key == "sabotage_timeline":
            narrative = f"Critical systems at {company}'s {location} were deliberately disabled on {incident_date}. System logs indicate the sabotage occurred during a narrow window between {time_window[0]} and {time_window[1]}. {len(suspects)} individuals had access."
            question = "Based on the available evidence, who is most likely responsible for the sabotage?"
        elif template_key == "data_breach_alibi":
            narrative = f"At {company}, unauthorized access to sensitive data was detected on {incident_date}. The breach occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had sufficient system privileges to access the compromised database."
            question = "Based on the available evidence, who is most likely responsible for the data breach?"
        elif template_key == "data_breach_timeline":
            narrative = f"At {company}, customer data was exfiltrated on {incident_date}. Network logs show the breach occurred in a specific timeframe between {time_window[0]} and {time_window[1]}. Investigation focuses on {len(suspects)} employees."
            question = "Based on the available evidence, who is most likely responsible for the data breach?"
        elif template_key == "contamination_alibi":
            narrative = f"At {company}'s {location}, a production batch was contaminated on {incident_date}, requiring a costly recall. The contamination occurred between {time_window[0]} and {time_window[1]}. {len(suspects)} staff members had access during production."
            question = "Based on the available evidence, who is most likely responsible for the contamination?"
        else:  # contamination_timeline
            narrative = f"A contamination incident at {company}'s {location} occurred on {incident_date}. Quality control traced the contamination to a specific production window between {time_window[0]} and {time_window[1]}. {len(suspects)} operators are under investigation."
            question = "Based on the available evidence, who is most likely responsible for the contamination?"

        # Build evidence
        evidence = []
        evidence_id = 1

        alibi_witnesses = rng.sample(WITNESSES, 3)
        alibi_locations = rng.sample([loc for loc in LOCATIONS if loc != location], 3)
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)

        # Phase 1: Incriminating evidence for each suspect
        for i, suspect in enumerate(suspects):
            content = f"Access logs show {suspect} was present near the scene at {access_times[i]}"
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": content,
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium"
            })
            evidence_id += 1

        # Phase 2: Alibis for each suspect
        for i, suspect in enumerate(suspects):
            content = f"{suspect} claims to have been at {alibi_locations[i]} during the incident, confirmed by {alibi_witnesses[i]}"
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": content,
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        # Phase 3: DECOY evidence for ALL suspects (balanced motive/behavior)
        decoy_templates = [
            "{suspect} had recently expressed frustration about work assignments in private conversations",
            "{suspect} was observed displaying unusual behavior during the week prior to the incident",
            "{suspect} had financial pressures that could theoretically provide motivation for misconduct"
        ]

        for i, suspect in enumerate(suspects):
            decoy_text = decoy_templates[i].format(suspect=suspect)
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": decoy_text,
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        # Phase 4: Cross-reference invalidating guilty suspect's alibi
        guilty_witness_idx = suspects.index(guilty)
        guilty_witness = alibi_witnesses[guilty_witness_idx]

        cross_ref = f"{guilty_witness} was actually on approved leave and not present at work on {incident_date}, according to HR records"
        evidence.append({
            "id": f"E{evidence_id:03d}",
            "content": cross_ref,
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high"
        })
        evidence_id += 1

        # Normalize evidence lengths
        self._normalize_evidence_lengths(evidence, rng)

        # Shuffle evidence
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        gold_reasoning = f"Despite equally salient decoy evidence for all suspects (motive and behavior), relational reasoning reveals {guilty}'s alibi is invalidated by the cross-reference. All suspects have identical evidence profiles (1 incriminating, 1 exonerating, 1 decoy), so name frequency and individual evidence strength are uninformative. Only the alibi invalidation distinguishes {guilty}."

        return T2Item(
            id=item_id,
            regime="DECOY",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{guilty} is responsible",
            gold_reasoning=gold_reasoning,
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "guilty_suspect": guilty,
                "guilty_position": guilty_idx,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "n_decoys": len(suspects),
                "mechanism": "alibi_chain_invalidation_with_decoys",
                "name_frequencies": {s: sum(1 for e in evidence if s in e["content"]) for s in suspects}
            }
        )

    def generate_conflict_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a CONFLICT regime item.

        All suspects get equal evidence (1 incriminating, 1 exonerating each).
        Two evidence sources disagree about a key fact.
        A precedence rule determines the answer.
        Name frequency and polarity are balanced across ALL suspects.
        """
        rng = self._get_seeded_rng("conflict", template_key, seed)

        # 3 suspects for CONFLICT (balanced like other regimes)
        suspects = rng.sample(SUSPECTS, 3)
        guilty_idx = rng.randint(0, 2)  # Uniform random
        guilty = suspects[guilty_idx]
        innocent = [s for s in suspects if s != guilty]

        location = rng.choice(LOCATIONS)
        company = rng.choice(COMPANIES)

        incident_date = rng.choice(["March 15, 2024", "April 22, 2024", "May 8, 2024"])
        time_window = rng.choice([("8:00 PM", "9:00 PM"), ("6:30 PM", "7:30 PM")])

        if "theft" in template_key:
            item_type = rng.choice(["prototype device", "confidential document set"])
            narrative = f"At {company}, a valuable {item_type} was stolen from {location} on {incident_date}. The investigation has produced conflicting evidence about responsibility. {len(suspects)} employees had access."
            question = "Based on the available evidence and precedence rules, who is most likely responsible for the theft?"
        elif "sabotage" in template_key:
            narrative = f"At {company}'s {location}, critical equipment was sabotaged on {incident_date}. Conflicting evidence has emerged regarding responsibility. {len(suspects)} employees had the technical knowledge and access."
            question = "Based on the available evidence and precedence rules, who is most likely responsible for the sabotage?"
        elif "data_breach" in template_key:
            narrative = f"At {company}, unauthorized data access was detected on {incident_date}. Investigation has produced conflicting evidence pointing to different individuals. {len(suspects)} employees had system access."
            question = "Based on the available evidence and precedence rules, who is most likely responsible for the data breach?"
        else:  # contamination
            narrative = f"At {company}'s {location}, a production batch was contaminated on {incident_date}. Conflicting evidence has emerged about who handled the batch improperly. {len(suspects)} staff members were involved in production."
            question = "Based on the available evidence and precedence rules, who is most likely responsible for the contamination?"

        precedence_rule = "When evidence sources conflict, official system logs and automated records take precedence over witness testimony and observational reports."

        evidence = []
        evidence_id = 1

        witnesses = rng.sample(WITNESSES, 3)
        access_times = rng.sample(["8:15 PM", "8:20 PM", "8:25 PM"], 3)

        # Phase 1: Each suspect gets 1 incriminating evidence (access/presence)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Access logs show {suspect} was present near the scene at {access_times[i]}",
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "medium"
            })
            evidence_id += 1

        # Phase 2: Each suspect gets 1 exonerating evidence (alibi/testimony)
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witnesses[i]} testified they saw {suspect} leaving the building before the incident window",
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "medium"
            })
            evidence_id += 1

        # Phase 3: Conflicting evidence — official record implicates guilty,
        # witness testimony implicates an innocent suspect
        conflict_innocent = rng.choice(innocent)

        # Official system log (high precedence) — points to guilty
        evidence.append({
            "id": f"E{evidence_id:03d}",
            "content": f"Automated security system recorded badge access in the restricted area during the incident window on {incident_date}. The badge belongs to an employee who was also logged entering at {access_times[suspects.index(guilty)]}",
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high"
        })
        evidence_id += 1

        # Witness testimony (low precedence) — points to innocent suspect
        evidence.append({
            "id": f"E{evidence_id:03d}",
            "content": f"An unverified witness report claims that the person seen at the restricted area matched the description of someone who was also logged entering at {access_times[suspects.index(conflict_innocent)]}",
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "high"
        })
        evidence_id += 1

        # Normalize lengths
        self._normalize_evidence_lengths(evidence, rng)

        # Shuffle
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        gold_reasoning = f"The evidence presents a conflict: an official automated security system log points to {guilty}, while an unverified witness report points to {conflict_innocent}. All suspects have equal surface-level evidence (1 incriminating, 1 exonerating each). Applying the stated precedence rule (official records > witness testimony), {guilty} is determined to be responsible. This requires understanding source reliability hierarchies, not counting mentions."

        return T2Item(
            id=item_id,
            regime="CONFLICT",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{guilty} is responsible",
            gold_reasoning=gold_reasoning,
            source_precedence_rule=precedence_rule,
            metadata={
                "template": template_key,
                "guilty_suspect": guilty,
                "guilty_position": guilty_idx,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "mechanism": "source_precedence_conflict",
                "name_frequencies": {s: sum(1 for e in evidence if s in e["content"]) for s in suspects}
            }
        )

    def generate_insufficient_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate an INSUFFICIENT regime item.

        All suspects are perfectly symmetric - no alibi is invalidated.
        Each suspect has identical evidence profiles.
        The correct answer is "Cannot be determined from available evidence".
        """
        rng = self._get_seeded_rng("insufficient", template_key, seed)

        suspects = rng.sample(SUSPECTS, 3)

        location = rng.choice(LOCATIONS)
        company = rng.choice(COMPANIES)

        incident_date = rng.choice(["March 15, 2024", "April 22, 2024", "May 8, 2024"])
        time_window = rng.choice([("8:00 PM", "9:00 PM"), ("6:30 PM", "7:30 PM")])

        if "theft" in template_key:
            item_type = rng.choice(["prototype device", "confidential document set"])
            narrative = f"At {company}, a valuable {item_type} was stolen from {location} on {incident_date} between {time_window[0]} and {time_window[1]}. Investigation has identified {len(suspects)} employees with access, but evidence is inconclusive."
            question = "Based on the available evidence, who is most likely responsible for the theft?"
        elif "sabotage" in template_key:
            narrative = f"At {company}'s {location}, equipment was sabotaged on {incident_date} between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had the technical knowledge and access, but evidence cannot distinguish between them."
            question = "Based on the available evidence, who is most likely responsible for the sabotage?"
        elif "data_breach" in template_key:
            narrative = f"At {company}, unauthorized data access occurred on {incident_date} between {time_window[0]} and {time_window[1]}. {len(suspects)} employees had sufficient privileges, but available evidence is insufficient to determine responsibility."
            question = "Based on the available evidence, who is most likely responsible for the data breach?"
        else:  # contamination
            narrative = f"At {company}'s {location}, a batch was contaminated on {incident_date} between {time_window[0]} and {time_window[1]}. {len(suspects)} staff members had access to the production line, but evidence is ambiguous."
            question = "Based on the available evidence, who is most likely responsible for the contamination?"

        evidence = []
        evidence_id = 1

        # Each suspect gets symmetric evidence
        alibi_witnesses = rng.sample(WITNESSES, 3)
        alibi_locations = rng.sample([loc for loc in LOCATIONS if loc != location], 3)
        access_times = ["8:15 PM", "8:20 PM", "8:25 PM"]

        # Phase 1: Each suspect has access evidence
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Access logs show {suspect} was present near the scene at {access_times[i]}",
                "supports": [suspect],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        # Phase 2: Each suspect has an alibi that CANNOT be verified or invalidated
        for i, suspect in enumerate(suspects):
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{suspect} claims to have been at {alibi_locations[i]}, but {alibi_witnesses[i]} cannot confirm or deny this",
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none"
            })
            evidence_id += 1

        # Phase 3: General ambiguity statement (no names)
        evidence.append({
            "id": f"E{evidence_id:03d}",
            "content": f"Critical security footage from the incident window is corrupted and cannot be recovered for analysis",
            "supports": [],
            "contradicts": [],
            "diagnostic_value": "none"
        })
        evidence_id += 1

        # Normalize lengths
        self._normalize_evidence_lengths(evidence, rng)

        # Shuffle
        rng.shuffle(evidence)
        for i, ev in enumerate(evidence, 1):
            ev["id"] = f"E{i:03d}"

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]
        hypotheses.append("Cannot be determined from available evidence")

        gold_reasoning = "The evidence is perfectly symmetric across all suspects. Each suspect has identical evidence profiles: access to the scene, unverifiable alibis, and no distinguishing factors. No alibi is invalidated, no precedence rule applies, and no relational inconsistency exists. A definitive determination cannot be made. This is genuinely insufficient evidence, not solvable through reasoning."

        return T2Item(
            id=item_id,
            regime="INSUFFICIENT",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer="Cannot be determined from available evidence",
            gold_reasoning=gold_reasoning,
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "mechanism": "perfect_symmetry",
                "name_frequencies": {s: sum(1 for e in evidence if s in e["content"]) for s in suspects}
            }
        )

    def generate_counterfactual_pair(self, base_item: T2Item, seed: int) -> T2Item:
        """
        Generate a counterfactual twin of a CLEAN/DECOY item.

        Uses the EXACT same suspect names and evidence structure,
        but invalidates a DIFFERENT suspect's alibi.
        This creates a minimal pair where token inventory is nearly identical
        but the correct answer flips.

        Handles both alibi variants (witness-based) and timeline variants
        (location-based) by detecting the cross-reference pattern.
        """
        import re

        if base_item.regime not in ["CLEAN", "DECOY"]:
            raise ValueError("Counterfactual pairs only supported for CLEAN/DECOY regimes")

        rng = random.Random(seed)

        # Extract suspects from base item
        suspects = [h.replace(" is responsible", "") for h in base_item.hypotheses]
        base_guilty = base_item.metadata["guilty_suspect"]

        # Choose a different guilty suspect
        other_suspects = [s for s in suspects if s != base_guilty]
        new_guilty = rng.choice(other_suspects)
        new_guilty_idx = suspects.index(new_guilty)

        template_key = base_item.metadata.get("template", "")
        is_alibi_variant = "alibi" in template_key

        # Parse evidence to find alibi witnesses/locations per suspect
        alibi_pattern = re.compile(
            r"(.+?) claims to have been at (.+?) during the incident, confirmed by (.+?)$"
        )
        timeline_pattern = re.compile(
            r"(.+?) reported being at (.+?) at the time, verified by (.+?)$"
        )
        witness_invalidation_pattern = re.compile(
            r"(.+?) was actually on approved leave"
        )
        location_invalidation_pattern = re.compile(
            r"(.+?)'s security logs show no entries"
        )

        alibi_info = {}  # suspect -> {"witness": ..., "location": ...}
        for ev in base_item.evidence:
            content = ev["content"]
            for pattern in [alibi_pattern, timeline_pattern]:
                match = pattern.search(content)
                if match:
                    suspect_name = match.group(1).strip()
                    location_name = match.group(2).strip()
                    witness_name = match.group(3).strip()
                    alibi_info[suspect_name] = {
                        "witness": witness_name,
                        "location": location_name
                    }
                    break

        # Find the cross-reference invalidation evidence
        cross_ref_idx = None
        old_invalidation_target = None  # the witness or location being invalidated
        for i, ev in enumerate(base_item.evidence):
            content = ev["content"]
            match_w = witness_invalidation_pattern.search(content)
            match_l = location_invalidation_pattern.search(content)
            if match_w:
                cross_ref_idx = i
                old_invalidation_target = match_w.group(1).strip()
                break
            elif match_l:
                cross_ref_idx = i
                old_invalidation_target = match_l.group(1).strip()
                break

        if cross_ref_idx is None or new_guilty not in alibi_info:
            # Fallback: create new item from scratch with same template
            return self.generate_clean_item(
                base_item.metadata["template"],
                seed,
                f"{base_item.id}_counterfactual"
            )

        # Determine what to replace in the cross-reference
        new_alibi = alibi_info[new_guilty]
        new_target = new_alibi["witness"] if is_alibi_variant else new_alibi["location"]

        # Create new evidence list with modified cross-reference
        new_evidence = []
        for i, ev in enumerate(base_item.evidence):
            if i == cross_ref_idx:
                new_content = ev["content"].replace(old_invalidation_target, new_target)
                new_evidence.append({
                    **ev,
                    "content": new_content
                })
            else:
                new_evidence.append(ev.copy())

        new_metadata = base_item.metadata.copy()
        new_metadata["guilty_suspect"] = new_guilty
        new_metadata["guilty_position"] = new_guilty_idx
        new_metadata["counterfactual_of"] = base_item.id
        new_metadata["name_frequencies"] = {s: sum(1 for e in new_evidence if s in e["content"]) for s in suspects}

        gold_reasoning = f"This is a counterfactual twin of another item. The evidence requires relational reasoning: {new_guilty}'s alibi is invalidated by the cross-reference. The innocent suspects' alibis remain valid. Token inventory is nearly identical to the base item, but the correct answer has flipped."

        return T2Item(
            id=f"{base_item.id}_counterfactual",
            regime=base_item.regime,
            narrative=base_item.narrative,
            question=base_item.question,
            hypotheses=base_item.hypotheses,
            evidence=new_evidence,
            gold_answer=f"{new_guilty} is responsible",
            gold_reasoning=gold_reasoning,
            source_precedence_rule=base_item.source_precedence_rule,
            metadata=new_metadata
        )

    def generate_dataset(self, n_per_regime: int = 8, seed: int = 42) -> List[T2Item]:
        """
        Generate a balanced dataset across all regimes and template families.

        Template families (8 total):
        - theft_alibi, theft_timeline
        - sabotage_alibi, sabotage_timeline
        - data_breach_alibi, data_breach_timeline
        - contamination_alibi, contamination_timeline

        Args:
            n_per_regime: Number of items per regime (distributed across templates)
            seed: Master seed for generation

        Returns:
            List of T2Item objects
        """
        self.master_seed = seed
        self.rng = random.Random(seed)

        templates = [
            "theft_alibi", "theft_timeline",
            "sabotage_alibi", "sabotage_timeline",
            "data_breach_alibi", "data_breach_timeline",
            "contamination_alibi", "contamination_timeline"
        ]

        regimes = ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]

        items = []
        item_counter = 0

        for regime in regimes:
            items_per_template = n_per_regime // len(templates)
            remainder = n_per_regime % len(templates)

            for template_idx, template in enumerate(templates):
                count = items_per_template + (1 if template_idx < remainder else 0)

                for i in range(count):
                    item_id = f"t2v2_{regime.lower()}_{template}_{item_counter:04d}"
                    item_seed = self.rng.randint(0, 1000000)

                    if regime == "CLEAN":
                        item = self.generate_clean_item(template, item_seed, item_id)
                    elif regime == "DECOY":
                        item = self.generate_decoy_item(template, item_seed, item_id)
                    elif regime == "CONFLICT":
                        item = self.generate_conflict_item(template, item_seed, item_id)
                    else:  # INSUFFICIENT
                        item = self.generate_insufficient_item(template, item_seed, item_id)

                    items.append(item)
                    item_counter += 1

        return items

    def generate_leakage_eval_corpus(self, n_per_regime: int = 50, seed: int = 42) -> List[T2Item]:
        """
        Generate a large balanced corpus for evaluating bag-of-words leakage.

        This corpus can be split into train/test sets to verify that:
        1. Name frequency heuristics achieve ~33% accuracy (chance level)
        2. Bag-of-words classifiers cannot exceed ~40% accuracy
        3. Only relational reasoning achieves high performance

        Args:
            n_per_regime: Number of items per regime (default 50)
            seed: Master seed

        Returns:
            Large list of T2Item objects with counterfactual pairs
        """
        base_items = self.generate_dataset(n_per_regime=n_per_regime, seed=seed)

        all_items = []
        for item in base_items:
            all_items.append(item)

            # Generate counterfactual twin for CLEAN/DECOY items
            if item.regime in ["CLEAN", "DECOY"]:
                try:
                    twin = self.generate_counterfactual_pair(item, seed=hash(item.id) % 1000000)
                    all_items.append(twin)
                except Exception:
                    # Skip if counterfactual generation fails
                    pass

        return all_items


def export_jsonl(items: List[T2Item], path: str):
    """Export items to JSONL format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')

    print(f"Exported {len(items)} items to {path}")


def analyze_leakage(items: List[T2Item]) -> Dict[str, Any]:
    """
    Analyze the dataset for potential leakage signals.

    Returns statistics on:
    - Name frequency distributions
    - Evidence count per suspect
    - Evidence length variance
    - Guilty suspect position distribution
    """
    stats = {
        "n_items": len(items),
        "regime_distribution": {},
        "guilty_position_distribution": {},
        "name_frequency_variance": [],
        "evidence_length_variance": [],
        "evidence_count_per_suspect": []
    }

    for item in items:
        # Regime distribution
        regime = item.regime
        stats["regime_distribution"][regime] = stats["regime_distribution"].get(regime, 0) + 1

        # Guilty position
        if "guilty_position" in item.metadata:
            pos = item.metadata["guilty_position"]
            stats["guilty_position_distribution"][pos] = stats["guilty_position_distribution"].get(pos, 0) + 1

        # Name frequency variance
        if "name_frequencies" in item.metadata:
            freqs = list(item.metadata["name_frequencies"].values())
            if len(freqs) > 1:
                mean_freq = sum(freqs) / len(freqs)
                variance = sum((f - mean_freq) ** 2 for f in freqs) / len(freqs)
                stats["name_frequency_variance"].append(variance)

        # Evidence length variance
        if item.evidence:
            lengths = [len(e["content"]) for e in item.evidence]
            mean_length = sum(lengths) / len(lengths)
            variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
            stats["evidence_length_variance"].append(variance)

    # Compute averages
    if stats["name_frequency_variance"]:
        stats["avg_name_frequency_variance"] = sum(stats["name_frequency_variance"]) / len(stats["name_frequency_variance"])

    if stats["evidence_length_variance"]:
        stats["avg_evidence_length_variance"] = sum(stats["evidence_length_variance"]) / len(stats["evidence_length_variance"])

    return stats


if __name__ == "__main__":
    print(f"T2 Generator Version {__version__}")
    print("=" * 60)

    # Initialize generator
    generator = T2Generator(seed=42)

    # Generate single examples from each regime
    print("\n### Generating single examples from each regime ###\n")

    clean_item = generator.generate_clean_item("theft_alibi", seed=1, item_id="t2v2_clean_theft_alibi_0001")
    print(f"CLEAN item generated: {clean_item.id}")
    print(f"  Guilty: {clean_item.metadata['guilty_suspect']}")
    print(f"  Name frequencies: {clean_item.metadata['name_frequencies']}")
    print(f"  Evidence count: {len(clean_item.evidence)}")

    decoy_item = generator.generate_decoy_item("sabotage_timeline", seed=2, item_id="t2v2_decoy_sabotage_timeline_0001")
    print(f"\nDECOY item generated: {decoy_item.id}")
    print(f"  Guilty: {decoy_item.metadata['guilty_suspect']}")
    print(f"  Name frequencies: {decoy_item.metadata['name_frequencies']}")
    print(f"  Evidence count: {len(decoy_item.evidence)}")

    conflict_item = generator.generate_conflict_item("data_breach_alibi", seed=3, item_id="t2v2_conflict_breach_alibi_0001")
    print(f"\nCONFLICT item generated: {conflict_item.id}")
    print(f"  Guilty: {conflict_item.metadata['guilty_suspect']}")
    print(f"  Name frequencies: {conflict_item.metadata['name_frequencies']}")
    print(f"  Precedence rule: {conflict_item.source_precedence_rule[:80]}...")

    insufficient_item = generator.generate_insufficient_item("contamination_alibi", seed=4, item_id="t2v2_insufficient_contam_alibi_0001")
    print(f"\nINSUFFICIENT item generated: {insufficient_item.id}")
    print(f"  Gold answer: {insufficient_item.gold_answer}")
    print(f"  Name frequencies: {insufficient_item.metadata['name_frequencies']}")

    # Generate counterfactual pair
    print("\n### Generating counterfactual pair ###\n")
    twin_item = generator.generate_counterfactual_pair(clean_item, seed=100)
    print(f"Counterfactual twin: {twin_item.id}")
    print(f"  Original guilty: {clean_item.metadata['guilty_suspect']}")
    print(f"  Twin guilty: {twin_item.metadata['guilty_suspect']}")
    print(f"  Name frequencies: {twin_item.metadata['name_frequencies']}")

    # Generate full dataset
    print("\n### Generating full dataset ###\n")
    dataset = generator.generate_dataset(n_per_regime=8, seed=42)
    print(f"Generated {len(dataset)} items")
    print(f"Regimes: {set(item.regime for item in dataset)}")
    print(f"Templates: {set(item.metadata.get('template') for item in dataset)}")

    # Analyze for leakage
    print("\n### Leakage Analysis ###\n")
    stats = analyze_leakage(dataset)
    print(f"Total items: {stats['n_items']}")
    print(f"Regime distribution: {stats['regime_distribution']}")
    print(f"Guilty position distribution: {stats['guilty_position_distribution']}")
    print(f"Avg name frequency variance: {stats.get('avg_name_frequency_variance', 'N/A'):.4f}")
    print(f"Avg evidence length variance: {stats.get('avg_evidence_length_variance', 'N/A'):.2f}")

    # Generate large leakage evaluation corpus
    print("\n### Generating large leakage evaluation corpus ###\n")
    leakage_corpus = generator.generate_leakage_eval_corpus(n_per_regime=20, seed=42)
    print(f"Generated {len(leakage_corpus)} items (includes counterfactual pairs)")

    # Export examples
    print("\n### Exporting examples ###\n")
    export_jsonl([clean_item, decoy_item, conflict_item, insufficient_item, twin_item],
                 "/home/user/work/output_files/t2v2_examples.jsonl")
    export_jsonl(dataset, "/home/user/work/output_files/t2v2_dataset.jsonl")
    export_jsonl(leakage_corpus, "/home/user/work/output_files/t2v2_leakage_corpus.jsonl")

    print("\n" + "=" * 60)
    print("T2 v2 Generator: Complete")
    print("=" * 60)
