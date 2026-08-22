"""
T2 Deterministic Diagnostic Generator for ACH Scaffolding Evaluation

Generates test items across four diagnostic regimes:
- CLEAN: Evidence uniquely identifies one hypothesis
- DECOY: Clean evidence + salient-but-non-diagnostic decoys
- CONFLICT: Sources genuinely disagree, precedence rule determines answer
- INSUFFICIENT: Evidence is genuinely ambiguous
"""

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


# Entity pools
SUSPECTS = [
    "Alex Chen", "Blake Rivera", "Casey Morgan", "Dana Patel",
    "Ellis Kim", "Finley Brooks", "Grey Martinez", "Harper Quinn",
    "Indigo Taylor", "Jordan Lee"
]

LOCATIONS = [
    "Building A - East Wing", "Building B - Research Lab", "Building C - Server Room",
    "Building D - Storage Facility", "Main Office - 3rd Floor", "Warehouse District",
    "Downtown Branch Office", "Research Campus - North Wing"
]

WITNESSES = [
    "Security Officer Sarah Johnson", "Custodian Mike Thompson",
    "IT Administrator Pat Williams", "Reception Staff Chris Anderson",
    "Maintenance Lead Sam Rodriguez", "Lab Supervisor Dr. Taylor Kim"
]

COMPANIES = [
    "TechCorp Industries", "DataVault Systems", "BioResearch Labs",
    "Manufacturing Solutions Inc", "Global Logistics Co"
]


# Scenario templates
SCENARIO_TEMPLATES = {
    "theft": {
        "setting": "At {company}, a valuable {item} was stolen from {location} on {date} between {time_start} and {time_end}. The security investigation has narrowed down to {n_suspects} employees who had access during that timeframe.",
        "question": "Based on the available evidence, who is most likely responsible for the theft?",
        "item_types": ["prototype device", "confidential research document", "server access key", "data backup drive"],
        "evidence_patterns": {
            "access_log": "{suspect} badge access recorded at {location} at {time}",
            "keycard": "{suspect}'s keycard was used to unlock {specific_area} at {time}",
            "witness": "{witness} reported seeing {suspect} near {location} at {time}",
            "alibi": "{suspect} was confirmed to be in {other_location} during the incident, verified by {verification}",
            "fingerprints": "Forensic analysis found {suspect}'s fingerprints on {item}",
            "camera": "Security camera footage shows {suspect} entering {location} at {time}",
            "motive": "{suspect} recently {motive_detail}",
            "item_found": "The stolen {item} was found in {suspect}'s workspace during the investigation"
        }
    },
    "sabotage": {
        "setting": "At {company}'s {location}, critical equipment was sabotaged on {date}, causing a production shutdown. Internal investigation identified {n_suspects} employees with the technical knowledge and access to perform the sabotage.",
        "question": "Based on the available evidence, who is most likely responsible for the sabotage?",
        "evidence_patterns": {
            "technical_logs": "System logs show {suspect}'s credentials were used to modify {system} settings at {time}",
            "witness": "{witness} observed {suspect} working on {equipment} at {time}",
            "alibi": "{suspect} was attending {event} during the sabotage window, confirmed by {verification}",
            "expertise": "{suspect} has specialized training in {technical_area}, required to perform this type of sabotage",
            "tools": "Specialized tools registered to {suspect} were found near the sabotaged equipment",
            "camera": "Security footage captured {suspect} near {location} at {time}",
            "motive": "{suspect} had recently {motive_detail}",
            "confession": "{suspect} admitted to {related_action} but denied involvement in the sabotage"
        }
    },
    "data_breach": {
        "setting": "At {company}, unauthorized access to sensitive customer data was detected on {date}. The breach occurred between {time_start} and {time_end}. Security analysis identified {n_suspects} employees with sufficient system privileges to access the compromised database.",
        "question": "Based on the available evidence, who is most likely responsible for the data breach?",
        "evidence_patterns": {
            "network_logs": "Network logs show data exfiltration from IP address assigned to {suspect}'s workstation at {time}",
            "database_access": "Database audit logs record {suspect}'s credentials accessing sensitive tables at {time}",
            "witness": "{witness} noticed {suspect} working unusually late in {location} on {date}",
            "alibi": "{suspect}'s laptop was powered off from {time_start} to {time_end}, verified by endpoint monitoring",
            "encryption": "Encrypted file transfer used {suspect}'s private key for authentication",
            "external_contact": "{suspect} exchanged emails with external contact at {company_name} shortly before the breach",
            "motive": "{suspect} had {motive_detail}",
            "usb_device": "Unauthorized USB device was detected on {suspect}'s computer at {time}"
        }
    },
    "contamination": {
        "setting": "At {company}'s {location}, a batch of products was contaminated on {date}, requiring a costly recall. Quality control investigation narrowed the incident to {n_suspects} staff members who handled the batch during production.",
        "question": "Based on the available evidence, who is most likely responsible for the contamination?",
        "evidence_patterns": {
            "production_log": "Production records show {suspect} was assigned to the contaminated batch during shift {shift}",
            "witness": "{witness} saw {suspect} handling materials near the contaminated production line at {time}",
            "alibi": "{suspect} was on approved leave during the contamination window, verified by HR records",
            "training": "{suspect} recently completed {training_type}, and was specifically trained on contamination prevention",
            "equipment": "Equipment assigned to {suspect} tested positive for the contaminant substance",
            "procedure": "{suspect} failed to follow {procedure_name} according to supervisor notes from {date}",
            "access": "{suspect} had access to {material} that contains the identified contaminant",
            "prior_incident": "{suspect} was involved in a similar contamination incident {time_period} ago"
        }
    }
}


class T2Generator:
    """Generator for T2 diagnostic test items."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed."""
        self.master_seed = seed
        self.rng = random.Random(seed)

    def _get_seeded_rng(self, *args) -> random.Random:
        """Create a seeded RNG from master seed and additional args."""
        seed_str = f"{self.master_seed}_{'_'.join(map(str, args))}"
        seed_val = hash(seed_str) % (2**32)
        return random.Random(seed_val)

    def _select_entities(self, rng: random.Random, n_suspects: int = 4):
        """Select suspects, locations, and other entities."""
        suspects = rng.sample(SUSPECTS, n_suspects)
        location = rng.choice(LOCATIONS)
        witness = rng.choice(WITNESSES)
        company = rng.choice(COMPANIES)
        return suspects, location, witness, company

    def _format_narrative(self, template_key: str, rng: random.Random, suspects: List[str]):
        """Format the narrative setting."""
        template = SCENARIO_TEMPLATES[template_key]
        company = rng.choice(COMPANIES)
        location = rng.choice(LOCATIONS)

        dates = ["March 15, 2024", "April 22, 2024", "May 8, 2024", "June 12, 2024"]
        times = [("8:00 PM", "11:00 PM"), ("6:30 PM", "9:30 PM"), ("7:00 PM", "10:00 PM")]

        setting = template["setting"].format(
            company=company,
            location=location,
            date=rng.choice(dates),
            time_start=rng.choice(times)[0],
            time_end=rng.choice(times)[1],
            n_suspects=len(suspects),
            item=rng.choice(template.get("item_types", ["item"]))
        )

        return setting, template["question"], company, location

    def generate_clean_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a CLEAN regime item.
        Evidence uniquely identifies one hypothesis. Multiple evidence items all point to guilty suspect.
        Other suspects have alibis.
        """
        rng = self._get_seeded_rng("clean", template_key, seed)

        suspects, location, witness, company = self._select_entities(rng, n_suspects=4)
        guilty = suspects[0]
        innocent = suspects[1:]

        narrative, question, company, location = self._format_narrative(template_key, rng, suspects)

        template = SCENARIO_TEMPLATES[template_key]
        patterns = template["evidence_patterns"]

        # Build evidence list
        evidence = []
        evidence_id = 1

        # Multiple strong diagnostic evidence pointing to guilty suspect
        times = ["8:15 PM", "8:45 PM", "9:20 PM"]

        if template_key == "theft":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Security camera footage shows {guilty} entering {location} at {times[0]} carrying an empty bag and exiting at {times[1]} with a bulging bag.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{guilty}'s keycard was used to unlock the secure storage area within {location} at {times[0]}, which contained the stolen prototype device.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"The stolen prototype device was found hidden in {guilty}'s personal locker during the investigation, wrapped in {guilty}'s distinctive jacket.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "sabotage":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"System logs show {guilty}'s credentials were used to modify critical safety settings on the equipment at {times[0]}, immediately before the malfunction occurred.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} observed {guilty} working alone on the equipment at {times[0]} and noted that {guilty} appeared nervous when questioned.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Specialized calibration tools registered to {guilty} were found at the scene with fresh fingerprints matching {guilty}.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "data_breach":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Network logs show 15 GB of customer data was exfiltrated from the IP address assigned to {guilty}'s workstation at {times[0]}.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Database audit logs record {guilty}'s credentials accessing sensitive customer tables at {times[0]}, with queries specifically extracting personal information.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"An unauthorized encrypted USB device was detected on {guilty}'s computer at {times[0]}, and the device was found in {guilty}'s desk drawer.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "contamination":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Production records show {guilty} was the sole operator assigned to the contaminated batch during the night shift when contamination occurred.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Equipment assigned to {guilty} tested positive for the specific contaminant substance, and {guilty}'s gloves also showed traces of the same substance.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} documented that {guilty} failed to follow the required sterilization protocol during the shift, which would have prevented the contamination.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        # Add alibis for innocent suspects
        for suspect in innocent:
            other_location = rng.choice([loc for loc in LOCATIONS if loc != location])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{suspect} was confirmed to be attending a company training session in {other_location} during the incident, verified by sign-in records and instructor confirmation.",
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        return T2Item(
            id=item_id,
            regime="CLEAN",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{guilty} is responsible",
            gold_reasoning=f"Multiple independent pieces of high-quality diagnostic evidence (access logs, physical evidence, witness testimony, and forensic analysis) all consistently point to {guilty}. All other suspects have verified alibis placing them elsewhere during the incident.",
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "guilty_suspect": guilty,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence)
            }
        )

    def generate_decoy_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a DECOY regime item.
        Same as CLEAN but adds salient-but-non-diagnostic decoy evidence pointing at innocent suspects.
        """
        rng = self._get_seeded_rng("decoy", template_key, seed)

        suspects, location, witness, company = self._select_entities(rng, n_suspects=4)
        guilty = suspects[0]
        innocent = suspects[1:]

        narrative, question, company, location = self._format_narrative(template_key, rng, suspects)

        template = SCENARIO_TEMPLATES[template_key]

        evidence = []
        evidence_id = 1
        times = ["8:15 PM", "8:45 PM", "9:20 PM"]

        # Strong diagnostic evidence pointing to guilty (same as CLEAN)
        if template_key == "theft":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Security camera footage shows {guilty} entering {location} at {times[0]} carrying an empty bag and exiting at {times[1]} with a bulging bag.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"The stolen prototype device was found hidden in {guilty}'s personal locker during the investigation, wrapped in {guilty}'s distinctive jacket.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # DECOY: Salient but non-diagnostic evidence for innocent suspects
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[0]} had recently expressed frustration about not being assigned to work on the prototype project, stating 'I should have been chosen for that team.'",
                "supports": [innocent[0]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[1]} was seen browsing competitor company websites during work hours the week before the theft.",
                "supports": [innocent[1]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        elif template_key == "sabotage":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"System logs show {guilty}'s credentials were used to modify critical safety settings on the equipment at {times[0]}, immediately before the malfunction occurred.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Specialized calibration tools registered to {guilty} were found at the scene with fresh fingerprints matching {guilty}.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # DECOY evidence
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[0]} was overheard complaining about management decisions regarding equipment maintenance schedules two weeks before the incident.",
                "supports": [innocent[0]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[1]} has extensive technical knowledge of the sabotaged equipment type and previously worked as a systems engineer at a competitor.",
                "supports": [innocent[1]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        elif template_key == "data_breach":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Network logs show 15 GB of customer data was exfiltrated from the IP address assigned to {guilty}'s workstation at {times[0]}.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"An unauthorized encrypted USB device was detected on {guilty}'s computer at {times[0]}, and the device was found in {guilty}'s desk drawer.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # DECOY evidence
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[0]} recently updated their LinkedIn profile and has been in contact with recruiters from competing firms.",
                "supports": [innocent[0]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[1]} was observed working unusually late hours in the week preceding the breach, though no specific suspicious activity was noted.",
                "supports": [innocent[1]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        elif template_key == "contamination":
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Production records show {guilty} was the sole operator assigned to the contaminated batch during the night shift when contamination occurred.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Equipment assigned to {guilty} tested positive for the specific contaminant substance, and {guilty}'s gloves also showed traces of the same substance.",
                "supports": [guilty],
                "contradicts": [],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # DECOY evidence
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[0]} had received a written warning three months ago for a minor quality control violation in an unrelated production area.",
                "supports": [innocent[0]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{innocent[1]} was seen entering the facility early on the day of contamination, though their shift didn't start until later.",
                "supports": [innocent[1]],
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        # Add alibis for innocent suspects (making the decoys clearly non-diagnostic)
        for suspect in innocent:
            other_location = rng.choice([loc for loc in LOCATIONS if loc != location])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{suspect} was confirmed to be in {other_location} during the critical incident window, verified by timestamped security badge records.",
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        return T2Item(
            id=item_id,
            regime="DECOY",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{guilty} is responsible",
            gold_reasoning=f"Despite salient decoy evidence suggesting motive or opportunity for other suspects, high-quality diagnostic evidence (access logs, physical evidence) definitively points to {guilty}. The decoy evidence for other suspects is circumstantial and non-diagnostic, and all innocent suspects have verified alibis.",
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "guilty_suspect": guilty,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "n_decoys": len(innocent) * 2
            }
        )

    def generate_conflict_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate a CONFLICT regime item.
        Sources genuinely disagree. A source-precedence rule determines the gold answer.
        """
        rng = self._get_seeded_rng("conflict", template_key, seed)

        suspects, location, witness, company = self._select_entities(rng, n_suspects=4)

        # Two suspects will have conflicting evidence
        suspect_a = suspects[0]
        suspect_b = suspects[1]
        others = suspects[2:]

        narrative, question, company, location = self._format_narrative(template_key, rng, suspects)

        template = SCENARIO_TEMPLATES[template_key]

        evidence = []
        evidence_id = 1
        times = ["8:15 PM", "8:45 PM", "9:20 PM"]

        # Source precedence rule: Official records > Witness testimony
        precedence_rule = "When evidence conflicts, official system logs and forensic records take precedence over witness testimony and circumstantial evidence."

        if template_key == "theft":
            # Official records point to suspect A
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Security system logs show {suspect_a}'s keycard was used to access {location} at {times[0]}, the exact time when the theft occurred.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Digital access logs indicate {suspect_a}'s credentials unlocked the secure storage containing the stolen item at {times[0]}.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # Witness testimony points to suspect B
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} testified that they clearly saw {suspect_b} leaving {location} at {times[1]} carrying a bag matching the description of the stolen item's container.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            witness2 = rng.choice([w for w in WITNESSES if w != witness])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness2} provided a statement saying they observed {suspect_b} acting suspiciously near {location} around {times[0]} and specifically remembers {suspect_b} looking around nervously.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "sabotage":
            # Official records point to suspect A
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"System audit logs show {suspect_a}'s network credentials were used to modify the equipment control settings at {times[0]}, causing the malfunction.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Automated monitoring systems recorded login from {suspect_a}'s account executing unauthorized configuration changes at {times[0]}.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # Witness testimony points to suspect B
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} testified that they saw {suspect_b} physically working on the equipment at {times[0]} with tools, and no one else was present in the area.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            witness2 = rng.choice([w for w in WITNESSES if w != witness])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness2} reported that {suspect_b} was the only person with the technical expertise present during the sabotage window and specifically saw {suspect_b} accessing the equipment control panel.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "data_breach":
            # Official records point to suspect A
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Network forensic logs show data exfiltration originated from IP address 192.168.1.42, which is assigned to {suspect_a}'s workstation.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Database access logs record {suspect_a}'s credentials querying and extracting sensitive customer records at {times[0]}.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # Witness testimony points to suspect B
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} stated they saw {suspect_b} working alone in the server room at {times[0]} with a laptop and USB drive, which is highly unusual and against protocol.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            witness2 = rng.choice([w for w in WITNESSES if w != witness])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness2} testified that {suspect_b} asked them detailed questions about database security protocols the week before the breach and seemed overly interested in access logs.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        elif template_key == "contamination":
            # Official records point to suspect A
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Production tracking system shows {suspect_a} was the logged operator for the contaminated batch, with their employee ID recorded at each production checkpoint.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Automated quality control logs indicate {suspect_a}'s workstation ID processed the contaminated batch during the critical time window.",
                "supports": [suspect_a],
                "contradicts": [suspect_b],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            # Witness testimony points to suspect B
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness} provided detailed testimony that they personally observed {suspect_b} handling the batch without proper protective equipment and skipping sterilization steps.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

            witness2 = rng.choice([w for w in WITNESSES if w != witness])
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{witness2} stated that {suspect_b} was the only person they saw working on that production line during the shift, despite what the system logs indicate.",
                "supports": [suspect_b],
                "contradicts": [suspect_a],
                "diagnostic_value": "high"
            })
            evidence_id += 1

        # Add neutral evidence for others
        for suspect in others:
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"{suspect} was working in a different department during the incident and has no apparent connection to the case.",
                "supports": [],
                "contradicts": [suspect],
                "diagnostic_value": "medium"
            })
            evidence_id += 1

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]

        # Gold answer determined by precedence rule (official records > witness testimony)
        gold_suspect = suspect_a

        return T2Item(
            id=item_id,
            regime="CONFLICT",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer=f"{gold_suspect} is responsible",
            gold_reasoning=f"The evidence presents a genuine conflict: official system logs and forensic records point to {suspect_a}, while witness testimony points to {suspect_b}. Applying the source precedence rule (official records take precedence over witness testimony), {suspect_a} is determined to be responsible. The official digital records are considered more reliable than human observation, which can be subject to error or misidentification.",
            source_precedence_rule=precedence_rule,
            metadata={
                "template": template_key,
                "suspect_a": suspect_a,
                "suspect_b": suspect_b,
                "conflict_type": "official_records_vs_witness",
                "n_suspects": len(suspects),
                "n_evidence": len(evidence)
            }
        )

    def generate_insufficient_item(self, template_key: str, seed: int, item_id: str) -> T2Item:
        """
        Generate an INSUFFICIENT regime item.
        Evidence is genuinely ambiguous. All suspects have roughly equal evidence.
        """
        rng = self._get_seeded_rng("insufficient", template_key, seed)

        suspects, location, witness, company = self._select_entities(rng, n_suspects=4)

        narrative, question, company, location = self._format_narrative(template_key, rng, suspects)

        template = SCENARIO_TEMPLATES[template_key]

        evidence = []
        evidence_id = 1
        times = ["8:15 PM", "8:30 PM", "8:45 PM", "9:00 PM"]

        if template_key == "theft":
            # Each suspect has some evidence but nothing definitive
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Badge access logs show all four suspects ({', '.join(suspects)}) accessed {location} at various times during the incident window between {times[0]} and {times[-1]}.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            for i, suspect in enumerate(suspects):
                evidence.append({
                    "id": f"E{evidence_id:03d}",
                    "content": f"{suspect} was observed near {location} around {times[i]}, which falls within the timeframe when the theft could have occurred.",
                    "supports": [suspect],
                    "contradicts": [],
                    "diagnostic_value": "low"
                })
                evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Security camera footage from the critical period is corrupted and does not show who removed the item from the secure area.",
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Forensic analysis found multiple sets of fingerprints on the storage cabinet, including partial prints from all four suspects, which is expected given they all work in this area regularly.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        elif template_key == "sabotage":
            # Each suspect has equal capability and access
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"All four suspects ({', '.join(suspects)}) have the required technical expertise and training to perform this type of sabotage.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"System logs indicate that all four suspects accessed the equipment control system on the day of the incident, which is routine for their roles.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            for i, suspect in enumerate(suspects):
                evidence.append({
                    "id": f"E{evidence_id:03d}",
                    "content": f"{suspect} was working in the area around {times[i]} and had unsupervised access to the equipment.",
                    "supports": [suspect],
                    "contradicts": [],
                    "diagnostic_value": "low"
                })
                evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"The sabotage method used is a standard technique that all four suspects learned in their technical training program.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

        elif template_key == "data_breach":
            # All suspects had access, no definitive logs
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Network analysis shows the breach originated from the shared network segment used by all four suspects' workstations, but the specific machine cannot be identified.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Database access logs show all four suspects ({', '.join(suspects)}) accessed the compromised database within the relevant time window as part of their normal duties.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            for i, suspect in enumerate(suspects):
                evidence.append({
                    "id": f"E{evidence_id:03d}",
                    "content": f"{suspect} has sufficient database privileges to extract the stolen data and was logged in during the breach window around {times[i]}.",
                    "supports": [suspect],
                    "contradicts": [],
                    "diagnostic_value": "low"
                })
                evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Detailed forensic logs that could identify the responsible party were not enabled on the database server at the time of the breach.",
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none"
            })
            evidence_id += 1

        elif template_key == "contamination":
            # All suspects handled the batch
            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Production records show the contaminated batch was processed through multiple stations, with all four suspects ({', '.join(suspects)}) handling it at different stages.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            for i, suspect in enumerate(suspects):
                evidence.append({
                    "id": f"E{evidence_id:03d}",
                    "content": f"{suspect} worked on the batch during their assigned shift around {times[i]} and had access to materials that could cause the observed contamination.",
                    "supports": [suspect],
                    "contradicts": [],
                    "diagnostic_value": "low"
                })
                evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"Testing of equipment and work areas found trace amounts of the contaminant at multiple stations where all four suspects worked.",
                "supports": suspects,
                "contradicts": [],
                "diagnostic_value": "low"
            })
            evidence_id += 1

            evidence.append({
                "id": f"E{evidence_id:03d}",
                "content": f"The contamination could have occurred at any of the four processing stages, and there is no physical evidence to determine which stage was the source.",
                "supports": [],
                "contradicts": [],
                "diagnostic_value": "none"
            })
            evidence_id += 1

        hypotheses = [f"{suspect} is responsible" for suspect in suspects]
        hypotheses.append("Cannot be determined from available evidence")

        return T2Item(
            id=item_id,
            regime="INSUFFICIENT",
            narrative=narrative,
            question=question,
            hypotheses=hypotheses,
            evidence=evidence,
            gold_answer="Cannot be determined from available evidence",
            gold_reasoning="The evidence is genuinely ambiguous and insufficient to identify a specific responsible party. All suspects had equal access, opportunity, and capability. No diagnostic evidence uniquely implicates any individual suspect, and critical forensic data is either unavailable or non-discriminating. A definitive determination cannot be made without additional evidence.",
            source_precedence_rule=None,
            metadata={
                "template": template_key,
                "n_suspects": len(suspects),
                "n_evidence": len(evidence),
                "ambiguity_type": "equal_evidence_all_suspects"
            }
        )

    def generate_dataset(self, n_per_regime: int = 6, seed: int = 42) -> List[T2Item]:
        """
        Generate a balanced dataset across all regimes and templates.

        Args:
            n_per_regime: Number of items per regime (will be distributed across templates)
            seed: Master seed for generation

        Returns:
            List of T2Item objects
        """
        self.master_seed = seed
        self.rng = random.Random(seed)

        templates = list(SCENARIO_TEMPLATES.keys())
        regimes = ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]

        items = []
        item_counter = 0

        for regime in regimes:
            # Distribute n_per_regime across templates evenly
            items_per_template = n_per_regime // len(templates)
            remainder = n_per_regime % len(templates)

            for template_idx, template in enumerate(templates):
                # Add extra item to first templates if there's remainder
                count = items_per_template + (1 if template_idx < remainder else 0)

                for i in range(count):
                    item_id = f"t2_{regime.lower()}_{template}_{item_counter:03d}"
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

    def generate_adversarial_permutations(self, item: T2Item, n_perms: int = 3, seed: int = 42) -> List[T2Item]:
        """
        Generate adversarial permutations of an item.
        Swaps names, shuffles hypothesis/evidence order, but maintains logical solution.

        Args:
            item: The base item to permute
            n_perms: Number of permutations to generate
            seed: Seed for permutation generation

        Returns:
            List of permuted T2Item objects
        """
        rng = random.Random(seed)
        permutations = []

        # Extract unique names from the item
        import re
        names_in_item = list(set([h.split(" is responsible")[0] for h in item.hypotheses if " is responsible" in h]))

        for perm_idx in range(n_perms):
            perm_rng = random.Random(seed + perm_idx)

            # Create name mapping (shuffle names)
            shuffled_names = names_in_item.copy()
            perm_rng.shuffle(shuffled_names)
            name_map = dict(zip(names_in_item, shuffled_names))

            # Swap names in all text
            def swap_names(text):
                if text is None:
                    return None
                result = text
                # Sort by length descending to avoid partial replacements
                for old_name in sorted(names_in_item, key=len, reverse=True):
                    result = result.replace(old_name, f"__TEMP_{name_map[old_name]}__")
                for old_name in names_in_item:
                    result = result.replace(f"__TEMP_{name_map[old_name]}__", name_map[old_name])
                return result

            # Create permuted item
            new_narrative = swap_names(item.narrative)
            new_question = swap_names(item.question)

            new_hypotheses = [swap_names(h) for h in item.hypotheses]
            perm_rng.shuffle(new_hypotheses)

            new_evidence = []
            for ev in item.evidence:
                new_ev = {
                    "id": ev["id"],
                    "content": swap_names(ev["content"]),
                    "supports": [swap_names(s) for s in ev["supports"]],
                    "contradicts": [swap_names(c) for c in ev["contradicts"]],
                    "diagnostic_value": ev["diagnostic_value"]
                }
                new_evidence.append(new_ev)
            perm_rng.shuffle(new_evidence)

            new_gold_answer = swap_names(item.gold_answer)
            new_gold_reasoning = swap_names(item.gold_reasoning)
            new_precedence = swap_names(item.source_precedence_rule)

            new_metadata = item.metadata.copy()
            new_metadata["permutation_of"] = item.id
            new_metadata["permutation_index"] = perm_idx

            perm_item = T2Item(
                id=f"{item.id}_perm{perm_idx:02d}",
                regime=item.regime,
                narrative=new_narrative,
                question=new_question,
                hypotheses=new_hypotheses,
                evidence=new_evidence,
                gold_answer=new_gold_answer,
                gold_reasoning=new_gold_reasoning,
                source_precedence_rule=new_precedence,
                metadata=new_metadata
            )

            permutations.append(perm_item)

        return permutations


def export_jsonl(items: List[T2Item], path: str):
    """Export items to JSONL format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')

    print(f"Exported {len(items)} items to {path}")


if __name__ == "__main__":
    # Example usage
    generator = T2Generator(seed=42)

    # Generate single items
    clean_item = generator.generate_clean_item("theft", seed=1, item_id="t2_clean_theft_001")
    decoy_item = generator.generate_decoy_item("sabotage", seed=2, item_id="t2_decoy_sabotage_001")
    conflict_item = generator.generate_conflict_item("data_breach", seed=3, item_id="t2_conflict_breach_001")
    insufficient_item = generator.generate_insufficient_item("contamination", seed=4, item_id="t2_insufficient_contam_001")

    # Generate full dataset
    dataset = generator.generate_dataset(n_per_regime=6, seed=42)

    print(f"Generated {len(dataset)} items")
    print(f"Regimes: {set(item.regime for item in dataset)}")
    print(f"Templates: {set(item.metadata.get('template') for item in dataset)}")
