#!/usr/bin/env python3
"""
Text-derived oracles for ALL FOUR regimes.

Each oracle takes ONLY a serialized prompt string.
It does NOT import, access, or reference any generator metadata,
assignment arrays, gold answers, labels, or viable_suspects.

Independence is enforced by construction: these functions accept
a single `str` argument and return a dict.
"""

import re
from typing import Dict, List, Optional


def _extract_suspect_names(prompt: str) -> List[str]:
    """Extract suspect names from hypothesis options, excluding abstention."""
    pattern = r"\([A-D]\) Based on the available evidence, (.+?) is uniquely identified as responsible\."
    raw = re.findall(pattern, prompt)
    return [n for n in raw if n != "no listed suspect"]


def _find_option_letter(prompt: str, text_fragment: str) -> Optional[str]:
    """Find which option letter (A-D) contains a text fragment."""
    options = re.findall(r"\(([A-D])\) (.+?)(?:\n|$)", prompt)
    for letter, body in options:
        if text_fragment in body:
            return letter
    return None


# ================================================================
# CONJUNCTION ORACLE (CLEAN, INSUFFICIENT, DECOY)
# ================================================================

def conjunction_oracle(prompt: str) -> Dict:
    """
    Text-derived oracle for conjunction-based regimes.
    Parses Level-2/Level-1 and flagged/cleared from evidence text.

    Args:
        prompt: the complete serialized prompt string (and NOTHING else)

    Returns dict with:
        status, parsed_facts, viable_suspects, n_viable, derived_label,
        derived_answer, derived_option
    """
    # Verify investigation rule is present
    if "INVESTIGATION RULE:" not in prompt:
        return {"status": "PARSE_FAIL", "reason": "No INVESTIGATION RULE found"}

    suspects = _extract_suspect_names(prompt)
    if not suspects:
        return {"status": "PARSE_FAIL", "reason": "No suspect names found in options"}

    parsed_facts = {}
    for name in suspects:
        esc = re.escape(name)

        # Criterion A: Level-2 vs Level-1
        has_l2 = bool(re.search(esc + r"\s+holds\s+a\s+Level-2\b", prompt))
        has_l1 = bool(re.search(esc + r"\s+holds\s+a\s+Level-1\b", prompt))

        # Criterion B: flagged vs cleared
        has_flag = bool(re.search(esc + r"\s+was\s+flagged\s+in\s+the\b", prompt))
        has_clear = bool(re.search(esc + r"\s+was\s+cleared\s+in\s+the\b", prompt))

        # Ambiguity checks — any ambiguity is a parse failure, not a silent answer
        if has_l2 and has_l1:
            return {"status": "PARSE_FAIL",
                    "reason": f"{name} has BOTH Level-2 AND Level-1"}
        if has_flag and has_clear:
            return {"status": "PARSE_FAIL",
                    "reason": f"{name} has BOTH flagged AND cleared"}
        if not has_l2 and not has_l1:
            return {"status": "PARSE_FAIL",
                    "reason": f"{name} has NEITHER Level-2 NOR Level-1"}
        if not has_flag and not has_clear:
            return {"status": "PARSE_FAIL",
                    "reason": f"{name} has NEITHER flagged NOR cleared"}

        parsed_facts[name] = {
            "criterion_a": has_l2,
            "criterion_b": has_flag,
            "meets_both": has_l2 and has_flag,
        }

    viable = [n for n, f in parsed_facts.items() if f["meets_both"]]
    n_viable = len(viable)

    if n_viable == 1:
        label = "ANSWERABLE"
        responsible = viable[0]
        answer = f"Based on the available evidence, {responsible} is uniquely identified as responsible."
        option = _find_option_letter(prompt, f"{responsible} is uniquely identified")
    elif n_viable >= 2:
        label = "INSUFFICIENT"
        responsible = None
        answer = "Based on the available evidence, no listed suspect is uniquely identified as responsible."
        option = _find_option_letter(prompt, "no listed suspect is uniquely identified")
    else:
        label = "INVALID"
        responsible = None
        answer = None
        option = None

    return {
        "status": "OK",
        "parsed_facts": parsed_facts,
        "viable_suspects": viable,
        "n_viable": n_viable,
        "derived_label": label,
        "derived_answer": answer,
        "derived_option": option,
        "responsible": responsible,
    }


# ================================================================
# CONFLICT ORACLE
# ================================================================

def conflict_oracle(prompt: str) -> Dict:
    """
    Text-derived oracle for CONFLICT regime.
    Parses source-precedence evidence from the prompt text.

    CONFLICT items do NOT use Level-2/Level-1 or flagged/cleared.
    Instead they use:
    - Access evidence (who was present, at what time)
    - Witness testimony (exonerating, low-reliability)
    - Automated system logs (high-reliability, identifies by access time)
    - Unverified reports (low-reliability, identifies by access time)
    - A source-precedence rule (official logs > testimony/unverified)

    The oracle must:
    1. Parse the EXPLICIT precedence rule — if absent, PARSE FAILURE
    2. Identify the automated log entry and which access time it references
    3. Map access times to suspects
    4. Apply precedence to determine the responsible suspect

    Args:
        prompt: the complete serialized prompt string (and NOTHING else)

    Returns dict with parsed evidence and derived answer.
    """
    suspects = _extract_suspect_names(prompt)
    if not suspects:
        return {"status": "PARSE_FAIL", "reason": "No suspect names found in options"}

    # MANDATORY: Parse the explicit precedence rule.
    # Absence or ambiguity -> PARSE FAILURE, never a silent answer.
    # Look for either the structured "SOURCE PRECEDENCE RULE:" prefix
    # or the phrase "official automated system logs take precedence"
    precedence_rule_match = re.search(
        r"SOURCE\s+PRECEDENCE\s+RULE:", prompt
    ) or re.search(
        r"official\s+automated\s+system\s+logs\s+take\s+precedence",
        prompt, re.IGNORECASE
    )
    if not precedence_rule_match:
        return {"status": "PARSE_FAIL",
                "reason": "No explicit precedence rule found "
                          "(expected 'SOURCE PRECEDENCE RULE:' or "
                          "'official automated system logs take precedence'). "
                          "Cannot resolve conflict without explicit precedence statement."}

    # DEFECT 3 FIX: Check for CONTRADICTORY precedence statements.
    # If multiple, mutually-exclusive precedence claims exist, the prompt
    # is ambiguous and must PARSE_FAIL — never silently pick one.
    # "Logs take precedence" contradicts "testimony takes precedence"
    # or "witness takes precedence" etc.
    log_precedence = bool(re.search(
        r"(automated\s+system\s+logs?|official\s+log|badge\s+system)\s+"
        r"(take|takes|has|have)\s+precedence",
        prompt, re.IGNORECASE
    ))
    witness_precedence = bool(re.search(
        r"(witness\s+testimon|witness\s+account|unverified\s+report|"
        r"testimony)\s+(take|takes|has|have)\s+precedence",
        prompt, re.IGNORECASE
    ))
    if log_precedence and witness_precedence:
        return {"status": "PARSE_FAIL",
                "reason": "Contradictory precedence rules detected: "
                          "both official logs AND witness testimony/reports "
                          "are claimed to take precedence. Cannot resolve "
                          "conflict with ambiguous precedence."}

    # Step 1: Parse access times for each suspect
    access_map = {}
    time_to_name = {}
    for name in suspects:
        esc = re.escape(name)
        m = re.search(esc + r".+?at\s+(\d+:\d+\s+[AP]M)", prompt)
        if m:
            t = m.group(1)
            access_map[name] = t
            time_to_name[t] = name

    if len(access_map) != len(suspects):
        return {"status": "PARSE_FAIL",
                "reason": f"Could not find access times for all suspects. "
                          f"Found: {access_map}"}

    # Step 2: Parse automated log entry (high-reliability)
    auto_match = re.search(
        r"Automated\s+badge\s+system\s+recorded\s+entry.+?"
        r"employee\s+who\s+entered\s+the\s+building\s+at\s+(\d+:\d+\s+[AP]M)",
        prompt
    )
    if not auto_match:
        return {"status": "PARSE_FAIL",
                "reason": "No automated badge system entry found. "
                          "Cannot identify suspect from official log."}

    auto_time = auto_match.group(1)
    auto_suspect = time_to_name.get(auto_time)
    if auto_suspect is None:
        return {"status": "PARSE_FAIL",
                "reason": f"Automated log references time {auto_time} but no suspect "
                          f"entered at that time. Known times: {time_to_name}"}

    # Step 3: Parse unverified report (low-reliability)
    unverified_match = re.search(
        r"unverified\s+witness\s+account.+?"
        r"entered\s+the\s+building\s+at\s+(\d+:\d+\s+[AP]M)",
        prompt, re.IGNORECASE
    )
    unverified_time = unverified_match.group(1) if unverified_match else None
    unverified_suspect = time_to_name.get(unverified_time) if unverified_time else None

    # Step 4: Verify genuine conflict exists (two sources disagree)
    if unverified_suspect is not None and unverified_suspect == auto_suspect:
        # Not actually a conflict — both sources agree
        pass  # Still valid, just no conflict to resolve

    # Step 5: Apply precedence — automated log wins
    responsible = auto_suspect
    label = "ANSWERABLE"
    answer = f"Based on the available evidence, {responsible} is uniquely identified as responsible."
    option = _find_option_letter(prompt, f"{responsible} is uniquely identified")

    return {
        "status": "OK",
        "parsed_facts": {
            "access_map": access_map,
            "automated_log_time": auto_time,
            "automated_log_suspect": auto_suspect,
            "unverified_report_time": unverified_time,
            "unverified_report_suspect": unverified_suspect,
            "precedence_rule_found": True,
            "conflict_present": (unverified_suspect is not None
                                 and unverified_suspect != auto_suspect),
        },
        "viable_suspects": [responsible],
        "n_viable": 1,
        "derived_label": label,
        "derived_answer": answer,
        "derived_option": option,
        "responsible": responsible,
    }


# ================================================================
# UNIFIED DISPATCHER
# ================================================================

def run_oracle(prompt: str, regime_hint: Optional[str] = None) -> Dict:
    """
    Run the appropriate oracle on a serialized prompt.

    If regime_hint is provided, use it to select the oracle.
    Otherwise, auto-detect from prompt content.
    """
    if regime_hint == "CONFLICT":
        return conflict_oracle(prompt)
    if regime_hint in ("CLEAN", "INSUFFICIENT", "DECOY"):
        return conjunction_oracle(prompt)

    # Auto-detect
    if "SOURCE PRECEDENCE RULE:" in prompt or "Automated badge system" in prompt:
        return conflict_oracle(prompt)
    if "INVESTIGATION RULE:" in prompt:
        return conjunction_oracle(prompt)

    return {"status": "PARSE_FAIL", "reason": "Could not determine regime from prompt content"}
