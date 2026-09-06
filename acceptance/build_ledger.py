#!/usr/bin/env python3
"""Build claim-to-artifact ledger comparing prior round claims to actual file contents."""

import json
import hashlib
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preserved_prior_rounds")


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file_contains(path, substring):
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        return substring in f.read()


def build_ledger():
    ledger = {"rounds": []}

    # ---- ROUND v1: v3_2_counterfactual_pairs_review.md ----
    v1_path = os.path.join(PRIOR_DIR, "v3_2_counterfactual_pairs_review.md")
    v1_exists = os.path.exists(v1_path)
    v1_content = open(v1_path).read() if v1_exists else ""
    v1_lines = v1_content.count("\n") if v1_exists else 0

    v1_round = {
        "round": "v1",
        "artifact": "v3_2_counterfactual_pairs_review.md",
        "artifact_sha256": sha256_file(v1_path),
        "artifact_lines": v1_lines,
        "claims_vs_actuals": []
    }

    v1_round["claims_vs_actuals"].append({
        "claim": "Contains full serialized prompt texts for all 8 pairs",
        "actual": "File contains truth tables and summaries but NOT complete serialized prompts. "
                  "No fenced code blocks with preamble+narrative+question+options+evidence.",
        "verdict": "OVERSTATED"
    })
    v1_round["claims_vs_actuals"].append({
        "claim": "Independent oracle verifies labels",
        "actual": "Both symbolic_label() and exhaustive_viability_oracle() in the generator consume "
                  "the same hidden assignment arrays (criterion_a_assignment, criterion_b_assignment). "
                  "No text-derived oracle exists in v1.",
        "verdict": "OVERSTATED — oracle is not independent"
    })
    v1_round["claims_vs_actuals"].append({
        "claim": "B1-B7 baselines computed",
        "actual": "No fitted classifiers exist. Prior doc contains expected-value prose, not empirical results.",
        "verdict": "OVERSTATED — baselines were not actually fitted"
    })
    v1_round["claims_vs_actuals"].append({
        "claim": "DECOY name-frequency balanced",
        "actual": "DECOY construction gave guilty suspect 4 mentions vs 3 for innocents (extra corroboration).",
        "verdict": "FALSE — 4-vs-3 imbalance existed"
    })
    v1_round["claims_vs_actuals"].append({
        "claim": "Name-length strata correct",
        "actual": "Hand-written strata had incorrect character-length groupings.",
        "verdict": "INCORRECT — lengths were hand-coded with errors"
    })

    ledger["rounds"].append(v1_round)

    # ---- ROUND v2: v3_2_counterfactual_pairs_review_v2.md ----
    v2_path = os.path.join(PRIOR_DIR, "v3_2_counterfactual_pairs_review_v2.md")
    v2_exists = os.path.exists(v2_path)
    v2_content = open(v2_path).read() if v2_exists else ""

    # Count how many complete pairs' prompts are actually printed
    pair_prompt_count = v2_content.count("#### Member A: CLEAN (ANSWERABLE)")
    pair_prompt_count_b = v2_content.count("#### Member B: INSUFFICIENT")

    v2_round = {
        "round": "v2",
        "artifact": "v3_2_counterfactual_pairs_review_v2.md",
        "artifact_sha256": sha256_file(v2_path),
        "artifact_lines": v2_content.count("\n") if v2_exists else 0,
        "claims_vs_actuals": []
    }

    v2_round["claims_vs_actuals"].append({
        "claim": "Full serialized prompts for ALL 8 pairs",
        "actual": f"File contains {pair_prompt_count} CLEAN member prompts and "
                  f"{pair_prompt_count_b} INSUFFICIENT member prompts in fenced code blocks. "
                  f"{'All 8 pairs shown.' if pair_prompt_count >= 8 else 'Only ' + str(pair_prompt_count) + ' of 8 pairs have full prompts; the rest are summarized in a table.'}",
        "verdict": "PARTIALLY_IMPLEMENTED" if pair_prompt_count < 8 else "IMPLEMENTED"
    })
    v2_round["claims_vs_actuals"].append({
        "claim": "Text-derived oracle is independent (parses from text only)",
        "actual": "verify_v322.py contains a text_derived_oracle() function that parses Level-2/Level-1 "
                  "and flagged/cleared from the serialized prompt string. It does not import or access "
                  "assignment arrays. However: (a) it only handles CLEAN/INSUFFICIENT/DECOY, not CONFLICT; "
                  "(b) it is embedded in the verification script, not a standalone tested module; "
                  "(c) no AST-level independence enforcement exists.",
        "verdict": "PARTIALLY_IMPLEMENTED — CONFLICT oracle missing, no independence test"
    })
    v2_round["claims_vs_actuals"].append({
        "claim": "B1-B7 baselines computed empirically",
        "actual": "B3 (token count), B4 (evidence slot length), B5 (name frequency) have accuracy numbers. "
                  "B1 (word unigram TF-IDF) and B2 (char n-gram) are prose descriptions, not fitted classifiers. "
                  "B6 (polarity) is a prose note. B7 (combined) shows partial vote tallies for 3 items, "
                  "not a fitted classifier with accuracy. No classifier was trained. No LOFO evaluation.",
        "verdict": "OVERSTATED — only B3/B4/B5 had simple heuristic accuracy; B1/B2/B6/B7 were prose"
    })
    v2_round["claims_vs_actuals"].append({
        "claim": "Inference amendment uses 95th percentile (one-sided)",
        "actual": "AMENDMENT_v3_2_inference_rule_v2.md correctly specifies 95th percentile. "
                  "However, the pseudocode's lofo_balanced_accuracy() reuses precomputed predictions "
                  "rather than retraining per fold.",
        "verdict": "PARTIALLY_IMPLEMENTED — correct percentile, but LOFO pseudocode is not genuine retraining"
    })
    v2_round["claims_vs_actuals"].append({
        "claim": "32-family taxonomy with genuinely distinct families",
        "actual": "TAXONOMY_32_families.md lists 32 families across 8 categories x 4 variants. "
                  "However, all 32 share the same conjunction mechanism, same 'Level-2/Level-1' + "
                  "'flagged/cleared' binary evidence pattern, and same sentence structure. "
                  "Variants within a category differ only in noun substitutions (e.g., 'vault access' "
                  "vs 'evidence handling'). These are surface-level template variations, not "
                  "genuinely distinct structural families.",
        "verdict": "OVERSTATED — 32 surface variants of 1 structural pattern, not 32 independent structures"
    })
    v2_round["claims_vs_actuals"].append({
        "claim": "Option-length asymmetry does not affect leakage",
        "actual": "Review v2 acknowledges abstention is 4-7 chars longer but dismisses it as 'does not "
                  "leak suspect identity.' This understates the issue: within INSUFFICIENT, an option-length "
                  "baseline can achieve perfect accuracy (always pick longest = abstention = gold). "
                  "This was not quantified or formally addressed.",
        "verdict": "UNDERSTATED — option-length baseline can achieve 100% on INSUFFICIENT"
    })

    ledger["rounds"].append(v2_round)

    # ---- ROUND v2 supplementary files ----
    verify_path = os.path.join(PRIOR_DIR, "verify_v322.py")
    verify_exists = os.path.exists(verify_path)
    verify_content = open(verify_path).read() if verify_exists else ""

    # Count how many pairs the verify script actually outputs prompts for
    # It outputs "PAIR {pid} MEMBER A" only for pair_ids[0] (first pair)
    verify_full_output_count = verify_content.count("MEMBER A (CLEAN/ANSWERABLE)")

    v2_verify = {
        "round": "v2_supplementary",
        "artifact": "verify_v322.py",
        "artifact_sha256": sha256_file(verify_path),
        "claims_vs_actuals": []
    }
    v2_verify["claims_vs_actuals"].append({
        "claim": "Verification script prints full prompts for all 8 pairs",
        "actual": f"Script contains {verify_full_output_count} prompt output block(s). "
                  f"The STEP 4 code block only prints prompts for pair_ids[0] (the first pair).",
        "verdict": "OVERSTATED" if verify_full_output_count < 8 else "IMPLEMENTED"
    })
    v2_verify["claims_vs_actuals"].append({
        "claim": "Text-derived oracle tested on all regimes",
        "actual": "Oracle tested on CLEAN, INSUFFICIENT, DECOY. CONFLICT items produce PARSE_FAIL "
                  "because CONFLICT uses source-precedence (not conjunction), and the oracle "
                  "only parses Level-2/Level-1 + flagged/cleared.",
        "verdict": "PARTIALLY_IMPLEMENTED — CONFLICT regime not covered"
    })

    ledger["rounds"].append(v2_verify)

    return ledger


def write_ledger():
    ledger = build_ledger()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(out_dir, "claim_to_artifact_ledger.json")
    with open(json_path, "w") as f:
        json.dump(ledger, f, indent=2)

    md_path = os.path.join(out_dir, "claim_to_artifact_ledger.md")
    with open(md_path, "w") as f:
        f.write("# Claim-to-Artifact Fidelity Ledger\n\n")
        f.write("Factual record of what prior review rounds CLAIMED versus what the artifacts ACTUALLY CONTAINED.\n\n")
        for rnd in ledger["rounds"]:
            f.write(f"## Round: {rnd['round']}\n\n")
            f.write(f"**Artifact:** `{rnd['artifact']}`\n")
            if rnd.get("artifact_sha256"):
                f.write(f"**SHA-256:** `{rnd['artifact_sha256']}`\n")
            if rnd.get("artifact_lines"):
                f.write(f"**Lines:** {rnd['artifact_lines']}\n")
            f.write("\n")
            f.write("| # | Claim | Actual | Verdict |\n")
            f.write("|---|-------|--------|--------|\n")
            for i, item in enumerate(rnd["claims_vs_actuals"], 1):
                claim = item["claim"].replace("|", "\\|")
                actual = item["actual"].replace("|", "\\|").replace("\n", " ")
                verdict = item["verdict"]
                f.write(f"| {i} | {claim} | {actual} | {verdict} |\n")
            f.write("\n---\n\n")

    return json_path, md_path


if __name__ == "__main__":
    jp, mp = write_ledger()
    print(f"Wrote {jp}")
    print(f"Wrote {mp}")
