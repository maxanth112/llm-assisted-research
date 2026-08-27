#!/usr/bin/env python3
"""
Generate hand-auditable worked examples for all 11 leakage baselines.

Produces analysis/leakage_worked_examples.md showing step-by-step
computation for each baseline on a single concrete item.

Phase A.2 Work Item 5.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from analysis.run_leakage_eval import (
    extract_name, ev_content_text, ev_content_list, gold_index,
    _extract_all_names, _target_normalize_text, _target_normalized_candidate_text,
    _compute_candidate_features, wilson_ci,
    pred_majority, pred_position, pred_mention_count,
    pred_evidence_count, pred_lexical_overlap,
)


def make_example_item():
    """Create a concrete example item for worked examples."""
    return {
        "id": "worked_example_001",
        "regime": "DECOY",
        "narrative": ("A fire broke out at the downtown warehouse on Tuesday night. "
                      "Three employees had access: Alice Chen, Bob Martinez, and Carol Davis. "
                      "Security footage was partially corrupted."),
        "question": "Who is most likely responsible for the fire?",
        "hypotheses": [
            "Alice Chen is responsible",
            "Bob Martinez is responsible",
            "Carol Davis is responsible",
        ],
        "evidence": [
            {"id": "E001", "content": "Alice Chen was seen leaving the warehouse at 9:15 PM, "
                                       "approximately 30 minutes before the fire started."},
            {"id": "E002", "content": "Bob Martinez had recently filed a complaint about "
                                       "unsafe wiring in the warehouse."},
            {"id": "E003", "content": "Carol Davis was on vacation during the week of the fire, "
                                       "according to HR records."},
            {"id": "E004", "content": "Alice Chen's keycard was used to access the storage room "
                                       "containing flammable materials at 9:10 PM."},
        ],
        "gold_answer": "Alice Chen is responsible",
        "gold_reasoning": "Alice Chen's keycard access and departure timing are most suspicious.",
        "metadata": {"template": "worked_example_template"},
    }


def generate_examples():
    item = make_example_item()
    names = _extract_all_names(item)
    gi = gold_index(item)

    lines = []
    lines.append("# Leakage Baseline Worked Examples")
    lines.append("")
    lines.append("Phase A.2 Work Item 5: One hand-auditable worked example per baseline,")
    lines.append("computed on a single concrete item using the actual predictor implementations.")
    lines.append("")
    lines.append("## Example Item")
    lines.append("")
    lines.append(f"- **ID:** {item['id']}")
    lines.append(f"- **Regime:** {item['regime']}")
    lines.append(f"- **Gold answer:** {item['gold_answer']} (index {gi})")
    lines.append(f"- **Hypotheses:** {item['hypotheses']}")
    lines.append(f"- **Extracted names:** {names}")
    lines.append(f"- **Narrative:** \"{item['narrative']}\"")
    lines.append(f"- **Evidence ({len(item['evidence'])} items):**")
    for ev in item["evidence"]:
        lines.append(f"  - {ev['id']}: \"{ev['content']}\"")
    lines.append("")

    # ---- Baseline 1: Majority class ----
    lines.append("---")
    lines.append("## 1. Majority Class (`pred_majority`)")
    lines.append("")
    lines.append("**Logic:** Predict the same label for every item (the majority gold")
    lines.append("label from the training set).")
    lines.append("")
    pred = pred_majority([item], majority_label=0)
    lines.append(f"- Majority label (from training): 0")
    lines.append(f"- **Prediction:** index {pred[0]} → \"{item['hypotheses'][pred[0]]}\"")
    lines.append(f"- **Correct:** {pred[0] == gi}")
    lines.append("")
    lines.append("**Why at chance:** In a balanced corpus with K candidates, each candidate")
    lines.append("is gold ~1/K of the time. Always predicting the same one gives ~1/K accuracy.")
    lines.append("")

    # ---- Baseline 2: Label position ----
    lines.append("---")
    lines.append("## 2. Label Position (`pred_position`)")
    lines.append("")
    lines.append("**Logic:** Always predict index 0 (first hypothesis).")
    lines.append("")
    pred = pred_position([item])
    lines.append(f"- **Prediction:** index {pred[0]} → \"{item['hypotheses'][pred[0]]}\"")
    lines.append(f"- **Correct:** {pred[0] == gi}")
    lines.append("")
    lines.append("**Why at chance:** If gold positions are uniformly distributed (enforced")
    lines.append("by S6), always picking position 0 gives ~1/K accuracy.")
    lines.append("")

    # ---- Baseline 3: Mention count ----
    lines.append("---")
    lines.append("## 3. Mention Count (`pred_mention_count`)")
    lines.append("")
    lines.append("**Logic:** For each hypothesis, extract the suspect name and count its")
    lines.append("occurrences in `narrative + evidence`. Predict the most-mentioned suspect.")
    lines.append("")
    text = (item.get("narrative", "") + " " + ev_content_text(item)).lower()
    pred = pred_mention_count([item])
    lines.append(f"- Combined text (lowered, {len(text)} chars): \"{text[:120]}...\"")
    lines.append("")
    for j, h in enumerate(item["hypotheses"]):
        n = extract_name(h).lower()
        count = text.count(n)
        lines.append(f"  - \"{n}\": count = {count}")
    lines.append(f"- Argmax → index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} → \"{item['hypotheses'][pred[0]]}\"")
    lines.append(f"- **Correct:** {pred[0] == gi}")
    lines.append("")
    lines.append("**Why at chance if no leak:** If evidence mentions each suspect equally")
    lines.append("often regardless of who is guilty, mention count has no predictive value.")
    lines.append("")

    # ---- Baseline 4: Evidence count ----
    lines.append("---")
    lines.append("## 4. Evidence Count (`pred_evidence_count`)")
    lines.append("")
    lines.append("**Logic:** Count how many evidence items mention each suspect's name.")
    lines.append("Predict the suspect mentioned in the most evidence items.")
    lines.append("")
    ev_contents = ev_content_list(item)
    pred = pred_evidence_count([item])
    for j, h in enumerate(item["hypotheses"]):
        n = extract_name(h).lower()
        ec = sum(1 for et in ev_contents if n in et.lower())
        lines.append(f"  - \"{n}\": appears in {ec}/{len(ev_contents)} evidence items")
    lines.append(f"- Argmax → index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} → \"{item['hypotheses'][pred[0]]}\"")
    lines.append(f"- **Correct:** {pred[0] == gi}")
    lines.append("")

    # ---- Baseline 5: Lexical overlap ----
    lines.append("---")
    lines.append("## 5. Lexical Overlap (`pred_lexical_overlap`)")
    lines.append("")
    lines.append("**Logic:** Compute word-level overlap (set intersection) between each")
    lines.append("hypothesis and the evidence text. Predict the hypothesis with most overlap.")
    lines.append("")
    ew = set(ev_content_text(item).lower().split())
    pred = pred_lexical_overlap([item])
    for j, h in enumerate(item["hypotheses"]):
        hw = set(h.lower().split())
        overlap = len(ew & hw)
        shared = sorted(ew & hw)[:8]
        lines.append(f"  - \"{h}\": overlap = {overlap} words ({shared}{'...' if len(ew & hw) > 8 else ''})")
    lines.append(f"- Argmax → index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} → \"{item['hypotheses'][pred[0]]}\"")
    lines.append(f"- **Correct:** {pred[0] == gi}")
    lines.append("")

    # ---- Baselines 6-7: TF-IDF (TARGET-normalized) ----
    lines.append("---")
    lines.append("## 6. TF-IDF Word (`pred_tfidf_word`)")
    lines.append("")
    lines.append("**Logic:** Expand each item into K candidate rows, each with TARGET-normalized")
    lines.append("text. Train a logistic regression on TF-IDF (word unigram+bigram, max 200 features).")
    lines.append("Predict the candidate with highest P(gold=1).")
    lines.append("")
    lines.append("**TARGET normalization for this item:**")
    lines.append("")
    for j in range(len(item["hypotheses"])):
        normalized = _target_normalized_candidate_text(item, j)
        lines.append(f"  - Candidate {j} (\"{names[j]}\" → TARGET):")
        lines.append(f"    \"{normalized[:150]}...\"")
    lines.append("")
    lines.append("**Key insight:** Each candidate row has DIFFERENT text because the")
    lines.append("TARGET/OTHER_k placeholders differ. This is what Phase A.2 fixed —")
    lines.append("in A.1, all rows had identical context (candidate name differences")
    lines.append("cancelled in TF-IDF).")
    lines.append("")
    lines.append("**Training:** Requires a train/test split (template-held-out CV).")
    lines.append("The classifier learns whether TARGET-implicated context predicts")
    lines.append("goldness. In a non-leaking corpus, TARGET mentions are balanced")
    lines.append("across gold/non-gold rows → accuracy ≈ chance.")
    lines.append("")

    lines.append("---")
    lines.append("## 7. TF-IDF Char (`pred_tfidf_char`)")
    lines.append("")
    lines.append("**Logic:** Same as baseline 6 but with character n-grams (2-4, char_wb).")
    lines.append("This catches subword patterns that word-level TF-IDF misses.")
    lines.append("")
    lines.append("**Same TARGET normalization as baseline 6** (only the vectorizer differs).")
    lines.append("")

    # ---- Baselines 8-11: Structured features (TARGET-normalized) ----
    lines.append("---")
    lines.append("## 8. Length Feature (`pred_length`)")
    lines.append("")
    lines.append("**Logic:** For each candidate row, compute `target_length_sum` (total")
    lines.append("character length of evidence items containing TARGET) and its delta")
    lines.append("vs other candidates. Train logistic regression on [target, delta] features.")
    lines.append("")
    lines.append("**Structured features for this item (TARGET-normalized):**")
    lines.append("")
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        lines.append(f"  - Candidate {j} ({names[j]} → TARGET):")
        lines.append(f"    length_sum = {feats['length_sum']}")
    lines.append("")
    lines.append("**Training uses columns [4, 5]** = target_length_sum, delta_length_sum.")
    lines.append("")

    lines.append("---")
    lines.append("## 9. Mention + Evidence (`pred_mention_evidence`)")
    lines.append("")
    lines.append("**Logic:** TARGET-normalized mention count and evidence count features.")
    lines.append("Columns [0,1,2,3] = target_mention, delta_mention, target_evidence, delta_evidence.")
    lines.append("")
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        lines.append(f"  - Candidate {j} ({names[j]} → TARGET):")
        lines.append(f"    mention_count={feats['mention_count']}, evidence_count={feats['evidence_count']}")
    lines.append("")

    lines.append("---")
    lines.append("## 10. First Mention Order (`pred_first_mention_order`)")
    lines.append("")
    lines.append("**Logic:** For each candidate, find the character position of the first")
    lines.append("TARGET occurrence in TARGET-normalized text (normalized by text length).")
    lines.append("Earlier mention → smaller value → potentially more salient.")
    lines.append("")
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        lines.append(f"  - Candidate {j} ({names[j]} → TARGET):")
        lines.append(f"    first_mention_pos = {feats['first_mention_pos']:.4f}")
    lines.append("")
    lines.append("**Training uses columns [6, 7]** = target_first_mention_pos, delta_first_mention_pos.")
    lines.append("")

    lines.append("---")
    lines.append("## 11. Combined Shallow (`pred_combined`)")
    lines.append("")
    lines.append("**Logic:** All 8 structured features combined (4 raw + 4 delta).")
    lines.append("This is the strongest structured baseline, using mention count,")
    lines.append("evidence count, length, and first-mention position together.")
    lines.append("")
    lines.append("**Full feature vector for this item:**")
    lines.append("")
    feat_names = ['mention_count', 'evidence_count', 'length_sum', 'first_mention_pos']
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        others = [_compute_candidate_features(item, k) for k in range(len(item["hypotheses"])) if k != j]
        row = []
        for fn in feat_names:
            t_val = feats[fn]
            o_mean = np.mean([o[fn] for o in others])
            row.append(f"{fn}_t={t_val:.3f}")
            row.append(f"{fn}_d={t_val - o_mean:.3f}")
        lines.append(f"  - Candidate {j}: [{', '.join(row)}]")
    lines.append("")
    lines.append("**Why at chance if no leak:** When evidence is balanced across candidates")
    lines.append("(each mentioned equally regardless of who is guilty), all features are")
    lines.append("~equal across gold and non-gold rows → classifier learns nothing → ~1/K.")
    lines.append("")

    # ---- Gate computation example ----
    lines.append("---")
    lines.append("## Gate Computation Example")
    lines.append("")
    lines.append("For a baseline with N=200 items at 3 options (chance=1/3):")
    lines.append("")
    # Example: 70 correct out of 200
    k, n = 70, 200
    ci_lo, ci_hi = wilson_ci(k, n)
    chance = 1/3
    threshold = chance + 0.05
    verdict = "PASS" if ci_hi <= threshold else "FAIL"
    lines.append(f"- Observed: {k}/{n} correct = {k/n:.3f}")
    lines.append(f"- Wilson 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    lines.append(f"- Chance: {chance:.4f}")
    lines.append(f"- Threshold (chance + 0.05): {threshold:.4f}")
    lines.append(f"- CI upper ({ci_hi:.4f}) {'<=' if ci_hi <= threshold else '>'} threshold ({threshold:.4f})")
    lines.append(f"- **Verdict: {verdict}**")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_examples()
    outpath = "analysis/leakage_worked_examples.md"
    with open(outpath, 'w') as f:
        f.write(report)
    print(f"Worked examples written to {outpath}", file=sys.stderr)
    print(report)
