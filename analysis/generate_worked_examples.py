#!/usr/bin/env python3
"""
Generate hand-auditable worked examples for all 11 leakage baselines.

Produces analysis/leakage_worked_examples.md showing step-by-step
computation for each baseline on a single concrete item.

Phase A.2 Work Item 5 (cleanup: now includes actual trained predictions
for baselines 6-11, not just input features).
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
    pred_tfidf_word, pred_tfidf_char, pred_length,
    pred_mention_evidence, pred_first_mention_order, pred_combined,
)


def make_training_items():
    """Create a small set of training items for trained baseline examples.

    These are used as the training split for baselines 6-11 which require
    fitting a classifier. The training set is deliberately small and balanced
    to make the worked example tractable to audit by hand.
    """
    items = []
    # Training item 1: Gold = suspect 0
    items.append({
        "id": "train_001",
        "regime": "CLEAN",
        "narrative": ("A robbery occurred at the jewelry store on Monday morning. "
                      "Three suspects were identified: David Park, Elena Rodriguez, "
                      "and Frank Wilson. The alarm was triggered at 8:45 AM."),
        "question": "Who committed the robbery?",
        "hypotheses": [
            "David Park is responsible",
            "Elena Rodriguez is responsible",
            "Frank Wilson is responsible",
        ],
        "evidence": [
            {"id": "T001", "content": "David Park was seen near the store entrance at 8:40 AM by a security camera."},
            {"id": "T002", "content": "Elena Rodriguez was at a doctor's appointment during the robbery, confirmed by medical records."},
            {"id": "T003", "content": "Frank Wilson had no prior connection to the jewelry store."},
            {"id": "T004", "content": "David Park's fingerprints were found on the display case."},
        ],
        "gold_answer": "David Park is responsible",
        "gold_reasoning": "Direct physical evidence links David Park.",
        "metadata": {"template": "train_template_A"},
    })
    # Training item 2: Gold = suspect 1
    items.append({
        "id": "train_002",
        "regime": "CLEAN",
        "narrative": ("A data breach was discovered at the tech company on Friday. "
                      "Three employees had admin access: Grace Kim, Henry Liu, "
                      "and Iris Johnson. The breach occurred between 2-4 PM."),
        "question": "Who caused the data breach?",
        "hypotheses": [
            "Grace Kim is responsible",
            "Henry Liu is responsible",
            "Iris Johnson is responsible",
        ],
        "evidence": [
            {"id": "T005", "content": "Grace Kim was in a meeting from 1 PM to 5 PM, confirmed by five colleagues."},
            {"id": "T006", "content": "Henry Liu's access logs show database queries at 2:30 PM and 3:15 PM."},
            {"id": "T007", "content": "Iris Johnson was working remotely from a different city that day."},
            {"id": "T008", "content": "Henry Liu had recently been denied a promotion and expressed frustration."},
        ],
        "gold_answer": "Henry Liu is responsible",
        "gold_reasoning": "Access logs and motive point to Henry Liu.",
        "metadata": {"template": "train_template_B"},
    })
    # Training item 3: Gold = suspect 2
    items.append({
        "id": "train_003",
        "regime": "DECOY",
        "narrative": ("A valuable painting was stolen from the gallery overnight. "
                      "Three people had keys: Jack Chen, Karen White, "
                      "and Leo Brown. The security system was disabled at 11 PM."),
        "question": "Who stole the painting?",
        "hypotheses": [
            "Jack Chen is responsible",
            "Karen White is responsible",
            "Leo Brown is responsible",
        ],
        "evidence": [
            {"id": "T009", "content": "Jack Chen was attending a concert that evening with tickets as proof."},
            {"id": "T010", "content": "Karen White mentioned wanting the painting but has a solid alibi."},
            {"id": "T011", "content": "Leo Brown's car was spotted in the gallery parking lot at 11:05 PM."},
            {"id": "T012", "content": "Leo Brown had recently taken out a large insurance policy."},
        ],
        "gold_answer": "Leo Brown is responsible",
        "gold_reasoning": "Physical presence and financial motive point to Leo Brown.",
        "metadata": {"template": "train_template_C"},
    })
    # Training item 4: Gold = suspect 0
    items.append({
        "id": "train_004",
        "regime": "DECOY",
        "narrative": ("An explosion damaged the chemical plant on Wednesday. "
                      "Three technicians were on duty: Maria Santos, Nathan Gray, "
                      "and Olivia Reed. The blast occurred in sector 7."),
        "question": "Who caused the explosion?",
        "hypotheses": [
            "Maria Santos is responsible",
            "Nathan Gray is responsible",
            "Olivia Reed is responsible",
        ],
        "evidence": [
            {"id": "T013", "content": "Maria Santos was last seen entering sector 7 at 3:20 PM."},
            {"id": "T014", "content": "Nathan Gray was working in sector 2, far from the blast site."},
            {"id": "T015", "content": "Olivia Reed had reported safety concerns about sector 7 earlier that week."},
            {"id": "T016", "content": "Maria Santos had modified the pressure valve settings without authorization."},
        ],
        "gold_answer": "Maria Santos is responsible",
        "gold_reasoning": "Unauthorized modifications and presence at scene.",
        "metadata": {"template": "train_template_D"},
    })
    # Training item 5: Gold = suspect 1
    items.append({
        "id": "train_005",
        "regime": "CONFLICT",
        "narrative": ("Money went missing from the charity fund on Thursday. "
                      "Three board members had access: Paul Anderson, Quinn Taylor, "
                      "and Rachel Davis. The discrepancy was $50,000."),
        "question": "Who took the money?",
        "hypotheses": [
            "Paul Anderson is responsible",
            "Quinn Taylor is responsible",
            "Rachel Davis is responsible",
        ],
        "evidence": [
            {"id": "T017", "content": "Paul Anderson had recently made large personal purchases."},
            {"id": "T018", "content": "Quinn Taylor's personal account received a $50,000 deposit on Friday."},
            {"id": "T019", "content": "Rachel Davis was traveling abroad during the period in question."},
            {"id": "T020", "content": "Quinn Taylor had access to the fund transfer system."},
        ],
        "gold_answer": "Quinn Taylor is responsible",
        "gold_reasoning": "Matching deposit and system access.",
        "metadata": {"template": "train_template_E"},
    })
    # Training item 6: Gold = suspect 2
    items.append({
        "id": "train_006",
        "regime": "CONFLICT",
        "narrative": ("Confidential documents were leaked to the press. "
                      "Three executives had clearance: Sam Morgan, Tina Chen, "
                      "and Victor Hall. The leak occurred last Tuesday."),
        "question": "Who leaked the documents?",
        "hypotheses": [
            "Sam Morgan is responsible",
            "Tina Chen is responsible",
            "Victor Hall is responsible",
        ],
        "evidence": [
            {"id": "T021", "content": "Sam Morgan had no motive and cooperated fully with the investigation."},
            {"id": "T022", "content": "Tina Chen's email account showed suspicious forwarding rules set up Monday."},
            {"id": "T023", "content": "Victor Hall was seen meeting with a journalist at a cafe on Tuesday morning."},
            {"id": "T024", "content": "Victor Hall's computer contained copies of the leaked documents in a hidden folder."},
        ],
        "gold_answer": "Victor Hall is responsible",
        "gold_reasoning": "Meeting with journalist and document copies on computer.",
        "metadata": {"template": "train_template_F"},
    })
    return items


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

    # ---- Trained baselines: create training set ----
    train_items = make_training_items()
    test_items = [item]

    lines.append("---")
    lines.append("## Training Items for Baselines 6-11")
    lines.append("")
    lines.append("Baselines 6-11 require a train/test split. The following 6 training")
    lines.append("items are used (the test item is the example item above):")
    lines.append("")
    for ti in train_items:
        ti_gi = gold_index(ti)
        lines.append(f"- **{ti['id']}** (regime={ti['regime']}, gold=index {ti_gi}: "
                      f"\"{ti['gold_answer']}\")")
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

    # Actual trained prediction
    pred_word = pred_tfidf_word(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_word[0]} → \"{item['hypotheses'][pred_word[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_word[0] == gi}")
    lines.append("")
    lines.append("**Why at chance if no leak:** In a non-leaking corpus, TARGET mentions are")
    lines.append("balanced across gold/non-gold rows → classifier learns nothing → ~1/K.")
    lines.append("")

    lines.append("---")
    lines.append("## 7. TF-IDF Char (`pred_tfidf_char`)")
    lines.append("")
    lines.append("**Logic:** Same as baseline 6 but with character n-grams (2-4, char_wb).")
    lines.append("This catches subword patterns that word-level TF-IDF misses.")
    lines.append("")
    lines.append("**Same TARGET normalization as baseline 6** (only the vectorizer differs).")
    lines.append("")

    # Actual trained prediction
    pred_char = pred_tfidf_char(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_char[0]} → \"{item['hypotheses'][pred_char[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_char[0] == gi}")
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
    feat_names_all = ['mention_count', 'evidence_count', 'length_sum', 'first_mention_pos']
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        lines.append(f"  - Candidate {j} ({names[j]} → TARGET):")
        lines.append(f"    length_sum = {feats['length_sum']}")
    lines.append("")
    lines.append("**Training uses columns [4, 5]** = target_length_sum, delta_length_sum.")
    lines.append("")

    # Actual trained prediction
    pred_len = pred_length(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_len[0]} → \"{item['hypotheses'][pred_len[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_len[0] == gi}")
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

    # Actual trained prediction
    pred_me = pred_mention_evidence(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_me[0]} → \"{item['hypotheses'][pred_me[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_me[0] == gi}")
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

    # Actual trained prediction
    pred_fmo = pred_first_mention_order(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_fmo[0]} → \"{item['hypotheses'][pred_fmo[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_fmo[0] == gi}")
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
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        others = [_compute_candidate_features(item, k) for k in range(len(item["hypotheses"])) if k != j]
        row = []
        for fn in feat_names_all:
            t_val = feats[fn]
            o_mean = np.mean([o[fn] for o in others])
            row.append(f"{fn}_t={t_val:.3f}")
            row.append(f"{fn}_d={t_val - o_mean:.3f}")
        lines.append(f"  - Candidate {j}: [{', '.join(row)}]")
    lines.append("")

    # Actual trained prediction
    pred_comb = pred_combined(train_items, test_items)
    lines.append(f"**Trained prediction:** index {pred_comb[0]} → \"{item['hypotheses'][pred_comb[0]]}\"")
    lines.append(f"**Gold:** index {gi} → \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {pred_comb[0] == gi}")
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
