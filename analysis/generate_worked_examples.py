#!/usr/bin/env python3
"""
Generate hand-auditable worked examples for all 11 leakage baselines.

Produces analysis/leakage_worked_examples.md showing step-by-step
computation for each baseline on a single concrete item.

Phase A.2 corrective rewrite: includes ACTUAL fitted candidate
probabilities/scores for baselines 6-11, with intermediate values
(normalized candidate rows, feature vectors, classifier coefficients,
per-candidate scores) sufficient to reproduce the selected candidate
by hand.
"""
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.linear_model import LogisticRegression
from analysis.run_leakage_eval import (
    extract_name, ev_content_text, ev_content_list, gold_index,
    _extract_all_names, _target_normalize_text, _target_normalized_candidate_text,
    _compute_candidate_features, wilson_ci,
    _prepare_candidate_rows, _build_structured_candidate_rows,
    pred_majority, pred_position, pred_mention_count,
    pred_evidence_count, pred_lexical_overlap,
)


def _candidate_predict_with_probs(train_items, test_items, build_features_fn):
    """Same as _candidate_predict but returns (predictions, per_item_probs, clf).

    per_item_probs: list of arrays, one per test item, each array has
    P(gold=1) for each candidate in that item's hypotheses.
    """
    tr_X, tr_labels, tr_item_ids, tr_valid = build_features_fn(train_items, is_train=True)
    te_X, te_labels, te_item_ids, te_valid = build_features_fn(test_items, is_train=False)

    tr_row_valid = np.array([tr_valid[iid] for iid in tr_item_ids])
    tr_X_v = tr_X[tr_row_valid]
    tr_y_v = tr_labels[tr_row_valid]

    if len(set(tr_y_v.tolist())) < 2:
        return (np.zeros(len(test_items), dtype=int),
                [np.zeros(len(it["hypotheses"])) for it in test_items],
                None)

    clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
    clf.fit(tr_X_v, tr_y_v)

    col_1 = clf.classes_.tolist().index(1) if 1 in clf.classes_ else -1
    if col_1 >= 0:
        probs = clf.predict_proba(te_X)[:, col_1]
    else:
        probs = np.zeros(te_X.shape[0])

    full_preds = np.full(len(test_items), -1, dtype=int)
    per_item_probs = []
    for item_idx in range(len(test_items)):
        row_mask = te_item_ids == item_idx
        if not row_mask.any():
            per_item_probs.append(np.array([]))
            continue
        item_probs = probs[row_mask]
        per_item_probs.append(item_probs)
        full_preds[item_idx] = int(np.argmax(item_probs))

    return full_preds, per_item_probs, clf


def make_training_items():
    """Create a small set of training items for trained baseline examples."""
    items = []
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


def _format_tfidf_baseline(item, names, gi, train_items, test_items,
                            baseline_num, baseline_name, func_name,
                            analyzer, ngram_range, max_features):
    """Format a TF-IDF baseline with full probability details."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    lines = []
    lines.append("---")
    lines.append(f"## {baseline_num}. {baseline_name} (`{func_name}`)")
    lines.append("")

    if baseline_num == 6:
        lines.append("**Logic:** Expand each item into K candidate rows, each with TARGET-normalized")
        lines.append("text. Train a logistic regression on TF-IDF (word unigram+bigram, max 200 features).")
        lines.append("Predict the candidate with highest P(gold=1).")
    else:
        lines.append("**Logic:** Same as baseline 6 but with character n-grams (2-4, char_wb).")
        lines.append("This catches subword patterns that word-level TF-IDF misses.")
    lines.append("")

    if baseline_num == 6:
        lines.append("**TARGET normalization for this item:**")
        lines.append("")
        for j in range(len(item["hypotheses"])):
            normalized = _target_normalized_candidate_text(item, j)
            lines.append(f"  - Candidate {j} (\"{names[j]}\" -> TARGET):")
            lines.append(f"    \"{normalized[:150]}...\"")
        lines.append("")
        lines.append("**Key insight:** Each candidate row has DIFFERENT text because the")
        lines.append("TARGET/OTHER_k placeholders differ.")
        lines.append("")

    # Build features and train manually to capture probabilities
    vec = TfidfVectorizer(
        max_features=max_features,
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words='english' if analyzer == 'word' else None,
        dtype=np.float32,
    )

    # Build candidate rows
    def build_fn(items, is_train=False):
        texts, labels, item_ids, valid_mask = _prepare_candidate_rows(items)
        if is_train:
            X = vec.fit_transform(texts)
        else:
            X = vec.transform(texts)
        return X, labels, item_ids, valid_mask

    preds, per_item_probs, clf = _candidate_predict_with_probs(
        train_items, test_items, build_fn)

    # Report fitted probabilities
    lines.append("**Fitted candidate probabilities P(gold=1):**")
    lines.append("")
    if per_item_probs and len(per_item_probs[0]) > 0:
        for j in range(len(item["hypotheses"])):
            prob = per_item_probs[0][j]
            marker = " <-- argmax" if j == preds[0] else ""
            lines.append(f"  - Candidate {j} ({names[j]}): P(gold=1) = {prob:.4f}{marker}")
    lines.append("")

    lines.append(f"**Prediction:** index {preds[0]} -> \"{item['hypotheses'][preds[0]]}\"")
    lines.append(f"**Gold:** index {gi} -> \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {preds[0] == gi}")
    lines.append("")

    if baseline_num == 6:
        lines.append("**Why at chance if no leak:** In a non-leaking corpus, TARGET mentions are")
        lines.append("balanced across gold/non-gold rows -> classifier learns nothing -> ~1/K.")
        lines.append("")

    return "\n".join(lines)


def _format_structured_baseline(item, names, gi, train_items, test_items,
                                 baseline_num, baseline_name, func_name,
                                 col_selector, feature_description):
    """Format a structured-feature baseline with full probability details."""
    lines = []
    lines.append("---")
    lines.append(f"## {baseline_num}. {baseline_name} (`{func_name}`)")
    lines.append("")
    lines.append(f"**Logic:** {feature_description}")
    lines.append("")

    feat_names_all = ['mention_count', 'evidence_count', 'length_sum', 'first_mention_pos']

    # Show features for test item
    lines.append("**Structured features for this item (TARGET-normalized):**")
    lines.append("")

    all_feats = []
    for j in range(len(item["hypotheses"])):
        feats = _compute_candidate_features(item, j)
        all_feats.append(feats)

    for j in range(len(item["hypotheses"])):
        feats = all_feats[j]
        others = [all_feats[k] for k in range(len(item["hypotheses"])) if k != j]
        if baseline_num == 11:
            # Show full feature vector
            row = []
            for fn in feat_names_all:
                t_val = feats[fn]
                o_mean = np.mean([o[fn] for o in others])
                row.append(f"{fn}_t={t_val:.3f}")
                row.append(f"{fn}_d={t_val - o_mean:.3f}")
            lines.append(f"  - Candidate {j} ({names[j]} -> TARGET): [{', '.join(row)}]")
        elif baseline_num == 8:
            lines.append(f"  - Candidate {j} ({names[j]} -> TARGET):")
            o_mean_len = np.mean([o['length_sum'] for o in others])
            lines.append(f"    length_sum = {feats['length_sum']}, delta = {feats['length_sum'] - o_mean_len:.1f}")
        elif baseline_num == 9:
            lines.append(f"  - Candidate {j} ({names[j]} -> TARGET):")
            o_mean_mc = np.mean([o['mention_count'] for o in others])
            o_mean_ec = np.mean([o['evidence_count'] for o in others])
            lines.append(f"    mention_count={feats['mention_count']} (delta={feats['mention_count'] - o_mean_mc:.1f}), "
                         f"evidence_count={feats['evidence_count']} (delta={feats['evidence_count'] - o_mean_ec:.1f})")
        elif baseline_num == 10:
            lines.append(f"  - Candidate {j} ({names[j]} -> TARGET):")
            o_mean_fm = np.mean([o['first_mention_pos'] for o in others])
            lines.append(f"    first_mention_pos = {feats['first_mention_pos']:.4f} "
                         f"(delta = {feats['first_mention_pos'] - o_mean_fm:.4f})")
    lines.append("")

    if col_selector is not None:
        lines.append(f"**Training uses feature columns {col_selector}.**")
        lines.append("")

    # Train and get probabilities
    def build_fn(items, is_train=False):
        X, labels, item_ids, valid_mask = _build_structured_candidate_rows(items)
        if col_selector is not None:
            cols = [c for c in col_selector if c < X.shape[1]]
            X = X[:, cols]
        return X, labels, item_ids, valid_mask

    preds, per_item_probs, clf = _candidate_predict_with_probs(
        train_items, test_items, build_fn)

    # Report classifier details
    if clf is not None:
        lines.append("**Classifier coefficients:**")
        lines.append(f"  - Intercept: {clf.intercept_[0]:.4f}")
        coefs = clf.coef_[0]
        if col_selector is not None:
            used_names = []
            raw_feat_names = ['mention_count_t', 'mention_count_d',
                              'evidence_count_t', 'evidence_count_d',
                              'length_sum_t', 'length_sum_d',
                              'first_mention_pos_t', 'first_mention_pos_d']
            for c in col_selector:
                if c < len(raw_feat_names):
                    used_names.append(raw_feat_names[c])
                else:
                    used_names.append(f"col_{c}")
        else:
            used_names = ['mention_count_t', 'mention_count_d',
                          'evidence_count_t', 'evidence_count_d',
                          'length_sum_t', 'length_sum_d',
                          'first_mention_pos_t', 'first_mention_pos_d']
        for i, c in enumerate(coefs):
            name = used_names[i] if i < len(used_names) else f"col_{i}"
            lines.append(f"  - {name}: {c:.4f}")
        lines.append("")

    # Report fitted probabilities
    lines.append("**Fitted candidate probabilities P(gold=1):**")
    lines.append("")
    if per_item_probs and len(per_item_probs[0]) > 0:
        for j in range(len(item["hypotheses"])):
            prob = per_item_probs[0][j]
            marker = " <-- argmax" if j == preds[0] else ""
            lines.append(f"  - Candidate {j} ({names[j]}): P(gold=1) = {prob:.4f}{marker}")
    lines.append("")

    lines.append(f"**Prediction:** index {preds[0]} -> \"{item['hypotheses'][preds[0]]}\"")
    lines.append(f"**Gold:** index {gi} -> \"{item['gold_answer']}\"")
    lines.append(f"**Correct:** {preds[0] == gi}")
    lines.append("")

    return "\n".join(lines)


def generate_examples():
    item = make_example_item()
    names = _extract_all_names(item)
    gi = gold_index(item)

    lines = []
    lines.append("# Leakage Baseline Worked Examples")
    lines.append("")
    lines.append("Phase A.2 Work Item 5 (corrective rewrite): One hand-auditable worked")
    lines.append("example per baseline, computed on a single concrete item using the actual")
    lines.append("predictor implementations. Includes fitted candidate probabilities for")
    lines.append("all trained baselines (6-11).")
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
    lines.append(f"- **Prediction:** index {pred[0]} -> \"{item['hypotheses'][pred[0]]}\"")
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
    lines.append(f"- **Prediction:** index {pred[0]} -> \"{item['hypotheses'][pred[0]]}\"")
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
    lines.append(f"- Argmax -> index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} -> \"{item['hypotheses'][pred[0]]}\"")
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
    lines.append(f"- Argmax -> index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} -> \"{item['hypotheses'][pred[0]]}\"")
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
    lines.append(f"- Argmax -> index {pred[0]}")
    lines.append(f"- **Prediction:** index {pred[0]} -> \"{item['hypotheses'][pred[0]]}\"")
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

    # ---- Baseline 6: TF-IDF Word ----
    lines.append(_format_tfidf_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=6, baseline_name="TF-IDF Word", func_name="pred_tfidf_word",
        analyzer='word', ngram_range=(1, 2), max_features=200,
    ))

    # ---- Baseline 7: TF-IDF Char ----
    lines.append(_format_tfidf_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=7, baseline_name="TF-IDF Char", func_name="pred_tfidf_char",
        analyzer='char_wb', ngram_range=(2, 4), max_features=200,
    ))

    # ---- Baseline 8: Length ----
    lines.append(_format_structured_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=8, baseline_name="Length Feature", func_name="pred_length",
        col_selector=[4, 5],
        feature_description=("For each candidate row, compute `target_length_sum` (total "
                              "character length of evidence items containing TARGET) and its delta "
                              "vs other candidates. Train logistic regression on [target, delta] features."),
    ))

    # ---- Baseline 9: Mention + Evidence ----
    lines.append(_format_structured_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=9, baseline_name="Mention + Evidence", func_name="pred_mention_evidence",
        col_selector=[0, 1, 2, 3],
        feature_description=("TARGET-normalized mention count and evidence count features. "
                              "Columns [0,1,2,3] = target_mention, delta_mention, "
                              "target_evidence, delta_evidence."),
    ))

    # ---- Baseline 10: First Mention Order ----
    lines.append(_format_structured_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=10, baseline_name="First Mention Order", func_name="pred_first_mention_order",
        col_selector=[6, 7],
        feature_description=("For each candidate, find the character position of the first "
                              "TARGET occurrence in TARGET-normalized text (normalized by text length). "
                              "Earlier mention -> smaller value -> potentially more salient."),
    ))

    # ---- Baseline 11: Combined Shallow ----
    lines.append(_format_structured_baseline(
        item, names, gi, train_items, test_items,
        baseline_num=11, baseline_name="Combined Shallow", func_name="pred_combined",
        col_selector=None,
        feature_description=("All 8 structured features combined (4 raw + 4 delta). "
                              "This is the strongest structured baseline, using mention count, "
                              "evidence count, length, and first-mention position together."),
    ))

    lines.append("**Why at chance if no leak:** When evidence is balanced across candidates")
    lines.append("(each mentioned equally regardless of who is guilty), all features are")
    lines.append("~equal across gold and non-gold rows -> classifier learns nothing -> ~1/K.")
    lines.append("")

    # ---- Gate computation example ----
    lines.append("---")
    lines.append("## Gate Computation Example")
    lines.append("")
    lines.append("For a baseline with N=200 items at 3 options (chance=1/3):")
    lines.append("")
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
