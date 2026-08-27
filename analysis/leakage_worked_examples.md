# Leakage Baseline Worked Examples

Phase A.2 Work Item 5: One hand-auditable worked example per baseline,
computed on a single concrete item using the actual predictor implementations.

## Example Item

- **ID:** worked_example_001
- **Regime:** DECOY
- **Gold answer:** Alice Chen is responsible (index 0)
- **Hypotheses:** ['Alice Chen is responsible', 'Bob Martinez is responsible', 'Carol Davis is responsible']
- **Extracted names:** ['Alice Chen', 'Bob Martinez', 'Carol Davis']
- **Narrative:** "A fire broke out at the downtown warehouse on Tuesday night. Three employees had access: Alice Chen, Bob Martinez, and Carol Davis. Security footage was partially corrupted."
- **Evidence (4 items):**
  - E001: "Alice Chen was seen leaving the warehouse at 9:15 PM, approximately 30 minutes before the fire started."
  - E002: "Bob Martinez had recently filed a complaint about unsafe wiring in the warehouse."
  - E003: "Carol Davis was on vacation during the week of the fire, according to HR records."
  - E004: "Alice Chen's keycard was used to access the storage room containing flammable materials at 9:10 PM."

---
## 1. Majority Class (`pred_majority`)

**Logic:** Predict the same label for every item (the majority gold
label from the training set).

- Majority label (from training): 0
- **Prediction:** index 0 → "Alice Chen is responsible"
- **Correct:** True

**Why at chance:** In a balanced corpus with K candidates, each candidate
is gold ~1/K of the time. Always predicting the same one gives ~1/K accuracy.

---
## 2. Label Position (`pred_position`)

**Logic:** Always predict index 0 (first hypothesis).

- **Prediction:** index 0 → "Alice Chen is responsible"
- **Correct:** True

**Why at chance:** If gold positions are uniformly distributed (enforced
by S6), always picking position 0 gives ~1/K accuracy.

---
## 3. Mention Count (`pred_mention_count`)

**Logic:** For each hypothesis, extract the suspect name and count its
occurrences in `narrative + evidence`. Predict the most-mentioned suspect.

- Combined text (lowered, 541 chars): "a fire broke out at the downtown warehouse on tuesday night. three employees had access: alice chen, bob martinez, and c..."

  - "alice chen": count = 3
  - "bob martinez": count = 2
  - "carol davis": count = 2
- Argmax → index 0
- **Prediction:** index 0 → "Alice Chen is responsible"
- **Correct:** True

**Why at chance if no leak:** If evidence mentions each suspect equally
often regardless of who is guilty, mention count has no predictive value.

---
## 4. Evidence Count (`pred_evidence_count`)

**Logic:** Count how many evidence items mention each suspect's name.
Predict the suspect mentioned in the most evidence items.

  - "alice chen": appears in 2/4 evidence items
  - "bob martinez": appears in 1/4 evidence items
  - "carol davis": appears in 1/4 evidence items
- Argmax → index 0
- **Prediction:** index 0 → "Alice Chen is responsible"
- **Correct:** True

---
## 5. Lexical Overlap (`pred_lexical_overlap`)

**Logic:** Compute word-level overlap (set intersection) between each
hypothesis and the evidence text. Predict the hypothesis with most overlap.

  - "Alice Chen is responsible": overlap = 2 words (['alice', 'chen'])
  - "Bob Martinez is responsible": overlap = 2 words (['bob', 'martinez'])
  - "Carol Davis is responsible": overlap = 2 words (['carol', 'davis'])
- Argmax → index 0
- **Prediction:** index 0 → "Alice Chen is responsible"
- **Correct:** True

---
## 6. TF-IDF Word (`pred_tfidf_word`)

**Logic:** Expand each item into K candidate rows, each with TARGET-normalized
text. Train a logistic regression on TF-IDF (word unigram+bigram, max 200 features).
Predict the candidate with highest P(gold=1).

**TARGET normalization for this item:**

  - Candidate 0 ("Alice Chen" → TARGET):
    "TARGET is responsible [SEP] A fire broke out at the downtown warehouse on Tuesday night. Three employees had access: TARGET, OTHER_1, and OTHER_2. Sec..."
  - Candidate 1 ("Bob Martinez" → TARGET):
    "TARGET is responsible [SEP] A fire broke out at the downtown warehouse on Tuesday night. Three employees had access: OTHER_1, TARGET, and OTHER_2. Sec..."
  - Candidate 2 ("Carol Davis" → TARGET):
    "TARGET is responsible [SEP] A fire broke out at the downtown warehouse on Tuesday night. Three employees had access: OTHER_1, OTHER_2, and TARGET. Sec..."

**Key insight:** Each candidate row has DIFFERENT text because the
TARGET/OTHER_k placeholders differ. This is what Phase A.2 fixed —
in A.1, all rows had identical context (candidate name differences
cancelled in TF-IDF).

**Training:** Requires a train/test split (template-held-out CV).
The classifier learns whether TARGET-implicated context predicts
goldness. In a non-leaking corpus, TARGET mentions are balanced
across gold/non-gold rows → accuracy ≈ chance.

---
## 7. TF-IDF Char (`pred_tfidf_char`)

**Logic:** Same as baseline 6 but with character n-grams (2-4, char_wb).
This catches subword patterns that word-level TF-IDF misses.

**Same TARGET normalization as baseline 6** (only the vectorizer differs).

---
## 8. Length Feature (`pred_length`)

**Logic:** For each candidate row, compute `target_length_sum` (total
character length of evidence items containing TARGET) and its delta
vs other candidates. Train logistic regression on [target, delta] features.

**Structured features for this item (TARGET-normalized):**

  - Candidate 0 (Alice Chen → TARGET):
    length_sum = 194
  - Candidate 1 (Bob Martinez → TARGET):
    length_sum = 75
  - Candidate 2 (Carol Davis → TARGET):
    length_sum = 76

**Training uses columns [4, 5]** = target_length_sum, delta_length_sum.

---
## 9. Mention + Evidence (`pred_mention_evidence`)

**Logic:** TARGET-normalized mention count and evidence count features.
Columns [0,1,2,3] = target_mention, delta_mention, target_evidence, delta_evidence.

  - Candidate 0 (Alice Chen → TARGET):
    mention_count=3, evidence_count=2
  - Candidate 1 (Bob Martinez → TARGET):
    mention_count=2, evidence_count=1
  - Candidate 2 (Carol Davis → TARGET):
    mention_count=2, evidence_count=1

---
## 10. First Mention Order (`pred_first_mention_order`)

**Logic:** For each candidate, find the character position of the first
TARGET occurrence in TARGET-normalized text (normalized by text length).
Earlier mention → smaller value → potentially more salient.

  - Candidate 0 (Alice Chen → TARGET):
    first_mention_pos = 0.1742
  - Candidate 1 (Bob Martinez → TARGET):
    first_mention_pos = 0.1914
  - Candidate 2 (Carol Davis → TARGET):
    first_mention_pos = 0.2168

**Training uses columns [6, 7]** = target_first_mention_pos, delta_first_mention_pos.

---
## 11. Combined Shallow (`pred_combined`)

**Logic:** All 8 structured features combined (4 raw + 4 delta).
This is the strongest structured baseline, using mention count,
evidence count, length, and first-mention position together.

**Full feature vector for this item:**

  - Candidate 0: [mention_count_t=3.000, mention_count_d=1.000, evidence_count_t=2.000, evidence_count_d=1.000, length_sum_t=194.000, length_sum_d=118.500, first_mention_pos_t=0.174, first_mention_pos_d=-0.030]
  - Candidate 1: [mention_count_t=2.000, mention_count_d=-0.500, evidence_count_t=1.000, evidence_count_d=-0.500, length_sum_t=75.000, length_sum_d=-60.000, first_mention_pos_t=0.191, first_mention_pos_d=-0.004]
  - Candidate 2: [mention_count_t=2.000, mention_count_d=-0.500, evidence_count_t=1.000, evidence_count_d=-0.500, length_sum_t=76.000, length_sum_d=-58.500, first_mention_pos_t=0.217, first_mention_pos_d=0.034]

**Why at chance if no leak:** When evidence is balanced across candidates
(each mentioned equally regardless of who is guilty), all features are
~equal across gold and non-gold rows → classifier learns nothing → ~1/K.

---
## Gate Computation Example

For a baseline with N=200 items at 3 options (chance=1/3):

- Observed: 70/200 correct = 0.350
- Wilson 95% CI: [0.2873, 0.4184]
- Chance: 0.3333
- Threshold (chance + 0.05): 0.3833
- CI upper (0.4184) > threshold (0.3833)
- **Verdict: FAIL**
