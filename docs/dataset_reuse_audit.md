# Dataset Reuse Audit: ACH Scaffolding Experiment

## Evaluation Criteria

This audit evaluates candidate datasets for suitability as test items in an Analysis of Competing Hypotheses (ACH) scaffolding experiment.

**Required Criteria:**
- **(a)** 3+ explicit competing hypotheses / answer candidates
- **(b)** Evidence associated with multiple hypotheses
- **(c)** Misleading/decoy/asymmetric/contradictory/insufficient evidence
- **(d)** Deterministic or independently defensible ground truth
- **(e)** License permitting modification + redistribution

**Legend:** ✓ = YES, ✗ = NO, ~ = PARTIAL/UNCERTAIN

---

## Dataset Evaluation Summary

| Dataset | Source | License | (a) | (b) | (c) | (d) | (e) | Verdict | Notes |
|---------|--------|---------|-----|-----|-----|-----|-----|---------|-------|
| **MuSR** | TAUR-Lab/MuSR (HuggingFace) | CC-BY-4.0 | ~ | ✓ | ~ | ✓ | ✓ | **ACCEPT as T1** | 2-suspect ceiling; evidence relates to both suspects; some misleading evidence but not systematically tagged |
| **FaithEval** | SalesforceAIResearch/FaithEval (HuggingFace) | Apache-2.0 | ✓ | ✓ | ✓ | ✓ | ✓ | **ACCEPT** | Multi-choice format; designed for conflicting signals; strong candidate for conflict/insufficient arms |
| **RAMDocs** | HanNight/RAMDocs (HuggingFace) | Apache-2.0/MIT (verify) | ~ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Retrieval-augmented with conflicting docs; verify license at retrieval time |
| **ClashEval** | kevinwu23/StanfordClashEval (GitHub) | Research-use (verify) | ✓ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Parametric vs contextual conflict; Stanford repos may restrict redistribution |
| **ConflictQA** | Research dataset | Unknown | ✓ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Systematic conflict design; license verification needed |
| **ConflictBank** | Research dataset | Unknown | ✓ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Similar to ConflictQA; license verification required |
| **AbstentionBench** | Research dataset | Unknown | ~ | ~ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Strong for insufficient evidence scenarios; partial on hypothesis count |
| **True Detective** | MaksymDel/true-detective | Academic-use-only (suspected) | ✓ | ✓ | ✓ | ✓ | ✗ | **REJECT** | License likely restricts redistribution |
| **DetectiveQA** | Research dataset | Unknown | ✓ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Pending license verification |
| **TurnaboutLLM** | Research dataset | Unknown | ✓ | ✓ | ✓ | ✓ | ~ | **CONDITIONAL ACCEPT** | Pending license verification |

---

## Detailed Evaluations

### 1. MuSR (TAUR-Lab/MuSR)
**Source:** https://huggingface.co/datasets/TAUR-Lab/MuSR
**License:** CC-BY-4.0
**Subset:** Murder mystery scenarios

- **Criterion (a): PARTIAL** — Only 2 suspects per item, creating a ceiling for ACH matrix evaluation. Ideally 3+ hypotheses for full ACH utility.
- **Criterion (b): YES** — Evidence elements explicitly relate to both suspects.
- **Criterion (c): PARTIAL** — Contains some misleading evidence but not systematically tagged or designed for deception testing.
- **Criterion (d): YES** — Deterministic gold answers provided.
- **Criterion (e): YES** — CC-BY-4.0 permits modification and redistribution.

**Verdict:** **ACCEPT as T1** with noted 2-suspect ceiling limitation. Suitable as baseline test set but constrains ACH matrix size.

---

### 2. FaithEval (SalesforceAIResearch/FaithEval)
**Source:** https://huggingface.co/datasets/SalesforceAIResearch/FaithEval
**License:** Apache-2.0

- **Criterion (a): YES** — Multiple answer candidates in multi-choice format.
- **Criterion (b): YES** — Evidence/context items with conflicting signals across options.
- **Criterion (c): YES** — Explicitly designed to test faithfulness under conflicting information.
- **Criterion (d): YES** — Deterministic ground truth.
- **Criterion (e): YES** — Apache-2.0 permits modification and redistribution.

**Verdict:** **ACCEPT** as strong candidate for conflict and insufficient evidence experimental arms.

---

### 3. RAMDocs (HanNight/RAMDocs)
**Source:** https://huggingface.co/datasets/HanNight/RAMDocs
**License:** Likely Apache-2.0 or MIT (verify)

- **Criterion (a): PARTIAL** — May have multiple answer options but structure varies across items.
- **Criterion (b): YES** — Retrieval-augmented design with potentially conflicting documents.
- **Criterion (c): YES** — Conflicting document evidence by design.
- **Criterion (d): YES** — Deterministic answers.
- **Criterion (e): UNCERTAIN** — License needs verification at retrieval time; HuggingFace metadata suggests Apache-2.0 or MIT.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

### 4. ClashEval (kevinwu23/StanfordClashEval)
**Source:** https://github.com/kevinwu23/StanfordClashEval
**License:** Likely research-use only (verify)

- **Criterion (a): YES** — Competing claims between parametric knowledge and contextual evidence.
- **Criterion (b): YES** — Evidence from multiple sources (memory vs context).
- **Criterion (c): YES** — Designed specifically for conflict scenarios.
- **Criterion (d): YES** — Ground truth derived from context.
- **Criterion (e): UNCERTAIN** — Stanford repos sometimes restrict redistribution; requires license verification.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

### 5. ConflictQA
**Source:** Research dataset
**License:** Unknown

- **Criterion (a): YES** — Questions with competing answer candidates.
- **Criterion (b): YES** — Evidence with competing signals.
- **Criterion (c): YES** — Systematic conflict design.
- **Criterion (d): YES** — Deterministic ground truth.
- **Criterion (e): UNCERTAIN** — License verification needed.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

### 6. ConflictBank
**Source:** Research dataset
**License:** Unknown

- Similar profile to ConflictQA.
- All criteria met except license verification.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

### 7. AbstentionBench
**Source:** Research dataset
**License:** Unknown

- **Criterion (a): PARTIAL** — May not always have 3+ explicit hypotheses.
- **Criterion (c): YES** — Strong focus on insufficient evidence scenarios.
- **Criterion (e): UNCERTAIN** — License verification needed.

**Verdict:** **CONDITIONAL ACCEPT** specifically for insufficient evidence experimental arm.

---

### 8. True Detective (MaksymDel/true-detective)
**Source:** https://huggingface.co/datasets/MaksymDel/true-detective
**License:** Academic-use-only (suspected)

- **Criterion (e): REJECT** — License likely restricts commercial use and redistribution.

**Verdict:** **REJECT** due to licensing restrictions.

---

### 9. DetectiveQA
**Source:** Research dataset
**License:** Unknown

- Meets criteria (a)-(d) conceptually.
- **Criterion (e): UNCERTAIN** — License verification required.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

### 10. TurnaboutLLM
**Source:** Research dataset
**License:** Unknown

- Similar profile to DetectiveQA.
- **Criterion (e): UNCERTAIN** — License verification required.

**Verdict:** **CONDITIONAL ACCEPT** pending license verification.

---

## Conclusions and Recommendations

### Immediate Actions

1. **T1 Test Set:** Use **MuSR** (murder mystery subset) as baseline T1 test set.
   - Note the 2-suspect ceiling limitation for ACH matrix evaluation.
   - CC-BY-4.0 license ensures redistribution rights.

2. **Conflict/Insufficient Evidence Arms:** Deploy **FaithEval** and (pending license verification) **RAMDocs** for experimental arms testing conflicting and insufficient evidence scenarios.

3. **Deterministic T2 Construction:** Build a clean, deterministic T2 test set for the investigative regime with:
   - 3+ competing hypotheses per item
   - Systematic tagging of decoy/misleading evidence
   - Clean ground truth derivation
   - Full redistribution rights

### License Verification Queue

Priority license verification needed for:
1. RAMDocs (HanNight/RAMDocs)
2. ClashEval (kevinwu23/StanfordClashEval)
3. ConflictQA
4. ConflictBank
5. AbstentionBench
6. DetectiveQA
7. TurnaboutLLM

### Rejected

- **True Detective** — Academic-use-only license incompatible with redistribution requirements.

---

## Next Steps

1. Retrieve and validate MuSR using the provided manifest and retrieval script.
2. Retrieve and validate FaithEval.
3. Execute license verification for conditional acceptance datasets.
4. Design and construct deterministic T2 with 3+ hypothesis ceiling and systematic evidence tagging.
5. Document all dataset provenance, modifications, and licenses in compliance with redistribution requirements.
