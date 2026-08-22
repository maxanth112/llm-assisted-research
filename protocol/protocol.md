# EXP-1: ETD-ACH Factorial Experiment Protocol

**Status**: PREREGISTRATION-READY (NOT YET SUBMITTED)

**Version**: 1.0.0

**Date**: 2026-08-22

**Protocol Hash**: TO_BE_COMPUTED (see PROTOCOL.lock.json)

---

## Research Question

Does explicit disconfirmation scoring (the D component of Analysis of Competing Hypotheses) improve LLM accuracy on investigative reasoning tasks beyond what is achieved by hypothesis enumeration (E) and evidence tabulation (T) alone?

### Secondary Questions

1. Are E, T, D effects additive or interactive?
2. Do scaffolding benefits persist after controlling for computational cost (tokens, API calls)?
3. Does evidence quality (CLEAN vs DECOY vs CONFLICT) moderate the effects of E/T/D?
4. Do models exhibit self-unfaithfulness (final answer contradicts their own ACH matrix conclusion)?
5. Does scaffolding reduce sensitivity to adversarial input ordering?

---

## Primary Hypotheses

### Main Effects

**H1**: E > baseline (Enumeration alone improves accuracy)
- Operationalization: `accuracy(E=1) > accuracy(E=0)`, averaged over T and D
- Rationale: Explicitly listing hypotheses forces systematic consideration of alternatives

**H2**: T > E (Tabulation improves accuracy beyond enumeration alone)
- Operationalization: `accuracy(T=1) > accuracy(T=0)`, averaged over E and D
- Rationale: Structured evidence-hypothesis tables reduce cognitive load and prevent evidence from being overlooked

**H3**: D > E+T (Disconfirmation scoring improves accuracy beyond enumeration+tabulation)
- Operationalization: `accuracy(D=1) > accuracy(D=0)`, averaged over E and T
- Rationale: Explicit disconfirmation logic (ACH's core mechanism) should improve beyond mere structure
- **Prior expectation**: Small or null effect (skeptical prior based on preliminary observations)

### Interactions and Moderators

**H4**: D's benefit is larger under DECOY/CONFLICT regimes than CLEAN (interaction with evidence quality)
- Operationalization: `[accuracy(D=1) - accuracy(D=0)]_DECOY/CONFLICT > [accuracy(D=1) - accuracy(D=0)]_CLEAN`
- Rationale: Disconfirmation scoring should be most valuable when misleading evidence is present

### Computational Cost Controls

**H5**: T/D gains vanish when token budget and call count are matched (compute artifact hypothesis)
- Operationalization:
  - Token-matched: `accuracy(E=1,T=1,D=1) ≈ accuracy(free_cot)` when both produce similar token counts
  - Call-matched: `accuracy(E=1,T=1,D=1) ≈ accuracy(prism_full)` when both use multiple reasoning passes
- Rationale: Scaffolding benefits may be artifacts of increased computation rather than structural advantages

### Faithfulness and Robustness

**H6**: Self-unfaithfulness rate > 0 (model's final answer contradicts its own ACH matrix conclusion)
- Operationalization: For conditions producing ACH matrices (D=1), measure frequency where `final_answer ≠ least_disconfirmed_hypothesis`
- Rationale: Models may perform correct reasoning but fail to follow through in final answer

**H7**: Scaffolding reduces order-flip rate (answers remain stable under hypothesis/evidence permutation)
- Operationalization: `order_flip_rate(scaffolded) < order_flip_rate(baseline)`
- Rationale: Structured reasoning should be more robust to input ordering artifacts

---

## Primary Outcomes

### 1. Accuracy (Primary)

**Definition**: Exact match between model's final answer and gold-standard answer, after normalization

**Normalization**:
- Case-insensitive
- Whitespace-stripped
- Common variations mapped to canonical form (e.g., "Hypothesis 1" → "H1")

**Measurement**: Binary (0 = incorrect, 1 = correct) per item-condition-run

**Aggregation**:
- Within-item: Average over k_runs
- Within-condition: Average over all items
- Overall: Mixed-effects model (see Statistical Model section)

### 2. Abstention Correctness (Primary for INSUFFICIENT Regime)

**Definition**: Proportion of INSUFFICIENT regime items where model appropriately abstains or indicates "cannot determine"

**Measurement**: Binary per item (1 = appropriate abstention, 0 = inappropriate commitment)

**Gold standard**: INSUFFICIENT regime items have no determinate answer; correct response is to abstain or express uncertainty

**Parser**: Regex matching variations of "insufficient", "cannot determine", "need more information", etc.

**Note**: Standard accuracy metric is inverted for INSUFFICIENT items (correct = recognizing indeterminacy)

---

## Secondary Outcomes

### 3. Calibration

**Metrics**:
- Expected Calibration Error (ECE): Mean absolute difference between confidence and empirical accuracy across bins
- Brier score: Mean squared error between predicted probabilities and binary outcomes

**Measurement**: Extracted from model's stated confidence (if provided in response) or logprobs (if available via API)

**Bins**: 10 equal-width bins [0-0.1, 0.1-0.2, ..., 0.9-1.0]

### 4. Sensitivity to Misleading Evidence

**Metric**: Accuracy delta between evidence quality regimes

**Operationalization**:
- `sensitivity_decoy = accuracy_CLEAN - accuracy_DECOY`
- `sensitivity_conflict = accuracy_CLEAN - accuracy_CONFLICT`

**Hypothesis**: Scaffolded conditions should show lower sensitivity (smaller accuracy drop) than baseline

### 5. Order-Flip Sensitivity

**Procedure**:
- For each item, generate adversarial permutations:
  - Hypothesis order reversed
  - Evidence order reversed
  - Both reversed
- Measure answer consistency across permutations

**Metric**: Flip rate = proportion of items where answer changes under any permutation

**Measurement**: Subset of items (e.g., 50 items × 3 permutations = 150 additional calls)

**Hypothesis**: Structured conditions (T=1, D=1) should exhibit lower flip rates

### 6. Cross-Run Consistency

**Metric**: Self-consistency rate = proportion of items where all k_runs produce the same answer

**Operationalization**: Binary per item-condition (1 = all k runs agree, 0 = disagreement across runs)

**Hypothesis**: Higher consistency indicates more deterministic/robust reasoning

### 7. Cost

**Metrics**:
- Tokens per item (input + output)
- API cost per item (in USD)
- Latency per item (wall-clock time)

**Purpose**: Cost-effectiveness analysis; identify whether gains justify increased expense

### 8. Matrix-Conclusion Faithfulness

**Definition**: For conditions producing ACH matrices (D=1), proportion where final answer matches the least-disconfirmed hypothesis from the model's own matrix

**Measurement**:
1. Parse ACH matrix to extract inconsistency counts per hypothesis
2. Identify hypothesis with lowest inconsistency count (model's "ACH conclusion")
3. Compare to final answer
4. Binary: 1 = faithful (answer matches conclusion), 0 = unfaithful (contradiction)

**Hypothesis**: Unfaithfulness rate should be low but non-zero (models sometimes ignore their own analysis)

---

## Experimental Design

### Factor Structure

**3-way factorial design**: 2³ = 8 possible combinations, but only 5 + 3 reference conditions tested

**Factors**:

| Factor | Levels | Description |
|--------|--------|-------------|
| **E** (Enumerate) | 0, 1 | 0 = no explicit hypothesis listing; 1 = "list all candidate hypotheses" |
| **T** (Tabulate) | 0, 1 | 0 = no structured table; 1 = evidence×hypothesis table |
| **D** (Disconfirm) | 0, 1 | 0 = no consistency scoring; 1 = C/I/N codes + inconsistency counts + "select least disconfirmed" |

**Tested Conditions**:

#### Factorial Conditions (5)

1. **000** (baseline): Direct answer, no scaffolding
2. **100** (E only): Enumerate hypotheses, then answer
3. **110** (E+T): Enumerate hypotheses, create evidence table, then answer
4. **101** (E+D): Enumerate hypotheses, score disconfirmation, then answer (note: D without T uses linear list format)
5. **111** (E+T+D, full ACH): Complete ACH matrix with C/I/N scoring and "least disconfirmed" selection

#### Reference Conditions (3)

6. **filter_only**: Minimal scaffolding, just "filter irrelevant evidence"
7. **prism_full**: 4-call decomposed reasoning (question→subquestions→subanswers→synthesis)
8. **free_cot**: Unrestricted chain-of-thought ("think step by step, show your reasoning")

**Untested Combinations** (excluded to reduce condition space):
- 010 (T only, without E): Theoretically incoherent (can't create hypothesis table without enumerating hypotheses)
- 011 (T+D without E): Same incoherence
- 001 (D only, without E): Same incoherence

---

## Contrasts and Comparisons

### Primary Contrasts (for hypothesis testing)

1. **E main effect**: Conditions where E=1 vs E=0 (averaged over T, D)
2. **T main effect**: Conditions where T=1 vs T=0 (averaged over E, D)
3. **D main effect**: Conditions where D=1 vs D=0 (averaged over E, T)

### Pairwise Contrasts (for interaction testing)

4. **E effect**: 100 vs 000 (E only vs baseline)
5. **T effect given E**: 110 vs 100 (adding T to E)
6. **D effect given E**: 101 vs 100 (adding D to E)
7. **D effect given E+T**: 111 vs 110 (adding D to E+T)

### Matched Controls (for cost-control analysis)

8. **Token-matched**: 111 (full ACH) vs free_cot (both produce long outputs)
9. **Call-matched**: 111 (single-call) vs prism_full (4-call decomposition)

---

## Dataset and Item Selection

### T2 Dataset (Synthetic Investigative Reasoning)

**Source**: Custom-generated via `src/data/generators/t2_generator.py`

**Regimes** (evidence quality manipulations):
- **CLEAN**: All evidence is valid and relevant
- **DECOY**: Contains misleading but consistent evidence favoring a wrong hypothesis
- **CONFLICT**: Contains contradictory evidence with no clear resolution
- **INSUFFICIENT**: Evidence is inadequate to determine the answer

**Target N**: 100 items (25 per regime)

**Pilot N**: 24 items (6 per regime)

**Inclusion**: All generated items that pass validation (no parse errors in gold answer)

### MuSR Dataset (Murder Mystery Reasoning)

**Source**: MuSR benchmark, murder_mystery subset (publicly available)

**Subset**: Murder mystery scenarios only (exclude other MuSR task types)

**Estimated N**: ~250 items after exclusions

**Exclusions**:
- Items where gold answer cannot be parsed reliably
- Items flagged as malformed in MuSR metadata (if available)

### Exclusion Rules

**Item-level exclusion** (before experiment):
- Items where gold answer cannot be extracted or validated

**Condition-level exclusion** (flagged in analysis, not excluded):
- Parse failures scored as incorrect (not dropped)
- If any condition shows >20% parse failure rate, flag for investigation

**No post-hoc exclusions**: Items are not excluded based on results (e.g., high variance, unexpected patterns)

---

## Sampling and Repetition

### Within-Item Design

**Key feature**: Each item is tested under ALL conditions
- Enables within-item paired comparisons (McNemar test)
- Controls for item-level difficulty variance
- Increases statistical power

### Repeated Runs (k_runs)

**Pilot**: k = 1 (or 3 if time permits)

**Confirmatory**: k = 3 (minimum) to 5 (preferred)

**Purpose**:
- Estimate model stochasticity (even at temperature=0, minor variations occur)
- Compute self-consistency metrics
- Increase robustness to transient API errors

**Aggregation**: Average over k_runs within each item×condition cell before analysis

---

## Model Selection

**Primary models** (see docs/model_access_plan.md):
1. Llama-3.3-70B-Instruct (via Together AI)
2. Qwen2.5-72B-Instruct (via Together AI)

**Cross-family check** (optional):
3. GPT-4o-mini (via OpenAI)

**Rationale**:
- Two genuinely different open-weight families (Llama vs Qwen)
- Cost-effective (~$81 for full confirmatory study)
- Reproducible (temperature=0 supported)

**Model as random effect**: In mixed-effects analysis, model is treated as a random effect (assumes effects generalize beyond these specific models)

---

## Statistical Analysis Plan

### Primary Analysis: Mixed-Effects Logistic Regression

**Model specification**:

```
logit(P(correct)) ~ E * T * D + regime + (1 | item) + (1 | model)
```

**Where**:
- `E`, `T`, `D`: Binary fixed effects (0/1)
- `E * T * D`: All main effects, 2-way interactions, and 3-way interaction
- `regime`: Fixed effect for evidence quality (CLEAN, DECOY, CONFLICT, INSUFFICIENT)
- `(1 | item)`: Random intercept for item (accounts for item difficulty variance)
- `(1 | model)`: Random intercept for model (accounts for base model capability)

**Software**: R `lme4` package or Python `statsmodels` mixed-effects logit

**Coefficient interpretation**:
- Main effect of D: Average log-odds improvement from adding disconfirmation scoring
- Interaction D × regime: Whether D's effect varies by evidence quality

### Secondary Analysis: Pairwise Comparisons

**Method**: McNemar test for paired binary outcomes

**Application**:
- Compare accuracy within-item across condition pairs (e.g., 111 vs 110)
- Non-parametric alternative to mixed-effects model

**Advantage**: Does not assume normality or homogeneity; robust to outliers

### Reference Condition Comparisons

**Token-matched**:
- Compare 111 vs free_cot
- Control for total token count (both conditions generate long outputs)
- Test: If accuracy_111 ≈ accuracy_free_cot, suggests gains are computational artifacts

**Call-matched**:
- Compare 111 vs prism_full
- Control for number of model calls (PRISM uses 4 calls; 111 uses 1)
- Test: If accuracy_111 < accuracy_prism_full, suggests decomposition (not ACH structure) drives gains

---

## Multiple Comparison Correction

### Family-Wise Strategy

**Primary family** (confirmatory tests): H1, H2, H3
- 3 tests (E main effect, T main effect, D main effect)
- Correction: Benjamini-Hochberg at FDR = 0.05
- Critical values adjusted for 3 comparisons

**Secondary family** (exploratory tests): H4, H5, H6, H7
- 4 tests (interaction with regime, compute controls, unfaithfulness, order-flip)
- Correction: Separate Benjamini-Hochberg at FDR = 0.05
- Critical values adjusted for 4 comparisons

**Rationale**:
- Primary family is confirmatory (preregistered hypotheses)
- Secondary family is exploratory (interesting but lower priority)
- Separate corrections prevent secondary tests from inflating primary family false discovery rate

---

## Parser Failure Handling

### Definition

**Parse failure**: Model output cannot be reliably mapped to a valid answer format
- Example: Output is malformed JSON, missing required fields, or contains ambiguous phrasing

### Treatment

**Scoring**: Parse failures are scored as **incorrect** (accuracy = 0)
- Rationale: A model that cannot produce parseable output has failed the task

**Recording**: Parse failure rate is recorded as a secondary outcome per condition

**Exclusion**: Parse failures are **NOT excluded** from analysis (intent-to-treat principle)

**Flagging criterion**: If any condition shows >20% parse failure rate:
- Flag for investigation
- Possible prompt revision (but this would require protocol modification and new preregistration)

---

## Power Analysis

### Assumptions (from design document D6)

**Baseline accuracy**: p₀ = 0.50 (50% correct without scaffolding)

**Minimum detectable effect**: δ = 0.03 (3 percentage points absolute improvement)

**Item random effect SD**: σ_item = 0.5 (logit scale)
- Implies substantial item-to-item difficulty variance

**Model random effect SD**: σ_model = 0.2 (logit scale)
- Implies modest model-to-model capability variance

**Recommended N**: 100 items minimum for 80% power at α = 0.05

**Recommended k_runs**: 3 (balances precision with cost)

### Sensitivity Analysis

**If baseline accuracy differs from 0.50**:
- Higher baseline (e.g., 0.70): Requires larger N to detect same δ (ceiling effects)
- Lower baseline (e.g., 0.30): Easier to detect improvements (more room to grow)

**If item variance is higher than expected**:
- Increase N or use within-item contrasts (McNemar test)

**If effect size is smaller than δ = 0.03**:
- May be underpowered; interpret null results cautiously

---

## Execution Procedure

### Phase 1: Pilot Study

**Objective**: Validate infrastructure, check parse rates, estimate variance

**Configuration**:
- N = 24 items (T2 only, 6 per regime)
- Conditions: 5 factorial (skip reference conditions to save cost)
- k_runs = 1 (or 3 if feasible)
- Models: 1 (Llama-3.3-70B)

**Success criteria**:
- Parse rate >80% across all conditions
- Accuracy variance across conditions is non-zero
- Harness runs without errors
- API latency acceptable (<5 sec per call)

**Deliverables**:
- Pilot results JSON
- Parse rate report
- Go/No-Go decision for confirmatory study

### Phase 2: Confirmatory Study

**Objective**: Full factorial experiment with cross-model validation

**Configuration**:
- N = 350 items (100 T2 + 250 MuSR)
- Conditions: 8 (5 factorial + 3 reference)
- k_runs = 3
- Models: 2 (Llama-3.3-70B + Qwen2.5-72B)

**Execution**:
- Run items in randomized order (prevent order effects in API)
- Log all raw outputs (for post-hoc parse debugging)
- Monitor parse rates in real-time (halt if <80% on any condition)

**Deliverables**:
- Complete results JSON (all items × conditions × runs × models)
- Summary statistics (accuracy, parse rates, costs per condition)
- Data archive for reproducibility

### Phase 3: Cross-Family Check (Optional)

**Objective**: Validate main findings on proprietary frontier model

**Configuration**:
- N = 100 items (T2 subset)
- Conditions: 5 factorial
- k_runs = 1
- Models: 1 (GPT-4o-mini)

**Purpose**: Check if D main effect replicates on different model family

---

## Decision Criteria (Continue / Modify / Kill)

### CONTINUE to Full Study

**If pilot shows**:
- Parse rate >80% on all conditions
- Accuracy variance across conditions is non-zero (SD > 0.05)
- Infrastructure works end-to-end without errors
- Cost estimates align with budget

**Action**: Proceed to confirmatory study with full N

### MODIFY Protocol

**If pilot shows**:
- Parse rate 50-80% on some conditions
  - **Action**: Revise prompts to improve parseability (requires re-piloting)
- Accuracy near ceiling (>90%) or floor (<20%)
  - **Action**: Adjust item difficulty, consider different datasets
- High item variance (item SD > 1.0 on logit scale)
  - **Action**: Increase N to maintain power
- Unexpected interactions or patterns
  - **Action**: Add exploratory conditions or analyses (update preregistration)

### KILL Study

**If pilot shows**:
- Parse rate <50% on any condition after prompt revision attempts
  - **Reason**: Fundamental mismatch between prompts and model capabilities
- All conditions produce identical accuracy (SD < 0.02)
  - **Reason**: No variance to explain; factors have zero effect
- Infrastructure failures (API rate limits, cost overruns, technical issues)
  - **Reason**: Study not feasible with current resources

---

## Preregistration and Transparency

### Preregistration Plan

**Platform**: OSF (Open Science Framework) or AsPredicted

**Contents**:
- This protocol document (markdown or PDF export)
- PROTOCOL.lock.json (hashes of all code and prompts)
- Dataset specifications (T2 generator parameters, MuSR subset criteria)

**Timing**: BEFORE any confirmatory data collection
- Pilot is exempt (exploratory phase)
- Confirmatory study must be preregistered

**Public/Private**: Preregister as private initially; make public upon paper submission

### Deviation Protocol

**If deviations from preregistration are necessary**:
1. Document the deviation and rationale in protocol/DEVIATIONS.md
2. Mark analyses as "post-hoc" or "exploratory" in results
3. Do NOT present post-hoc analyses as confirmatory

**Acceptable deviations**:
- Bug fixes in parsing code (if bug is discovered mid-study)
- Increased N (more data is fine; less is not)
- Additional exploratory analyses (clearly labeled)

**Unacceptable deviations without re-preregistration**:
- Changing hypothesis definitions
- Excluding items based on results
- Changing statistical models after seeing data

---

## Reproducibility Measures

### Code Versioning

**Repository**: Git-tracked, with tagged releases for each phase
- Tag: `pilot-v1.0` before pilot execution
- Tag: `confirmatory-v1.0` before confirmatory execution

**Locked dependencies**: requirements.txt or pyproject.toml with pinned versions

### Prompt Locking

**Mechanism**: protocol/freeze_protocol.py computes SHA-256 hashes of all prompt templates

**Storage**: PROTOCOL.lock.json records hashes + timestamp

**Verification**: Before analysis, re-compute hashes and compare to locked values

### Data Archiving

**Raw outputs**: Store all model responses (even parse failures) in JSON format

**Processed data**: Store parsed results with parser version metadata

**Archive location**: Cloud storage (e.g., OSF, Zenodo) for long-term access

---

## Ethical Considerations

### No Human Subjects

This study involves only synthetic data (T2) and fictional scenarios (MuSR). No human participants, no IRB required.

### Model Usage

All models used within their intended use cases (reasoning, QA). No attempts to elicit harmful outputs.

### Data Privacy

No personally identifiable information (PII) in datasets. API providers' data retention policies reviewed (see model_access_plan.md).

### Environmental Impact

Total compute: ~23,100 API calls (~46M tokens). Carbon footprint minimal compared to training large models. Using shared inference infrastructure (Together AI, OpenAI) rather than dedicated GPU resources.

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Protocol finalization | 1 day | This document + review |
| Preregistration | 1 day | OSF submission |
| Pilot execution | 1-2 days | Infrastructure ready |
| Pilot analysis | 1 day | Pilot data collected |
| Go/No-Go decision | 1 day | Pilot results reviewed |
| Confirmatory execution | 3-5 days | Pilot success |
| Data analysis | 3-5 days | Confirmatory data collected |
| Cross-family check | 1 day | (Optional) Main analysis complete |
| **Total** | **2-3 weeks** | (Excluding paper writing) |

---

## Appendix: Outcome Variable Definitions

### Accuracy (Primary)

```python
def compute_accuracy(model_answer: str, gold_answer: str) -> int:
    """
    Returns 1 if model_answer matches gold_answer after normalization, else 0.
    """
    norm_model = normalize_answer(model_answer)
    norm_gold = normalize_answer(gold_answer)
    return int(norm_model == norm_gold)
```

### Parse Failure Rate

```python
def compute_parse_failure_rate(results: list) -> float:
    """
    Returns proportion of items where parser failed to extract an answer.
    """
    failures = sum(1 for r in results if r['parsed_answer'] is None)
    return failures / len(results)
```

### Calibration (ECE)

```python
def compute_ece(predictions: list, actuals: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: mean absolute difference between
    predicted confidence and empirical accuracy across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if bin_mask.sum() > 0:
            bin_conf = predictions[bin_mask].mean()
            bin_acc = actuals[bin_mask].mean()
            ece += abs(bin_conf - bin_acc) * bin_mask.sum()
    return ece / len(predictions)
```

### Matrix Faithfulness

```python
def compute_matrix_faithfulness(model_output: dict) -> int:
    """
    For ACH conditions (D=1), check if final answer matches
    least-disconfirmed hypothesis from model's own matrix.
    Returns 1 if faithful, 0 if unfaithful, None if no matrix.
    """
    if 'ach_matrix' not in model_output:
        return None

    matrix = model_output['ach_matrix']
    least_disconf = min(matrix, key=lambda h: h['inconsistency_count'])
    final_answer = model_output['final_answer']

    return int(normalize_answer(least_disconf['hypothesis']) ==
               normalize_answer(final_answer))
```

---

## Version History

- **v1.0.0** (2026-08-22): Initial protocol for ETD-ACH factorial experiment (preregistration-ready)

---

## Acknowledgments

This protocol builds on:
- ACH methodology from Richards Heuer's "Psychology of Intelligence Analysis"
- Factorial design principles from Montgomery "Design and Analysis of Experiments"
- LLM evaluation best practices from BIG-Bench and HELM benchmarks

---

**END OF PROTOCOL**
