# T2 Dataset Data Card

## Overview

**Dataset Name**: T2 (Task 2: Deterministic Diagnostic Items for ACH Scaffolding Evaluation)

**Purpose**: Controlled evaluation of Analysis of Competing Hypotheses (ACH) scaffolding effects on language model reasoning. T2 consists of deterministic diagnostic items designed for within-item factorial comparisons across evidence presentation modes, reasoning trajectories, and dataset regimes.

**Version**: 2.0.0 (T2 v2, per AMENDMENT-001)
**Format**: JSONL (JSON Lines)
**License**: MIT (generated content, no external copyrights)
**Repository**: https://github.com/maxanth112/llm-assisted-research

## Dataset Statistics

### Overall

- **Total items**: 6,000 (5,250 train + 750 final-audit)
- **Item format**: JSON objects with narrative, hypotheses, evidence, and metadata
- **Hypotheses per item**: 3-4 (mean: 3.17; INSUFFICIENT items have 4, others have 3)
- **Evidence items per item**: 7-10 (mean: 8.17)

### Distribution by Regime

| Regime | Count | Percentage | Description |
|--------|-------|------------|-------------|
| CLEAN | 2,000 | 33.3% | Straightforward cases with clear evidence |
| DECOY | 2,000 | 33.3% | Includes plausible but irrelevant distractors |
| CONFLICT | 1,000 | 16.7% | Contains contradictory or ambiguous evidence |
| INSUFFICIENT | 1,000 | 16.7% | Requires acknowledging gaps in evidence |

### Distribution by Scenario Template

| Template | Count | Percentage |
|----------|-------|------------|
| theft (alibi + timeline) | 1,500 | 25.0% |
| sabotage (alibi + timeline) | 1,500 | 25.0% |
| data_breach (alibi + timeline) | 1,500 | 25.0% |
| contamination (alibi + timeline) | 1,500 | 25.0% |

**Template families (8):** theft_alibi, theft_timeline, sabotage_alibi, sabotage_timeline,
data_breach_alibi, data_breach_timeline, contamination_alibi, contamination_timeline.
Each family contributes 750 items. The contamination_timeline family is reserved as the
final-audit set; the remaining 7 families form the train/held-out evaluation set.

## Regime Descriptions

### CLEAN

**Characteristics**:
- Evidence clearly and consistently points to one hypothesis
- No significant distractors or contradictions
- Straightforward deductive reasoning sufficient

**Example Structure**:
- Hypothesis A: Supported by 3 pieces of consistent evidence
- Hypothesis B: No supporting evidence
- Hypothesis C: No supporting evidence

**Purpose**: Baseline condition to verify model can perform basic reasoning with ACH structure.

### DECOY

**Characteristics**:
- Includes plausible but ultimately irrelevant alternative hypotheses
- Correct hypothesis has strongest evidence, but decoys have superficial appeal
- Requires distinguishing relevant from irrelevant evidence

**Example Structure**:
- Hypothesis A (correct): Supported by 2 pieces of strong evidence
- Hypothesis B (decoy): Supported by 1 piece of weak/circumstantial evidence
- Hypothesis C (decoy): Mentioned in narrative but lacks evidential support

**Purpose**: Test whether ACH scaffolding helps models avoid surface-level distractors.

### CONFLICT

**Characteristics**:
- Contains apparently contradictory evidence
- Requires resolving conflicts through evidence strength, credibility, or specificity
- Correct hypothesis has preponderance of evidence despite conflicts

**Example Structure**:
- Hypothesis A (correct): Supported by 3 pieces of evidence, contradicted by 1 weak piece
- Hypothesis B: Supported by 1 piece of evidence
- Hypothesis C: No direct evidence

**Purpose**: Test whether ACH scaffolding helps models systematically weigh conflicting evidence.

### INSUFFICIENT

**Characteristics**:
- Evidence is genuinely insufficient to definitively confirm any hypothesis
- Correct answer may be "inconclusive" or requires acknowledging uncertainty
- Tests whether models can recognize epistemic limits

**Example Structure**:
- Hypothesis A: Supported by 1 ambiguous piece of evidence
- Hypothesis B: Supported by 1 ambiguous piece of evidence
- Hypothesis C: No evidence
- Correct response: "Insufficient evidence to determine"

**Purpose**: Test whether ACH scaffolding improves calibration and uncertainty awareness.

## Scenario Templates

### Theft

**Domain**: Property crime investigation
**Typical structure**:
- **Hypotheses**: Multiple suspects who had opportunity to take the item
- **Evidence**: Access logs, witness statements, motive, alibi
- **Variation**: Location (office, warehouse, residence), item type, number of suspects

### Sabotage

**Domain**: Industrial/organizational misconduct
**Typical structure**:
- **Hypotheses**: Multiple actors who could have damaged equipment or processes
- **Evidence**: Timeline, technical knowledge, motive, physical access
- **Variation**: Industry setting, sabotage method, consequence severity

### Data Breach

**Domain**: Cybersecurity incident response
**Typical structure**:
- **Hypotheses**: Multiple potential threat actors or insider threats
- **Evidence**: Log files, access patterns, technical capability, motive
- **Variation**: Attack vector, data type, organization size

### Contamination

**Domain**: Food safety or environmental incident
**Typical structure**:
- **Hypotheses**: Multiple potential contamination sources
- **Evidence**: Timeline, contamination pattern, handling procedures, test results
- **Variation**: Industry (food, pharmaceutical, environmental), contaminant type

## Construction Rules

### Generation Process

1. **Seeded RNG**: All items generated with deterministic random seed for reproducibility
2. **Entity Pools**: Names, locations, and entities drawn from predefined pools to ensure diversity
3. **Evidence Patterns**: Each regime has specific evidence generation rules:
   - CLEAN: Evidence distribution heavily favors one hypothesis
   - DECOY: Evidence includes red herrings with superficial plausibility
   - CONFLICT: Evidence includes contradictions requiring resolution
   - INSUFFICIENT: Evidence is sparse or ambiguous across hypotheses

### Counterbalancing

To mitigate spurious correlations and lexical leakage:

1. **Name Shuffling**: Entity names randomized across roles (guilty vs innocent)
2. **Position Shuffling**: Hypothesis order randomized in presentation
3. **Adversarial Permutations**: For each item, create variants where:
   - Correct hypothesis is in different positions (first, middle, last)
   - Entity names are rotated across hypotheses
   - Evidence order is permuted

**Result**: Correct answer cannot be predicted by:
- Hypothesis position
- Entity name patterns
- Evidence ordering
- Narrative structure alone

## Leakage Check Results

**Evaluated:** 2026-08-24 per AMENDMENT-001 acceptance criteria.
**Script:** `analysis/run_leakage_eval.py`
**Full results:** `analysis/leakage_results_v2.json`

### Baseline Methods (Template-Held-Out Evaluation)

| # | Method | Accuracy | 95% Wilson CI | Verdict | Notes |
|---|--------|----------|---------------|---------|-------|
| 1 | Majority class | 0.167 | [0.157, 0.177] | PASS | Guessing most frequent answer |
| 2 | Label position | 0.277 | [0.265, 0.290] | PASS | Always predict first hypothesis |
| 3 | Mention count | 0.277 | [0.265, 0.290] | PASS | Predict most-mentioned suspect |
| 4 | Evidence count | 0.277 | [0.265, 0.290] | PASS | Predict suspect in most evidence items |
| 5 | Lexical overlap | 0.277 | [0.265, 0.290] | PASS | Hypothesis-evidence word overlap |
| 6 | TF-IDF word | 0.435 | [0.422, 0.449] | FAIL | Word-level TF-IDF + LogReg |
| 7 | TF-IDF char | 0.433 | [0.420, 0.447] | FAIL | Char-level TF-IDF + LogReg |
| 8 | Length feature | 0.445 | [0.432, 0.459] | FAIL | Evidence length per candidate |
| 9 | Polarity feature | 0.442 | [0.429, 0.456] | FAIL | Support/contradict counts |
| 10 | Positional feature | 0.448 | [0.434, 0.461] | FAIL | Evidence order features |
| 11 | Combined shallow | 0.447 | [0.434, 0.461] | FAIL | All structured features |

### Interpretation

- **Chance level**: 1 / 3.17 = 0.316
- **Threshold**: 0.316 + 0.05 = 0.366
- **Overall verdict**: **FAIL**

### Per-Regime Analysis

| Regime | Heuristic baselines (1-5) | Classifier baselines (6-11) | Notes |
|--------|---------------------------|----------------------------|-------|
| CLEAN | All PASS | All PASS | No leakage detected |
| DECOY | All PASS | All PASS | No leakage detected |
| CONFLICT | All PASS | 4/6 PASS, 2/6 marginal FAIL | Length and combined: CI upper 0.391 vs threshold 0.383 |
| INSUFFICIENT | 4/5 PASS | All FAIL (100% accuracy) | Structural leak: answer format categorically different |

### Diagnosis

**Primary failure:** The INSUFFICIENT regime uses a categorically different gold
answer ("Cannot be determined from available evidence") that is trivially
distinguishable from named-suspect answers. Any classifier achieves 100%
accuracy on INSUFFICIENT items, inflating aggregate scores.

**Secondary failure:** Minor length-based leakage in CONFLICT items (2 baselines
exceed threshold by ~0.8 percentage points).

**For CLEAN and DECOY regimes, all 11 baselines pass.** The T2 v2
counterbalancing invariants (name-frequency equalization, evidence-count parity,
polarity balance, counterfactual pairs) work as intended for the primary
reasoning regimes.

**Recommendation:** Use CLEAN and DECOY items for primary ACH scaffolding
analysis. Report results separately by regime. See AMENDMENT-001 for full
details.

## Known Limitations

### Template-Based Generation

**Limitation**: Items are generated from templates, creating systematic lexical patterns.

**Specific Risks**:
1. **Guilty Suspect Bias**: The correct hypothesis is necessarily mentioned in evidence, potentially creating a "most mentioned = guilty" heuristic
2. **Narrative Structure**: Template-based generation may produce predictable sentence structures
3. **Entity Pool Constraints**: Limited entity names may create spurious name-outcome correlations

**Mitigations**:
1. **Adversarial Permutations**: Randomize entity-to-role assignments across items
2. **Within-Item Design**: Primary DV is accuracy *difference* between D=0 and D=1 conditions *for the same item*
3. **Leakage Checks**: Automated detection of trivial shortcuts (see Leakage Check Results)

### Validity Constraints

**What T2 IS**:
- A controlled diagnostic set for measuring ACH scaffolding effects
- Designed for *within-item factorial comparisons* (D=0 vs D=1 on same item)
- Appropriate for testing if deconfounding mechanisms improve reasoning

**What T2 IS NOT**:
- A validated benchmark for general reasoning ability
- Representative of real-world ACH complexity
- Suitable for absolute accuracy comparisons across models

### External Validity

**Generalization Limits**:
1. **Synthetic Nature**: Templates do not capture full complexity of real intelligence analysis
2. **Domain Coverage**: Limited to 4 scenario types (theft, sabotage, data breach, contamination)
3. **Deterministic Answers**: Real ACH often involves genuine uncertainty; T2 has ground truth

**Appropriate Use**: T2 is designed for *mechanistic* evaluation of scaffolding effects, not as a proxy for real-world performance.

## Prohibited Interpretations

### DO NOT

1. ❌ **Cite absolute accuracy as model capability**: "Model X achieves 73% on T2" is meaningless without context
   - **Why**: T2 difficulty is arbitrary and template-based
   - **What instead**: Report *comparative* findings: "Model X improves by 5pp with ACH scaffolding (p<0.01)"

2. ❌ **Compare T2 scores across different models**: "Model A (68%) outperforms Model B (62%)"
   - **Why**: Absolute scores reflect template artifacts, not capability
   - **What instead**: Run factorial experiment for each model; compare *effect sizes*

3. ❌ **Use T2 as a benchmark for leaderboard**: "State-of-the-art on T2: 81%"
   - **Why**: T2 is a diagnostic set, not a validated benchmark
   - **What instead**: Use T2 for controlled experiments, not ranking

4. ❌ **Claim T2 measures "reasoning ability"**: "Model X has strong ACH reasoning"
   - **Why**: T2 tests response to scaffolding, not general reasoning
   - **What instead**: "Model X shows sensitivity to ACH deconfounding in controlled settings"

### DO

1. ✅ **Report within-item effect sizes**: "Full ACH scaffolding improved accuracy by 3.2pp (95% CI: [1.1, 5.3]) compared to enumerate-only"

2. ✅ **Conduct factorial experiments**: "We tested 2×2×2 design (E×T×D) on 150 T2 items with 3 runs per condition"

3. ✅ **Acknowledge limitations**: "T2 is a synthetic diagnostic set; findings may not generalize to real-world ACH tasks"

4. ✅ **Use for mechanistic insights**: "T2 results suggest deconfounding mechanisms reduce decoy susceptibility"

## Intended Use

### Primary Use Case

**Within-item factorial comparison of scaffolding conditions**:

1. For each item in T2:
   - Run under D=0 (enumerate-only) condition
   - Run under D=1 (full ACH with deconfounding) condition
   - (Optionally vary E and T factors)

2. Compute item-level accuracy difference: Δ_j = Acc(D=1) - Acc(D=0)

3. Aggregate across items: Mean(Δ_j) with 95% CI

4. Interpret: Does ACH scaffolding systematically improve accuracy?

### Secondary Use Cases

1. **Regime-specific effects**: Does ACH help more in DECOY vs CLEAN regimes?
2. **Mechanism isolation**: Which ACH components (matrix, deconfounding, diagnosticity) drive effects?
3. **Robustness checks**: Do effects replicate across scenario templates?

### Statistical Approach

See [`docs/power_analysis.md`](./power_analysis.md) for:
- Recommended sample sizes
- Multiple comparison procedures
- Interpretation of null findings

## Data Schema

### Item Structure

```json
{
  "id": "t2v2_clean_theft_alibi_0000",
  "regime": "CLEAN",
  "narrative": "At Manufacturing Solutions Inc, a valuable confidential document set was stolen...",
  "question": "Based on the available evidence, who is most likely responsible for the theft?",
  "hypotheses": [
    "Blake Rivera is responsible",
    "Indigo Taylor is responsible",
    "Finley Brooks is responsible"
  ],
  "evidence": [
    {
      "id": "E001",
      "content": "Blake Rivera claims to have been at Building C...",
      "supports": [],
      "contradicts": ["Blake Rivera"],
      "diagnostic_value": "high"
    }
  ],
  "gold_answer": "Finley Brooks is responsible",
  "gold_reasoning": "...",
  "source_precedence_rule": null,
  "metadata": {
    "template": "theft_alibi",
    "guilty_suspect": "Finley Brooks",
    "guilty_position": 2,
    "n_suspects": 3,
    "n_evidence": 9,
    "mechanism": "alibi_invalidation",
    "name_frequencies": {"Blake Rivera": 6, "Indigo Taylor": 6, "Finley Brooks": 6}
  }
}
```

### Field Descriptions

- **id**: Unique identifier (format: `t2v2_{regime}_{scenario}_{template}_{number}`)
- **regime**: One of [CLEAN, DECOY, CONFLICT, INSUFFICIENT]
- **narrative**: Natural language description of the investigation scenario
- **question**: The question posed to the model
- **hypotheses**: List of hypothesis strings (e.g., "Name is responsible")
- **evidence**: List of evidence objects with content, supports, contradicts, diagnostic_value
- **gold_answer**: Ground truth answer string (matches one hypothesis or "Cannot be determined...")
- **gold_reasoning**: Explanation of why the gold answer is correct
- **source_precedence_rule**: For CONFLICT items, the rule for resolving source conflicts; null otherwise
- **metadata**: Generation parameters including template, suspect details, and name frequencies

## Versioning and Updates

### Current Version: 2.0.0

**Release Date**: 2026-08-24
**Changes**: T2 v2 with counterbalancing invariants per AMENDMENT-001

### Future Updates

Planned enhancements:
- Additional scenario templates (fraud, espionage, accident investigation)
- Multilingual variants (if applicable)
- Increased item count per regime
- Refined counterbalancing strategies

**Versioning Policy**: Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes to schema or regime definitions
- MINOR: New items, scenarios, or regimes added
- PATCH: Bug fixes, typo corrections, metadata updates

## Citation

If you use the T2 dataset in research, please cite:

```
[To be filled - suggested format:]

@misc{t2_dataset_2026,
  title={T2: Deterministic Diagnostic Items for ACH Scaffolding Evaluation},
  author={[Authors]},
  year={2026},
  version={1.0.0},
  url={[Repository URL]}
}
```

## Contact and Contributions

**Maintainers**: [To be filled]
**Issues**: [To be filled - repository issue tracker]
**Contributions**: [To be filled - contribution guidelines]

## Changelog

### Version 1.0.0 (Initial Release)

- T2 v1 generator with basic counterbalancing
- Invalidated by leakage battery (see AMENDMENT-001 for details)

### Version 2.0.0 (T2 v2, AMENDMENT-001)

- Generated 6,000 items across 4 regimes, 8 template families
- Relational reasoning via structural rules (alibi-chain invalidation, temporal-sequence inconsistency, source-credibility cascade)
- Counterbalancing invariants: name-frequency equalization, evidence-count parity, polarity balance, length matching, positional uniformity
- Counterfactual minimal pairs with >85% token overlap
- 11-baseline leakage battery: overall FAIL (INSUFFICIENT structural leak), CLEAN/DECOY PASS all 11 baselines

---

**Document version**: 2.0
**Last updated**: 2026-08-24
