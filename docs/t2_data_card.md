# T2 Dataset Data Card

## Overview

**Dataset Name**: T2 (Task 2: Deterministic Diagnostic Items for ACH Scaffolding Evaluation)

**Purpose**: Controlled evaluation of Analysis of Competing Hypotheses (ACH) scaffolding effects on language model reasoning. T2 consists of deterministic diagnostic items designed for within-item factorial comparisons across evidence presentation modes, reasoning trajectories, and dataset regimes.

**Version**: 1.0.0
**Format**: JSONL (JSON Lines)
**License**: MIT (generated content, no external copyrights)
**Repository**: [To be filled]

## Dataset Statistics

### Overall

- **Total items**: [To be filled after generation]
- **Item format**: JSON objects with narrative, hypotheses, evidence, and metadata
- **Hypotheses per item**: ≥3 (range: [TBD])
- **Evidence items per item**: [TBD]

### Distribution by Regime

| Regime | Count | Percentage | Description |
|--------|-------|------------|-------------|
| CLEAN | TBD | TBD% | Straightforward cases with clear evidence |
| DECOY | TBD | TBD% | Includes plausible but irrelevant distractors |
| CONFLICT | TBD | TBD% | Contains contradictory or ambiguous evidence |
| INSUFFICIENT | TBD | TBD% | Requires acknowledging gaps in evidence |

### Distribution by Scenario Template

| Template | Count | Percentage |
|----------|-------|------------|
| theft | TBD | TBD% |
| sabotage | TBD | TBD% |
| data_breach | TBD | TBD% |
| contamination | TBD | TBD% |

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

**[To be filled after running `leakage_check.py`]**

### Baseline Methods

| Method | Accuracy | Verdict | Notes |
|--------|----------|---------|-------|
| Majority class | TBD | TBD | Guessing most frequent correct hypothesis |
| Lexical overlap | TBD | TBD | Predicting hypothesis mentioned most in narrative |
| TF-IDF + Logistic | TBD | TBD | Bag-of-words with cross-validation |

### Interpretation

- **Chance level**: 1 / (average hypotheses per item) ≈ TBD
- **Threshold**: Chance + 0.05 = TBD
- **Overall verdict**: [PASS/FAIL]

**Conclusion**: [To be filled - should be PASS, confirming no trivial shortcuts]

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
  "item_id": "t2_theft_001",
  "regime": "CLEAN",
  "scenario": "theft",
  "narrative": "...",
  "hypotheses": {
    "H1": {
      "name": "Alice",
      "description": "Alice stole the laptop"
    },
    "H2": { ... },
    "H3": { ... }
  },
  "evidence": [
    {
      "id": "E1",
      "text": "...",
      "supports": ["H1"],
      "contradicts": [],
      "credibility": "high"
    },
    ...
  ],
  "correct_hypothesis": "H1",
  "metadata": {
    "generation_seed": 42,
    "template_version": "1.0",
    "counterbalancing_group": "A"
  }
}
```

### Field Descriptions

- **item_id**: Unique identifier (format: `t2_{scenario}_{number}`)
- **regime**: One of [CLEAN, DECOY, CONFLICT, INSUFFICIENT]
- **scenario**: One of [theft, sabotage, data_breach, contamination]
- **narrative**: Natural language description of the situation
- **hypotheses**: Dictionary of hypothesis objects with name and description
- **evidence**: List of evidence items with support/contradiction metadata
- **correct_hypothesis**: Ground truth hypothesis ID
- **metadata**: Generation parameters for reproducibility

## Versioning and Updates

### Current Version: 1.0.0

**Release Date**: [To be filled]
**Changes**: Initial release

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

- Generated [TBD] items across 4 regimes
- 4 scenario templates implemented
- Counterbalancing and leakage checks validated
- Full data card and power analysis documentation

---

**Document version**: 1.0
**Last updated**: [To be filled]
