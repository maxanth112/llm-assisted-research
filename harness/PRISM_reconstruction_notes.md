# PRISM Reconstruction Notes

This document records all assumptions and design decisions made when reconstructing the PRISM multi-agent reasoning system based on the ACL 2026 paper.

## Critical Context

The PRISM system described in the ACL 2026 paper **did not release code or complete prompts**. This implementation is a best-effort reconstruction based on the paper's descriptions, examples, and Appendix C.

## Reconstruction Assumptions

### 1. Prompt Text Approximation
- **Assumption**: Prompt text is approximate and reconstructed from Appendix C descriptions
- **Rationale**: The paper provides high-level descriptions of agent roles and instructions, but not the exact prompt text
- **Implementation**: We reconstructed prompts to match the spirit and structure described in the paper
- **Uncertainty**: High - actual prompts may have differed significantly in wording, emphasis, or additional instructions

### 2. Agent Personas
- **Assumption**: Agent personas match paper descriptions exactly
  - A1 = Forensic analyst with 20 years of experience
  - A2 = Criminal investigator using M.O.M.A framework
  - A3 = Intelligence analyst trained in ACH
  - A4 = Judge presiding over a murder case
- **Rationale**: These personas are explicitly stated in the paper
- **Uncertainty**: Low - these are direct quotes or close paraphrases from the paper

### 3. Output Schemas
- **Assumption**: JSON output schemas are inferred from paper examples
- **Rationale**: The paper shows example outputs but does not specify exact JSON structure
- **Implementation**: We designed schemas that capture the information shown in examples
- **Uncertainty**: Medium - field names and nesting structure may differ from original implementation

### 4. Evidence Classification Tiers (A1)
- **Assumption**: We use HIGH_VALUE/MEDIUM_VALUE/LOW_VALUE/IRRELEVANT categories
- **Paper Description**: Core (8-10), Circumstantial (5-7), Background (2-4), Noise (0-1) on a 0-10 scale
- **Rationale**: We simplified to four categorical tiers for clearer distinction and easier parsing
- **Uncertainty**: Medium - original may have used numeric scores or different category names
- **Impact**: Should not affect core functionality, as the goal is to prioritize evidence

### 5. MOMA Ratings
- **Assumption**: We use STRONG/MODERATE/WEAK/ABSENT ratings
- **Paper Description**: The paper mentions "strong/moderate/weak/unknown" ratings
- **Rationale**: We use ABSENT instead of UNKNOWN for clarity (evidence is either present or absent)
- **Uncertainty**: Low - this is a minor terminology adjustment
- **Impact**: Minimal - semantic meaning is preserved

### 6. ACH Matrix Codes
- **Assumption**: We use C (consistent), I (inconsistent), N (neutral) codes and HIGH/MEDIUM/LOW diagnosticity
- **Rationale**: These codes are explicitly mentioned in the paper's ACH description
- **Uncertainty**: Low - this matches the paper's description
- **Implementation Note**: The matrix is represented as a list of evidence items, each with consistency codes for all hypotheses

### 7. Confidence Scaling
- **Assumption**: Judge confidence is on a 0-100 scale
- **Rationale**: The paper shows percentage-based confidence scores in examples
- **Uncertainty**: Low - this appears consistent with paper examples
- **Implementation**: All templates use 0-100 integer scale

### 8. Debate Variant Out of Scope
- **Assumption**: We do NOT implement the "Structured Debate" ensemble variant
- **Rationale**: The debate variant is mentioned but not detailed in the paper
- **Uncertainty**: N/A - this is a deliberate scope limitation
- **Impact**: Our implementation focuses on the sequential pipeline variant only

### 9. Sequential Pipeline Order
- **Assumption**: Agent execution order is A1 → A2 → A3 → A4
- **Rationale**: The paper describes a sequential pipeline where:
  - A1 filters and categorizes evidence
  - A2 performs MOMA analysis on suspects
  - A3 conducts ACH matrix analysis
  - A4 delivers final verdict as judge
- **Uncertainty**: Low - this is clearly described in the paper
- **Implementation**: Orchestrator should execute agents in this order, passing outputs forward

### 10. Temperature and Sampling Parameters
- **Assumption**: We default to temperature=0.0 for reproducibility
- **Rationale**: The paper does not specify temperature or sampling parameters
- **Uncertainty**: High - original implementation may have used different temperature settings
- **Implementation Decision**: We use temperature=0.0 for deterministic, reproducible results
- **Future Work**: Could experiment with temperature as a hyperparameter

### 11. Parsing Rules
- **Assumption**: All JSON parsers are our own implementation
- **Rationale**: No code or detailed parsing rules were released with the paper
- **Uncertainty**: High - original implementation may have used different parsing strategies
- **Implementation Considerations**:
  - We expect well-formed JSON responses
  - Should implement robust error handling for malformed JSON
  - May need fallback strategies for parsing failures

### 12. Token Budgets
- **Assumption**: We use max_tokens=2048 per agent call
- **Rationale**: The paper does not specify token budgets or length constraints
- **Uncertainty**: High - original implementation may have used different limits
- **Implementation Decision**: 2048 tokens provides room for detailed reasoning while staying within typical context limits
- **Future Work**: Could tune as a hyperparameter based on task complexity

## Information Flow Between Agents

Based on paper descriptions:

1. **A1 → A2**: A1's evidence categorization (HIGH_VALUE/MEDIUM_VALUE evidence) is passed to A2
2. **A2 → A3**: A2's MOMA analysis and suspect rankings inform A3's hypothesis set
3. **A3 → A4**: A3's full ACH matrix and inconsistency counts are primary inputs to A4
4. **A4 Output**: Final verdict with confidence and reasoning

**Uncertainty**: Medium - exact information passing mechanism not fully detailed in paper

## Known Differences from Paper

1. **Exact prompt wording**: Our prompts are reconstructions, not originals
2. **JSON schema details**: Field names and structure may differ
3. **Error handling**: Our implementation includes error handling not discussed in paper
4. **Evidence extraction**: Paper does not detail how evidence is extracted from narratives - we assume this is done via LLM parsing
5. **Hypothesis generation**: Not clear if hypotheses are pre-specified or generated - we assume they correspond to answer choices

## Validation Strategy

To validate our reconstruction:

1. **Qualitative**: Review agent outputs to ensure they match the style and content shown in paper examples
2. **Structural**: Verify that outputs contain the key information types described (MOMA ratings, ACH matrix, etc.)
3. **Comparative**: If possible, compare results on same cases to paper's reported outputs
4. **Ablation**: Test individual agents to ensure they perform their specialized roles

## Open Questions

1. How were hypotheses generated or specified in the original implementation?
2. What was the exact token budget for each agent?
3. How were ties handled (e.g., multiple hypotheses with same inconsistency count)?
4. Were there any prompt engineering techniques (few-shot examples, etc.) not mentioned in the paper?
5. How were malformed or incomplete agent outputs handled?
6. What post-processing was applied to agent outputs before passing to next agent?

## References

- PRISM paper: ACL 2026 (full citation to be added when paper details are finalized)
- Appendix C: Agent prompt descriptions
- Paper Section 3: PRISM system architecture

## Version History

- v1.0.0 (2026-08-22): Initial reconstruction based on ACL 2026 paper
