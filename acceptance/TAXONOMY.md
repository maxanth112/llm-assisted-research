# Template Taxonomy — Honest Assessment (v4, Corrected)

## Current Structure

The T2 v3.2.2 generator uses **ONE controlled conjunction mechanism** instantiated
across **EIGHT surface templates** nested within **FOUR incident domains**.

### Domains and Templates

| Domain | Template A | Template B |
|--------|-----------|-----------|
| Theft | theft_alibi | theft_timeline |
| Sabotage | sabotage_alibi | sabotage_timeline |
| Data Breach | data_breach_alibi | data_breach_timeline |
| Contamination | contamination_alibi | contamination_timeline |

### What Is Shared (ALL 8 templates)

Every template uses the identical logical structure:

1. **Conjunction decision rule**: "A suspect is responsible iff BOTH criteria met"
2. **Criterion A keywords**: "Level-2" (positive) vs "Level-1" (negative)
3. **Criterion B keywords**: "flagged" (positive) vs "cleared" (negative)
4. **Evidence structure**: 3 access + 3 criterion A + 3 criterion B + 1 neutral = 10 slots
5. **Permutation mechanism**: Single transposition in criterion B (S1 <-> S2)
6. **Option structure**: 3 suspect hypotheses + 1 abstention hypothesis
7. **Name-frequency balance**: Each suspect mentioned exactly 3 times
8. **Name-length matching**: Suspects drawn from same character-length stratum

### What Varies Across Templates

Only **surface-level noun phrases** in the evidence templates differ:

- Criterion A: "security clearance for the restricted zone" vs
  "access authorization for the secured vault" vs
  "technical certification for the critical system" vs ...
- Criterion B: "anomaly detection log for the incident period" vs
  "inventory discrepancy report for the audit window" vs ...
- Narrative: theft vs sabotage vs data breach vs contamination framing

These substitutions do NOT change the logical structure, the keywords that
the oracle parses (Level-2/Level-1, flagged/cleared), or the conjunction mechanism.

### Effective Structural Diversity

**One (1) reasoning structure**, not eight independent ones.

The 8 templates are surface variants of a single conjunction-over-binary-criteria
rule. A classifier that learns to detect "Level-2" and "flagged" from one template
applies identically to all others. The leave-one-template-out (LOTO) evaluation
is therefore a **diagnostic of surface-token robustness**, not evidence that
the pipeline produces independent reasoning families.

### CONFLICT Regime

The CONFLICT regime uses a different mechanism (source-precedence resolution
based on automated log vs unverified witness report) but still shares:
- Same suspect name pool and name-length matching
- Same evidence slot count (10)
- Same option structure (3 suspects + 1 abstention)

CONFLICT is a second logical structure, making the total: **2 structures**
(conjunction + source-precedence), not 8 or 32.

### Prior Claims (Corrected)

- **Prior v3 TAXONOMY_32_families.md**: Claimed 32 independent structures
  (8 templates x 4 variants). This was misleading — all 32 share one
  conjunction mechanism with surface noun substitutions.
- **Prior v3 TAXONOMY.md**: Acknowledged 8 templates share mechanism but
  did not explicitly state the effective structural count is 1.

### Implications for Evaluation

1. **LOTO is a diagnostic, not a population-level gate**: With only 1 effective
   structure, leaving out one surface template does not test generalization
   across reasoning types.
2. **Bootstrap confidence intervals are wide**: 8 resampling units (templates)
   provide limited statistical power for the binary leakage question.
3. **The real leakage gate requires a structurally diverse corpus**: At minimum
   >=500 items per regime with >=5 genuinely distinct logical structures.

### Path to Genuine Structural Diversity

To create truly independent reasoning families, future extensions should vary:

1. **Logical connective**: AND (current), OR, XOR, threshold (>=k of n criteria),
   negation, conditional chains
2. **Evidence format**: Narrative (current), tabular, temporal sequences,
   quantitative measurements, nested hierarchies
3. **Decision framework**: Binary conjunction (current), multi-step reasoning,
   probabilistic weighting, contradictory-source resolution (partially via CONFLICT)
4. **Scale parameters**: Number of suspects (currently fixed at 3), number of
   criteria (currently fixed at 2), number of evidence slots (currently fixed at 10)
5. **Answer structure**: Currently 3 suspects + 1 abstention. Could vary to
   4-5 suspects, multiple valid answers, or graded confidence levels.
