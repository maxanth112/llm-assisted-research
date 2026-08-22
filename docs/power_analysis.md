# Power Analysis for ETD-ACH Factorial Experiment

## 1. Statistical Model Specification

The ETD-ACH factorial experiment uses a mixed-effects logistic regression model to account for item-level variability:

```
Y_ij ~ Bernoulli(p_ij)
logit(p_ij) = μ + α_E·E + α_T·T + α_D·D + β_ET·E·T + u_j
u_j ~ N(0, σ²_item)
```

Where:
- **Y_ij**: Binary outcome (correct/incorrect) for item *j* in condition *i*
- **E**: Evidence presentation (0 = enumerate-only, 1 = full ACH scaffolding)
- **T**: Trajectory (0 = direct answer, 1 = chain-of-thought)
- **D**: Dataset regime (0 = enumerate-only, 1 = full ACH with deconfounding)
- **u_j**: Item random effect capturing item-specific difficulty
- **σ_item**: Standard deviation of item random effects

### Model Rationale

- **Logistic link**: Appropriate for binary accuracy outcomes
- **Random item effects**: Account for heterogeneity in item difficulty; critical because we use deterministic diagnostic items rather than a large sample from a population
- **Factorial structure**: Allows testing of main effects and interactions
- **Within-item design**: Each item appears in all conditions (between runs), maximizing power

## 2. Primary Contrast: D Main Effect

**Research Question**: Does full ACH scaffolding with deconfounding (D=1) improve accuracy compared to enumerate-only scaffolding (D=0)?

**Statistical Test**: Two-sample t-test comparing mean accuracy across D=1 vs D=0 conditions, aggregating across all factorial cells and items.

**Null Hypothesis**: H₀: α_D = 0 (no difference in accuracy between dataset regimes)

**Alternative Hypothesis**: H₁: α_D > 0 (full ACH improves accuracy)

This is the **primary confirmatory hypothesis** for the experiment.

## 3. Secondary Contrasts

While the D main effect is primary, we also examine:

1. **E Main Effect** (α_E): Does evidence presentation mode affect accuracy?
2. **T Main Effect** (α_T): Does reasoning trajectory (CoT vs direct) affect accuracy?
3. **E × T Interaction** (β_ET): Does the effect of evidence depend on trajectory type?
4. **Higher-order interactions**: E×D, T×D, E×T×D

These are **exploratory analyses** and will be corrected for multiple comparisons.

## 4. Effect Sizes

We sweep the following effect sizes for the D main effect (absolute accuracy difference on probability scale):

| Effect Size | Description | Percentage Points |
|-------------|-------------|------------------|
| 0.005 | Very small | 0.5pp |
| 0.01 | Small | 1pp |
| 0.02 | Moderate | 2pp |
| 0.03 | Medium | 3pp |
| 0.05 | Large | 5pp |

**Interpretation**: These represent the absolute increase in accuracy (e.g., from 50% to 53% is a 3pp effect).

**Minimal Detectable Effect**: Based on power simulations, we aim to design the study to detect a **3pp effect with 80% power**. This is considered the smallest practically meaningful effect for ACH scaffolding evaluation.

## 5. Design Parameters

The power simulation sweeps the following design parameters:

### Sample Size (n_items)
- Values: 25, 50, 100, 150, 200, 250
- Definition: Number of distinct T2 items used in the experiment

### Runs per Condition (k_runs)
- Values: 1, 3, 5
- Definition: Number of independent runs per factorial cell per item
- Note: k_runs=1 provides minimal variance; k_runs≥3 recommended for robustness

### Baseline Accuracy (baseline_acc)
- Values: 0.4, 0.5, 0.6
- Definition: Expected accuracy in the reference condition (E=0, T=0, D=0)
- Rationale: Models uncertainty in true model capability

### Item Variance (σ_item)
- Values: 0.3, 0.5, 0.8
- Definition: Standard deviation of item random effects on logit scale
- Interpretation:
  - σ=0.3: Low heterogeneity (items similar difficulty)
  - σ=0.5: Moderate heterogeneity (typical for diagnostic items)
  - σ=0.8: High heterogeneity (some items much harder than others)

## 6. Key Findings

**[To be filled after running `power_simulation.py`]**

### Power Curves

The following table summarizes the minimum n_items required to achieve 80% power for each effect size:

| Effect Size | Min n_items | Configuration |
|-------------|-------------|---------------|
| 0.5pp | TBD | TBD |
| 1pp | TBD | TBD |
| 2pp | TBD | TBD |
| 3pp | TBD | TBD |
| 5pp | TBD | TBD |

### Sensitivity Analysis

- **Impact of baseline accuracy**: [TBD]
- **Impact of item variance**: [TBD]
- **Impact of k_runs**: [TBD]

### Smallest Detectable Effect

At the recommended design (n_items=TBD, k_runs=TBD):
- **80% power to detect**: TBD pp effect
- **95% power to detect**: TBD pp effect
- **Precision for null findings**: 95% CI excludes effects larger than TBD pp

## 7. Recommendations

### Recommended Design

Based on the power simulations:

- **n_items**: [TBD] items from T2 dataset
- **k_runs**: [TBD] independent runs per condition
- **Factorial design**: Full 2×2×2 (8 cells)
- **Total model calls**: n_items × k_runs × 8 = [TBD]

### Interpretation Constraints

1. **Confirmatory vs Exploratory**:
   - Primary contrast (D main effect) is confirmatory
   - All other contrasts are exploratory and require multiple comparison correction

2. **Null Findings**:
   - A non-significant D main effect is **only informative if** the 95% confidence interval excludes effects above the minimal detectable threshold (3pp)
   - Report precision: "We can rule out effects larger than X pp with 95% confidence"

3. **Trade-offs**:
   - **Clean vs Adversarial Items**: More adversarial items (DECOY, CONFLICT) increase σ_item, reducing power; however, they provide stronger validity for the deconfounding mechanism
   - **Single vs Multiple Model Families**: Testing multiple model families reduces power for any single model but increases generalizability
   - Recommendation: Run power analysis separately for each model family; pool only if effects are homogeneous

### Statistical Practices

1. **Pre-registration**: Register the analysis plan, including:
   - Primary contrast (D main effect)
   - Minimal detectable effect (3pp)
   - Multiple comparison procedure (see below)

2. **Robustness Checks**:
   - McNemar test for paired within-item contrasts (more powerful)
   - Mixed-effects logistic regression with item random effects (confirmatory)
   - Subgroup analyses by regime (CLEAN, DECOY, CONFLICT, INSUFFICIENT)

3. **Reporting**:
   - Always report effect sizes with 95% CIs
   - Never report only p-values
   - For null findings, report precision (maximum excluded effect)

## 8. Multiple Comparison Procedure

Given multiple secondary contrasts, we use the **Benjamini-Hochberg procedure** to control False Discovery Rate (FDR):

### Procedure

1. Compute p-values for all contrasts: p₁, p₂, ..., p_m
2. Rank p-values: p(₁) ≤ p(₂) ≤ ... ≤ p(m)
3. Find largest k such that p(k) ≤ (k/m) × α
4. Reject H₀ for all contrasts with p ≤ p(k)

### Settings

- **α (FDR level)**: 0.05
- **Number of contrasts**: 7 (E, T, D, E×T, E×D, T×D, E×T×D)
- **Primary contrast**: D main effect is tested at α=0.05 without correction (pre-specified)
- **Secondary contrasts**: Remaining 6 contrasts use Benjamini-Hochberg at FDR=0.05

### Rationale

- FDR control is less conservative than Bonferroni but still protects against false discoveries
- Primary contrast exemption is justified by pre-registration
- BH procedure has good power when most tests are non-null (as expected in factorial designs)

## 9. McNemar Test for Paired Contrasts

For within-item comparisons (same item in two conditions), the **McNemar test** is more powerful than an independent t-test.

### When to Use

- Comparing D=0 vs D=1 **within the same item**
- Each item contributes a pair: (Y_j | D=0, E=e, T=t) vs (Y_j | D=1, E=e, T=t)

### Test Statistic

For paired binary outcomes:

```
χ² = (n₁₀ - n₀₁)² / (n₁₀ + n₀₁)
```

Where:
- n₁₀: number of items correct in D=1 but incorrect in D=0
- n₀₁: number of items correct in D=0 but incorrect in D=1

### Advantages

- Exploits within-item pairing for increased power
- Requires fewer items than independent t-test
- Robust to item difficulty heterogeneity

### Recommendation

- **Primary analysis**: McNemar test for D=0 vs D=1 comparison
- **Robustness check**: Mixed-effects logistic regression (accounts for random effects explicitly)

## 10. Limitations and Assumptions

### Model Assumptions

1. **Independence**: Outcomes are independent across items (within and between runs)
   - Potential violation: Model may learn patterns across items in a session
   - Mitigation: Randomize item order; use fresh model instances per run

2. **Logistic link**: True data-generating process follows logistic function
   - Robustness: t-tests on proportions are robust to link misspecification

3. **Normality of random effects**: Item effects are normally distributed
   - Robustness: Central Limit Theorem applies for large n_items (≥50)

### External Validity

- **Item representativeness**: T2 items are synthetic and template-based; findings may not generalize to real-world ACH tasks
- **Model generalization**: Power estimates assume stable model performance across items; adaptive or meta-learning models may violate this

### Design Constraints

- **Fixed factorial**: Power calculations assume full 2×2×2 design; partial factorials reduce power
- **Item selection**: Power estimates assume random sampling of items; adversarial selection may increase σ_item

## 11. References

- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.

---

**Document version**: 1.0
**Last updated**: [To be filled]
**Contact**: [To be filled]
