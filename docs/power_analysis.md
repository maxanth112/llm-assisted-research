# Power Analysis for ETD-ACH Factorial Experiment

**Document version**: 2.0 (Phase A.1 rewrite per AMENDMENT-002)
**Last updated**: 2026-08-27
**Simulation script**: `analysis/power_simulation.py`

## 1. Design Summary

### 1.1 Conditions

The experiment uses 5 conditions (not a full 2x2x2 factorial):

| Condition | E | T | D | Description |
|-----------|---|---|---|-------------|
| 000 | 0 | 0 | 0 | Baseline: no enumeration, no trajectory, no deconfounding |
| 100 | 1 | 0 | 0 | Enumerate-only |
| 110 | 1 | 1 | 0 | Enumerate + trajectory |
| 101 | 1 | 0 | 1 | Enumerate + deconfounding (ACH) |
| 111 | 1 | 1 | 1 | Full scaffold: enumerate + trajectory + deconfounding |

E=0 cells with T=1 or D=1 are **incoherent** (cannot tabulate or deconfound
without first enumerating hypotheses) and are excluded.

### 1.2 Estimable Contrasts

| # | Contrast | Operationalization | Type |
|---|----------|-------------------|------|
| 1 | Enumeration | 100 vs 000 | Independent (different condition families) |
| 2 | T\|E=1 | mean(110, 111) vs mean(100, 101) | Paired within-item |
| 3 | D\|E=1 | mean(101, 111) vs mean(100, 110) | Paired within-item |
| 4 | T×D\|E=1 | (111 − 110) − (101 − 100) | Paired within-item |

**Primary confirmatory contrast:** #3 (D|E=1) — does ACH deconfounding
improve accuracy over enumerate-only, conditional on enumeration being active?

### 1.3 Within-Item Design

Each item appears in all 5 conditions (across separate runs). For the E=1
contrasts (#2-4), each item contributes paired outcomes in conditions 100,
110, 101, 111. This pairing removes item-difficulty variance and enables
the McNemar test.

### 1.4 Regimes

Items come from 4 regimes. The primary contrast (D|E=1) is most meaningful
for adversarial regimes where deconfounding has diagnostic value:

| Regime | Role in primary analysis |
|--------|------------------------|
| CLEAN | Informational (deconfounding may not help when evidence is clean) |
| DECOY | **Primary**: deconfounding should help distinguish diagnostic from non-diagnostic evidence |
| CONFLICT | **Primary**: deconfounding should help resolve source conflicts |
| INSUFFICIENT | Informational (tests uncertainty acknowledgment, not deconfounding) |

### 1.5 Model as Fixed Effect

With 2-3 model families, model is a fixed effect (insufficient levels for
random effect estimation). Power analysis is conducted per-model and
results reported model-by-model. No generalization claim is made beyond
tested models.

## 2. Statistical Tests

### 2.1 Primary: McNemar Test (Paired Within-Item)

For the D|E=1 contrast, each item contributes a paired binary outcome:

- D=0 outcome: average of (item under 100, item under 110) → binarize
- D=1 outcome: average of (item under 101, item under 111) → binarize

With k_runs per condition, each cell has k_runs binary outcomes. The
per-item D=0 and D=1 scores are the mean accuracy across their respective
cells and runs.

**McNemar statistic:**
```
χ² = (n₁₀ - n₀₁)² / (n₁₀ + n₀₁)
```

Where:
- n₁₀: items correct under D=1 but incorrect under D=0
- n₀₁: items correct under D=0 but incorrect under D=1

**Advantages:**
- Exploits within-item pairing (removes σ_item from denominator)
- Exact binomial version available for small N
- More powerful than independent-sample tests when item variance is large

### 2.2 Confirmatory: Mixed-Effects Logistic Regression

```
logit(P(correct)) ~ T * D + regime + model + (1 | item)
```

This is the pre-registered confirmatory model (AMENDMENT-002 §4.2).
The McNemar test is a robustness check that does not assume the logistic
link.

## 3. Power Simulation

### 3.1 Data-Generating Process

For each simulation replicate:

1. Draw item random effects: u_j ~ N(0, σ²_item), j = 1...N
2. For each item j in each E=1 condition (100, 110, 101, 111):
   - Compute logit(p_j) = μ + α_T·T + α_D·D + β_TD·T·D + u_j
   - Draw k_runs Bernoulli outcomes with probability sigmoid(logit(p_j))
3. Compute per-item D=0 accuracy (mean of 100 and 110 runs) and D=1
   accuracy (mean of 101 and 111 runs)
4. Binarize: item "correct under D=d" iff mean accuracy > 0.5
5. Run McNemar test
6. Record whether p < α

Power = proportion of replicates where p < α.

### 3.2 Parameter Grid

| Parameter | Values | Notes |
|-----------|--------|-------|
| N (items) | 50, 100, 150, 200, 300 | Per-regime or total (specified per row) |
| k_runs | 1, 3, 5 | Runs per condition per item |
| σ_item | 0.3, 0.5, 0.8 | Item random effect SD (logit scale) |
| p₀ | 0.35, 0.50 | Baseline accuracy (E=1, T=0, D=0) |
| δ_D | 0.03, 0.05, 0.07, 0.10 | D effect on probability scale |
| α | 0.05 | Significance level |
| n_sims | 2000 | Replicates per configuration |

### 3.3 Results

*To be computed when model-inference credentials are confirmed and the
experiment design is finalized. The simulation script
(`analysis/power_simulation.py`) implements the data-generating process
described above.*

**Placeholder recommendations (to be replaced by simulation results):**

| Scenario | N_items | k_runs | Detectable effect |
|----------|---------|--------|-------------------|
| Conservative (σ=0.8, p₀=0.35) | 200 | 3 | ~7pp at 80% power |
| Moderate (σ=0.5, p₀=0.50) | 200 | 3 | ~5pp at 80% power |
| Optimistic (σ=0.3, p₀=0.50) | 200 | 3 | ~3pp at 80% power |

These estimates will be replaced by exact simulation results when
`analysis/power_simulation.py` is updated to the Phase A.1 specification.

## 4. Minimum Detectable Effect (MDE)

The MDE is the smallest effect size detectable at 80% power given the
chosen design parameters (N, k_runs, σ_item).

**Single MDE claim:** The MDE depends on the design configuration. We
do NOT claim a single fixed MDE across all scenarios. Instead, we:

1. Report MDE as a function of (N, k_runs, σ_item) in a table
2. State the MDE for the CHOSEN design configuration
3. Pre-register the chosen configuration before data collection

**Regime-specific power:** The primary analysis focuses on adversarial
regimes (DECOY + CONFLICT). If these have different item variance than
CLEAN, the MDE may differ by regime. Report per-regime power estimates.

## 5. Multiple Comparison Procedure

### 5.1 Primary Family (3 tests, Bonferroni-Holm at α=0.05)

1. Enumeration contrast (100 vs 000)
2. T|E=1: mean(110, 111) vs mean(100, 101)
3. D|E=1: mean(101, 111) vs mean(100, 110)

### 5.2 Secondary Family (exploratory, BH FDR at α=0.05)

1. T×D|E=1 interaction
2. D×regime interaction (does D effect vary by regime?)
3. Per-regime D estimates (CLEAN, DECOY, CONFLICT, INSUFFICIENT)

### 5.3 Rationale

- Primary family uses Bonferroni-Holm (strong FWER control, 3 tests)
- Secondary family uses BH FDR (less conservative, appropriate for
  exploratory analyses)

## 6. Interpretation Constraints

1. **Null findings**: A non-significant D|E=1 effect is informative ONLY IF
   the 95% CI excludes effects above the pre-registered MDE.
   Report: "We can rule out D effects larger than X pp with 95% confidence."

2. **Regime interactions**: If D|E=1 is significant only in DECOY but not
   CONFLICT (or vice versa), this is an exploratory finding, not a
   pre-registered prediction.

3. **Model specificity**: Results apply to tested model families only.
   No generalization claim is made.

## 7. References

- McNemar, Q. (1947). Note on the sampling error of the difference
  between correlated proportions or percentages. *Psychometrika*, 12(2),
  153-157.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
  rate. *JRSS-B*, 57(1), 289-300.
- Holm, S. (1979). A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
