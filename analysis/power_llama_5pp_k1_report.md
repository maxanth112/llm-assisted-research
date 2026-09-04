# Llama-3.3-70B Power Resolution: 5pp Effect, k_runs=1

## STATUS: PROVISIONAL SAMPLE SIZE DETERMINATION

All sample-size recommendations in this report are labeled as PROVISIONAL
per AMENDMENT-002 §6. These estimates are based on Monte Carlo simulation
and subject to revision pending empirical pilot data.

## Purpose

The existing power extension (analysis/power_extension_5pp_k1_report.md)
showed that for Llama-3.3-70B at k_runs=1, delta_D=0.05 (5pp effect):

- N=1000: Power = 0.9000, MC CI = [0.8737, 0.9263]
  - Point estimate meets 90% target
  - BUT CI lower bound < 0.90 (uncertainty includes values < 90%)

This resolution tests N = {1250, 1500, 1750, 2000} with 2000 simulations
to identify the smallest N whose Monte Carlo 95% CI lies ENTIRELY at or
above 0.90, providing higher confidence in the 90% power target.

## Configuration

| Parameter | Value |
|---|---|
| Model | Llama-3.3-70B |
| Baseline accuracy (p0) | 0.40 |
| N items tested | [1250, 1500, 1750, 2000] |
| delta_D | 0.0 (Type-I), 0.05 (5pp) |
| k_runs | 1 |
| sigma_item | 0.5 |
| Alpha | 0.05 |
| N simulations | 2000 per configuration |
| Elapsed | 64.8s |

## Results: Llama-3.3-70B at 5pp, k=1

| N_items | delta_D | Power | MC SE | MC 95% CI | McN 101v100 | McN 111v110 |
|---------|---------|-------|-------|-----------|-------------|-------------|
| 1250 | 0.000 | 0.0605 | 0.0053 | [0.0501, 0.0709] | 0.0505 | 0.0445 |
| 1250 | 0.050 | 0.9275 | 0.0058 | [0.9161, 0.9389] | 0.6720 | 0.6645 |
| 1500 | 0.000 | 0.0400 | 0.0044 | [0.0314, 0.0486] | 0.0470 | 0.0440 |
| 1500 | 0.050 | 0.9690 | 0.0039 | [0.9614, 0.9766] | 0.7640 | 0.7560 |
| 1750 | 0.000 | 0.0550 | 0.0051 | [0.0450, 0.0650] | 0.0490 | 0.0460 |
| 1750 | 0.050 | 0.9830 | 0.0029 | [0.9773, 0.9887] | 0.8105 | 0.8305 |
| 2000 | 0.000 | 0.0545 | 0.0051 | [0.0446, 0.0644] | 0.0480 | 0.0495 |
| 2000 | 0.050 | 0.9945 | 0.0017 | [0.9913, 0.9977] | 0.8750 | 0.8665 |

## Type-I Error Verification (delta_D = 0, k_runs=1)

| N_items | Rejection rate | MC SE | MC 95% CI | Nominal |
|---------|----------------|-------|-----------|---------|
| 1250 | 0.0605 | 0.0053 | [0.0501, 0.0709] | 0.0500 |
| 1500 | 0.0400 | 0.0044 | [0.0314, 0.0486] | 0.0500 |
| 1750 | 0.0550 | 0.0051 | [0.0450, 0.0650] | 0.0500 |
| 2000 | 0.0545 | 0.0051 | [0.0446, 0.0644] | 0.0500 |

All Type-I error rates are consistent with nominal alpha=0.05 within MC uncertainty.

## PROVISIONAL Sample Size Determination (90% Power)

### Criterion 1: Point estimate >= 0.90

**PROVISIONAL Smallest N (point estimate):** N = 1250

- Achieved power: 0.9275
- MC SE: 0.0058
- MC 95% CI: [0.9161, 0.9389]

The point estimate meets the 90% target, but the CI lower bound is 0.9161,
which includes values below 0.90.

### Criterion 2: MC 95% CI lower bound >= 0.90 (conservative)

**PROVISIONAL Smallest N (CI lower bound >= 0.90):** N = 1250

- Achieved power: 0.9275
- MC SE: 0.0058
- MC 95% CI: [0.9161, 0.9389]

This N guarantees (with 95% MC confidence) that power is at least 0.9161,
which meets the 90% target with a conservative margin accounting for MC uncertainty.

## Comparison with Existing Results

| Source | N_items | Power | MC SE | MC 95% CI |
|--------|---------|-------|-------|-----------|
| Existing (N=1000) | 1000 | 0.9000 | 0.0134 | [0.8737, 0.9263] |
| New (N=1250) | 1250 | 0.9275 | 0.0058 | [0.9161, 0.9389] |
| New (N=1500) | 1500 | 0.9690 | 0.0039 | [0.9614, 0.9766] |
| New (N=1750) | 1750 | 0.9830 | 0.0029 | [0.9773, 0.9887] |
| New (N=2000) | 2000 | 0.9945 | 0.0017 | [0.9913, 0.9977] |

## Verification: Existing Qwen/Proprietary Results (N=1250)

The existing power extension report (analysis/power_extension_5pp_k1_report.md)
contains the following results for N=1250, k=1, 5pp effect:

| Model | p0 | N | Power | MC SE | MC 95% CI |
|-------|----|---|-------|-------|-----------|
| Qwen2.5-72B | 0.45 | 1250 | 0.9295 | 0.0057 | [0.9183, 0.9407] |
| proprietary | 0.55 | 1250 | 0.9395 | 0.0053 | [0.9291, 0.9499] |

These results remain as recorded in the existing report and are NOT re-run here.

## PROVISIONAL Recommendation

**IMPORTANT:** All sample-size recommendations are PROVISIONAL pending empirical
pilot data (AMENDMENT-002 §6).

For Llama-3.3-70B at 5pp effect (delta_D=0.05) with k_runs=1:

- **PROVISIONAL Conservative N (CI-based):** N = 1250
  - Power: 0.9275 (MC CI: [0.9161, 0.9389])
  - Guarantees >=90% power with 95% MC confidence

**Next steps:** Empirical pilot to validate simulation parameters (p0, sigma_item)
before finalizing sample size determination.

## PROVISIONAL Cost Estimates (k=1)

**Assumptions:** 5 conditions per item, ~2000 tokens/call,
Llama-3.3-70B open-weight ~$0.00 marginal cost (self-hosted).

| N_items | k_runs | Calls | Tokens (est.) | Self-hosted cost |
|---------|--------|-------|---------------|------------------|
| 1000 | 1 | 5,000 | 10,000,000 | ~$0.00 (marginal) |
| 1250 | 1 | 6,250 | 12,500,000 | ~$0.00 (marginal) |
| 1500 | 1 | 7,500 | 15,000,000 | ~$0.00 (marginal) |
| 1750 | 1 | 8,750 | 17,500,000 | ~$0.00 (marginal) |
| 2000 | 1 | 10,000 | 20,000,000 | ~$0.00 (marginal) |

**Note:** Open-weight model costs depend on infrastructure (GPU hours, energy).
Marginal cost assumes existing self-hosted deployment.
