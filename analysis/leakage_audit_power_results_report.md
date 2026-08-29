# Leakage Audit Joint-Gate Power Simulation

P(ALL baselines x regimes pass simultaneously | true accuracy = chance)

**Baselines:** 11
**Regimes:** 4 (CLEAN, DECOY, CONFLICT, INSUFFICIENT)
**Gate:** Wilson 95% CI upper <= 0.30
**Chance:** 0.25 (v3 universal 4-option)
**Margin:** 0.05
**Total cells:** 11 x 4 = 44

**Correlation model:** block: baselines within regime share rho, baselines across regimes independent
**rho=0 method:** exact binomial (closed-form)
**rho>0 method:** Monte Carlo (200000 sims per config)
**Seed:** 42

## Power Table: P(gate passes | true accuracy = chance)

| N/regime | Total N | Marginal P(PASS) | Joint (rho=0.0) | Joint (rho=0.3) | Joint (rho=0.6) |
|---|---|---|---|---|---|
| 100 | 400 | 0.211435 | 0.000000 | 0.000000 | 0.000000 |
| 125 | 500 | 0.221383 | 0.000000 | 0.000000 | 0.000000 |
| 150 | 600 | 0.227441 | 0.000000 | 0.000000 | 0.000005 |
| 175 | 700 | 0.288683 | 0.000000 | 0.000000 | 0.000000 |
| 200 | 800 | 0.345804 | 0.000000 | 0.000000 | 0.000005 |
| 225 | 900 | 0.398449 | 0.000000 | 0.000000 | 0.000025 |
| 250 | 1000 | 0.389420 | 0.000000 | 0.000000 | 0.000010 |
| 275 | 1100 | 0.435368 | 0.000000 | 0.000000 | 0.000045 |
| 300 | 1200 | 0.477848 | 0.000000 | 0.000000 | 0.000170 |
| 325 | 1300 | 0.517029 | 0.000000 | 0.000000 | 0.000300 |
| 350 | 1400 | 0.553120 | 0.000000 | 0.000010 | 0.000590 |
| 375 | 1500 | 0.586347 | 0.000000 | 0.000010 | 0.001070 |
| 400 | 1600 | 0.616932 | 0.000000 | 0.000035 | 0.001850 |
| 425 | 1700 | 0.602854 | 0.000000 | 0.000025 | 0.001340 |
| 450 | 1800 | 0.631052 | 0.000000 | 0.000015 | 0.002120 |
| 475 | 1900 | 0.657162 | 0.000000 | 0.000095 | 0.003495 |
| 500 | 2000 | 0.681337 | 0.000000 | 0.000220 | 0.005040 |
| 550 | 2200 | 0.724454 | 0.000001 | 0.000595 | 0.010200 |
| 600 | 2400 | 0.761459 | 0.000006 | 0.001810 | 0.018685 |
| 650 | 2600 | 0.817853 | 0.000144 | 0.006720 | 0.044140 |
| 700 | 2800 | 0.842270 | 0.000525 | 0.012485 | 0.063975 |
| 750 | 3000 | 0.863235 | 0.001548 | 0.021135 | 0.088290 |
| 800 | 3200 | 0.881272 | 0.003844 | 0.033985 | 0.117335 |
| 850 | 3400 | 0.896817 | 0.008298 | 0.049955 | 0.148685 |
| 900 | 3600 | 0.921889 | 0.027917 | 0.096925 | 0.223840 |
| 950 | 3800 | 0.932042 | 0.045203 | 0.126720 | 0.264420 |
| 1000 | 4000 | 0.940820 | 0.068278 | 0.161760 | 0.305565 |
| 1100 | 4400 | 0.961115 | 0.174625 | 0.284235 | 0.437520 |
| 1200 | 4800 | 0.970371 | 0.266236 | 0.372065 | 0.519705 |
| 1300 | 5200 | 0.980531 | 0.421024 | 0.510205 | 0.633645 |
| 1400 | 5600 | 0.987227 | 0.568002 | 0.632955 | 0.728115 |
| 1500 | 6000 | 0.991629 | 0.690819 | 0.732710 | 0.803485 |
| 1600 | 6400 | 0.994518 | 0.785155 | 0.811595 | 0.860310 |
| 1700 | 6800 | 0.995779 | 0.830159 | 0.848385 | 0.887070 |
| 1800 | 7200 | 0.997232 | 0.885176 | 0.896335 | 0.920900 |
| 1900 | 7600 | 0.998186 | 0.923203 | 0.928615 | 0.946135 |
| 1925 | 7700 | 0.998559 | 0.938541 | 0.942125 | 0.956145 |
| 1950 | 7800 | 0.998649 | 0.942244 | 0.946090 | 0.957960 |
| 1975 | 7900 | 0.998733 | 0.945727 | 0.949090 | 0.960690 |
| 2000 | 8000 | 0.998811 | 0.949003 | 0.950730 | 0.962585 |
| 2025 | 8100 | 0.999057 | 0.959353 | 0.961680 | 0.970090 |
| 2050 | 8200 | 0.999116 | 0.961813 | 0.963880 | 0.971595 |
| 2075 | 8300 | 0.999170 | 0.964125 | 0.966395 | 0.972835 |
| 2100 | 8400 | 0.999221 | 0.966297 | 0.968225 | 0.975285 |
| 2125 | 8500 | 0.999383 | 0.973218 | 0.974730 | 0.979800 |
| 2150 | 8600 | 0.999421 | 0.974840 | 0.976570 | 0.981145 |
| 2500 | 10000 | 0.999880 | 0.994719 | 0.994785 | 0.995485 |
| 3000 | 12000 | 0.999986 | 0.999364 | 0.999320 | 0.999485 |

## First Tested N Achieving Target P(gate passes)

**Note:** These are the smallest N values *in the tested grid* that
achieve the target. The true mathematical minimum may lie between grid
points. Values should be interpreted as sufficient, not necessary.

| Target | rho=0.0 (N/regime) | rho=0.3 (N/regime) | rho=0.6 (N/regime) |
|---|---|---|---|
| 0.80 | 1700 (total=6800) | 1600 (total=6400) | 1500 (total=6000) |
| 0.90 | 1900 (total=7600) | 1900 (total=7600) | 1800 (total=7200) |
| 0.95 | 2025 (total=8100) | 2000 (total=8000) | 1925 (total=7700) |

## FROZEN Audit-Size Decision

**FROZEN:** 2,000 items per regime, 8,000 total.

- Target: P(overall leakage gate passes | every baseline's true accuracy = chance) >= 0.90
- Design basis: rho=0 (independence) as conservative default
- At N=2,000/regime under rho=0: joint pass probability ~0.949
- This meets the >=0.90 target but does NOT claim 0.95
- Structured within-regime correlation (rho>0) is reported as sensitivity analysis only

## Aggregate Gate Cells

The per-regime gate is modeled above with 44 cells (11 baselines x 4 regimes).
The full gate additionally includes aggregate cells (11 baselines on all items pooled).
At N=8,000 total (2,000/regime), the marginal pass probability for each aggregate
baseline (N=8,000 at chance=0.25, margin=0.05) is effectively 1.0:

- Marginal P(PASS) for one aggregate baseline at N=8,000: 1.000000
- Joint P(all 11 aggregate baselines pass): 1.000000
- Including aggregate cells does not alter the selected design point (2,000/regime)

## Notes

- **Marginal P(PASS)**: probability that ONE baseline on ONE regime passes
- **Joint**: probability that ALL (baselines x regimes) pass simultaneously
- rho=0 (independence): exact binomial computation, no Monte Carlo
- Independence is the CONSERVATIVE default for the joint gate — positive
  correlation increases joint pass probability (failures cluster)
- rho>0: block correlation as sensitivity analysis only
- Only the audit split is modeled; held-out split passes with ~1.0
- v3 universal 4-option design: chance=0.25 for all regimes
