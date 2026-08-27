# Leakage Audit Joint-Gate Power Simulation

**Baselines:** 11
**Regimes:** 4
**Gate:** Wilson 95% CI upper <= chance + alpha
**Alpha (margin):** 0.05
**Chance (v3 universal 4-option):** 0.25 for all regimes

## Power Table: P(joint gate passes | H0)

| N/regime | Total N | Marginal P(PASS) | Joint (rho=0.0) | Joint (rho=0.3) | Joint (rho=0.6) |
|---|---|---|---|---|---|---|
| 500 | 2000 | 0.6770 | 0.0000 | 0.0184 | 0.1317 |
| 750 | 3000 | 0.8640 | 0.0016 | 0.1360 | 0.3619 |
| 1000 | 4000 | 0.9413 | 0.0690 | 0.3452 | 0.5823 |
| 1500 | 6000 | 0.9917 | 0.6915 | 0.7994 | 0.8862 |
| 2000 | 8000 | 0.9989 | 0.9515 | 0.9610 | 0.9765 |
| 2500 | 10000 | 0.9999 | 0.9956 | 0.9959 | 0.9973 |
| 3000 | 12000 | 1.0000 | 0.9996 | 0.9996 | 0.9997 |
| 4000 | 16000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5000 | 20000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7500 | 30000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 10000 | 40000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Minimum N for Target Joint Power

| Target | rho=0.0 (N/regime) | rho=0.3 (N/regime) | rho=0.6 (N/regime) |
|---|---|---|---|
| 0.80 | 2000 (total=8000) | 2000 (total=8000) | 1500 (total=6000) |
| 0.90 | 2000 (total=8000) | 2000 (total=8000) | 2000 (total=8000) |
| 0.95 | 2000 (total=8000) | 2000 (total=8000) | 2000 (total=8000) |

## Notes

- **Marginal P(PASS)** = probability one baseline on one regime passes
- **Joint** = probability ALL (baselines x regimes) pass simultaneously
- Under independence (rho=0), joint = marginal^K where K = baselines x regimes
- Positive correlation (rho>0) increases joint probability (failures cluster)
- Only the audit split is modeled; the held-out split passes with ~1.0 probability
- These results assume v3 universal 4-option design (chance=0.25 for all regimes)
