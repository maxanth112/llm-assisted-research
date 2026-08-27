#!/usr/bin/env python3
"""
Joint-gate power simulation for the leakage equivalence audit.

Phase A.2 rewrite.

Computes P(ALL baselines x regimes pass simultaneously | true accuracy = chance)
as a function of per-regime N.

Key improvements over Phase A.1:
  * Exact binomial for rho=0 (closed-form, no Monte Carlo)
  * Fixed integer seeds (no hash(regime))
  * Fine N grid (step=25 from 100 to 3000)
  * Block correlation structure (baselines within regime correlated,
    baselines across regimes independent)
  * Reports P(gate passes | true chance), not "H0"
  * Aggregate cells: ALL baselines x ALL regimes simultaneously

Gate: Wilson 95% CI upper bound <= chance + 0.05

Usage:
    python analysis/leakage_audit_power.py [--quick] [--output FILE]
"""

import json
import math
import sys
import time
import numpy as np
from typing import Tuple, Dict, List
from scipy.stats import norm, binom


def wilson_ci_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score 95% CI upper bound."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return min(1.0, c + m)


def exact_marginal_pass_prob(n: int, chance: float, margin: float = 0.05) -> float:
    """Exact P(PASS for one cell | true accuracy = chance).

    PASS = Wilson CI upper <= chance + margin.
    Under true accuracy = chance, k ~ Binomial(n, chance).
    Sum P(k) over all k where Wilson CI upper <= threshold.

    This is exact (no Monte Carlo) via scipy.stats.binom.
    """
    threshold = chance + margin
    pass_prob = 0.0
    for k in range(n + 1):
        ci_upper = wilson_ci_upper(k, n)
        if ci_upper <= threshold:
            pass_prob += binom.pmf(k, n, chance)
    return pass_prob


def exact_joint_pass_prob_independent(
    marginal_probs: List[float],
) -> float:
    """Joint pass probability assuming independence: product of marginals."""
    p = 1.0
    for pp in marginal_probs:
        p *= pp
    return p


def joint_pass_prob_block_correlated(
    regime_marginals: Dict[str, float],
    n_baselines: int,
    rho_within: float,
    n_sims: int = 200000,
    seed: int = 42,
) -> float:
    """Monte Carlo estimate of joint pass probability with BLOCK correlation.

    Block structure: baselines WITHIN the same regime share correlation
    rho_within, baselines ACROSS different regimes are independent.

    For each regime r with marginal pass probability p_r:
      latent_ij = sqrt(rho)*W_r + sqrt(1-rho)*e_ij
      cell (r,j) passes iff Phi(latent_ij) < p_r

    Joint = all cells across all regimes pass.
    """
    if rho_within <= 0:
        # All independent
        marginal_list = []
        for regime, p_r in regime_marginals.items():
            marginal_list.extend([p_r] * n_baselines)
        return exact_joint_pass_prob_independent(marginal_list)

    rng = np.random.RandomState(seed)
    regimes = sorted(regime_marginals.keys())
    n_regimes = len(regimes)

    # Precompute thresholds: Phi^{-1}(p_r) for each regime
    thresholds = {}
    for r in regimes:
        p_r = regime_marginals[r]
        thresholds[r] = norm.ppf(p_r) if p_r < 1.0 else 10.0

    sqrt_rho = math.sqrt(rho_within)
    sqrt_1mrho = math.sqrt(1 - rho_within)

    n_pass = 0
    for _ in range(n_sims):
        all_pass = True
        for r in regimes:
            W_r = rng.randn()  # shared within regime
            e = rng.randn(n_baselines)  # independent per baseline
            z = sqrt_rho * W_r + sqrt_1mrho * e
            if not np.all(z <= thresholds[r]):
                all_pass = False
                break  # early exit
        if all_pass:
            n_pass += 1

    return n_pass / n_sims


def compute_power_table(
    n_per_regime_range: List[int],
    n_baselines: int = 11,
    chance: float = 0.25,
    regime_names: List[str] = None,
    rho_values: List[float] = None,
    margin: float = 0.05,
    n_sims_joint: int = 200000,
    seed: int = 42,
) -> Dict:
    """Compute the joint-gate power table.

    For each per-regime N, compute:
    1. Exact marginal pass probability (one cell)
    2. Exact joint pass probability under independence (rho=0)
    3. Monte Carlo joint pass probability under block correlation (rho > 0)

    Args:
        n_per_regime_range: list of per-regime N values
        n_baselines: number of baselines
        chance: chance level (0.25 for universal 4-option)
        regime_names: list of regime names
        rho_values: within-regime baseline correlation values
        margin: equivalence margin (0.05)
        n_sims_joint: MC sims for correlated joint probability
        seed: master seed
    """
    if regime_names is None:
        regime_names = ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]
    if rho_values is None:
        rho_values = [0.0, 0.3, 0.6]

    n_regimes = len(regime_names)
    results = []
    total = len(n_per_regime_range)

    for idx, n_per_regime in enumerate(n_per_regime_range):
        print(f"  N={n_per_regime} ({idx+1}/{total})...", file=sys.stderr)

        # Exact marginal pass probability (same for all regimes at same chance)
        marginal = exact_marginal_pass_prob(n_per_regime, chance, margin)

        # Per-regime marginals (all same under universal 4-option)
        regime_marginals = {r: marginal for r in regime_names}

        # Joint pass probability for each rho
        rho_results = {}
        for rho in rho_values:
            if rho == 0:
                # Exact: product of marginals across all cells
                n_cells = n_baselines * n_regimes
                joint_p = marginal ** n_cells
            else:
                # Monte Carlo with block correlation
                # Use fixed integer seed offset (no hash)
                rho_seed = seed + int(rho * 1000) + idx * 7
                joint_p = joint_pass_prob_block_correlated(
                    regime_marginals, n_baselines, rho,
                    n_sims=n_sims_joint, seed=rho_seed,
                )
            rho_results[f"rho_{rho:.1f}"] = round(joint_p, 6)

        results.append({
            "n_per_regime": n_per_regime,
            "total_audit_n": n_per_regime * n_regimes,
            "n_cells": n_baselines * n_regimes,
            "marginal_pass_prob": round(marginal, 6),
            "joint_pass_prob": rho_results,
        })

    # Find minimum N for each target power and rho
    targets = [0.80, 0.90, 0.95]
    min_n_table = {}
    for target in targets:
        min_n_table[f"target_{target:.2f}"] = {}
        for rho in rho_values:
            rho_key = f"rho_{rho:.1f}"
            found = None
            for r in results:
                if r["joint_pass_prob"][rho_key] >= target:
                    found = r
                    break
            if found:
                min_n_table[f"target_{target:.2f}"][rho_key] = {
                    "n_per_regime": found["n_per_regime"],
                    "total_audit_n": found["total_audit_n"],
                    "joint_pass_prob": found["joint_pass_prob"][rho_key],
                }
            else:
                min_n_table[f"target_{target:.2f}"][rho_key] = {
                    "n_per_regime": None,
                    "message": f"No N in range achieved P(gate passes) >= {target}"
                }

    return {
        "description": "Joint-gate power: P(ALL baselines x regimes pass | true accuracy = chance)",
        "design": {
            "n_baselines": n_baselines,
            "n_regimes": n_regimes,
            "regime_names": regime_names,
            "chance": chance,
            "margin": margin,
            "gate": f"Wilson 95% CI upper <= {chance + margin:.2f}",
        },
        "correlation_model": {
            "rho_values": rho_values,
            "structure": "block: baselines within regime share rho, "
                         "baselines across regimes independent",
        },
        "power_table": results,
        "minimum_n": min_n_table,
        "simulation_params": {
            "rho_0_method": "exact binomial (closed-form)",
            "rho_gt0_method": f"Monte Carlo ({n_sims_joint} sims per config)",
            "seed": seed,
        },
    }


def format_report(results: Dict) -> str:
    """Format results as markdown report."""
    lines = []
    lines.append("# Leakage Audit Joint-Gate Power Simulation")
    lines.append("")
    lines.append("P(ALL baselines x regimes pass simultaneously | true accuracy = chance)")
    lines.append("")
    lines.append(f"**Baselines:** {results['design']['n_baselines']}")
    lines.append(f"**Regimes:** {results['design']['n_regimes']} "
                 f"({', '.join(results['design']['regime_names'])})")
    lines.append(f"**Gate:** {results['design']['gate']}")
    lines.append(f"**Chance:** {results['design']['chance']} (v3 universal 4-option)")
    lines.append(f"**Margin:** {results['design']['margin']}")
    lines.append(f"**Total cells:** {results['design']['n_baselines']} x "
                 f"{results['design']['n_regimes']} = "
                 f"{results['design']['n_baselines'] * results['design']['n_regimes']}")
    lines.append("")
    lines.append(f"**Correlation model:** {results['correlation_model']['structure']}")
    lines.append(f"**rho=0 method:** {results['simulation_params']['rho_0_method']}")
    lines.append(f"**rho>0 method:** {results['simulation_params']['rho_gt0_method']}")
    lines.append(f"**Seed:** {results['simulation_params']['seed']}")
    lines.append("")

    # Power table
    lines.append("## Power Table: P(gate passes | true accuracy = chance)")
    lines.append("")
    rho_values = results['correlation_model']['rho_values']
    rho_keys = [f"rho_{rho:.1f}" for rho in rho_values]
    header = "| N/regime | Total N | Marginal P(PASS) | " + " | ".join(
        f"Joint (rho={rho:.1f})" for rho in rho_values
    ) + " |"
    sep = "|" + "|".join(["---"] * (3 + len(rho_keys))) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results['power_table']:
        marginal = r['marginal_pass_prob']
        joint_vals = [f"{r['joint_pass_prob'][k]:.6f}" for k in rho_keys]
        lines.append(
            f"| {r['n_per_regime']} | {r['total_audit_n']} | {marginal:.6f} | "
            + " | ".join(joint_vals) + " |"
        )

    lines.append("")

    # Minimum N table
    lines.append("## Minimum N for Target P(gate passes)")
    lines.append("")
    lines.append("| Target | " + " | ".join(
        f"rho={rho:.1f} (N/regime)" for rho in rho_values
    ) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(rho_keys))) + "|")

    for target in [0.80, 0.90, 0.95]:
        tkey = f"target_{target:.2f}"
        cells = []
        for rho_key in rho_keys:
            entry = results['minimum_n'][tkey][rho_key]
            if entry.get('n_per_regime') is not None:
                cells.append(f"{entry['n_per_regime']} (total={entry['total_audit_n']})")
            else:
                cells.append("N/A")
        lines.append(f"| {target:.2f} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Marginal P(PASS)**: probability that ONE baseline on ONE regime passes")
    lines.append("- **Joint**: probability that ALL (baselines x regimes) pass simultaneously")
    lines.append("- rho=0 (independence): exact binomial computation, no Monte Carlo")
    lines.append("- rho>0: block correlation (within-regime baselines correlated, "
                 "across-regime independent)")
    lines.append("- Higher correlation increases joint probability (failures cluster)")
    lines.append("- Only the audit split is modeled; held-out split passes with ~1.0")
    lines.append("- v3 universal 4-option design: chance=0.25 for all regimes")
    lines.append("")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint-gate power simulation")
    parser.add_argument("--output", default="analysis/leakage_audit_power_results.json")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()

    t0 = time.time()

    if args.quick:
        # Fine grid but fewer MC sims
        n_range = list(range(100, 1001, 100)) + [1500, 2000]
        n_sims_j = 50000
    else:
        # Fine grid from 100 to 3000
        n_range = list(range(100, 501, 25)) + list(range(550, 1001, 50)) + \
                  list(range(1100, 2001, 100)) + [2500, 3000]
        n_sims_j = 200000

    print(f"Running joint-gate power simulation ({len(n_range)} N values)...",
          file=sys.stderr)
    results = compute_power_table(
        n_per_regime_range=n_range,
        n_baselines=11,
        chance=0.25,
        rho_values=[0.0, 0.3, 0.6],
        n_sims_joint=n_sims_j,
        seed=42,
    )

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}", file=sys.stderr)

    report = format_report(results)
    report_path = args.output.replace('.json', '_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # Print summary
    print("\n" + report, file=sys.stderr)


if __name__ == "__main__":
    main()
