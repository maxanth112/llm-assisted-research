#!/usr/bin/env python3
"""
Joint-gate power simulation for the leakage equivalence audit.

Computes P(ALL baselines pass simultaneously | true accuracy = chance)
as a function of per-regime N, accounting for correlation between baselines.

This addresses Work Item 3 from Phase A.1: the original AMENDMENT-002 section 3.2
powered each baseline individually at 0.90, but the overall gate requires ALL
baselines x regimes x splits to pass simultaneously. Under independence, the
joint pass probability is the product of marginal pass probabilities, which
can be far below 0.90 even when each individual baseline achieves 0.90.

Key concepts:
  - Gate: Wilson 95% CI upper bound <= chance + 0.05
  - Under H0 (true acc = chance), PASS probability depends on N
  - Joint gate: all K cells in {baseline x regime x split} must PASS
  - With B=11 baselines, R=4 regimes, S=2 splits: K = 11*4*2 = 88 cells
    (but not all independent — shared predictions, correlated baselines)
  - Sensitivity analysis for baseline correlation rho = {0, 0.3, 0.6}

Usage:
    python analysis/leakage_audit_power.py [--output FILE]
"""

import json
import math
import sys
import time
import numpy as np
from typing import Tuple, Dict, List


def wilson_ci_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score 95% CI upper bound."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return min(1.0, c + m)


def per_baseline_pass_prob(n: int, chance: float, alpha: float = 0.05,
                           n_sims: int = 50000, seed: int = 42) -> float:
    """Monte Carlo estimate of P(PASS | true accuracy = chance) for one baseline
    on one regime/split cell with N items.

    PASS = Wilson CI upper <= chance + alpha.
    Under H0, each item is correct with probability = chance.
    """
    rng = np.random.RandomState(seed)
    threshold = chance + alpha
    n_pass = 0
    for _ in range(n_sims):
        k = rng.binomial(n, chance)
        ci_upper = wilson_ci_upper(k, n)
        if ci_upper <= threshold:
            n_pass += 1
    return n_pass / n_sims


def joint_gate_pass_prob_independent(
    per_cell_probs: List[float]
) -> float:
    """Joint pass probability assuming independence: product of marginals."""
    p = 1.0
    for pp in per_cell_probs:
        p *= pp
    return p


def joint_gate_pass_prob_correlated(
    per_cell_probs: List[float],
    rho: float,
    n_sims: int = 100000,
    seed: int = 42
) -> float:
    """Monte Carlo estimate of joint pass probability with equicorrelated
    baseline pass/fail indicators.

    Model: each cell i has latent z_i = sqrt(rho)*W + sqrt(1-rho)*e_i
    where W ~ N(0,1) is shared, e_i ~ N(0,1) are independent.
    Cell i passes iff Phi(z_i) < per_cell_probs[i] (matching marginal).

    This models equicorrelation rho between all cells.
    """
    if rho <= 0:
        return joint_gate_pass_prob_independent(per_cell_probs)

    from scipy.stats import norm

    rng = np.random.RandomState(seed)
    K = len(per_cell_probs)
    thresholds = np.array([norm.ppf(p) if p < 1.0 else 10.0 for p in per_cell_probs])

    sqrt_rho = math.sqrt(rho)
    sqrt_1mrho = math.sqrt(1 - rho)

    n_pass = 0
    for _ in range(n_sims):
        W = rng.randn()
        e = rng.randn(K)
        z = sqrt_rho * W + sqrt_1mrho * e
        if np.all(z <= thresholds):
            n_pass += 1
    return n_pass / n_sims


def compute_power_table(
    n_per_regime_range: List[int],
    n_baselines: int = 11,
    n_regimes: int = 4,
    chance_values: Dict[str, float] = None,
    rho_values: List[float] = None,
    alpha: float = 0.05,
    n_sims_marginal: int = 50000,
    n_sims_joint: int = 100000,
    seed: int = 42,
) -> Dict:
    """Compute the joint-gate power table.

    For each per-regime N, compute:
    1. Per-cell marginal pass probabilities
    2. Joint pass probability under independence (rho=0)
    3. Joint pass probability under equicorrelation (rho > 0)

    Args:
        n_per_regime_range: list of per-regime N values to evaluate
        n_baselines: number of baselines in the gate
        n_regimes: number of regimes
        chance_values: per-regime chance levels (default: all 0.25 for v3 universal 4-option)
        rho_values: correlation values for sensitivity analysis
        alpha: equivalence margin
        n_sims_marginal: simulations for marginal pass probability
        n_sims_joint: simulations for joint pass probability

    Returns:
        dict with power table and metadata
    """
    if chance_values is None:
        # v3 universal 4-option design: chance = 0.25 for all regimes
        chance_values = {
            "CLEAN": 0.25,
            "DECOY": 0.25,
            "CONFLICT": 0.25,
            "INSUFFICIENT": 0.25,
        }
    if rho_values is None:
        rho_values = [0.0, 0.3, 0.6]

    results = []
    total = len(n_per_regime_range)

    for idx, n_per_regime in enumerate(n_per_regime_range):
        print(f"  N={n_per_regime} ({idx+1}/{total})...", file=sys.stderr)

        # Compute per-cell marginal pass probabilities
        # Each cell = (baseline, regime) on one split
        # Under H0, all baselines on a given regime have the same P(PASS)
        # because they all observe the same binomial(N, chance) correct count
        # (well, not exactly — different baselines give different predictions,
        # but under H0 each predicts at chance).
        # However, the gate evaluates each baseline independently, so each
        # baseline on each regime is a separate cell.

        # For the gate: each baseline/regime cell is evaluated on BOTH splits
        # (held-out and audit). The held-out split has much larger N (template-
        # held-out), so its pass probability is ~1.0. The audit split is the
        # binding constraint.

        # Simplification: model only the audit split (the held-out split
        # passes with near-certainty for any reasonable N).
        # K = n_baselines * n_regimes cells on the audit split.

        per_cell_probs = []
        regime_info = {}
        for regime, chance in chance_values.items():
            pp = per_baseline_pass_prob(
                n_per_regime, chance, alpha,
                n_sims=n_sims_marginal,
                seed=seed + hash(regime) % 10000
            )
            regime_info[regime] = {
                "n": n_per_regime,
                "chance": chance,
                "threshold": chance + alpha,
                "marginal_pass_prob": round(pp, 5),
            }
            # Each baseline on this regime has the same marginal pass prob
            for _ in range(n_baselines):
                per_cell_probs.append(pp)

        K = len(per_cell_probs)

        # Joint pass probability for each rho
        rho_results = {}
        for rho in rho_values:
            if rho == 0:
                joint_p = joint_gate_pass_prob_independent(per_cell_probs)
            else:
                joint_p = joint_gate_pass_prob_correlated(
                    per_cell_probs, rho,
                    n_sims=n_sims_joint,
                    seed=seed + int(rho * 1000)
                )
            rho_results[f"rho_{rho:.1f}"] = round(joint_p, 5)

        results.append({
            "n_per_regime": n_per_regime,
            "total_audit_n": n_per_regime * n_regimes,
            "n_cells": K,
            "per_regime": regime_info,
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
                    "message": f"No N in range achieved joint power >= {target}"
                }

    return {
        "description": "Joint-gate power simulation for leakage equivalence audit",
        "design": {
            "n_baselines": n_baselines,
            "n_regimes": n_regimes,
            "n_splits_modeled": 1,  # audit split only (held-out ~1.0)
            "chance_values": chance_values,
            "alpha": alpha,
            "gate": "Wilson 95% CI upper <= chance + alpha",
        },
        "sensitivity": {
            "rho_values": rho_values,
            "note": "rho = equicorrelation between baseline pass/fail indicators",
        },
        "power_table": results,
        "minimum_n": min_n_table,
        "simulation_params": {
            "n_sims_marginal": n_sims_marginal,
            "n_sims_joint": n_sims_joint,
            "seed": seed,
        },
    }


def format_report(results: Dict) -> str:
    """Format results as markdown report."""
    lines = []
    lines.append("# Leakage Audit Joint-Gate Power Simulation")
    lines.append("")
    lines.append(f"**Baselines:** {results['design']['n_baselines']}")
    lines.append(f"**Regimes:** {results['design']['n_regimes']}")
    lines.append(f"**Gate:** {results['design']['gate']}")
    lines.append(f"**Alpha (margin):** {results['design']['alpha']}")
    lines.append(f"**Chance (v3 universal 4-option):** 0.25 for all regimes")
    lines.append("")

    # Power table
    lines.append("## Power Table: P(joint gate passes | H0)")
    lines.append("")
    rho_keys = [f"rho_{rho:.1f}" for rho in results['sensitivity']['rho_values']]
    header = "| N/regime | Total N | Marginal P(PASS) | " + " | ".join(
        f"Joint (rho={rho:.1f})" for rho in results['sensitivity']['rho_values']
    ) + " |"
    sep = "|" + "|".join(["---"] * (4 + len(rho_keys))) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results['power_table']:
        # Use first regime's marginal (all same under universal 4-option)
        marginal = list(r['per_regime'].values())[0]['marginal_pass_prob']
        joint_vals = [f"{r['joint_pass_prob'][k]:.4f}" for k in rho_keys]
        lines.append(
            f"| {r['n_per_regime']} | {r['total_audit_n']} | {marginal:.4f} | "
            + " | ".join(joint_vals) + " |"
        )

    lines.append("")

    # Minimum N table
    lines.append("## Minimum N for Target Joint Power")
    lines.append("")
    lines.append("| Target | " + " | ".join(
        f"rho={rho:.1f} (N/regime)" for rho in results['sensitivity']['rho_values']
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
    lines.append("- **Marginal P(PASS)** = probability one baseline on one regime passes")
    lines.append("- **Joint** = probability ALL (baselines x regimes) pass simultaneously")
    lines.append("- Under independence (rho=0), joint = marginal^K where K = baselines x regimes")
    lines.append("- Positive correlation (rho>0) increases joint probability (failures cluster)")
    lines.append("- Only the audit split is modeled; the held-out split passes with ~1.0 probability")
    lines.append("- These results assume v3 universal 4-option design (chance=0.25 for all regimes)")
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
        n_range = [100, 200, 300, 400, 500, 600, 800, 1000]
        n_sims_m = 10000
        n_sims_j = 20000
    else:
        n_range = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000, 1500]
        n_sims_m = 50000
        n_sims_j = 100000

    print(f"Running joint-gate power simulation...", file=sys.stderr)
    results = compute_power_table(
        n_per_regime_range=n_range,
        n_baselines=11,
        n_regimes=4,
        rho_values=[0.0, 0.3, 0.6],
        n_sims_marginal=n_sims_m,
        n_sims_joint=n_sims_j,
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
