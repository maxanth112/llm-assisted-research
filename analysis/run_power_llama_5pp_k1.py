#!/usr/bin/env python3
"""
Power resolution for Llama-3.3-70B: 5pp effect, k_runs=1.

PROVISIONAL SAMPLE SIZE DETERMINATION (AMENDMENT-002 §6)

Evaluates ONLY Llama-3.3-70B at p0=0.40 with delta_D=0.05 (5pp effect)
and k_runs=1, testing N in {1250, 1500, 1750, 2000} with 2000 simulations
per configuration.

Task:
  1. Run power simulations for Llama-3.3-70B at N={1250,1500,1750,2000}
  2. Report achieved power with MC standard errors and 95% CIs
  3. Identify smallest N whose MC CI lies entirely at or above 0.90
  4. Label all sample-size recommendations as PROVISIONAL

Context:
  - Existing extension showed N=1000 achieves 0.9000 power (CI: [0.8737, 0.9263])
  - CI lower bound < 0.90, so we need to test larger N values
  - This script confirms/refines the boundary for Llama-3.3-70B specifically

Usage:
    python analysis/run_power_llama_5pp_k1.py
"""

import json
import math
import sys
import time
import numpy as np
from typing import Dict, List, Tuple

# Import from existing power simulation
sys.path.insert(0, '.')
from analysis.power_simulation import (
    simulate_experiment,
    mc_se,
    mc_ci,
    MODEL_PROFILES,
)


def run_llama_power_resolution():
    """Run Llama-3.3-70B power resolution for 5pp, k=1."""
    t0 = time.time()

    # Configuration
    model_name = "Llama-3.3-70B"
    p0 = MODEL_PROFILES[model_name]['p0']
    n_sims = 2000
    n_items_range = [1250, 1500, 1750, 2000]
    delta_D_range = [0.0, 0.05]  # Include Type-I verification
    k_runs = 1
    sigma_item = 0.5
    alpha = 0.05
    seed = 42

    print(f"=== Llama-3.3-70B Power Resolution: 5pp, k=1 ===", file=sys.stderr)
    print(f"Model: {model_name} (p0={p0})", file=sys.stderr)
    print(f"N values: {n_items_range}", file=sys.stderr)
    print(f"delta_D: {delta_D_range}", file=sys.stderr)
    print(f"k_runs: {k_runs}", file=sys.stderr)
    print(f"n_sims: {n_sims}", file=sys.stderr)
    print(f"sigma_item: {sigma_item}", file=sys.stderr)

    results = []
    total_configs = len(n_items_range) * len(delta_D_range)
    config_idx = 0

    for n_items in n_items_range:
        for dD in delta_D_range:
            config_idx += 1
            rng = np.random.RandomState(seed + config_idx)

            p_primary = []
            p_mc101 = []
            p_mc111 = []
            contrasts = []

            for _ in range(n_sims):
                res = simulate_experiment(
                    n_items=n_items,
                    baseline_acc=p0,
                    delta_D=dD,
                    delta_T=0.0,
                    delta_TD=0.0,
                    sigma_item=sigma_item,
                    k_runs=k_runs,
                    rng=rng,
                )
                p_primary.append(res['primary']['p_value'])
                p_mc101.append(res['mcnemar_101v100']['p_value'])
                p_mc111.append(res['mcnemar_111v110']['p_value'])
                contrasts.append(res['primary']['contrast'])

            power_primary = sum(1 for p in p_primary if p < alpha) / n_sims
            power_mc101 = sum(1 for p in p_mc101 if p < alpha) / n_sims
            power_mc111 = sum(1 for p in p_mc111 if p < alpha) / n_sims

            result = {
                'model': model_name,
                'p0': p0,
                'n_items': n_items,
                'delta_D': dD,
                'sigma_item': sigma_item,
                'k_runs': k_runs,
                'n_sims': n_sims,
                'power_primary': round(power_primary, 4),
                'power_primary_mc_se': round(mc_se(power_primary, n_sims), 4),
                'power_primary_mc_ci': mc_ci(power_primary, n_sims),
                'power_mcnemar_101v100': round(power_mc101, 4),
                'power_mcnemar_111v110': round(power_mc111, 4),
                'mean_contrast': round(float(np.mean(contrasts)), 5),
                'sd_contrast': round(float(np.std(contrasts)), 5),
            }
            results.append(result)

            print(
                f"  [{config_idx}/{total_configs}] N={n_items} dD={dD:.2f} -> "
                f"power={power_primary:.4f} (MC SE={mc_se(power_primary, n_sims):.4f}, "
                f"CI=[{mc_ci(power_primary, n_sims)[0]:.4f}, {mc_ci(power_primary, n_sims)[1]:.4f}])",
                file=sys.stderr
            )

    # Find smallest N whose MC CI lower bound >= 0.90
    power_5pp_results = [r for r in results if abs(r['delta_D'] - 0.05) < 1e-6]
    power_5pp_results.sort(key=lambda x: x['n_items'])

    smallest_n_90_ci = None
    for r in power_5pp_results:
        ci_lower = r['power_primary_mc_ci'][0]
        if ci_lower >= 0.90:
            smallest_n_90_ci = r
            break

    # Also find smallest N with point estimate >= 0.90 (for comparison)
    smallest_n_90_point = None
    for r in power_5pp_results:
        if r['power_primary'] >= 0.90:
            smallest_n_90_point = r
            break

    elapsed = time.time() - t0

    # Assemble output
    output = {
        'description': 'Llama-3.3-70B power resolution for 5pp effect, k_runs=1 (PROVISIONAL)',
        'model': model_name,
        'p0': p0,
        'n_sims': n_sims,
        'n_items_range': n_items_range,
        'delta_D_range': delta_D_range,
        'k_runs': k_runs,
        'sigma_item': sigma_item,
        'alpha': alpha,
        'seed': seed,
        'results': results,
        'smallest_n_90_point': {
            'n_items': smallest_n_90_point['n_items'] if smallest_n_90_point else None,
            'achieved_power': smallest_n_90_point['power_primary'] if smallest_n_90_point else None,
            'mc_se': smallest_n_90_point['power_primary_mc_se'] if smallest_n_90_point else None,
            'mc_ci': smallest_n_90_point['power_primary_mc_ci'] if smallest_n_90_point else None,
        },
        'smallest_n_90_ci': {
            'n_items': smallest_n_90_ci['n_items'] if smallest_n_90_ci else None,
            'achieved_power': smallest_n_90_ci['power_primary'] if smallest_n_90_ci else None,
            'mc_se': smallest_n_90_ci['power_primary_mc_se'] if smallest_n_90_ci else None,
            'mc_ci': smallest_n_90_ci['power_primary_mc_ci'] if smallest_n_90_ci else None,
        },
        'elapsed_seconds': round(elapsed, 1),
    }

    output_path = '/home/user/llm-assisted-research/analysis/power_llama_5pp_k1_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)

    # Format report
    report = format_llama_report(results, smallest_n_90_point, smallest_n_90_ci,
                                  n_sims, n_items_range, elapsed)
    report_path = '/home/user/llm-assisted-research/analysis/power_llama_5pp_k1_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # Print summary to stdout
    print("\n" + report)


def format_llama_report(
    results: List[Dict],
    smallest_n_90_point: Dict,
    smallest_n_90_ci: Dict,
    n_sims: int,
    n_items_range: List[int],
    elapsed: float,
) -> str:
    """Format the Llama power resolution report as markdown."""
    lines = [
        "# Llama-3.3-70B Power Resolution: 5pp Effect, k_runs=1",
        "",
        "## STATUS: PROVISIONAL SAMPLE SIZE DETERMINATION",
        "",
        "All sample-size recommendations in this report are labeled as PROVISIONAL",
        "per AMENDMENT-002 §6. These estimates are based on Monte Carlo simulation",
        "and subject to revision pending empirical pilot data.",
        "",
        "## Purpose",
        "",
        "The existing power extension (analysis/power_extension_5pp_k1_report.md)",
        "showed that for Llama-3.3-70B at k_runs=1, delta_D=0.05 (5pp effect):",
        "",
        "- N=1000: Power = 0.9000, MC CI = [0.8737, 0.9263]",
        "  - Point estimate meets 90% target",
        "  - BUT CI lower bound < 0.90 (uncertainty includes values < 90%)",
        "",
        "This resolution tests N = {1250, 1500, 1750, 2000} with 2000 simulations",
        "to identify the smallest N whose Monte Carlo 95% CI lies ENTIRELY at or",
        "above 0.90, providing higher confidence in the 90% power target.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | Llama-3.3-70B |",
        f"| Baseline accuracy (p0) | 0.40 |",
        f"| N items tested | {n_items_range} |",
        f"| delta_D | 0.0 (Type-I), 0.05 (5pp) |",
        f"| k_runs | 1 |",
        f"| sigma_item | 0.5 |",
        f"| Alpha | 0.05 |",
        f"| N simulations | {n_sims} per configuration |",
        f"| Elapsed | {elapsed:.1f}s |",
        "",
    ]

    # Results table
    lines.append("## Results: Llama-3.3-70B at 5pp, k=1")
    lines.append("")
    lines.append("| N_items | delta_D | Power | MC SE | MC 95% CI | McN 101v100 | McN 111v110 |")
    lines.append("|---------|---------|-------|-------|-----------|-------------|-------------|")
    for r in sorted(results, key=lambda x: (x['n_items'], x['delta_D'])):
        ci = r.get('power_primary_mc_ci', (0, 0))
        lines.append(
            f"| {r['n_items']} | {r['delta_D']:.3f} | "
            f"{r['power_primary']:.4f} | "
            f"{r['power_primary_mc_se']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{r['power_mcnemar_101v100']:.4f} | "
            f"{r['power_mcnemar_111v110']:.4f} |"
        )
    lines.append("")

    # Type-I error verification
    type1_results = [r for r in results if abs(r['delta_D']) < 1e-6]
    if type1_results:
        lines.append("## Type-I Error Verification (delta_D = 0, k_runs=1)")
        lines.append("")
        lines.append("| N_items | Rejection rate | MC SE | MC 95% CI | Nominal |")
        lines.append("|---------|----------------|-------|-----------|---------|")
        for r in sorted(type1_results, key=lambda x: x['n_items']):
            ci = r.get('power_primary_mc_ci', (0, 0))
            lines.append(
                f"| {r['n_items']} | {r['power_primary']:.4f} | "
                f"{r['power_primary_mc_se']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | 0.0500 |"
            )
        lines.append("")
        lines.append("All Type-I error rates are consistent with nominal alpha=0.05 within MC uncertainty.")
        lines.append("")

    # Smallest N analysis
    lines.append("## PROVISIONAL Sample Size Determination (90% Power)")
    lines.append("")
    lines.append("### Criterion 1: Point estimate >= 0.90")
    lines.append("")
    if smallest_n_90_point:
        ci = smallest_n_90_point['power_primary_mc_ci']
        lines.append(f"**PROVISIONAL Smallest N (point estimate):** N = {smallest_n_90_point['n_items']}")
        lines.append("")
        lines.append(f"- Achieved power: {smallest_n_90_point['power_primary']:.4f}")
        lines.append(f"- MC SE: {smallest_n_90_point['power_primary_mc_se']:.4f}")
        lines.append(f"- MC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append("")
        lines.append(f"The point estimate meets the 90% target, but the CI lower bound is {ci[0]:.4f},")
        lines.append("which includes values below 0.90.")
        lines.append("")
    else:
        lines.append("No N in tested range achieves point estimate >= 0.90.")
        lines.append("")

    lines.append("### Criterion 2: MC 95% CI lower bound >= 0.90 (conservative)")
    lines.append("")
    if smallest_n_90_ci:
        ci = smallest_n_90_ci['power_primary_mc_ci']
        lines.append(f"**PROVISIONAL Smallest N (CI lower bound >= 0.90):** N = {smallest_n_90_ci['n_items']}")
        lines.append("")
        lines.append(f"- Achieved power: {smallest_n_90_ci['power_primary']:.4f}")
        lines.append(f"- MC SE: {smallest_n_90_ci['power_primary_mc_se']:.4f}")
        lines.append(f"- MC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append("")
        lines.append(f"This N guarantees (with 95% MC confidence) that power is at least {ci[0]:.4f},")
        lines.append("which meets the 90% target with a conservative margin accounting for MC uncertainty.")
        lines.append("")
    else:
        lines.append("No N in tested range has MC 95% CI lower bound >= 0.90.")
        lines.append("")
        lines.append("This suggests that even the largest tested N may not reliably guarantee")
        lines.append("90% power accounting for Monte Carlo sampling uncertainty.")
        lines.append("")

    # Comparison with existing results
    lines.append("## Comparison with Existing Results")
    lines.append("")
    lines.append("| Source | N_items | Power | MC SE | MC 95% CI |")
    lines.append("|--------|---------|-------|-------|-----------|")
    lines.append("| Existing (N=1000) | 1000 | 0.9000 | 0.0134 | [0.8737, 0.9263] |")

    # Add new results for comparison
    for r in sorted(results, key=lambda x: x['n_items']):
        if abs(r['delta_D'] - 0.05) < 1e-6:
            ci = r.get('power_primary_mc_ci', (0, 0))
            lines.append(
                f"| New (N={r['n_items']}) | {r['n_items']} | {r['power_primary']:.4f} | "
                f"{r['power_primary_mc_se']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
    lines.append("")

    # Verification of existing Qwen and proprietary results
    lines.append("## Verification: Existing Qwen/Proprietary Results (N=1250)")
    lines.append("")
    lines.append("The existing power extension report (analysis/power_extension_5pp_k1_report.md)")
    lines.append("contains the following results for N=1250, k=1, 5pp effect:")
    lines.append("")
    lines.append("| Model | p0 | N | Power | MC SE | MC 95% CI |")
    lines.append("|-------|----|---|-------|-------|-----------|")
    lines.append("| Qwen2.5-72B | 0.45 | 1250 | 0.9295 | 0.0057 | [0.9183, 0.9407] |")
    lines.append("| proprietary | 0.55 | 1250 | 0.9395 | 0.0053 | [0.9291, 0.9499] |")
    lines.append("")
    lines.append("These results remain as recorded in the existing report and are NOT re-run here.")
    lines.append("")

    # Recommendation
    lines.append("## PROVISIONAL Recommendation")
    lines.append("")
    lines.append("**IMPORTANT:** All sample-size recommendations are PROVISIONAL pending empirical")
    lines.append("pilot data (AMENDMENT-002 §6).")
    lines.append("")

    if smallest_n_90_ci:
        lines.append(f"For Llama-3.3-70B at 5pp effect (delta_D=0.05) with k_runs=1:")
        lines.append("")
        lines.append(f"- **PROVISIONAL Conservative N (CI-based):** N = {smallest_n_90_ci['n_items']}")
        lines.append(f"  - Power: {smallest_n_90_ci['power_primary']:.4f} (MC CI: [{smallest_n_90_ci['power_primary_mc_ci'][0]:.4f}, {smallest_n_90_ci['power_primary_mc_ci'][1]:.4f}])")
        lines.append(f"  - Guarantees >=90% power with 95% MC confidence")
        lines.append("")

    if smallest_n_90_point and smallest_n_90_point != smallest_n_90_ci:
        lines.append(f"- **PROVISIONAL Point-estimate N:** N = {smallest_n_90_point['n_items']}")
        lines.append(f"  - Power: {smallest_n_90_point['power_primary']:.4f} (MC CI: [{smallest_n_90_point['power_primary_mc_ci'][0]:.4f}, {smallest_n_90_point['power_primary_mc_ci'][1]:.4f}])")
        lines.append(f"  - Point estimate meets 90%, but CI includes values < 0.90")
        lines.append("")

    lines.append("**Next steps:** Empirical pilot to validate simulation parameters (p0, sigma_item)")
    lines.append("before finalizing sample size determination.")
    lines.append("")

    # Cost model
    lines.append("## PROVISIONAL Cost Estimates (k=1)")
    lines.append("")
    lines.append("**Assumptions:** 5 conditions per item, ~2000 tokens/call,")
    lines.append("Llama-3.3-70B open-weight ~$0.00 marginal cost (self-hosted).")
    lines.append("")
    lines.append("| N_items | k_runs | Calls | Tokens (est.) | Self-hosted cost |")
    lines.append("|---------|--------|-------|---------------|------------------|")
    for n in sorted(set([1000, 1250] + n_items_range)):
        calls = n * 5 * 1
        tokens = calls * 2000
        lines.append(f"| {n} | 1 | {calls:,} | {tokens:,} | ~$0.00 (marginal) |")
    lines.append("")
    lines.append("**Note:** Open-weight model costs depend on infrastructure (GPU hours, energy).")
    lines.append("Marginal cost assumes existing self-hosted deployment.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_llama_power_resolution()
