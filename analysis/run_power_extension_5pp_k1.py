#!/usr/bin/env python3
"""
Power analysis extension: 5pp effect, k_runs=1.

Extends the existing power simulation (analysis/power_simulation.py) to cover
larger N values (1250, 1500, 1750, 2000) at k_runs=1 with delta_D=0.05,
specifically targeting the question:

    "What is the smallest N achieving 90% power at 5pp, k=1, per model?"

The existing analysis showed that at k=1, 5pp, the smallest N achieving 80%
power was N=1000 for all three models, but 90% power was NOT achieved at
any tested N (max was 1500) for Qwen and proprietary models.

This extension:
  1. Runs N = {1250, 1500, 1750, 2000} x delta_D = {0.0, 0.05} x k_runs=1
     for each of the 3 model profiles, with 2000 simulations per config.
  2. Reports per-model smallest N for 90% power at 5pp, k=1.
  3. Includes delta_D=0 for Type-I error verification.
  4. Reports MC SEs, MC 95% CIs, and cost estimates.

Usage:
    python analysis/run_power_extension_5pp_k1.py
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


def run_extension():
    """Run the 5pp/k=1 power extension."""
    t0 = time.time()

    n_sims = 2000
    n_items_range = [1250, 1500, 1750, 2000]
    delta_D_range = [0.0, 0.05]
    k_runs = 1
    sigma_item = 0.5
    alpha = 0.05
    seed = 42

    print(f"=== Power Extension: 5pp, k=1 ===", file=sys.stderr)
    print(f"N values: {n_items_range}", file=sys.stderr)
    print(f"delta_D: {delta_D_range}", file=sys.stderr)
    print(f"k_runs: {k_runs}", file=sys.stderr)
    print(f"n_sims: {n_sims}", file=sys.stderr)
    print(f"Models: {list(MODEL_PROFILES.keys())}", file=sys.stderr)

    all_results = {}
    total_configs = len(MODEL_PROFILES) * len(n_items_range) * len(delta_D_range)
    config_idx = 0

    for model_name, profile in MODEL_PROFILES.items():
        p0 = profile['p0']
        model_results = []

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
                model_results.append(result)

                print(
                    f"  [{config_idx}/{total_configs}] {model_name} "
                    f"N={n_items} dD={dD:.2f} -> power={power_primary:.3f} "
                    f"(MC SE={mc_se(power_primary, n_sims):.4f})",
                    file=sys.stderr
                )

        all_results[model_name] = model_results

    # ================================================================
    # Find smallest N for 90% power at 5pp, k=1, per model
    # ================================================================
    # Also include existing results at N <= 1000 from the screening/refinement
    # for a complete picture. Load them from the existing results file.
    existing_path = 'analysis/power_simulation_results.json'
    try:
        with open(existing_path) as f:
            existing = json.load(f)
        print(f"\nLoaded existing results from {existing_path}", file=sys.stderr)
    except FileNotFoundError:
        existing = None
        print(f"\nNo existing results found at {existing_path}", file=sys.stderr)

    # Build combined results per model (existing + extension)
    combined_per_model = {}
    for model_name, profile in MODEL_PROFILES.items():
        p0 = profile['p0']
        combined = []

        # Add existing per-model refinement results for k=1, delta_D=0.05
        if existing and 'per_model_refinement' in existing:
            existing_model = existing['per_model_refinement'].get(model_name, [])
            for r in existing_model:
                if (r['k_runs'] == 1
                        and abs(r['delta_D'] - 0.05) < 1e-6):
                    combined.append({
                        'n_items': r['n_items'],
                        'power_primary': r['power_primary'],
                        'power_primary_mc_se': r.get('power_primary_mc_se', 0),
                        'power_primary_mc_ci': r.get('power_primary_mc_ci', (0, 0)),
                        'n_sims': r.get('n_sims', 2000),
                        'source': 'existing_refinement',
                    })

        # Add extension results for delta_D=0.05
        for r in all_results[model_name]:
            if abs(r['delta_D'] - 0.05) < 1e-6:
                combined.append({
                    'n_items': r['n_items'],
                    'power_primary': r['power_primary'],
                    'power_primary_mc_se': r['power_primary_mc_se'],
                    'power_primary_mc_ci': r['power_primary_mc_ci'],
                    'n_sims': r['n_sims'],
                    'source': 'extension',
                })

        combined.sort(key=lambda x: x['n_items'])
        combined_per_model[model_name] = combined

    # Find smallest N for 80% and 90% power
    smallest_n_table = []
    for model_name, combined in combined_per_model.items():
        for target_power in [0.80, 0.90]:
            found = None
            for r in combined:
                if r['power_primary'] >= target_power:
                    found = r
                    break
            smallest_n_table.append({
                'model': model_name,
                'p0': MODEL_PROFILES[model_name]['p0'],
                'delta_D': 0.05,
                'k_runs': k_runs,
                'target_power': target_power,
                'smallest_n': found['n_items'] if found else None,
                'achieved_power': found['power_primary'] if found else None,
                'mc_se': found.get('power_primary_mc_se') if found else None,
                'mc_ci': found.get('power_primary_mc_ci') if found else None,
                'n_sims': found.get('n_sims') if found else n_sims,
                'source': found.get('source', 'N/A') if found else 'N/A',
            })

    elapsed = time.time() - t0

    # ================================================================
    # ASSEMBLE OUTPUT
    # ================================================================
    output = {
        'description': 'Power extension for 5pp effect, k_runs=1',
        'n_sims': n_sims,
        'n_items_range': n_items_range,
        'delta_D_range': delta_D_range,
        'k_runs': k_runs,
        'sigma_item': sigma_item,
        'alpha': alpha,
        'seed': seed,
        'per_model_results': all_results,
        'combined_power_curve': {
            model_name: combined
            for model_name, combined in combined_per_model.items()
        },
        'smallest_n_table': smallest_n_table,
        'elapsed_seconds': round(elapsed, 1),
    }

    output_path = 'analysis/power_extension_5pp_k1_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)

    # ================================================================
    # FORMAT REPORT
    # ================================================================
    report = format_extension_report(
        all_results, combined_per_model, smallest_n_table,
        n_sims, n_items_range, elapsed
    )
    report_path = 'analysis/power_extension_5pp_k1_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # Print summary to stdout
    print("\n" + report)


def format_extension_report(
    all_results: Dict,
    combined: Dict,
    smallest_n_table: List,
    n_sims: int,
    n_items_range: List[int],
    elapsed: float,
) -> str:
    """Format the extension report as markdown."""
    lines = [
        "# Power Analysis Extension: 5pp Effect, k_runs=1",
        "",
        "## Purpose",
        "",
        "The existing power analysis (analysis/power_simulation_results_report.md)",
        "showed that at k_runs=1, delta_D=0.05 (5pp effect):",
        "",
        "- 80% power achieved at N=1000 for all three models",
        "- 90% power NOT achieved at N<=1500 for Qwen2.5-72B and proprietary",
        "- 90% power achieved at N=1000 for Llama-3.3-70B (barely: 0.900)",
        "",
        "This extension tests N = {1250, 1500, 1750, 2000} with 2000 simulations",
        "per configuration to find the smallest N achieving 90% power at 5pp, k=1.",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| N items tested | {n_items_range} |",
        f"| delta_D | 0.0 (Type-I), 0.05 (5pp) |",
        f"| k_runs | 1 |",
        f"| sigma_item | 0.5 |",
        f"| Alpha | 0.05 |",
        f"| N simulations | {n_sims} per configuration |",
        f"| Models | Llama-3.3-70B (p0=0.40), Qwen2.5-72B (p0=0.45), proprietary (p0=0.55) |",
        f"| Elapsed | {elapsed:.1f}s |",
        "",
    ]

    # Per-model extension results
    lines.append("## Extension Results (N = 1250-2000, k_runs=1)")
    lines.append("")

    for model_name, results in all_results.items():
        profile = MODEL_PROFILES[model_name]
        lines.append(f"### {model_name} (p0={profile['p0']})")
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
    lines.append("## Type-I Error Verification (delta_D = 0, k_runs=1)")
    lines.append("")
    lines.append("| Model | p0 | N_items | Rejection rate | MC SE | MC 95% CI | Nominal |")
    lines.append("|-------|----|---------|----------------|-------|-----------|---------|")
    for model_name, results in all_results.items():
        for r in sorted(results, key=lambda x: x['n_items']):
            if abs(r['delta_D']) < 1e-6:
                ci = r.get('power_primary_mc_ci', (0, 0))
                lines.append(
                    f"| {model_name} | {r['p0']} | {r['n_items']} | "
                    f"{r['power_primary']:.4f} | {r['power_primary_mc_se']:.4f} | "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}] | 0.0500 |"
                )
    lines.append("")

    # Combined power curve (existing + extension)
    lines.append("## Combined Power Curve: 5pp, k=1 (existing + extension)")
    lines.append("")
    lines.append("Merges existing refined results (N <= 1500) with new extension (N >= 1250).")
    lines.append("")

    for model_name, curve in combined.items():
        profile = MODEL_PROFILES[model_name]
        lines.append(f"### {model_name} (p0={profile['p0']})")
        lines.append("")
        lines.append("| N_items | Power | MC SE | MC 95% CI | Source |")
        lines.append("|---------|-------|-------|-----------|--------|")
        for r in curve:
            ci = r.get('power_primary_mc_ci', (0, 0))
            lines.append(
                f"| {r['n_items']} | {r['power_primary']:.4f} | "
                f"{r.get('power_primary_mc_se', 0):.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{r.get('source', 'unknown')} |"
            )
        lines.append("")

    # Smallest N table
    lines.append("## Per-Model Smallest N for 5pp, k=1 (combined)")
    lines.append("")
    lines.append("| Model | p0 | Target Power | Smallest N | Achieved | MC SE | MC 95% CI | n_sims | Source |")
    lines.append("|-------|----|-------------|------------|----------|-------|-----------|--------|--------|")
    for s in smallest_n_table:
        if s['smallest_n'] is not None:
            ci = s.get('mc_ci', (0, 0))
            lines.append(
                f"| {s['model']} | {s['p0']} | {s['target_power']} | "
                f"{s['smallest_n']} | {s['achieved_power']:.4f} | "
                f"{s.get('mc_se', 0):.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{s['n_sims']} | {s['source']} |"
            )
        else:
            lines.append(
                f"| {s['model']} | {s['p0']} | {s['target_power']} | "
                f">max tested | N/A | N/A | N/A | {s['n_sims']} | N/A |"
            )
    lines.append("")

    # Cost model extension
    lines.append("## Cost Estimates for Extended N (k=1)")
    lines.append("")
    lines.append("**Assumptions:** 5 conditions per item, 3 models, ~2000 tokens/call,")
    lines.append("open-weight ~$0.00 marginal, proprietary ~$0.01/1K tokens.")
    lines.append("")
    lines.append("| N_items | k_runs | Calls/model | Total calls (3 models) | Proprietary tokens | Proprietary cost |")
    lines.append("|---------|--------|-------------|------------------------|--------------------|--------------------|")
    for n in [1000] + sorted(set(n_items_range)):
        calls_per_model = n * 5 * 1  # k=1
        total_calls = calls_per_model * 3
        prop_tokens = calls_per_model * 2000
        prop_cost = prop_tokens * 0.01 / 1000
        lines.append(
            f"| {n} | 1 | {calls_per_model:,} | {total_calls:,} | "
            f"{prop_tokens:,} | ${prop_cost:,.2f} |"
        )
    lines.append("")
    lines.append("**Note:** Open-weight model costs depend on infrastructure.")
    lines.append("Proprietary cost uses $0.01/1K tokens as a rough estimate.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_extension()
