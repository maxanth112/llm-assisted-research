#!/usr/bin/env python3
"""
Monte Carlo power simulation for the ETD-ACH factorial experiment.

Statistical model:
Y_ij ~ Bernoulli(p_ij)
logit(p_ij) = mu + alpha_E*E + alpha_T*T + alpha_D*D + beta_ET*E*T + u_j
u_j ~ N(0, sigma_item^2)

where:
- E: Evidence (enumerate-only vs full)
- T: Trajectory (CoT vs direct)
- D: Dataset (full ACH with deconfounding vs enumerate-only)
- u_j: item random effect
"""

import argparse
import json
import sys
from typing import Dict, List, Any, Tuple
import random


def sigmoid(x: float) -> float:
    """Sigmoid function for logit to probability conversion."""
    import math
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    """Logit function for probability to log-odds conversion."""
    import math
    p = max(0.001, min(0.999, p))  # Clamp to avoid log(0)
    return math.log(p / (1 - p))


def simulate_single_experiment(
    n_items: int,
    baseline_acc: float,
    effects: Dict[str, float],
    sigma_item: float,
    k_runs: int,
    seed: int
) -> Dict[str, Any]:
    """Simulate a single factorial experiment.

    Args:
        n_items: Number of items
        baseline_acc: Baseline accuracy (intercept on probability scale)
        effects: Dictionary of effect sizes on probability scale
                 {'E': 0.02, 'T': 0.01, 'D': 0.03, 'ET': 0.005}
        sigma_item: Standard deviation of item random effects
        k_runs: Number of runs per condition per item
        seed: Random seed

    Returns:
        Dictionary with p_value and effect_estimate
    """
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        raise ImportError("numpy and scipy required for power simulation")

    rng = np.random.RandomState(seed)

    # Convert baseline accuracy to logit scale
    mu = logit(baseline_acc)

    # Convert effects to logit scale (approximate)
    alpha_E = logit(baseline_acc + effects.get('E', 0.0)) - mu
    alpha_T = logit(baseline_acc + effects.get('T', 0.0)) - mu
    alpha_D = logit(baseline_acc + effects.get('D', 0.0)) - mu
    beta_ET = effects.get('ET', 0.0)  # Interaction on logit scale

    # Generate item random effects
    u_items = rng.normal(0, sigma_item, n_items)

    # Generate data for 2x2x2 factorial design
    conditions = [
        (0, 0, 0),  # E=0, T=0, D=0 (baseline)
        (1, 0, 0),  # E=1, T=0, D=0
        (0, 1, 0),  # E=0, T=1, D=0
        (1, 1, 0),  # E=1, T=1, D=0
        (0, 0, 1),  # E=0, T=0, D=1
        (1, 0, 1),  # E=1, T=0, D=1
        (0, 1, 1),  # E=0, T=1, D=1
        (1, 1, 1),  # E=1, T=1, D=1
    ]

    # Collect outcomes for D=0 and D=1 conditions
    outcomes_D0 = []
    outcomes_D1 = []

    for item_idx in range(n_items):
        u_j = u_items[item_idx]

        for E, T, D in conditions:
            # Compute logit for this condition
            logit_p = mu + alpha_E * E + alpha_T * T + alpha_D * D + beta_ET * E * T + u_j
            p = sigmoid(logit_p)

            # Simulate k_runs Bernoulli trials
            outcomes = rng.binomial(1, p, k_runs)

            # Store outcomes by D condition
            if D == 0:
                outcomes_D0.extend(outcomes)
            else:
                outcomes_D1.extend(outcomes)

    # Run two-sample t-test comparing D=1 vs D=0
    t_stat, p_value = stats.ttest_ind(outcomes_D1, outcomes_D0)

    # Effect estimate (mean difference)
    effect_estimate = np.mean(outcomes_D1) - np.mean(outcomes_D0)

    return {
        "p_value": float(p_value),
        "effect_estimate": float(effect_estimate),
        "t_statistic": float(t_stat),
        "mean_D0": float(np.mean(outcomes_D0)),
        "mean_D1": float(np.mean(outcomes_D1))
    }


def run_power_sweep(
    n_sims: int,
    n_items_range: List[int],
    baseline_accs: List[float],
    effect_sizes: List[float],
    sigma_items: List[float],
    k_runs_range: List[int],
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """Run power sweep across design parameters.

    Args:
        n_sims: Number of simulations per configuration
        n_items_range: List of item counts to sweep [25, 50, 100, 150, 200, 250]
        baseline_accs: List of baseline accuracies [0.4, 0.5, 0.6]
        effect_sizes: List of D main effects to sweep [0.005, 0.01, 0.02, 0.03, 0.05]
        sigma_items: List of item SD values [0.3, 0.5, 0.8]
        k_runs_range: List of runs per condition [1, 3, 5]
        alpha: Significance level (default: 0.05)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with power curves and configuration details
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy required for power simulation")

    rng = np.random.RandomState(seed)

    results = []
    total_configs = (len(n_items_range) * len(baseline_accs) * len(effect_sizes) *
                     len(sigma_items) * len(k_runs_range))

    print(f"Running power sweep: {total_configs} configurations × {n_sims} simulations = {total_configs * n_sims} total",
          file=sys.stderr)

    config_idx = 0
    for n_items in n_items_range:
        for baseline_acc in baseline_accs:
            for effect_size in effect_sizes:
                for sigma_item in sigma_items:
                    for k_runs in k_runs_range:
                        config_idx += 1

                        # Run simulations for this configuration
                        effects = {'E': 0.0, 'T': 0.0, 'D': effect_size, 'ET': 0.0}
                        p_values = []
                        effect_estimates = []

                        for sim_idx in range(n_sims):
                            sim_seed = rng.randint(0, 2**31)
                            result = simulate_single_experiment(
                                n_items=n_items,
                                baseline_acc=baseline_acc,
                                effects=effects,
                                sigma_item=sigma_item,
                                k_runs=k_runs,
                                seed=sim_seed
                            )
                            p_values.append(result['p_value'])
                            effect_estimates.append(result['effect_estimate'])

                        # Compute power (proportion of significant results)
                        power = sum(1 for p in p_values if p < alpha) / n_sims

                        config_result = {
                            'n_items': n_items,
                            'baseline_acc': baseline_acc,
                            'effect_size': effect_size,
                            'sigma_item': sigma_item,
                            'k_runs': k_runs,
                            'power': power,
                            'mean_effect_estimate': float(np.mean(effect_estimates)),
                            'sd_effect_estimate': float(np.std(effect_estimates))
                        }

                        results.append(config_result)

                        if config_idx % 10 == 0:
                            print(f"Progress: {config_idx}/{total_configs} configurations completed",
                                  file=sys.stderr)

    return {
        'n_sims': n_sims,
        'alpha': alpha,
        'seed': seed,
        'results': results,
        'n_configurations': len(results)
    }


def find_minimum_design(
    sweep_results: Dict[str, Any],
    target_power: float = 0.80
) -> Dict[str, Any]:
    """Find minimum n_items achieving target power for each effect size.

    Args:
        sweep_results: Output from run_power_sweep
        target_power: Target statistical power (default: 0.80)

    Returns:
        Dictionary with minimum design recommendations
    """
    results = sweep_results['results']

    # Group by effect size
    by_effect_size = {}
    for r in results:
        effect = r['effect_size']
        if effect not in by_effect_size:
            by_effect_size[effect] = []
        by_effect_size[effect].append(r)

    # For each effect size, find minimum n_items achieving target power
    recommendations = {}

    for effect_size, configs in by_effect_size.items():
        # Filter configs that achieve target power
        achieving_power = [c for c in configs if c['power'] >= target_power]

        if achieving_power:
            # Find minimum n_items
            min_config = min(achieving_power, key=lambda c: c['n_items'])
            recommendations[f"effect_{effect_size}"] = {
                'effect_size': effect_size,
                'min_n_items': min_config['n_items'],
                'power': min_config['power'],
                'baseline_acc': min_config['baseline_acc'],
                'sigma_item': min_config['sigma_item'],
                'k_runs': min_config['k_runs']
            }
        else:
            recommendations[f"effect_{effect_size}"] = {
                'effect_size': effect_size,
                'min_n_items': None,
                'message': f"No configuration achieved {target_power} power"
            }

    return {
        'target_power': target_power,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo power simulation for ETD-ACH factorial experiment"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick mode for CI/testing (fewer simulations)"
    )
    parser.add_argument(
        "--output",
        default="power_simulation_results.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Number of simulations per configuration (default: 500 for quick, 1000 for full)"
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_sims = args.n_sims or 500
        n_items_range = [25, 50, 100, 150, 200]
        baseline_accs = [0.5]
        sigma_items = [0.5]
        k_runs_range = [1, 3]
        print("Running in QUICK mode for CI/testing", file=sys.stderr)
    else:
        n_sims = args.n_sims or 1000
        n_items_range = [25, 50, 100, 150, 200, 250]
        baseline_accs = [0.4, 0.5, 0.6]
        sigma_items = [0.3, 0.5, 0.8]
        k_runs_range = [1, 3, 5]
        print("Running in FULL mode", file=sys.stderr)

    effect_sizes = [0.005, 0.01, 0.02, 0.03, 0.05]

    # Run power sweep
    print(f"Starting power sweep with {n_sims} simulations per configuration...",
          file=sys.stderr)

    sweep_results = run_power_sweep(
        n_sims=n_sims,
        n_items_range=n_items_range,
        baseline_accs=baseline_accs,
        effect_sizes=effect_sizes,
        sigma_items=sigma_items,
        k_runs_range=k_runs_range,
        alpha=0.05,
        seed=42
    )

    # Find minimum designs
    print("Finding minimum designs for 80% power...", file=sys.stderr)
    min_designs = find_minimum_design(sweep_results, target_power=0.80)

    # Combine results
    output = {
        'sweep_results': sweep_results,
        'minimum_designs': min_designs,
        'summary': {
            'n_configurations': sweep_results['n_configurations'],
            'n_sims_per_config': n_sims,
            'total_simulations': sweep_results['n_configurations'] * n_sims
        }
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}", file=sys.stderr)

    # Print summary
    print("\nMinimum Design Recommendations (80% power):")
    print("=" * 60)
    for effect_key, rec in min_designs['recommendations'].items():
        effect = rec['effect_size']
        if rec.get('min_n_items'):
            print(f"Effect size {effect:.3f} ({effect*100:.1f}pp):")
            print(f"  Minimum n_items: {rec['min_n_items']}")
            print(f"  Achieved power: {rec['power']:.3f}")
            print(f"  k_runs: {rec['k_runs']}")
        else:
            print(f"Effect size {effect:.3f} ({effect*100:.1f}pp):")
            print(f"  {rec['message']}")
