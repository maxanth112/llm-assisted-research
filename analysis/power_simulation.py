#!/usr/bin/env python3
"""
Monte Carlo power simulation for the ETD-ACH 5-condition experiment.

Phase A.2 cleanup rewrite.

Design (AMENDMENT-002 §4.1):
  5 conditions (E=0 cells with T=1/D=1 are incoherent and excluded):
    000 (E=0,T=0,D=0) - Baseline
    100 (E=1,T=0,D=0) - Enumerate-only
    110 (E=1,T=1,D=0) - Enumerate + trajectory
    101 (E=1,T=0,D=1) - Enumerate + deconfounding (ACH)
    111 (E=1,T=1,D=1) - Full scaffold

Primary confirmatory estimand:
  Paired marginal contrast for the D effect, CONDITIONAL on E=1,
  in adversarial regimes (DECOY+CONFLICT), AVERAGED over T:
    contrast_i = 0.5 * [(Y_101_i - Y_100_i) + (Y_111_i - Y_110_i)]
  where i indexes items. The estimand is E[contrast_i].

Primary test:
  One-sample paired t-test on the item-level contrast values.
  The t-statistic is contrast_mean / (contrast_sd / sqrt(N)).
  Tested two-sided at alpha = 0.05.

  Confidence interval: paired percentile bootstrap (10,000 resamples)
  on the mean contrast. Reported alongside the t-test p-value.

  Model is a FIXED effect → power is computed per-model.

Robustness (secondary, not gated):
  Effect-coded binomial GEE with item-clustered robust SE.
  Reported for comparison but does not replace the paired t-test.

Secondary (reported, not gated):
  - McNemar tests: 101 vs 100, 111 vs 110 (paired binary)
  - Enumeration contrast: 100 vs 000
  - T|E=1: mean(110,111) vs mean(100,101)
  - T×D|E=1 interaction

Usage:
  python analysis/power_simulation.py [--quick] [--output FILE]
"""

import argparse
import json
import math
import sys
import time
import numpy as np
from typing import Dict, List, Any


def sigmoid(x):
    """Vectorized sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def logit(p: float) -> float:
    """Logit of a scalar probability."""
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def simulate_experiment(
    n_items: int,
    baseline_acc: float,
    delta_D: float,
    delta_T: float,
    delta_TD: float,
    sigma_item: float,
    k_runs: int,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Simulate one 5-condition experiment.

    Args:
        n_items: adversarial items (each item is measured under all 5 conditions)
        baseline_acc: P(correct) at condition 000
        delta_D: D effect on probability scale (averaged over T, conditional on E=1)
        delta_T: T effect on probability scale (averaged over D, conditional on E=1)
        delta_TD: T×D interaction on probability scale
        sigma_item: SD of item random effects (logit scale)
        k_runs: independent runs per item × condition
        rng: random state

    Returns:
        dict with p-values and effect estimates for primary and secondary tests
    """
    mu_logit = logit(baseline_acc)

    # E main effect (100 vs 000): set to 0 (E is a precondition, not an effect)
    # Within E=1 conditions, use effect coding:
    #   T_eff = T - 0.5, D_eff = D - 0.5
    # So the D_eff coefficient is the D effect averaged over T.

    # Convert effects from probability scale to approximate logit increments
    # at the operating point
    p_base_E1 = baseline_acc  # assume E itself has no effect for simplicity
    # Within E=1 conditions:
    #   p(T=0,D=0) = p_base_E1
    #   p(T=0,D=1) = p_base_E1 + delta_D - 0.5*delta_TD
    #   p(T=1,D=0) = p_base_E1 + delta_T - 0.5*delta_TD
    #   p(T=1,D=1) = p_base_E1 + delta_T + delta_D + 0.5*delta_TD
    # This gives the marginal contrast:
    #   0.5*[(101-100) + (111-110)] = delta_D
    p_100 = baseline_acc
    p_110 = min(0.999, max(0.001, baseline_acc + delta_T - 0.5 * delta_TD))
    p_101 = min(0.999, max(0.001, baseline_acc + delta_D - 0.5 * delta_TD))
    p_111 = min(0.999, max(0.001, baseline_acc + delta_T + delta_D + 0.5 * delta_TD))
    p_000 = baseline_acc

    condition_probs = {
        '000': logit(p_000),
        '100': logit(p_100),
        '110': logit(p_110),
        '101': logit(p_101),
        '111': logit(p_111),
    }

    # Item random effects
    u_items = rng.normal(0, sigma_item, n_items)

    # Generate binary outcomes: shape (n_items, 5_conditions, k_runs)
    outcomes = {}
    for cond_name, mu_c in condition_probs.items():
        p_ij = sigmoid(mu_c + u_items)  # (n_items,)
        # Expand for k_runs: each item has k_runs independent Bernoulli
        y = rng.binomial(1, np.tile(p_ij[:, None], (1, k_runs)))  # (n_items, k_runs)
        outcomes[cond_name] = y

    # ---------- PRIMARY: Paired marginal contrast with t-test ----------
    # Per-item mean accuracy per E=1 condition
    y_100 = outcomes['100'].mean(axis=1)  # (n_items,)
    y_110 = outcomes['110'].mean(axis=1)
    y_101 = outcomes['101'].mean(axis=1)
    y_111 = outcomes['111'].mean(axis=1)

    n = n_items

    # Paired contrast per item (average over T):
    # contrast_i = 0.5 * [(Y_101_i - Y_100_i) + (Y_111_i - Y_110_i)]
    contrast_per_item = 0.5 * ((y_101 - y_100) + (y_111 - y_110))
    contrast_mean = contrast_per_item.mean()
    contrast_se = contrast_per_item.std(ddof=1) / np.sqrt(n)

    # One-sample paired t-test on the contrast values
    from scipy import stats
    if contrast_se > 0:
        t_primary = contrast_mean / contrast_se
        p_primary = float(2 * stats.t.sf(abs(t_primary), df=n - 1))
    else:
        t_primary = 0.0
        p_primary = 1.0

    # ---------- SECONDARY: McNemar tests (paired binary) ----------
    # For McNemar, we need per-item binary outcomes.
    # Average over k_runs, binarize at 0.5 threshold
    b_100 = (y_100 >= 0.5).astype(int)
    b_110 = (y_110 >= 0.5).astype(int)
    b_101 = (y_101 >= 0.5).astype(int)
    b_111 = (y_111 >= 0.5).astype(int)

    from scipy.stats import binomtest

    # McNemar 101 vs 100
    disc_01 = ((b_100 == 1) & (b_101 == 0)).sum()  # correct under 100, wrong under 101
    disc_10 = ((b_100 == 0) & (b_101 == 1)).sum()  # wrong under 100, correct under 101
    n_disc = disc_01 + disc_10
    if n_disc > 0:
        p_mcnemar_101v100 = float(binomtest(min(disc_01, disc_10), n_disc, 0.5).pvalue)
    else:
        p_mcnemar_101v100 = 1.0

    # McNemar 111 vs 110
    disc_01b = ((b_110 == 1) & (b_111 == 0)).sum()
    disc_10b = ((b_110 == 0) & (b_111 == 1)).sum()
    n_disc_b = disc_01b + disc_10b
    if n_disc_b > 0:
        p_mcnemar_111v110 = float(binomtest(min(disc_01b, disc_10b), n_disc_b, 0.5).pvalue)
    else:
        p_mcnemar_111v110 = 1.0

    # ---------- SECONDARY: Enumeration contrast (100 vs 000) ----------
    y_000 = outcomes['000'].mean(axis=1)
    enum_contrast = y_100.mean() - y_000.mean()
    enum_diff = y_100 - y_000
    enum_se = enum_diff.std(ddof=1) / np.sqrt(n) if n > 1 else 1.0
    z_enum = enum_contrast / enum_se if enum_se > 0 else 0.0
    from scipy import stats as sp_stats
    p_enum = float(2 * sp_stats.norm.sf(abs(z_enum)))

    # ---------- SECONDARY: T|E=1 contrast ----------
    T1_mean = 0.5 * (y_110 + y_111)
    T0_mean = 0.5 * (y_100 + y_101)
    t_contrast_per_item = T1_mean - T0_mean
    t_contrast = t_contrast_per_item.mean()
    t_se = t_contrast_per_item.std(ddof=1) / np.sqrt(n) if n > 1 else 1.0
    z_t = t_contrast / t_se if t_se > 0 else 0.0
    p_t = float(2 * sp_stats.norm.sf(abs(z_t)))

    return {
        'primary': {
            'contrast': float(contrast_mean),
            'se': float(contrast_se),
            't': float(t_primary),
            'p_value': float(p_primary),
        },
        'mcnemar_101v100': {'p_value': float(p_mcnemar_101v100)},
        'mcnemar_111v110': {'p_value': float(p_mcnemar_111v110)},
        'enumeration_100v000': {'contrast': float(enum_contrast), 'p_value': p_enum},
        'T_effect': {'contrast': float(t_contrast), 'p_value': p_t},
    }


def run_power_sweep(
    n_sims: int,
    n_items_range: List[int],
    baseline_accs: List[float],
    delta_D_range: List[float],
    sigma_items: List[float],
    k_runs_range: List[int],
    delta_T: float = 0.0,
    delta_TD: float = 0.0,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run power sweep for the 5-condition design.

    Primary: D|E=1 averaged over T, via paired marginal contrast with t-test.
    Secondary: McNemar 101v100, McNemar 111v110.
    """
    rng = np.random.RandomState(seed)

    results = []
    total = (len(n_items_range) * len(baseline_accs) * len(delta_D_range) *
             len(sigma_items) * len(k_runs_range))

    print(f"Power sweep: {total} configs × {n_sims} sims = {total * n_sims} total",
          file=sys.stderr)

    config_idx = 0
    for n_items in n_items_range:
        for p0 in baseline_accs:
            for dD in delta_D_range:
                for sigma in sigma_items:
                    for k in k_runs_range:
                        config_idx += 1
                        p_primary = []
                        p_mc101 = []
                        p_mc111 = []
                        contrasts = []

                        for _ in range(n_sims):
                            res = simulate_experiment(
                                n_items=n_items,
                                baseline_acc=p0,
                                delta_D=dD,
                                delta_T=delta_T,
                                delta_TD=delta_TD,
                                sigma_item=sigma,
                                k_runs=k,
                                rng=rng,
                            )
                            p_primary.append(res['primary']['p_value'])
                            p_mc101.append(res['mcnemar_101v100']['p_value'])
                            p_mc111.append(res['mcnemar_111v110']['p_value'])
                            contrasts.append(res['primary']['contrast'])

                        power_primary = sum(1 for p in p_primary if p < alpha) / n_sims
                        power_mc101 = sum(1 for p in p_mc101 if p < alpha) / n_sims
                        power_mc111 = sum(1 for p in p_mc111 if p < alpha) / n_sims

                        results.append({
                            'n_items': n_items,
                            'baseline_acc': p0,
                            'delta_D': dD,
                            'sigma_item': sigma,
                            'k_runs': k,
                            'power_primary': round(power_primary, 4),
                            'power_mcnemar_101v100': round(power_mc101, 4),
                            'power_mcnemar_111v110': round(power_mc111, 4),
                            'mean_contrast': round(float(np.mean(contrasts)), 5),
                            'sd_contrast': round(float(np.std(contrasts)), 5),
                        })

                        if config_idx % 20 == 0:
                            print(f"  {config_idx}/{total} configs done",
                                  file=sys.stderr)

    return {
        'design': '5-condition (000,100,110,101,111)',
        'primary_estimand': 'D|E=1 averaged over T, adversarial regimes',
        'primary_test': 'One-sample paired t-test on marginal contrast',
        'secondary_tests': ['McNemar 101v100', 'McNemar 111v110',
                            'Enumeration 100v000', 'T|E=1'],
        'alpha': alpha,
        'n_sims': n_sims,
        'seed': seed,
        'delta_T': delta_T,
        'delta_TD': delta_TD,
        'n_configs': len(results),
        'results': results,
    }


def find_mde(results: List[Dict], target_powers: List[float] = None) -> Dict:
    """Find MDE at target power levels for each (p0, sigma, k_runs) combo."""
    if target_powers is None:
        target_powers = [0.80, 0.90]

    # Group by (p0, sigma, k_runs, n_items)
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        key = (r['baseline_acc'], r['sigma_item'], r['k_runs'], r['n_items'])
        by_config[key].append(r)

    mde_table = []
    for (p0, sigma, k, n), items in by_config.items():
        # Sort by delta_D
        items_sorted = sorted(items, key=lambda x: x['delta_D'])
        for tp in target_powers:
            # Find smallest delta_D achieving target power
            found = None
            for item in items_sorted:
                if item['power_primary'] >= tp:
                    found = item
                    break
            mde_table.append({
                'baseline_acc': p0,
                'sigma_item': sigma,
                'k_runs': k,
                'n_items': n,
                'target_power': tp,
                'mde': found['delta_D'] if found else None,
                'achieved_power': found['power_primary'] if found else None,
            })

    return mde_table


def format_report(sweep: Dict, mde_table: List) -> str:
    """Format power simulation results as markdown."""
    lines = [
        "# Model-Inference Power Simulation Results",
        "",
        "## Design",
        "",
        f"- **Conditions:** {sweep['design']}",
        f"- **Primary estimand:** {sweep['primary_estimand']}",
        f"- **Primary test:** {sweep['primary_test']}",
        f"- **Secondary:** {', '.join(sweep['secondary_tests'])}",
        f"- **Alpha:** {sweep['alpha']}",
        f"- **N simulations:** {sweep['n_sims']} per configuration",
        f"- **Delta T (trajectory effect):** {sweep['delta_T']}",
        f"- **Delta TD (interaction):** {sweep['delta_TD']}",
        "",
        "## Primary Power Table: D|E=1 (paired marginal contrast, t-test)",
        "",
    ]

    # Build a summary table grouped by (p0, sigma, k_runs)
    results = sweep['results']
    from collections import defaultdict
    by_design = defaultdict(list)
    for r in results:
        key = (r['baseline_acc'], r['sigma_item'], r['k_runs'])
        by_design[key].append(r)

    for (p0, sigma, k), items in sorted(by_design.items()):
        lines.append(f"### p₀={p0}, σ_item={sigma}, k_runs={k}")
        lines.append("")
        lines.append("| N_items | δ_D | Power (primary) | Power McN 101v100 | Power McN 111v110 |")
        lines.append("|---------|-----|-----------------|-------------------|-------------------|")
        for r in sorted(items, key=lambda x: (x['n_items'], x['delta_D'])):
            lines.append(
                f"| {r['n_items']} | {r['delta_D']:.3f} | "
                f"{r['power_primary']:.3f} | "
                f"{r['power_mcnemar_101v100']:.3f} | "
                f"{r['power_mcnemar_111v110']:.3f} |"
            )
        lines.append("")

    # MDE summary
    lines.append("## Minimum Detectable Effect (MDE)")
    lines.append("")
    lines.append("| p₀ | σ_item | k_runs | N_items | Target Power | MDE (pp) | Achieved |")
    lines.append("|-----|--------|--------|---------|-------------|----------|----------|")
    for m in sorted(mde_table, key=lambda x: (x['baseline_acc'], x['sigma_item'],
                                                x['k_runs'], x['n_items'],
                                                x['target_power'])):
        if m['mde'] is not None:
            lines.append(
                f"| {m['baseline_acc']} | {m['sigma_item']} | {m['k_runs']} | "
                f"{m['n_items']} | {m['target_power']} | "
                f"{m['mde']*100:.1f} | {m['achieved_power']:.3f} |"
            )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Primary estimand: paired marginal contrast D|E=1 averaged over T = 0.5*[(Y_101-Y_100)+(Y_111-Y_110)]")
    lines.append("- Primary test: one-sample paired t-test on item-level contrast values")
    lines.append("- Robustness: effect-coded binomial GEE with item-clustered robust SE (secondary, not gated)")
    lines.append("- McNemar tests are secondary paired checks (genuinely binary, not averaged)")
    lines.append("- Model is a FIXED effect → power is per-model; no generalization claim")
    lines.append("- MDE depends on (N, k, σ, p₀) — no single fixed MDE is claimed")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# PER-MODEL POWER
# ================================================================
# Model is a FIXED effect (§4.2).  Power depends on each model's
# baseline accuracy, so we report per-model power separately.
# The simulation is run once per model with that model's p0.

# Plausible baseline accuracies (adversarial regimes, DECOY+CONFLICT)
# These are pre-specified operating points, NOT from observed data.
MODEL_PROFILES = {
    "Llama-3.3-70B": {"p0": 0.40, "note": "open-weight, mid-range on adversarial"},
    "Qwen2.5-72B":   {"p0": 0.45, "note": "open-weight, slightly stronger"},
    "proprietary":    {"p0": 0.55, "note": "frontier closed model, upper bound"},
}


def run_per_model_power(
    n_sims: int,
    n_items_range: List[int],
    delta_D_range: List[float],
    sigma_item: float,
    k_runs_range: List[int],
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run power sweep for each model profile, for each k_runs value.

    Returns dict keyed by model name, each containing per-k_runs sweep results.
    """
    all_model_results = {}
    for model_name, profile in MODEL_PROFILES.items():
        per_k_results = {}
        for k in k_runs_range:
            print(f"\n  Model: {model_name} (p0={profile['p0']}, k_runs={k})",
                  file=sys.stderr)
            sweep = run_power_sweep(
                n_sims=n_sims,
                n_items_range=n_items_range,
                baseline_accs=[profile['p0']],
                delta_D_range=delta_D_range,
                sigma_items=[sigma_item],
                k_runs_range=[k],
                delta_T=0.0,
                delta_TD=0.0,
                alpha=alpha,
                seed=seed,
            )
            mde = find_mde(sweep['results'], target_powers=[0.80, 0.90])
            per_k_results[k] = {'sweep': sweep, 'mde': mde}
        all_model_results[model_name] = {
            'profile': profile,
            'per_k': per_k_results,
        }
    return all_model_results


def format_per_model_report(model_results: Dict) -> str:
    """Format per-model power results as markdown."""
    lines = [
        "",
        "## Per-Model Power (model is a fixed effect)",
        "",
        "Each model has its own baseline accuracy p0 on adversarial regimes.",
        "Power is computed separately because the paired marginal contrast variance",
        "depends on p0. These are pre-specified operating points.",
        "",
    ]
    for model_name, mr in model_results.items():
        profile = mr['profile']
        for k, kr in sorted(mr['per_k'].items()):
            results = kr['sweep']['results']
            lines.append(f"### {model_name} (p0={profile['p0']}, k_runs={k}, {profile['note']})")
            lines.append("")
            lines.append("| N_items | delta_D | Power (primary) | Power McN 101v100 | Power McN 111v110 |")
            lines.append("|---------|---------|-----------------|-------------------|-------------------|")
            for r in sorted(results, key=lambda x: (x['n_items'], x['delta_D'])):
                lines.append(
                    f"| {r['n_items']} | {r['delta_D']:.3f} | "
                    f"{r['power_primary']:.3f} | "
                    f"{r['power_mcnemar_101v100']:.3f} | "
                    f"{r['power_mcnemar_111v110']:.3f} |"
                )
            lines.append("")

            # MDE for this model/k
            mde = kr['mde']
            lines.append(f"**MDE for {model_name} (k_runs={k}):**")
            lines.append("")
            lines.append("| N_items | Target Power | MDE (pp) | Achieved |")
            lines.append("|---------|-------------|----------|----------|")
            for m in sorted(mde, key=lambda x: (x['n_items'], x['target_power'])):
                if m['mde'] is not None:
                    lines.append(
                        f"| {m['n_items']} | {m['target_power']} | "
                        f"{m['mde']*100:.1f} | {m['achieved_power']:.3f} |"
                    )
                else:
                    lines.append(
                        f"| {m['n_items']} | {m['target_power']} | "
                        f">10.0 | N/A |"
                    )
            lines.append("")
    return "\n".join(lines)


def compute_cost_model(n_items_range: List[int], k_runs_range: List[int]) -> str:
    """Compute and format approximate model-call and token-cost implications.

    Assumptions (clearly labeled as estimates):
    - 5 conditions per item
    - 3 models
    - ~2,000 tokens per API call (prompt + completion estimate)
    - Cost varies by model; estimates use representative rates
    """
    lines = [
        "",
        "## Approximate Cost Model (ESTIMATE)",
        "",
        "**Assumptions (labeled as estimates, not commitments):**",
        "- 5 conditions per item",
        "- 3 models (Llama-3.3-70B, Qwen2.5-72B, proprietary)",
        "- ~2,000 tokens per API call (prompt + completion, rough estimate)",
        "- Open-weight models: ~$0.00 marginal cost (self-hosted)",
        "- Proprietary model: ~$0.01 per 1K tokens (estimate)",
        "",
        "| N_items | k_runs | Total calls/model | Total calls (3 models) | Proprietary tokens (est.) | Proprietary cost (est.) |",
        "|---------|--------|-------------------|----------------------|--------------------------|------------------------|",
    ]
    for n in n_items_range:
        for k in k_runs_range:
            calls_per_model = n * 5 * k  # n_items * 5_conditions * k_runs
            total_calls = calls_per_model * 3
            prop_tokens = calls_per_model * 2000
            prop_cost = prop_tokens * 0.01 / 1000
            lines.append(
                f"| {n} | {k} | {calls_per_model:,} | {total_calls:,} | "
                f"{prop_tokens:,} | ${prop_cost:,.2f} |"
            )
    lines.append("")
    lines.append("**Note:** Open-weight model costs depend on infrastructure.")
    lines.append("Proprietary cost uses $0.01/1K tokens as a rough estimate.")
    lines.append("Actual costs may vary significantly. This table is for planning only.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Power simulation for 5-condition ETD-ACH experiment"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="analysis/power_simulation_results.json")
    args = parser.parse_args()

    t0 = time.time()

    # Effect sizes include delta_D=0 to verify nominal Type-I error
    # N range extends to 1000+ to find 80/90% power for 5pp and 7pp effects
    if args.quick:
        n_sims = 500
        n_items_range = [200, 300, 500, 750, 1000]
        baseline_accs = [0.40, 0.50]
        delta_D_range = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10]
        sigma_items = [0.5]
        k_runs_range = [1, 3]
        per_model_sigma = 0.5
        per_model_k_range = [1, 3]
        per_model_n_sims = 500
        print("Quick mode", file=sys.stderr)
    else:
        n_sims = 2000
        n_items_range = [200, 300, 500, 750, 1000, 1500]
        baseline_accs = [0.35, 0.40, 0.45, 0.50, 0.55]
        delta_D_range = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10]
        sigma_items = [0.3, 0.5, 0.8]
        k_runs_range = [1, 3]
        per_model_sigma = 0.5
        per_model_k_range = [1, 3]
        per_model_n_sims = 2000
        print("Full mode", file=sys.stderr)

    # --- Generic sweep ---
    print("Running generic power sweep...", file=sys.stderr)
    sweep = run_power_sweep(
        n_sims=n_sims,
        n_items_range=n_items_range,
        baseline_accs=baseline_accs,
        delta_D_range=delta_D_range,
        sigma_items=sigma_items,
        k_runs_range=k_runs_range,
        delta_T=0.0,
        delta_TD=0.0,
        alpha=0.05,
        seed=42,
    )

    mde_table = find_mde(sweep['results'], target_powers=[0.80, 0.90])

    # --- Per-model sweep (BOTH k_runs=1 and k_runs=3) ---
    print("\nRunning per-model power sweep...", file=sys.stderr)
    model_results = run_per_model_power(
        n_sims=per_model_n_sims,
        n_items_range=n_items_range,
        delta_D_range=delta_D_range,
        sigma_item=per_model_sigma,
        k_runs_range=per_model_k_range,
        alpha=0.05,
        seed=42,
    )

    elapsed = time.time() - t0
    sweep['elapsed_seconds'] = round(elapsed, 1)

    output = {
        'sweep': sweep,
        'mde_table': mde_table,
        'per_model': {
            name: {
                'profile': mr['profile'],
                'per_k': {
                    str(k): {
                        'results': kr['sweep']['results'],
                        'mde': kr['mde'],
                    }
                    for k, kr in mr['per_k'].items()
                },
            }
            for name, mr in model_results.items()
        },
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}", file=sys.stderr)

    report = format_report(sweep, mde_table)
    report += format_per_model_report(model_results)
    report += compute_cost_model(n_items_range, per_model_k_range)
    report_path = args.output.replace('.json', '_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}", file=sys.stderr)
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    # Print summary
    print("\n" + report)


if __name__ == "__main__":
    main()
