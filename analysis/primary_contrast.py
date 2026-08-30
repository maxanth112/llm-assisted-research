#!/usr/bin/env python3
"""
Primary contrast analysis for the ETD-ACH 5-condition experiment.

Implements the pre-specified primary estimator (AMENDMENT-002 §4.2):

  Primary estimand:
    Paired marginal contrast for the D effect, conditional on E=1,
    in adversarial regimes (DECOY+CONFLICT), averaged over T:
      contrast_i = 0.5 * [(Y_101_i - Y_100_i) + (Y_111_i - Y_110_i)]
    where i indexes items. The estimand is E[contrast_i].

  Primary test:
    One-sample paired t-test on the item-level contrast values.

  Primary CI:
    Frozen-seed, item-clustered paired percentile bootstrap (10,000
    resamples) on the mean contrast.

  Robustness (secondary, not gated):
    Effect-coded binomial GEE with item-clustered robust SE.

Usage:
  from analysis.primary_contrast import (
      compute_primary_contrast,
      paired_bootstrap_ci,
      paired_t_test,
  )
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def compute_primary_contrast(
    y_100: np.ndarray,
    y_110: np.ndarray,
    y_101: np.ndarray,
    y_111: np.ndarray,
) -> np.ndarray:
    """Compute per-item primary contrast.

    Args:
        y_100, y_110, y_101, y_111: Per-item mean accuracy arrays (shape: n_items).
            Each element is the mean accuracy for that item under the condition,
            averaged over k_runs if k_runs > 1.

    Returns:
        Per-item contrast values: 0.5 * [(Y_101 - Y_100) + (Y_111 - Y_110)]
    """
    return 0.5 * ((y_101 - y_100) + (y_111 - y_110))


def paired_t_test(contrast_per_item: np.ndarray) -> Dict[str, float]:
    """One-sample paired t-test on item-level contrast values.

    Tests H0: E[contrast_i] = 0 (two-sided).

    Args:
        contrast_per_item: Per-item contrast values from compute_primary_contrast.

    Returns:
        Dict with: mean, se, t_statistic, p_value, n_items, df
    """
    from scipy import stats

    n = len(contrast_per_item)
    mean = float(contrast_per_item.mean())
    se = float(contrast_per_item.std(ddof=1) / np.sqrt(n))

    if se > 0:
        t_stat = mean / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        t_stat = 0.0
        p_value = 1.0

    return {
        'mean': mean,
        'se': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'n_items': n,
        'df': n - 1,
    }


def paired_bootstrap_ci(
    contrast_per_item: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Frozen-seed, item-clustered paired percentile bootstrap CI.

    Resamples ITEMS (not individual observations), preserving the pairing
    structure. Each bootstrap replicate resamples n items with replacement
    and computes the mean of their contrast values.

    Args:
        contrast_per_item: Per-item contrast values from compute_primary_contrast.
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        alpha: Significance level for CI (default 0.05 -> 95% CI).
        seed: Fixed seed for reproducibility.

    Returns:
        Dict with: ci_lower, ci_upper, bootstrap_mean, bootstrap_se,
                   n_bootstrap, seed, alpha
    """
    rng = np.random.RandomState(seed)
    n = len(contrast_per_item)

    bootstrap_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means[b] = contrast_per_item[indices].mean()

    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_mean': float(bootstrap_means.mean()),
        'bootstrap_se': float(bootstrap_means.std()),
        'n_bootstrap': n_bootstrap,
        'seed': seed,
        'alpha': alpha,
    }


def full_primary_analysis(
    y_100: np.ndarray,
    y_110: np.ndarray,
    y_101: np.ndarray,
    y_111: np.ndarray,
    bootstrap_seed: int = 42,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run the complete primary contrast analysis.

    Returns paired t-test results and bootstrap CI in one call.

    Args:
        y_100, y_110, y_101, y_111: Per-item mean accuracy arrays.
        bootstrap_seed: Fixed seed for bootstrap reproducibility.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Dict with 'contrast_per_item', 't_test', 'bootstrap_ci' keys.
    """
    contrast = compute_primary_contrast(y_100, y_110, y_101, y_111)
    t_result = paired_t_test(contrast)
    boot_ci = paired_bootstrap_ci(
        contrast,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    return {
        'contrast_per_item': contrast.tolist(),
        't_test': t_result,
        'bootstrap_ci': boot_ci,
    }
