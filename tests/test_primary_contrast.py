"""
Tests for the primary contrast analysis module (AMENDMENT-002 §4.2).

Verifies:
- Contrast computation correctness
- Paired t-test against scipy reference
- Bootstrap CI determinism (frozen seed)
- Bootstrap CI coverage on known-effect synthetic data
"""

import numpy as np
import pytest

from analysis.primary_contrast import (
    compute_primary_contrast,
    paired_t_test,
    paired_bootstrap_ci,
    full_primary_analysis,
)


class TestComputePrimaryContrast:
    """Verify contrast_i = 0.5 * [(Y_101 - Y_100) + (Y_111 - Y_110)]."""

    def test_zero_effect(self):
        """When all conditions are equal, contrast is zero."""
        n = 50
        y = np.ones(n) * 0.5
        contrast = compute_primary_contrast(y, y, y, y)
        np.testing.assert_array_almost_equal(contrast, np.zeros(n))

    def test_known_effect(self):
        """Manual computation check."""
        y_100 = np.array([0.0, 1.0, 0.0])
        y_110 = np.array([0.0, 0.0, 1.0])
        y_101 = np.array([1.0, 1.0, 0.0])
        y_111 = np.array([1.0, 0.0, 1.0])

        contrast = compute_primary_contrast(y_100, y_110, y_101, y_111)
        # Item 0: 0.5*[(1-0)+(1-0)] = 1.0
        # Item 1: 0.5*[(1-1)+(0-0)] = 0.0
        # Item 2: 0.5*[(0-0)+(1-1)] = 0.0
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(contrast, expected)

    def test_symmetric(self):
        """D effect of +0.1 should produce contrast of ~0.1."""
        rng = np.random.RandomState(42)
        n = 1000
        base = 0.4
        delta_D = 0.1
        y_100 = rng.binomial(1, base, n).astype(float)
        y_110 = rng.binomial(1, base, n).astype(float)
        y_101 = rng.binomial(1, base + delta_D, n).astype(float)
        y_111 = rng.binomial(1, base + delta_D, n).astype(float)

        contrast = compute_primary_contrast(y_100, y_110, y_101, y_111)
        assert abs(contrast.mean() - delta_D) < 0.05  # within 5pp of true effect


class TestPairedTTest:
    """Verify paired t-test against scipy reference."""

    def test_scipy_agreement(self):
        """Should match scipy.stats.ttest_1samp."""
        from scipy.stats import ttest_1samp

        rng = np.random.RandomState(42)
        contrast = rng.normal(0.05, 0.3, 100)

        result = paired_t_test(contrast)
        scipy_result = ttest_1samp(contrast, 0)

        assert abs(result['t_statistic'] - scipy_result.statistic) < 1e-10
        assert abs(result['p_value'] - scipy_result.pvalue) < 1e-10

    def test_zero_contrast(self):
        """Zero contrast should yield p near 1."""
        contrast = np.zeros(50)
        result = paired_t_test(contrast)
        assert result['p_value'] == 1.0
        assert result['mean'] == 0.0


class TestPairedBootstrapCI:
    """Verify bootstrap CI properties."""

    def test_determinism(self):
        """Same seed should produce identical CI."""
        rng = np.random.RandomState(42)
        contrast = rng.normal(0.05, 0.3, 100)

        ci1 = paired_bootstrap_ci(contrast, seed=42)
        ci2 = paired_bootstrap_ci(contrast, seed=42)

        assert ci1['ci_lower'] == ci2['ci_lower']
        assert ci1['ci_upper'] == ci2['ci_upper']

    def test_different_seed_different_ci(self):
        """Different seeds should produce different CIs."""
        rng = np.random.RandomState(42)
        contrast = rng.normal(0.05, 0.3, 100)

        ci1 = paired_bootstrap_ci(contrast, seed=42)
        ci2 = paired_bootstrap_ci(contrast, seed=99)

        # At least one bound should differ
        assert ci1['ci_lower'] != ci2['ci_lower'] or ci1['ci_upper'] != ci2['ci_upper']

    def test_coverage_on_known_effect(self):
        """95% CI should cover the true mean ~95% of the time."""
        true_mean = 0.05
        n_items = 100
        n_trials = 200  # number of experiments to check coverage
        covered = 0

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            contrast = rng.normal(true_mean, 0.3, n_items)
            ci = paired_bootstrap_ci(contrast, seed=trial + 10000)
            if ci['ci_lower'] <= true_mean <= ci['ci_upper']:
                covered += 1

        coverage = covered / n_trials
        # Should be near 0.95, allow 0.88 to 1.0 range for finite-sample variability
        assert 0.88 <= coverage <= 1.0, f"Coverage {coverage:.3f} out of range"

    def test_ci_contains_sample_mean(self):
        """CI should contain the sample mean for reasonable data."""
        rng = np.random.RandomState(42)
        contrast = rng.normal(0.1, 0.2, 200)
        ci = paired_bootstrap_ci(contrast, seed=42)
        sample_mean = contrast.mean()
        assert ci['ci_lower'] <= sample_mean <= ci['ci_upper']


class TestFullPrimaryAnalysis:
    """Verify the combined analysis function."""

    def test_returns_all_components(self):
        """Should return t_test, bootstrap_ci, and contrast values."""
        rng = np.random.RandomState(42)
        n = 50
        y_100 = rng.binomial(1, 0.4, n).astype(float)
        y_110 = rng.binomial(1, 0.4, n).astype(float)
        y_101 = rng.binomial(1, 0.5, n).astype(float)
        y_111 = rng.binomial(1, 0.5, n).astype(float)

        result = full_primary_analysis(y_100, y_110, y_101, y_111)

        assert 't_test' in result
        assert 'bootstrap_ci' in result
        assert 'contrast_per_item' in result
        assert len(result['contrast_per_item']) == n

        # t-test should have required keys
        assert 'p_value' in result['t_test']
        assert 'mean' in result['t_test']

        # bootstrap should have CI bounds
        assert 'ci_lower' in result['bootstrap_ci']
        assert 'ci_upper' in result['bootstrap_ci']
