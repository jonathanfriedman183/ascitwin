from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from ascitwin.interpretation import (
    bootstrap_ci,
    calibration_error,
    cohens_d,
    cross_cohort_degradation,
    delong_test,
    interaction_ratio,
    oneway_anova,
    roc_auc,
    spatial_concordance,
)


def _data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=200).astype(np.float64)
    scores = 0.6 * labels + rng.normal(0.0, 1.0, size=200)
    return scores, labels


def test_roc_auc_matches_sklearn() -> None:
    scores, labels = _data(0)
    assert abs(roc_auc(scores, labels) - float(roc_auc_score(labels, scores))) < 1e-9


def test_bootstrap_interval_brackets_point_estimate() -> None:
    scores, labels = _data(1)
    point = roc_auc(scores, labels)
    low, high = bootstrap_ci(scores, labels, n_resamples=300, seed=3)
    assert low <= point <= high


def test_delong_recovers_auc_and_valid_pvalue() -> None:
    scores, labels = _data(2)
    other, _ = _data(5)
    z, p = delong_test(scores, other, labels)
    assert 0.0 <= p <= 1.0
    assert np.isfinite(z)


def test_delong_identical_scores_is_insignificant() -> None:
    scores, labels = _data(4)
    _, p = delong_test(scores, scores.copy(), labels)
    assert p > 0.99


def test_cohens_d_sign() -> None:
    high = np.array([3.0, 4.0, 5.0])
    low = np.array([0.0, 1.0, 2.0])
    assert cohens_d(high, low) > 0.0


def test_spatial_concordance_perfect() -> None:
    grid = np.random.default_rng(0).random((10, 10))
    pearson, spearman = spatial_concordance(grid, grid)
    assert pearson > 0.999
    assert spearman > 0.999


def test_oneway_anova_matches_scipy() -> None:
    groups = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([1.5, 2.5, 0.5])]
    f_value, p_value = oneway_anova(groups)
    reference = stats.f_oneway(*groups)
    assert abs(f_value - reference.statistic) < 1e-9
    assert abs(p_value - reference.pvalue) < 1e-9


def test_calibration_error_of_perfect_predictor_is_small() -> None:
    labels = np.array([0.0, 1.0] * 50)
    probs = labels.copy()
    assert calibration_error(probs, labels) < 1e-6


def test_degradation_and_interaction_ratio() -> None:
    assert abs(cross_cohort_degradation(0.9, 0.864) - (1.0 - 0.864 / 0.9)) < 1e-9
    assert abs(interaction_ratio(-0.107, -0.055, -0.024) - (-0.107 / -0.079)) < 1e-6
