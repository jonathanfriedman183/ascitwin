from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray


def roc_auc(scores: FloatArray, labels: FloatArray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty_like(order, dtype=np.float64)
    ranked[order] = _average_ranks(scores[order])
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(ranked[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_ranks(sorted_scores: FloatArray) -> FloatArray:
    n = sorted_scores.shape[0]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    start = 0
    for index in range(1, n + 1):
        if index == n or sorted_scores[index] != sorted_scores[start]:
            ranks[start:index] = (start + 1 + index) / 2.0
            start = index
    return ranks


def classification_metrics(
    probs: FloatArray, labels: FloatArray, threshold: float = 0.5
) -> dict[str, float]:
    predicted = (probs >= threshold).astype(np.float64)
    truth = labels.astype(np.float64)
    tp = float(((predicted == 1) & (truth == 1)).sum())
    tn = float(((predicted == 0) & (truth == 0)).sum())
    fp = float(((predicted == 1) & (truth == 0)).sum())
    fn = float(((predicted == 0) & (truth == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn > 0 else float("nan")
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )
    accuracy = (tp + tn) / float(len(labels))
    return {
        "auc": roc_auc(probs, labels),
        "f1": f1,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def bootstrap_ci(
    scores: FloatArray,
    labels: FloatArray,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    size = len(labels)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        draw = rng.integers(0, size, size=size)
        estimates[index] = roc_auc(scores[draw], labels[draw])
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low = float(np.quantile(finite, alpha / 2.0))
    high = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return low, high


def cohens_d(group_a: FloatArray, group_b: FloatArray) -> float:
    n_a = len(group_a)
    n_b = len(group_b)
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    spread = float(np.sqrt(pooled)) + 1e-12
    return (float(np.mean(group_a)) - float(np.mean(group_b))) / spread


def _compute_midrank(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    midrank = np.zeros(n, dtype=np.float64)
    index = 0
    while index < n:
        run = index
        while run < n and sorted_values[run] == sorted_values[index]:
            run += 1
        midrank[index:run] = 0.5 * (index + run - 1) + 1
        index = run
    result = np.empty(n, dtype=np.float64)
    result[order] = midrank
    return result


def _fast_delong(positive: FloatArray, negative: FloatArray) -> tuple[FloatArray, FloatArray]:
    m = positive.shape[1]
    n = negative.shape[1]
    k = positive.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for row in range(k):
        tx[row] = _compute_midrank(positive[row])
        ty[row] = _compute_midrank(negative[row])
        tz[row] = _compute_midrank(np.concatenate([positive[row], negative[row]]))
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return auc, np.atleast_2d(covariance)


def delong_test(
    scores_a: FloatArray, scores_b: FloatArray, labels: FloatArray
) -> tuple[float, float]:
    order = np.argsort(-labels)
    label_sorted = labels[order]
    n_pos = int(label_sorted.sum())
    stacked = np.vstack([scores_a[order], scores_b[order]])
    positive = stacked[:, :n_pos]
    negative = stacked[:, n_pos:]
    auc, covariance = _fast_delong(positive, negative)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T)[0, 0])
    if variance <= 0.0:
        return 0.0, 1.0
    z = float((auc[0] - auc[1]) / np.sqrt(variance))
    p = float(2.0 * stats.norm.sf(abs(z)))
    return z, p


def spatial_concordance(predicted: FloatArray, truth: FloatArray) -> tuple[float, float]:
    flat_pred = predicted.reshape(-1)
    flat_true = truth.reshape(-1)
    pearson = float(stats.pearsonr(flat_pred, flat_true).statistic)
    spearman = float(stats.spearmanr(flat_pred, flat_true).statistic)
    return pearson, spearman


def radial_residual(predicted: FloatArray, truth: FloatArray) -> float:
    resolution = predicted.shape[0]
    axis = (np.arange(resolution) + 0.5) / resolution
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2)
    bins = np.linspace(0.0, radius.max() + 1e-9, 12)
    digitised = np.digitize(radius.reshape(-1), bins)
    pred_flat = predicted.reshape(-1).copy()
    true_flat = truth.reshape(-1).copy()
    for level in np.unique(digitised):
        mask = digitised == level
        pred_flat[mask] -= pred_flat[mask].mean()
        true_flat[mask] -= true_flat[mask].mean()
    return float(stats.pearsonr(pred_flat, true_flat).statistic)


def oneway_anova(groups: list[FloatArray]) -> tuple[float, float]:
    result = stats.f_oneway(*groups)
    return float(result.statistic), float(result.pvalue)


def calibration_error(probs: FloatArray, labels: FloatArray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(labels))
    error = 0.0
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (probs >= lower) & (probs < upper if index < n_bins - 1 else probs <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        error += (count / total) * abs(confidence - accuracy)
    return error


def interval_coverage(lower: FloatArray, upper: FloatArray, truth: FloatArray) -> float:
    inside = (truth >= lower) & (truth <= upper)
    return float(inside.mean())


def ablation_delta(full_auc: float, reduced_auc: float) -> float:
    return reduced_auc - full_auc


def cross_cohort_degradation(internal_auc: float, external_auc: float) -> float:
    return 1.0 - external_auc / internal_auc


def interaction_ratio(delta_joint: float, delta_a: float, delta_b: float) -> float:
    individual = delta_a + delta_b
    if individual == 0.0:
        return float("nan")
    return delta_joint / individual
