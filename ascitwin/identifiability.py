from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray

CHI2_1_95 = 3.841458820694124


def finite_difference_jacobian(
    predict: Callable[[FloatArray], FloatArray], theta: FloatArray, eps: float = 1e-4
) -> FloatArray:
    base = predict(theta)
    columns: list[FloatArray] = []
    for index in range(theta.shape[0]):
        shifted = theta.copy()
        step = eps * max(abs(theta[index]), 1.0)
        shifted[index] += step
        derivative = (predict(shifted) - base) / step
        columns.append(derivative.astype(np.float64))
    return cast(FloatArray, np.stack(columns, axis=1))


def fisher_information(jacobian: FloatArray, noise_variance: float = 1.0) -> FloatArray:
    return cast(FloatArray, (jacobian.T @ jacobian) / noise_variance)


def fisher_condition_number(jacobian: FloatArray, noise_variance: float = 1.0) -> float:
    fim = fisher_information(jacobian, noise_variance)
    eigenvalues = np.linalg.eigvalsh(fim)
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def profile_interval(
    loss: Callable[[FloatArray], float],
    theta: FloatArray,
    index: int,
    bounds: tuple[float, float],
    threshold: float = CHI2_1_95,
    n_steps: int = 64,
) -> tuple[float, float]:
    base_loss = loss(theta)
    centre = theta[index]
    low, high = bounds
    down = _scan(loss, theta, index, np.linspace(centre, low, n_steps), base_loss, threshold)
    up = _scan(loss, theta, index, np.linspace(centre, high, n_steps), base_loss, threshold)
    return down, up


def _scan(
    loss: Callable[[FloatArray], float],
    theta: FloatArray,
    index: int,
    grid: FloatArray,
    base_loss: float,
    threshold: float,
) -> float:
    last = float(grid[0])
    for value in grid:
        probe = theta.copy()
        probe[index] = value
        if 2.0 * (loss(probe) - base_loss) > threshold:
            return float(value)
        last = float(value)
    return last


def profile_likelihood_intervals(
    loss: Callable[[FloatArray], float],
    theta: FloatArray,
    bounds: list[tuple[float, float]],
    threshold: float = CHI2_1_95,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for index in range(theta.shape[0]):
        intervals.append(profile_interval(loss, theta, index, bounds[index], threshold))
    return intervals


def is_identifiable(interval: tuple[float, float], bounds: tuple[float, float]) -> bool:
    low, high = interval
    span_low, span_high = bounds
    touches_low = low <= span_low + 1e-9
    touches_high = high >= span_high - 1e-9
    return not (touches_low and touches_high)
