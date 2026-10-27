from __future__ import annotations

import numpy as np

from ascitwin.identifiability import (
    finite_difference_jacobian,
    fisher_condition_number,
    is_identifiable,
    profile_interval,
)


def test_jacobian_of_linear_map() -> None:
    matrix = np.array([[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]])

    def predict(theta: np.ndarray) -> np.ndarray:
        return matrix @ theta

    jacobian = finite_difference_jacobian(predict, np.array([0.5, 0.5]))
    assert np.allclose(jacobian, matrix, atol=1e-4)


def test_condition_number_finite_for_full_rank() -> None:
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    kappa = fisher_condition_number(jacobian)
    assert np.isfinite(kappa)
    assert kappa >= 1.0


def test_condition_number_infinite_for_rank_deficient() -> None:
    jacobian = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert fisher_condition_number(jacobian) == float("inf")


def test_profile_interval_brackets_optimum() -> None:
    centre = np.array([1.0])

    def loss(theta: np.ndarray) -> float:
        return float((theta[0] - 1.0) ** 2)

    low, high = profile_interval(loss, centre, 0, (-5.0, 5.0))
    assert low < 1.0 < high


def test_flat_parameter_is_not_identifiable() -> None:
    assert is_identifiable((-5.0, 5.0), (-5.0, 5.0)) is False
    assert is_identifiable((0.5, 1.5), (-5.0, 5.0)) is True
