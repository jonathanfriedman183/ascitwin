from __future__ import annotations

import numpy as np
import pytest

from ascitwin.gathers import (
    CIBERSORT_CELL_TYPES,
    OBSERVATION_DIM,
    PATHWAY_SCORES,
    AggregateObservation,
    SpatialGrid,
)


def _observation() -> AggregateObservation:
    fractions = np.full(len(CIBERSORT_CELL_TYPES), 1.0 / len(CIBERSORT_CELL_TYPES))
    pathways = np.array([0.4, 0.3, 0.2])
    return AggregateObservation(fractions=fractions, pathways=pathways)


def test_observation_vector_length() -> None:
    vector = _observation().vector()
    assert vector.shape == (OBSERVATION_DIM,)
    assert len(CIBERSORT_CELL_TYPES) + len(PATHWAY_SCORES) == OBSERVATION_DIM


def test_wrong_fraction_length_raises() -> None:
    with pytest.raises(ValueError):
        AggregateObservation(fractions=np.zeros(3), pathways=np.zeros(len(PATHWAY_SCORES)))


def test_spatial_grid_cell_area() -> None:
    grid = SpatialGrid(extent=1.0, resolution=50, density={"tumour": np.zeros((50, 50))})
    assert grid.cell_area() == pytest.approx((1.0 / 50.0) ** 2)
    assert grid.map_for("tumour").shape == (50, 50)
