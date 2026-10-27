from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray

SPECIES: tuple[str, ...] = ("tumour", "immune", "cytokine", "drug", "pdl1")

CIBERSORT_CELL_TYPES: tuple[str, ...] = (
    "b_naive",
    "b_memory",
    "plasma",
    "cd8_t",
    "cd4_naive",
    "cd4_memory_resting",
    "cd4_memory_activated",
    "t_follicular_helper",
    "t_regulatory",
    "t_gamma_delta",
    "nk_resting",
    "nk_activated",
    "monocyte",
    "macrophage_m0",
    "macrophage_m1",
    "macrophage_m2",
    "dendritic_resting",
    "dendritic_activated",
    "mast_resting",
    "mast_activated",
    "eosinophil",
    "neutrophil",
)

PATHWAY_SCORES: tuple[str, ...] = ("ifn_gamma", "tgf_beta", "angiogenesis")

OBSERVATION_DIM: int = len(CIBERSORT_CELL_TYPES) + len(PATHWAY_SCORES)

CMS_LABELS: tuple[str, ...] = ("CMS1", "CMS2", "CMS3", "CMS4", "Unclassified")


@dataclass(frozen=True)
class AggregateObservation:
    fractions: FloatArray
    pathways: FloatArray

    def vector(self) -> FloatArray:
        return cast(FloatArray, np.concatenate([self.fractions, self.pathways]).astype(np.float64))

    def __post_init__(self) -> None:
        if self.fractions.shape[0] != len(CIBERSORT_CELL_TYPES):
            raise ValueError("fraction vector has the wrong length")
        if self.pathways.shape[0] != len(PATHWAY_SCORES):
            raise ValueError("pathway vector has the wrong length")


@dataclass(frozen=True)
class PatientSample:
    patient_id: str
    cohort: str
    observation: AggregateObservation
    cms: str
    msi_high: bool
    tmb_per_mb: float
    cd8_fraction: float
    gep_score: float
    treated: bool
    responder: int


@dataclass(frozen=True)
class SpatialGrid:
    extent: float
    resolution: int
    density: dict[str, FloatArray] = field(default_factory=dict)

    def cell_area(self) -> float:
        step = self.extent / float(self.resolution)
        return step * step

    def map_for(self, species: str) -> FloatArray:
        return self.density[species]


def stack_observations(samples: list[PatientSample]) -> FloatArray:
    rows = [sample.observation.vector() for sample in samples]
    return np.stack(rows, axis=0).astype(np.float64)


def response_labels(samples: list[PatientSample]) -> FloatArray:
    return np.asarray([float(sample.responder) for sample in samples], dtype=np.float64)
