from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import nnls

from ascitwin.gathers import (
    CIBERSORT_CELL_TYPES,
    CMS_LABELS,
    PATHWAY_SCORES,
    AggregateObservation,
    PatientSample,
    SpatialGrid,
)
from ascitwin.surrogate import CiaiThresholds, is_responder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray


@dataclass(frozen=True)
class CohortSpec:
    name: str
    n: int
    platform: str
    license: str
    access: str
    treated: bool


COHORTS: tuple[CohortSpec, ...] = (
    CohortSpec("TCGA-COAD/READ", 623, "rna_seq", "open", "GDC Data Portal", False),
    CohortSpec("GSE39582", 566, "affymetrix_u133plus2", "open", "GEO", False),
    CohortSpec("GSE17536", 177, "affymetrix_gpl570", "open", "GEO", False),
    CohortSpec("GSE236581", 22, "rna_seq_recist", "open", "GEO", True),
    CohortSpec("IMvigor210", 298, "rna_seq_atezolizumab", "CC BY 3.0", "R package", True),
    CohortSpec("CPTAC-CRC", 290, "proteomics_sequencing", "open", "NCI PDC", False),
    CohortSpec("Visium-HD-CRC", 5, "visium_hd_ffpe", "open", "10x Genomics", False),
)


class CohortRegistry:
    def __init__(self, specs: tuple[CohortSpec, ...] = COHORTS) -> None:
        self._by_name = {spec.name: spec for spec in specs}

    def get(self, name: str) -> CohortSpec:
        return self._by_name[name]

    def names(self) -> list[str]:
        return list(self._by_name)


def deconvolve_fractions(expression: FloatArray, signature: FloatArray) -> FloatArray:
    weights, _ = nnls(signature, expression)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(signature.shape[1], 1.0 / signature.shape[1], dtype=np.float64)
    return cast(FloatArray, (weights / total).astype(np.float64))


def assign_cms(observation: AggregateObservation) -> str:
    ifn, tgf, angio = observation.pathways
    if ifn >= 0.6 and ifn >= tgf:
        return CMS_LABELS[0]
    if tgf >= 0.6 and tgf > ifn:
        return CMS_LABELS[3]
    if angio >= 0.55:
        return CMS_LABELS[2]
    if max(float(ifn), float(tgf), float(angio)) < 0.35:
        return CMS_LABELS[4]
    return CMS_LABELS[1]


def _fraction_profile(activation: float, stroma: float, rng: np.random.Generator) -> FloatArray:
    base = rng.uniform(0.01, 0.03, size=len(CIBERSORT_CELL_TYPES))
    index = {name: position for position, name in enumerate(CIBERSORT_CELL_TYPES)}
    base[index["cd8_t"]] += 0.20 * activation
    base[index["t_follicular_helper"]] += 0.06 * activation
    base[index["nk_activated"]] += 0.05 * activation
    base[index["macrophage_m1"]] += 0.05 * activation
    base[index["t_regulatory"]] += 0.12 * stroma
    base[index["macrophage_m2"]] += 0.18 * stroma
    base[index["neutrophil"]] += 0.08 * stroma
    base = np.clip(base, 0.0, None)
    return cast(FloatArray, (base / base.sum()).astype(np.float64))


def _spatial_maps(activation: float, stroma: float, resolution: int) -> dict[str, FloatArray]:
    axis = (np.arange(resolution) + 0.5) / resolution
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    centre = ((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2) / 0.08
    tumour = np.exp(-centre)
    ring = np.exp(-((np.sqrt((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2) - 0.30) ** 2) / 0.01)
    immune = (0.3 + activation) * ring
    barrier = (0.2 + stroma) * (1.0 - tumour)
    return {
        "tumour": tumour.astype(np.float64),
        "cd8_t": immune.astype(np.float64),
        "t_regulatory": barrier.astype(np.float64),
    }


class SyntheticCohort:
    def __init__(self, cohort: str = "TCGA-COAD/READ", resolution: int = 50) -> None:
        self.cohort = cohort
        self.resolution = resolution
        self._registry = CohortRegistry()

    def sample(self, n: int, seed: int) -> list[PatientSample]:
        rng = np.random.default_rng(seed)
        treated = self._registry.get(self.cohort).treated
        thresholds = CiaiThresholds()
        samples: list[PatientSample] = []
        for position in range(n):
            activation = float(rng.uniform(0.0, 1.0))
            stroma = float(rng.uniform(0.0, 1.0))
            fractions = _fraction_profile(activation, stroma, rng)
            pathways = np.asarray(
                [
                    activation + float(rng.normal(0.0, 0.03)),
                    stroma + float(rng.normal(0.0, 0.03)),
                    0.5 * stroma + float(rng.normal(0.0, 0.03)),
                ],
                dtype=np.float64,
            )
            pathways = np.clip(pathways, 0.0, 1.0)
            observation = AggregateObservation(fractions=fractions, pathways=pathways)
            cd8_fraction = float(fractions[CIBERSORT_CELL_TYPES.index("cd8_t")])
            gep_score = float(2.0 * (activation - 0.5) + rng.normal(0.0, 0.1))
            tmb = float(max(0.0, 30.0 * activation + rng.normal(0.0, 4.0)))
            sample = PatientSample(
                patient_id=f"{self.cohort}-{seed}-{position:04d}",
                cohort=self.cohort,
                observation=observation,
                cms=assign_cms(observation),
                msi_high=activation > 0.7,
                tmb_per_mb=tmb,
                cd8_fraction=cd8_fraction,
                gep_score=gep_score,
                treated=treated,
                responder=0,
            )
            label = is_responder(sample, thresholds)
            samples.append(_relabel(sample, label))
        return samples

    def spatial_reference(self, sample: PatientSample) -> SpatialGrid:
        activation = sample.gep_score / 2.0 + 0.5
        stroma = float(sample.observation.pathways[1])
        density = _spatial_maps(activation, stroma, self.resolution)
        return SpatialGrid(extent=1.0, resolution=self.resolution, density=density)


def _relabel(sample: PatientSample, responder: int) -> PatientSample:
    return PatientSample(
        patient_id=sample.patient_id,
        cohort=sample.cohort,
        observation=sample.observation,
        cms=sample.cms,
        msi_high=sample.msi_high,
        tmb_per_mb=sample.tmb_per_mb,
        cd8_fraction=sample.cd8_fraction,
        gep_score=sample.gep_score,
        treated=sample.treated,
        responder=responder,
    )


def stratified_folds(
    samples: list[PatientSample], n_folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)
    positives = [i for i, s in enumerate(samples) if s.responder == 1]
    negatives = [i for i, s in enumerate(samples) if s.responder == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for offset, index in enumerate(positives + negatives):
        buckets[offset % n_folds].append(index)
    folds: list[tuple[list[int], list[int]]] = []
    for held in range(n_folds):
        val = sorted(buckets[held])
        train = sorted(i for b in range(n_folds) if b != held for i in buckets[b])
        folds.append((train, val))
    return folds


class ManifestLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[PatientSample]:
        records = json.loads(self.path.read_text())
        samples: list[PatientSample] = []
        for record in records:
            fractions = np.asarray(record["fractions"], dtype=np.float64)
            pathways = np.asarray(record["pathways"], dtype=np.float64)
            observation = AggregateObservation(fractions=fractions, pathways=pathways)
            samples.append(
                PatientSample(
                    patient_id=str(record["patient_id"]),
                    cohort=str(record.get("cohort", "manifest")),
                    observation=observation,
                    cms=str(record.get("cms", assign_cms(observation))),
                    msi_high=bool(record["msi_high"]),
                    tmb_per_mb=float(record["tmb_per_mb"]),
                    cd8_fraction=float(record["cd8_fraction"]),
                    gep_score=float(record["gep_score"]),
                    treated=bool(record.get("treated", False)),
                    responder=int(record.get("responder", 0)),
                )
            )
        return samples


def identity_signature() -> FloatArray:
    size = len(CIBERSORT_CELL_TYPES)
    return np.eye(size, dtype=np.float64)


def pathway_names() -> tuple[str, ...]:
    return PATHWAY_SCORES
