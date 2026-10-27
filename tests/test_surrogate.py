from __future__ import annotations

import numpy as np

from ascitwin.gathers import CIBERSORT_CELL_TYPES, AggregateObservation, PatientSample
from ascitwin.surrogate import CiaiThresholds, composite_immune_activation_index, is_responder


def _sample(msi: bool, tmb: float, cd8: float, gep: float) -> PatientSample:
    observation = AggregateObservation(
        fractions=np.full(len(CIBERSORT_CELL_TYPES), 1.0 / len(CIBERSORT_CELL_TYPES)),
        pathways=np.array([0.5, 0.3, 0.2]),
    )
    return PatientSample(
        patient_id="p",
        cohort="c",
        observation=observation,
        cms="CMS1",
        msi_high=msi,
        tmb_per_mb=tmb,
        cd8_fraction=cd8,
        gep_score=gep,
        treated=False,
        responder=0,
    )


def test_all_four_criteria_count_four() -> None:
    sample = _sample(True, 20.0, 0.3, 1.0)
    assert composite_immune_activation_index(sample) == 4
    assert is_responder(sample) == 1


def test_none_satisfied_is_non_responder() -> None:
    sample = _sample(False, 1.0, 0.01, -1.0)
    assert composite_immune_activation_index(sample) == 0
    assert is_responder(sample) == 0


def test_two_of_four_is_responder_boundary() -> None:
    sample = _sample(True, 20.0, 0.01, -1.0)
    assert composite_immune_activation_index(sample) == 2
    assert is_responder(sample) == 1


def test_one_of_four_is_non_responder() -> None:
    sample = _sample(True, 1.0, 0.01, -1.0)
    assert composite_immune_activation_index(sample) == 1
    assert is_responder(sample) == 0


def test_custom_thresholds_change_count() -> None:
    sample = _sample(False, 12.0, 0.05, 0.5)
    strict = CiaiThresholds(tmb_cut=15.0, cd8_cut=0.2, gep_cut=1.0)
    assert composite_immune_activation_index(sample, strict) == 0
