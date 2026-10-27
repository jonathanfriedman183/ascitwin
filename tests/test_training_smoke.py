from __future__ import annotations

import math

from ascitwin.acquisition import SyntheticCohort
from ascitwin.campaign import fit_patient
from ascitwin.config import load_config


def test_training_smoke_loss_trends_down() -> None:
    exp = load_config("_smoke", ["train.epochs=40", "train.grad_balance=false"])
    builder = SyntheticCohort(exp.data.cohort, resolution=exp.data.resolution)
    sample = builder.sample(exp.data.n_patients, exp.data.seed)[0]
    result = fit_patient(sample, exp, exp.data.seed)
    assert len(result.history) == 40
    assert all(math.isfinite(value) for value in result.history)
    early = sum(result.history[:10]) / 10.0
    late = sum(result.history[-10:]) / 10.0
    assert late < early
