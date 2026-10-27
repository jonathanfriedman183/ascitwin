from __future__ import annotations

from dataclasses import dataclass

from ascitwin.gathers import PatientSample


@dataclass(frozen=True)
class CiaiThresholds:
    tmb_cut: float = 10.0
    cd8_cut: float = 0.10
    gep_cut: float = 0.0


def msi_criterion(msi_high: bool) -> bool:
    return msi_high


def tmb_criterion(tmb_per_mb: float, thresholds: CiaiThresholds) -> bool:
    return tmb_per_mb >= thresholds.tmb_cut


def cd8_criterion(cd8_fraction: float, thresholds: CiaiThresholds) -> bool:
    return cd8_fraction >= thresholds.cd8_cut


def gep_criterion(gep_score: float, thresholds: CiaiThresholds) -> bool:
    return gep_score >= thresholds.gep_cut


def criteria_vector(sample: PatientSample, thresholds: CiaiThresholds) -> list[bool]:
    return [
        msi_criterion(sample.msi_high),
        tmb_criterion(sample.tmb_per_mb, thresholds),
        cd8_criterion(sample.cd8_fraction, thresholds),
        gep_criterion(sample.gep_score, thresholds),
    ]


def composite_immune_activation_index(
    sample: PatientSample, thresholds: CiaiThresholds | None = None
) -> int:
    resolved = thresholds if thresholds is not None else CiaiThresholds()
    return sum(1 for flag in criteria_vector(sample, resolved) if flag)


def is_responder(sample: PatientSample, thresholds: CiaiThresholds | None = None) -> int:
    return int(composite_immune_activation_index(sample, thresholds) >= 2)
