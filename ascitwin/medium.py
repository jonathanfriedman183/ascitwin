from __future__ import annotations

from dataclasses import dataclass, replace

import torch

FREE_PARAMETERS: tuple[str, ...] = (
    "d_t",
    "rho",
    "k_cap",
    "delta_ti",
    "mu_i",
    "d_i",
    "s_i",
    "alpha",
    "beta",
    "gamma",
    "d_c",
    "lambda_c",
    "k_el",
    "p_basal",
    "k_up",
)

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "d_t": (1e-4, 1e-3),
    "rho": (1e-2, 1e-1),
    "k_cap": (1e-1, 1e1),
    "delta_ti": (1e-3, 1e-1),
    "mu_i": (1e-2, 2e-1),
    "d_i": (1e-3, 1e-1),
    "s_i": (1e-3, 5e-1),
    "alpha": (1e-2, 5e-1),
    "beta": (1e-2, 1e0),
    "gamma": (1e-2, 1e0),
    "d_c": (1e-3, 1e-1),
    "lambda_c": (1e-2, 1e0),
    "k_el": (1e-2, 2e0),
    "p_basal": (1e-3, 1e-1),
    "k_up": (1e-2, 1e0),
}

FIXED_CONSTANTS: dict[str, float] = {
    "kappa_t": 0.10,
    "kappa_i": 0.20,
    "k_deg": 0.10,
    "k_p": 0.50,
    "k_block": 0.30,
}


@dataclass(frozen=True)
class PdeParameters:
    d_t: float = 5.3e-4
    rho: float = 0.052
    k_cap: float = 1.0
    delta_ti: float = 0.020
    mu_i: float = 0.044
    d_i: float = 0.025
    s_i: float = 0.050
    alpha: float = 0.060
    beta: float = 0.050
    gamma: float = 0.100
    d_c: float = 0.025
    lambda_c: float = 0.180
    k_el: float = 0.700
    p_basal: float = 0.010
    k_up: float = 0.150
    kappa_t: float = FIXED_CONSTANTS["kappa_t"]
    kappa_i: float = FIXED_CONSTANTS["kappa_i"]
    k_deg: float = FIXED_CONSTANTS["k_deg"]
    k_p: float = FIXED_CONSTANTS["k_p"]
    k_block: float = FIXED_CONSTANTS["k_block"]

    def free_vector(self) -> list[float]:
        return [getattr(self, name) for name in FREE_PARAMETERS]

    def with_free_vector(self, values: list[float]) -> PdeParameters:
        updates = dict(zip(FREE_PARAMETERS, values, strict=True))
        return replace(self, **updates)

    def clamp_to_bounds(self) -> PdeParameters:
        clamped: dict[str, float] = {}
        for name in FREE_PARAMETERS:
            low, high = PARAMETER_BOUNDS[name]
            clamped[name] = float(min(max(getattr(self, name), low), high))
        return replace(self, **clamped)


ALL_PARAMETERS: tuple[str, ...] = (
    *FREE_PARAMETERS,
    "kappa_t",
    "kappa_i",
    "k_deg",
    "k_p",
    "k_block",
)


@dataclass(frozen=True)
class TensorParameters:
    d_t: torch.Tensor
    rho: torch.Tensor
    k_cap: torch.Tensor
    delta_ti: torch.Tensor
    mu_i: torch.Tensor
    d_i: torch.Tensor
    s_i: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    d_c: torch.Tensor
    lambda_c: torch.Tensor
    k_el: torch.Tensor
    p_basal: torch.Tensor
    k_up: torch.Tensor
    kappa_t: torch.Tensor
    kappa_i: torch.Tensor
    k_deg: torch.Tensor
    k_p: torch.Tensor
    k_block: torch.Tensor


def tensor_parameters_from_free(
    free: list[torch.Tensor], base: PdeParameters, device: torch.device
) -> TensorParameters:
    values: dict[str, torch.Tensor] = dict(zip(FREE_PARAMETERS, free, strict=True))
    for name in ("kappa_t", "kappa_i", "k_deg", "k_p", "k_block"):
        values[name] = torch.tensor(getattr(base, name), dtype=torch.float32, device=device)
    return TensorParameters(**values)


def default_parameters() -> PdeParameters:
    return PdeParameters()


def to_tensors(params: PdeParameters, device: torch.device | None = None) -> TensorParameters:
    target = device if device is not None else torch.device("cpu")
    values = {
        name: torch.tensor(getattr(params, name), dtype=torch.float32, device=target)
        for name in ALL_PARAMETERS
    }
    return TensorParameters(**values)


def sample_parameters(generator: torch.Generator) -> PdeParameters:
    values: list[float] = []
    for name in FREE_PARAMETERS:
        low, high = PARAMETER_BOUNDS[name]
        unit = torch.rand(1, generator=generator).item()
        values.append(low + (high - low) * unit)
    return default_parameters().with_free_vector(values)


@dataclass(frozen=True)
class StateFields:
    tumour: torch.Tensor
    immune: torch.Tensor
    cytokine: torch.Tensor
    drug: torch.Tensor
    pdl1: torch.Tensor

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> StateFields:
        columns = matrix.unbind(dim=-1)
        return cls(
            tumour=columns[0],
            immune=columns[1],
            cytokine=columns[2],
            drug=columns[3],
            pdl1=columns[4],
        )

    def as_matrix(self) -> torch.Tensor:
        return torch.stack(
            [self.tumour, self.immune, self.cytokine, self.drug, self.pdl1],
            dim=-1,
        )
