from __future__ import annotations

from dataclasses import dataclass

import torch

from ascitwin.medium import StateFields, TensorParameters

SPATIAL_DIMS: tuple[int, int] = (0, 1)
TIME_DIM: int = 2


def _first_gradient(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if not field.requires_grad:
        return torch.zeros_like(coords)
    grad = torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(coords)
    return grad


def time_derivative(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return _first_gradient(field, coords)[:, TIME_DIM]


def laplacian(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grad = _first_gradient(field, coords)
    accumulated = torch.zeros_like(field)
    for axis in SPATIAL_DIMS:
        component = grad[:, axis]
        second = _first_gradient(component, coords)[:, axis]
        accumulated = accumulated + second
    return accumulated


def neumann_flux(field: torch.Tensor, coords: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    grad = _first_gradient(field, coords)
    spatial = grad[:, list(SPATIAL_DIMS)]
    return (spatial * normals).sum(dim=-1)


@dataclass(frozen=True)
class FieldDynamics:
    dt: torch.Tensor
    lap: torch.Tensor


def _dynamics(field: torch.Tensor, coords: torch.Tensor, diffusive: bool) -> FieldDynamics:
    dt = time_derivative(field, coords)
    lap = laplacian(field, coords) if diffusive else torch.zeros_like(field)
    return FieldDynamics(dt=dt, lap=lap)


def tumour_residual(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters
) -> torch.Tensor:
    dyn = _dynamics(fields.tumour, coords, diffusive=True)
    growth = params.rho * fields.tumour * (1.0 - fields.tumour / params.k_cap)
    kill = params.delta_ti * fields.immune * fields.tumour
    flux = params.d_t * dyn.lap + growth - kill
    return dyn.dt - flux


def immune_residual(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters
) -> torch.Tensor:
    dyn = _dynamics(fields.immune, coords, diffusive=True)
    recruitment = params.alpha * (fields.tumour / (params.beta + fields.tumour)) * fields.immune
    decay = params.mu_i * fields.immune
    exhaustion = params.gamma * fields.pdl1 * fields.immune
    flux = params.d_i * dyn.lap + params.s_i + recruitment - decay - exhaustion
    return dyn.dt - flux


def cytokine_residual(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters
) -> torch.Tensor:
    dyn = _dynamics(fields.cytokine, coords, diffusive=True)
    production = params.kappa_t * fields.tumour + params.kappa_i * fields.immune
    flux = params.d_c * dyn.lap + production - params.lambda_c * fields.cytokine
    return dyn.dt - flux


def drug_residual(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters, dose: torch.Tensor
) -> torch.Tensor:
    dyn = _dynamics(fields.drug, coords, diffusive=False)
    flux = -params.k_el * fields.drug + dose
    return dyn.dt - flux


def pdl1_residual(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters
) -> torch.Tensor:
    dyn = _dynamics(fields.pdl1, coords, diffusive=False)
    upregulation = params.k_up * (fields.tumour / (params.k_p + fields.tumour))
    blockade = params.k_block * fields.drug * fields.pdl1
    flux = params.p_basal - params.k_deg * fields.pdl1 + upregulation - blockade
    return dyn.dt - flux


def all_residuals(
    fields: StateFields, coords: torch.Tensor, params: TensorParameters, dose: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "tumour": tumour_residual(fields, coords, params),
        "immune": immune_residual(fields, coords, params),
        "cytokine": cytokine_residual(fields, coords, params),
        "drug": drug_residual(fields, coords, params, dose),
        "pdl1": pdl1_residual(fields, coords, params),
    }


def constant_dose(t: torch.Tensor, rate: float, treated: bool) -> torch.Tensor:
    if not treated:
        return torch.zeros_like(t)
    return torch.full_like(t, rate)
