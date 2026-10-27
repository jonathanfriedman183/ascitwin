from __future__ import annotations

import torch

from ascitwin.governing import laplacian, neumann_flux, time_derivative, tumour_residual
from ascitwin.medium import PdeParameters, StateFields, to_tensors


def _coords(n: int) -> torch.Tensor:
    coords = torch.rand(n, 3, dtype=torch.float32)
    coords.requires_grad_(True)
    return coords


def _constant_on(coords: torch.Tensor, value: float) -> torch.Tensor:
    return value + 0.0 * coords[:, 0]


def test_laplacian_of_quadratic_is_four() -> None:
    coords = _coords(64)
    field = coords[:, 0] ** 2 + coords[:, 1] ** 2
    value = laplacian(field, coords)
    assert torch.allclose(value, torch.full_like(value, 4.0), atol=1e-4)


def test_time_derivative_matches_analytic() -> None:
    coords = _coords(64)
    field = coords[:, 2] ** 2
    value = time_derivative(field, coords)
    assert torch.allclose(value, 2.0 * coords[:, 2], atol=1e-4)


def test_neumann_flux_of_constant_field_is_zero() -> None:
    coords = _coords(32)
    field = _constant_on(coords, 1.0)
    normals = torch.zeros(coords.shape[0], 2)
    normals[:, 0] = 1.0
    flux = neumann_flux(field, coords, normals)
    assert torch.allclose(flux, torch.zeros_like(flux), atol=1e-5)


def test_tumour_residual_reaction_only() -> None:
    coords = _coords(16)
    constant = 0.3
    fields = StateFields(
        tumour=_constant_on(coords, constant),
        immune=torch.full((16,), 0.2),
        cytokine=torch.zeros(16),
        drug=torch.zeros(16),
        pdl1=torch.zeros(16),
    )
    params = to_tensors(PdeParameters())
    residual = tumour_residual(fields, coords, params)
    growth = params.rho * constant * (1.0 - constant / params.k_cap)
    kill = params.delta_ti * 0.2 * constant
    expected = -(growth - kill)
    assert torch.allclose(residual, expected.expand_as(residual), atol=1e-5)
