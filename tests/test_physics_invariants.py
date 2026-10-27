from __future__ import annotations

import torch

from ascitwin.config import ModelConfig
from ascitwin.governing import drug_residual, tumour_residual
from ascitwin.inversion import FieldNetwork
from ascitwin.medium import PdeParameters, StateFields, to_tensors


def test_field_network_output_is_non_negative() -> None:
    network = FieldNetwork(ModelConfig(width=16, depth=3, dropout=0.0))
    coords = torch.rand(128, 3)
    fields = network(coords)
    assert torch.all(fields >= 0.0)


def test_fisher_kpp_carrying_capacity_fixed_point() -> None:
    coords = torch.rand(16, 3, requires_grad=True)
    params = to_tensors(PdeParameters())
    capacity = float(params.k_cap.item())
    fields = StateFields(
        tumour=capacity + 0.0 * coords[:, 0],
        immune=torch.zeros(16),
        cytokine=torch.zeros(16),
        drug=torch.zeros(16),
        pdl1=torch.zeros(16),
    )
    residual = tumour_residual(fields, coords, params)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_pharmacokinetic_exponential_decay_has_zero_residual() -> None:
    coords = torch.rand(32, 3, requires_grad=True)
    params = to_tensors(PdeParameters())
    rate = float(params.k_el.item())
    drug = torch.exp(-rate * coords[:, 2])
    fields = StateFields(
        tumour=torch.zeros(32),
        immune=torch.zeros(32),
        cytokine=torch.zeros(32),
        drug=drug,
        pdl1=torch.zeros(32),
    )
    dose = torch.zeros(32)
    residual = drug_residual(fields, coords, params, dose)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)
