from __future__ import annotations

import torch

from ascitwin.acquisition import SyntheticCohort
from ascitwin.collocation import CollocationBatch, DomainConfig, DomainSampler
from ascitwin.config import ModelConfig
from ascitwin.inversion import DigitalTwin, fraction_constraint, observation_to_targets
from ascitwin.rig import seeded_generator, set_seed


def _twin() -> DigitalTwin:
    return DigitalTwin(ModelConfig(width=16, depth=3, dropout=0.0), treated=False, dose_rate=1.0)


def _batch() -> CollocationBatch:
    return DomainSampler(DomainConfig(n_interior=128, n_boundary=32, n_initial=32)).batch(
        seeded_generator(0)
    )


def test_fraction_constraint_normalised() -> None:
    twin = _twin()
    fractions = fraction_constraint(twin, resolution=8)
    assert torch.allclose(fractions.sum(), torch.tensor(1.0), atol=1e-5)


def test_gradient_reaches_parameters_and_backbone() -> None:
    set_seed(0)
    twin = _twin()
    batch = _batch()
    sample = SyntheticCohort().sample(1, seed=1)[0]
    target = observation_to_targets(sample.observation)
    loss = (
        twin.residual_loss(batch)
        + twin.boundary_loss(batch)
        + twin.initial_loss(batch)
        + twin.data_loss(target, resolution=8)
    )
    torch.autograd.backward(loss)
    assert twin.parameters_module.raw.grad is not None
    assert float(twin.parameters_module.raw.grad.norm()) > 0.0
    first_weight = twin.field.body[0]
    assert isinstance(first_weight, torch.nn.Linear)
    assert first_weight.weight.grad is not None


def test_overfit_single_batch_reduces_loss() -> None:
    set_seed(0)
    twin = _twin()
    batch = _batch()
    optimiser = torch.optim.Adam(twin.parameters(), lr=0.01)
    losses: list[float] = []
    for _ in range(60):
        optimiser.zero_grad()
        loss = twin.residual_loss(batch) + twin.boundary_loss(batch) + twin.initial_loss(batch)
        torch.autograd.backward(loss)
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
