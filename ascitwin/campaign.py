from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ascitwin.collocation import CollocationBatch, DomainConfig, DomainSampler
from ascitwin.config import ExperimentConfig
from ascitwin.gathers import PatientSample
from ascitwin.inversion import (
    CompositeLoss,
    DigitalTwin,
    GradNormBalancer,
    LossWeights,
    observation_to_targets,
)
from ascitwin.medium import FREE_PARAMETERS, PdeParameters
from ascitwin.rig import resolve_device, seeded_generator, set_seed

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray

_FRAMEWORK_SPATIAL = "no_spatial"
_FRAMEWORK_PHYSICS = "no_physics"
_SPECIES = ("tumour", "immune", "cytokine", "drug", "pdl1")


@dataclass
class InversionResult:
    parameters: PdeParameters
    final_loss: float
    history: list[float]


def _domain_from(exp: ExperimentConfig) -> DomainConfig:
    return DomainConfig(
        extent=exp.domain.extent,
        t_final=exp.domain.t_final,
        n_interior=exp.domain.n_interior,
        n_boundary=exp.domain.n_boundary,
        n_initial=exp.domain.n_initial,
    )


def _weights(exp: ExperimentConfig) -> LossWeights:
    disabled = set(exp.train.disabled_equations)
    w_pde = 0.0 if _FRAMEWORK_PHYSICS in disabled else exp.train.w_pde
    w_data = 0.0 if _FRAMEWORK_SPATIAL in disabled else exp.train.w_data
    return LossWeights(w_data=w_data, w_pde=w_pde, w_ic=exp.train.w_ic, w_bc=exp.train.w_bc)


def _residual_disabled(exp: ExperimentConfig) -> frozenset[str]:
    return frozenset(name for name in exp.train.disabled_equations if name in _SPECIES)


def fit_patient(sample: PatientSample, exp: ExperimentConfig, seed: int) -> InversionResult:
    set_seed(seed)
    device = resolve_device(exp.train.device)
    twin = DigitalTwin(exp.model, treated=sample.treated, dose_rate=exp.train.dose_rate).to(device)
    optimiser = torch.optim.Adam(
        twin.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(exp.train.epochs, 1)
    )
    sampler = DomainSampler(_domain_from(exp))
    generator = seeded_generator(seed)
    target = observation_to_targets(sample.observation).to(device)
    weights = _weights(exp)
    composite = CompositeLoss(weights)
    balancer = GradNormBalancer()
    disabled = _residual_disabled(exp)
    history: list[float] = []
    twin.train()
    for _ in range(exp.train.epochs):
        batch = sampler.batch(generator)
        batch = _to_device(batch, device)
        optimiser.zero_grad()
        pde = twin.residual_loss(batch, disabled)
        bc = twin.boundary_loss(batch)
        ic = twin.initial_loss(batch)
        data = (
            twin.data_loss(target, exp.data.resolution) if weights.w_data > 0.0 else torch.zeros(())
        )
        if exp.train.grad_balance:
            scale = balancer.update(
                {
                    "data": float(data.detach()),
                    "pde": float(pde.detach()),
                    "ic": float(ic.detach()),
                    "bc": float(bc.detach()),
                }
            )
            composite.weights = LossWeights(
                w_data=weights.w_data * scale["data"],
                w_pde=weights.w_pde * scale["pde"],
                w_ic=weights.w_ic * scale["ic"],
                w_bc=weights.w_bc * scale["bc"],
            )
        loss = composite.combine(data, pde, ic, bc)
        torch.autograd.backward(loss)
        optimiser.step()
        scheduler.step()
        history.append(float(loss.detach()))
    return InversionResult(
        parameters=twin.parameters_module.as_struct(),
        final_loss=history[-1] if history else float("nan"),
        history=history,
    )


def _to_device(batch: CollocationBatch, device: torch.device) -> CollocationBatch:
    interior = batch.interior.detach().to(device).requires_grad_(True)
    boundary = batch.boundary.detach().to(device).requires_grad_(True)
    initial = batch.initial.detach().to(device).requires_grad_(True)
    return CollocationBatch(
        interior=interior,
        boundary=boundary,
        boundary_normals=batch.boundary_normals.to(device),
        initial=initial,
    )


def invert_cohort(
    samples: list[PatientSample], exp: ExperimentConfig, seed: int
) -> tuple[FloatArray, FloatArray]:
    rows: list[list[float]] = []
    labels: list[float] = []
    for index, sample in enumerate(samples):
        result = fit_patient(sample, exp, seed + index)
        rows.append(result.parameters.free_vector())
        labels.append(float(sample.responder))
    theta = np.asarray(rows, dtype=np.float64)
    return theta, np.asarray(labels, dtype=np.float64)


class ResponsePredictor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.linear(features).squeeze(-1)
        return logits


@dataclass
class Standardiser:
    mean: FloatArray
    scale: FloatArray

    def apply(self, theta: FloatArray) -> FloatArray:
        return ((theta - self.mean) / self.scale).astype(np.float64)


def fit_standardiser(theta: FloatArray) -> Standardiser:
    mean = theta.mean(axis=0)
    scale = theta.std(axis=0) + 1e-8
    return Standardiser(mean=mean, scale=scale)


def fit_response_model(
    theta: FloatArray, labels: FloatArray, seed: int, steps: int = 400
) -> tuple[ResponsePredictor, Standardiser]:
    set_seed(seed)
    standardiser = fit_standardiser(theta)
    features = torch.as_tensor(standardiser.apply(theta), dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.float32)
    model = ResponsePredictor(theta.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimiser.zero_grad()
        loss = criterion(model(features), target)
        torch.autograd.backward(loss)
        optimiser.step()
    return model, standardiser


def predict_proba(
    model: ResponsePredictor, standardiser: Standardiser, theta: FloatArray
) -> FloatArray:
    features = torch.as_tensor(standardiser.apply(theta), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(features))
    return probs.numpy().astype(np.float64)


def free_parameter_names() -> tuple[str, ...]:
    return FREE_PARAMETERS
