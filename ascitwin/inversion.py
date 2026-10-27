from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ascitwin.collocation import CollocationBatch, grid_coordinates
from ascitwin.config import ModelConfig
from ascitwin.gathers import CIBERSORT_CELL_TYPES, AggregateObservation
from ascitwin.governing import all_residuals, constant_dose, neumann_flux
from ascitwin.medium import (
    FREE_PARAMETERS,
    PARAMETER_BOUNDS,
    PdeParameters,
    StateFields,
    tensor_parameters_from_free,
)

DIFFUSIVE_SPECIES: tuple[str, ...] = ("tumour", "immune", "cytokine")
FRACTION_SPECIES: tuple[str, ...] = ("tumour", "immune", "cytokine", "pdl1")

_LYMPHOID = ("cd8_t", "t_follicular_helper", "nk_activated", "b_memory", "plasma")
_MYELOID = ("macrophage_m2", "neutrophil", "monocyte", "t_regulatory")


class FieldNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, config.width), nn.Tanh()]
        for _ in range(config.depth - 1):
            layers.append(nn.Linear(config.width, config.width))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.width, 5))
        self.body = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.body(coords)
        return nn.functional.softplus(raw)


class ParameterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(len(FREE_PARAMETERS)))

    def bounded(self) -> list[torch.Tensor]:
        gated = torch.sigmoid(self.raw)
        values: list[torch.Tensor] = []
        for index, name in enumerate(FREE_PARAMETERS):
            low, high = PARAMETER_BOUNDS[name]
            values.append(low + (high - low) * gated[index])
        return values

    def as_struct(self) -> PdeParameters:
        scalars = [float(value.item()) for value in self.bounded()]
        return PdeParameters().with_free_vector(scalars)


class DigitalTwin(nn.Module):
    def __init__(self, config: ModelConfig, treated: bool, dose_rate: float) -> None:
        super().__init__()
        self.field = FieldNetwork(config)
        self.parameters_module = ParameterModule()
        self.treated = treated
        self.dose_rate = dose_rate

    def forward(self, coords: torch.Tensor) -> StateFields:
        return StateFields.from_matrix(self.field(coords))

    def _live_parameters(self) -> PdeParameters:
        return self.parameters_module.as_struct()

    def residual_loss(
        self, batch: CollocationBatch, disabled: frozenset[str] = frozenset()
    ) -> torch.Tensor:
        device = batch.interior.device
        params = tensor_parameters_from_free(
            self.parameters_module.bounded(), PdeParameters(), device
        )
        fields = self.forward(batch.interior)
        dose = constant_dose(batch.interior[:, 2], self.dose_rate, self.treated)
        residuals = all_residuals(fields, batch.interior, params, dose)
        total = torch.zeros(())
        for name, value in residuals.items():
            if name in disabled:
                continue
            total = total + value.pow(2).mean()
        return total

    def boundary_loss(self, batch: CollocationBatch) -> torch.Tensor:
        fields = self.forward(batch.boundary)
        total = torch.zeros(())
        for species in DIFFUSIVE_SPECIES:
            field = getattr(fields, species)
            flux = neumann_flux(field, batch.boundary, batch.boundary_normals)
            total = total + flux.pow(2).mean()
        return total

    def initial_loss(self, batch: CollocationBatch) -> torch.Tensor:
        fields = self.forward(batch.initial)
        seed = _initial_condition(batch.initial)
        return (fields.tumour - seed).pow(2).mean()

    def data_loss(self, target: torch.Tensor, resolution: int) -> torch.Tensor:
        predicted = fraction_constraint(self, resolution)
        return (predicted - target).pow(2).sum()


def _initial_condition(coords: torch.Tensor) -> torch.Tensor:
    centre = ((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2) / 0.08
    return torch.exp(-centre)


def fraction_constraint(twin: DigitalTwin, resolution: int) -> torch.Tensor:
    coords = grid_coordinates(1.0, resolution, twin_biopsy_time())
    coords = coords.to(next(twin.field.parameters()).device)
    fields = twin.forward(coords)
    integrals = torch.stack([getattr(fields, species).mean() for species in FRACTION_SPECIES])
    total = integrals.sum()
    normaliser = total + torch.finfo(integrals.dtype).eps
    return integrals / normaliser


def twin_biopsy_time() -> float:
    return 1.0


def observation_to_targets(observation: AggregateObservation) -> torch.Tensor:
    index = {name: position for position, name in enumerate(CIBERSORT_CELL_TYPES)}
    lymphoid = float(sum(observation.fractions[index[name]] for name in _LYMPHOID))
    myeloid = float(sum(observation.fractions[index[name]] for name in _MYELOID))
    ifn = float(observation.pathways[0])
    tumour = max(1.0 - lymphoid - myeloid, 0.0)
    immune = lymphoid
    cytokine = ifn
    pdl1 = 0.5 * ifn + 0.5 * myeloid
    raw = torch.tensor([tumour, immune, cytokine, pdl1], dtype=torch.float32)
    return raw / (raw.sum() + torch.finfo(raw.dtype).eps)


@dataclass
class LossWeights:
    w_data: float
    w_pde: float
    w_ic: float
    w_bc: float


class CompositeLoss:
    def __init__(self, weights: LossWeights) -> None:
        self.weights = weights

    def combine(
        self, data: torch.Tensor, pde: torch.Tensor, ic: torch.Tensor, bc: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.weights.w_data * data
            + self.weights.w_pde * pde
            + self.weights.w_ic * ic
            + self.weights.w_bc * bc
        )


class GradNormBalancer:
    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def update(self, magnitudes: dict[str, float]) -> dict[str, float]:
        for key, value in magnitudes.items():
            previous = self._state.get(key, value)
            self._state[key] = self.alpha * previous + (1.0 - self.alpha) * value
        reference = max(self._state.values()) if self._state else 1.0
        weights: dict[str, float] = {}
        for key, value in self._state.items():
            weights[key] = reference / (value + 1e-8)
        return weights
