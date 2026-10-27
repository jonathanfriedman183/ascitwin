from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DomainConfig:
    extent: float = 1.0
    t_final: float = 1.0
    n_interior: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000


@dataclass(frozen=True)
class CollocationBatch:
    interior: torch.Tensor
    boundary: torch.Tensor
    boundary_normals: torch.Tensor
    initial: torch.Tensor


def _scaled(samples: torch.Tensor, extent: float) -> torch.Tensor:
    return samples * extent


class DomainSampler:
    def __init__(self, config: DomainConfig) -> None:
        self.config = config

    def interior(self, generator: torch.Generator) -> torch.Tensor:
        raw = torch.rand(self.config.n_interior, 3, generator=generator)
        space = _scaled(raw[:, :2], self.config.extent)
        time = raw[:, 2:3] * self.config.t_final
        coords = torch.cat([space, time], dim=1)
        coords.requires_grad_(True)
        return coords

    def initial(self, generator: torch.Generator) -> torch.Tensor:
        raw = torch.rand(self.config.n_initial, 2, generator=generator)
        space = _scaled(raw, self.config.extent)
        time = torch.zeros(self.config.n_initial, 1)
        coords = torch.cat([space, time], dim=1)
        coords.requires_grad_(True)
        return coords

    def boundary(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.config.n_boundary
        side = torch.randint(0, 4, (count,), generator=generator)
        along = torch.rand(count, generator=generator) * self.config.extent
        time = torch.rand(count, 1, generator=generator) * self.config.t_final
        x = torch.empty(count)
        y = torch.empty(count)
        normals = torch.zeros(count, 2)
        for index in range(count):
            edge = int(side[index].item())
            position = float(along[index].item())
            if edge == 0:
                x[index], y[index] = 0.0, position
                normals[index] = torch.tensor([-1.0, 0.0])
            elif edge == 1:
                x[index], y[index] = self.config.extent, position
                normals[index] = torch.tensor([1.0, 0.0])
            elif edge == 2:
                x[index], y[index] = position, 0.0
                normals[index] = torch.tensor([0.0, -1.0])
            else:
                x[index], y[index] = position, self.config.extent
                normals[index] = torch.tensor([0.0, 1.0])
        coords = torch.cat([x.unsqueeze(1), y.unsqueeze(1), time], dim=1)
        coords.requires_grad_(True)
        return coords, normals

    def batch(self, generator: torch.Generator) -> CollocationBatch:
        boundary, normals = self.boundary(generator)
        return CollocationBatch(
            interior=self.interior(generator),
            boundary=boundary,
            boundary_normals=normals,
            initial=self.initial(generator),
        )


def grid_coordinates(extent: float, resolution: int, t_value: float) -> torch.Tensor:
    axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution * extent
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = grid_x.reshape(-1, 1)
    flat_y = grid_y.reshape(-1, 1)
    time = torch.full_like(flat_x, t_value)
    return torch.cat([flat_x, flat_y, time], dim=1)
