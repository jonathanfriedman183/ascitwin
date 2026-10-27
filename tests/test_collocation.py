from __future__ import annotations

import torch

from ascitwin.collocation import DomainConfig, DomainSampler, grid_coordinates
from ascitwin.rig import seeded_generator


def test_interior_count_and_bounds() -> None:
    sampler = DomainSampler(DomainConfig(n_interior=256, extent=2.0, t_final=3.0))
    coords = sampler.interior(seeded_generator(0))
    assert coords.shape == (256, 3)
    detached = coords.detach()
    assert float(detached[:, :2].max()) <= 2.0
    assert float(detached[:, 2].max()) <= 3.0
    assert coords.requires_grad


def test_initial_points_have_zero_time() -> None:
    sampler = DomainSampler(DomainConfig(n_initial=64))
    coords = sampler.initial(seeded_generator(1))
    assert torch.allclose(coords[:, 2], torch.zeros(64))


def test_boundary_normals_are_unit() -> None:
    sampler = DomainSampler(DomainConfig(n_boundary=128))
    _, normals = sampler.boundary(seeded_generator(2))
    lengths = normals.norm(dim=1)
    assert torch.allclose(lengths, torch.ones_like(lengths))


def test_sampler_is_reproducible() -> None:
    sampler = DomainSampler(DomainConfig(n_interior=32))
    first = sampler.interior(seeded_generator(7))
    second = sampler.interior(seeded_generator(7))
    assert torch.allclose(first, second)


def test_grid_coordinates_shape() -> None:
    coords = grid_coordinates(1.0, 10, 1.0)
    assert coords.shape == (100, 3)
    assert torch.allclose(coords[:, 2], torch.ones(100))
