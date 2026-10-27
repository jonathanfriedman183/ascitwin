from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    width: int = 512
    depth: int = 4
    dropout: float = 0.2
    activation: str = "tanh"


@dataclass
class DomainConfigSchema:
    extent: float = 1.0
    t_final: float = 1.0
    n_interior: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000


@dataclass
class DataConfig:
    cohort: str = "TCGA-COAD/READ"
    n_patients: int = 623
    resolution: int = 50
    n_folds: int = 5
    seed: int = 0


@dataclass
class TrainConfig:
    epochs: int = 8000
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_seeds: int = 20
    dose_rate: float = 1.0
    w_data: float = 1.0
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    grad_balance: bool = True
    disabled_equations: list[str] = field(default_factory=list)
    spatial_recovery: bool = True
    device: str | None = None


@dataclass
class ExperimentConfig:
    name: str = "main"
    model: ModelConfig = field(default_factory=ModelConfig)
    domain: DomainConfigSchema = field(default_factory=DomainConfigSchema)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "experiment"


def load_config(experiment: str, overrides: list[str] | None = None) -> ExperimentConfig:
    base = OmegaConf.structured(ExperimentConfig())
    path = experiment_dir() / f"{experiment}.yaml"
    if path.exists():
        base = OmegaConf.merge(base, OmegaConf.load(path))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    resolved = cast(ExperimentConfig, OmegaConf.to_object(base))
    return resolved


def parse_experiment_token(token: str) -> tuple[str, list[str]]:
    if "=" in token and token.split("=", 1)[0] == "experiment":
        return token.split("=", 1)[1], []
    return token, []
