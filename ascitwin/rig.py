from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> dict[str, Any]:
    location = device if device is not None else torch.device("cpu")
    loaded: dict[str, Any] = torch.load(str(path), map_location=location, weights_only=False)
    return loaded


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def as_tensor(values: Any, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
