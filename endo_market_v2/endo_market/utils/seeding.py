"""Centralized seeding for full reproducibility from ``(config, seed)``."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch RNGs.

    Parameters
    ----------
    seed:
        The base seed.
    deterministic:
        If ``True``, force single-threaded, deterministic torch behaviour so
        results are bit-reproducible on CPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Single CPU thread => deterministic reductions.
        torch.set_num_threads(1)


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Return a seeded ``torch.Generator`` for explicit, local randomness."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """Context manager that temporarily sets and then restores RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
