"""Global (inter-domain) agent interfaces and baselines."""

from __future__ import annotations

import random
from typing import Iterable, List


class GlobalAgent:
    """Interface for inter-domain scheduling policies."""

    def select_action(self, state: Iterable[float]) -> List[int]:
        raise NotImplementedError

    def update(self, batch: dict) -> None:
        raise NotImplementedError


class RandomGlobalAgent(GlobalAgent):
    """Random baseline that outputs cut points for K domains."""

    def __init__(self, num_layers: int, num_domains: int, seed: int = 42) -> None:
        self.num_layers = num_layers
        self.num_domains = num_domains
        self._rng = random.Random(seed)

    def select_action(self, state: Iterable[float]) -> List[int]:
        if self.num_domains <= 1:
            return []
        # Choose K-1 cut points between layers.
        cut_points = sorted(
            self._rng.sample(range(1, self.num_layers), k=self.num_domains - 1)
        )
        return cut_points

    def update(self, batch: dict) -> None:
        # Random baseline has no update.
        return None
