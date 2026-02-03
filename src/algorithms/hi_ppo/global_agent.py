"""Global (inter-domain) agent interfaces and baselines."""

from __future__ import annotations

import random
from typing import Iterable, List, Optional

import numpy as np


class GlobalAgent:
    """Interface for inter-domain scheduling policies."""

    def select_action(self, state: Iterable[float], deterministic: bool = False) -> List[int]:
        raise NotImplementedError

    def update(self, batch: dict) -> None:
        raise NotImplementedError


class RandomGlobalAgent(GlobalAgent):
    """Random baseline that outputs cut points for K domains."""

    def __init__(self, num_layers: int, num_domains: int, seed: int = 42) -> None:
        self.num_layers = num_layers
        self.num_domains = num_domains
        self._rng = random.Random(seed)

    def select_action(self, state: Iterable[float], deterministic: bool = False) -> List[int]:
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


class PPOGlobalAgent(GlobalAgent):
    """Adapter to use PPOAgent as a GlobalAgent."""

    def __init__(self, ppo_agent: "PPOAgent") -> None:
        self._agent = ppo_agent
        self._last_action: Optional[np.ndarray] = None
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0

    def select_action(self, state: Iterable[float], deterministic: bool = False) -> List[int]:
        action, log_prob, value = self._agent.select_action(
            np.asarray(list(state), dtype=np.float32),
            deterministic=deterministic,
        )
        self._last_action = np.asarray(action, dtype=np.int64)
        self._last_log_prob = float(log_prob)
        self._last_value = float(value)
        return action.tolist()

    def get_last_transition(self) -> tuple[np.ndarray, float, float]:
        if self._last_action is None:
            raise RuntimeError("select_action must be called before get_last_transition().")
        return self._last_action, self._last_log_prob, self._last_value

    def update(self, batch: dict) -> None:
        self._agent.update()

