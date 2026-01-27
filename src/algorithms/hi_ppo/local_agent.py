"""Local (intra-domain) agent interfaces and baselines."""

from __future__ import annotations

from typing import Iterable, List


class LocalAgent:
    """Interface for intra-domain scheduling policies."""

    def select_action(self, state: Iterable[float], domain_id: int) -> List[int]:
        raise NotImplementedError

    def update(self, batch: dict) -> None:
        raise NotImplementedError


class NoOpLocalAgent(LocalAgent):
    """Placeholder local agent that performs no refinement."""

    def select_action(self, state: Iterable[float], domain_id: int) -> List[int]:
        return []

    def update(self, batch: dict) -> None:
        return None
