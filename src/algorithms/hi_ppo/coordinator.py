"""Coordinator to combine global and local policies into a placement action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .global_agent import GlobalAgent
from .local_agent import LocalAgent
from ...utils.gnn_encoder import GNNConfig, GNNEncoder, TopologyGraph
from ...utils.types import NetworkState


@dataclass
class CoordinatorConfig:
    num_layers: int
    num_domains: int


class HiPPOCoordinator:
    """Combine global cut points with optional local refinement."""

    def __init__(
        self,
        config: CoordinatorConfig,
        global_agent: GlobalAgent,
        local_agent: LocalAgent,
        gnn_config: GNNConfig | None = None,
        gnn_encoder: GNNEncoder | None = None,
    ) -> None:
        self.config = config
        self.global_agent = global_agent
        self.local_agent = local_agent
        self.gnn_encoder = gnn_encoder or GNNEncoder(gnn_config or GNNConfig())

    def select_action(
        self,
        state_high: Iterable[float],
        state_low: Iterable[Iterable[float]],
        network_state: NetworkState | None = None,
        domain_loads: Optional[List[float]] = None,
        deterministic: bool = False,
    ) -> List[int]:
        state_high_vec = list(state_high)
        if network_state is not None:
            graph = TopologyGraph.from_network_state(network_state, domain_loads)
            h_topo = self.gnn_encoder.encode(graph)
            state_high_vec = state_high_vec + h_topo

        cut_points = self.global_agent.select_action(state_high_vec, deterministic=deterministic)
        placement = self._cut_points_to_placement(cut_points)
        # Local agent can be used to refine per-domain placement later.
        _ = state_low
        return placement

    def update(self, batch: dict) -> None:
        self.global_agent.update(batch)
        self.local_agent.update(batch)

    def _cut_points_to_placement(self, cut_points: List[int]) -> List[int]:
        if self.config.num_domains <= 1:
            return [0 for _ in range(self.config.num_layers)]
        cut_points = sorted(cut_points)
        if len(cut_points) != self.config.num_domains - 1:
            raise ValueError("cut_points length must be num_domains - 1")
        placement: List[int] = []
        current_domain = 0
        next_cut_index = 0
        next_cut = cut_points[next_cut_index] if cut_points else self.config.num_layers
        for layer_idx in range(self.config.num_layers):
            if layer_idx >= next_cut and current_domain < self.config.num_domains - 1:
                current_domain += 1
                next_cut_index += 1
                next_cut = cut_points[next_cut_index] if next_cut_index < len(cut_points) else self.config.num_layers
            placement.append(current_domain)
        return placement
