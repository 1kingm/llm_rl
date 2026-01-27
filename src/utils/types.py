"""Shared dataclasses for environment and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NetworkState:
    bandwidth_gbps: List[List[float]]
    latency_ms: List[List[float]]


@dataclass(frozen=True)
class RunMetrics:
    total_cycles: float
    comm_volume_gb: float
    utilization: float
    cross_domain_comm_gb: float
    comm_cycles: float


@dataclass(frozen=True)
class RewardBreakdown:
    r_eff: float
    r_util: float
    r_cost: float
    reward: float
