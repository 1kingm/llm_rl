"""Reward computation for Hi-PPO scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ..utils.types import RewardBreakdown, RunMetrics


class WeightPreset(Enum):
    """预设的奖励权重配置.

    基于多目标优化理论和实际应用场景设计。
    """

    # 均衡模式：三个目标等权重
    BALANCED = "balanced"
    # 效率优先：适合对训练时间敏感的场景
    EFFICIENCY_FIRST = "efficiency_first"
    # 成本敏感：适合跨云计费场景
    COST_SENSITIVE = "cost_sensitive"
    # 利用率优先：适合资源紧张场景
    UTILIZATION_FIRST = "utilization_first"


# 预设权重值（基于 Pareto 前沿分析和实际需求）
WEIGHT_PRESETS = {
    WeightPreset.BALANCED: (0.34, 0.33, 0.33),
    WeightPreset.EFFICIENCY_FIRST: (0.5, 0.25, 0.25),
    WeightPreset.COST_SENSITIVE: (0.25, 0.25, 0.5),
    WeightPreset.UTILIZATION_FIRST: (0.25, 0.5, 0.25),
}


@dataclass
class RewardWeights:
    """奖励权重配置.

    权重设计原则：
    1. 归一化：w_eff + w_util + w_cost = 1.0
    2. 可解释：每个权重对应一个明确的优化目标
    3. 可调节：支持预设和自定义

    理论依据：
    - 基于线性加权和（Weighted Sum）方法的多目标优化
    - 权重决定了 Pareto 前沿上的搜索方向
    - 不同权重组合对应不同的应用场景

    默认权重 (0.4, 0.3, 0.3) 的设计考虑：
    - 效率权重略高：训练时间是分布式训练的核心指标
    - 利用率和成本均衡：避免过度优化单一目标
    """

    w_eff: float = 0.4   # 效率权重：训练周期缩短
    w_util: float = 0.3  # 利用率权重：资源利用率
    w_cost: float = 0.3  # 成本权重：跨域通信成本

    def __post_init__(self) -> None:
        """验证权重有效性."""
        total = self.w_eff + self.w_util + self.w_cost
        if abs(total - 1.0) > 0.01:
            # 自动归一化
            self.w_eff /= total
            self.w_util /= total
            self.w_cost /= total

    @classmethod
    def from_preset(cls, preset: WeightPreset) -> "RewardWeights":
        """从预设创建权重配置.

        Args:
            preset: 预设类型

        Returns:
            RewardWeights 实例
        """
        w_eff, w_util, w_cost = WEIGHT_PRESETS[preset]
        return cls(w_eff=w_eff, w_util=w_util, w_cost=w_cost)

    @classmethod
    def from_priority(
        cls,
        efficiency_priority: float = 1.0,
        utilization_priority: float = 1.0,
        cost_priority: float = 1.0,
    ) -> "RewardWeights":
        """基于优先级创建权重配置.

        优先级会被归一化为权重。

        Args:
            efficiency_priority: 效率优先级（相对值）
            utilization_priority: 利用率优先级（相对值）
            cost_priority: 成本优先级（相对值）

        Returns:
            RewardWeights 实例

        Example:
            # 效率是成本的两倍重要
            weights = RewardWeights.from_priority(
                efficiency_priority=2.0,
                utilization_priority=1.0,
                cost_priority=1.0
            )
            # 结果: w_eff=0.5, w_util=0.25, w_cost=0.25
        """
        total = efficiency_priority + utilization_priority + cost_priority
        if total <= 0:
            total = 3.0
            efficiency_priority = utilization_priority = cost_priority = 1.0

        return cls(
            w_eff=efficiency_priority / total,
            w_util=utilization_priority / total,
            w_cost=cost_priority / total,
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组."""
        return (self.w_eff, self.w_util, self.w_cost)


@dataclass
class RewardBaselines:
    """奖励函数的基线值.

    基线值用于归一化各奖励分量，使其量级一致。
    支持两种模式：
    1. 静态模式：使用固定值（适合调试）
    2. 动态模式：基于工作负载参数计算（推荐用于训练）
    """

    baseline_cycles: float = 1.0e7
    baseline_comm_gb: float = 50.0
    baseline_comm_cycles: float = 1.0e7

    # 动态计算所需的工作负载参数
    num_layers: int = 96
    layer_runtime_us: int = 100
    comm_size_bytes: int = 1_000_000
    num_domains: int = 3
    clock_freq_ghz: float = 1.5  # GPU 时钟频率，用于 us -> cycles 转换

    # 历史数据用于自适应更新
    _history_cycles: List[float] = field(default_factory=list)
    _history_comm_cycles: List[float] = field(default_factory=list)

    @classmethod
    def from_workload(
        cls,
        num_layers: int,
        layer_runtime_us: int,
        comm_size_bytes: int,
        num_domains: int,
        clock_freq_ghz: float = 1.5,
    ) -> "RewardBaselines":
        """基于工作负载参数动态计算基线值.

        计算逻辑：
        - baseline_cycles: 假设所有层串行执行的总周期数
        - baseline_comm_cycles: 假设最坏情况（每层都跨域）的通信周期
        - baseline_comm_gb: 假设最坏情况的通信量（GB）

        Args:
            num_layers: 模型层数
            layer_runtime_us: 每层计算时间（微秒）
            comm_size_bytes: 每次跨域通信的数据量（字节）
            num_domains: 域数量
            clock_freq_ghz: GPU 时钟频率（GHz）

        Returns:
            动态计算的 RewardBaselines 实例
        """
        # 计算周期数: cycles = time_us * freq_ghz * 1000
        cycles_per_us = clock_freq_ghz * 1000  # 1 GHz = 1000 cycles/us

        # baseline_cycles: 串行执行所有层的周期数（无并行优化）
        # 乘以 1.5 作为安全边际，考虑内存访问等开销
        baseline_cycles = num_layers * layer_runtime_us * cycles_per_us * 1.5

        # baseline_comm_cycles: 最坏情况下的通信周期
        # 假设每层都需要跨域通信（num_layers - 1 次）
        # 通信周期 ≈ 数据量 / 带宽，假设跨域带宽 5 Gbps
        inter_bw_gbps = 5.0
        bytes_per_cycle = (inter_bw_gbps * 1e9 / 8) / (clock_freq_ghz * 1e9)
        comm_cycles_per_transfer = comm_size_bytes / bytes_per_cycle
        max_cross_edges = num_layers - 1
        baseline_comm_cycles = max_cross_edges * comm_cycles_per_transfer

        # baseline_comm_gb: 最坏情况的通信量
        baseline_comm_gb = (max_cross_edges * comm_size_bytes) / 1e9

        return cls(
            baseline_cycles=baseline_cycles,
            baseline_comm_gb=baseline_comm_gb,
            baseline_comm_cycles=baseline_comm_cycles,
            num_layers=num_layers,
            layer_runtime_us=layer_runtime_us,
            comm_size_bytes=comm_size_bytes,
            num_domains=num_domains,
            clock_freq_ghz=clock_freq_ghz,
        )

    def update_from_metrics(self, metrics: RunMetrics, alpha: float = 0.1) -> None:
        """基于实际运行指标自适应更新基线值.

        使用指数移动平均（EMA）平滑更新，避免剧烈波动。

        Args:
            metrics: 实际运行指标
            alpha: EMA 平滑系数，越大越敏感（默认 0.1）
        """
        if metrics.total_cycles > 0:
            self._history_cycles.append(metrics.total_cycles)
            # 使用 75 分位数作为基线（略高于平均，鼓励优化）
            if len(self._history_cycles) >= 10:
                sorted_cycles = sorted(self._history_cycles[-100:])  # 最近 100 个样本
                p75_cycles = sorted_cycles[int(len(sorted_cycles) * 0.75)]
                self.baseline_cycles = (1 - alpha) * self.baseline_cycles + alpha * p75_cycles

        if metrics.comm_cycles > 0:
            self._history_comm_cycles.append(metrics.comm_cycles)
            if len(self._history_comm_cycles) >= 10:
                sorted_comm = sorted(self._history_comm_cycles[-100:])
                p75_comm = sorted_comm[int(len(sorted_comm) * 0.75)]
                self.baseline_comm_cycles = (1 - alpha) * self.baseline_comm_cycles + alpha * p75_comm


def compute_cross_domain_comm(
    placement: List[int],
    comm_size_bytes: int,
) -> tuple[int, float]:
    """精确计算跨域通信量.

    Args:
        placement: 层到域的映射列表
        comm_size_bytes: 每次跨域通信的数据量（字节）

    Returns:
        (cross_edges, cross_domain_comm_gb): 跨域边数和跨域通信量（GB）
    """
    if len(placement) < 2:
        return 0, 0.0

    cross_edges = sum(
        1 for i in range(len(placement) - 1)
        if placement[i] != placement[i + 1]
    )
    cross_domain_comm_gb = (cross_edges * comm_size_bytes) / 1e9

    return cross_edges, cross_domain_comm_gb


def _count_cross_edges(placement: List[int]) -> int:
    if len(placement) < 2:
        return 0
    return sum(1 for i in range(len(placement) - 1) if placement[i] != placement[i + 1])


def _balance_score(placement: List[int], num_domains: Optional[int] = None) -> float:
    if not placement:
        return 0.0
    num_layers = len(placement)
    if num_domains is None:
        num_domains = max(placement) + 1
    if num_domains <= 0:
        return 0.0

    layers_per_domain = [0] * num_domains
    for domain_id in placement:
        if 0 <= domain_id < num_domains:
            layers_per_domain[domain_id] += 1

    mean_layers = num_layers / num_domains
    if mean_layers <= 0:
        return 0.0
    variance = sum((x - mean_layers) ** 2 for x in layers_per_domain) / num_domains
    std_dev = variance ** 0.5
    cv = std_dev / mean_layers
    return max(0.0, 1.0 - cv)


class RewardCalculator:
    """奖励计算器.

    支持两种通信成本计算模式：
    1. 基于 Astra-sim 输出的 comm_cycles（精确）
    2. 基于 placement 计算的跨域通信量（当 Astra-sim 不可用时）
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        baselines: RewardBaselines | None = None,
        num_domains: Optional[int] = None,
    ) -> None:
        self.weights = weights or RewardWeights()
        self.baselines = baselines or RewardBaselines()
        self.num_domains = num_domains

    def compute(
        self,
        metrics: RunMetrics,
        placement: Optional[List[int]] = None,
        comm_size_bytes: Optional[int] = None,
    ) -> RewardBreakdown:
        """计算奖励分解.

        Args:
            metrics: 运行指标
            placement: 层到域的映射（用于精确计算跨域通信）
            comm_size_bytes: 每次跨域通信的数据量

        Returns:
            RewardBreakdown: 奖励分解
        """
        # 效率奖励: (baseline - actual) / baseline
        # 正值表示比基线快，负值表示比基线慢
        r_eff = (self.baselines.baseline_cycles - metrics.total_cycles) / max(
            1.0, self.baselines.baseline_cycles
        )

        r_util = metrics.utilization

        # 通信成本: 优先使用 comm_cycles，其次使用精确计算的跨域通信量
        if metrics.comm_cycles > 0:
            # 使用 Astra-sim 输出的通信周期
            r_cost = metrics.comm_cycles / max(1.0, self.baselines.baseline_comm_cycles)
        elif placement is not None and comm_size_bytes is not None:
            # 精确计算跨域通信量
            _, cross_comm_gb = compute_cross_domain_comm(placement, comm_size_bytes)
            r_cost = cross_comm_gb / max(1.0, self.baselines.baseline_comm_gb)
        else:
            # 回退到 metrics 中的近似值（可能不准确）
            r_cost = metrics.cross_domain_comm_gb / max(1.0, self.baselines.baseline_comm_gb)

        # 基于放置的奖励修正：鼓励均衡、惩罚过多跨域切分
        if placement is not None:
            balance_score = _balance_score(placement, self.num_domains)
            r_util = metrics.utilization * (0.5 + 0.5 * balance_score)
            cross_edges = _count_cross_edges(placement)
            denom = max(1, (self.num_domains or (max(placement) + 1)) - 1)
            cut_ratio = cross_edges / denom
            r_cost = r_cost * (1.0 + cut_ratio)

        # 总奖励: 效率 + 利用率 - 成本
        reward = (
            self.weights.w_eff * r_eff
            + self.weights.w_util * r_util
            - self.weights.w_cost * r_cost
        )

        return RewardBreakdown(r_eff=r_eff, r_util=r_util, r_cost=r_cost, reward=reward)
