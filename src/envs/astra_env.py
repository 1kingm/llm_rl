"""Gym-like environment wrapper for Astra-sim backed scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import random
from typing import Iterable, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None

from ..algorithms.reward_functions import (
    RewardBaselines,
    RewardCalculator,
    RewardWeights,
    compute_cross_domain_comm,
)
from ..utils.astra_adapter import (
    AstraSystemConfig,
    MultiDomainTopologyConfig,
    find_ns3_binary,
    generate_allreduce_workload,
    generate_network_config_yaml,
    generate_network_config_yaml_2d,
    generate_logical_topology_json,
    generate_remote_memory_config,
    generate_system_config_v2,
    generate_workload_et,
    mock_run,
    run_astra,
    run_astra_ns3,
)
from ..utils.network_dynamics import NetworkDynamics
from ..utils.types import NetworkState, RunMetrics


@dataclass
class EnvConfig:
    num_domains: int = 3
    num_layers: int = 96
    nodes_per_domain: int = 8
    topology: str = "ring"
    out_dir: str = "configs/generated"
    results_path: str = "results/run_stats.csv"
    use_mock: bool = False
    backend: str = "ns3"  # "analytical" or "ns3"
    astra_bin: str = "astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware"
    ns3_bin: str = "astra-sim/extern/network_backend/ns-3-src/build/scratch/ns3.42-AstraSimNetwork-default"
    ns3_build_dir: str = "astra-sim/extern/network_backend/ns-3-src/build/scratch"
    ns3_network_config: str = "astra-sim/extern/network_backend/ns-3-src/scratch/config/config_clos.txt"
    ns3_comm_group_config: str = "empty"
    ns3_logical_topology_dims: Optional[List[int]] = None
    remote_mem_config: str = "astra-sim/examples/remote_memory/analytical/no_memory_expansion.json"
    peak_perf_tflops: float = 120.0
    local_mem_bw_gbps: float = 1600.0
    layer_runtime_us: int = 100
    comm_size_bytes: int = 1_000_000
    allow_et_fallback: bool = False
    seed: int = 42
    bandwidth_fluctuation: float = 0.3
    latency_jitter: float = 0.2
    inter_bandwidth_range_gbps: Tuple[float, float] = (2.0, 5.0)
    inter_latency_range_ms: Tuple[float, float] = (20.0, 100.0)
    intra_bandwidth_gbps: float = 400.0
    intra_latency_ms: float = 0.5
    network_aggregation: str = "min"  # min or avg
    # 网络波动模式: "uniform" 或 "ou" (Ornstein-Uhlenbeck)
    fluctuation_mode: str = "ou"
    # OU 过程参数
    ou_theta: float = 0.5  # 均值回归速度
    ou_dt: float = 1.0  # 时间步长（秒）
    # 动态 baseline 模式：设为 True 时基于工作负载参数计算 baseline
    use_dynamic_baseline: bool = True
    # 自适应 baseline 更新：设为 True 时基于历史指标更新 baseline
    adaptive_baseline: bool = False
    adaptive_baseline_alpha: float = 0.1
    # 静态 baseline 值（仅当 use_dynamic_baseline=False 时使用）
    baseline_cycles: float = 1.0e7
    baseline_comm_gb: float = 50.0
    baseline_comm_cycles: float = 1.0e7

    # ========== 多域拓扑配置 (2D Topology) ==========
    # 启用多域模式时，使用 2D 拓扑 (域内 + 域间)
    use_multi_domain: bool = False
    gpus_per_domain: int = 4  # 每域 GPU 数
    intra_topology: str = "Switch"  # 域内拓扑: Switch, Ring, FullyConnected
    inter_topology: str = "Ring"  # 域间拓扑: Switch, Ring, FullyConnected
    intra_bandwidth_gbs: float = 400.0  # 域内带宽 (GB/s)
    inter_bandwidth_gbs: float = 25.0  # 域间带宽 (GB/s)
    intra_latency_ns: float = 500.0  # 域内延迟 (ns)
    inter_latency_ns: float = 10_000_000.0  # 域间延迟 (ns) = 10 ms

    # ========== 工作负载类型 ==========
    # "p2p": 点对点通信 (原始模式)
    # "allreduce": AllReduce 集合通信 (推荐用于 LLM 训练)
    workload_type: str = "p2p"
    allreduce_data_size_bytes: int = 64 * 1024 * 1024  # AllReduce 数据量 (64 MB)
    allreduce_iterations: int = 5  # AllReduce 迭代次数


BaseEnv = gym.Env if gym is not None else object


class AstraSimEnv(BaseEnv):
    """
    Gym-like environment for cross-domain scheduling using Astra-sim configs.

    The environment produces a flattened observation:
    [bandwidth_matrix_norm, latency_matrix_norm, domain_loads]
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self._rng = random.Random(self.config.seed)

        self.network = NetworkDynamics(
            num_domains=self.config.num_domains,
            intra_bandwidth_gbps=self.config.intra_bandwidth_gbps,
            inter_bandwidth_range_gbps=self.config.inter_bandwidth_range_gbps,
            intra_latency_ms=self.config.intra_latency_ms,
            inter_latency_range_ms=self.config.inter_latency_range_ms,
            bandwidth_fluctuation=self.config.bandwidth_fluctuation,
            latency_jitter=self.config.latency_jitter,
            seed=self.config.seed,
            fluctuation_mode=self.config.fluctuation_mode,
            ou_theta=self.config.ou_theta,
            ou_dt=self.config.ou_dt,
        )

        # 初始化 baseline：动态计算或使用静态值
        if self.config.use_dynamic_baseline:
            baselines = RewardBaselines.from_workload(
                num_layers=self.config.num_layers,
                layer_runtime_us=self.config.layer_runtime_us,
                comm_size_bytes=self.config.comm_size_bytes,
                num_domains=self.config.num_domains,
            )
        else:
            baselines = RewardBaselines(
                baseline_cycles=self.config.baseline_cycles,
                baseline_comm_gb=self.config.baseline_comm_gb,
                baseline_comm_cycles=self.config.baseline_comm_cycles,
            )

        self.reward_calculator = RewardCalculator(
            weights=RewardWeights(),
            baselines=baselines,
        )
        self.current_network_state: Optional[NetworkState] = None
        self._last_domain_loads: List[float] = self._default_domain_loads()
        self._last_placement: Optional[List[int]] = None

        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([self.config.num_domains] * self.config.num_layers)
            state_dim = self._state_dim()
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=float)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[float], dict]:
        if seed is not None:
            self.config.seed = seed
            self._rng = random.Random(seed)
        self.network.reset(seed=seed)
        self.current_network_state = self.network.sample_state()
        self._last_domain_loads = self._default_domain_loads()
        self._last_placement = None
        observation = self._build_observation(self.current_network_state, self._last_domain_loads)
        return observation, {}

    def step(self, action: Iterable[int]) -> Tuple[List[float], float, bool, bool, dict]:
        placement = list(action)
        if len(placement) != self.config.num_layers:
            raise ValueError(f"placement length {len(placement)} != num_layers {self.config.num_layers}")

        self._last_placement = placement

        # 计算总 NPU 数
        if self.config.use_multi_domain:
            total_npus = self.config.num_domains * self.config.gpus_per_domain
        else:
            total_npus = self.config.num_domains

        # 精确计算跨域通信量
        cross_edges, cross_domain_comm_gb = compute_cross_domain_comm(
            placement, self.config.comm_size_bytes
        )
        estimated_comm_gb = cross_domain_comm_gb

        network_state = self.current_network_state or self.network.sample_state()
        self.current_network_state = network_state
        domain_loads = self._compute_domain_loads(placement)
        self._last_domain_loads = domain_loads

        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        backend = self.config.backend.lower()
        if backend == "ns3":
            network_cfg = Path(self.config.ns3_network_config)
        else:
            network_cfg = self._write_network_config(out_dir / "network_config.yml", network_state)

        system_cfg = generate_system_config_v2(
            out_dir / "system_config.json",
            AstraSystemConfig(
                preferred_dataset_splits=total_npus,
                local_mem_bw_gbps=self.config.local_mem_bw_gbps,
                peak_perf_tflops=self.config.peak_perf_tflops,
                roofline_enabled=1,
            ),
        )

        # 根据工作负载类型生成不同的工作负载
        if self.config.workload_type == "allreduce":
            workload_cfg = generate_allreduce_workload(
                out_dir / "workload",
                num_npus=total_npus,
                data_size_bytes=self.config.allreduce_data_size_bytes,
                num_iterations=self.config.allreduce_iterations,
                compute_us=self.config.layer_runtime_us,
                allow_fallback=self.config.allow_et_fallback,
            )
            # AllReduce 通信量估算
            from ..utils.astra_adapter import estimate_allreduce_comm_volume_gb
            estimated_comm_gb = estimate_allreduce_comm_volume_gb(
                num_npus=total_npus,
                data_size_bytes=self.config.allreduce_data_size_bytes,
                num_iterations=self.config.allreduce_iterations,
            )
        else:
            workload_cfg = generate_workload_et(
                out_dir / "workload",
                placement,
                num_npus=total_npus,
                layer_runtime_us=self.config.layer_runtime_us,
                comm_size_bytes=self.config.comm_size_bytes,
                allow_fallback=self.config.allow_et_fallback,
            )

        output_path = Path(self.config.results_path)
        if self.config.use_mock:
            mock_run(output_path, placement, comm_size_bytes=self.config.comm_size_bytes)
        else:
            remote_mem_cfg = Path(self.config.remote_mem_config)
            if not remote_mem_cfg.exists():
                remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")
            if backend == "ns3":
                logical_dims = self.config.ns3_logical_topology_dims
                if not logical_dims:
                    if self.config.use_multi_domain:
                        logical_dims = [self.config.gpus_per_domain, self.config.num_domains]
                    else:
                        logical_dims = [total_npus]
                logical_topology_cfg = generate_logical_topology_json(
                    out_dir / "logical_topology.json",
                    logical_dims,
                )
                ns3_bin = Path(self.config.ns3_bin)
                if not ns3_bin.exists():
                    ns3_bin = find_ns3_binary(Path(self.config.ns3_build_dir))
                run_astra_ns3(
                    ns3_bin,
                    network_cfg,
                    system_cfg,
                    workload_cfg,
                    logical_topology_cfg,
                    remote_mem_cfg=remote_mem_cfg,
                    comm_group_cfg=self.config.ns3_comm_group_config,
                    output_path=output_path,
                    log_path=Path("results/astra_stdout.log"),
                    comm_volume_gb=estimated_comm_gb,
                )
            else:
                run_astra(
                    Path(self.config.astra_bin),
                    network_cfg,
                    system_cfg,
                    workload_cfg,
                    remote_mem_cfg=remote_mem_cfg,
                    output_path=output_path,
                    log_path=Path("results/astra_stdout.log"),
                    comm_volume_gb=estimated_comm_gb,
                )

        metrics = self._parse_metrics(output_path, estimated_comm_gb)

        # 使用精确的跨域通信量计算奖励
        reward_breakdown = self.reward_calculator.compute(
            metrics,
            placement=placement,
            comm_size_bytes=self.config.comm_size_bytes,
        )

        # 自适应更新 baseline
        if self.config.adaptive_baseline:
            self.reward_calculator.baselines.update_from_metrics(
                metrics, alpha=self.config.adaptive_baseline_alpha
            )

        next_network_state = self.network.sample_state()
        self.current_network_state = next_network_state
        observation = self._build_observation(next_network_state, domain_loads)
        info = {
            "metrics": metrics,
            "reward_breakdown": reward_breakdown,
            "network_config": str(network_cfg),
            "backend": backend,
            "system_config": str(system_cfg),
            "workload_config": str(workload_cfg),
            "network_state": network_state,
            "domain_loads": domain_loads,
            "cross_edges": cross_edges,
            "cross_domain_comm_gb": cross_domain_comm_gb,
        }
        return observation, reward_breakdown.reward, False, False, info

    def sample_random_action(self) -> List[int]:
        return [self._rng.randrange(self.config.num_domains) for _ in range(self.config.num_layers)]

    def get_total_npus(self) -> int:
        """获取总 NPU 数量.

        多域模式: num_domains × gpus_per_domain
        单域模式: num_domains
        """
        if self.config.use_multi_domain:
            return self.config.num_domains * self.config.gpus_per_domain
        return self.config.num_domains

    def _state_dim(self) -> int:
        k = self.config.num_domains
        return k * k * 2 + k

    def _build_observation(self, network_state: NetworkState, domain_loads: Optional[List[float]] = None) -> List[float]:
        bw_norm = self._normalize_matrix(network_state.bandwidth_gbps, self.config.intra_bandwidth_gbps)
        lat_norm = self._normalize_matrix(
            network_state.latency_ms,
            max(1.0, self.config.inter_latency_range_ms[1]),
        )
        loads = domain_loads or self._default_domain_loads()

        return self._flatten(bw_norm) + self._flatten(lat_norm) + loads

    @staticmethod
    def _flatten(matrix: List[List[float]]) -> List[float]:
        return [value for row in matrix for value in row]

    @staticmethod
    def _normalize_matrix(matrix: List[List[float]], scale: float) -> List[List[float]]:
        if scale <= 0:
            scale = 1.0
        return [[value / scale for value in row] for row in matrix]

    def _write_network_config(self, path: Path, network_state: NetworkState) -> Path:
        """
        生成 Astra-sim 网络配置文件。

        支持两种模式：
        1. 单域模式 (use_multi_domain=False): 使用 1D 拓扑
        2. 多域模式 (use_multi_domain=True): 使用 2D 拓扑 (域内 + 域间)

        注意: Astra-sim analytical 后端的 YAML 格式仅支持单一带宽/延迟值，
        不支持逐链路的异构网络参数矩阵。因此必须将 N×N 矩阵聚合为标量。
        聚合策略由 config.network_aggregation 控制:
        - "min": 使用最小值（保守估计，适合瓶颈分析）
        - "avg": 使用平均值（适合整体性能估计）

        如需完整异构网络模拟，请使用 Astra-sim NS3 后端。
        """
        # 多域模式：使用 2D 拓扑
        if self.config.use_multi_domain:
            topo_config = MultiDomainTopologyConfig(
                num_domains=self.config.num_domains,
                gpus_per_domain=self.config.gpus_per_domain,
                intra_topology=self.config.intra_topology,
                inter_topology=self.config.inter_topology,
                intra_bandwidth_gbs=self.config.intra_bandwidth_gbs,
                inter_bandwidth_gbs=self.config.inter_bandwidth_gbs,
                intra_latency_ns=self.config.intra_latency_ns,
                inter_latency_ns=self.config.inter_latency_ns,
            )
            return generate_network_config_yaml_2d(path, topo_config)

        # 单域模式：使用 1D 拓扑，聚合网络状态矩阵
        agg_bw = self._aggregate_offdiag(network_state.bandwidth_gbps, self.config.network_aggregation)
        agg_lat = self._aggregate_offdiag(network_state.latency_ms, self.config.network_aggregation)

        # 检测异构性损失并记录警告
        # 使用基于统计学的动态阈值
        bw_values = self._collect_offdiag(network_state.bandwidth_gbps)
        lat_values = self._collect_offdiag(network_state.latency_ms)
        heterogeneity_threshold = self._compute_heterogeneity_threshold(self.config.num_domains)

        if bw_values:
            bw_cv = self._coefficient_of_variation(bw_values)
            if bw_cv > heterogeneity_threshold:
                import logging
                logging.getLogger(__name__).warning(
                    f"带宽矩阵异构性较高 (CV={bw_cv:.2%} > 阈值{heterogeneity_threshold:.2%})，"
                    f"聚合为 {agg_bw:.2f} Gbps 可能损失精度。考虑使用 NS3 后端。"
                )
        if lat_values:
            lat_cv = self._coefficient_of_variation(lat_values)
            if lat_cv > heterogeneity_threshold:
                import logging
                logging.getLogger(__name__).warning(
                    f"延迟矩阵异构性较高 (CV={lat_cv:.2%} > 阈值{heterogeneity_threshold:.2%})，"
                    f"聚合为 {agg_lat:.2f} ms 可能损失精度。考虑使用 NS3 后端。"
                )

        return generate_network_config_yaml(
            path,
            num_npus=self.config.num_domains,
            topology=self.config.topology,
            bandwidth_gbps=agg_bw,
            latency_ms=agg_lat,
        )

    @staticmethod
    def _aggregate_offdiag(matrix: List[List[float]], mode: str) -> float:
        values: List[float] = []
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if i != j:
                    values.append(value)
        if not values:
            return 0.0
        if mode == "min":
            return min(values)
        total = sum(values)
        return total / len(values)

    @staticmethod
    def _collect_offdiag(matrix: List[List[float]]) -> List[float]:
        """收集矩阵中所有非对角线元素。"""
        values: List[float] = []
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if i != j:
                    values.append(value)
        return values

    @staticmethod
    def _coefficient_of_variation(values: List[float]) -> float:
        """计算变异系数 (CV = std / mean)，用于衡量异构性。"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        return std / mean

    @staticmethod
    def _compute_heterogeneity_threshold(num_domains: int) -> float:
        """基于统计学计算异构性阈值.

        阈值设计原则：
        1. 域数越少，允许的异构性越高（因为聚合损失的信息相对较少）
        2. 域数越多，阈值越严格（因为聚合会丢失更多链路信息）

        理论依据：
        - 对于 N 个域，有 N*(N-1) 条跨域链路
        - 聚合为单一值时，信息损失与链路数成正比
        - 使用 Herfindahl 指数的思想：1/N 作为基准

        阈值公式：
        threshold = 0.5 / sqrt(num_domains - 1)  (当 num_domains >= 2)

        示例：
        - 2 域: 0.5 (允许较高异构性)
        - 3 域: 0.35
        - 5 域: 0.25
        - 10 域: 0.17
        """
        if num_domains <= 1:
            return 1.0  # 单域无异构性问题
        if num_domains == 2:
            return 0.5  # 2 域允许较高异构性

        import math
        # 基于链路数的动态阈值
        threshold = 0.5 / math.sqrt(num_domains - 1)
        # 设置下限，避免阈值过于严格
        return max(0.15, threshold)

    def _default_domain_loads(self) -> List[float]:
        if self.config.num_domains <= 0:
            return []
        return [1.0 / self.config.num_domains for _ in range(self.config.num_domains)]

    def _compute_domain_loads(self, placement: List[int]) -> List[float]:
        if self.config.num_domains <= 0:
            return []
        counts = [0] * self.config.num_domains
        for domain_id in placement:
            if 0 <= domain_id < self.config.num_domains:
                counts[domain_id] += 1
        total = max(1, sum(counts))
        return [count / total for count in counts]

    def _parse_metrics(self, result_file: Path, estimated_comm_gb: float) -> RunMetrics:
        if not result_file.exists():
            raise FileNotFoundError(f"Astra-sim result file not found: {result_file}")
        with result_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError(f"No metrics in result file: {result_file}")

        last = rows[-1]
        total_cycles = float(last.get("total_cycles", 0.0))
        comm_volume = float(last.get("comm_volume_gb", 0.0))
        comm_cycles = float(last.get("comm_cycles", 0.0)) if "comm_cycles" in last else 0.0
        utilization = float(last.get("utilization", 0.0))

        if comm_volume == 0.0 and estimated_comm_gb > 0.0:
            comm_volume = estimated_comm_gb

        # For the initial phase, approximate cross-domain comm with total comm volume.
        cross_domain_comm = float(last.get("cross_domain_comm_gb", 0.0))
        if cross_domain_comm == 0.0 and estimated_comm_gb > 0.0:
            cross_domain_comm = estimated_comm_gb

        return RunMetrics(
            total_cycles=total_cycles,
            comm_volume_gb=comm_volume,
            utilization=utilization,
            cross_domain_comm_gb=cross_domain_comm,
            comm_cycles=comm_cycles,
        )
