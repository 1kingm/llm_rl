"""Network dynamics model for cross-domain bandwidth/latency fluctuations.

支持两种波动模型：
1. 均匀分布（简单，无时间相关性）
2. Ornstein-Uhlenbeck 过程（真实，具有时间相关性和均值回归特性）

网络动态性模型公式：
$$B_{ij}(t) = B_{ij}^{base} \cdot (1 + \epsilon_{ij}(t)), \quad |\epsilon_{ij}(t)| \leq 0.3$$

其中 ε(t) 可以是：
- 均匀分布: ε ~ U(-σ, σ)
- OU 过程: dε = -θ(ε - μ)dt + σ dW
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .types import NetworkState


def _init_matrix(n: int, diag_value: float, off_value: float) -> List[List[float]]:
    return [[diag_value if i == j else off_value for j in range(n)] for i in range(n)]


@dataclass
class OUProcessConfig:
    """Ornstein-Uhlenbeck 过程配置.

    OU 过程是一种均值回归的随机过程，适合模拟网络波动：
    dX_t = θ(μ - X_t)dt + σ dW_t

    参数说明：
    - theta: 均值回归速度，越大回归越快（典型值 0.1-1.0）
    - mu: 长期均值（通常为 0，表示围绕基线波动）
    - sigma: 波动强度（典型值 0.1-0.3）
    - dt: 时间步长（秒）
    """

    theta: float = 0.5  # 均值回归速度
    mu: float = 0.0  # 长期均值
    sigma: float = 0.15  # 波动强度
    dt: float = 1.0  # 时间步长（秒）
    max_deviation: float = 0.3  # 最大偏离（用于裁剪）


class OUProcess:
    """Ornstein-Uhlenbeck 过程实现.

    用于生成具有时间相关性的网络波动序列。
    """

    def __init__(
        self,
        config: OUProcessConfig | None = None,
        initial_value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or OUProcessConfig()
        self._value = initial_value
        self._rng = random.Random(seed)

    def step(self) -> float:
        """执行一步 OU 过程，返回新值.

        使用 Euler-Maruyama 方法离散化：
        X_{t+dt} = X_t + θ(μ - X_t)dt + σ√dt * Z
        其中 Z ~ N(0, 1)
        """
        theta = self.config.theta
        mu = self.config.mu
        sigma = self.config.sigma
        dt = self.config.dt

        # 生成标准正态随机数
        z = self._rng.gauss(0, 1)

        # Euler-Maruyama 更新
        drift = theta * (mu - self._value) * dt
        diffusion = sigma * math.sqrt(dt) * z
        self._value = self._value + drift + diffusion

        # 裁剪到允许范围
        self._value = max(-self.config.max_deviation, min(self.config.max_deviation, self._value))

        return self._value

    def reset(self, initial_value: float = 0.0, seed: Optional[int] = None) -> None:
        """重置过程状态."""
        self._value = initial_value
        if seed is not None:
            self._rng = random.Random(seed)

    @property
    def value(self) -> float:
        """当前值."""
        return self._value


@dataclass
class NetworkDynamics:
    """网络动态模型.

    支持两种波动模式：
    1. "uniform": 均匀分布波动（简单，无时间相关性）
    2. "ou": Ornstein-Uhlenbeck 过程（真实，具有时间相关性）

    支持非对称带宽（上行/下行不同）：
    - asymmetric_bandwidth=True 时，B[i][j] 和 B[j][i] 独立波动
    - asymmetric_bandwidth=False 时，B[i][j] = B[j][i]（对称）
    """

    num_domains: int
    intra_bandwidth_gbps: float = 400.0
    inter_bandwidth_range_gbps: Tuple[float, float] = (2.0, 5.0)
    intra_latency_ms: float = 0.5
    inter_latency_range_ms: Tuple[float, float] = (20.0, 100.0)
    bandwidth_fluctuation: float = 0.3
    latency_jitter: float = 0.2
    seed: Optional[int] = None

    # 波动模式: "uniform" 或 "ou"
    fluctuation_mode: str = "ou"

    # OU 过程参数
    ou_theta: float = 0.5  # 均值回归速度
    ou_dt: float = 1.0  # 时间步长（秒）

    # 非对称带宽支持
    asymmetric_bandwidth: bool = False  # 是否支持非对称带宽（上行/下行不同）
    asymmetry_factor: float = 0.2  # 非对称因子：上行/下行带宽差异比例

    _rng: random.Random = field(init=False, repr=False)
    _base_bandwidth: List[List[float]] = field(init=False, repr=False)
    _base_latency: List[List[float]] = field(init=False, repr=False)
    _bw_ou_processes: List[List[Optional[OUProcess]]] = field(init=False, repr=False)
    _lat_ou_processes: List[List[Optional[OUProcess]]] = field(init=False, repr=False)
    _step_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._base_bandwidth = self._init_base_bandwidth()
        self._base_latency = self._init_base_latency()
        self._init_ou_processes()
        self._step_count = 0

    def _init_ou_processes(self) -> None:
        """初始化 OU 过程矩阵."""
        n = self.num_domains
        self._bw_ou_processes = [[None for _ in range(n)] for _ in range(n)]
        self._lat_ou_processes = [[None for _ in range(n)] for _ in range(n)]

        if self.fluctuation_mode != "ou":
            return

        # 为每条跨域链路创建独立的 OU 过程
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 带宽 OU 过程
                    bw_config = OUProcessConfig(
                        theta=self.ou_theta,
                        mu=0.0,
                        sigma=self.bandwidth_fluctuation * 0.5,  # σ_OU ≈ σ_uniform / 2
                        dt=self.ou_dt,
                        max_deviation=self.bandwidth_fluctuation,
                    )
                    self._bw_ou_processes[i][j] = OUProcess(
                        bw_config,
                        initial_value=self._rng.uniform(-0.1, 0.1),
                        seed=self._rng.randint(0, 2**31),
                    )

                    # 延迟 OU 过程
                    lat_config = OUProcessConfig(
                        theta=self.ou_theta,
                        mu=0.0,
                        sigma=self.latency_jitter * 0.5,
                        dt=self.ou_dt,
                        max_deviation=self.latency_jitter,
                    )
                    self._lat_ou_processes[i][j] = OUProcess(
                        lat_config,
                        initial_value=self._rng.uniform(-0.1, 0.1),
                        seed=self._rng.randint(0, 2**31),
                    )

    def reset(self, seed: Optional[int] = None) -> None:
        """重置网络状态."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._base_bandwidth = self._init_base_bandwidth()
        self._base_latency = self._init_base_latency()
        self._init_ou_processes()
        self._step_count = 0

    def sample_state(self) -> NetworkState:
        """采样当前网络状态.

        根据 fluctuation_mode 选择波动模型：
        - "uniform": 独立均匀分布波动
        - "ou": Ornstein-Uhlenbeck 过程波动
        """
        self._step_count += 1

        if self.fluctuation_mode == "ou":
            return self._sample_state_ou()
        return self._sample_state_uniform()

    def _sample_state_uniform(self) -> NetworkState:
        """使用均匀分布采样网络状态（原始实现）."""
        bandwidth = []
        for i in range(self.num_domains):
            row = []
            for j in range(self.num_domains):
                base = self._base_bandwidth[i][j]
                epsilon = self._rng.uniform(-self.bandwidth_fluctuation, self.bandwidth_fluctuation)
                row.append(max(0.0, base * (1.0 + epsilon)))
            bandwidth.append(row)

        latency = []
        for i in range(self.num_domains):
            row = []
            for j in range(self.num_domains):
                base = self._base_latency[i][j]
                jitter = self._rng.uniform(-self.latency_jitter, self.latency_jitter)
                row.append(max(0.0, base * (1.0 + jitter)))
            latency.append(row)

        return NetworkState(bandwidth_gbps=bandwidth, latency_ms=latency)

    def _sample_state_ou(self) -> NetworkState:
        """使用 Ornstein-Uhlenbeck 过程采样网络状态.

        OU 过程的特点：
        1. 均值回归：波动会自然回归到基线
        2. 时间相关性：连续采样的值具有相关性
        3. 平滑变化：避免剧烈跳变
        """
        bandwidth = []
        for i in range(self.num_domains):
            row = []
            for j in range(self.num_domains):
                base = self._base_bandwidth[i][j]
                if i == j:
                    # 域内带宽不波动
                    row.append(base)
                else:
                    # 使用 OU 过程生成波动
                    ou = self._bw_ou_processes[i][j]
                    if ou is not None:
                        epsilon = ou.step()
                    else:
                        epsilon = 0.0
                    row.append(max(0.0, base * (1.0 + epsilon)))
            bandwidth.append(row)

        latency = []
        for i in range(self.num_domains):
            row = []
            for j in range(self.num_domains):
                base = self._base_latency[i][j]
                if i == j:
                    # 域内延迟不波动
                    row.append(base)
                else:
                    # 使用 OU 过程生成波动
                    ou = self._lat_ou_processes[i][j]
                    if ou is not None:
                        jitter = ou.step()
                    else:
                        jitter = 0.0
                    row.append(max(0.0, base * (1.0 + jitter)))
            latency.append(row)

        return NetworkState(bandwidth_gbps=bandwidth, latency_ms=latency)

    def get_ou_state(self) -> dict:
        """获取 OU 过程的当前状态（用于调试和可视化）."""
        if self.fluctuation_mode != "ou":
            return {"mode": "uniform", "step": self._step_count}

        bw_state = []
        lat_state = []
        for i in range(self.num_domains):
            bw_row = []
            lat_row = []
            for j in range(self.num_domains):
                if i != j and self._bw_ou_processes[i][j] is not None:
                    bw_row.append(self._bw_ou_processes[i][j].value)
                    lat_row.append(self._lat_ou_processes[i][j].value)
                else:
                    bw_row.append(0.0)
                    lat_row.append(0.0)
            bw_state.append(bw_row)
            lat_state.append(lat_row)

        return {
            "mode": "ou",
            "step": self._step_count,
            "bandwidth_epsilon": bw_state,
            "latency_epsilon": lat_state,
        }

    def _init_base_bandwidth(self) -> List[List[float]]:
        """初始化基线带宽矩阵.

        支持两种模式：
        1. 对称模式 (asymmetric_bandwidth=False): B[i][j] = B[j][i]
        2. 非对称模式 (asymmetric_bandwidth=True): B[i][j] 和 B[j][i] 独立

        非对称模式模拟真实 WAN 场景，上行/下行带宽可能不同。
        """
        bw = _init_matrix(self.num_domains, self.intra_bandwidth_gbps, 0.0)
        low, high = self.inter_bandwidth_range_gbps

        if self.asymmetric_bandwidth:
            # 非对称模式：每条链路独立采样
            for i in range(self.num_domains):
                for j in range(self.num_domains):
                    if i != j:
                        base_bw = self._rng.uniform(low, high)
                        # 添加非对称因子
                        asymmetry = self._rng.uniform(-self.asymmetry_factor, self.asymmetry_factor)
                        bw[i][j] = base_bw * (1.0 + asymmetry)
        else:
            # 对称模式：B[i][j] = B[j][i]
            for i in range(self.num_domains):
                for j in range(i + 1, self.num_domains):
                    base_bw = self._rng.uniform(low, high)
                    bw[i][j] = base_bw
                    bw[j][i] = base_bw

        return bw

    def _init_base_latency(self) -> List[List[float]]:
        """初始化基线延迟矩阵."""
        lat = _init_matrix(self.num_domains, self.intra_latency_ms, 0.0)
        low, high = self.inter_latency_range_ms
        for i in range(self.num_domains):
            for j in range(self.num_domains):
                if i != j:
                    lat[i][j] = self._rng.uniform(low, high)
        return lat

    def get_base_bandwidth(self) -> List[List[float]]:
        """获取基线带宽矩阵."""
        return [row[:] for row in self._base_bandwidth]

    def get_base_latency(self) -> List[List[float]]:
        """获取基线延迟矩阵."""
        return [row[:] for row in self._base_latency]

    def set_base_bandwidth(self, i: int, j: int, value: float) -> None:
        """设置特定链路的基线带宽."""
        if 0 <= i < self.num_domains and 0 <= j < self.num_domains:
            self._base_bandwidth[i][j] = value

    def set_base_latency(self, i: int, j: int, value: float) -> None:
        """设置特定链路的基线延迟."""
        if 0 <= i < self.num_domains and 0 <= j < self.num_domains:
            self._base_latency[i][j] = value


def simulate_network_trajectory(
    dynamics: NetworkDynamics,
    num_steps: int,
    link: Tuple[int, int] = (0, 1),
) -> Tuple[List[float], List[float]]:
    """模拟网络轨迹（用于可视化和分析）.

    Args:
        dynamics: NetworkDynamics 实例
        num_steps: 模拟步数
        link: 要跟踪的链路 (i, j)

    Returns:
        (bandwidth_trajectory, latency_trajectory): 带宽和延迟的时间序列
    """
    i, j = link
    bw_trajectory = []
    lat_trajectory = []

    for _ in range(num_steps):
        state = dynamics.sample_state()
        bw_trajectory.append(state.bandwidth_gbps[i][j])
        lat_trajectory.append(state.latency_ms[i][j])

    return bw_trajectory, lat_trajectory


def compute_autocorrelation(values: List[float], max_lag: int = 20) -> List[float]:
    """计算自相关函数（用于验证 OU 过程的时间相关性）.

    Args:
        values: 时间序列
        max_lag: 最大滞后

    Returns:
        autocorrelation: 自相关系数列表
    """
    n = len(values)
    if n < 2:
        return [1.0]

    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var == 0:
        return [1.0] * min(max_lag + 1, n)

    autocorr = []
    for lag in range(min(max_lag + 1, n)):
        if lag == 0:
            autocorr.append(1.0)
        else:
            cov = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)) / (n - lag)
            autocorr.append(cov / var)

    return autocorr


@dataclass
class MultiDomainNetworkDynamics:
    """多域网络动态模型.

    专为跨域 LLM 训练场景设计，提供更直观的接口：
    - 域间带宽矩阵 (N×N)
    - 域间延迟矩阵 (N×N)
    - 每对域之间有独立的 OU 过程

    与 NetworkDynamics 的区别：
    1. 更清晰的域间/域内参数分离
    2. 支持直接设置域间带宽矩阵
    3. 提供域间通信成本估算
    """

    num_domains: int
    inter_bandwidth_gbs: float = 25.0  # 域间带宽 (GB/s)
    inter_latency_ms: float = 10.0  # 域间延迟 (ms)
    bandwidth_fluctuation: float = 0.3
    latency_jitter: float = 0.2
    seed: Optional[int] = None
    fluctuation_mode: str = "ou"
    ou_theta: float = 0.5
    ou_dt: float = 1.0
    asymmetric_bandwidth: bool = False

    _rng: random.Random = field(init=False, repr=False)
    _base_bandwidth: List[List[float]] = field(init=False, repr=False)
    _base_latency: List[List[float]] = field(init=False, repr=False)
    _bw_ou_processes: List[List[Optional[OUProcess]]] = field(init=False, repr=False)
    _lat_ou_processes: List[List[Optional[OUProcess]]] = field(init=False, repr=False)
    _step_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._base_bandwidth = self._init_base_bandwidth()
        self._base_latency = self._init_base_latency()
        self._init_ou_processes()
        self._step_count = 0

    def _init_base_bandwidth(self) -> List[List[float]]:
        """初始化域间带宽矩阵."""
        n = self.num_domains
        bw = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    # 域内带宽设为无穷大（用大值表示）
                    bw[i][j] = 1000.0
                else:
                    # 域间带宽
                    if self.asymmetric_bandwidth:
                        # 非对称：添加随机偏移
                        offset = self._rng.uniform(-0.2, 0.2)
                        bw[i][j] = self.inter_bandwidth_gbs * (1.0 + offset)
                    else:
                        bw[i][j] = self.inter_bandwidth_gbs

        return bw

    def _init_base_latency(self) -> List[List[float]]:
        """初始化域间延迟矩阵."""
        n = self.num_domains
        lat = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    # 域内延迟很小
                    lat[i][j] = 0.001  # 1 µs
                else:
                    # 域间延迟
                    lat[i][j] = self.inter_latency_ms

        return lat

    def _init_ou_processes(self) -> None:
        """初始化 OU 过程矩阵."""
        n = self.num_domains
        self._bw_ou_processes = [[None for _ in range(n)] for _ in range(n)]
        self._lat_ou_processes = [[None for _ in range(n)] for _ in range(n)]

        if self.fluctuation_mode != "ou":
            return

        for i in range(n):
            for j in range(n):
                if i != j:
                    # 带宽 OU 过程
                    bw_config = OUProcessConfig(
                        theta=self.ou_theta,
                        mu=0.0,
                        sigma=self.bandwidth_fluctuation * 0.5,
                        dt=self.ou_dt,
                        max_deviation=self.bandwidth_fluctuation,
                    )
                    self._bw_ou_processes[i][j] = OUProcess(
                        bw_config,
                        initial_value=self._rng.uniform(-0.1, 0.1),
                        seed=self._rng.randint(0, 2**31),
                    )

                    # 延迟 OU 过程
                    lat_config = OUProcessConfig(
                        theta=self.ou_theta,
                        mu=0.0,
                        sigma=self.latency_jitter * 0.5,
                        dt=self.ou_dt,
                        max_deviation=self.latency_jitter,
                    )
                    self._lat_ou_processes[i][j] = OUProcess(
                        lat_config,
                        initial_value=self._rng.uniform(-0.1, 0.1),
                        seed=self._rng.randint(0, 2**31),
                    )

    def reset(self, seed: Optional[int] = None) -> None:
        """重置网络状态."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._base_bandwidth = self._init_base_bandwidth()
        self._base_latency = self._init_base_latency()
        self._init_ou_processes()
        self._step_count = 0

    def sample_bandwidth_matrix(self) -> List[List[float]]:
        """采样当前域间带宽矩阵 (N×N).

        返回值单位: GB/s
        """
        self._step_count += 1
        n = self.num_domains
        bw = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                base = self._base_bandwidth[i][j]
                if i == j:
                    bw[i][j] = base
                else:
                    if self.fluctuation_mode == "ou" and self._bw_ou_processes[i][j] is not None:
                        epsilon = self._bw_ou_processes[i][j].step()
                    else:
                        epsilon = self._rng.uniform(-self.bandwidth_fluctuation, self.bandwidth_fluctuation)
                    bw[i][j] = max(0.0, base * (1.0 + epsilon))

        return bw

    def sample_latency_matrix(self) -> List[List[float]]:
        """采样当前域间延迟矩阵 (N×N).

        返回值单位: ms
        """
        n = self.num_domains
        lat = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                base = self._base_latency[i][j]
                if i == j:
                    lat[i][j] = base
                else:
                    if self.fluctuation_mode == "ou" and self._lat_ou_processes[i][j] is not None:
                        jitter = self._lat_ou_processes[i][j].step()
                    else:
                        jitter = self._rng.uniform(-self.latency_jitter, self.latency_jitter)
                    lat[i][j] = max(0.0, base * (1.0 + jitter))

        return lat

    def estimate_cross_domain_cost(
        self,
        placement: List[int],
        data_size_bytes: int,
    ) -> Tuple[float, float]:
        """估算跨域通信成本.

        Args:
            placement: 层到域的映射
            data_size_bytes: 每次通信的数据量

        Returns:
            (total_time_ms, total_data_gb): 总通信时间和总数据量
        """
        bw_matrix = self.sample_bandwidth_matrix()
        lat_matrix = self.sample_latency_matrix()

        total_time_ms = 0.0
        total_data_gb = 0.0

        for i in range(len(placement) - 1):
            src = placement[i]
            dst = placement[i + 1]
            if src != dst:
                # 跨域通信
                bw_gbs = bw_matrix[src][dst]
                lat_ms = lat_matrix[src][dst]
                data_gb = data_size_bytes / 1e9

                # 通信时间 = 延迟 + 数据量/带宽
                if bw_gbs > 0:
                    transfer_time_ms = (data_gb / bw_gbs) * 1000  # GB / (GB/s) = s -> ms
                else:
                    transfer_time_ms = float('inf')

                total_time_ms += lat_ms + transfer_time_ms
                total_data_gb += data_gb

        return total_time_ms, total_data_gb

    def get_bandwidth_matrix(self) -> List[List[float]]:
        """获取当前基线带宽矩阵."""
        return [row[:] for row in self._base_bandwidth]

    def get_latency_matrix(self) -> List[List[float]]:
        """获取当前基线延迟矩阵."""
        return [row[:] for row in self._base_latency]

    def set_bandwidth(self, src: int, dst: int, bandwidth_gbs: float) -> None:
        """设置特定域对的带宽."""
        if 0 <= src < self.num_domains and 0 <= dst < self.num_domains:
            self._base_bandwidth[src][dst] = bandwidth_gbs

    def set_latency(self, src: int, dst: int, latency_ms: float) -> None:
        """设置特定域对的延迟."""
        if 0 <= src < self.num_domains and 0 <= dst < self.num_domains:
            self._base_latency[src][dst] = latency_ms
