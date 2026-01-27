# Phase 2: Simulation Environment Setup (Astra-sim Backend)

> **Project:** Intelligent Cross-Domain Scheduling for Large Model Training (Task 3)
> **Objective:** Build a `gym`-compatible environment that uses **Astra-sim** as the physics engine
> **Version:** v1.0
> **Date:** 2026-01-26

**核心假设**: 同构算力环境（各域 GPU 型号相同），核心挑战聚焦于**跨域网络动态性**。

---

## 1. 为什么选择 Astra-sim？

| 特性 | SimPy | **Astra-sim** |
|------|-------|---------------|
| 学术认可度 | 通用离散事件模拟 | **顶会标准** (ISCA, MICRO, HPCA, NSDI) |
| 通信模拟精度 | 简化模型 | **精确模拟** All-Reduce, All-Gather |
| 网络拓扑支持 | 需自行实现 | **内置** Torus, Fat-Tree, WAN |
| 输出指标 | 自定义 | **Cycle 级精度**，直接对齐 KPI |

**Astra-sim** 是 Meta、Intel 等大厂主导的开源模拟器，能精确模拟集合通信在不同网络拓扑下的真实延迟。

> **说明**: SimPy 适合快速原型与算法验证；Astra-sim 用于高保真评估与最终实验结果对齐。
> **资源受限声明**: 若缺乏真实集群，主评估以 Astra-sim 为主，配合小规模微基准校准模型参数以提高可信度。

---

## 2. 系统架构 (System Architecture)

构建 "RL-Loop"：PPO Agent 生成配置，Astra-sim 负责"跑分"。

```
┌─────────────────────────────────────────────────────────────────┐
│                        RL Training Loop                          │
│                                                                   │
│  ┌──────────────┐     Action: Split Strategy     ┌────────────┐ │
│  │  Hi-PPO      │ ─────────────────────────────> │  Astra     │ │
│  │  Agent       │                                │  Wrapper   │ │
│  │  (Python)    │ <───────────────────────────── │  (Gym Env) │ │
│  └──────────────┘     Reward & Next State        └─────┬──────┘ │
│                                                        │        │
└────────────────────────────────────────────────────────┼────────┘
                                                         │
                    ┌────────────────────────────────────┼────────┐
                    │           Astra-sim Backend        │        │
                    │                                    ▼        │
                    │  ┌─────────────┐    ┌─────────────────────┐ │
                    │  │ Network     │    │ Workload Trace      │ │
                    │  │ Config      │    │ (Chakra ET)         │ │
                    │  │ (YAML)      │    │                     │ │
                    │  └──────┬──────┘    └──────────┬──────────┘ │
                    │         │                      │            │
                    │         └──────────┬───────────┘            │
                    │                    ▼                        │
                    │         ┌─────────────────────┐             │
                    │         │  Astra-sim Binary   │             │
                    │         │  (C++ Core)         │             │
                    │         └──────────┬──────────┘             │
                    │                    │                        │
                    │                    ▼                        │
                    │         ┌─────────────────────┐             │
                    │         │  Run Statistics     │             │
                    │         │  (cycles, latency)  │             │
                    │         └─────────────────────┘             │
                    └─────────────────────────────────────────────┘
```

---

## 3. 关键组件实现 (Implementation Details)

### 3.1 网络拓扑配置 (`network.yml`) —— 模拟"东数西算"

Astra-sim 允许定义网络带宽和延迟。我们需要模拟**网络带宽的层次差异**（域内高带宽，域间低带宽）。

**关键点**: 算力同构，但**网络连接异构**。

```yaml
# network_config.yml - 东数西算网络拓扑
# 注意：算力同构，网络带宽/延迟异构

topology:
  domains:
    - name: "East_DataCenter"
      num_gpus: 8
      intra_bandwidth: "400Gbps"  # NVLink
      intra_latency: "500ns"

    - name: "Central_DataCenter"
      num_gpus: 8
      intra_bandwidth: "400Gbps"
      intra_latency: "500ns"

    - name: "West_DataCenter"
      num_gpus: 8
      intra_bandwidth: "400Gbps"
      intra_latency: "500ns"

  # 跨域链路 - 核心瓶颈
  cross_domain_links:
    - source: "East_DataCenter"
      dest: "Central_DataCenter"
      bandwidth: "5Gbps"      # WAN 低带宽
      latency: "30ms"         # 物理距离延迟

    - source: "Central_DataCenter"
      dest: "West_DataCenter"
      bandwidth: "3Gbps"      # 更低带宽
      latency: "50ms"         # 更远距离

    - source: "East_DataCenter"
      dest: "West_DataCenter"
      bandwidth: "2Gbps"      # 最低带宽
      latency: "80ms"         # 最远距离

# 网络动态性参数
dynamics:
  bandwidth_fluctuation: 0.3  # ±30% 波动
  latency_jitter: 0.2         # ±20% 抖动
  update_interval: "per_epoch"
```

> **实现对齐说明**：当前 analytical backend 仅支持 1D 基础拓扑（Ring/Switch/FullyConnected）。
> 代码实现中会将域间带宽/延迟矩阵压缩为**单值**生成 `network_config.yml`，
> 推荐使用 `min` 聚合以保守反映最差链路（可通过 `network_aggregation` 配置调整）。
> 完整矩阵仍用于策略输入（GNN 编码），但仿真侧采用简化拓扑。

### 3.2 工作负载追踪 (Workload Trace via Chakra)

Astra-sim 使用 **Chakra Execution Trace (ET)** 来描述大模型训练任务。

**PPO Action → Chakra ET 映射**:
- PPO Action: "Layer 1-12 on Domain A, Layer 13-24 on Domain B"
- Chakra ET: 计算量不变，**通信节点 (Comm Nodes)** 的位置和大小改变

```python
# workload_generator.py - 生成 Chakra ET

def generate_chakra_trace(model_config: dict, placement: list) -> str:
    """
    根据模型配置和放置策略生成 Chakra Execution Trace

    Args:
        model_config: 模型配置 (layers, hidden_size, etc.)
        placement: 每层的域分配 [0, 0, 1, 1, 2, ...]

    Returns:
        Chakra ET 文件路径
    """
    trace = {
        "schema": "chakra_0.0.4",
        "nodes": []
    }

    for i, layer in enumerate(model_config["layers"]):
        # 计算节点
        compute_node = {
            "id": i * 2,
            "name": f"layer_{i}_compute",
            "type": "COMP_NODE",
            "domain": placement[i],
            "runtime": layer["flops"] / PEAK_FLOPS,  # 同构，FLOPS 相同
        }
        trace["nodes"].append(compute_node)

        # 通信节点（如果跨域）
        if i < len(placement) - 1 and placement[i] != placement[i+1]:
            comm_node = {
                "id": i * 2 + 1,
                "name": f"layer_{i}_cross_domain_comm",
                "type": "COMM_SEND_NODE",
                "src_domain": placement[i],
                "dst_domain": placement[i+1],
                "comm_size": layer["activation_size"],
            }
            trace["nodes"].append(comm_node)

    return trace
```

### 3.3 Python 仿真环境封装 (`astra_env.py`)

遵循 OpenAI Gym 接口，直接接入 RL 算法。

```python
# envs/astra_env.py

import gymnasium as gym
import subprocess
import json
import numpy as np
from typing import Tuple, Dict, Any

class AstraSimEnv(gym.Env):
    """
    基于 Astra-sim 的跨域调度仿真环境

    核心假设：
    - 同构算力环境（各域 GPU 型号相同）
    - 网络带宽动态波动（±30%）
    - 域间带宽远小于域内带宽
    """

    def __init__(
        self,
        num_domains: int = 3,
        num_layers: int = 96,
        task_profile: str = "llama_70b",
        bandwidth_fluctuation: float = 0.3,
    ):
        super().__init__()

        self.num_domains = num_domains
        self.num_layers = num_layers
        self.task_profile = task_profile
        self.bandwidth_fluctuation = bandwidth_fluctuation

        # 动作空间：每层分配到哪个 Domain [0, 1, ..., K-1]
        self.action_space = gym.spaces.MultiDiscrete(
            [num_domains] * num_layers
        )

        # 状态空间：网络状态 + 资源状态
        # [域间带宽矩阵(K*K) + 域间延迟矩阵(K*K) + 各域负载(K)]
        state_dim = num_domains * num_domains * 2 + num_domains
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )

        # 基准网络配置
        self.base_bandwidth = self._init_base_bandwidth()
        self.base_latency = self._init_base_latency()

        # 奖励权重 (对应 Phase 1 公式)
        self.w1 = 0.4  # 训练效率
        self.w2 = 0.3  # 利用率
        self.w3 = 0.3  # 通信成本

    def _init_base_bandwidth(self) -> np.ndarray:
        """初始化基准带宽矩阵 (Gbps)"""
        B = np.zeros((self.num_domains, self.num_domains))
        # 域内带宽：400 Gbps
        np.fill_diagonal(B, 400)
        # 域间带宽：2-5 Gbps (远小于域内)
        for i in range(self.num_domains):
            for j in range(self.num_domains):
                if i != j:
                    B[i, j] = np.random.uniform(2, 5)
        return B

    def _init_base_latency(self) -> np.ndarray:
        """初始化基准延迟矩阵 (ms)"""
        L = np.zeros((self.num_domains, self.num_domains))
        # 域内延迟：< 1ms
        np.fill_diagonal(L, 0.5)
        # 域间延迟：20-100ms
        for i in range(self.num_domains):
            for j in range(self.num_domains):
                if i != j:
                    L[i, j] = np.random.uniform(20, 100)
        return L

    def _get_current_network_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前网络状态（含动态波动）

        网络动态性模型：B_ij(t) = B_ij^base * (1 + ε(t))
        其中 |ε(t)| <= 0.3
        """
        epsilon = np.random.uniform(
            -self.bandwidth_fluctuation,
            self.bandwidth_fluctuation,
            size=self.base_bandwidth.shape
        )
        current_bandwidth = self.base_bandwidth * (1 + epsilon)

        # 延迟也有波动
        latency_jitter = np.random.uniform(-0.2, 0.2, size=self.base_latency.shape)
        current_latency = self.base_latency * (1 + latency_jitter)

        return current_bandwidth, current_latency

    def _get_observation(self) -> np.ndarray:
        """构建状态向量"""
        bandwidth, latency = self._get_current_network_state()

        # 归一化
        bandwidth_norm = bandwidth / 400  # 相对于域内带宽
        latency_norm = latency / 100      # 相对于最大延迟
        # 负载来自最近一次 placement 的域内层占比（避免随机噪声）
        load_norm = self._compute_domain_loads(self.last_placement)

        state = np.concatenate([
            bandwidth_norm.flatten(),
            latency_norm.flatten(),
            load_norm
        ])
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步仿真

        Args:
            action: 每层的域分配 [0, 0, 1, 1, 2, ...]

        Returns:
            observation: 下一状态
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # Step 1: 生成 Astra-sim 配置
        self._generate_astra_config(action)

        # Step 2: 运行 Astra-sim 仿真
        result_file = self._run_astra_binary()

        # Step 3: 解析仿真结果
        metrics = self._parse_astra_logs(result_file)

        # Step 4: 计算奖励 (Phase 1 公式)
        reward = self._calculate_reward(metrics)

        # Step 5: 获取下一状态
        self.last_placement = action  # 缓存放置用于负载估计
        next_state = self._get_observation()

        return next_state, reward, False, False, metrics

    def _generate_astra_config(self, action: np.ndarray):
        """
        将 PPO 决策转换为 Astra-sim 配置

        Args:
            action: 每层的域分配
        """
        # 生成网络拓扑配置 (Astra-sim 2.0: YAML)
        network_cfg = generate_network_config_yaml(
            "configs/network_config.yml",
            num_npus=self.num_domains,
            topology="Ring",
            bandwidth_gbps=5.0,
            latency_ms=30.0,
        )

        # 生成 System 配置 (JSON)
        system_cfg = generate_system_config_v2(
            "configs/system_config.json",
            AstraSystemConfig(
                preferred_dataset_splits=self.num_domains,
                local_mem_bw_gbps=1600.0,
                peak_perf_tflops=120.0,
            ),
        )

        # 生成 Workload (Chakra ET)
        # 输出为 workload/trace.{rank}.et，传参时使用前缀 workload/trace
        workload_prefix = generate_workload_et(
            "configs/workload",
            placement=action.tolist(),
            num_npus=self.num_domains,
            allow_fallback=False,  # 如 protobuf 版本不匹配，建议升级而非回退
        )

        # 若本地 protobuf 与 Chakra gencode 版本不匹配，
        # 默认直接报错（避免 placement 无效），如需应急可显式开启 allow_fallback。

    def _run_astra_binary(self) -> str:
        """
        执行 Astra-sim 仿真

        Returns:
            结果文件路径
        """
        cmd = [
            "./astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware",
            f"--network-configuration={network_cfg}",
            f"--system-configuration={system_cfg}",
            f"--workload-configuration={workload_prefix}",
            "--remote-memory-configuration=configs/remote_memory.json",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Astra-sim error: {e.stderr.decode()}")

        return "results/run_stats.csv"

    def _parse_astra_logs(self, result_file: str) -> Dict[str, float]:
        """
        解析 Astra-sim 输出

        Returns:
            metrics: {total_cycles, comm_volume, utilization}
        """
        # 实际实现需要解析 CSV 文件（Astra-sim analytical 输出 comm_cycles）
        last_row = read_csv_last_row(result_file)
        est_comm_gb = estimate_comm_volume_gb(action, comm_size_bytes)
        metrics = {
            "total_cycles": last_row["total_cycles"],
            "comm_cycles": last_row["comm_cycles"],
            "comm_volume_gb": est_comm_gb,  # 若 CSV 未提供，使用 placement 估算
            "utilization": last_row["utilization"],
            "cross_domain_comm_gb": est_comm_gb,
        }
        return metrics

    # 注意：若使用 Astra-sim 真实运行，优先采用 comm_cycles 作为通信成本，
    # 并配套 baseline_comm_cycles，避免 GB 与 cycles 混用造成奖励失真。

    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        计算奖励值 (Phase 1 公式)

        R = w1 * R_eff + w2 * R_util - w3 * R_cost
        """
        # 训练效率奖励 (cycles 越少越好)
        baseline_cycles = 1e7
        r_eff = (baseline_cycles - metrics["total_cycles"]) / baseline_cycles

        # 利用率奖励
        r_util = metrics["utilization"]

        # 通信成本惩罚 (跨域通信越少越好)
        baseline_comm = 50  # GB
        r_cost = metrics["cross_domain_comm_gb"] / baseline_comm

        reward = self.w1 * r_eff + self.w2 * r_util - self.w3 * r_cost

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 重新初始化网络状态
        self.base_bandwidth = self._init_base_bandwidth()
        self.base_latency = self._init_base_latency()

        observation = self._get_observation()
        return observation, {}

    def render(self):
        """可视化（可选）"""
        pass
```

---

### 3.3.x 切换到 NS3 后端（当前默认）

本项目已将环境默认后端切到 NS3（并关闭 mock）。关键点：

- `backend: ns3`，`use_mock: false`
- NS3 运行需要：
  - NS3 二进制：`astra-sim/extern/network_backend/ns-3-src/build/scratch/ns3.42-AstraSimNetwork-default`
  - 网络配置：`astra-sim/extern/network_backend/ns-3-src/scratch/config/config_clos.txt`
  - 逻辑拓扑：运行时自动生成（单域 `[num_domains]`；多域 `[gpus_per_domain, num_domains]`）

`configs/algo/hi_ppo.yaml` 中的示例：

```yaml
env:
  backend: "ns3"
  use_mock: false
  ns3_build_dir: "astra-sim/extern/network_backend/ns-3-src/build/scratch"
  ns3_network_config: "astra-sim/extern/network_backend/ns-3-src/scratch/config/config_clos.txt"
  ns3_comm_group_config: "empty"
  ns3_bin: "astra-sim/extern/network_backend/ns-3-src/build/scratch/ns3.42-AstraSimNetwork-default"
```

## 3.4 Astra-sim 配置与 Chakra ET 完整示例（补充）

### 3.4.1 Network 配置示例（JSON）

```json
{
  "topology": "hierarchical",
  "num_domains": 3,
  "bandwidth_matrix": [
    [400, 5, 2],
    [5, 400, 3],
    [2, 3, 400]
  ],
  "latency_matrix": [
    [0.5, 30, 80],
    [30, 0.5, 50],
    [80, 50, 0.5]
  ]
}
```

### 3.4.2 Chakra ET 生成示例（Python）

```python
def generate_chakra_trace(model_config: dict, placement: list) -> dict:
    trace = {"schema": "chakra_0.0.4", "nodes": []}
    for i, layer in enumerate(model_config["layers"]):
        trace["nodes"].append({
            "id": i * 2,
            "name": f"layer_{i}_compute",
            "type": "COMP_NODE",
            "domain": placement[i],
            "runtime": layer["flops"] / model_config["peak_flops"],
        })
        if i < len(placement) - 1 and placement[i] != placement[i + 1]:
            trace["nodes"].append({
                "id": i * 2 + 1,
                "name": f"layer_{i}_cross_domain_comm",
                "type": "COMM_SEND_NODE",
                "src_domain": placement[i],
                "dst_domain": placement[i + 1],
                "comm_size": layer["activation_size"],
            })
    return trace
```

---

## 4. 对齐 KPI 的配置

确保仿真结果符合课题要求：

### 4.1 网络动态性配置

**要求**: 网络带宽波动 ±30%

```python
# 在 step() 中动态更新带宽
def _apply_network_fluctuation(self):
    epsilon = np.random.uniform(-0.3, 0.3, size=self.base_bandwidth.shape)
    self.current_bandwidth = self.base_bandwidth * (1 + epsilon)
```

### 4.2 通信成本计算

**要求**: 通信成本降低 ≥30%

```python
# 只计算跨域通信
def _calculate_cross_domain_comm(self, placement):
    cross_domain_comm = 0
    for i in range(len(placement) - 1):
        if placement[i] != placement[i+1]:
            cross_domain_comm += self.layer_activation_sizes[i]
    return cross_domain_comm
```

### 4.3 决策延迟约束

**要求**: 决策延迟 < 1秒

```python
# 在训练时监控推理时间
import time

def get_action_with_timing(agent, state):
    start = time.time()
    action = agent.get_action(state)
    latency = time.time() - start
    assert latency < 1.0, f"Decision latency {latency}s exceeds 1s limit"
    return action
```

---

## 4.5 Astra-sim 构建与验证（2026-01-27）

已检查 analytical 后端二进制存在且可运行（需要完整配置参数）。

- **二进制路径**：
  - `astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware`
  - `astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Aware`
- **验证结果**：直接调用二进制会提示缺少 `workload-configuration` 参数，说明入口可用但需要完整配置文件运行。

---

## 5. 目录结构

```
智能算力跨域调度研究/
├── CLAUDE.md
├── docs/
│   ├── 00_问题定义说明书.md
│   ├── 01_相关论文综述.md
│   └── 02_数学建模与形式化描述.md
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── astra_env.py          # Gym 环境封装
│   │   └── astra_adapter.py      # Astra-sim 配置生成器
│   ├── algorithms/
│   │   ├── hi_ppo/
│   │   │   ├── global_agent.py
│   │   │   ├── local_agent.py
│   │   │   └── coordinator.py
│   │   └── reward_functions.py
│   └── utils/
│       ├── workload_generator.py  # Chakra ET 生成
│       └── network_config.py      # 网络配置生成
├── configs/
│   ├── network_config.yml
│   ├── system_config.json
│   ├── remote_memory.json
│   └── workload/
│       └── trace.{rank}.et
├── astra-sim/                     # Astra-sim 源码（git submodule）
└── results/
```

---

## 6. 下一步行动

**建议执行的 Prompt**:

> "Phase 2 环境设计已确认。请帮我实现 `astra_adapter.py`：
> 1. 接收 placement list `[0, 0, 1, 1, 2, ...]`（每层的域分配）
> 2. 生成 Astra-sim 所需的 **Network Topology 文件**（3 个同构域，域间带宽动态波动）
> 3. 生成 **Workload Configuration**（Llama-70B 的层配置）
> 4. 包含网络波动模拟：$B_{ij}(t) = B_{ij}^{base} \cdot (1 + \epsilon_{ij}(t))$，$|\epsilon_{ij}(t)| \leq 0.3$"

---

*本文档定义了基于 Astra-sim 的仿真环境架构*
*核心假设：同构算力环境，聚焦网络动态性优化*
