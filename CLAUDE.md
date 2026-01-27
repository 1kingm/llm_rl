# Project Master Context: Task 3 - Intelligent Cross-Domain Scheduling

> **SYSTEM CRITICAL INSTRUCTION:**
> This file is the **absolute source of truth** for the project.
> The AI MUST strictly adhere to the scope of **"Task 3"** defined in the source document `面向AI大模型分布式训练的智能算力跨域调度方法研究@v0.2.pdf`.
> **Any context outside Task 3 is OUT OF SCOPE.**

> **核心假设**: 本项目假设**同构算力环境**（各域 GPU 型号相同），核心挑战聚焦于**跨域网络动态性**和**通信优化**。

---

## 1. 研究目标 (Research Objective)

* **核心课题**: 课题 3 - 面向大模型训练的智能跨域调度算法设计
* **学术定位**: 针对 **CCF-A** 类会议 (SIGCOMM, OSDI, NeurIPS) 的系统/算法级论文
* **核心问题**: 解决多云战略下，大模型分布式训练在动态广域网环境中的资源分配难题

---

## 2. 核心技术架构: Hi-PPO (Hierarchical PPO)

根据文档要求，必须采用 **分层调度 (Hierarchical Scheduling)** 结合 **强化学习 (RL)**。

### 2.1 优化目标函数 (Objective Function)

构建多目标优化模型，最大化奖励 $R$：

$$R(\pi) = w_1 \cdot R_{eff} + w_2 \cdot U_{global} - w_3 \cdot C_{comm}$$

其中 $R_{eff} = \frac{T_{step}^{baseline} - T_{step}}{T_{step}^{baseline}}$ 表示训练效率改进率。

* **训练效率最大化**: 缩短模型训练周期 ($T_{step}$)
* **算力利用率最大化**: 提升跨域资源的整体利用率 ($U_{global}$)
* **跨域通信成本最小化**: 降低带宽占用计费 ($C_{comm}$)

### 2.2 分层架构 (Layered Architecture)

**1. 上层：域间调度 (Inter-Domain Agent)**
* **决策粒度**: 粗粒度
* **动作**: 决定任务在不同数据中心（域）间的分配比例
* **更新频率**: 低频（每 $N$ 个 Iterations 或网络状态剧烈变化时）

**2. 下层：域内调度 (Intra-Domain Agent)**
* **决策粒度**: 细粒度
* **动作**: 决定具体计算节点（GPU）的任务绑定、并行策略
* **更新频率**: 高频（实时响应）

---

## 3. 严格约束与 KPI (Hard Constraints & Metrics)

所有算法设计与实验验证必须满足以下 KPI，任何低于此标准的方案均视为失败：

| 维度 | 指标名称 (Metric) | 目标值 (Target) |
| --- | --- | --- |
| **核心算法** | 算法框架 | **PPO (Proximal Policy Optimization)** |
| **求解质量** | 帕累托最优解覆盖率 (Pareto Coverage) | **≥ 85%** |
| **实时性** | 域间决策延迟 (Inter-Domain Latency) | **< 1 秒** (在 10 个域场景下) |
| **实时性** | 域内决策延迟 (Intra-Domain Latency) | **< 100 ms** |
| **性能提升** | 训练周期缩短 (Training Time Reduction) | **≥ 25%** (vs 单域调度) |
| **资源效率** | 跨域算力利用率提升 (Util Improvement) | **≥ 35%** |
| **成本控制** | 跨域通信成本降低 (Cost Reduction) | **≥ 30%** |
| **协同效率** | 分层协同响应时间 | **≤ 2 秒** |
| **分配精度** | 域间算力分配误差 | **< 5%** |
| **稳定性** | 负载波动±20%时稳定性 | **≥ 99%** |
| **鲁棒性** | 网络波动±30%时性能下降 | **< 10%** |

---

## 4. 关键输入变量 (Key Inputs - Environment State)

虽然只做课题三，但我们需要假设以下输入（作为 Environment State）：

1. **算力状态**: 各域的 GPU 利用率、显存占用、任务队列长度（同构环境）
2. **网络状态**: 跨域链路的**实时带宽** $B_{ij}(t)$ 与**延迟** $L_{ij}(t)$（核心动态输入）
3. **任务特征**: 大模型训练的计算量与通信量需求

**网络动态性模型**:
$$B_{ij}(t) = B_{ij}^{base} \cdot (1 + \epsilon_{ij}(t)), \quad |\epsilon_{ij}(t)| \leq 0.3$$

---

## 5. 执行指令 (Directives for AI)

1. **LaTeX Required**: 涉及目标函数、约束条件、状态空间定义的描述，必须使用 LaTeX 公式
2. **Python Simulation**: 代码实现首选 Python（`SimPy` 用于快速原型验证，`Astra-sim` 用于高保真评估；`PyTorch` / `Ray` 用于算法逻辑）
3. **Strict Scope**: 不要生成课题一（感知）、课题二（预测）或课题四（容错）的代码，除非它们直接作为课题三的"黑盒环境参数"存在
4. **同构假设**: 不考虑 GPU 型号差异，聚焦网络动态性优化
5. **SOTA 对比**: 任何算法设计必须说明与 MAST/Sailor 的差异化

---

## 6. 关键参考论文 (Must-Read Papers)

### 直接竞争者（必须超越）
1. **MAST** (OSDI 2024) - 跨域调度
2. **Sailor** (SOSP 2025) - 动态适应
3. **CrossPipe** - 跨域流水线

### 方法论参考
4. **Crux** (SIGCOMM 2024) - 通信调度
5. **Hyperion** - 分层调度
6. **DD-PPO** - 分布式 PPO

### 工业基准
7. **DiLoCo** - Google 低通信训练
8. **DeepSpeed-ZeRO** - Microsoft 内存优化
9. **Megatron-LM** - NVIDIA 模型并行

---

## 7. 资源链接

- **企业方案**: `面向AI大模型分布式训练的智能算力跨域调度方法研究@v0.2.pdf`
- **问题定义**: `docs/00_问题定义说明书.md`
- **论文综述**: `docs/01_相关论文综述.md`
- **科研记忆**: `Project_Master_Context.md`

---

## 8. 已实现功能记录 (Implementation Log)

### 8.1 代码与模块
- **环境框架**: `src/envs/astra_env.py` 实现 Gym-like 环境，支持 mock 或真实 Astra-sim（默认真实运行）；已对齐 Astra-sim 2.0 配置格式（network.yml / system.json / workload .et），支持 remote memory 配置
- **网络动态模型**: `src/utils/network_dynamics.py` 提供带宽/延迟动态波动采样
- **奖励计算**: `src/algorithms/reward_functions.py` 提供多目标奖励计算与基线
- **Hi-PPO 骨架**: `src/algorithms/hi_ppo/*`（global/local agent、coordinator、rollout）
- **Astra 适配器**: `src/utils/astra_adapter.py` 生成 Astra-sim 2.0 规范配置，并解析 stdout 输出指标
- **Chakra ET 生成**: `src/utils/workload_generator.py` 提供 ET 生成骨架

### 8.2 实验脚本与输出
- **最小联通验证**: `experiments/run_mock_episode.py`、`experiments/run_minimal_rollout.py`
- **真实 Astra-sim 对比测试**: `experiments/run_astra_end_to_end.py`（congestion-unaware vs congestion-aware）
- **输出示例**: `results/astra_unaware.csv`、`results/astra_aware.csv`、`results/astra_unaware.log`、`results/astra_aware.log`

### 8.3 Astra-sim 集成状态
- **源码与构建**: 已克隆 `astra-sim/` 并完成 analytical backend 构建
- **二进制路径**:
  - `astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware`
  - `astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Aware`
- **构建验证**: 2026-01-27 已确认二进制存在并可执行（无配置参数时提示缺少 `workload-configuration`）
- **ET 生成兼容性**: 当前 Python protobuf 运行时为 5.x，Chakra `et_def_pb2.py` 为 protoc 6.x 生成，存在版本不兼容；适配器默认阻断并提示升级（如需临时跑通，可显式开启 `allow_fallback` 使用 Astra-sim 示例 ET）

### 8.4 精度修正 (Accuracy Fixes)
- **通信体积一致性**: `run_astra` 支持传入估算 comm_volume_gb（基于 placement + comm_size_bytes），避免 comm_cycles 混写到 comm_volume_gb
- **负载观测一致性**: `AstraSimEnv` 观测中的 domain_loads 改为基于 placement 的真实占比，替代随机采样
- **解释/日志一致性**: `env.step()` 记录实际使用的 network_state，解释模块优先使用该状态
- **ET 兼容性**: `generate_workload_et` 为 COMP_NODE 补齐 `num_ops` 与 `tensor_size`，满足 Chakra v3 feeder 需求

### 8.5 运行验证 (Runtime Verification)
- **2026-01-27 真实 E2E**: `experiments/run_astra_end_to_end.py` 已跑通（aware/unaware 均成功输出 metrics）

---

*本文件是项目的"宪法"，所有设计决策必须符合此处定义的标准*
*核心假设：同构算力环境，聚焦网络动态性优化*
*唯一聚焦：课题三 - 智能跨域调度算法设计*
