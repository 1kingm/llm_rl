# Phase 3: Hi-PPO 算法设计与实现规范

> **Project:** Intelligent Cross-Domain Scheduling for Large Model Training (Task 3)
> **Target:** CCF-A Class Conference (System/AI Track)
> **Version:** v1.0
> **Date:** 2026-01-26

**核心假设**: 同构算力环境（各域 GPU 型号相同），核心挑战聚焦于**跨域网络动态性**与**通信优化**。

---

## 1. 设计目标与约束

**目标函数（统一版本）**:
$$R = w_1 \cdot R_{eff} + w_2 \cdot R_{util} - w_3 \cdot R_{cost}$$

其中：
- $R_{eff} = \frac{T_{step}^{baseline} - T_{step}}{T_{step}^{baseline}}$
- $R_{util} = \frac{1}{K}\sum_k U_k$
- $R_{cost} = C_{comm}^{cross}$

**硬约束**:
- 域间决策延迟 < 1 秒
- 域内决策延迟 < 100 ms
- 网络波动 ±30% 时性能下降 < 10%
- 资源利用率提升 ≥ 35%（可选绝对值 > 85%）

---

## 2. 分层策略结构

### 2.1 上层策略（域间调度）
**输入**:
- $B(t)$, $L(t)$: 域间带宽/延迟矩阵
- $R_k(t)$: 域内 GPU/内存利用率
- $Q_k(t)$: 任务队列/负载
- $h_{topo}$: GNN 拓扑编码

**输出**:
- 切分点集合 $\{c_1, ..., c_{K-1}\}$ 或域分配比例 $\pi_{domain}$

**目标**: 最小化跨域通信成本并控制全局 $T_{step}$

### 2.2 下层策略（域内编排）
**输入**:
- 当前域内拓扑 $G_k$
- 显存与负载状态 $\vec{M}_k$, $Q_k$
- 上层切分结果约束

**输出**:
- 并行策略 $(dp, tp, pp)$
- 设备放置 placement

**目标**: 最大化域内利用率并降低域内通信开销

---

## 3. GNN 网络拓扑编码（核心模块）

### 3.1 图定义
- 节点: 域 $v_k$
- 边: 域间链路 $e_{ij}$
- 节点特征: $[U_k^{GPU}, U_k^{Mem}, Q_k]$
- 边特征: $[B_{ij}(t), L_{ij}(t), \gamma_{ij}]$

### 3.2 编码函数
$$h_{topo} = \text{GNN}(\mathcal{V}, \mathcal{E}, X_v, X_e)$$

**建议架构**: 2 层 GraphSAGE 或 GAT，输出全局 pooling 向量

---

## 4. 关键算法流程（伪代码）

```text
Algorithm Hi-PPO
Input: Environment state s_t
Output: Action a_t = {a_high, a_low}

1: h_topo <- GNN(B(t), L(t), R_k(t))
2: a_high <- pi_high(s_high, h_topo)
3: for each domain k:
4:     a_low_k <- pi_low_k(s_low_k | a_high)
5: Execute a_t in simulator / cluster
6: Observe T_step, U_k, C_comm
7: Compute reward R
8: Update PPO policies (high + low)
```

---

## 5. 约束实现

### 5.1 Action Masking
- 层切分导致 OOM 的动作直接 Mask
- 带宽不足的跨域切分动作加大惩罚

### 5.2 Lagrangian Penalty
$$L(\theta, \mu) = L^{CLIP}(\theta) - \mu \cdot g(s,a)$$

---

## 6. 训练流程设计

1. 使用 SimPy 进行小规模快速原型验证
2. 切换到 Astra-sim 进行高保真训练
3. 使用 offline trace + online roll-out 结合
4. 每 N steps 更新上层策略，每 step 更新下层策略

---

## 7. 模块接口（建议）

```
src/
  algorithms/
    hi_ppo/
      global_agent.py   # 上层策略
      local_agent.py    # 下层策略
      coordinator.py    # 协调器
  envs/
    astra_env.py        # 训练环境
  utils/
    gnn_encoder.py      # 拓扑编码
    explainability.py   # 解释性输出
```

---

## 8. 可解释性设计（补充）

为满足“调度算法可解释性”要求，输出可读的决策依据：

- **规则摘要**: 展示关键链路带宽/延迟阈值触发原因  
- **贡献分解**: 输出 $R_{eff}/R_{util}/R_{cost}$ 分量  
- **可视化**: 记录域间切分点与跨域流量热力图

---

## 9. 初版超参数配置（可执行）

已提供可执行 YAML 配置：`configs/algo/hi_ppo.yaml`。  
建议先用小规模仿真验证，再逐步放大域数量与模型规模。

```yaml
ppo:
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.02
  num_steps: 2048
  update_epochs: 10
  minibatch_size: 256

gnn:
  type: "GraphSAGE"
  num_layers: 2
  hidden_dim: 128
  out_dim: 128
  aggregation: "mean"
```

---

## 10. 与 SOTA 的差异化

- **MAST**: 启发式 + 静态带宽假设
- **Sailor**: 单层 RL + 动态资源
- **Hi-PPO**: 分层 RL + 动态网络感知 + 多目标优化

---

*本文件为 Phase 3 算法设计的统一规范* 
