"""Explainability module for Hi-PPO scheduling decisions.

接入方式:
=========
1. 在 rollout.py 中使用:
   ```python
   from ...utils.explainability import build_explanation

   def run_rollout(env, coordinator, ...):
       for step in range(steps_per_episode):
           ...
           next_obs, reward, terminated, truncated, info = env.step(action)

           # 生成解释
           explanation = build_explanation(
               placement=action,
               reward=info.get("reward_breakdown"),
               network_state=info.get("network_state", env.current_network_state),
           )
           info["explanation"] = explanation
           row["explanation_summary"] = explanation.get("summary", "")
   ```

2. 在日志中记录:
   ```python
   if explanation:
       row["dominant_factor"] = explanation["reward_breakdown"].get("dominant_factor", "")
       row["cross_cuts"] = explanation["placement"]["cross_domain_cuts"]
   ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import NetworkState, RewardBreakdown


# 阈值配置
LOW_BANDWIDTH_THRESHOLD_GBPS: float = 3.0
HIGH_LATENCY_THRESHOLD_MS: float = 50.0


@dataclass(frozen=True)
class ExplanationItem:
    """单条解释项."""

    key: str
    value: str


def summarize_reward(
    reward: RewardBreakdown,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """分解奖励并计算各分量贡献.

    Args:
        reward: 奖励分解
        weights: 奖励权重 {w_eff, w_util, w_cost}

    Returns:
        包含奖励分量和贡献占比的字典
    """
    weights = weights or {"w_eff": 0.4, "w_util": 0.3, "w_cost": 0.3}

    # 计算加权贡献
    eff_contrib = weights["w_eff"] * reward.r_eff
    util_contrib = weights["w_util"] * reward.r_util
    cost_contrib = weights["w_cost"] * reward.r_cost

    total_abs = abs(eff_contrib) + abs(util_contrib) + abs(cost_contrib)

    if total_abs > 0:
        eff_pct = abs(eff_contrib) / total_abs * 100
        util_pct = abs(util_contrib) / total_abs * 100
        cost_pct = abs(cost_contrib) / total_abs * 100
    else:
        eff_pct = util_pct = cost_pct = 33.3

    # 确定主导因素
    contributions = {
        "efficiency": abs(eff_contrib),
        "utilization": abs(util_contrib),
        "cost": abs(cost_contrib),
    }
    dominant = max(contributions, key=lambda k: contributions[k])

    return {
        "r_eff": reward.r_eff,
        "r_util": reward.r_util,
        "r_cost": reward.r_cost,
        "reward": reward.reward,
        "eff_contribution_pct": eff_pct,
        "util_contribution_pct": util_pct,
        "cost_contribution_pct": cost_pct,
        "dominant_factor": dominant,
    }


def rule_summary(
    placement: List[int],
    network_state: Optional[NetworkState] = None,
    bandwidth_threshold_gbps: float = LOW_BANDWIDTH_THRESHOLD_GBPS,
    latency_threshold_ms: float = HIGH_LATENCY_THRESHOLD_MS,
    avg_inter_bandwidth_gbps: Optional[float] = None,
    avg_inter_latency_ms: Optional[float] = None,
) -> List[ExplanationItem]:
    """生成规则摘要：检测触发调度决策的关键因素.

    Args:
        placement: 放置动作
        network_state: 网络状态（可选，用于详细告警）
        bandwidth_threshold_gbps: 带宽告警阈值
        latency_threshold_ms: 延迟告警阈值
        avg_inter_bandwidth_gbps: 平均域间带宽（兼容旧接口）
        avg_inter_latency_ms: 平均域间延迟（兼容旧接口）

    Returns:
        解释项列表
    """
    items: List[ExplanationItem] = []

    # 跨域切分次数
    cross_edges = sum(1 for i in range(len(placement) - 1) if placement[i] != placement[i + 1])
    items.append(ExplanationItem("cross_domain_cuts", str(cross_edges)))

    # 从 network_state 提取详细告警
    if network_state is not None:
        num_domains = len(network_state.bandwidth_gbps)
        bw_alerts = []
        lat_alerts = []

        for i in range(num_domains):
            for j in range(num_domains):
                if i == j:
                    continue
                bw = network_state.bandwidth_gbps[i][j]
                lat = network_state.latency_ms[i][j]

                if bw < bandwidth_threshold_gbps:
                    bw_alerts.append(f"域{i}→域{j}: {bw:.2f}Gbps")
                if lat > latency_threshold_ms:
                    lat_alerts.append(f"域{i}→域{j}: {lat:.1f}ms")

        if bw_alerts:
            items.append(ExplanationItem("bandwidth_alerts", "; ".join(bw_alerts[:3])))
        if lat_alerts:
            items.append(ExplanationItem("latency_alerts", "; ".join(lat_alerts[:3])))

    # 兼容旧接口
    elif avg_inter_bandwidth_gbps is not None or avg_inter_latency_ms is not None:
        if avg_inter_bandwidth_gbps is not None and avg_inter_bandwidth_gbps < bandwidth_threshold_gbps:
            items.append(ExplanationItem(
                "bandwidth_alert",
                f"avg {avg_inter_bandwidth_gbps:.2f} Gbps < {bandwidth_threshold_gbps}"
            ))
        if avg_inter_latency_ms is not None and avg_inter_latency_ms > latency_threshold_ms:
            items.append(ExplanationItem(
                "latency_alert",
                f"avg {avg_inter_latency_ms:.2f} ms > {latency_threshold_ms}"
            ))

    return items


def analyze_placement(placement: List[int]) -> Dict[str, Any]:
    """分析放置策略.

    Args:
        placement: 放置动作（每层分配的域 ID）

    Returns:
        放置分析结果
    """
    if not placement:
        return {
            "num_domains": 0,
            "num_layers": 0,
            "layers_per_domain": [],
            "cross_domain_cuts": 0,
            "balance_score": 0.0,
        }

    num_layers = len(placement)
    num_domains = max(placement) + 1 if placement else 0

    # 统计每个域的层数
    layers_per_domain = [0] * num_domains
    for domain_id in placement:
        layers_per_domain[domain_id] += 1

    # 计算跨域切分次数
    cross_cuts = sum(1 for i in range(1, num_layers) if placement[i] != placement[i-1])

    # 计算负载均衡得分 (1 - 变异系数)
    if num_domains > 0 and num_layers > 0:
        mean_layers = num_layers / num_domains
        variance = sum((x - mean_layers) ** 2 for x in layers_per_domain) / num_domains
        std_dev = variance ** 0.5
        cv = std_dev / mean_layers if mean_layers > 0 else 0
        balance_score = max(0.0, 1.0 - cv)
    else:
        balance_score = 0.0

    return {
        "num_domains": num_domains,
        "num_layers": num_layers,
        "layers_per_domain": layers_per_domain,
        "cross_domain_cuts": cross_cuts,
        "balance_score": round(balance_score, 3),
    }


def generate_summary_text(
    reward_summary: Dict[str, Any],
    placement_analysis: Dict[str, Any],
    rules: List[Dict[str, str]],
) -> str:
    """生成人类可读的摘要文本.

    Args:
        reward_summary: 奖励分解结果
        placement_analysis: 放置分析结果
        rules: 规则摘要列表

    Returns:
        人类可读的摘要字符串
    """
    lines = []

    # 奖励摘要
    lines.append(f"[奖励] 总分: {reward_summary.get('reward', 0):.4f}")
    lines.append(f"  - 效率: {reward_summary.get('r_eff', 0):.4f} ({reward_summary.get('eff_contribution_pct', 0):.1f}%)")
    lines.append(f"  - 利用率: {reward_summary.get('r_util', 0):.4f} ({reward_summary.get('util_contribution_pct', 0):.1f}%)")
    lines.append(f"  - 成本: {reward_summary.get('r_cost', 0):.4f} ({reward_summary.get('cost_contribution_pct', 0):.1f}%)")
    lines.append(f"  - 主导因素: {reward_summary.get('dominant_factor', 'unknown')}")

    # 放置摘要
    lines.append(f"[放置] {placement_analysis.get('num_layers', 0)}层 → {placement_analysis.get('num_domains', 0)}域")
    lines.append(f"  - 各域层数: {placement_analysis.get('layers_per_domain', [])}")
    lines.append(f"  - 跨域切分: {placement_analysis.get('cross_domain_cuts', 0)}次")
    lines.append(f"  - 均衡得分: {placement_analysis.get('balance_score', 0):.2f}")

    # 告警摘要
    alerts = [r for r in rules if "alert" in r.get("key", "").lower()]
    if alerts:
        lines.append("[告警]")
        for alert in alerts[:3]:
            lines.append(f"  ⚠ {alert.get('key', '')}: {alert.get('value', '')}")

    return "\n".join(lines)


def build_explanation(
    placement: List[int],
    reward: Optional[RewardBreakdown] = None,
    network_state: Optional[NetworkState] = None,
    avg_inter_bandwidth_gbps: Optional[float] = None,
    avg_inter_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """组装完整的解释载荷.

    Args:
        placement: 放置动作
        reward: 奖励分解
        network_state: 网络状态（可选）
        avg_inter_bandwidth_gbps: 平均域间带宽（兼容旧接口）
        avg_inter_latency_ms: 平均域间延迟（兼容旧接口）

    Returns:
        完整的解释载荷字典
    """
    # 奖励分解
    if reward is not None:
        reward_summary = summarize_reward(reward)
    else:
        reward_summary = {
            "r_eff": 0.0, "r_util": 0.0, "r_cost": 0.0, "reward": 0.0,
            "dominant_factor": "unknown"
        }

    # 规则摘要
    rules = [item.__dict__ for item in rule_summary(
        placement,
        network_state=network_state,
        avg_inter_bandwidth_gbps=avg_inter_bandwidth_gbps,
        avg_inter_latency_ms=avg_inter_latency_ms,
    )]

    # 放置分析
    placement_analysis = analyze_placement(placement)

    # 生成摘要文本
    summary = generate_summary_text(reward_summary, placement_analysis, rules)

    return {
        "reward_breakdown": reward_summary,
        "rules": rules,
        "placement": placement_analysis,
        "summary": summary,
    }
