"""PPO Agent implementation with GAE advantage estimation.

实现完整的 PPO 算法，包括：
- Clipped surrogate objective
- Value function clipping
- Entropy bonus
- Early stopping (KL divergence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorCritic, NetworkConfig
from .buffer import BufferConfig, RolloutBuffer


@dataclass
class PPOConfig:
    """PPO 算法配置."""

    # 学习率
    lr: float = 3e-4
    lr_schedule: str = "linear"  # linear, constant

    # PPO 超参数
    clip_range: float = 0.2  # 策略裁剪范围
    clip_range_vf: Optional[float] = None  # 价值函数裁剪范围（None 表示不裁剪）
    entropy_coef: float = 0.01  # 熵系数
    value_coef: float = 0.5  # 价值损失系数
    max_grad_norm: float = 0.5  # 梯度裁剪

    # 训练参数
    update_epochs: int = 10  # 每次更新的 epoch 数
    minibatch_size: int = 256  # 小批量大小
    target_kl: Optional[float] = 0.02  # KL 散度阈值（用于早停）

    # GAE 参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


class PPOAgent:
    """PPO 智能体.

    实现 Proximal Policy Optimization 算法。
    """

    def __init__(
        self,
        config: PPOConfig,
        network_config: NetworkConfig,
        num_layers: int,
        num_domains: int,
        device: torch.device = None,
    ) -> None:
        self.config = config
        self.network_config = network_config
        self.num_layers = num_layers
        self.num_domains = num_domains
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建 Actor-Critic 网络
        self.actor_critic = ActorCritic(
            config=network_config,
            num_layers=num_layers,
            num_domains=num_domains,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.lr,
            eps=1e-5,
        )

        # 学习率调度器
        self.lr_scheduler = None
        self._total_timesteps = 0
        self._current_progress = 0.0

        # 创建缓冲区
        buffer_config = BufferConfig(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_advantages=config.normalize_advantages,
        )
        self.buffer = RolloutBuffer(
            config=buffer_config,
            state_dim=network_config.state_dim,
            action_dim=num_domains - 1,  # K-1 个切分点
            device=self.device,
        )

        # 训练统计
        self.train_stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "clip_fraction": [],
        }

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """选择动作.

        Args:
            state: 状态 [state_dim]
            deterministic: 是否确定性选择

        Returns:
            action: 切分点动作 [num_cuts]
            log_prob: 动作对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions, log_probs, _, values = self.actor_critic(state_tensor, deterministic)

        return (
            actions.cpu().numpy()[0],
            log_probs.cpu().item(),
            values.cpu().item(),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """存储转移."""
        self.buffer.add(state, action, reward, done, value, log_prob)

    def compute_returns(self, last_state: np.ndarray, last_done: bool) -> None:
        """计算 GAE 和 returns.

        Args:
            last_state: 最后状态
            last_done: 是否终止
        """
        with torch.no_grad():
            state_tensor = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.actor_critic.value(state_tensor).cpu().item()

        self.buffer.compute_gae(last_value, last_done)

    def update(self) -> Dict[str, float]:
        """执行 PPO 更新.

        Returns:
            训练统计信息
        """
        config = self.config

        # 重置统计
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []

        for epoch in range(config.update_epochs):
            for batch in self.buffer.get_batches(config.minibatch_size, shuffle=True):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # 评估当前策略
                log_probs, entropy, values = self.actor_critic.evaluate_actions(states, actions)

                # 计算比率
                ratio = torch.exp(log_probs - old_log_probs)

                # 策略损失（Clipped Surrogate Objective）
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # 价值损失
                if config.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -config.clip_range_vf, config.clip_range_vf
                    )
                    value_loss_1 = (values - returns) ** 2
                    value_loss_2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    + config.entropy_coef * entropy_loss
                )

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), config.max_grad_norm)
                self.optimizer.step()

                # 统计
                with torch.no_grad():
                    # KL 散度近似
                    log_ratio = log_probs - old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                    # Clip fraction
                    clip_fraction = ((ratio - 1).abs() > config.clip_range).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                kl_divs.append(approx_kl)
                clip_fractions.append(clip_fraction)

            # 早停检查
            if config.target_kl is not None:
                mean_kl = np.mean(kl_divs[-len(list(self.buffer.get_batches(config.minibatch_size))):])
                if mean_kl > config.target_kl:
                    break

        # 更新学习率
        self._update_learning_rate()

        # 重置缓冲区
        self.buffer.reset()

        # 返回统计
        stats = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divs),
            "clip_fraction": np.mean(clip_fractions),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        # 记录历史
        for key in ["policy_loss", "value_loss", "entropy", "kl_divergence", "clip_fraction"]:
            self.train_stats[key].append(stats[key])

        return stats

    def _update_learning_rate(self) -> None:
        """更新学习率（线性衰减）."""
        if self.config.lr_schedule == "linear" and self._total_timesteps > 0:
            progress = self._current_progress
            lr = self.config.lr * (1.0 - progress)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def set_training_progress(self, current_timestep: int, total_timesteps: int) -> None:
        """设置训练进度（用于学习率调度）."""
        self._total_timesteps = total_timesteps
        self._current_progress = current_timestep / total_timesteps

    def save(self, path: str) -> None:
        """保存模型."""
        torch.save({
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "network_config": self.network_config,
            "num_layers": self.num_layers,
            "num_domains": self.num_domains,
            "train_stats": self.train_stats,
        }, path)

    def load(self, path: str) -> None:
        """加载模型."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_stats = checkpoint.get("train_stats", self.train_stats)

    def get_cut_points(self, state: np.ndarray) -> List[int]:
        """获取切分点（用于与现有接口兼容）.

        Args:
            state: 状态

        Returns:
            切分点列表
        """
        action, _, _ = self.select_action(state, deterministic=True)
        return action.tolist()
