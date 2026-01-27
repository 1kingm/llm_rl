"""Experience buffer for PPO training.

实现 Rollout Buffer，存储轨迹数据并计算 GAE 优势估计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BufferConfig:
    """缓冲区配置."""

    buffer_size: int = 2048  # 缓冲区大小（步数）
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE lambda
    normalize_advantages: bool = True  # 是否标准化优势


@dataclass
class RolloutBuffer:
    """Rollout 经验缓冲区.

    存储一个 rollout 周期的轨迹数据，支持 GAE 优势估计。
    """

    config: BufferConfig
    state_dim: int
    action_dim: int  # 切分点数量 (K-1)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # 存储数组
    states: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)

    # GAE 计算结果
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)

    # 指针
    ptr: int = field(init=False, default=0)
    full: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """重置缓冲区."""
        size = self.config.buffer_size
        self.states = np.zeros((size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((size, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """添加一条经验.

        Args:
            state: 状态 [state_dim]
            action: 动作（切分点）[action_dim]
            reward: 奖励
            done: 是否终止
            value: 价值估计
            log_prob: 动作对数概率
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1
        if self.ptr >= self.config.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """计算 GAE (Generalized Advantage Estimation).

        GAE 公式:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...

        Args:
            last_value: 最后状态的价值估计
            last_done: 最后状态是否终止
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        last_gae = 0.0
        size = self.ptr if not self.full else self.config.buffer_size

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(last_done)
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]

            # TD 误差
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # 计算 returns = advantages + values
        self.returns[:size] = self.advantages[:size] + self.values[:size]

        # 标准化优势
        if self.config.normalize_advantages:
            adv = self.advantages[:size]
            self.advantages[:size] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[dict, None, None]:
        """生成训练批次.

        Args:
            batch_size: 批次大小
            shuffle: 是否打乱

        Yields:
            包含训练数据的字典
        """
        size = self.ptr if not self.full else self.config.buffer_size
        indices = np.arange(size)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield {
                "states": torch.tensor(self.states[batch_indices], device=self.device),
                "actions": torch.tensor(self.actions[batch_indices], device=self.device),
                "old_log_probs": torch.tensor(self.log_probs[batch_indices], device=self.device),
                "advantages": torch.tensor(self.advantages[batch_indices], device=self.device),
                "returns": torch.tensor(self.returns[batch_indices], device=self.device),
                "old_values": torch.tensor(self.values[batch_indices], device=self.device),
            }

    def get_all(self) -> dict:
        """获取所有数据."""
        size = self.ptr if not self.full else self.config.buffer_size
        return {
            "states": torch.tensor(self.states[:size], device=self.device),
            "actions": torch.tensor(self.actions[:size], device=self.device),
            "old_log_probs": torch.tensor(self.log_probs[:size], device=self.device),
            "advantages": torch.tensor(self.advantages[:size], device=self.device),
            "returns": torch.tensor(self.returns[:size], device=self.device),
            "old_values": torch.tensor(self.values[:size], device=self.device),
        }

    @property
    def size(self) -> int:
        """当前存储的经验数量."""
        return self.config.buffer_size if self.full else self.ptr

    def is_full(self) -> bool:
        """缓冲区是否已满."""
        return self.full


@dataclass
class MultiAgentBuffer:
    """多智能体缓冲区（用于分层 PPO）.

    分别存储上层（域间）和下层（域内）的经验。
    """

    global_buffer: RolloutBuffer
    local_buffers: List[RolloutBuffer] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        config: BufferConfig,
        state_dim: int,
        global_action_dim: int,
        local_action_dim: int,
        num_domains: int,
        device: torch.device,
    ) -> "MultiAgentBuffer":
        """创建多智能体缓冲区.

        Args:
            config: 缓冲区配置
            state_dim: 状态维度
            global_action_dim: 全局动作维度（切分点数）
            local_action_dim: 本地动作维度
            num_domains: 域数量
            device: 设备

        Returns:
            MultiAgentBuffer 实例
        """
        global_buffer = RolloutBuffer(
            config=config,
            state_dim=state_dim,
            action_dim=global_action_dim,
            device=device,
        )

        local_buffers = [
            RolloutBuffer(
                config=config,
                state_dim=state_dim,
                action_dim=local_action_dim,
                device=device,
            )
            for _ in range(num_domains)
        ]

        return cls(global_buffer=global_buffer, local_buffers=local_buffers)

    def reset(self) -> None:
        """重置所有缓冲区."""
        self.global_buffer.reset()
        for buf in self.local_buffers:
            buf.reset()
