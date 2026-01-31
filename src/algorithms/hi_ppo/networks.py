"""Neural network architectures for Hi-PPO policy and value functions.

策略网络: 输出动作分布（切分点的概率）
价值网络: 输出状态价值估计
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class NetworkConfig:
    """网络配置."""

    state_dim: int = 128  # 状态维度
    hidden_sizes: Optional[List[int]] = None  # 隐藏层大小
    activation: str = "tanh"  # 激活函数: tanh, relu, gelu
    layer_norm: bool = True  # 是否使用 LayerNorm
    dropout: float = 0.1  # Dropout 比例
    orthogonal_init: bool = True  # 是否使用正交初始化

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


def get_activation(name: str) -> nn.Module:
    """获取激活函数."""
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }
    return activations.get(name.lower(), nn.Tanh())


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """正交初始化."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """多层感知机基础模块."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: str = "tanh",
        layer_norm: bool = True,
        dropout: float = 0.1,
        output_activation: bool = False,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(get_activation(activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """策略网络：输出切分点的动作分布.

    对于 K 个域，需要选择 K-1 个切分点。
    使用自回归方式依次选择每个切分点。
    """

    def __init__(
        self,
        config: NetworkConfig,
        num_layers: int,
        num_domains: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.num_domains = num_domains
        self.num_cuts = num_domains - 1  # K-1 个切分点

        # 共享特征提取器
        self.feature_extractor = MLP(
            input_dim=config.state_dim,
            hidden_sizes=config.hidden_sizes[:-1] if len(config.hidden_sizes) > 1 else config.hidden_sizes,
            output_dim=config.hidden_sizes[-1],
            activation=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
            output_activation=True,
        )

        # 切分点选择头（每个切分点一个）
        # 输入: 特征 + 已选切分点的 embedding
        self.cut_embedding_dim = 32
        self.cut_embedding = nn.Embedding(num_layers + 1, self.cut_embedding_dim)

        # 切分点预测头
        head_input_dim = config.hidden_sizes[-1] + self.cut_embedding_dim * self.num_cuts
        self.cut_heads = nn.ModuleList([
            nn.Linear(head_input_dim, num_layers) for _ in range(self.num_cuts)
        ])

        # 初始化
        if config.orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
            for head in self.cut_heads:
                orthogonal_init(head, gain=0.01)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            state: 状态张量 [batch_size, state_dim]
            deterministic: 是否确定性选择（取 argmax）

        Returns:
            actions: 切分点动作 [batch_size, num_cuts]
            log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
        """
        batch_size = state.shape[0]
        device = state.device

        if self.num_cuts == 0:
            empty_actions = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            zeros = torch.zeros(batch_size, device=device)
            return empty_actions, zeros, zeros

        # 提取特征
        features = self.feature_extractor(state)  # [batch, hidden]

        # 初始化已选切分点 embedding（用 0 表示未选）
        selected_cuts = torch.zeros(batch_size, self.num_cuts, dtype=torch.long, device=device)
        cut_embeds = self.cut_embedding(selected_cuts)  # [batch, num_cuts, embed_dim]

        actions = []
        log_probs = []
        entropies = []

        prev_cut = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(self.num_cuts):
            # 拼接特征和已选切分点 embedding
            cut_embeds_flat = cut_embeds.view(batch_size, -1)  # [batch, num_cuts * embed_dim]
            head_input = torch.cat([features, cut_embeds_flat], dim=-1)

            # 计算 logits
            logits = self.cut_heads[i](head_input)  # [batch, num_layers]

            # 应用约束 mask：切分点必须递增
            idx = torch.arange(self.num_layers, device=device).unsqueeze(0)  # [1, num_layers]
            max_cut = self.num_layers - (self.num_cuts - i)  # 保留后续切分点的空间
            mask = (idx <= prev_cut.unsqueeze(1)) | (idx > max_cut)  # [batch, num_layers]
            logits = logits.masked_fill(mask, float('-inf'))

            # 采样或确定性选择
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            # 更新已选切分点
            updated_cuts = selected_cuts.clone()
            updated_cuts[:, i] = action
            selected_cuts = updated_cuts
            cut_embeds = self.cut_embedding(selected_cuts)
            prev_cut = action

            actions.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        actions = torch.stack(actions, dim=-1)  # [batch, num_cuts]
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)  # [batch]
        entropy = torch.stack(entropies, dim=-1).mean(dim=-1)  # [batch]

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率和熵.

        Args:
            state: 状态张量 [batch_size, state_dim]
            actions: 动作张量 [batch_size, num_cuts]

        Returns:
            log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
        """
        batch_size = state.shape[0]
        device = state.device

        if self.num_cuts == 0:
            zeros = torch.zeros(batch_size, device=device)
            return zeros, zeros

        features = self.feature_extractor(state)

        selected_cuts = torch.zeros(batch_size, self.num_cuts, dtype=torch.long, device=device)
        cut_embeds = self.cut_embedding(selected_cuts)

        log_probs = []
        entropies = []
        prev_cut = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(self.num_cuts):
            cut_embeds_flat = cut_embeds.view(batch_size, -1)
            head_input = torch.cat([features, cut_embeds_flat], dim=-1)
            logits = self.cut_heads[i](head_input)

            # 约束 mask
            idx = torch.arange(self.num_layers, device=device).unsqueeze(0)
            max_cut = self.num_layers - (self.num_cuts - i)
            mask = (idx <= prev_cut.unsqueeze(1)) | (idx > max_cut)
            logits = logits.masked_fill(mask, float('-inf'))

            dist = Categorical(logits=logits)
            action = actions[:, i]

            updated_cuts = selected_cuts.clone()
            updated_cuts[:, i] = action
            selected_cuts = updated_cuts
            cut_embeds = self.cut_embedding(selected_cuts)
            prev_cut = action

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).mean(dim=-1)

        return log_probs, entropy


class ValueNetwork(nn.Module):
    """价值网络：估计状态价值 V(s)."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        self.net = MLP(
            input_dim=config.state_dim,
            hidden_sizes=config.hidden_sizes,
            output_dim=1,
            activation=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
            output_activation=False,
        )

        if config.orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
            # 价值网络输出层使用较小的初始化
            orthogonal_init(self.net.net[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            value: 状态价值 [batch_size]
        """
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor-Critic 组合模块."""

    def __init__(
        self,
        config: NetworkConfig,
        num_layers: int,
        num_domains: int,
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(config, num_layers, num_domains)
        self.value = ValueNetwork(config)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播.

        Returns:
            actions, log_probs, entropy, values
        """
        actions, log_probs, entropy = self.policy(state, deterministic)
        values = self.value(state)
        return actions, log_probs, entropy, values

    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作.

        Returns:
            log_probs, entropy, values
        """
        log_probs, entropy = self.policy.evaluate_actions(state, actions)
        values = self.value(state)
        return log_probs, entropy, values
