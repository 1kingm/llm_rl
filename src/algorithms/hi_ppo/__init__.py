"""Hi-PPO hierarchy package.

完整的分层 PPO 实现，包括：
- 策略网络和价值网络 (networks.py)
- 经验缓冲区 (buffer.py)
- PPO Agent (ppo_agent.py)
- 训练循环 (trainer.py)
- 协调器 (coordinator.py)
"""

from .global_agent import GlobalAgent, RandomGlobalAgent
from .local_agent import LocalAgent, NoOpLocalAgent
from .coordinator import CoordinatorConfig, HiPPOCoordinator
from .rollout import run_rollout
from .networks import NetworkConfig, PolicyNetwork, ValueNetwork, ActorCritic
from .buffer import BufferConfig, RolloutBuffer, MultiAgentBuffer
from .ppo_agent import PPOConfig, PPOAgent
from .trainer import TrainerConfig, TrainingStats, HiPPOTrainer, train_hi_ppo

__all__ = [
    # 基础接口
    "GlobalAgent",
    "RandomGlobalAgent",
    "LocalAgent",
    "NoOpLocalAgent",
    "CoordinatorConfig",
    "HiPPOCoordinator",
    "run_rollout",
    # 网络
    "NetworkConfig",
    "PolicyNetwork",
    "ValueNetwork",
    "ActorCritic",
    # 缓冲区
    "BufferConfig",
    "RolloutBuffer",
    "MultiAgentBuffer",
    # PPO Agent
    "PPOConfig",
    "PPOAgent",
    # 训练
    "TrainerConfig",
    "TrainingStats",
    "HiPPOTrainer",
    "train_hi_ppo",
]
