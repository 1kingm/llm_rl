#!/usr/bin/env python3
"""Hi-PPO 训练脚本.

使用方法:
    # 使用默认配置（真实 Astra-sim，需已构建 binary）
    python experiments/train_hi_ppo.py

    # 指定配置文件
    python experiments/train_hi_ppo.py --config configs/algo/hi_ppo.yaml

    # 快速测试（使用 mock 模式）
    python experiments/train_hi_ppo.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.envs.astra_env import EnvConfig
from src.algorithms.hi_ppo import (
    PPOConfig,
    NetworkConfig,
    TrainerConfig,
    HiPPOTrainer,
    train_hi_ppo,
)
from src.algorithms.reward_functions import RewardConstraints, RewardWeights
from src.utils.gnn_encoder import GNNConfig


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_configs_from_yaml(yaml_config: dict) -> tuple:
    """从 YAML 配置创建各个配置对象."""
    reward_cfg = yaml_config.get("reward", {})
    default_weights = RewardWeights()
    reward_weights = RewardWeights(
        w_eff=reward_cfg.get("w_eff", default_weights.w_eff),
        w_util=reward_cfg.get("w_util", default_weights.w_util),
        w_cost=reward_cfg.get("w_cost", default_weights.w_cost),
    )
    constraint_cfg = yaml_config.get("reward_constraints", {})
    default_constraints = RewardConstraints()
    reward_constraints = RewardConstraints(
        min_active_domains=constraint_cfg.get("min_active_domains", default_constraints.min_active_domains),
        active_domain_penalty=constraint_cfg.get("active_domain_penalty", default_constraints.active_domain_penalty),
        balance_target=constraint_cfg.get("balance_target", default_constraints.balance_target),
        balance_penalty=constraint_cfg.get("balance_penalty", default_constraints.balance_penalty),
        bandwidth_threshold_gbps=constraint_cfg.get(
            "bandwidth_threshold_gbps", default_constraints.bandwidth_threshold_gbps
        ),
        bandwidth_penalty=constraint_cfg.get("bandwidth_penalty", default_constraints.bandwidth_penalty),
        constraint_weight=constraint_cfg.get("constraint_weight", default_constraints.constraint_weight),
    )

    # 环境配置
    env_cfg = yaml_config.get("env", {})
    env_config = EnvConfig(
        num_domains=env_cfg.get("num_domains", 3),
        num_layers=env_cfg.get("num_layers", 96),
        episode_length=env_cfg.get("episode_length", EnvConfig().episode_length),
        bandwidth_fluctuation=env_cfg.get("bandwidth_fluctuation", 0.3),
        latency_jitter=env_cfg.get("latency_jitter", 0.2),
        use_mock=env_cfg.get("use_mock", False),
        backend=env_cfg.get("backend", EnvConfig().backend),
        astra_bin=env_cfg.get("astra_bin", EnvConfig().astra_bin),
        ns3_bin=env_cfg.get("ns3_bin", EnvConfig().ns3_bin),
        ns3_build_dir=env_cfg.get("ns3_build_dir", EnvConfig().ns3_build_dir),
        ns3_network_config=env_cfg.get("ns3_network_config", EnvConfig().ns3_network_config),
        ns3_comm_group_config=env_cfg.get("ns3_comm_group_config", EnvConfig().ns3_comm_group_config),
        ns3_logical_topology_dims=env_cfg.get("ns3_logical_topology_dims"),
        reward_weights=reward_weights,
        reward_constraints=reward_constraints,
        seed=yaml_config.get("seed", 42),
    )

    # PPO 配置
    ppo_cfg = yaml_config.get("ppo", {})
    ppo_config = PPOConfig(
        lr=ppo_cfg.get("lr", 3e-4),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        clip_range_vf=ppo_cfg.get("clip_range_vf"),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        update_epochs=ppo_cfg.get("update_epochs", 10),
        minibatch_size=ppo_cfg.get("minibatch_size", 256),
        target_kl=ppo_cfg.get("target_kl", 0.02),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        normalize_advantages=ppo_cfg.get("normalize_adv", True),
    )

    # 网络配置
    model_cfg = yaml_config.get("model", {})
    network_config = NetworkConfig(
        hidden_sizes=model_cfg.get("policy_hidden_sizes", [256, 256]),
        activation=model_cfg.get("activation", "tanh"),
        layer_norm=model_cfg.get("layer_norm", True),
        dropout=model_cfg.get("dropout", 0.1),
        orthogonal_init=True,
    )

    # GNN 配置
    gnn_cfg = yaml_config.get("gnn", {})
    gnn_config = GNNConfig(
        model_type=gnn_cfg.get("type", "GraphSAGE"),
        num_layers=gnn_cfg.get("num_layers", 2),
        hidden_dim=gnn_cfg.get("hidden_dim", 128),
        out_dim=gnn_cfg.get("out_dim", 128),
        aggregation=gnn_cfg.get("aggregation", "mean"),
        dropout=gnn_cfg.get("dropout", 0.1),
    )

    # 训练配置
    train_cfg = yaml_config.get("training", {})
    trainer_config = TrainerConfig(
        total_timesteps=train_cfg.get("total_timesteps", 2_000_000),
        steps_per_rollout=ppo_cfg.get("num_steps", 2048),
        eval_interval=train_cfg.get("eval_interval", 50_000),
        save_interval=train_cfg.get("save_interval", 100_000),
        log_interval=train_cfg.get("log_interval", 1000),
        seed=yaml_config.get("seed", 42),
        verbose=1,
    )

    return env_config, ppo_config, network_config, gnn_config, trainer_config


def main():
    parser = argparse.ArgumentParser(description="Hi-PPO 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/algo/hi_ppo.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试模式（减少训练步数）",
    )
    parser.add_argument(
        "--domains",
        type=int,
        default=None,
        help="覆盖域数量",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="覆盖层数量",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="覆盖总训练步数",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="输出详细程度",
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        print(f"加载配置: {config_path}")
        yaml_config = load_config(str(config_path))
        env_config, ppo_config, network_config, gnn_config, trainer_config = create_configs_from_yaml(yaml_config)
    else:
        print(f"配置文件不存在: {config_path}，使用默认配置")
        env_config = EnvConfig()
        ppo_config = PPOConfig()
        network_config = NetworkConfig()
        gnn_config = GNNConfig()
        trainer_config = TrainerConfig()

    # 应用命令行覆盖
    if args.domains is not None:
        env_config.num_domains = args.domains
    if args.layers is not None:
        env_config.num_layers = args.layers
    if args.timesteps is not None:
        trainer_config.total_timesteps = args.timesteps
    trainer_config.verbose = args.verbose

    # 快速测试模式
    if args.quick:
        print("快速测试模式")
        trainer_config.total_timesteps = 5000
        trainer_config.eval_interval = 1000
        trainer_config.save_interval = 2500
        trainer_config.log_interval = 500
        env_config.num_layers = 12  # 减少层数加速
        env_config.use_mock = True

    # 打印配置摘要
    print("\n" + "=" * 50)
    print("训练配置摘要")
    print("=" * 50)
    print(f"环境: {env_config.num_domains}域, {env_config.num_layers}层")
    print(f"总步数: {trainer_config.total_timesteps:,}")
    print(f"Rollout 步数: {trainer_config.steps_per_rollout}")
    print(f"PPO epochs: {ppo_config.update_epochs}")
    print(f"学习率: {ppo_config.lr}")
    print(f"GNN: {gnn_config.model_type}, out_dim={gnn_config.out_dim}")
    print("=" * 50 + "\n")

    # 创建训练器并开始训练
    trainer = HiPPOTrainer(
        env_config=env_config,
        ppo_config=ppo_config,
        network_config=network_config,
        trainer_config=trainer_config,
        gnn_config=gnn_config,
    )

    stats = trainer.train()

    # 打印最终结果
    print("\n" + "=" * 50)
    print("训练完成")
    print("=" * 50)
    print(f"总步数: {stats.timesteps:,}")
    print(f"总 episodes: {stats.episodes}")
    print(f"总更新次数: {stats.updates}")
    print(f"训练时间: {stats.wall_time:.2f}s")
    if stats.episode_rewards:
        print(f"最终平均奖励 (last 100): {sum(stats.episode_rewards[-100:]) / min(100, len(stats.episode_rewards)):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
