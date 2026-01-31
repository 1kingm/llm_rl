"""Training loop for Hi-PPO algorithm.

实现完整的训练循环，包括：
- 环境交互
- 经验收集
- PPO 更新
- 日志记录
- 模型保存
"""

from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from ...envs.astra_env import AstraSimEnv, EnvConfig
from ...utils.explainability import build_explanation
from ...utils.gnn_encoder import GNNEncoder, GNNConfig
from .ppo_agent import PPOAgent, PPOConfig
from .networks import NetworkConfig


@dataclass
class TrainerConfig:
    """训练器配置."""

    # 训练参数
    total_timesteps: int = 2_000_000  # 总训练步数
    steps_per_rollout: int = 2048  # 每次 rollout 的步数
    eval_interval: int = 50_000  # 评估间隔
    save_interval: int = 100_000  # 保存间隔
    log_interval: int = 1000  # 日志间隔
    live_log_path: str = "results/training/train_live.log"  # 实时日志路径

    # 路径
    log_dir: str = "results/training"
    model_dir: str = "results/models"

    # 可解释性
    enable_explainability: bool = True  # 是否启用可解释性记录
    explanation_log_interval: int = 100  # 解释记录间隔（步数）
    explainability_history_size: int = 10_000  # 可解释性历史保留长度（0=不限制）

    # GNN 训练
    train_gnn: bool = False  # 是否在 PPO 更新中联合训练 GNN 编码器

    # 其他
    seed: int = 42
    verbose: int = 1  # 0: 无输出, 1: 进度, 2: 详细


@dataclass
class TrainingStats:
    """训练统计."""

    timesteps: int = 0
    episodes: int = 0
    updates: int = 0
    total_reward: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    update_stats: List[Dict[str, float]] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    wall_time: float = 0.0

    # 可解释性统计
    explanations: List[Dict] = field(default_factory=list)
    dominant_factors: Dict[str, int] = field(default_factory=lambda: {"efficiency": 0, "utilization": 0, "cost": 0})
    avg_cross_cuts: float = 0.0
    avg_balance_score: float = 0.0


class HiPPOTrainer:
    """Hi-PPO 训练器.

    管理完整的训练流程。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        network_config: NetworkConfig,
        trainer_config: TrainerConfig,
        gnn_config: Optional[GNNConfig] = None,
    ) -> None:
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.network_config = network_config
        self.trainer_config = trainer_config
        self.gnn_config = gnn_config or GNNConfig()

        # 设置随机种子
        self._set_seed(trainer_config.seed)

        # 设备：优先 CUDA，其次 MPS（macOS），否则 CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 创建环境
        self.env = AstraSimEnv(env_config)

        # 计算状态维度
        base_state_dim = self.env._state_dim()
        gnn_out_dim = self.gnn_config.out_dim
        total_state_dim = base_state_dim + gnn_out_dim

        # 更新网络配置
        self.network_config.state_dim = total_state_dim

        # 创建 GNN 编码器
        self.gnn_encoder = GNNEncoder(self.gnn_config).to(str(self.device))
        train_gnn = bool(trainer_config.train_gnn and self.gnn_encoder.has_model())
        if trainer_config.train_gnn and not train_gnn and trainer_config.verbose >= 1:
            print("GNN 训练已请求，但当前环境不可用（缺少 PyG 或未启用 GNN）。", flush=True)

        # 创建 PPO Agent
        self.agent = PPOAgent(
            config=ppo_config,
            network_config=self.network_config,
            num_layers=env_config.num_layers,
            num_domains=env_config.num_domains,
            device=self.device,
            buffer_size=trainer_config.steps_per_rollout,
            gnn_encoder=self.gnn_encoder,
            gnn_out_dim=self.gnn_config.out_dim,
            train_gnn=train_gnn,
        )

        # 训练统计
        self.stats = TrainingStats()
        self._live_log_file: Optional[object] = None

        # 可解释性追踪
        self._cross_cuts_history: List[int] = []
        self._balance_scores_history: List[float] = []

        # 创建目录
        Path(trainer_config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(trainer_config.model_dir).mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        """设置随机种子."""
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _augment_state(self, obs: List[float], network_state) -> np.ndarray:
        """增强状态（添加 GNN 编码）.

        Args:
            obs: 原始观测
            network_state: 网络状态

        Returns:
            增强后的状态
        """
        base_state = np.array(obs, dtype=np.float32)

        if network_state is not None:
            h_topo = self.gnn_encoder.encode_from_network_state(network_state)
            h_topo = np.array(h_topo, dtype=np.float32)
            return np.concatenate([base_state, h_topo])
        else:
            # 填充零向量
            h_topo = np.zeros(self.gnn_config.out_dim, dtype=np.float32)
            return np.concatenate([base_state, h_topo])

    def train(self, callback: Optional[Callable[[TrainingStats], bool]] = None) -> TrainingStats:
        """执行训练.

        Args:
            callback: 回调函数，返回 False 时停止训练

        Returns:
            训练统计
        """
        config = self.trainer_config
        start_time = time.time()

        # 初始化
        obs, _ = self.env.reset(seed=config.seed)
        state = self._augment_state(obs, self.env.current_network_state)

        episode_reward = 0.0
        episode_length = 0

        if config.verbose >= 1:
            print(f"开始训练: 总步数={config.total_timesteps}, 设备={self.device}", flush=True)
            print(f"环境: {self.env_config.num_domains}域, {self.env_config.num_layers}层", flush=True)

        pbar = None
        with contextlib.ExitStack() as stack:
            if config.verbose >= 1 and config.live_log_path:
                live_log_path = Path(config.live_log_path)
                live_log_path.parent.mkdir(parents=True, exist_ok=True)
                self._live_log_file = stack.enter_context(
                    open(live_log_path, "a", encoding="utf-8", buffering=1)
                )
                self._write_live_log(
                    f"[start] ts={datetime.now().isoformat(timespec='seconds')} "
                    f"timesteps={config.total_timesteps}, device={self.device}, "
                    f"domains={self.env_config.num_domains}, layers={self.env_config.num_layers}"
                )

            if config.verbose >= 1 and tqdm is not None:
                pbar = tqdm(total=config.total_timesteps, desc="Hi-PPO Training", unit="step")
                if self.stats.timesteps:
                    pbar.update(self.stats.timesteps)
                stack.callback(pbar.close)

            try:
                while self.stats.timesteps < config.total_timesteps:
                    # 收集 rollout
                    for _ in range(config.steps_per_rollout):
                        step_id = self.stats.timesteps + 1
                        if config.verbose >= 2 and config.log_interval > 0 and step_id % config.log_interval == 0:
                            print(f"[Step {step_id}] started", flush=True)
                            self._write_live_log(f"[Step {step_id}] started")
                        # 选择动作
                        action, log_prob, value = self.agent.select_action(state)

                        # 将切分点转换为放置向量
                        placement = self._cut_points_to_placement(action.tolist())

                        # 执行动作
                        next_obs, reward, terminated, truncated, info = self.env.step(placement)
                        done = terminated or truncated

                        # 生成可解释性记录
                        if config.enable_explainability:
                            self._record_explanation(
                                placement=placement,
                                reward_breakdown=info.get("reward_breakdown"),
                                network_state=info.get("network_state", self.env.current_network_state),
                            )

                        # 存储转移
                        self.agent.store_transition(
                            state,
                            action,
                            reward,
                            done,
                            value,
                            log_prob,
                            network_state=info.get("network_state", self.env.current_network_state),
                            domain_loads=info.get("domain_loads"),
                        )

                        # 更新状态
                        state = self._augment_state(next_obs, self.env.current_network_state)
                        episode_reward += reward
                        episode_length += 1
                        self.stats.timesteps += 1
                        if pbar is not None:
                            pbar.update(1)

                        # Episode 结束
                        if done:
                            self.stats.episodes += 1
                            self.stats.episode_rewards.append(episode_reward)
                            self.stats.episode_lengths.append(episode_length)
                            self.stats.total_reward += episode_reward

                            if config.verbose >= 2:
                                print(
                                    f"Episode {self.stats.episodes}: reward={episode_reward:.4f}, length={episode_length}",
                                    flush=True,
                                )

                            # 重置
                            obs, _ = self.env.reset()
                            state = self._augment_state(obs, self.env.current_network_state)
                            episode_reward = 0.0
                            episode_length = 0

                        # 检查是否达到总步数
                        if self.stats.timesteps >= config.total_timesteps:
                            break

                    # 计算 GAE
                    self.agent.compute_returns(state, done)

                    # PPO 更新
                    self.agent.set_training_progress(self.stats.timesteps, config.total_timesteps)
                    update_stats = self.agent.update()
                    self.stats.updates += 1
                    self.stats.update_stats.append(update_stats)

                    # 日志
                    if self.stats.timesteps % config.log_interval < config.steps_per_rollout:
                        self._log_progress(update_stats)

                    # 评估
                    if self.stats.timesteps % config.eval_interval < config.steps_per_rollout:
                        eval_reward = self._evaluate()
                        self.stats.eval_rewards.append(eval_reward)
                        if config.verbose >= 1:
                            print(f"[Eval] timesteps={self.stats.timesteps}, reward={eval_reward:.4f}", flush=True)
                            self._write_live_log(
                                f"[Eval] timesteps={self.stats.timesteps}, reward={eval_reward:.4f}"
                            )

                    # 保存
                    if self.stats.timesteps % config.save_interval < config.steps_per_rollout:
                        self._save_checkpoint()

                    # 回调
                    if callback is not None and not callback(self.stats):
                        if config.verbose >= 1:
                            print("训练被回调函数终止", flush=True)
                            self._write_live_log("[stop] callback requested stop")
                        break

                    if pbar is not None:
                        pbar.set_postfix(
                            {
                                "updates": self.stats.updates,
                                "episodes": self.stats.episodes,
                            }
                        )

                # 训练结束
                self.stats.wall_time = time.time() - start_time
                self._finalize_explainability_stats()
                self._save_checkpoint(final=True)
                self._save_stats()

                if self._live_log_file is not None:
                    self._write_live_log(
                        f"[done] timesteps={self.stats.timesteps}, wall_time={self.stats.wall_time:.2f}s"
                    )

                if config.verbose >= 1:
                    print(f"训练完成: 总步数={self.stats.timesteps}, 耗时={self.stats.wall_time:.2f}s", flush=True)
                    if self.stats.episode_rewards:
                        print(f"平均奖励={np.mean(self.stats.episode_rewards[-100:]):.4f}", flush=True)
                    else:
                        print("平均奖励=N/A (无完整 episode)", flush=True)

                    # 可解释性摘要
                    if config.enable_explainability:
                        print("\n[可解释性摘要]", flush=True)
                        print(f"  平均跨域切分: {self.stats.avg_cross_cuts:.2f}", flush=True)
                        print(f"  平均均衡得分: {self.stats.avg_balance_score:.2f}", flush=True)
                        print(f"  主导因素分布: {self.stats.dominant_factors}", flush=True)

                return self.stats
            finally:
                self._live_log_file = None

    def _write_live_log(self, message: str) -> None:
        """写入实时日志文件（若启用）."""
        if self._live_log_file is not None:
            self._live_log_file.write(message + "\n")

    def _cut_points_to_placement(self, cut_points: List[int]) -> List[int]:
        """将切分点转换为放置向量."""
        num_layers = self.env_config.num_layers
        num_domains = self.env_config.num_domains

        if num_domains <= 1:
            return [0] * num_layers

        cut_points = sorted(cut_points)
        placement = []
        current_domain = 0
        next_cut_idx = 0
        next_cut = cut_points[next_cut_idx] if cut_points else num_layers

        for layer_idx in range(num_layers):
            if layer_idx >= next_cut and current_domain < num_domains - 1:
                current_domain += 1
                next_cut_idx += 1
                next_cut = cut_points[next_cut_idx] if next_cut_idx < len(cut_points) else num_layers
            placement.append(current_domain)

        return placement

    def _evaluate(self, num_episodes: int = 5) -> float:
        """评估当前策略.

        Args:
            num_episodes: 评估 episode 数

        Returns:
            平均奖励
        """
        rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            state = self._augment_state(obs, self.env.current_network_state)
            episode_reward = 0.0
            done = False

            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                placement = self._cut_points_to_placement(action.tolist())
                next_obs, reward, terminated, truncated, _ = self.env.step(placement)
                done = terminated or truncated
                state = self._augment_state(next_obs, self.env.current_network_state)
                episode_reward += reward

            rewards.append(episode_reward)

        return np.mean(rewards)

    def _log_progress(self, update_stats: Dict[str, float]) -> None:
        """记录训练进度."""
        config = self.trainer_config

        if config.verbose >= 1:
            recent_rewards = self.stats.episode_rewards[-10:] if self.stats.episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)

            # 基础日志
            log_msg = (
                f"[Train] timesteps={self.stats.timesteps}, "
                f"episodes={self.stats.episodes}, "
                f"avg_reward={avg_reward:.4f}, "
                f"policy_loss={update_stats['policy_loss']:.4f}, "
                f"value_loss={update_stats['value_loss']:.4f}, "
                f"entropy={update_stats['entropy']:.4f}"
            )

            # 添加可解释性信息
            if config.enable_explainability and self._cross_cuts_history:
                avg_cuts = np.mean(self._cross_cuts_history[-100:])
                avg_balance = np.mean(self._balance_scores_history[-100:])
                log_msg += f", avg_cuts={avg_cuts:.1f}, balance={avg_balance:.2f}"

            print(log_msg)

    def _save_checkpoint(self, final: bool = False) -> None:
        """保存检查点."""
        config = self.trainer_config
        suffix = "final" if final else f"step_{self.stats.timesteps}"
        path = Path(config.model_dir) / f"hi_ppo_{suffix}.pt"
        self.agent.save(str(path))

        if config.verbose >= 1:
            print(f"模型已保存: {path}")

    def _save_stats(self) -> None:
        """保存训练统计."""
        config = self.trainer_config
        path = Path(config.log_dir) / "training_stats.json"

        stats_dict = {
            "timesteps": self.stats.timesteps,
            "episodes": self.stats.episodes,
            "updates": self.stats.updates,
            "total_reward": self.stats.total_reward,
            "wall_time": self.stats.wall_time,
            "episode_rewards": self.stats.episode_rewards,
            "episode_lengths": self.stats.episode_lengths,
            "eval_rewards": self.stats.eval_rewards,
            "final_avg_reward": float(np.mean(self.stats.episode_rewards[-100:])) if self.stats.episode_rewards else 0.0,
            # 可解释性统计
            "explainability": {
                "dominant_factors": self.stats.dominant_factors,
                "avg_cross_cuts": self.stats.avg_cross_cuts,
                "avg_balance_score": self.stats.avg_balance_score,
                "sample_explanations": self.stats.explanations[-10:],  # 最后 10 条解释
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)

        if self.trainer_config.verbose >= 1:
            print(f"统计已保存: {path}")

    def _record_explanation(
        self,
        placement: List[int],
        reward_breakdown,
        network_state,
    ) -> None:
        """记录可解释性信息.

        Args:
            placement: 放置动作
            reward_breakdown: 奖励分解
            network_state: 网络状态
        """
        config = self.trainer_config

        # 生成解释
        explanation = build_explanation(
            placement=placement,
            reward=reward_breakdown,
            network_state=network_state,
        )

        # 更新统计
        placement_info = explanation.get("placement", {})
        cross_cuts = placement_info.get("cross_domain_cuts", 0)
        balance_score = placement_info.get("balance_score", 0.0)

        self._cross_cuts_history.append(cross_cuts)
        self._balance_scores_history.append(balance_score)
        max_hist = int(config.explainability_history_size)
        if max_hist > 0:
            if len(self._cross_cuts_history) > max_hist:
                self._cross_cuts_history = self._cross_cuts_history[-max_hist:]
            if len(self._balance_scores_history) > max_hist:
                self._balance_scores_history = self._balance_scores_history[-max_hist:]

        # 更新主导因素统计
        reward_info = explanation.get("reward_breakdown", {})
        dominant = reward_info.get("dominant_factor", "")
        if dominant in self.stats.dominant_factors:
            self.stats.dominant_factors[dominant] += 1

        # 定期记录完整解释
        if self.stats.timesteps % config.explanation_log_interval == 0:
            self.stats.explanations.append({
                "timestep": self.stats.timesteps,
                "summary": explanation.get("summary", ""),
                "cross_cuts": cross_cuts,
                "balance_score": balance_score,
                "dominant_factor": dominant,
            })

            # 详细输出
            if config.verbose >= 2:
                print(f"\n[Explanation @ step {self.stats.timesteps}]")
                print(explanation.get("summary", ""))
                print()

    def _finalize_explainability_stats(self) -> None:
        """汇总可解释性统计."""
        if self._cross_cuts_history:
            self.stats.avg_cross_cuts = float(np.mean(self._cross_cuts_history))
        if self._balance_scores_history:
            self.stats.avg_balance_score = float(np.mean(self._balance_scores_history))


def train_hi_ppo(
    env_config: Optional[EnvConfig] = None,
    ppo_config: Optional[PPOConfig] = None,
    network_config: Optional[NetworkConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    gnn_config: Optional[GNNConfig] = None,
) -> TrainingStats:
    """便捷训练函数.

    Args:
        env_config: 环境配置
        ppo_config: PPO 配置
        network_config: 网络配置
        trainer_config: 训练器配置
        gnn_config: GNN 配置

    Returns:
        训练统计
    """
    env_config = env_config or EnvConfig()
    ppo_config = ppo_config or PPOConfig()
    network_config = network_config or NetworkConfig()
    trainer_config = trainer_config or TrainerConfig()
    gnn_config = gnn_config or GNNConfig()

    trainer = HiPPOTrainer(
        env_config=env_config,
        ppo_config=ppo_config,
        network_config=network_config,
        trainer_config=trainer_config,
        gnn_config=gnn_config,
    )

    return trainer.train()
