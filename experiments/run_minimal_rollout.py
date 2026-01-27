"""Minimal rollout that wires HiPPOCoordinator to the AstraSimEnv (mock mode)."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.hi_ppo import CoordinatorConfig, HiPPOCoordinator, NoOpLocalAgent, RandomGlobalAgent
from src.algorithms.hi_ppo.rollout import run_rollout
from src.envs.astra_env import AstraSimEnv, EnvConfig


def main() -> None:
    env_config = EnvConfig(use_mock=True, num_layers=24, num_domains=3)
    env = AstraSimEnv(env_config)

    coordinator_config = CoordinatorConfig(num_layers=env_config.num_layers, num_domains=env_config.num_domains)
    global_agent = RandomGlobalAgent(env_config.num_layers, env_config.num_domains, seed=env_config.seed)
    local_agent = NoOpLocalAgent()
    coordinator = HiPPOCoordinator(coordinator_config, global_agent, local_agent)

    logs = run_rollout(
        env,
        coordinator,
        episodes=1,
        steps_per_episode=3,
        log_path=Path("results/minimal_rollout.csv"),
    )

    print(f"Completed rollout steps: {len(logs)}")
    print("Saved log to results/minimal_rollout.csv")


if __name__ == "__main__":
    main()
