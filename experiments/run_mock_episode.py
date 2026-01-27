"""Run a single mock episode step to validate wiring (mock mode)."""

from __future__ import annotations

from pprint import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.astra_env import AstraSimEnv, EnvConfig


def main() -> None:
    config = EnvConfig(use_mock=True, num_layers=12, num_domains=3)
    env = AstraSimEnv(config)
    obs, _ = env.reset(seed=config.seed)
    action = env.sample_random_action()
    next_obs, reward, terminated, truncated, info = env.step(action)

    print("Observation length:", len(obs))
    print("Next observation length:", len(next_obs))
    print("Reward:", reward)
    print("Terminated:", terminated, "Truncated:", truncated)
    pprint(info)


if __name__ == "__main__":
    main()
