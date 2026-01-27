"""Minimal rollout loop connecting Hi-PPO coordinator with the environment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import csv

from ...envs.astra_env import AstraSimEnv
from ...utils.explainability import build_explanation
from .coordinator import HiPPOCoordinator


def run_rollout(
    env: AstraSimEnv,
    coordinator: HiPPOCoordinator,
    episodes: int = 1,
    steps_per_episode: int = 1,
    log_path: Optional[Path] = None,
) -> List[dict]:
    """Run a minimal rollout with optional CSV logging."""
    logs: List[dict] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=env.config.seed + episode)
        for step in range(steps_per_episode):
            state_high = obs
            state_low = [obs for _ in range(env.config.num_domains)]
            domain_loads = list(obs[-env.config.num_domains:]) if len(obs) >= env.config.num_domains else None
            action = coordinator.select_action(
                state_high,
                state_low,
                network_state=env.current_network_state,
                domain_loads=domain_loads,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)

            metrics = info.get("metrics")
            reward_breakdown = info.get("reward_breakdown")
            network_state_used = info.get("network_state", env.current_network_state)

            row = {
                "episode": episode,
                "step": step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
            if metrics is not None:
                row.update(asdict(metrics))
            if reward_breakdown is not None:
                row.update({f"rb_{k}": v for k, v in asdict(reward_breakdown).items()})

            explanation = None
            if reward_breakdown is not None:
                explanation = build_explanation(
                    placement=action,
                    reward=reward_breakdown,
                    network_state=network_state_used,
                )
                info["explanation"] = explanation

            row["explanation_summary"] = explanation.get("summary", "") if explanation else ""
            row["dominant_factor"] = (
                explanation.get("reward_breakdown", {}).get("dominant_factor", "")
                if explanation
                else ""
            )
            placement_info = explanation.get("placement", {}) if explanation else {}
            row["cross_cuts"] = placement_info.get("cross_domain_cuts", 0)
            row["balance_score"] = placement_info.get("balance_score", 0.0)

            logs.append(row)
            obs = next_obs

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(log_path, logs)

    return logs


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
