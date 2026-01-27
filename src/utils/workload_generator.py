"""Chakra execution trace generator for placement actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class LayerProfile:
    flops: float
    activation_size: float


@dataclass
class ModelProfile:
    layers: List[LayerProfile]
    peak_flops: float


def generate_chakra_trace(model: ModelProfile, placement: List[int]) -> Dict:
    """Generate a Chakra ET-like trace from layer profiles and placement."""
    nodes = []
    for idx, layer in enumerate(model.layers):
        nodes.append(
            {
                "id": idx * 2,
                "name": f"layer_{idx}_compute",
                "type": "COMP_NODE",
                "domain": placement[idx],
                "runtime": layer.flops / model.peak_flops,
            }
        )
        if idx < len(placement) - 1 and placement[idx] != placement[idx + 1]:
            nodes.append(
                {
                    "id": idx * 2 + 1,
                    "name": f"layer_{idx}_cross_domain_comm",
                    "type": "COMM_SEND_NODE",
                    "src_domain": placement[idx],
                    "dst_domain": placement[idx + 1],
                    "comm_size": layer.activation_size,
                }
            )

    return {"schema": "chakra_0.0.4", "nodes": nodes}


def write_chakra_trace(path: Path, trace: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path
