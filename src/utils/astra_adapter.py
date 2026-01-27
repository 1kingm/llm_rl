#!/usr/bin/env python3
"""
Astra-sim adapter: map layer placement actions to config files.

Features:
- Accept placement list like [0,0,1,1,2,...]
- Generate Astra-sim 2.0 configs (network.yml / system.json / workload .et)
- Optionally run Astra-sim binary or produce a mock result to close the loop
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class TopologyConfig:
    num_domains: int
    topology: str  # "ring" or "star"
    intra_bandwidth_gbps: float = 400.0
    inter_bandwidth_gbps: float = 5.0
    intra_latency_ms: float = 0.5
    inter_latency_ms: float = 30.0
    disconnected_bandwidth_gbps: float = 0.1
    disconnected_latency_ms: float = 999.0


@dataclass(frozen=True)
class AstraSystemConfig:
    scheduling_policy: str = "LIFO"
    collective_optimization: str = "localBWAware"
    endpoint_delay: int = 10
    active_chunks_per_dimension: int = 1
    preferred_dataset_splits: int = 1
    local_mem_bw_gbps: float = 1600.0
    peak_perf_tflops: float = 120.0
    roofline_enabled: int = 0
    trace_enabled: int = 0
    replay_only: int = 0
    track_local_mem: int = 0
    all_reduce_impl: str = "ring"
    all_gather_impl: str = "ring"
    reduce_scatter_impl: str = "ring"
    all_to_all_impl: str = "ring"


def parse_placement(value: str) -> List[int]:
    value = value.strip()
    if value.startswith("["):
        return list(json.loads(value))
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def estimate_comm_volume_gb(placement: Iterable[int], comm_size_bytes: int) -> float:
    placement = list(placement)
    if len(placement) < 2:
        return 0.0
    cross_edges = sum(1 for i in range(len(placement) - 1) if placement[i] != placement[i + 1])
    return (cross_edges * comm_size_bytes) / 1e9


def infer_num_domains(placement: Iterable[int], default: int = 3) -> int:
    placement = list(placement)
    if not placement:
        return default
    max_id = max(placement)
    return max(default, max_id + 1)


def _init_matrix(n: int, diag_value: float, off_value: float) -> List[List[float]]:
    return [[diag_value if i == j else off_value for j in range(n)] for i in range(n)]


def build_bandwidth_latency(topo: TopologyConfig) -> tuple[list[list[float]], list[list[float]]]:
    n = topo.num_domains
    bw = _init_matrix(n, topo.intra_bandwidth_gbps, topo.disconnected_bandwidth_gbps)
    lat = _init_matrix(n, topo.intra_latency_ms, topo.disconnected_latency_ms)

    if topo.topology == "ring":
        for i in range(n):
            j = (i + 1) % n
            bw[i][j] = topo.inter_bandwidth_gbps
            bw[j][i] = topo.inter_bandwidth_gbps
            lat[i][j] = topo.inter_latency_ms
            lat[j][i] = topo.inter_latency_ms
    elif topo.topology == "star":
        center = 0
        for i in range(1, n):
            bw[center][i] = topo.inter_bandwidth_gbps
            bw[i][center] = topo.inter_bandwidth_gbps
            lat[center][i] = topo.inter_latency_ms
            lat[i][center] = topo.inter_latency_ms
    else:
        raise ValueError(f"Unsupported topology: {topo.topology}")

    return bw, lat


def generate_network_config(out_path: Path, topo: TopologyConfig) -> Path:
    warnings.warn(
        "generate_network_config is deprecated for Astra-sim 2.0. "
        "Use generate_network_config_yaml instead.",
        DeprecationWarning,
    )
    bw, lat = build_bandwidth_latency(topo)
    payload = {
        "topology": topo.topology,
        "num_domains": topo.num_domains,
        "bandwidth_matrix": bw,
        "latency_matrix": lat,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def generate_network_config_from_matrices(
    out_path: Path,
    bandwidth_matrix: List[List[float]],
    latency_matrix: List[List[float]],
    topology: str = "custom",
) -> Path:
    payload = {
        "topology": topology,
        "num_domains": len(bandwidth_matrix),
        "bandwidth_matrix": bandwidth_matrix,
        "latency_matrix": latency_matrix,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def generate_network_config_yaml(
    out_path: Path,
    num_npus: int,
    topology: str,
    bandwidth_gbps: float,
    latency_ms: float,
) -> Path:
    topology_map = {
        "ring": "Ring",
        "star": "Switch",
        "switch": "Switch",
        "fully": "FullyConnected",
        "full": "FullyConnected",
        "fullyconnected": "FullyConnected",
        "ring": "Ring",
    }
    topo_key = topology.lower()
    topo_name = topology_map.get(topo_key, topology)
    # Analytical backend expects GB/s and ns.
    bandwidth_gbps = max(0.0, bandwidth_gbps)
    bandwidth_gbps_to_gbs = bandwidth_gbps / 8.0
    latency_ns = max(0.0, latency_ms * 1_000_000.0)

    payload = (
        f"topology: [ {topo_name} ]\n"
        f"npus_count: [ {num_npus} ]\n"
        f"bandwidth: [ {bandwidth_gbps_to_gbs:.6f} ]\n"
        f"latency: [ {latency_ns:.6f} ]\n"
    )
    out_path.write_text(payload, encoding="utf-8")
    return out_path


@dataclass
class MultiDomainTopologyConfig:
    """多域网络拓扑配置.

    支持 2D 拓扑结构：
    - 第一维: 域内拓扑 (Switch/Ring/FullyConnected)
    - 第二维: 域间拓扑 (Ring/Switch/FullyConnected)

    这种结构模拟真实的跨数据中心 LLM 训练场景：
    - 域内: 高速 NVLink/NVSwitch 互联
    - 域间: WAN 广域网连接
    """

    num_domains: int = 4  # 域数量
    gpus_per_domain: int = 4  # 每域 GPU 数
    intra_topology: str = "Switch"  # 域内拓扑
    inter_topology: str = "Ring"  # 域间拓扑
    intra_bandwidth_gbs: float = 400.0  # 域内带宽 (GB/s)
    inter_bandwidth_gbs: float = 25.0  # 域间带宽 (GB/s)
    intra_latency_ns: float = 500.0  # 域内延迟 (ns)
    inter_latency_ns: float = 10_000_000.0  # 域间延迟 (ns) = 10 ms


def generate_network_config_yaml_2d(
    out_path: Path,
    config: MultiDomainTopologyConfig,
) -> Path:
    """生成 2D 多域网络拓扑配置.

    Astra-sim 2.0 支持多维拓扑，格式为：
    topology: [ dim0_type, dim1_type, ... ]
    npus_count: [ dim0_count, dim1_count, ... ]
    bandwidth: [ dim0_bw, dim1_bw, ... ]  # GB/s
    latency: [ dim0_lat, dim1_lat, ... ]  # ns

    对于跨域 LLM 训练：
    - 第一维 (dim0): 域内拓扑，高带宽低延迟
    - 第二维 (dim1): 域间拓扑，低带宽高延迟

    总 NPU 数 = gpus_per_domain × num_domains
    """
    topology_map = {
        "ring": "Ring",
        "switch": "Switch",
        "fullyconnected": "FullyConnected",
    }

    intra_topo = topology_map.get(config.intra_topology.lower(), config.intra_topology)
    inter_topo = topology_map.get(config.inter_topology.lower(), config.inter_topology)

    payload = (
        f"# 多域跨数据中心拓扑 (2D)\n"
        f"# 第一维: {intra_topo} (域内), 第二维: {inter_topo} (域间)\n"
        f"# 总 NPU 数: {config.gpus_per_domain} × {config.num_domains} = {config.gpus_per_domain * config.num_domains}\n"
        f"topology: [ {intra_topo}, {inter_topo} ]\n"
        f"npus_count: [ {config.gpus_per_domain}, {config.num_domains} ]\n"
        f"bandwidth: [ {config.intra_bandwidth_gbs}, {config.inter_bandwidth_gbs} ]\n"
        f"latency: [ {config.intra_latency_ns}, {config.inter_latency_ns} ]\n"
    )
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def generate_logical_topology_json(out_path: Path, dims: List[int]) -> Path:
    """生成 NS3 逻辑拓扑配置 (logical-dims)."""
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"logical dims must be positive: {dims}")
    payload = {"logical-dims": [str(d) for d in dims]}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def find_ns3_binary(ns3_build_dir: Path) -> Path:
    """在 ns-3 build/scratch 目录中查找 AstraSimNetwork 二进制."""
    if not ns3_build_dir.exists():
        raise FileNotFoundError(f"ns-3 build dir not found: {ns3_build_dir}")
    candidates = sorted(ns3_build_dir.glob("ns3.*AstraSimNetwork-*"))
    if not candidates:
        raise FileNotFoundError(
            f"No ns3 AstraSimNetwork binary found in {ns3_build_dir}. "
            "Please build ns-3 backend first."
        )
    for cand in candidates:
        if "default" in cand.name:
            return cand
    return candidates[0]


def generate_system_config_v2(out_path: Path, config: AstraSystemConfig) -> Path:
    payload = {
        "scheduling-policy": config.scheduling_policy,
        "collective-optimization": config.collective_optimization,
        "endpoint-delay": config.endpoint_delay,
        "active-chunks-per-dimension": config.active_chunks_per_dimension,
        "preferred-dataset-splits": config.preferred_dataset_splits,
        "local-mem-bw": config.local_mem_bw_gbps,
        "peak-perf": config.peak_perf_tflops,
        "roofline-enabled": config.roofline_enabled,
        "trace-enabled": config.trace_enabled,
        "replay-only": config.replay_only,
        "track-local-mem": config.track_local_mem,
        "all-reduce-implementation": [config.all_reduce_impl],
        "all-gather-implementation": [config.all_gather_impl],
        "reduce-scatter-implementation": [config.reduce_scatter_impl],
        "all-to-all-implementation": [config.all_to_all_impl],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def generate_remote_memory_config(out_path: Path, memory_type: str = "NO_MEMORY_EXPANSION") -> Path:
    payload = {"memory-type": memory_type}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _load_chakra_modules() -> tuple[object, object]:
    chakra_root = Path(__file__).resolve().parents[2] / "astra-sim" / "extern" / "graph_frontend" / "chakra"
    proto_dir = chakra_root / "schema" / "protobuf"
    utils_dir = chakra_root / "src" / "third_party" / "utils"
    if not proto_dir.exists() or not utils_dir.exists():
        raise FileNotFoundError("Chakra protobuf modules not found. Ensure astra-sim submodules are initialized.")
    sys.path.insert(0, str(proto_dir))
    sys.path.insert(0, str(utils_dir))
    import importlib

    et_def_pb2 = importlib.import_module("et_def_pb2")
    protolib = importlib.import_module("protolib")
    return et_def_pb2, protolib


def generate_workload_et(
    out_dir: Path,
    placement: List[int],
    num_npus: int,
    layer_runtime_us: int = 100,
    comm_size_bytes: int = 1_000_000,
    allow_fallback: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "trace"

    try:
        et_def_pb2, protolib = _load_chakra_modules()
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Chakra protobuf modules unavailable or incompatible. "
                "Upgrade protobuf to match protoc (e.g., 6.32.1) "
                "or pass allow_fallback=True to use Astra-sim microbench ET."
            ) from exc
        available = [4, 8, 16]
        chosen = next((n for n in available if n >= num_npus), max(available))
        example_dir = (
            Path(__file__).resolve().parents[2]
            / "astra-sim"
            / "examples"
            / "workload"
            / "microbenchmarks"
            / "all_reduce"
            / f"{chosen}npus_1MB"
        )
        if not example_dir.exists():
            raise
        for idx in range(num_npus):
            source = example_dir / f"all_reduce.{idx}.et"
            target = out_dir / f"trace.{idx}.et"
            shutil.copyfile(source, target)
        return prefix

    for npu_id in range(num_npus):
        filename = f"{prefix}.{npu_id}.et"
        with open(filename, "wb") as handle:
            protolib.encodeMessage(handle, et_def_pb2.GlobalMetadata(version="0.0.4"))
            node_id = 0
            prev_id: Optional[int] = None
            tensor_bytes = max(1, int(comm_size_bytes))
            num_ops = max(1, int(layer_runtime_us)) * 1_000_000

            for idx, domain in enumerate(placement):
                # Compute node for layers mapped to this NPU.
                if domain == npu_id:
                    node = et_def_pb2.Node()
                    node.id = node_id
                    node.name = f"layer_{idx}_compute"
                    node.type = et_def_pb2.COMP_NODE
                    node.duration_micros = int(layer_runtime_us)
                    node.attr.extend([et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False)])
                    node.attr.extend(
                        [
                            et_def_pb2.AttributeProto(name="num_ops", uint64_val=num_ops),
                            et_def_pb2.AttributeProto(name="tensor_size", uint64_val=tensor_bytes),
                        ]
                    )
                    if prev_id is not None:
                        node.data_deps.append(prev_id)
                    protolib.encodeMessage(handle, node)
                    prev_id = node_id
                    node_id += 1

                # Cross-domain communication between consecutive layers.
                if idx < len(placement) - 1 and placement[idx] != placement[idx + 1]:
                    src = placement[idx]
                    dst = placement[idx + 1]
                    if npu_id == src:
                        node = et_def_pb2.Node()
                        node.id = node_id
                        node.name = f"layer_{idx}_send"
                        node.type = et_def_pb2.COMM_SEND_NODE
                        node.attr.extend(
                            [
                                et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                                et_def_pb2.AttributeProto(name="comm_src", uint32_val=src),
                                et_def_pb2.AttributeProto(name="comm_dst", uint32_val=dst),
                                et_def_pb2.AttributeProto(name="comm_size", uint64_val=comm_size_bytes),
                                et_def_pb2.AttributeProto(name="comm_tag", uint32_val=0),
                            ]
                        )
                        if prev_id is not None:
                            node.data_deps.append(prev_id)
                        protolib.encodeMessage(handle, node)
                        prev_id = node_id
                        node_id += 1
                    elif npu_id == dst:
                        node = et_def_pb2.Node()
                        node.id = node_id
                        node.name = f"layer_{idx}_recv"
                        node.type = et_def_pb2.COMM_RECV_NODE
                        node.attr.extend(
                            [
                                et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                                et_def_pb2.AttributeProto(name="comm_src", uint32_val=src),
                                et_def_pb2.AttributeProto(name="comm_dst", uint32_val=dst),
                                et_def_pb2.AttributeProto(name="comm_size", uint64_val=comm_size_bytes),
                                et_def_pb2.AttributeProto(name="comm_tag", uint32_val=0),
                            ]
                        )
                        if prev_id is not None:
                            node.data_deps.append(prev_id)
                        protolib.encodeMessage(handle, node)
                        prev_id = node_id
                        node_id += 1

    return prefix


def generate_allreduce_workload(
    out_dir: Path,
    num_npus: int,
    data_size_bytes: int = 1_000_000,
    num_iterations: int = 10,
    compute_us: int = 100,
    allow_fallback: bool = False,
) -> Path:
    """生成 AllReduce 集合通信工作负载.

    AllReduce 是分布式训练中最常用的集合通信操作，用于梯度同步。
    所有 NPU 同时参与 AllReduce，创建并发通信流量，
    这是测试 congestion-aware 调度的关键场景。

    工作负载结构（每个 NPU）：
    1. 计算节点 (模拟前向/反向传播)
    2. AllReduce 集合通信节点 (梯度同步)
    3. 重复 num_iterations 次

    Args:
        out_dir: 输出目录
        num_npus: NPU 数量
        data_size_bytes: 每次 AllReduce 的数据量 (bytes)
        num_iterations: 迭代次数
        compute_us: 每次迭代的计算时间 (微秒)
        allow_fallback: 是否允许回退到示例 ET

    Returns:
        工作负载文件前缀路径
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "allreduce"

    try:
        et_def_pb2, protolib = _load_chakra_modules()
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Chakra protobuf modules unavailable or incompatible. "
                "Upgrade protobuf to match protoc (e.g., 6.32.1) "
                "or pass allow_fallback=True to use Astra-sim microbench ET."
            ) from exc
        # 回退到 Astra-sim 示例 AllReduce ET
        available = [4, 8, 16]
        chosen = next((n for n in available if n >= num_npus), max(available))
        example_dir = (
            Path(__file__).resolve().parents[2]
            / "astra-sim"
            / "examples"
            / "workload"
            / "microbenchmarks"
            / "all_reduce"
            / f"{chosen}npus_1MB"
        )
        if not example_dir.exists():
            raise
        for idx in range(num_npus):
            source = example_dir / f"all_reduce.{idx}.et"
            target = out_dir / f"allreduce.{idx}.et"
            shutil.copyfile(source, target)
        return prefix

    # 为每个 NPU 生成 ET 文件
    for npu_id in range(num_npus):
        filename = f"{prefix}.{npu_id}.et"
        with open(filename, "wb") as handle:
            protolib.encodeMessage(handle, et_def_pb2.GlobalMetadata(version="0.0.4"))
            node_id = 0
            prev_id: Optional[int] = None
            tensor_bytes = max(1, int(data_size_bytes))
            num_ops = max(1, int(compute_us)) * 1_000_000

            for iteration in range(num_iterations):
                # 1. 计算节点 (模拟前向/反向传播)
                comp_node = et_def_pb2.Node()
                comp_node.id = node_id
                comp_node.name = f"iter_{iteration}_compute"
                comp_node.type = et_def_pb2.COMP_NODE
                comp_node.duration_micros = int(compute_us)
                comp_node.attr.extend([
                    et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                    et_def_pb2.AttributeProto(name="num_ops", uint64_val=num_ops),
                    et_def_pb2.AttributeProto(name="tensor_size", uint64_val=tensor_bytes),
                ])
                if prev_id is not None:
                    comp_node.data_deps.append(prev_id)
                protolib.encodeMessage(handle, comp_node)
                prev_id = node_id
                node_id += 1

                # 2. AllReduce 集合通信节点
                allreduce_node = et_def_pb2.Node()
                allreduce_node.id = node_id
                allreduce_node.name = f"iter_{iteration}_allreduce"
                allreduce_node.type = et_def_pb2.COMM_COLL_NODE
                allreduce_node.attr.extend([
                    et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                    et_def_pb2.AttributeProto(name="comm_type", int64_val=et_def_pb2.ALL_REDUCE),
                    et_def_pb2.AttributeProto(name="comm_size", int64_val=data_size_bytes),
                ])
                if prev_id is not None:
                    allreduce_node.data_deps.append(prev_id)
                protolib.encodeMessage(handle, allreduce_node)
                prev_id = node_id
                node_id += 1

    return prefix


def estimate_allreduce_comm_volume_gb(
    num_npus: int,
    data_size_bytes: int,
    num_iterations: int = 1,
) -> float:
    """估算 AllReduce 的通信量.

    Ring AllReduce 的通信量公式：
    每个 NPU 发送和接收 2 * (N-1) / N * data_size 的数据
    总通信量 = N * 2 * (N-1) / N * data_size = 2 * (N-1) * data_size

    Args:
        num_npus: NPU 数量
        data_size_bytes: 每次 AllReduce 的数据量
        num_iterations: 迭代次数

    Returns:
        总通信量 (GB)
    """
    if num_npus <= 1:
        return 0.0
    # Ring AllReduce: 每个 NPU 发送 2*(N-1)/N 的数据
    comm_per_iteration = 2 * (num_npus - 1) * data_size_bytes
    return (comm_per_iteration * num_iterations) / 1e9


def generate_alltoall_workload(
    out_dir: Path,
    num_npus: int,
    data_size_bytes: int = 1_000_000,
    num_iterations: int = 10,
    compute_us: int = 100,
    allow_fallback: bool = False,
) -> Path:
    """生成 All-to-All 集合通信工作负载.

    All-to-All 是一种全交换通信模式，每个 NPU 向所有其他 NPU 发送数据。
    在 Ring 拓扑上，All-to-All 会产生严重的链路竞争，
    这是测试 congestion-aware 调度的理想场景。

    工作负载结构（每个 NPU）：
    1. 计算节点 (模拟前向/反向传播)
    2. All-to-All 集合通信节点
    3. 重复 num_iterations 次

    Args:
        out_dir: 输出目录
        num_npus: NPU 数量
        data_size_bytes: 每次 All-to-All 的数据量 (bytes)
        num_iterations: 迭代次数
        compute_us: 每次迭代的计算时间 (微秒)
        allow_fallback: 是否允许回退到示例 ET

    Returns:
        工作负载文件前缀路径
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "alltoall"

    try:
        et_def_pb2, protolib = _load_chakra_modules()
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Chakra protobuf modules unavailable or incompatible. "
                "Upgrade protobuf to match protoc (e.g., 6.32.1) "
                "or pass allow_fallback=True to use Astra-sim microbench ET."
            ) from exc
        # 回退到 Astra-sim 示例 All-to-All ET
        available = [4, 8, 16]
        chosen = next((n for n in available if n >= num_npus), max(available))
        example_dir = (
            Path(__file__).resolve().parents[2]
            / "astra-sim"
            / "examples"
            / "workload"
            / "microbenchmarks"
            / "all_to_all"
            / f"{chosen}npus_1MB"
        )
        if not example_dir.exists():
            raise
        for idx in range(num_npus):
            source = example_dir / f"all_to_all.{idx}.et"
            target = out_dir / f"alltoall.{idx}.et"
            shutil.copyfile(source, target)
        return prefix

    # 为每个 NPU 生成 ET 文件
    for npu_id in range(num_npus):
        filename = f"{prefix}.{npu_id}.et"
        with open(filename, "wb") as handle:
            protolib.encodeMessage(handle, et_def_pb2.GlobalMetadata(version="0.0.4"))
            node_id = 0
            prev_id: Optional[int] = None
            tensor_bytes = max(1, int(data_size_bytes))
            num_ops = max(1, int(compute_us)) * 1_000_000

            for iteration in range(num_iterations):
                # 1. 计算节点 (模拟前向/反向传播)
                comp_node = et_def_pb2.Node()
                comp_node.id = node_id
                comp_node.name = f"iter_{iteration}_compute"
                comp_node.type = et_def_pb2.COMP_NODE
                comp_node.duration_micros = int(compute_us)
                comp_node.attr.extend([
                    et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                    et_def_pb2.AttributeProto(name="num_ops", uint64_val=num_ops),
                    et_def_pb2.AttributeProto(name="tensor_size", uint64_val=tensor_bytes),
                ])
                if prev_id is not None:
                    comp_node.data_deps.append(prev_id)
                protolib.encodeMessage(handle, comp_node)
                prev_id = node_id
                node_id += 1

                # 2. All-to-All 集合通信节点
                alltoall_node = et_def_pb2.Node()
                alltoall_node.id = node_id
                alltoall_node.name = f"iter_{iteration}_alltoall"
                alltoall_node.type = et_def_pb2.COMM_COLL_NODE
                alltoall_node.attr.extend([
                    et_def_pb2.AttributeProto(name="is_cpu_op", bool_val=False),
                    et_def_pb2.AttributeProto(name="comm_type", int64_val=et_def_pb2.ALL_TO_ALL),
                    et_def_pb2.AttributeProto(name="comm_size", int64_val=data_size_bytes),
                ])
                if prev_id is not None:
                    alltoall_node.data_deps.append(prev_id)
                protolib.encodeMessage(handle, alltoall_node)
                prev_id = node_id
                node_id += 1

    return prefix


def estimate_alltoall_comm_volume_gb(
    num_npus: int,
    data_size_bytes: int,
    num_iterations: int = 1,
) -> float:
    """估算 All-to-All 的通信量.

    All-to-All 的通信量公式：
    每个 NPU 向其他 N-1 个 NPU 发送 data_size 的数据
    总通信量 = N * (N-1) * data_size

    Args:
        num_npus: NPU 数量
        data_size_bytes: 每次 All-to-All 的数据量
        num_iterations: 迭代次数

    Returns:
        总通信量 (GB)
    """
    if num_npus <= 1:
        return 0.0
    # All-to-All: 每个 NPU 发送 (N-1) * data_size 的数据
    comm_per_iteration = num_npus * (num_npus - 1) * data_size_bytes
    return (comm_per_iteration * num_iterations) / 1e9


def parse_astra_stdout(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    cycles_match = re.search(r"finished,\s+(\d+)\s+cycles,\s+exposed communication\s+(\d+)\s+cycles", output)
    if cycles_match:
        metrics["total_cycles"] = float(cycles_match.group(1))
        metrics["comm_cycles"] = float(cycles_match.group(2))
    util_match = re.search(r"Average compute utilization:\s+([0-9.]+)%", output)
    if util_match:
        metrics["utilization"] = float(util_match.group(1)) / 100.0
    return metrics


def generate_system_config(
    out_path: Path,
    num_domains: int,
    nodes_per_domain: int,
    peak_perfs: List[float],
) -> Path:
    warnings.warn(
        "generate_system_config is deprecated for Astra-sim 2.0. "
        "Use generate_system_config_v2 instead.",
        DeprecationWarning,
    )
    if len(peak_perfs) != num_domains:
        raise ValueError("peak_perfs length must match num_domains")
    payload = {
        "num_domains": num_domains,
        "domains": [
            {
                "id": i,
                "num_nodes": nodes_per_domain,
                # unit: TFLOPS (homogeneous by default)
                "peak_perf": peak_perfs[i],
            }
            for i in range(num_domains)
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def generate_workload_config(out_path: Path, placement: List[int]) -> Path:
    warnings.warn(
        "generate_workload_config is deprecated for Astra-sim 2.0. "
        "Use generate_workload_et instead.",
        DeprecationWarning,
    )
    payload = {
        "num_layers": len(placement),
        "placement": placement,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_astra(
    astra_bin: Path,
    network_cfg: Path,
    system_cfg: Path,
    workload_cfg: Path,
    remote_mem_cfg: Optional[Path] = None,
    output_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    comm_volume_gb: Optional[float] = None,
) -> dict[str, float]:
    cmd = [
        str(astra_bin),
        f"--network-configuration={network_cfg}",
        f"--system-configuration={system_cfg}",
        f"--workload-configuration={workload_cfg}",
    ]
    if remote_mem_cfg is not None:
        cmd.append(f"--remote-memory-configuration={remote_mem_cfg}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")

    metrics = parse_astra_stdout(combined)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_cycles = metrics.get("total_cycles", 0.0)
        comm_cycles = metrics.get("comm_cycles", 0.0)
        utilization = metrics.get("utilization", 0.0)
        if comm_volume_gb is None:
            comm_volume_gb = 0.0
        # 注意: Astra-sim analytical 后端不直接输出 comm_volume_gb（可通过 placement 估算传入）。
        output_path.write_text(
            "total_cycles,comm_cycles,comm_volume_gb,utilization\n"
            f"{int(total_cycles)},{int(comm_cycles)},{comm_volume_gb},{utilization}\n",
            encoding="utf-8",
        )
    return metrics


def run_astra_ns3(
    ns3_bin: Path,
    network_cfg: Path,
    system_cfg: Path,
    workload_cfg: Path,
    logical_topology_cfg: Path,
    remote_mem_cfg: Optional[Path] = None,
    comm_group_cfg: str = "empty",
    output_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    comm_volume_gb: Optional[float] = None,
) -> dict[str, float]:
    """运行 Astra-sim NS3 后端.

    注意: NS3 配置文件中的相对路径依赖当前工作目录，默认使用 ns3_bin.parent 作为 cwd。
    """
    cmd = [
        str(ns3_bin),
        f"--workload-configuration={workload_cfg}",
        f"--system-configuration={system_cfg}",
        f"--network-configuration={network_cfg}",
        f"--logical-topology-configuration={logical_topology_cfg}",
        f"--comm-group-configuration={comm_group_cfg}",
    ]
    if remote_mem_cfg is not None:
        cmd.append(f"--remote-memory-configuration={remote_mem_cfg}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ns3_bin.parent),
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")

    metrics = parse_astra_stdout(combined)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_cycles = metrics.get("total_cycles", 0.0)
        comm_cycles = metrics.get("comm_cycles", 0.0)
        utilization = metrics.get("utilization", 0.0)
        if comm_volume_gb is None:
            comm_volume_gb = 0.0
        output_path.write_text(
            "total_cycles,comm_cycles,comm_volume_gb,utilization\n"
            f"{int(total_cycles)},{int(comm_cycles)},{comm_volume_gb},{utilization}\n",
            encoding="utf-8",
        )
    return metrics


def mock_run(
    output_path: Path,
    placement: List[int],
    comm_size_bytes: int = 1_000_000,
    seed: int = 42,
    num_domains: int = 3,
    layer_runtime_us: int = 100,
) -> Path:
    """生成模拟的 Astra-sim 运行结果.

    使用非线性性能模型，考虑：
    1. 跨域通信的拥塞效应（非线性惩罚）
    2. 负载不均衡对利用率的影响
    3. 通信与计算的重叠

    Args:
        output_path: 输出文件路径
        placement: 层到域的映射
        comm_size_bytes: 每次跨域通信的数据量
        seed: 随机种子
        num_domains: 域数量
        layer_runtime_us: 每层计算时间（微秒）

    Returns:
        输出文件路径
    """
    import random as rng
    rng.seed(seed)

    num_layers = len(placement)
    if num_layers == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "total_cycles,comm_cycles,comm_volume_gb,utilization\n"
            "0,0,0.0,0.0\n",
            encoding="utf-8",
        )
        return output_path

    # 计算跨域边数
    cross_edges = sum(
        1 for i in range(num_layers - 1)
        if placement[i] != placement[i + 1]
    )

    # 计算负载分布
    domain_counts = [0] * num_domains
    for d in placement:
        if 0 <= d < num_domains:
            domain_counts[d] += 1

    # 负载不均衡度（基尼系数）
    total_layers = sum(domain_counts)
    if total_layers > 0:
        sorted_counts = sorted(domain_counts)
        cumsum = 0.0
        gini_sum = 0.0
        for i, c in enumerate(sorted_counts):
            cumsum += c
            gini_sum += cumsum - c / 2
        load_imbalance = 1 - 2 * gini_sum / (num_domains * total_layers) if total_layers > 0 else 0
    else:
        load_imbalance = 0.0

    # 基础计算周期（假设 1.5 GHz 时钟）
    cycles_per_us = 1500
    base_compute_cycles = num_layers * layer_runtime_us * cycles_per_us

    # 非线性通信惩罚模型
    # 拥塞效应：每条额外的跨域边会增加通信开销
    # 使用对数模型：penalty = 1 + α * log(1 + cross_edges) + β * cross_edges^γ
    if cross_edges == 0:
        comm_penalty = 1.0
        comm_cycles = 0
    else:
        import math
        alpha = 0.1  # 对数项系数
        beta = 0.02  # 幂次项系数
        gamma = 1.3  # 幂次（>1 表示超线性增长）

        comm_penalty = 1.0 + alpha * math.log(1 + cross_edges) + beta * (cross_edges ** gamma)

        # 通信周期：基于带宽和数据量
        # 假设跨域带宽 5 Gbps，每次传输 comm_size_bytes
        inter_bw_gbps = 5.0
        bytes_per_cycle = (inter_bw_gbps * 1e9 / 8) / (1.5 * 1e9)  # 1.5 GHz
        base_comm_cycles_per_edge = comm_size_bytes / bytes_per_cycle

        # 拥塞导致的额外延迟（非线性）
        congestion_factor = 1.0 + 0.1 * math.log(1 + cross_edges)
        comm_cycles = int(cross_edges * base_comm_cycles_per_edge * congestion_factor)

    # 总周期数
    total_cycles = int(base_compute_cycles * comm_penalty)

    # 通信量（精确计算）
    comm_volume_gb = (cross_edges * comm_size_bytes) / 1e9

    # 利用率模型
    # 考虑：1) 负载不均衡 2) 通信开销 3) 随机波动
    # 基础利用率 80%，负载不均衡和通信都会降低利用率
    base_util = 0.8
    imbalance_penalty = 0.15 * load_imbalance  # 负载不均衡惩罚
    comm_overhead_ratio = comm_cycles / max(1, total_cycles)
    comm_penalty_util = 0.2 * comm_overhead_ratio  # 通信开销惩罚

    utilization = max(0.3, base_util - imbalance_penalty - comm_penalty_util)
    # 添加小幅随机波动
    utilization = max(0.3, min(0.95, utilization + rng.uniform(-0.02, 0.02)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "total_cycles,comm_cycles,comm_volume_gb,utilization\n"
        f"{total_cycles},{comm_cycles},{comm_volume_gb:.6f},{utilization:.4f}\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Astra-sim configs from placement list.")
    parser.add_argument("--placement", required=True, help="Placement list, e.g. 0,0,1,1,2 or [0,0,1,1,2]")
    parser.add_argument("--topology", choices=["ring", "star"], default="ring")
    parser.add_argument("--num-domains", type=int, default=3)
    parser.add_argument("--nodes-per-domain", type=int, default=8)
    parser.add_argument(
        "--peak-perf",
        default="120",
        help="Comma-separated peak perf per domain (length must match num_domains).",
    )
    parser.add_argument("--local-mem-bw", type=float, default=1600.0, help="Local memory bandwidth (GB/s).")
    parser.add_argument("--layer-runtime-us", type=int, default=100, help="Per-layer runtime in microseconds.")
    parser.add_argument("--comm-size-bytes", type=int, default=1_000_000, help="Cross-domain comm size in bytes.")
    parser.add_argument(
        "--allow-et-fallback",
        action="store_true",
        help="Allow fallback to Astra-sim microbench ET when protobuf is incompatible.",
    )
    parser.add_argument("--out-dir", default="configs/generated")
    parser.add_argument("--run", action="store_true", help="Run Astra-sim if binary is provided.")
    parser.add_argument("--mock", action="store_true", help="Generate mock output if Astra-sim is unavailable.")
    parser.add_argument(
        "--astra-bin",
        default="astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware",
    )
    parser.add_argument(
        "--remote-mem-config",
        default="astra-sim/examples/remote_memory/analytical/no_memory_expansion.json",
    )
    parser.add_argument("--output", default="results/run_stats.csv")
    parser.add_argument("--log-path", default="results/astra_stdout.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    placement = parse_placement(args.placement)
    num_domains = max(args.num_domains, infer_num_domains(placement, args.num_domains))

    peak_perfs = [float(x) for x in args.peak_perf.split(",") if x.strip()]
    if len(peak_perfs) == 1:
        peak_perfs = peak_perfs * num_domains
    if len(peak_perfs) != num_domains:
        raise ValueError(
            f"peak_perfs length {len(peak_perfs)} must match num_domains {num_domains}. "
            "Use --peak-perf with the correct length."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = TopologyConfig(num_domains=num_domains, topology=args.topology)
    num_npus = num_domains
    network_cfg = generate_network_config_yaml(
        out_dir / "network_config.yml",
        num_npus=num_npus,
        topology=args.topology,
        bandwidth_gbps=topo.inter_bandwidth_gbps,
        latency_ms=topo.inter_latency_ms,
    )
    system_cfg = generate_system_config_v2(
        out_dir / "system_config.json",
        AstraSystemConfig(
            preferred_dataset_splits=num_npus,
            local_mem_bw_gbps=args.local_mem_bw,
            peak_perf_tflops=peak_perfs[0],
        ),
    )
    workload_cfg = generate_workload_et(
        out_dir / "workload",
        placement=placement,
        num_npus=num_npus,
        layer_runtime_us=args.layer_runtime_us,
        comm_size_bytes=args.comm_size_bytes,
        allow_fallback=args.allow_et_fallback,
    )

    output_path = Path(args.output)

    estimated_comm_gb = estimate_comm_volume_gb(placement, args.comm_size_bytes)

    if args.run:
        astra_bin = Path(args.astra_bin)
        if not astra_bin.exists():
            raise FileNotFoundError(f"Astra-sim binary not found: {astra_bin}")
        remote_mem_cfg = Path(args.remote_mem_config)
        if not remote_mem_cfg.exists():
            remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")
        run_astra(
            astra_bin,
            network_cfg,
            system_cfg,
            workload_cfg,
            remote_mem_cfg=remote_mem_cfg,
            output_path=output_path,
            log_path=Path(args.log_path),
            comm_volume_gb=estimated_comm_gb,
        )
    elif args.mock:
        mock_run(output_path, placement, comm_size_bytes=args.comm_size_bytes)

    print("network_config:", network_cfg)
    print("system_config:", system_cfg)
    print("workload_config:", workload_cfg)
    if args.run or args.mock:
        print("output:", output_path)


if __name__ == "__main__":
    main()
