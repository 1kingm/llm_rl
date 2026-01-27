#!/usr/bin/env python3
"""跨域 LLM 训练测试脚本 (NS3 后端, 2D 拓扑).

本脚本使用 NS3 网络后端运行 2D 多域拓扑：
- 维度 0: 域内 (gpus_per_domain)
- 维度 1: 域间 (num_domains)

注意：NS3 需要先编译，且 network config 内的相对路径依赖工作目录。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.astra_adapter import (
    AstraSystemConfig,
    estimate_allreduce_comm_volume_gb,
    find_ns3_binary,
    generate_allreduce_workload,
    generate_logical_topology_json,
    generate_remote_memory_config,
    generate_system_config_v2,
    run_astra_ns3,
)


@dataclass
class TestConfig:
    num_domains: int = 4
    gpus_per_domain: int = 4
    data_size_mb: int = 64
    num_iterations: int = 5
    compute_us: int = 1000


@dataclass
class TestResult:
    name: str
    total_cycles: float
    comm_cycles: float
    utilization: float
    comm_volume_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-domain test with NS3 backend (2D topology).")
    parser.add_argument("--out-dir", default="results/cross_domain_test_ns3", help="Output directory")
    parser.add_argument(
        "--ns3-build-dir",
        default="astra-sim/extern/network_backend/ns-3-src/build/scratch",
        help="ns-3 build/scratch directory",
    )
    parser.add_argument(
        "--ns3-network-config",
        default="astra-sim/extern/network_backend/ns-3-src/scratch/config/config_clos.txt",
        help="ns-3 network config file",
    )
    parser.add_argument("--num-domains", type=int, default=4)
    parser.add_argument("--gpus-per-domain", type=int, default=4)
    parser.add_argument("--data-size-mb", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--compute-us", type=int, default=1000)
    return parser.parse_args()


def run_test(config: TestConfig, out_dir: Path, ns3_bin: Path, network_cfg: Path) -> TestResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_npus = config.num_domains * config.gpus_per_domain

    system_cfg = generate_system_config_v2(
        out_dir / "system_config.json",
        AstraSystemConfig(
            preferred_dataset_splits=total_npus,
            local_mem_bw_gbps=1600.0,
            peak_perf_tflops=120.0,
            roofline_enabled=1,
        ),
    )

    data_size_bytes = config.data_size_mb * 1024 * 1024
    workload_prefix = generate_allreduce_workload(
        out_dir / "workload",
        num_npus=total_npus,
        data_size_bytes=data_size_bytes,
        num_iterations=config.num_iterations,
        compute_us=config.compute_us,
        allow_fallback=True,
    )

    logical_topology_cfg = generate_logical_topology_json(
        out_dir / "logical_topology.json",
        [config.gpus_per_domain, config.num_domains],
    )

    estimated_comm_gb = estimate_allreduce_comm_volume_gb(
        num_npus=total_npus,
        data_size_bytes=data_size_bytes,
        num_iterations=config.num_iterations,
    )

    remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")

    output_path = out_dir / "ns3_results.csv"
    log_path = out_dir / "ns3.log"

    metrics = run_astra_ns3(
        ns3_bin,
        network_cfg,
        system_cfg,
        workload_prefix,
        logical_topology_cfg,
        remote_mem_cfg=remote_mem_cfg,
        output_path=output_path,
        log_path=log_path,
        comm_volume_gb=estimated_comm_gb,
    )

    return TestResult(
        name="ns3_backend",
        total_cycles=metrics.get("total_cycles", 0.0),
        comm_cycles=metrics.get("comm_cycles", 0.0),
        utilization=metrics.get("utilization", 0.0),
        comm_volume_gb=estimated_comm_gb,
    )


def print_result(result: TestResult) -> None:
    print("\n" + "=" * 80)
    print("NS3 2D 拓扑测试结果")
    print("=" * 80)
    print(f"测试名称: {result.name}")
    print(f"总周期数: {result.total_cycles:,.0f}")
    print(f"通信周期: {result.comm_cycles:,.0f}")
    print(f"利用率: {result.utilization:.2%}")
    print(f"通信量(GB): {result.comm_volume_gb:.2f}")
    if result.total_cycles == 0:
        print("[提示] 未解析到指标，建议查看 ns3.log 确认输出格式。")


def main() -> None:
    args = parse_args()

    config = TestConfig(
        num_domains=args.num_domains,
        gpus_per_domain=args.gpus_per_domain,
        data_size_mb=args.data_size_mb,
        num_iterations=args.num_iterations,
        compute_us=args.compute_us,
    )

    out_dir = Path(args.out_dir)
    ns3_build_dir = Path(args.ns3_build_dir)
    network_cfg = Path(args.ns3_network_config)

    if not network_cfg.exists():
        print(f"[错误] NS3 network config 不存在: {network_cfg}")
        sys.exit(1)

    try:
        ns3_bin = find_ns3_binary(ns3_build_dir)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        print("请先编译 NS3 后端:")
        print("  cd astra-sim/build/astra_ns3 && ./build.sh")
        sys.exit(1)

    print("=" * 80)
    print("跨域 LLM 训练测试 (NS3, 2D 拓扑)")
    print("=" * 80)
    print("配置:")
    print(f"  - 域数量: {config.num_domains}")
    print(f"  - 每域 GPU: {config.gpus_per_domain}")
    print(f"  - 总 NPU: {config.num_domains * config.gpus_per_domain}")
    print(f"  - AllReduce 数据量: {config.data_size_mb} MB")
    print(f"  - 迭代次数: {config.num_iterations}")
    print(f"  - 计算时间: {config.compute_us} µs")
    print(f"  - NS3 network config: {network_cfg}")
    print(f"  - NS3 binary: {ns3_bin}")

    try:
        result = run_test(config, out_dir, ns3_bin, network_cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] NS3 运行失败: {exc}")
        sys.exit(1)

    print_result(result)
    print("\n" + "=" * 80)
    print(f"详细日志保存在: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
