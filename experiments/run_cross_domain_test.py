#!/usr/bin/env python3
"""跨域 LLM 训练测试脚本.

本脚本验证 congestion-aware 和 congestion-unaware 模式在集合通信场景下的差异。

重要限制：
- Astra-sim congestion-aware 后端仅支持 1D 拓扑
- 因此使用 1D Ring 拓扑

测试场景设计：
1. 使用 1D Ring 拓扑 (16 NPU)
2. 使用 All-to-All 集合通信（在 Ring 上产生严重链路竞争）
3. 验证 congestion-aware vs unaware 的差异

为什么使用 All-to-All 而不是 AllReduce：
- Ring AllReduce 在 Ring 拓扑上是最优匹配，不会产生链路竞争
- All-to-All 需要每个 NPU 向所有其他 NPU 发送数据，在 Ring 上会产生严重拥塞

预期结果：
- Congestion-aware 模式应该比 unaware 模式有更长的执行时间
  （因为 aware 模式会考虑拥塞，可能选择更保守的调度）
- 或者 aware 模式通过更好的调度减少拥塞
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.astra_adapter import (
    AstraSystemConfig,
    estimate_alltoall_comm_volume_gb,
    generate_alltoall_workload,
    generate_remote_memory_config,
    generate_system_config_v2,
    run_astra,
)


@dataclass
class TestConfig:
    """测试配置."""

    num_npus: int = 16  # NPU 数量
    topology: str = "Ring"  # 拓扑类型 (1D)
    data_size_mb: int = 64  # AllReduce 数据量 (MB)
    num_iterations: int = 5  # 迭代次数
    compute_us: int = 1000  # 每次迭代计算时间 (微秒)
    bandwidth_gbs: float = 25.0  # 带宽 (GB/s)
    latency_ns: float = 10_000_000.0  # 延迟 (ns) = 10 ms


@dataclass
class TestResult:
    """测试结果."""

    name: str
    total_cycles: float
    comm_cycles: float
    utilization: float
    comm_volume_gb: float


def generate_network_config_1d(
    out_path: Path,
    num_npus: int,
    topology: str,
    bandwidth_gbs: float,
    latency_ns: float,
) -> Path:
    """生成 1D 网络拓扑配置.

    Args:
        out_path: 输出路径
        num_npus: NPU 数量
        topology: 拓扑类型 (Ring, Switch, FullyConnected)
        bandwidth_gbs: 带宽 (GB/s)
        latency_ns: 延迟 (ns)

    Returns:
        配置文件路径
    """
    payload = (
        f"# 1D {topology} 拓扑 - 用于 congestion-aware 测试\n"
        f"topology: [ {topology} ]\n"
        f"npus_count: [ {num_npus} ]\n"
        f"bandwidth: [ {bandwidth_gbs} ]\n"
        f"latency: [ {latency_ns} ]\n"
    )
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def run_test(
    config: TestConfig,
    out_dir: Path,
    astra_bin: Path,
    test_name: str,
) -> TestResult:
    """运行单个测试.

    Args:
        config: 测试配置
        out_dir: 输出目录
        astra_bin: Astra-sim 二进制路径
        test_name: 测试名称

    Returns:
        测试结果
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成 1D 网络配置（congestion-aware 仅支持 1D）
    network_cfg = generate_network_config_1d(
        out_dir / "network_config.yml",
        num_npus=config.num_npus,
        topology=config.topology,
        bandwidth_gbs=config.bandwidth_gbs,
        latency_ns=config.latency_ns,
    )

    # 生成系统配置
    system_cfg = generate_system_config_v2(
        out_dir / "system_config.json",
        AstraSystemConfig(
            preferred_dataset_splits=config.num_npus,
            local_mem_bw_gbps=1600.0,
            peak_perf_tflops=120.0,
            roofline_enabled=1,
        ),
    )

    # 生成 All-to-All 工作负载（在 Ring 上产生链路竞争）
    data_size_bytes = config.data_size_mb * 1024 * 1024
    workload_prefix = generate_alltoall_workload(
        out_dir / "workload",
        num_npus=config.num_npus,
        data_size_bytes=data_size_bytes,
        num_iterations=config.num_iterations,
        compute_us=config.compute_us,
        allow_fallback=True,  # 允许回退到示例 ET
    )

    # 估算通信量
    estimated_comm_gb = estimate_alltoall_comm_volume_gb(
        num_npus=config.num_npus,
        data_size_bytes=data_size_bytes,
        num_iterations=config.num_iterations,
    )

    # 生成远程内存配置
    remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")

    # 运行 Astra-sim
    output_path = out_dir / f"{test_name}_results.csv"
    log_path = out_dir / f"{test_name}.log"

    metrics = run_astra(
        astra_bin,
        network_cfg,
        system_cfg,
        workload_prefix,
        remote_mem_cfg=remote_mem_cfg,
        output_path=output_path,
        log_path=log_path,
        comm_volume_gb=estimated_comm_gb,
    )

    return TestResult(
        name=test_name,
        total_cycles=metrics.get("total_cycles", 0.0),
        comm_cycles=metrics.get("comm_cycles", 0.0),
        utilization=metrics.get("utilization", 0.0),
        comm_volume_gb=estimated_comm_gb,
    )


def print_results(results: List[TestResult]) -> None:
    """打印测试结果."""
    print("\n" + "=" * 80)
    print("跨域 LLM 训练测试结果")
    print("=" * 80)

    # 表头
    print(f"{'测试名称':<30} {'总周期数':>15} {'通信周期':>15} {'利用率':>10} {'通信量(GB)':>12}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.name:<30} "
            f"{result.total_cycles:>15,.0f} "
            f"{result.comm_cycles:>15,.0f} "
            f"{result.utilization:>10.2%} "
            f"{result.comm_volume_gb:>12.2f}"
        )

    print("-" * 80)


def analyze_results(unaware: TestResult, aware: TestResult) -> None:
    """分析 congestion-aware vs unaware 的差异."""
    print("\n" + "=" * 80)
    print("Congestion-Aware vs Unaware 分析")
    print("=" * 80)

    if unaware.total_cycles > 0:
        cycle_diff = (unaware.total_cycles - aware.total_cycles) / unaware.total_cycles * 100
        print(f"总周期数差异: {cycle_diff:+.2f}% (正值表示 aware 更优)")
    else:
        print("总周期数差异: N/A (unaware 周期数为 0)")

    if unaware.comm_cycles > 0:
        comm_diff = (unaware.comm_cycles - aware.comm_cycles) / unaware.comm_cycles * 100
        print(f"通信周期差异: {comm_diff:+.2f}% (正值表示 aware 更优)")
    else:
        print("通信周期差异: N/A (unaware 通信周期为 0)")

    if aware.utilization > 0:
        util_diff = (aware.utilization - unaware.utilization) / aware.utilization * 100
        print(f"利用率差异: {util_diff:+.2f}% (正值表示 aware 更优)")
    else:
        print("利用率差异: N/A (aware 利用率为 0)")

    # 判断是否有显著差异
    if unaware.total_cycles > 0 and abs(cycle_diff) > 5:
        print("\n[结论] Congestion-aware 和 unaware 模式有显著差异 (>5%)")
    elif unaware.total_cycles > 0:
        print("\n[警告] Congestion-aware 和 unaware 模式差异不显著 (<5%)")
        print("可能原因:")
        print("  1. 工作负载通信量不足")
        print("  2. 网络拓扑未产生足够的链路竞争")
        print("  3. 计算时间过长，掩盖了通信差异")
    else:
        print("\n[警告] 无法计算差异 (unaware 周期数为 0)")


def main() -> None:
    """主函数."""
    # 检查 Astra-sim 二进制
    unaware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware")
    aware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Aware")

    if not unaware_bin.exists():
        print(f"[错误] Astra-sim Congestion-Unaware 二进制不存在: {unaware_bin}")
        print("请先构建 Astra-sim:")
        print("  cd astra-sim && ./build/astra_analytical/build.sh")
        sys.exit(1)

    if not aware_bin.exists():
        print(f"[错误] Astra-sim Congestion-Aware 二进制不存在: {aware_bin}")
        print("请先构建 Astra-sim:")
        print("  cd astra-sim && ./build/astra_analytical/build.sh")
        sys.exit(1)

    # 测试配置
    # 注意: congestion-aware 仅支持 1D 拓扑，因此使用 Ring
    config = TestConfig(
        num_npus=16,  # 16 个 NPU
        topology="Ring",  # 1D Ring 拓扑
        data_size_mb=64,  # 64 MB AllReduce
        num_iterations=5,
        compute_us=1000,  # 1 ms 计算
        bandwidth_gbs=25.0,  # 25 GB/s (200 Gbps)
        latency_ns=10_000_000.0,  # 10 ms
    )

    print("=" * 80)
    print("跨域 LLM 训练测试 (1D Ring + All-to-All)")
    print("=" * 80)
    print(f"配置:")
    print(f"  - NPU 数量: {config.num_npus}")
    print(f"  - 拓扑类型: {config.topology} (1D)")
    print(f"  - All-to-All 数据量: {config.data_size_mb} MB")
    print(f"  - 迭代次数: {config.num_iterations}")
    print(f"  - 计算时间: {config.compute_us} µs")
    print(f"  - 带宽: {config.bandwidth_gbs} GB/s")
    print(f"  - 延迟: {config.latency_ns / 1e6:.1f} ms")
    print()
    print("注意: Congestion-aware 后端仅支持 1D 拓扑")
    print("      使用 All-to-All 在 Ring 上创建链路竞争以产生拥塞")

    out_dir = Path("results/cross_domain_test")
    results: List[TestResult] = []

    # 运行 Congestion-Unaware 测试
    print("\n[1/2] 运行 Congestion-Unaware 测试...")
    try:
        unaware_result = run_test(
            config,
            out_dir / "unaware",
            unaware_bin,
            "congestion_unaware",
        )
        results.append(unaware_result)
        print(f"  完成: {unaware_result.total_cycles:,.0f} 周期")
    except Exception as e:
        print(f"  [错误] {e}")
        unaware_result = TestResult(
            name="congestion_unaware",
            total_cycles=0,
            comm_cycles=0,
            utilization=0,
            comm_volume_gb=0,
        )
        results.append(unaware_result)

    # 运行 Congestion-Aware 测试
    print("\n[2/2] 运行 Congestion-Aware 测试...")
    try:
        aware_result = run_test(
            config,
            out_dir / "aware",
            aware_bin,
            "congestion_aware",
        )
        results.append(aware_result)
        print(f"  完成: {aware_result.total_cycles:,.0f} 周期")
    except Exception as e:
        print(f"  [错误] {e}")
        aware_result = TestResult(
            name="congestion_aware",
            total_cycles=0,
            comm_cycles=0,
            utilization=0,
            comm_volume_gb=0,
        )
        results.append(aware_result)

    # 打印结果
    print_results(results)

    # 分析差异
    analyze_results(unaware_result, aware_result)

    print("\n" + "=" * 80)
    print(f"详细日志保存在: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
