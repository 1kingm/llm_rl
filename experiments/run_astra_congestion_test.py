"""测试 Astra-sim Congestion-Aware vs Unaware 的差异.

问题分析：
=========
原始 run_astra_end_to_end.py 中，Congestion-Aware 和 Unaware 结果相同，原因是：

1. 通信模式是串行的点对点（SEND/RECV），没有并发通信
2. 单维度 Ring 拓扑没有链路竞争
3. 带宽过低导致通信完全主导，拥塞效应可忽略

解决方案：
=========
1. 使用 AllReduce 集合通信（所有 NPU 同时参与）
2. 增加带宽使计算和通信更均衡
3. 使用 Astra-sim 自带的 microbench 工作负载

本脚本对比两种场景：
- 场景 A: 使用自带的 AllReduce microbench（会产生拥塞）
- 场景 B: 使用自定义的串行点对点通信（不会产生拥塞）
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.astra_adapter import (  # noqa: E402
    AstraSystemConfig,
    generate_network_config_yaml,
    generate_remote_memory_config,
    generate_system_config_v2,
    parse_astra_stdout,
)


def run_with_microbench(
    astra_bin: Path,
    network_cfg: Path,
    system_cfg: Path,
    remote_mem_cfg: Path,
    num_npus: int,
    log_path: Path,
) -> dict:
    """使用 Astra-sim 自带的 AllReduce microbench 运行."""
    # 使用 Astra-sim 自带的 microbench 工作负载
    available = [4, 8, 16]
    chosen = next((n for n in available if n >= num_npus), max(available))
    example_dir = ROOT / "astra-sim" / "examples" / "workload" / "microbenchmarks" / "all_reduce" / f"{chosen}npus_1MB"

    if not example_dir.exists():
        raise FileNotFoundError(f"Microbench not found: {example_dir}")

    workload_prefix = example_dir / "all_reduce"

    cmd = [
        str(astra_bin),
        f"--network-configuration={network_cfg}",
        f"--system-configuration={system_cfg}",
        f"--workload-configuration={workload_prefix}",
        f"--remote-memory-configuration={remote_mem_cfg}",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")

    return parse_astra_stdout(combined)


def main() -> None:
    out_dir = Path("configs/congestion_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_npus = 4

    # 配置 1: 高带宽（计算和通信更均衡）
    network_cfg_high_bw = generate_network_config_yaml(
        out_dir / "network_high_bw.yml",
        num_npus=num_npus,
        topology="Ring",
        bandwidth_gbps=100.0,  # 100 Gbps
        latency_ms=0.5,  # 0.5 ms
    )

    # 配置 2: 低带宽（通信主导）
    network_cfg_low_bw = generate_network_config_yaml(
        out_dir / "network_low_bw.yml",
        num_npus=num_npus,
        topology="Ring",
        bandwidth_gbps=1.0,  # 1 Gbps
        latency_ms=80.0,  # 80 ms
    )

    system_cfg = generate_system_config_v2(
        out_dir / "system_config.json",
        AstraSystemConfig(
            preferred_dataset_splits=num_npus,
            local_mem_bw_gbps=1600.0,
            peak_perf_tflops=120.0,
            roofline_enabled=1,
        ),
    )

    remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")

    unaware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware")
    aware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Aware")

    if not unaware_bin.exists():
        raise FileNotFoundError(f"Binary not found: {unaware_bin}")
    if not aware_bin.exists():
        raise FileNotFoundError(f"Binary not found: {aware_bin}")

    results_dir = Path("results/congestion_test")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("场景 1: 高带宽 + AllReduce Microbench")
    print("=" * 60)

    unaware_high = run_with_microbench(
        unaware_bin, network_cfg_high_bw, system_cfg, remote_mem_cfg, num_npus,
        results_dir / "unaware_high_bw.log"
    )
    aware_high = run_with_microbench(
        aware_bin, network_cfg_high_bw, system_cfg, remote_mem_cfg, num_npus,
        results_dir / "aware_high_bw.log"
    )

    print(f"Unaware (高带宽): {unaware_high}")
    print(f"Aware (高带宽):   {aware_high}")

    if unaware_high.get("total_cycles", 0) > 0 and aware_high.get("total_cycles", 0) > 0:
        diff_high = (aware_high["total_cycles"] - unaware_high["total_cycles"]) / unaware_high["total_cycles"] * 100
        print(f"差异: {diff_high:+.2f}%")
    print()

    print("=" * 60)
    print("场景 2: 低带宽 + AllReduce Microbench")
    print("=" * 60)

    unaware_low = run_with_microbench(
        unaware_bin, network_cfg_low_bw, system_cfg, remote_mem_cfg, num_npus,
        results_dir / "unaware_low_bw.log"
    )
    aware_low = run_with_microbench(
        aware_bin, network_cfg_low_bw, system_cfg, remote_mem_cfg, num_npus,
        results_dir / "aware_low_bw.log"
    )

    print(f"Unaware (低带宽): {unaware_low}")
    print(f"Aware (低带宽):   {aware_low}")

    if unaware_low.get("total_cycles", 0) > 0 and aware_low.get("total_cycles", 0) > 0:
        diff_low = (aware_low["total_cycles"] - unaware_low["total_cycles"]) / unaware_low["total_cycles"] * 100
        print(f"差异: {diff_low:+.2f}%")
    print()

    print("=" * 60)
    print("分析")
    print("=" * 60)
    print("""
Congestion-Aware 和 Congestion-Unaware 的区别：

1. Congestion-Unaware:
   - 假设网络链路容量无限
   - 多个流同时使用同一链路时，不考虑带宽竞争
   - 适合快速估算，但可能低估实际通信时间

2. Congestion-Aware:
   - 模拟网络拥塞效应
   - 当多个流竞争同一链路时，带宽被分摊
   - 更接近真实系统行为

差异产生的条件：
- 需要并发通信（如 AllReduce）
- 需要链路竞争（多个流使用同一链路）
- 带宽不能过低（否则拥塞效应被基础延迟淹没）
""")


if __name__ == "__main__":
    main()
