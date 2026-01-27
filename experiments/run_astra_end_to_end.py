"""Run Astra-sim (congestion-aware vs unaware) end-to-end with generated configs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.astra_adapter import (  # noqa: E402
    AstraSystemConfig,
    estimate_comm_volume_gb,
    generate_network_config_yaml,
    generate_remote_memory_config,
    generate_system_config_v2,
    generate_workload_et,
    run_astra,
)


def main() -> None:
    out_dir = Path("configs/astra_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_domains = 4
    num_layers = 64
    placement = [idx % num_domains for idx in range(num_layers)]

    network_cfg = generate_network_config_yaml(
        out_dir / "network_config.yml",
        num_npus=num_domains,
        topology="Ring",
        bandwidth_gbps=1.0,
        latency_ms=80.0,
    )

    system_cfg = generate_system_config_v2(
        out_dir / "system_config.json",
        AstraSystemConfig(
            preferred_dataset_splits=num_domains,
            local_mem_bw_gbps=1600.0,
            peak_perf_tflops=120.0,
            roofline_enabled=1,
        ),
    )

    comm_size_bytes = 128 * 1024 * 1024
    workload_prefix = generate_workload_et(
        out_dir / "workload",
        placement,
        num_npus=num_domains,
        layer_runtime_us=200,
        comm_size_bytes=comm_size_bytes,
    )
    estimated_comm_gb = estimate_comm_volume_gb(placement, comm_size_bytes)

    remote_mem_cfg = generate_remote_memory_config(out_dir / "remote_memory.json")

    unaware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Unaware")
    aware_bin = Path("astra-sim/build/astra_analytical/build_proto/bin/AstraSim_Analytical_Congestion_Aware")

    if not unaware_bin.exists():
        raise FileNotFoundError(f"Binary not found: {unaware_bin}")
    if not aware_bin.exists():
        raise FileNotFoundError(f"Binary not found: {aware_bin}")

    unaware_metrics = run_astra(
        unaware_bin,
        network_cfg,
        system_cfg,
        workload_prefix,
        remote_mem_cfg=remote_mem_cfg,
        output_path=Path("results/astra_unaware.csv"),
        log_path=Path("results/astra_unaware.log"),
        comm_volume_gb=estimated_comm_gb,
    )

    aware_metrics = run_astra(
        aware_bin,
        network_cfg,
        system_cfg,
        workload_prefix,
        remote_mem_cfg=remote_mem_cfg,
        output_path=Path("results/astra_aware.csv"),
        log_path=Path("results/astra_aware.log"),
        comm_volume_gb=estimated_comm_gb,
    )

    print("Unaware metrics:", unaware_metrics)
    print("Aware metrics:", aware_metrics)


if __name__ == "__main__":
    main()
