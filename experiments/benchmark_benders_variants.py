#!/usr/bin/env python3
# experiments/benchmark_benders_variants.py
"""
Benchmark script comparing three Benders methods: Farkas, Pi, Deepest-Cuts.

Protocol (FR-9):
  - Hardware: documented (CPU model, RAM, OS)
  - Solver: Gurobi 13.0.1
  - Time limit: 60,000 s (matching FR-9 / HP25)
  - Seeds: deterministic (Seed=0, Threads=1)
  - Instances: us-night-00008, us-night-00010, us-night-00020
  - Metrics per method: root_lp, final_ip, gap %, time_s, nodes
  - Output: CSV + markdown report with bottleneck analysis

Invocation:
  .venv/bin/python experiments/benchmark_benders_variants.py
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure repo root is in path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.utils import compute_triangles, read_indexed_instance


def get_system_info() -> dict:
    """Capture hardware and environment info per FR-9."""
    try:
        cpu_info = subprocess.check_output("wmic cpu get name", shell=True).decode()
        cpu_model = cpu_info.split("\n")[1].strip() if "\n" in cpu_info else "Unknown"
    except Exception:
        cpu_model = platform.processor()

    try:
        ram_info = subprocess.check_output("wmic MemoryChip get Capacity", shell=True)
        total_ram_bytes = sum(
            int(x) for x in ram_info.decode().split("\n")[1:] if x.strip()
        )
        ram_gb = total_ram_bytes / (1024**3)
    except Exception:
        ram_gb = 0.0

    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "ram_gb": round(ram_gb, 1),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat(),
    }


def run_single_solve(
    instance_path: str,
    method: str,
    time_limit: int = 60000,
) -> dict:
    """
    Run a single solve with one Benders method.

    Args:
        instance_path: Path to .instance file
        method: "farkas", "pi", or "deepest"
        time_limit: Gurobi time limit in seconds

    Returns:
        dict with keys: instance, method, root_lp, final_ip, gap_pct, time_s, nodes, status
    """
    try:
        from models import OAPBendersModel

        logger.info(
            f"Running {method.upper()} on {Path(instance_path).name} (limit={time_limit}s)"
        )

        points = read_indexed_instance(instance_path)
        triangles = compute_triangles(points)

        model = OAPBendersModel(
            points, triangles, name=f"bench-{Path(instance_path).stem}-{method}"
        )

        # Configure based on method
        use_deepest = method == "deepest"
        benders_method = "farkas" if method in ("farkas", "deepest") else "pi"

        model.build(
            objective="Fekete",
            maximize=True,
            benders_method=benders_method,
            use_deepest_cuts=use_deepest,
        )

        # Solve with deterministic settings
        start_time = time.time()
        model.solve(time_limit=time_limit, verbose=False)
        elapsed = time.time() - start_time

        lp, gap, ip, time_s, nodes = model.get_model_stats()

        return {
            "instance": Path(instance_path).stem,
            "method": method.upper(),
            "root_lp": round(lp, 2) if lp is not None else None,
            "final_ip": round(ip, 2) if ip is not None else None,
            "gap_pct": round(gap, 2) if gap is not None else None,
            "time_s": round(time_s, 2) if time_s is not None else None,
            "nodes": int(nodes) if nodes is not None else None,
            "status": "OK",
        }

    except Exception as e:
        logger.error(f"Error on {instance_path} / {method}: {e}")
        return {
            "instance": Path(instance_path).stem,
            "method": method.upper(),
            "root_lp": None,
            "final_ip": None,
            "gap_pct": None,
            "time_s": None,
            "nodes": None,
            "status": f"FAILED: {str(e)[:50]}",
        }


def main():
    """Run benchmark suite and generate reports."""
    # Validate instance files exist
    instances = [
        "instance/us-night-0000008.instance",
        "instance/us-night-0000010.instance",
        "instance/us-night-0000020.instance",
    ]

    missing = [p for p in instances if not Path(p).exists()]
    if missing:
        logger.warning(f"Missing instances: {missing} (skipping)")
        instances = [p for p in instances if Path(p).exists()]

    if not instances:
        logger.error("No instances found; aborting")
        return 1

    logger.info(f"Benchmarking {len(instances)} instances × 3 methods")

    # System info for report
    sys_info = get_system_info()
    logger.info(f"Hardware: {sys_info['cpu_model']} / {sys_info['ram_gb']}GB RAM")
    logger.info(f"Python {sys_info['python_version']}")

    # Run all combinations
    results = []
    methods = ["farkas", "pi", "deepest"]

    for instance_path in instances:
        for method in methods:
            result = run_single_solve(instance_path, method, time_limit=60000)
            results.append(result)

            # Brief pause between solves to avoid thermal throttling
            time.sleep(2)

    # Write CSV output
    csv_path = REPO_ROOT / "outputs" / "CSV" / "benchmark_benders_variants.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instance",
        "method",
        "root_lp",
        "final_ip",
        "gap_pct",
        "time_s",
        "nodes",
        "status",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"CSV output: {csv_path}")

    # Generate markdown report
    report_path = (
        REPO_ROOT / "context" / "notes" / "2026-04-30-cgsp-benchmark-results.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# CGSP Benchmark Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"**Hardware:** {sys_info['cpu_model']} / {sys_info['ram_gb']}GB / {sys_info['platform']}\n"
        )
        f.write(f"**Solver:** Gurobi 13.0.1 (deterministic: Seed=0, Threads=1)\n")
        f.write(f"**Time Limit:** 60,000 s (FR-9 / HP25 protocol)\n\n")

        f.write("## Raw Results\n\n")

        # Group by instance
        for instance in [Path(p).stem for p in instances]:
            f.write(f"### {instance}\n\n")
            f.write("| Method | Root LP | Final IP | Gap % | Time (s) | Nodes | Status |\n")
            f.write("|--------|---------|----------|-------|----------|-------|--------|\n")

            for result in results:
                if result["instance"] == instance:
                    f.write(
                        f"| {result['method']} | {result['root_lp']} | {result['final_ip']} | "
                        f"{result['gap_pct']} | {result['time_s']} | {result['nodes']} | {result['status']} |\n"
                    )

            f.write("\n")

        # Analysis section
        f.write("## Bottleneck Analysis\n\n")

        # Time per method
        time_by_method = {}
        for method in methods:
            times = [
                r["time_s"]
                for r in results
                if r["method"] == method.upper() and r["time_s"] is not None
            ]
            if times:
                time_by_method[method] = {
                    "total": sum(times),
                    "avg": sum(times) / len(times),
                    "max": max(times),
                }

        f.write("### Time per Method\n\n")
        for method, stats in time_by_method.items():
            f.write(
                f"- **{method.upper()}**: {stats['total']:.1f}s total, {stats['avg']:.1f}s avg, {stats['max']:.1f}s max\n"
            )
        f.write("\n")

        # Gap analysis
        f.write("### Gap Analysis\n\n")
        for method in methods:
            gaps = [
                r["gap_pct"]
                for r in results
                if r["method"] == method.upper() and r["gap_pct"] is not None
            ]
            if gaps:
                f.write(
                    f"- **{method.upper()}**: median gap = {sorted(gaps)[len(gaps)//2]:.2f}%, "
                    f"max gap = {max(gaps):.2f}%\n"
                )
        f.write("\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        farkas_time = time_by_method.get("farkas", {}).get("total", 0)
        pi_time = time_by_method.get("pi", {}).get("total", 0)
        deepest_time = time_by_method.get("deepest", {}).get("total", 0)

        if deepest_time > farkas_time * 1.2:
            f.write(
                f"- **CGSP overhead detected**: Deepest-Cuts {deepest_time:.0f}s > Farkas {farkas_time:.0f}s (+{((deepest_time/farkas_time)-1)*100:.0f}%)\n"
            )
        else:
            f.write(
                f"- **CGSP overhead minimal**: Deepest-Cuts {deepest_time:.0f}s ≈ Farkas {farkas_time:.0f}s\n"
            )

        f.write(
            "- Refer to CSV output for detailed metrics per instance and method.\n"
        )

    logger.info(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
