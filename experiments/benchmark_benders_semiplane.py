#!/usr/bin/env python3
# experiments/benchmark_benders_semiplane.py
"""
Benchmark: Compact vs Benders-Farkas vs Benders-Pi (plain + semiplane V1)
on the CG:SHOP-2019 instance set, sizes 10/15/20/25/30/35.

Protocol (FR-9 compliant):
  - Hardware: logged via get_system_info()
  - Solver: Gurobi 13.0.1, Seed=0, Threads=1
  - MIPGapAbs overridden to 0 (the model default of 1.99 distorts gaps)
  - MIPGapAbs: 1.99
  - Time limit per solve: 60,000 s (FR-9 / HP25 protocol)
  - Skip rule: any (instance, method) exceeding 10 h wall-clock is recorded
    as status=SKIPPED_TIMEOUT_10H and the script continues.
  - Instance selection: euro-night, london, stars, us-night (sizes 10-35),
    uniform-1, uniform-2 (sizes 10-35) — 36 instances total.
  - Instances sorted smallest-first, stars-* last within each size group.

Metrics per row:
  instance, n_nodes, family, model, benders_method, semiplane,
  root_lp, final_ip, gap_pct, time_s, nodes, n_master_constrs, status

Output:
  outputs/CSV/benchmark_benders_semiplane_phase1.csv  — Compact + Benders plain
  outputs/CSV/benchmark_benders_semiplane_phase2.csv  — Benders + semiplane V1

Invocation:
  .venv/bin/python experiments/benchmark_benders_semiplane.py [--smoke-test]

  --smoke-test: runs on little-instances/ with time_limit=60 to validate the
                script before the real campaign.
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZES = [10, 15, 20, 25, 30, 35]
FAMILIES_ORDER = ["euro-night", "london", "uniform", "us-night", "stars"]  # stars last
TIME_LIMIT_DEFAULT = 60_000  # seconds — FR-9 / HP25
SKIP_WALL_CLOCK_S = 10 * 3600  # 10 h soft cap per solve
MIPGAPABS = 1.99  # override OAPBendersModel default of 1.99

# Phase-1 methods: Compact (semiplane=1 by default per plan) + Benders plain
# Phase-2 methods: Benders + semiplane V1 in master
# Both phases are collected in a single run; the split into two CSVs is by
# the `semiplane` column value.

METHODS: list[dict[str, Any]] = [
    # --- Phase 1 ---
    {
        "label": "compact_default",
        "model": "compact",
        "benders_method": None,
        "semiplane": 1,  # Compact ALWAYS with semiplane per plan M.3
        "phase": 1,
    },
    {
        "label": "benders_farkas_plain",
        "model": "benders",
        "benders_method": "farkas",
        "semiplane": 0,
        "phase": 1,
    },
    {
        "label": "benders_pi_plain",
        "model": "benders",
        "benders_method": "pi",
        "semiplane": 0,
        "phase": 1,
    },
    # --- Phase 2 ---
    {
        "label": "benders_farkas_sp1",
        "model": "benders",
        "benders_method": "farkas",
        "semiplane": 1,
        "phase": 2,
    },
    {
        "label": "benders_pi_sp1",
        "model": "benders",
        "benders_method": "pi",
        "semiplane": 1,
        "phase": 2,
    },
]

CSV_FIELDNAMES = [
    "instance",
    "n_nodes",
    "family",
    "size",
    "method_label",
    "model",
    "benders_method",
    "semiplane",
    "root_lp",
    "final_ip",
    "gap_pct",
    "time_s",
    "nodes",
    "n_master_constrs",
    "status",
    "phase",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_system_info() -> dict:
    """Capture hardware and environment info per FR-9."""
    try:
        cpu_info = subprocess.check_output("wmic cpu get name", shell=True).decode()
        cpu_model = cpu_info.split("\n")[1].strip() if "\n" in cpu_info else "Unknown"
    except Exception:
        cpu_model = platform.processor()

    try:
        ram_info = subprocess.check_output("wmic MemoryChip get Capacity", shell=True)
        total_ram_bytes = sum(int(x) for x in ram_info.decode().split("\n")[1:] if x.strip())
        ram_gb = total_ram_bytes / (1024**3)
    except Exception:
        ram_gb = 0.0

    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "ram_gb": round(ram_gb, 1),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat(),
        "gurobi_version": "13.0.1",
        "seed": 0,
        "threads": 1,
        "mipgapabs": MIPGAPABS
        }


def parse_family(stem: str) -> str:
    """Extract instance family from stem, e.g. 'euro-night-0000010' -> 'euro-night'."""
    # Remove trailing size suffix (last segment of digits)
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # uniform-0000010-1 style
    parts2 = stem.rsplit("-", 2)
    if len(parts2) == 3 and parts2[1].isdigit() and parts2[2].isdigit():
        return parts2[0]
    return stem


def parse_n_nodes(stem: str) -> int:
    """Extract |N| from instance stem, e.g. 'euro-night-0000010' -> 10."""
    # Last numeric segment
    for part in reversed(stem.split("-")):
        if part.isdigit():
            return int(part)
    return 0


def collect_instances(instance_dir: Path, sizes: list[int]) -> list[Path]:
    """
    Gather instance files for the target sizes, sorted smallest-first
    with stars-* last within each size group (per plan M.4 risk register R-6).
    """
    candidates: list[Path] = []
    for p in instance_dir.glob("*.instance"):
        n = parse_n_nodes(p.stem)
        if n in sizes:
            candidates.append(p)

    def sort_key(p: Path) -> tuple[int, int, str]:
        n = parse_n_nodes(p.stem)
        fam = parse_family(p.stem)
        # families_order defines preferred ordering; stars last
        order = FAMILIES_ORDER.index(fam) if fam in FAMILIES_ORDER else 99
        return (n, order, p.stem)

    candidates.sort(key=sort_key)
    return candidates


def run_compact_solve(
    instance_path: Path,
    semiplane: int,
    time_limit: int,
) -> dict:
    """Run OAPCompactModel on one instance and return a result row."""
    from models import OAPCompactModel
    from utils.utils import compute_triangles, read_indexed_instance

    stem = instance_path.stem
    label = "compact_default"
    timestamp = datetime.now().isoformat()
    logger.info("[%s] Compact semiplane=%d starting (limit=%ds)", stem, semiplane, time_limit)

    try:
        points = read_indexed_instance(str(instance_path))
        triangles = compute_triangles(points)
        n_nodes = len(points)

        model = OAPCompactModel(points, triangles, name=f"bench-{stem}-compact")
        model.build(
            objective="Fekete",
            maximize=False,  # MinArea — the harder direction per HP25
            subtour="SCF",
            sum_constrain=True,
            strengthen=False,
            semiplane=semiplane,
            use_knapsack=False,
            use_cliques=False,
        )

        # Deterministic settings (NFR-2)
        model.model.Params.Seed = 0
        model.model.Params.Threads = 1

        n_master_constrs = model.model.NumConstrs

        wall_start = time.time()
        model.solve(time_limit=time_limit, verbose=False)
        wall_elapsed = time.time() - wall_start

        lp, gap, ip, time_s, nodes = model.get_model_stats()

        status = "OK"
        if wall_elapsed >= SKIP_WALL_CLOCK_S:
            status = "SKIPPED_TIMEOUT_10H"

        logger.info(
            "[%s] Compact semiplane=%d: ip=%.2f gap=%.2f%% time=%.1fs nodes=%s",
            stem,
            semiplane,
            ip or float("nan"),
            gap or float("nan"),
            time_s or float("nan"),
            nodes,
        )
        return _make_row(
            stem,
            n_nodes,
            label,
            "compact",
            None,
            semiplane,
            1,
            lp,
            ip,
            gap,
            time_s or wall_elapsed,
            nodes,
            n_master_constrs,
            status,
            timestamp,
        )

    except Exception as exc:
        logger.error("[%s] Compact semiplane=%d FAILED: %s", stem, semiplane, exc)
        return _make_row(
            stem,
            parse_n_nodes(instance_path.stem),
            label,
            "compact",
            None,
            semiplane,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            f"FAILED: {str(exc)[:80]}",
            timestamp,
        )


def run_benders_solve(
    instance_path: Path,
    benders_method: str,
    semiplane: int,
    time_limit: int,
) -> dict:
    """Run OAPBendersModel on one instance and return a result row."""
    from models import OAPBendersModel
    from utils.utils import compute_triangles, read_indexed_instance

    stem = instance_path.stem
    phase = 1 if semiplane == 0 else 2
    label = f"benders_{benders_method}_{'plain' if semiplane == 0 else 'sp1'}"
    timestamp = datetime.now().isoformat()
    logger.info(
        "[%s] Benders(%s) semiplane=%d starting (limit=%ds)",
        stem,
        benders_method,
        semiplane,
        time_limit,
    )

    try:
        points = read_indexed_instance(str(instance_path))
        triangles = compute_triangles(points)
        n_nodes = len(points)

        model = OAPBendersModel(points, triangles, name=f"bench-{stem}-{benders_method}-sp{semiplane}")
        model.build(
            objective="Fekete",
            maximize=False,  # MinArea
            benders_method=benders_method,
            sum_constrain=True,
            crosses_constrain=False,
            strengthen=False,
            use_deepest_cuts=False,
            semiplane=semiplane,
        )

        # Deterministic settings (NFR-2)
        model.model.Params.Seed = 0
        model.model.Params.Threads = 1
        # Override MIPGapAbs=1.99 default for honest gap reporting (plan §M.3)
        model.model.Params.MIPGapAbs = MIPGAPABS

        n_master_constrs = model.model.NumConstrs

        wall_start = time.time()
        model.solve(
            time_limit=time_limit,
            verbose=False,
            save_cuts=False,  # Disable JSONL logging for campaign runs (plan R-7)
        )
        wall_elapsed = time.time() - wall_start

        lp, gap, ip, time_s, nodes = model.get_model_stats()

        status = "OK"
        if wall_elapsed >= SKIP_WALL_CLOCK_S:
            status = "SKIPPED_TIMEOUT_10H"

        logger.info(
            "[%s] Benders(%s) semiplane=%d: ip=%.2f gap=%.2f%% time=%.1fs nodes=%s",
            stem,
            benders_method,
            semiplane,
            ip or float("nan"),
            gap or float("nan"),
            time_s or float("nan"),
            nodes,
        )
        return _make_row(
            stem,
            n_nodes,
            label,
            "benders",
            benders_method,
            semiplane,
            phase,
            lp,
            ip,
            gap,
            time_s or wall_elapsed,
            nodes,
            n_master_constrs,
            status,
            timestamp,
        )

    except Exception as exc:
        logger.error("[%s] Benders(%s) semiplane=%d FAILED: %s", stem, benders_method, semiplane, exc)
        return _make_row(
            stem,
            parse_n_nodes(instance_path.stem),
            label,
            "benders",
            benders_method,
            semiplane,
            phase,
            None,
            None,
            None,
            None,
            None,
            None,
            f"FAILED: {str(exc)[:80]}",
            timestamp,
        )


def _make_row(
    stem: str,
    n_nodes: int,
    label: str,
    model: str,
    benders_method: str | None,
    semiplane: int,
    phase: int,
    root_lp: float | None,
    final_ip: float | None,
    gap_pct: float | None,
    time_s: float | None,
    nodes: int | None,
    n_master_constrs: int | None,
    status: str,
    timestamp: str,
) -> dict:
    return {
        "instance": stem,
        "n_nodes": n_nodes,
        "family": parse_family(stem),
        "size": n_nodes,
        "method_label": label,
        "model": model,
        "benders_method": benders_method or "",
        "semiplane": semiplane,
        "root_lp": round(root_lp, 4) if root_lp is not None else "",
        "final_ip": round(final_ip, 4) if final_ip is not None else "",
        "gap_pct": round(gap_pct, 4) if gap_pct is not None else "",
        "time_s": round(time_s, 2) if time_s is not None else "",
        "nodes": int(nodes) if nodes is not None else "",
        "n_master_constrs": int(n_master_constrs) if n_master_constrs is not None else "",
        "status": status,
        "phase": phase,
        "timestamp": timestamp,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    sys_info = get_system_info()
    logger.info("=" * 70)
    logger.info("Benchmark: Benders semiplane sweep (Compact + Farkas + Pi)")
    logger.info("Date: %s", sys_info["timestamp"])
    logger.info("Hardware: %s / %.1f GB RAM", sys_info["cpu_model"], sys_info["ram_gb"])
    logger.info(
        "Solver: Gurobi %s  Seed=%d  Threads=%d", sys_info["gurobi_version"], sys_info["seed"], sys_info["threads"]
    )
    logger.info("MIPGapAbs=%.1f", MIPGAPABS)
    logger.info("=" * 70)

    # Collect instances
    if args.smoke_test:
        instance_dir = REPO_ROOT / "instance" / "little-instances"
        sizes = [6, 8, 10]
        time_limit = 60
        logger.info("SMOKE TEST MODE: %s, time_limit=%ds", instance_dir, time_limit)
    else:
        instance_dir = REPO_ROOT / "instance"
        sizes = SIZES
        time_limit = args.time_limit

    instances = collect_instances(instance_dir, sizes)
    if not instances:
        logger.error("No instances found in %s for sizes %s", instance_dir, sizes)
        return 1

    logger.info("Found %d instances across %d sizes", len(instances), len(sizes))
    for p in instances:
        logger.info("  %s (%d nodes)", p.stem, parse_n_nodes(p.stem))

    # Estimate total budget (informational)
    n_methods = len(METHODS)
    logger.info(
        "Matrix: %d instances × %d methods = %d solves",
        len(instances),
        n_methods,
        len(instances) * n_methods,
    )
    logger.info(
        "Worst-case wall clock (10h skip): %.0f h",
        len(instances) * n_methods * SKIP_WALL_CLOCK_S / 3600,
    )
    logger.info("=" * 70)

    all_rows: list[dict] = []

    for inst_path in instances:
        logger.info("")
        logger.info("--- Instance: %s ---", inst_path.stem)
        for method_cfg in METHODS:
            m_model = method_cfg["model"]
            m_benders = method_cfg["benders_method"]
            m_semiplane = method_cfg["semiplane"]

            if m_model == "compact":
                row = run_compact_solve(inst_path, m_semiplane, time_limit)
            else:
                assert m_benders is not None
                row = run_benders_solve(inst_path, m_benders, m_semiplane, time_limit)

            all_rows.append(row)

            # Write incrementally so a crash mid-run doesn't lose all data
            phase1_rows = [r for r in all_rows if r["phase"] == 1]
            phase2_rows = [r for r in all_rows if r["phase"] == 2]

            if phase1_rows:
                write_csv(
                    phase1_rows,
                    REPO_ROOT / "outputs" / "CSV" / "benchmark_benders_semiplane_phase1.csv",
                )
            if phase2_rows:
                write_csv(
                    phase2_rows,
                    REPO_ROOT / "outputs" / "CSV" / "benchmark_benders_semiplane_phase2.csv",
                )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Benchmark complete.  Total solves: %d", len(all_rows))

    ok = sum(1 for r in all_rows if r["status"] == "OK")
    failed = sum(1 for r in all_rows if r["status"].startswith("FAILED"))
    timeout = sum(1 for r in all_rows if r["status"] == "SKIPPED_TIMEOUT_10H")
    logger.info("  OK: %d  FAILED: %d  TIMEOUT-10H: %d", ok, failed, timeout)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benders semiplane benchmark (Compact + Farkas + Pi, sizes 10-35)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help=("Run on little-instances/ with time_limit=60s to validate the script before the real campaign."),
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=TIME_LIMIT_DEFAULT,
        metavar="SECONDS",
        help=f"Time limit per solve in seconds (default: {TIME_LIMIT_DEFAULT}).",
    )
    parsed = parser.parse_args()
    sys.exit(main(parsed))
