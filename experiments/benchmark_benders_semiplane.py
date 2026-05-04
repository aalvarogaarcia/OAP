#!/usr/bin/env python3
# experiments/benchmark_benders_semiplane.py
"""
Benchmark: Basic theory — Compact vs Benders-Farkas vs Benders-Pi
(semiplane=0, no CGSP) on the CG:SHOP-2019 instance set, sizes 10/15/20/25/30/35.

Protocol (FR-9 compliant):
  - Hardware: logged via get_system_info()
  - Solver: Gurobi 13.0.1, Seed=0, Threads=1
  - MIPGapAbs: 1.99
  - Time limit per solve: 60,000 s (FR-9 / HP25 protocol)
  - Skip rule: any (instance, method) exceeding 10 h wall-clock is recorded
    as status=SKIPPED_TIMEOUT_10H and the script continues.
  - Instance selection: euro-night, london, stars, us-night (sizes 10-35),
    uniform-1, uniform-2 (sizes 10-35) — 36 instances total.
  - Instances sorted smallest-first, stars-* last within each size group.

Metrics per row:
  instance, n_nodes, family, model, benders_method,
  root_lp, final_ip, gap_pct, time_s, nodes, n_master_constrs, status

Output:
  outputs/CSV/benchmark_basic_theory.csv

Profiling output:
  outputs/profiling/profile_{method}_{stem}.csv    (per-instance)
  outputs/profiling/profile_summary_{method}.csv   (aggregated top-20)

Invocation:
  .venv/bin/python experiments/benchmark_benders_semiplane.py [--smoke-test]

  --smoke-test: runs on little-instances/ with time_limit=60 to validate the
                script before the real campaign.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import logging
import platform
import pstats
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

METHODS: list[dict[str, Any]] = [
    {"label": "compact",        "model": "compact",  "benders_method": None},
    {"label": "benders_farkas", "model": "benders",  "benders_method": "farkas"},
]

CSV_FIELDNAMES = [
    "instance", "n_nodes", "family", "size", "method_label",
    "model", "benders_method",
    "root_lp", "final_ip", "gap_pct", "time_s", "nodes",
    "n_master_constrs", "status", "timestamp",
]

PROFILE_FIELDNAMES = [
    "ncalls", "tottime", "percall_tot", "cumtime", "percall_cum",
    "filename", "lineno", "function",
]
PROFILE_SUMMARY_FIELDNAMES = [
    "function", "filename", "lineno",
    "total_ncalls", "total_tottime", "total_cumtime", "n_instances",
]
PROFILING_OUT_DIR = REPO_ROOT / "outputs" / "profiling"
RESULTS_CSV_PATH  = REPO_ROOT / "outputs" / "CSV" / "benchmark_basic_theory.csv"


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
        "mipgapabs": MIPGAPABS,
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


def _is_project_func(filename: str) -> bool:
    return str(REPO_ROOT) in filename


def _dump_profile_csv(
    profiler: cProfile.Profile,
    method_label: str,
    stem: str,
    out_dir: Path,
) -> list[dict]:
    """Dump per-instance profiling data to CSV; return rows for summary accumulation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.calc_callees()

    rows = []
    for (filename, lineno, funcname), (prim_calls, total_calls, tottime, cumtime, _callers) in stats.stats.items():
        if not _is_project_func(filename):
            continue
        row = {
            "ncalls":      total_calls,
            "tottime":     round(tottime, 6),
            "percall_tot": round(tottime / total_calls, 6) if total_calls else 0.0,
            "cumtime":     round(cumtime, 6),
            "percall_cum": round(cumtime / prim_calls, 6) if prim_calls else 0.0,
            "filename":    Path(filename).name,
            "lineno":      lineno,
            "function":    funcname,
        }
        rows.append(row)

    rows.sort(key=lambda r: r["cumtime"], reverse=True)

    out_path = out_dir / f"profile_{method_label}_{stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROFILE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Profile CSV written: %s (%d functions)", out_path, len(rows))
    return rows


def _write_profile_summary(accumulated: dict[str, list[dict]], out_dir: Path) -> None:
    """Aggregate per-instance profile rows by function and write summary CSVs."""
    for method_label, all_rows in accumulated.items():
        agg: dict[tuple, dict] = {}
        for row in all_rows:
            key = (row["function"], row["filename"], row["lineno"])
            if key not in agg:
                agg[key] = {
                    "function":      row["function"],
                    "filename":      row["filename"],
                    "lineno":        row["lineno"],
                    "total_ncalls":  0,
                    "total_tottime": 0.0,
                    "total_cumtime": 0.0,
                    "n_instances":   0,
                }
            agg[key]["total_ncalls"]  += row["ncalls"]
            agg[key]["total_tottime"] += row["tottime"]
            agg[key]["total_cumtime"] += row["cumtime"]
            agg[key]["n_instances"]   += 1

        summary_rows = sorted(agg.values(), key=lambda r: r["total_cumtime"], reverse=True)[:20]
        for r in summary_rows:
            r["total_tottime"] = round(r["total_tottime"], 6)
            r["total_cumtime"] = round(r["total_cumtime"], 6)

        out_path = out_dir / f"profile_summary_{method_label}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=PROFILE_SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Profile summary written: %s", out_path)


def run_compact_solve(instance_path: Path, time_limit: int) -> tuple[dict, list[dict]]:
    """Run OAPCompactModel. Returns (result_row, profile_rows)."""
    from models import OAPCompactModel
    from utils.utils import compute_triangles, read_indexed_instance

    stem = instance_path.stem
    label = "compact"
    timestamp = datetime.now().isoformat()
    logger.info("[%s] Compact starting (limit=%ds)", stem, time_limit)

    try:
        points = read_indexed_instance(str(instance_path))
        triangles = compute_triangles(points)
        n_nodes = len(points)

        profiler = cProfile.Profile()
        profiler.enable()

        model = OAPCompactModel(points, triangles, name=f"bench-{stem}-compact")
        model.build(
            objective="Fekete",
            maximize=False,
            subtour="SCF",
            sum_constrain=True,
            strengthen=False,
            semiplane=0,
            use_knapsack=False,
            use_cliques=False,
        )
        model.model.Params.Seed = 0
        model.model.Params.Threads = 1
        n_master_constrs = model.model.NumConstrs

        wall_start = time.time()
        model.solve(time_limit=time_limit, verbose=False)
        wall_elapsed = time.time() - wall_start

        profiler.disable()

        lp, gap, ip, time_s, nodes = model.get_model_stats()
        status = "OK"
        if wall_elapsed >= SKIP_WALL_CLOCK_S:
            status = "SKIPPED_TIMEOUT_10H"

        logger.info("[%s] Compact: ip=%s gap=%s time=%.1fs", stem, ip, gap, time_s or wall_elapsed)

        profile_rows = _dump_profile_csv(profiler, label, stem, PROFILING_OUT_DIR)

        return _make_row(stem, n_nodes, label, "compact", None, lp, ip, gap,
                         time_s or wall_elapsed, nodes, n_master_constrs, status, timestamp), profile_rows

    except Exception as exc:
        logger.error("[%s] Compact FAILED: %s", stem, exc)
        return _make_row(stem, parse_n_nodes(instance_path.stem), label, "compact", None,
                         None, None, None, None, None, None, f"FAILED: {str(exc)[:80]}", timestamp), []


def run_benders_solve(
    instance_path: Path, benders_method: str, time_limit: int
) -> tuple[dict, list[dict]]:
    """Run OAPBendersModel. Returns (result_row, profile_rows)."""
    from models import OAPBendersModel
    from utils.utils import compute_triangles, read_indexed_instance

    stem = instance_path.stem
    label = f"benders_{benders_method}"
    timestamp = datetime.now().isoformat()
    logger.info("[%s] Benders(%s) starting (limit=%ds)", stem, benders_method, time_limit)

    try:
        points = read_indexed_instance(str(instance_path))
        triangles = compute_triangles(points)
        n_nodes = len(points)

        profiler = cProfile.Profile()
        profiler.enable()

        model = OAPBendersModel(points, triangles, name=f"bench-{stem}-{benders_method}")
        model.build(
            objective="Fekete",
            maximize=False,
            benders_method=benders_method,
            sum_constrain=True,
            crosses_constrain=False,
            strengthen=False,
            use_deepest_cuts=False,
            semiplane=0,
        )
        model.model.Params.Seed = 0
        model.model.Params.Threads = 1
        model.model.Params.MIPGapAbs = MIPGAPABS
        n_master_constrs = model.model.NumConstrs

        wall_start = time.time()
        model.solve(time_limit=time_limit, verbose=False, save_cuts=False)
        wall_elapsed = time.time() - wall_start

        profiler.disable()

        lp, gap, ip, time_s, nodes = model.get_model_stats()
        status = "OK"
        if wall_elapsed >= SKIP_WALL_CLOCK_S:
            status = "SKIPPED_TIMEOUT_10H"

        logger.info("[%s] Benders(%s): ip=%s gap=%s time=%.1fs", stem, benders_method, ip, gap, time_s or wall_elapsed)

        profile_rows = _dump_profile_csv(profiler, label, stem, PROFILING_OUT_DIR)

        return _make_row(stem, n_nodes, label, "benders", benders_method, lp, ip, gap,
                         time_s or wall_elapsed, nodes, n_master_constrs, status, timestamp), profile_rows

    except Exception as exc:
        logger.error("[%s] Benders(%s) FAILED: %s", stem, benders_method, exc)
        return _make_row(stem, parse_n_nodes(instance_path.stem), label, "benders", benders_method,
                         None, None, None, None, None, None, f"FAILED: {str(exc)[:80]}", timestamp), []


def _make_row(
    stem: str,
    n_nodes: int,
    label: str,
    model: str,
    benders_method: str | None,
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
        "root_lp":   round(root_lp,  4) if isinstance(root_lp,  (int, float)) else "",
        "final_ip":  round(final_ip, 4) if isinstance(final_ip, (int, float)) else "",
        "gap_pct":   round(gap_pct,  4) if isinstance(gap_pct,  (int, float)) else "",
        "time_s":    round(time_s,   2) if isinstance(time_s,   (int, float)) else "",
        "nodes":     int(nodes) if nodes is not None else "",
        "n_master_constrs": int(n_master_constrs) if n_master_constrs is not None else "",
        "status": status,
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
    logger.info("Benchmark: Basic theory (Compact vs Farkas vs Pi, semiplane=0, no CGSP)")
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
    profile_accumulated: dict[str, list[dict]] = {}

    for inst_path in instances:
        logger.info("")
        logger.info("--- Instance: %s ---", inst_path.stem)
        for method_cfg in METHODS:
            m_model   = method_cfg["model"]
            m_benders = method_cfg["benders_method"]
            m_label   = method_cfg["label"]

            if m_model == "compact":
                row, prof_rows = run_compact_solve(inst_path, time_limit)
            else:
                row, prof_rows = run_benders_solve(inst_path, m_benders, time_limit)

            all_rows.append(row)
            profile_accumulated.setdefault(m_label, []).extend(prof_rows)

            # Incremental write
            write_csv(all_rows, RESULTS_CSV_PATH)

    logger.info("Benchmark complete. Total solves: %d", len(all_rows))
    ok      = sum(1 for r in all_rows if r["status"] == "OK")
    failed  = sum(1 for r in all_rows if r["status"].startswith("FAILED"))
    timeout = sum(1 for r in all_rows if r["status"] == "SKIPPED_TIMEOUT_10H")
    logger.info("  OK: %d  FAILED: %d  TIMEOUT-10H: %d", ok, failed, timeout)

    _write_profile_summary(profile_accumulated, PROFILING_OUT_DIR)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic theory benchmark (Compact vs Farkas vs Pi, semiplane=0)")
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
