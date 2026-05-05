#!/usr/bin/env python3
# experiments/benchmark_benders_general.py
"""
General benchmark comparing five Benders cut strategies:

  farkas       — standard Farkas cuts (baseline)
  cgsp_farkas  — deepest cuts (CGSP) with Farkas dual base
  cgsp_pi      — deepest cuts (CGSP) with Pi dual base
  mw_lp        — Magnanti-Wong Pareto-optimal cuts, core-point from LP relaxation
  mw_uniform   — Magnanti-Wong Pareto-optimal cuts, uniform core-point

Protocol (FR-9):
  - Solver: Gurobi 13.0.1
  - Seeds: deterministic (Seed=0, Threads=1)
  - Time limit: configurable (default 60,000 s)
  - Metrics: root_lp, final_ip, gap_pct, time_s, nodes, status

Invocation:
  # Interactive (inquirer prompts):
  .venv/bin/python experiments/benchmark_benders_general.py

  # Headless from JSON config:
  .venv/bin/python experiments/benchmark_benders_general.py --config config.json

  # Quick smoke-test (little-instances/, 60 s, all methods):
  .venv/bin/python experiments/benchmark_benders_general.py --smoke-test

Outputs (all timestamped with run_id = YYYYMMDD_HHMMSS):
  outputs/CSV/benchmark_general_{run_id}.csv
  outputs/CSV/benchmark_general_{run_id}_config.json
  outputs/reports/benchmark_general_{run_id}.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# Each entry maps a short label to the kwargs forwarded to model.build().
# Keys deliberately match OAPBendersModel.build() parameter names.
METHOD_CONFIG: dict[str, dict] = {
    "farkas": dict(
        benders_method="farkas",
        use_deepest_cuts=False,
        use_magnanti_wong=False,
    ),
    "cgsp_farkas": dict(
        benders_method="farkas",
        use_deepest_cuts=True,
        use_magnanti_wong=False,
    ),
    "cgsp_pi": dict(
        benders_method="pi",
        use_deepest_cuts=True,
        use_magnanti_wong=False,
    ),
    "mw_lp": dict(
        benders_method="farkas",
        use_deepest_cuts=False,
        use_magnanti_wong=True,
        core_point_strategy="lp_relaxation",
    ),
    "mw_uniform": dict(
        benders_method="farkas",
        use_deepest_cuts=False,
        use_magnanti_wong=True,
        core_point_strategy="uniform",
    ),
    # F3 — DDMA (Algorithm 3, Hosseini & Turner 2025 §4.1)
    "ddma_farkas": dict(
        benders_method="farkas",
        use_deepest_cuts=False,
        use_magnanti_wong=False,
        use_ddma=True,
    ),
    "ddma_pi": dict(
        benders_method="pi",
        use_deepest_cuts=False,
        use_magnanti_wong=False,
        use_ddma=True,
    ),
}

ALL_METHODS = list(METHOD_CONFIG.keys())

CSV_FIELDNAMES = [
    "instance",
    "n_nodes",
    "method",
    "root_lp",
    "final_ip",
    "gap_pct",
    "time_s",
    "nodes",
    "status",
]

# ---------------------------------------------------------------------------
# System info
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
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _safe_round(value, ndigits: int):
    """Round numeric values; pass through anything else (None, "-", strings) as None.

    F1.2 (audit ref. .claude/context/reviews/2026-05-04-cgsp-paper-dissonance.md §2.4):
    ``OAPStatsMixin.get_model_stats`` returns the literal string ``"-"`` when
    the manual LP-relaxation Benders loop fails to converge (or when SolCount
    is zero).  The previous implementation called ``round("-", 4)`` which
    raised ``TypeError: type str doesn't define __round__ method`` and made
    the whole method appear ``FAILED`` in the report — masking the real
    diagnostic, which is LP non-convergence inside CGSP.
    """
    if isinstance(value, (int, float)):
        return round(value, ndigits)
    return None


def run_single_solve(
    instance_path: str,
    method: str,
    objective: str = "Fekete",
    maximize: bool = True,
    time_limit: int = 60000,
) -> dict:
    """
    Build and solve one OAPBendersModel instance with the given method.

    Returns a dict with CSV_FIELDNAMES keys.
    """
    stem = Path(instance_path).stem
    row: dict = {
        "instance": stem,
        "n_nodes": None,
        "method": method,
        "root_lp": None,
        "final_ip": None,
        "gap_pct": None,
        "time_s": None,
        "nodes": None,
        "status": "FAILED",
    }

    try:
        from models import OAPBendersModel
        from utils.utils import compute_triangles, read_indexed_instance

        logger.info(
            "  [%s] %s  (limit=%ds, obj=%s, max=%s)",
            method,
            stem,
            time_limit,
            objective,
            maximize,
        )

        points = read_indexed_instance(instance_path)
        triangles = compute_triangles(points)

        row["n_nodes"] = len(points)

        model = OAPBendersModel(
            points,
            triangles,
            name=f"bench-{stem}-{method}",
        )

        build_kwargs = dict(METHOD_CONFIG[method])  # shallow copy
        model.build(
            objective=objective,
            maximize=maximize,
            **build_kwargs,
        )

        model.solve(time_limit=time_limit, verbose=False)

        lp, gap, ip, time_s, nodes = model.get_model_stats()

        row.update(
            {
                "root_lp": _safe_round(lp, 4),
                "final_ip": _safe_round(ip, 4),
                "gap_pct": _safe_round(gap, 4),
                "time_s": _safe_round(time_s, 2),
                "nodes": int(nodes) if isinstance(nodes, (int, float)) else None,
                "status": "OK",
            }
        )

    except Exception as exc:
        # F1.3 — emit the full traceback so failures don't get truncated to a
        # single line; without this the only diagnostic in the benchmark log is
        # ``FAILED: <type>: <first 80 chars>``, which is what hid the real
        # ``round("-")`` origin of the previous run.
        import traceback
        logger.error(
            "  FAILED [%s] %s\n%s",
            method,
            stem,
            traceback.format_exc(),
        )
        row["status"] = f"FAILED: {type(exc).__name__}: {str(exc)[:80]}"

    return row


# ---------------------------------------------------------------------------
# Interactive configuration
# ---------------------------------------------------------------------------

def prompt_config() -> dict:
    """Collect benchmark configuration via inquirer prompts."""
    try:
        import inquirer
    except ImportError:
        logger.error("inquirer is not installed. Run: pip install inquirer")
        sys.exit(1)

    # --- Step 1: instance directory + glob ---
    dir_answers = inquirer.prompt(
        [
            inquirer.Text(
                "instance_dir",
                message="Instance directory",
                default="instance",
            ),
            inquirer.Text(
                "glob",
                message="File glob pattern",
                default="*.instance",
            ),
        ]
    )
    if dir_answers is None:
        sys.exit(0)

    instance_dir = Path(dir_answers["instance_dir"])
    glob_pattern = dir_answers["glob"]

    found = sorted(instance_dir.glob(glob_pattern))
    if not found:
        logger.error(
            "No files matching '%s' in '%s'", glob_pattern, instance_dir
        )
        sys.exit(1)

    # --- Step 2: instance selection ---
    inst_answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "instances",
                message="Select instances to benchmark (space to toggle, enter to confirm)",
                choices=[p.name for p in found],
                default=[p.name for p in found],
            ),
        ]
    )
    if inst_answers is None or not inst_answers["instances"]:
        logger.error("No instances selected.")
        sys.exit(1)

    selected_instances = [
        str(instance_dir / name) for name in inst_answers["instances"]
    ]

    # --- Step 3: method selection ---
    # Use (label, value) tuples so the user sees a human-readable description
    # in the checkbox while the stored value is the METHOD_CONFIG key.
    _METHOD_LABELS: dict[str, str] = {
        "farkas":       "farkas          — Benders Farkas (baseline)",
        "cgsp_farkas":  "cgsp_farkas     — Deepest cuts via CGSP, Farkas-mode  (Hosseini & Turner §3)",
        "cgsp_pi":      "cgsp_pi         — Deepest cuts via CGSP, Pi-mode",
        "mw_lp":        "mw_lp           — Magnanti-Wong Pareto-optimal, core point = LP relax",
        "mw_uniform":   "mw_uniform      — Magnanti-Wong Pareto-optimal, core point = uniform",
        "ddma_farkas":  "ddma_farkas     — DDMA Algorithm 3, Farkas-mode  [Hosseini & Turner §4.1]",
        "ddma_pi":      "ddma_pi         — DDMA Algorithm 3, Pi-mode      [Hosseini & Turner §4.1]",
    }
    method_choices = [(_METHOD_LABELS.get(m, m), m) for m in ALL_METHODS]
    method_answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "methods",
                message="Select cut strategies to benchmark  (space = toggle, enter = confirm)",
                choices=method_choices,
                default=[label for label, _ in method_choices],
            ),
        ]
    )
    if method_answers is None or not method_answers["methods"]:
        logger.error("No methods selected.")
        sys.exit(1)

    selected_methods = method_answers["methods"]

    # --- Step 4: solver options ---
    solver_answers = inquirer.prompt(
        [
            inquirer.List(
                "objective",
                message="Objective function",
                choices=["Fekete", "Internal", "External", "Diagonals"],
                default="Fekete",
            ),
            inquirer.Confirm(
                "maximize",
                message="Maximize objective?",
                default=True,
            ),
            inquirer.Text(
                "time_limit",
                message="Time limit per solve (seconds)",
                default="60000",
                validate=lambda _, v: v.isdigit() and int(v) > 0,
            ),
        ]
    )
    if solver_answers is None:
        sys.exit(0)

    return {
        "instances": selected_instances,
        "methods": selected_methods,
        "objective": solver_answers["objective"],
        "maximize": solver_answers["maximize"],
        "time_limit": int(solver_answers["time_limit"]),
    }


def smoke_test_config() -> dict:
    """Return hardcoded config for --smoke-test mode."""
    smoke_dir = REPO_ROOT / "little-instances"
    if not smoke_dir.exists():
        logger.error("little-instances/ directory not found at %s", smoke_dir)
        sys.exit(1)

    instances = sorted(smoke_dir.glob("*.instance"))
    if not instances:
        logger.error("No .instance files found in %s", smoke_dir)
        sys.exit(1)

    logger.info("Smoke-test mode: %d instances in %s", len(instances), smoke_dir)
    return {
        "instances": [str(p) for p in instances],
        "methods": ALL_METHODS,
        "objective": "Fekete",
        "maximize": True,
        "time_limit": 60,
    }


def load_json_config(path: str) -> dict:
    """Load benchmark configuration from a JSON file."""
    with open(path) as f:
        cfg = json.load(f)

    required = {"instances", "methods", "objective", "maximize", "time_limit"}
    missing = required - set(cfg.keys())
    if missing:
        logger.error("Config JSON missing keys: %s", missing)
        sys.exit(1)

    unknown_methods = set(cfg["methods"]) - set(ALL_METHODS)
    if unknown_methods:
        logger.error("Unknown methods in config: %s", unknown_methods)
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV: %s", path)


def write_config_snapshot(cfg: dict, sys_info: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {"config": cfg, "system": sys_info}
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info("Config snapshot: %s", path)


def write_report(
    results: list[dict],
    path: Path,
    sys_info: dict,
    cfg: dict,
    run_id: str,
) -> None:
    """Write a markdown report with raw tables, time/gap analysis, and conclusions."""
    path.parent.mkdir(parents=True, exist_ok=True)

    methods: list[str] = cfg["methods"]
    instances_order = list(dict.fromkeys(r["instance"] for r in results))

    def _fmt(v) -> str:
        return str(v) if v is not None else "—"

    with open(path, "w", encoding="utf-8") as f:

        # --- Header ---
        f.write("# Benders Variants General Benchmark\n\n")
        f.write(f"**Run ID:** `{run_id}`  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(
            f"**Hardware:** {sys_info['cpu_model']} / "
            f"{sys_info['ram_gb']} GB / {sys_info['platform']}  \n"
        )
        f.write("**Solver:** Gurobi 13.0.1 (deterministic: Seed=0, Threads=1)  \n")
        f.write(
            f"**Protocol:** FR-9 — time limit {cfg['time_limit']} s, "
            f"objective={cfg['objective']}, maximize={cfg['maximize']}  \n"
        )
        f.write(f"**Methods:** {', '.join(methods)}  \n\n")

        f.write("---\n\n")

        # --- Method descriptions ---
        f.write("## Method Descriptions\n\n")
        f.write("| Label | benders_method | use_deepest_cuts | use_magnanti_wong | use_ddma | core_point_strategy |\n")
        f.write("|-------|---------------|-----------------|------------------|----------|--------------------|\n")
        for label, kwargs in METHOD_CONFIG.items():
            if label not in methods:
                continue
            bm = kwargs.get("benders_method", "—")
            dc = kwargs.get("use_deepest_cuts", False)
            mw = kwargs.get("use_magnanti_wong", False)
            ddma = kwargs.get("use_ddma", False)
            cp = kwargs.get("core_point_strategy", "—")
            f.write(f"| `{label}` | {bm} | {dc} | {mw} | {ddma} | {cp} |\n")
        f.write("\n")

        # --- Raw results per instance ---
        f.write("## Raw Results\n\n")
        for inst in instances_order:
            f.write(f"### {inst}\n\n")
            f.write(
                "| Method | n_nodes | Root LP | Final IP | Gap (%) | Time (s) | Nodes | Status |\n"
            )
            f.write(
                "|--------|---------|---------|----------|---------|----------|-------|--------|\n"
            )
            for r in results:
                if r["instance"] != inst:
                    continue
                f.write(
                    f"| `{r['method']}` | {_fmt(r['n_nodes'])} | {_fmt(r['root_lp'])} | "
                    f"{_fmt(r['final_ip'])} | {_fmt(r['gap_pct'])} | "
                    f"{_fmt(r['time_s'])} | {_fmt(r['nodes'])} | {r['status']} |\n"
                )
            f.write("\n")

        # --- Time analysis ---
        f.write("## Time Analysis\n\n")
        f.write("| Method | Total (s) | Avg (s) | Max (s) | Solved |\n")
        f.write("|--------|----------|---------|---------|--------|\n")
        time_by_method: dict[str, dict] = {}
        for method in methods:
            times = [
                r["time_s"]
                for r in results
                if r["method"] == method and r["time_s"] is not None
            ]
            solved = len(times)
            if times:
                stats = {
                    "total": sum(times),
                    "avg": sum(times) / solved,
                    "max": max(times),
                    "solved": solved,
                }
            else:
                stats = {"total": 0.0, "avg": 0.0, "max": 0.0, "solved": 0}
            time_by_method[method] = stats
            f.write(
                f"| `{method}` | {stats['total']:.2f} | {stats['avg']:.2f} | "
                f"{stats['max']:.2f} | {stats['solved']} |\n"
            )
        f.write("\n")

        # --- Gap analysis ---
        f.write("## Gap Analysis\n\n")
        f.write("| Method | Median Gap (%) | Max Gap (%) | Solved |\n")
        f.write("|--------|---------------|------------|--------|\n")
        for method in methods:
            gaps = [
                r["gap_pct"]
                for r in results
                if r["method"] == method and r["gap_pct"] is not None
            ]
            if gaps:
                sorted_gaps = sorted(gaps)
                median = sorted_gaps[len(sorted_gaps) // 2]
                f.write(
                    f"| `{method}` | {median:.4f} | {max(gaps):.4f} | {len(gaps)} |\n"
                )
            else:
                f.write(f"| `{method}` | — | — | 0 |\n")
        f.write("\n")

        # --- Conclusions ---
        f.write("## Conclusions\n\n")
        baseline_time = time_by_method.get("farkas", {}).get("total", 0.0)

        def _overhead_line(label: str, label_time: float, baseline: float) -> str:
            if baseline <= 0:
                return f"- `{label}`: baseline unavailable for comparison.\n"
            ratio = label_time / baseline
            pct = (ratio - 1.0) * 100
            if ratio > 1.2:
                return (
                    f"- `{label}` is **{pct:.0f}% slower** than `farkas` "
                    f"({label_time:.1f}s vs {baseline:.1f}s).\n"
                )
            elif ratio < 0.8:
                return (
                    f"- `{label}` is **{abs(pct):.0f}% faster** than `farkas` "
                    f"({label_time:.1f}s vs {baseline:.1f}s).\n"
                )
            else:
                return (
                    f"- `{label}` is roughly equivalent to `farkas` "
                    f"({label_time:.1f}s vs {baseline:.1f}s, {pct:+.0f}%).\n"
                )

        for method in methods:
            if method == "farkas":
                continue
            t = time_by_method.get(method, {}).get("total", 0.0)
            f.write(_overhead_line(method, t, baseline_time))

        f.write(
            "\n> Refer to the CSV for per-instance detail and the config JSON "
            "for full reproducibility metadata.\n"
        )

    logger.info("Report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="General Benders variants benchmark (farkas / cgsp_farkas / cgsp_pi / mw_lp / mw_uniform)"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick validation: little-instances/, all methods, 60 s time limit",
    )
    mode_group.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a JSON config file (headless mode)",
    )
    args = parser.parse_args()

    # ---- Build configuration ----
    if args.smoke_test:
        cfg = smoke_test_config()
    elif args.config:
        cfg = load_json_config(args.config)
    else:
        cfg = prompt_config()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    sys_info = get_system_info()
    logger.info("Hardware: %s / %.1f GB RAM", sys_info["cpu_model"], sys_info["ram_gb"])
    logger.info("Python %s", sys_info["python_version"])

    instances: list[str] = cfg["instances"]
    methods: list[str] = cfg["methods"]
    objective: str = cfg["objective"]
    maximize: bool = cfg["maximize"]
    time_limit: int = cfg["time_limit"]

    # Validate instance files
    missing = [p for p in instances if not Path(p).exists()]
    if missing:
        logger.warning("Missing instance files (skipping): %s", missing)
        instances = [p for p in instances if Path(p).exists()]
    if not instances:
        logger.error("No valid instance files found. Aborting.")
        return 1

    total_runs = len(instances) * len(methods)
    logger.info(
        "Starting: %d instance(s) × %d method(s) = %d solves",
        len(instances), len(methods), total_runs,
    )

    # ---- Run all combinations ----
    results: list[dict] = []
    completed = 0

    for instance_path in instances:
        for method in methods:
            completed += 1
            logger.info(
                "Solve %d/%d — instance=%s  method=%s",
                completed,
                total_runs,
                Path(instance_path).stem,
                method,
            )
            row = run_single_solve(
                instance_path=instance_path,
                method=method,
                objective=objective,
                maximize=maximize,
                time_limit=time_limit,
            )
            results.append(row)
            # Brief pause to avoid thermal throttling between solves
            if completed < total_runs:
                time.sleep(2)

    # ---- Write outputs ----
    out_csv = REPO_ROOT / "outputs" / "CSV" / f"benchmark_general_{run_id}.csv"
    out_cfg = REPO_ROOT / "outputs" / "CSV" / f"benchmark_general_{run_id}_config.json"
    out_rpt = REPO_ROOT / "outputs" / "reports" / f"benchmark_general_{run_id}.md"

    write_csv(results, out_csv)
    write_config_snapshot(cfg, sys_info, out_cfg)
    write_report(results, out_rpt, sys_info, cfg, run_id)

    # ---- Quick summary to stdout ----
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - ok_count
    logger.info("Done. %d OK / %d failed.", ok_count, fail_count)

    if fail_count:
        failed = [(r["instance"], r["method"], r["status"]) for r in results if r["status"] != "OK"]
        for inst, meth, status in failed:
            logger.warning("  FAILED  %-30s  %-14s  %s", inst, meth, status)

    return 0


if __name__ == "__main__":
    sys.exit(main())
