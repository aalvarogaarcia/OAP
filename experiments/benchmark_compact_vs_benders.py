#!/usr/bin/env python3
"""
Benchmark: OAPCompactModel vs 3 OAPBendersModel variants.

Compares:
  1. OAPCompactModel (Threads=1)
  2. OAPBendersModel with farkas (vanilla)
  3. OAPBendersModel with Magnanti-Wong (LP core point)
  4. OAPBendersModel with DDMA (Algorithm 3)

All with sum_constrain=True, objective=Fekete, run over both maximize=True and maximize=False.

Supports:
  - Interactive configuration (instances, models, time limit, profiling)
  - Smoke test (quick validation)
  - Headless mode (JSON config)
  - Profiling (cProfile, per-instance and summary CSVs)
  - Markdown report with comparative analysis
"""

import argparse
import cProfile
import csv
import json
import logging
import pstats
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import OAPBendersModel, OAPCompactModel  # noqa: E402
from utils.utils import compute_triangles, read_indexed_instance  # noqa: E402

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

METHOD_CONFIG: dict[str, dict[str, Any]] = {
    "compact_threads1": {
        "model_type": "compact",
        "threads": 1,
        "label": "OAPCompactModel (Threads=1)",
        "build_kwargs": {
            "objective": "Fekete",
            "subtour": "SCF",
            "sum_constrain": True,
            "strengthen": False,
            "semiplane": 0,
            "use_knapsack": False,
            "use_cliques": False,
        },
    },
    "benders_farkas_vanilla": {
        "model_type": "benders",
        "threads": 1,
        "label": "Benders Farkas (Vanilla)",
        "build_kwargs": {
            "objective": "Fekete",
            "benders_method": "farkas",
            "subtour": "SCF",
            "sum_constrain": True,
            "strengthen": False,
            "crosses_constrain": False,
            "use_magnanti_wong": False,
            "use_deepest_cuts": False,
            "use_ddma": False,
            "semiplane": 0,
            "use_knapsack": False,
            "use_cliques": False,
        },
    },
    "benders_mw_lp": {
        "model_type": "benders",
        "threads": 1,
        "label": "Benders Magnanti-Wong (LP core)",
        "build_kwargs": {
            "objective": "Fekete",
            "benders_method": "farkas",
            "subtour": "SCF",
            "sum_constrain": True,
            "strengthen": False,
            "crosses_constrain": False,
            "use_magnanti_wong": True,
            "core_point_strategy": "lp_relaxation",
            "use_deepest_cuts": False,
            "use_ddma": False,
            "semiplane": 0,
            "use_knapsack": False,
            "use_cliques": False,
        },
    },
    "benders_ddma": {
        "model_type": "benders",
        "threads": 1,
        "label": "Benders DDMA (Algorithm 3)",
        "build_kwargs": {
            "objective": "Fekete",
            "benders_method": "farkas",
            "subtour": "SCF",
            "sum_constrain": True,
            "strengthen": False,
            "crosses_constrain": False,
            "use_magnanti_wong": False,
            "use_deepest_cuts": False,
            "use_ddma": True,
            "semiplane": 0,
            "use_knapsack": False,
            "use_cliques": False,
        },
    },
}

CSV_FIELDNAMES = [
    "instance",
    "n_nodes",
    "method",
    "maximize",
    "root_lp",
    "final_ip",
    "gap_pct",
    "time_s",
    "nodes",
    "total_cuts",
    "status",
]

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    dir_path: str
    ext: str
    instances: list[str]
    methods: list[str]
    time_limit: int
    enable_profiling: bool


@dataclass
class ResultRow:
    """Single result row (instance, method, maximize)."""

    instance: str
    n_nodes: int | None = None
    method: str = ""
    maximize: bool = True
    root_lp: float | None = None
    final_ip: float | None = None
    gap_pct: float | None = None
    time_s: float | None = None
    nodes: int | None = None
    total_cuts: int | None = None  # Benders cuts added during solve; 0 for Compact
    status: str = "PENDING"


# ============================================================================
# UTILITIES
# ============================================================================


def _safe_round(val: Any, decimals: int) -> str | float | int:
    """Round a value or return as-is if sentinel."""
    if val is None or val == "-":
        return "-"
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return "-"


def get_system_info() -> dict[str, str]:
    """Capture system info for reproducibility."""
    import platform

    return {
        "platform": platform.system(),
        "platform_version": platform.release(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }


def _ensure_dir(path: Path | str) -> None:
    """Create parent directory if needed."""
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================


def prompt_config() -> BenchmarkConfig:
    """Interactive configuration via inquirer."""
    import inquirer
    logger.info("Starting interactive configuration...")

    # Step 1: Instance directory and glob
    questions_step1 = [
        inquirer.Text(
            "dir_path",
            message="Instance directory",
            default="instance",
            validate=lambda _, x: Path(x).is_dir() or f"Directory {x} not found",
        ),
        inquirer.Text("ext", message="File glob pattern", default="*.instance"),
    ]
    config1 = inquirer.prompt(questions_step1)
    if config1 is None:
        sys.exit(0)

    dir_path = config1["dir_path"]
    ext = config1["ext"]

    # Discover instances
    instances = sorted(Path(dir_path).glob(ext))
    if not instances:
        logger.error("No instances found in %s matching %s", dir_path, ext)
        sys.exit(1)

    instance_names = [str(p) for p in instances]

    # Step 2: Select instances
    questions_step2 = [
        inquirer.Checkbox(
            "instances",
            message="Select instances to benchmark",
            choices=instance_names,
            default=instance_names,
        ),
    ]
    config2 = inquirer.prompt(questions_step2)
    if config2 is None:
        sys.exit(0)

    selected_instances = config2["instances"]

    # Step 3: Select methods
    method_choices = [
        f"{key}: {METHOD_CONFIG[key]['label']}" for key in METHOD_CONFIG.keys()
    ]
    method_keys = list(METHOD_CONFIG.keys())

    questions_step3 = [
        inquirer.Checkbox(
            "methods",
            message="Select models to benchmark",
            choices=method_choices,
            default=method_choices,
        ),
    ]
    config3 = inquirer.prompt(questions_step3)
    if config3 is None:
        sys.exit(0)

    selected_labels = config3["methods"]
    selected_methods = [
        method_keys[method_choices.index(label)] for label in selected_labels
    ]

    # Step 4: Solver options
    questions_step4 = [
        inquirer.Text("time_limit", message="Time limit (s)", default="7200"),
        inquirer.Confirm("enable_profiling", message="Enable profiling?", default=False),
    ]
    config4 = inquirer.prompt(questions_step4)
    if config4 is None:
        sys.exit(0)

    return BenchmarkConfig(
        dir_path=dir_path,
        ext=ext,
        instances=selected_instances,
        methods=selected_methods,
        time_limit=int(config4["time_limit"]),
        enable_profiling=config4["enable_profiling"],
    )


def smoke_test_config() -> BenchmarkConfig:
    """Quick smoke test configuration."""
    logger.info("Smoke test mode: 2 small instances, all methods, 60s limit")
    little_instances_dir = REPO_ROOT / "instance" / "little-instances"
    if not little_instances_dir.exists():
        logger.warning("little-instances dir not found, using main instance dir")
        instances_dir = REPO_ROOT / "instance"
    else:
        instances_dir = little_instances_dir

    instances = sorted(instances_dir.glob("*.instance"))[:2]
    if not instances:
        logger.error("No instances found for smoke test")
        sys.exit(1)

    return BenchmarkConfig(
        dir_path=str(instances_dir),
        ext="*.instance",
        instances=[str(p) for p in instances],
        methods=list(METHOD_CONFIG.keys()),
        time_limit=60,
        enable_profiling=False,
    )


def load_json_config(path: str) -> BenchmarkConfig:
    """Load configuration from JSON file."""
    logger.info("Loading config from %s", path)
    with open(path, "r") as f:
        cfg_dict = json.load(f)

    return BenchmarkConfig(
        dir_path=cfg_dict["dir_path"],
        ext=cfg_dict["ext"],
        instances=cfg_dict["instances"],
        methods=cfg_dict["methods"],
        time_limit=cfg_dict["time_limit"],
        enable_profiling=cfg_dict.get("enable_profiling", False),
    )


# ============================================================================
# SOLVER
# ============================================================================


def run_single_solve(
    instance_path: str,
    method: str,
    maximize: bool,
    time_limit: int = 7200,
    enable_profiling: bool = False,
) -> tuple[ResultRow, list[dict[str, Any]]]:
    """
    Run a single solve (instance, method, maximize).

    Returns (result_row, profile_rows).
    profile_rows = [] if profiling disabled.
    """
    stem = Path(instance_path).stem
    method_cfg = METHOD_CONFIG[method]
    model_type = method_cfg["model_type"]

    row = ResultRow(
        instance=stem,
        method=method,
        maximize=maximize,
    )
    profile_rows: list[dict[str, Any]] = []

    try:
        # Load instance
        points = read_indexed_instance(instance_path)
        triangles = compute_triangles(points)
        row.n_nodes = len(points)

        # Instantiate model
        if model_type == "compact":
            model = OAPCompactModel(points, triangles, name=f"bench-{stem}-{method}")
            model.build(
                maximize=maximize,
                **method_cfg["build_kwargs"],
            )
        else:  # benders
            model = OAPBendersModel(points, triangles, name=f"bench-{stem}-{method}")
            model.build(
                maximize=maximize,
                **method_cfg["build_kwargs"],
            )

        # Set Gurobi parameters (deterministic)
        model.model.Params.Seed = 0
        model.model.Params.Threads = 1
        model.model.Params.TimeLimit = time_limit

        # Profile if requested
        profiler = None
        if enable_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        # Snapshot constraint count before solve to measure Benders cuts added
        num_constrs_before = model.model.NumConstrs

        # Measure wall-clock time
        wall_start = time.perf_counter()
        model.solve(time_limit=time_limit, verbose=False)
        wall_elapsed = time.perf_counter() - wall_start

        # Stop profiler before stats extraction
        if profiler:
            profiler.disable()

        # Extract stats
        lp, gap, ip, gurobi_time_s, nodes = model.get_model_stats()

        row.root_lp = _safe_round(lp, 4)
        row.final_ip = _safe_round(ip, 4)
        row.gap_pct = _safe_round(gap, 4)
        row.time_s = _safe_round(wall_elapsed, 2)
        row.nodes = int(nodes) if isinstance(nodes, (int, float)) and nodes != "-" else None
        # Cuts added as lazy constraints during B&B (0 for Compact model)
        row.total_cuts = max(0, model.model.NumConstrs - num_constrs_before)
        row.status = "OK"

        # Dump profiling if enabled
        if profiler:
            profile_rows = _dump_profile_csv(profiler, method, stem)

    except Exception as exc:
        import traceback

        logger.error("FAILED [%s] %s\n%s", method, stem, traceback.format_exc())
        row.status = f"FAILED: {type(exc).__name__}: {str(exc)[:80]}"

    return row, profile_rows


# ============================================================================
# PROFILING
# ============================================================================


def _is_project_row(fname: str) -> bool:
    """Return True for project-code frames; discard Python stdlib / site-packages."""
    if fname == "~":
        # built-in methods — keep, they show real cost
        return True
    try:
        Path(fname).resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _dump_profile_csv(
    profiler: cProfile.Profile, method: str, instance_stem: str
) -> list[dict[str, Any]]:
    """Dump profiling data to CSV."""
    profile_rows: list[dict[str, Any]] = []

    try:
        stats = pstats.Stats(profiler)
        stats.strip_dirs()

        profiling_dir = REPO_ROOT / "outputs" / "profiling"
        _ensure_dir(profiling_dir)

        profile_csv = profiling_dir / f"profile_{method}_{instance_stem}.csv"

        for func, (_cc, nc, tt, ct, _callers) in stats.stats.items():
            fname, lineno, funcname = func
            if not _is_project_row(fname):
                continue
            profile_rows.append(
                {
                    "ncalls": nc,
                    "tottime": tt,
                    "percall_tot": tt / nc if nc > 0 else 0,
                    "cumtime": ct,
                    "percall_cum": ct / nc if nc > 0 else 0,
                    "filename": fname,
                    "lineno": lineno,
                    "function": funcname,
                }
            )

        profile_rows.sort(key=lambda r: r["tottime"], reverse=True)

        with open(profile_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ncalls",
                    "tottime",
                    "percall_tot",
                    "cumtime",
                    "percall_cum",
                    "filename",
                    "lineno",
                    "function",
                ],
            )
            writer.writeheader()
            writer.writerows(profile_rows)

        logger.info("Profiling CSV written to %s", profile_csv)

    except Exception as e:
        logger.warning("Failed to dump profiling: %s", e)

    return profile_rows


# ============================================================================
# OUTPUT
# ============================================================================


def write_csv(results: list[ResultRow], output_path: Path) -> None:
    """Write results to CSV."""
    _ensure_dir(output_path)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for row in results:
            row_dict = asdict(row)
            writer.writerow(row_dict)

    logger.info("CSV written to %s", output_path)


def write_config_snapshot(
    cfg: BenchmarkConfig, sys_info: dict[str, str], output_path: Path
) -> None:
    """Write config and system info snapshot."""
    _ensure_dir(output_path)

    snapshot = {
        "config": {
            "dir_path": cfg.dir_path,
            "ext": cfg.ext,
            "instances": cfg.instances,
            "methods": cfg.methods,
            "time_limit": cfg.time_limit,
            "enable_profiling": cfg.enable_profiling,
        },
        "system_info": sys_info,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info("Config snapshot written to %s", output_path)


def write_report(
    results: list[ResultRow],
    output_path: Path,
    sys_info: dict[str, str],
    cfg: BenchmarkConfig,
    run_id: str,
) -> None:
    """Write markdown report with analysis."""
    _ensure_dir(output_path)

    # Group results by instance and method
    by_instance: dict[str, dict[str, list[ResultRow]]] = {}
    for row in results:
        if row.instance not in by_instance:
            by_instance[row.instance] = {}
        if row.method not in by_instance[row.instance]:
            by_instance[row.instance][row.method] = []
        by_instance[row.instance][row.method].append(row)

    # Build report
    lines: list[str] = [
        "# Compact vs Benders Benchmark Report\n",
        f"**Run ID:** `{run_id}`\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Hardware:** {sys_info.get('platform', 'unknown')} {sys_info.get('platform_version', '')}\n",
        f"**Python:** {sys_info.get('python_version', 'unknown')}\n",
        f"**Instances:** {len(cfg.instances)}\n",
        f"**Methods:** {len(cfg.methods)}\n",
        f"**Time limit:** {cfg.time_limit}s\n",
        f"**Profiling:** {'Enabled' if cfg.enable_profiling else 'Disabled'}\n",
        "\n---\n",
        "## Method Descriptions\n",
        "\n| Method | Model Type | Label |\n",
        "|--------|------------|-------|\n",
    ]

    for method_key, method_cfg in METHOD_CONFIG.items():
        if method_key in cfg.methods:
            lines.append(
                f"| `{method_key}` | {method_cfg['model_type']} | {method_cfg['label']} |\n"
            )

    lines.append("\n---\n")
    lines.append("## Raw Results\n\n")

    # Per-instance tables
    for instance_name in sorted(by_instance.keys()):
        lines.append(f"### {instance_name}\n\n")
        lines.append(
            "| Method | Maximize | Root LP | Final IP | Gap (%) | Time (s) | Nodes | Cuts | Status |\n"
        )
        lines.append("|--------|----------|---------|----------|---------|----------|-------|------|--------|\n")

        for method_key in cfg.methods:
            if method_key in by_instance[instance_name]:
                for row in by_instance[instance_name][method_key]:
                    maximize_str = "Max" if row.maximize else "Min"
                    cuts_str = str(row.total_cuts) if row.total_cuts is not None else "-"
                    lines.append(
                        f"| `{row.method}` | {maximize_str} | {row.root_lp} | {row.final_ip} | "
                        f"{row.gap_pct} | {row.time_s} | {row.nodes} | {cuts_str} | {row.status} |\n"
                    )

        lines.append("\n")

    # Summary statistics
    lines.append("---\n")
    lines.append("## Summary Statistics\n\n")

    # Time analysis
    lines.append("### Time Performance (seconds)\n\n")
    lines.append("| Method | Total | Avg | Min | Max | OK Instances |\n")
    lines.append("|--------|-------|-----|-----|-----|---------------|\n")

    for method_key in cfg.methods:
        times = [
            row.time_s
            for result_list in by_instance.values()
            if method_key in result_list
            for row in result_list[method_key]
            if row.time_s and row.time_s != "-"
        ]
        if times:
            total = sum(times)
            avg = total / len(times)
            min_t = min(times)
            max_t = max(times)
            ok_count = sum(1 for t_list in by_instance.values()
                         if method_key in t_list
                         for row in t_list[method_key]
                         if row.status == "OK")
            lines.append(f"| `{method_key}` | {total:.2f} | {avg:.2f} | {min_t:.2f} | {max_t:.2f} | {ok_count} |\n")

    lines.append("\n### Total Cuts Added\n\n")
    lines.append("| Method | Total | Avg | Max |\n")
    lines.append("|--------|-------|-----|-----|\n")

    for method_key in cfg.methods:
        cuts = [
            row.total_cuts
            for result_list in by_instance.values()
            if method_key in result_list
            for row in result_list[method_key]
            if row.total_cuts is not None
        ]
        if cuts:
            total_c = sum(cuts)
            avg_c = total_c / len(cuts)
            max_c = max(cuts)
            lines.append(f"| `{method_key}` | {total_c} | {avg_c:.1f} | {max_c} |\n")

    lines.append("\n### Optimality Gap (%)\n\n")
    lines.append("| Method | Median | Max | Avg |\n")
    lines.append("|--------|--------|-----|-----|\n")

    for method_key in cfg.methods:
        gaps = [
            row.gap_pct
            for result_list in by_instance.values()
            if method_key in result_list
            for row in result_list[method_key]
            if row.gap_pct and row.gap_pct != "-"
        ]
        if gaps:
            gaps_sorted = sorted(gaps)
            median = gaps_sorted[len(gaps_sorted) // 2]
            max_gap = max(gaps)
            avg_gap = sum(gaps) / len(gaps)
            lines.append(f"| `{method_key}` | {median:.4f} | {max_gap:.4f} | {avg_gap:.4f} |\n")

    lines.append("\n---\n")

    if cfg.enable_profiling:
        lines.append("## Profiling\n\n")
        lines.append("Profiling data available in `outputs/profiling/`:\n")
        lines.append("- Per-instance: `profile_{method}_{instance}.csv`\n")
        lines.append("- Summary: `profile_summary_{method}.csv` (top functions)\n\n")

    # Write file
    with open(output_path, "w") as f:
        f.writelines(lines)

    logger.info("Report written to %s", output_path)


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare OAPCompactModel vs 3 Benders variants"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick validation: little-instances/, all methods, 60s limit",
    )
    mode_group.add_argument(
        "--config",
        metavar="PATH",
        help="Path to JSON config file (headless mode)",
    )
    args = parser.parse_args()

    # Load config
    if args.smoke_test:
        cfg = smoke_test_config()
    elif args.config:
        cfg = load_json_config(args.config)
    else:
        cfg = prompt_config()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_info = get_system_info()

    # Validate instances
    missing = [p for p in cfg.instances if not Path(p).exists()]
    if missing:
        logger.warning("Missing instances: %s", missing)
        cfg.instances = [p for p in cfg.instances if Path(p).exists()]

    if not cfg.instances:
        logger.error("No valid instances. Aborting.")
        return 1

    logger.info("Benchmark start: %d instances × %d methods", len(cfg.instances), len(cfg.methods))

    # Run all solves
    all_results: list[ResultRow] = []
    total_runs = len(cfg.instances) * len(cfg.methods) * 2  # × 2 for maximize
    completed = 0

    for instance_path in cfg.instances:
        for method in cfg.methods:
            for maximize in [True, False]:
                completed += 1
                logger.info(
                    "Solve %d/%d — instance=%s method=%s maximize=%s",
                    completed,
                    total_runs,
                    Path(instance_path).stem,
                    method,
                    maximize,
                )

                row, profile_rows = run_single_solve(
                    instance_path=instance_path,
                    method=method,
                    maximize=maximize,
                    time_limit=cfg.time_limit,
                    enable_profiling=cfg.enable_profiling,
                )
                all_results.append(row)

                # Small pause to avoid thermal throttling
                if completed < total_runs:
                    time.sleep(0.5)

    # Write outputs
    out_dir_csv = REPO_ROOT / "outputs" / "CSV"
    out_dir_reports = REPO_ROOT / "outputs" / "reports"

    out_csv = out_dir_csv / f"benchmark_compact_vs_benders_{run_id}.csv"
    out_cfg = out_dir_csv / f"benchmark_compact_vs_benders_{run_id}_config.json"
    out_rpt = out_dir_reports / f"benchmark_compact_vs_benders_{run_id}.md"

    write_csv(all_results, out_csv)
    write_config_snapshot(cfg, sys_info, out_cfg)
    write_report(all_results, out_rpt, sys_info, cfg, run_id)

    # Summary
    ok_count = sum(1 for r in all_results if r.status == "OK")
    failed_count = len(all_results) - ok_count
    logger.info("Done. %d OK / %d failed.", ok_count, failed_count)
    logger.info("Outputs:")
    logger.info("  CSV: %s", out_csv)
    logger.info("  Config: %s", out_cfg)
    logger.info("  Report: %s", out_rpt)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
