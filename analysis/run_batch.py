# analysis/run_batch.py
"""Batch runner: solve a directory of instances with OAPBendersModel and/or
OAPInverseBendersModel and generate one PDF of Benders cuts per
(instance, model-family, variant) combination.

Output layout
-------------
<output_dir>/
    Benders/
        MILP/<instance_stem>.pdf
        LP/<instance_stem>.pdf      (only when --lp is passed)
    InvBenders/
        MILP/<instance_stem>.pdf
        LP/<instance_stem>.pdf      (only when --lp is passed)

Log files (JSONL, git-ignored)
------------------------------
outputs/Logs/Benders/MILP/<instance_stem>.jsonl
outputs/Logs/Benders/LP/<instance_stem>.jsonl
outputs/Logs/InvBenders/MILP/<instance_stem>.jsonl
outputs/Logs/InvBenders/LP/<instance_stem>.jsonl

Usage
-----
.venv/Scripts/python.exe analysis/run_batch.py -d instance/little-instances --lp
.venv/Scripts/python.exe analysis/run_batch.py --skip-benders --lp -t 120
.venv/Scripts/python.exe analysis/run_batch.py -f us-night -m pi --objective Internal
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

from models import OAPBendersModel, OAPInverseBendersModel
from utils.utils import compute_triangles, read_indexed_instance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-instance runners
# ---------------------------------------------------------------------------

def _run_benders(
    instance_path: Path,
    time_limit: int,
    benders_method: Literal["farkas", "pi"],
    objective: str,
    maximize: bool,
    lp: bool,
    output_dir: str,
) -> None:
    """Solve one instance with OAPBendersModel and write cut PDFs.

    LP relaxation (``--lp``) is run on the *same* model object after the MILP
    solve.  ``solve_lp_relaxation`` permanently drops integrality, so MILP must
    always come first.
    """
    name      = instance_path.stem
    points    = read_indexed_instance(str(instance_path))
    triangles = compute_triangles(points)

    model = OAPBendersModel(points, triangles, name=name)
    model.build(
        objective=objective,
        maximize=maximize,
        benders_method=benders_method,
        sum_constrain=True,
    )

    # ── MILP ────────────────────────────────────────────────────────────────
    milp_log = f"outputs/Logs/Benders/MILP/{name}.jsonl"
    os.makedirs(os.path.dirname(milp_log), exist_ok=True)
    model.set_log_path(milp_log)
    model.solve(time_limit=time_limit, verbose=False, save_cuts=True)

    milp_pdf = os.path.join(output_dir, "Benders", "MILP", f"{name}.pdf")
    os.makedirs(os.path.dirname(milp_pdf), exist_ok=True)
    model.generate_benders_report(output_pdf_path=milp_pdf)
    logger.info("[Benders MILP] PDF saved: %s", milp_pdf)

    # ── LP relaxation (destructive — must follow MILP) ──────────────────────
    if lp:
        lp_log = f"outputs/Logs/Benders/LP/{name}.jsonl"
        os.makedirs(os.path.dirname(lp_log), exist_ok=True)
        model.set_log_path(lp_log)
        model.solve(time_limit=time_limit, verbose=False, save_cuts=True, relaxed=True)

        lp_pdf = os.path.join(output_dir, "Benders", "LP", f"{name}.pdf")
        os.makedirs(os.path.dirname(lp_pdf), exist_ok=True)
        model.generate_benders_report(output_pdf_path=lp_pdf)
        logger.info("[Benders LP  ] PDF saved: %s", lp_pdf)


def _run_inv_benders(
    instance_path: Path,
    time_limit: int,
    maximize: bool,
    lp: bool,
    output_dir: str,
) -> None:
    """Solve one instance with OAPInverseBendersModel and write cut PDFs.

    OAPInverseBendersModel only supports ``objective='Internal'``.
    LP relaxation (``--lp``) is run on the *same* model object after the MILP
    solve.  ``solve_lp_relaxation`` permanently drops integrality, so MILP must
    always come first.
    """
    name      = instance_path.stem
    points    = read_indexed_instance(str(instance_path))
    triangles = compute_triangles(points)

    model = OAPInverseBendersModel(points, triangles, name=name)
    model.build(objective="Internal", maximize=maximize, sum_constrain=True)

    # ── MILP ────────────────────────────────────────────────────────────────
    milp_log = f"outputs/Logs/InvBenders/MILP/{name}.jsonl"
    os.makedirs(os.path.dirname(milp_log), exist_ok=True)
    model.set_log_path(milp_log)
    model.solve(time_limit=time_limit, verbose=False, save_cuts=True)

    milp_pdf = os.path.join(output_dir, "InvBenders", "MILP", f"{name}.pdf")
    os.makedirs(os.path.dirname(milp_pdf), exist_ok=True)
    model.generate_inv_benders_report(output_pdf_path=milp_pdf)
    logger.info("[InvBenders MILP] PDF saved: %s", milp_pdf)

    # ── LP relaxation (destructive — must follow MILP) ──────────────────────
    if lp:
        lp_log = f"outputs/Logs/InvBenders/LP/{name}.jsonl"
        os.makedirs(os.path.dirname(lp_log), exist_ok=True)
        model.set_log_path(lp_log)
        model.solve(time_limit=time_limit, verbose=False, save_cuts=True, relaxed=True)

        lp_pdf = os.path.join(output_dir, "InvBenders", "LP", f"{name}.pdf")
        os.makedirs(os.path.dirname(lp_pdf), exist_ok=True)
        model.generate_inv_benders_report(output_pdf_path=lp_pdf)
        logger.info("[InvBenders LP  ] PDF saved: %s", lp_pdf)


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def run_batch(
    instances_dir: str,
    filter_texts: list[str],
    time_limit: int,
    benders_method: Literal["farkas", "pi"],
    objective: str,
    maximize: bool,
    lp: bool,
    run_benders_flag: bool,
    run_inv_benders_flag: bool,
    output_dir: str,
) -> None:
    """Discover instances, run the requested models, and write cut PDFs."""
    directory = Path(instances_dir)
    if not directory.exists():
        logger.error("Instance directory not found: %s", instances_dir)
        return

    all_instances = sorted(directory.glob("*.instance"))
    if filter_texts:
        all_instances = [
            p for p in all_instances
            if any(text in p.name for text in filter_texts)
        ]

    if not all_instances:
        logger.warning(
            "No instances found matching filter %r in %s.", filter_texts, directory
        )
        return

    total = len(all_instances)
    direction = "MAX" if maximize else "MIN"

    logger.info("=" * 60)
    logger.info("Processing %d instance(s) | filter=%r", total, filter_texts)
    logger.info(
        "  Benders    : %s | method=%s | obj=%s | %s",
        "yes" if run_benders_flag else "no",
        benders_method, objective, direction,
    )
    logger.info(
        "  InvBenders : %s | obj=Internal | %s",
        "yes" if run_inv_benders_flag else "no",
        direction,
    )
    logger.info("  LP relax   : %s", "yes" if lp else "no")
    logger.info("  Time limit : %ds per solve", time_limit)
    logger.info("  Output dir : %s", output_dir)
    logger.info("=" * 60)

    ok = err = 0
    for idx, path in enumerate(all_instances, 1):
        logger.info("\n--- %d/%d: %s ---", idx, total, path.stem)
        try:
            if run_benders_flag:
                _run_benders(
                    path, time_limit, benders_method,
                    objective, maximize, lp, output_dir,
                )
            if run_inv_benders_flag:
                _run_inv_benders(path, time_limit, maximize, lp, output_dir)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", path.stem, exc, exc_info=True)
            err += 1

    logger.info("=" * 60)
    logger.info("Done. %d/%d succeeded, %d failed.", ok, total, err)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Batch solver: generate Benders cut PDFs for a directory of instances.\n"
            "Both OAPBendersModel and OAPInverseBendersModel are run by default.\n"
            "Use --skip-benders or --skip-inv-benders to run only one family."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d", "--dir",
        default="instance/little-instances",
        metavar="DIR",
        help="Directory containing *.instance files (default: instance/little-instances).",
    )
    parser.add_argument(
        "-f", "--filter",
        nargs="+",
        default=[],
        metavar="TEXT",
        help=(
            "Keep only instances whose filename contains any of these strings. "
            "If omitted, all instances in DIR are processed."
        ),
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Time limit per solve in seconds (default: 300).",
    )
    parser.add_argument(
        "-m", "--method",
        choices=["farkas", "pi"],
        default="farkas",
        help="Benders subproblem method for OAPBendersModel (default: farkas). Ignored for InvBenders.",
    )
    parser.add_argument(
        "--objective",
        choices=["Fekete", "Internal", "External", "Diagonals"],
        default="Fekete",
        help=(
            "Objective for OAPBendersModel (default: Fekete). "
            "OAPInverseBendersModel always uses Internal and ignores this flag."
        ),
    )

    # maximize / minimize are mutually exclusive but maximize is the default
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--maximize",
        dest="maximize",
        action="store_true",
        default=True,
        help="Maximise the objective (default).",
    )
    direction.add_argument(
        "--minimize",
        dest="maximize",
        action="store_false",
        help="Minimise the objective instead of maximising.",
    )

    parser.add_argument(
        "--lp",
        action="store_true",
        help=(
            "Also run the LP relaxation for each instance/model and write PDFs to …/LP/. "
            "The LP solve is destructive (drops integrality), so MILP always runs first."
        ),
    )
    parser.add_argument(
        "--skip-benders",
        action="store_true",
        help="Do not run OAPBendersModel.",
    )
    parser.add_argument(
        "--skip-inv-benders",
        action="store_true",
        help="Do not run OAPInverseBendersModel.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/Analysis",
        metavar="DIR",
        help="Root output directory for PDF files (default: outputs/Analysis).",
    )

    args = parser.parse_args()

    if args.skip_benders and args.skip_inv_benders:
        parser.error("--skip-benders and --skip-inv-benders cannot both be set.")

    run_batch(
        instances_dir=args.dir,
        filter_texts=args.filter,
        time_limit=args.time,
        benders_method=args.method,
        objective=args.objective,
        maximize=args.maximize,
        lp=args.lp,
        run_benders_flag=not args.skip_benders,
        run_inv_benders_flag=not args.skip_inv_benders,
        output_dir=args.output_dir,
    )
