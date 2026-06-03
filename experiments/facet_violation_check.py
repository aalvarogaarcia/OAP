"""experiments/facet_violation_check.py

Enumerate all feasible polygons for a given instance via Gurobi's solution
pool, extract the facets of their convex hull in arc-indicator space (using
cdd / pycddlib), then check which of those facets are violated by the LP
relaxation solution.

Only violated facets are reported — these are valid inequalities for the
integer hull that the LP relaxation does NOT satisfy, and are therefore
candidates for cutting planes.

Run from the repo root:
    .venv/bin/python experiments/facet_violation_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import OAPCompactModel  # noqa: E402
from utils.polygon_enumeration import (  # noqa: E402
    enumerate_all_polygons,
    extract_facets_from_enumerated_polygons,
)
from utils.utils import compute_triangles, read_indexed_instance  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FILE = "euro-night-0000010"

INSTANCE_DIR: Path = _REPO_ROOT / "instance"
OUTPUT_DIR: Path = _REPO_ROOT / "outputs" / "FacetCheck"

OBJECTIVE = "Internal"
MAXIMIZE = True
SUBTOUR = "SCF"
SUM_CONSTRAIN = True

POOL_TIME_LIMIT = 300     # seconds for the MIP solution pool enumeration
LP_TIME_LIMIT = 120       # seconds for the LP relaxation solve

VIOLATION_TOL = 1e-6      # lhs > rhs + tol  ⟹  facet violated

VERBOSE_SOLVER = True

_W = 65   # terminal width for separators


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _step(label: str) -> float:
    """Print a step header and return the start timestamp."""
    print(f"\n  ▶  {label} ...")
    return time.perf_counter()


def _done(t0: float, **info: Any) -> None:
    """Print elapsed time and optional key=value pairs on completion."""
    elapsed = time.perf_counter() - t0
    parts = "  ".join(f"{k}={v}" for k, v in info.items())
    suffix = f"  ({parts})" if parts else ""
    print(f"     done  {elapsed:.2f}s{suffix}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(name: str) -> OAPCompactModel:
    instance_file = INSTANCE_DIR / f"{FILE}.instance"
    points = read_indexed_instance(str(instance_file))
    triangles = compute_triangles(points)
    model = OAPCompactModel(points, triangles, name=name)
    model.build(
        objective=OBJECTIVE,  # type: ignore[arg-type]
        maximize=MAXIMIZE,
        subtour=SUBTOUR,      # type: ignore[arg-type]
        sum_constrain=SUM_CONSTRAIN,
    )
    return model


def _check_violations(
    arc_names: list[str],
    facets: list[list[float]],
    lin_set: set[int],
    x_relaxed: dict[tuple[int, int], float],
    arc_list: list[tuple[int, int]],
    tol: float = VIOLATION_TOL,
) -> list[dict[str, Any]]:
    """Return one dict per facet that is violated by *x_relaxed*.

    Facet convention (cdd H-rep):  row = [b, -a_0, ..., -a_{m-1}]
    Inequality:  a · x <= b  (equality when row index in lin_set)
    """
    violated: list[dict[str, Any]] = []

    for row_idx, row in enumerate(facets):
        b = row[0]
        lhs = sum(-row[k + 1] * x_relaxed.get(arc_list[k], 0.0)
                  for k in range(len(arc_list)))
        slack = b - lhs

        if slack < -tol:
            sense = "==" if row_idx in lin_set else "<="
            coefs = {arc_names[k]: -row[k + 1]
                     for k in range(len(arc_list)) if abs(row[k + 1]) > 1e-9}
            violated.append({
                "row_index": row_idx,
                "sense": sense,
                "rhs": b,
                "lhs": lhs,
                "slack": slack,
                "coefficients": coefs,
            })

    return violated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    instance_file = INSTANCE_DIR / f"{FILE}.instance"
    if not instance_file.exists():
        print(f"[ERROR] Instance file not found: {instance_file}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*_W}")
    print(f"  Instance : {FILE}")
    print(f"  Objective: {'MAX' if MAXIMIZE else 'MIN'} {OBJECTIVE}  |  Subtour: {SUBTOUR}")
    print(f"{'='*_W}")

    # ------------------------------------------------------------------
    # Phase 1: enumerate all feasible polygons via Gurobi solution pool
    # ------------------------------------------------------------------
    t = _step("[1/4] Enumerating polygons — MIP solve, PoolSearchMode=2")
    enum_model = _build_model(name=f"{FILE}_enum")
    enum_model.solve(time_limit=POOL_TIME_LIMIT, verbose=VERBOSE_SOLVER, all_polygons=True)

    if enum_model.model.SolCount == 0:
        print("[ERROR] No feasible solution found. Aborting.")
        sys.exit(1)

    polygons = enumerate_all_polygons(enum_model.model, enum_model.x, enum_model.points)

    if not polygons:
        print("[ERROR] Solution pool returned 0 valid polygons. Aborting.")
        sys.exit(1)

    _done(t, polygons=len(polygons))

    # ------------------------------------------------------------------
    # Phase 2: extract facets — cdd V→H conversion (black box, no hooks)
    # ------------------------------------------------------------------
    t = _step("[2/4] Extracting facets — cdd V→H  (no sub-step progress available)")
    arc_names, facets, lin_set = extract_facets_from_enumerated_polygons(
        polygons, enum_model.x, verbose=False
    )
    arc_list = sorted(enum_model.x.keys())
    _done(t, facets=len(facets), equalities=len(lin_set))

    # ------------------------------------------------------------------
    # Phase 3: solve LP relaxation
    # ------------------------------------------------------------------
    t = _step("[3/4] Solving LP relaxation")
    lp_model = _build_model(name=f"{FILE}_lp")
    lp_model.solve(time_limit=LP_TIME_LIMIT, verbose=VERBOSE_SOLVER, relaxed=True)

    if lp_model.model.SolCount == 0:
        print("[ERROR] LP relaxation found no solution. Aborting.")
        sys.exit(1)

    x_relaxed = lp_model.x_relaxed
    lp_obj = lp_model.model.ObjVal
    _done(t, lp_obj=f"{lp_obj:.6f}")

    # ------------------------------------------------------------------
    # Phase 4: check violations (pure Python loop — deterministic cost)
    # ------------------------------------------------------------------
    t = _step(f"[4/4] Checking {len(facets)} facets against LP solution")
    violated = _check_violations(arc_names, facets, lin_set, x_relaxed, arc_list)
    n_total = len(facets)
    n_violated = len(violated)
    _done(t, violated=f"{n_violated}/{n_total}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*_W}")
    print(f"  VIOLATED FACETS  ({n_violated} of {n_total})")
    print(f"{'='*_W}")

    for entry in violated:
        coefs = entry["coefficients"]
        terms = "  ".join(f"{v:+.4f}*{name}" for name, v in sorted(coefs.items()))
        print(f"\n  [{entry['row_index']}]  {terms}  {entry['sense']}  {entry['rhs']:.4f}")
        print(f"        LP lhs = {entry['lhs']:.6f}   slack = {entry['slack']:.6f}  (VIOLATED)")

    print(f"\n{'='*_W}\n")

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    output_file = OUTPUT_DIR / f"{FILE}_violated_facets.json"
    output: dict[str, Any] = {
        "instance": FILE,
        "objective": OBJECTIVE,
        "maximize": MAXIMIZE,
        "subtour": SUBTOUR,
        "num_polygons_enumerated": len(polygons),
        "num_facets_total": n_total,
        "num_facets_violated": n_violated,
        "lp_objective": lp_obj,
        "violated_facets": violated,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
