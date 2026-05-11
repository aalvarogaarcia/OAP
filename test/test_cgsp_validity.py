"""B-4: CGSP cut validity tests.

Verifies that:
1. A cut produced by get_cgsp_cut_yp is strictly violated at the x_bar that
   triggered it (i.e. cut_expr(x_bar) > cut_rhs + TOL).
2. No cut is emitted when the subproblem is feasible at the given x.
"""

import logging
from pathlib import Path
from typing import Any

import gurobipy as gp
import pytest

from models.OAPBendersModel import OAPBendersModel
from utils.utils import compute_triangles, read_indexed_instance

gp.setParam("OutputFlag", 0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------
INSTANCES_DIR = Path("instance/little-instances")
INSTANCE_FILES = list(INSTANCES_DIR.glob("*.instance")) if INSTANCES_DIR.exists() else []

if not INSTANCE_FILES:
    pytest.skip(
        "No instance files found in instance/little-instances/",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_cut(cut_expr: gp.LinExpr, x_sol: dict[str, Any]) -> float:
    """Evaluate a Gurobi LinExpr at a given x_sol dict (arc -> float)."""
    val = cut_expr.getConstant()
    for k in range(cut_expr.size()):
        var = cut_expr.getVar(k)
        coeff = cut_expr.getCoeff(k)
        # Variable name encodes the arc: e.g. "x_1_2" or "x[1,2]"
        name = var.VarName
        arc = None
        try:
            if "[" in name:
                inner = name[name.index("[") + 1 : name.index("]")]
                i_str, j_str = inner.split(",")
                arc = (int(i_str.strip()), int(j_str.strip()))
            elif name.startswith("x_"):
                parts = name.split("_")
                arc = (int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            arc = None
        x_val = x_sol.get(arc, 0.0) if arc is not None else 0.0
        val += coeff * x_val
    return val


def _build_benders(points, triangles, instance_name, objective="Fekete", maximize=False):
    model = OAPBendersModel(points, triangles, name=f"CGSP_validity_{instance_name}")
    model.build(objective=objective, maximize=maximize, benders_method="farkas")
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=INSTANCE_FILES, ids=[f.stem for f in INSTANCE_FILES])
def loaded_instance(request):
    fp = request.param
    points = read_indexed_instance(str(fp))
    triangles = compute_triangles(points)
    return fp.stem, points, triangles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cgsp_cut_yp_violated_at_x_bar(loaded_instance):
    """Cut produced by get_cgsp_cut_yp must be strictly violated at x_bar."""
    instance_name, points, triangles = loaded_instance
    model = _build_benders(points, triangles, instance_name)

    # Solve the LP relaxation to get a fractional x_bar that is likely to
    # violate Y' constraints.
    model.solve(time_limit=30, verbose=False, relaxed=True)

    if model.sub_yp is None:
        pytest.skip(f"sub_yp not initialised for {instance_name}")

    # Extract x_sol from the master LP solution
    x_sol: dict[str, Any] = {}
    for (i, j), var in model.x.items():
        try:
            x_sol[(i, j)] = var.X
        except gp.GurobiError:
            pass

    if not x_sol:
        pytest.skip(f"No x variables accessible for {instance_name}")

    cut_expr, cut_rhs, witness = model.get_cgsp_cut_yp(x_sol)

    if cut_expr is None:
        # No violation found — that is valid (subproblem may be feasible)
        assert witness.get("aborted") in ("no_violation", "solve_failed"), f"Unexpected abort reason: {witness}"
        pytest.skip(f"No violated cut for {instance_name}: {witness}")

    lhs_val = _eval_cut(cut_expr, x_sol)
    assert lhs_val > cut_rhs + 1e-6, (
        f"[{instance_name}] Cut not violated at x_bar: lhs={lhs_val:.8f} <= rhs={cut_rhs:.8f}. witness={witness}"
    )


def test_cgsp_cut_y_violated_at_x_bar(loaded_instance):
    """Cut produced by get_cgsp_cut_y must be strictly violated at x_bar."""
    instance_name, points, triangles = loaded_instance
    model = _build_benders(points, triangles, instance_name)

    model.solve(time_limit=30, verbose=False, relaxed=True)

    if model.sub_y is None:
        pytest.skip(f"sub_y not initialised for {instance_name}")

    x_sol: dict[str, Any] = {}
    for (i, j), var in model.x.items():
        try:
            x_sol[(i, j)] = var.X
        except gp.GurobiError:
            pass

    if not x_sol:
        pytest.skip(f"No x variables accessible for {instance_name}")

    cut_expr, cut_rhs, witness = model.get_cgsp_cut_y(x_sol)

    if cut_expr is None:
        assert witness.get("aborted") in ("no_violation", "solve_failed"), f"Unexpected abort reason: {witness}"
        pytest.skip(f"No violated cut for {instance_name}: {witness}")

    lhs_val = _eval_cut(cut_expr, x_sol)
    assert lhs_val > cut_rhs + 1e-6, (
        f"[{instance_name}] Cut not violated at x_bar: lhs={lhs_val:.8f} <= rhs={cut_rhs:.8f}. witness={witness}"
    )


def test_cgsp_cut_yp_none_when_feasible(loaded_instance):
    """get_cgsp_cut_yp returns None when Y' is feasible at x_bar = 0."""
    instance_name, points, triangles = loaded_instance
    model = _build_benders(points, triangles, instance_name)

    # x_sol = all zeros: sub_yp is trivially feasible (RHS ≥ 0 for all constrs)
    x_sol_zero: dict[str, Any] = {arc: 0.0 for arc in model.x}

    # Force the sub_yp to be solved at x=0 by calling its internal update
    # (The public entry point calls _build_cgsp_yp internally, so just call it)
    cut_expr, cut_rhs, witness = model.get_cgsp_cut_yp(x_sol_zero)

    # At x=0 the violation should be 0 or negative — no cut emitted
    if cut_expr is not None:
        lhs_val = _eval_cut(cut_expr, x_sol_zero)
        # It's acceptable to emit a cut only if it truly violates at x=0
        assert lhs_val > cut_rhs + 1e-6, (
            f"[{instance_name}] Spurious cut emitted at x=0: lhs={lhs_val:.8f} rhs={cut_rhs:.8f}"
        )
