"""
DDMA cut validity and equivalence tests.

Verifies for BendersDDMAMixin (Algorithm 3, Hosseini & Turner 2025):

1. test_ddma_cut_validity        — every DDMA cut is strictly violated at the
                                   x̄ that triggered it (feasibility check).
2. test_ddma_ip_equals_farkas    — DDMA produces the same IP optimal value as
                                   plain Farkas (same problem, different cuts).
3. test_ddma_lp_equals_farkas    — DDMA LP relaxation converges to the same
                                   bound as Farkas LP relaxation.
4. test_ddma_early_termination   — DDMA with max_iter=1 still produces a valid
                                   cut (Remark 6 in the paper).
5. test_ddma_no_cut_on_feasible  — when the subproblem is already feasible at
                                   the given x, DDMA must not emit a cut.

All tests skip gracefully when instance files are absent.
"""
from pathlib import Path

import gurobipy as gp
import pytest

from models.OAPBendersModel import OAPBendersModel
from utils.utils import compute_triangles, read_indexed_instance

gp.setParam("OutputFlag", 0)

# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

INSTANCE_DIR = Path("instance/little-instances")
INSTANCE_FILES = (
    list(INSTANCE_DIR.glob("*.instance")) if INSTANCE_DIR.exists() else []
)

if not INSTANCE_FILES:
    pytest.skip(
        "No instance files in instance/little-instances/ — skipping DDMA tests.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_cut(cut_expr: gp.LinExpr, x_sol: dict) -> float:
    """Evaluate a Gurobi LinExpr at a given x_sol dict."""
    val = cut_expr.getConstant()
    for i in range(cut_expr.size()):
        v = cut_expr.getVar(i)
        coef = cut_expr.getCoeff(i)
        # find arc key from variable name "x_i_j"
        name = v.VarName
        if name.startswith("x_"):
            parts = name.split("_")
            arc = (int(parts[1]), int(parts[2]))
            val += coef * x_sol.get(arc, 0.0)
    return val


def _build_ddma_model(points, triangles, name, benders_method="farkas"):
    m = OAPBendersModel(points, triangles, name=name)
    m.build(
        objective="Fekete",
        maximize=True,
        benders_method=benders_method,
        use_ddma=True,
    )
    return m


def _build_farkas_model(points, triangles, name):
    m = OAPBendersModel(points, triangles, name=name)
    m.build(
        objective="Fekete",
        maximize=True,
        benders_method="farkas",
        use_ddma=False,
    )
    return m


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(params=INSTANCE_FILES, ids=[p.stem for p in INSTANCE_FILES])
def instance(request):
    path: Path = request.param
    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    if len(triangles) == 0:
        pytest.skip(f"Degenerate instance '{path.stem}' — empty triangulation.")
    return path.stem, points, triangles


# ---------------------------------------------------------------------------
# Test 1 — Cut validity: every DDMA cut violates x̄
# ---------------------------------------------------------------------------

def test_ddma_cut_validity(instance):
    """
    A cut produced by get_ddma_cut_yp / get_ddma_cut_y must satisfy:
        cut_expr(x̄) > cut_rhs + TOL
    i.e. the current master solution actually violates the cut.
    """
    name, points, triangles = instance
    m = _build_ddma_model(points, triangles, f"DDMA_valid_{name}")

    # Use the LP relaxation solution as a synthetic x̄
    # (it may violate the subproblem feasibility)
    x_sol = {arc: 1.0 / m.N for arc in m.x.keys()}
    m._update_subproblem_rhs(x_sol)
    m.sub_y.optimize()
    m.sub_yp.optimize()

    TOL = 1e-6

    # Y'
    if m.sub_yp.Status == gp.GRB.INFEASIBLE:
        cut_expr, cut_rhs, witness = m.get_ddma_cut_yp(x_sol)
        if cut_expr is not None:
            lhs = _eval_cut(cut_expr, x_sol)
            assert lhs > cut_rhs + TOL, (
                f"DDMA Y' cut NOT violated at x̄ for {name}: "
                f"cut_expr(x̄)={lhs:.6f} ≤ cut_rhs={cut_rhs:.6f}"
            )

    # Y
    if m.sub_y.Status == gp.GRB.INFEASIBLE:
        cut_expr, cut_rhs, witness = m.get_ddma_cut_y(x_sol)
        if cut_expr is not None:
            lhs = _eval_cut(cut_expr, x_sol)
            assert lhs > cut_rhs + TOL, (
                f"DDMA Y cut NOT violated at x̄ for {name}: "
                f"cut_expr(x̄)={lhs:.6f} ≤ cut_rhs={cut_rhs:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 2 — IP equivalence: DDMA ≡ Farkas on optimal value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maximize", [True, False], ids=["max", "min"])
def test_ddma_ip_equals_farkas(instance, maximize: bool):
    """
    DDMA and plain Farkas must reach the same IP optimal value.

    Both methods solve the same MILP, differing only in the quality
    (depth) of the Benders cuts injected.  The optimal IP value is
    determined by the problem, not the cut strategy.
    """
    name, points, triangles = instance
    direction = "max" if maximize else "min"

    # Farkas (ground truth)
    m_farkas = OAPBendersModel(points, triangles, name=f"Farkas_IP_{name}_{direction}")
    m_farkas.build(objective="Fekete", maximize=maximize, benders_method="farkas")
    m_farkas.solve(time_limit=120, verbose=False)

    if m_farkas.model.Status != gp.GRB.OPTIMAL:
        pytest.skip(f"Farkas did not reach OPTIMAL for {name} — skipping parity check.")

    expected = m_farkas.get_objval_int()
    assert expected is not None

    # DDMA
    m_ddma = _build_ddma_model(
        points, triangles, f"DDMA_IP_{name}_{direction}"
    )
    m_ddma.build.__func__  # silence; model was already built above
    # Rebuild with maximize setting:
    m_ddma2 = OAPBendersModel(points, triangles, name=f"DDMA2_IP_{name}_{direction}")
    m_ddma2.build(
        objective="Fekete",
        maximize=maximize,
        benders_method="farkas",
        use_ddma=True,
    )
    m_ddma2.solve(time_limit=120, verbose=False)

    assert m_ddma2.model.Status == gp.GRB.OPTIMAL, (
        f"DDMA did not reach OPTIMAL for {name} {direction} "
        f"(status={m_ddma2.model.Status})"
    )
    actual = m_ddma2.get_objval_int()
    assert actual is not None

    # Allow MIPGapAbs tolerance
    ABS_TOL = 1.99
    assert actual == pytest.approx(expected, abs=ABS_TOL, rel=1e-4), (
        f"IP divergence [{name} {direction}]: "
        f"Farkas={expected:.4f}  DDMA={actual:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — LP equivalence: DDMA LP ≡ Farkas LP
# ---------------------------------------------------------------------------

def test_ddma_lp_equals_farkas(instance):
    """
    DDMA LP relaxation must converge to the same bound as Farkas LP.

    Both methods enumerate Benders cuts for the same LP, so the
    relaxation optimum is the same (up to numerical tolerance).
    """
    name, points, triangles = instance

    # Farkas LP
    m_f = OAPBendersModel(points, triangles, name=f"Farkas_LP_{name}")
    m_f.build(objective="Fekete", maximize=True, benders_method="farkas")
    m_f.solve(time_limit=60, relaxed=True, verbose=False)
    lp_farkas = m_f.get_objval_lp()

    if lp_farkas == "-":
        pytest.skip(f"Farkas LP not available for {name}.")

    # DDMA LP
    m_d = OAPBendersModel(points, triangles, name=f"DDMA_LP_{name}")
    m_d.build(objective="Fekete", maximize=True, benders_method="farkas", use_ddma=True)
    m_d.solve(time_limit=60, relaxed=True, verbose=False)
    lp_ddma = m_d.get_objval_lp()

    if lp_ddma == "-":
        pytest.skip(f"DDMA LP not available for {name}.")

    assert lp_ddma == pytest.approx(lp_farkas, rel=1e-4), (
        f"LP divergence [{name}]: Farkas={lp_farkas:.6f}  DDMA={lp_ddma:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Early termination: DDMA with max_iter=1 still gives valid cut
# ---------------------------------------------------------------------------

def test_ddma_early_termination_valid(instance):
    """
    DDMA with max_iter=1 (single Benders solve) must still return a
    cut that is violated at x̄ (Remark 6 in Hosseini & Turner 2025).
    """
    name, points, triangles = instance
    m = _build_ddma_model(points, triangles, f"DDMA_early_{name}")

    x_sol = {arc: 1.0 / m.N for arc in m.x.keys()}
    m._update_subproblem_rhs(x_sol)
    m.sub_yp.optimize()

    TOL = 1e-6

    if m.sub_yp.Status != gp.GRB.INFEASIBLE:
        pytest.skip(f"Y' is not infeasible at uniform x for {name} — skip early-term test.")

    # max_iter=1 must still return a valid cut
    cut_expr, cut_rhs, witness = m.get_ddma_cut_yp(x_sol, max_iter=1)
    assert cut_expr is not None, (
        f"DDMA (max_iter=1) returned None for {name} when Y' is infeasible."
    )
    lhs = _eval_cut(cut_expr, x_sol)
    assert lhs > cut_rhs + TOL, (
        f"DDMA (max_iter=1) cut NOT violated at x̄ for {name}: "
        f"cut_expr(x̄)={lhs:.6f} ≤ cut_rhs={cut_rhs:.6f}"
    )
    assert "ddma_iters" in witness
    assert witness["ddma_iters"] == 1


# ---------------------------------------------------------------------------
# Test 5 — No spurious cut when subproblem is feasible at x_sol
# ---------------------------------------------------------------------------

def test_ddma_no_cut_on_feasible(instance):
    """
    When both subproblems are feasible for the given x̄, DDMA must not
    emit a cut (no violation → correct convergence signal).
    """
    name, points, triangles = instance
    m = _build_ddma_model(points, triangles, f"DDMA_nofeas_{name}")

    # Use actual optimal integer solution (if model reaches OPTIMAL)
    m_ref = OAPBendersModel(points, triangles, name=f"Ref_{name}")
    m_ref.build(objective="Fekete", maximize=True, benders_method="farkas")
    m_ref.solve(time_limit=60, verbose=False)

    if m_ref.model.Status != gp.GRB.OPTIMAL or m_ref.model.SolCount == 0:
        pytest.skip(f"Reference model did not reach OPTIMAL for {name}.")

    x_sol = {arc: var.X for arc, var in m_ref.x.items()}

    m._update_subproblem_rhs(x_sol)
    m.sub_y.optimize()
    m.sub_yp.optimize()

    if m.sub_yp.Status != gp.GRB.OPTIMAL:
        pytest.skip(
            f"Y' not OPTIMAL at optimal x for {name} "
            f"(status={m.sub_yp.Status}) — skip no-cut test."
        )

    cut_expr, cut_rhs, witness = m.get_ddma_cut_yp(x_sol)
    assert cut_expr is None, (
        f"DDMA emitted a spurious Y' cut at a feasible x̄ for {name}. "
        f"Witness: {witness}"
    )
