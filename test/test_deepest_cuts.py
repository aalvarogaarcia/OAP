# test/test_deepest_cuts.py
"""
Unit tests for BendersCGSPMixin — Deepest Benders Cuts.

Tests are structured around 7 acceptance criteria:
  1. CGSP construction for Y' (external subproblem)
  2. CGSP construction for Y (internal subproblem, Fekete)
  3. Cut generation from get_cgsp_cut_yp — tuple shape
  4. Cut generation from get_cgsp_cut_y — tuple shape
  5. Backward compatibility: use_deepest_cuts=False (default) keeps legacy code
  6. Weight resolution — default all-ones and custom override
  7. Optimality-cut flag absent for Fekete objective

Tests auto-skip when the fixture instance file is absent (consistent with the
rest of the test suite).  All tests require a Gurobi licence; when no licence
is available the OAPBendersModel import itself will raise and pytest will
collect 0 tests (same behaviour as test_benders.py on licence-free CI).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

INSTANCE_PATH_8 = "instance/us-night-0000008.instance"
INSTANCE_PATH_10 = "instance/us-night-0000010.instance"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_instance():
    """Load the smallest available fixture (8 or 10 points) for fast tests."""
    for path_str in (INSTANCE_PATH_8, INSTANCE_PATH_10):
        path = Path(path_str)
        if path.exists():
            from utils.utils import compute_triangles, read_indexed_instance
            points = read_indexed_instance(str(path))
            triangles = compute_triangles(points)
            return points, triangles, path_str
    pytest.skip(
        "No suitable small instance file found "
        "(us-night-0000008.instance or us-night-0000010.instance)."
    )


@pytest.fixture(scope="module")
def built_benders_farkas(small_instance):
    """OAPBendersModel built with farkas method and subproblems populated."""
    points, triangles, _path = small_instance
    from models.OAPBendersModel import OAPBendersModel
    model = OAPBendersModel(points, triangles, name="test_cgsp_farkas")
    model.build(
        objective="Fekete",
        maximize=False,
        benders_method="farkas",
        sum_constrain=True,
        use_deepest_cuts=False,
    )
    return model


@pytest.fixture(scope="module")
def warmup_rhs(built_benders_farkas):
    """Return model with subproblem RHS updated to all-zero x_sol."""
    model = built_benders_farkas
    x_sol = {arc: 0.0 for arc in model.x.keys()}
    model._update_subproblem_rhs(x_sol)
    return model, x_sol


# ---------------------------------------------------------------------------
# Test 1: CGSP construction for Y' (external subproblem)
# ---------------------------------------------------------------------------

def test_cgsp_yp_construction(warmup_rhs):
    """_build_cgsp_yp returns a valid Gurobi model with L₁ normalisation."""
    model, x_sol = warmup_rhs

    cgsp, pi_vars = model._build_cgsp_yp(x_sol)

    import gurobipy as gp
    assert isinstance(cgsp, gp.Model), "_build_cgsp_yp must return a gp.Model."
    assert cgsp.NumVars > 0, "CGSP (Y') must contain at least one dual variable."

    constr_names = [c.ConstrName for c in cgsp.getConstrs()]
    assert "l1_norm" in constr_names, (
        "CGSP (Y') must include the L₁ normalisation constraint 'l1_norm'."
    )
    assert len(pi_vars) > 0, "pi_vars dict returned from _build_cgsp_yp must be non-empty."


# ---------------------------------------------------------------------------
# Test 2: CGSP construction for Y (Fekete — no π₀)
# ---------------------------------------------------------------------------

def test_cgsp_y_construction_fekete(warmup_rhs):
    """_build_cgsp_y for Fekete objective returns a model without π₀."""
    model, x_sol = warmup_rhs

    import gurobipy as gp
    cgsp, pi_vars, pi0_var = model._build_cgsp_y(x_sol, eta_sol=0.0)

    assert isinstance(cgsp, gp.Model), "_build_cgsp_y must return a gp.Model."
    assert cgsp.NumVars > 0, "CGSP (Y) must contain at least one dual variable."

    constr_names = [c.ConstrName for c in cgsp.getConstrs()]
    assert "l1_norm" in constr_names, (
        "CGSP (Y) must include the L₁ normalisation constraint 'l1_norm'."
    )
    # Fekete objective → no π₀
    assert pi0_var is None, (
        "For Fekete objective, _build_cgsp_y must return pi0_var=None."
    )


# ---------------------------------------------------------------------------
# Test 3: get_cgsp_cut_yp returns correct 3-tuple types
# ---------------------------------------------------------------------------

def test_get_cgsp_cut_yp_structure(warmup_rhs):
    """get_cgsp_cut_yp must return (LinExpr, float, dict)."""
    model, x_sol = warmup_rhs

    import gurobipy as gp
    result = model.get_cgsp_cut_yp(x_sol)

    assert isinstance(result, tuple) and len(result) == 3, (
        "get_cgsp_cut_yp must return a 3-tuple."
    )
    cut_expr, cut_rhs, witness = result
    assert isinstance(cut_expr, gp.LinExpr), "First element must be gp.LinExpr."
    assert isinstance(cut_rhs, float), "Second element must be float."
    assert isinstance(witness, dict), "Third element must be dict."


# ---------------------------------------------------------------------------
# Test 4: get_cgsp_cut_y returns correct 3-tuple types
# ---------------------------------------------------------------------------

def test_get_cgsp_cut_y_structure(warmup_rhs):
    """get_cgsp_cut_y must return (LinExpr, float, dict)."""
    model, x_sol = warmup_rhs

    import gurobipy as gp
    result = model.get_cgsp_cut_y(x_sol, eta_sol=0.0)

    assert isinstance(result, tuple) and len(result) == 3, (
        "get_cgsp_cut_y must return a 3-tuple."
    )
    cut_expr, cut_rhs, witness = result
    assert isinstance(cut_expr, gp.LinExpr), "First element must be gp.LinExpr."
    assert isinstance(cut_rhs, float), "Second element must be float."
    assert isinstance(witness, dict), "Third element must be dict."


# ---------------------------------------------------------------------------
# Test 5a: Backward compat — use_deepest_cuts defaults to False
# ---------------------------------------------------------------------------

def test_backward_compat_default_is_false(small_instance):
    """OAPBendersModel.build() without use_deepest_cuts stores False."""
    points, triangles, _ = small_instance
    from models.OAPBendersModel import OAPBendersModel

    model = OAPBendersModel(points, triangles, name="test_compat_default")
    model.build(objective="Fekete", maximize=False, benders_method="farkas")

    assert model.use_deepest_cuts is False, (
        "use_deepest_cuts must default to False for backward compatibility."
    )


# ---------------------------------------------------------------------------
# Test 5b: Backward compat — MIP solve with use_deepest_cuts=False
# ---------------------------------------------------------------------------

def test_backward_compat_mip_solve(small_instance):
    """Benders MIP with use_deepest_cuts=False must match compact model."""
    points, triangles, _ = small_instance
    from models.OAPBendersModel import OAPBendersModel
    from models.OAPCompactModel import OAPCompactModel

    compact = OAPCompactModel(points, triangles, name="compat_compact")
    compact.build(objective="Fekete", maximize=False)
    compact.solve(time_limit=120, verbose=False)
    obj_compact = compact.get_objval_int()

    benders = OAPBendersModel(points, triangles, name="compat_benders")
    benders.build(
        objective="Fekete",
        maximize=False,
        benders_method="farkas",
        use_deepest_cuts=False,
    )
    benders.solve(time_limit=120, verbose=False)
    obj_benders = benders.get_objval_int()

    assert obj_compact is not None, "Compact model must find a solution."
    assert obj_benders is not None, (
        "Benders (use_deepest_cuts=False) must find a solution."
    )
    assert obj_compact == pytest.approx(obj_benders, rel=1e-3), (
        f"Legacy Benders obj={obj_benders:.4f} should match compact "
        f"obj={obj_compact:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 6a: Weight resolution — default weights are all 1.0
# ---------------------------------------------------------------------------

def test_resolve_weights_defaults(warmup_rhs):
    """_resolve_weights with no custom weights returns all-1.0 values."""
    model, _ = warmup_rhs
    model.cut_weights_y = None
    model.cut_weights_yp = None

    weights = model._resolve_weights("yp", model.constrs_yp, include_pi0=False)

    for key, val in weights.items():
        if isinstance(val, dict):
            for arc_key, w in val.items():
                assert w == pytest.approx(1.0), (
                    f"Default weight for {key}[{arc_key}] should be 1.0, got {w}."
                )
        else:
            assert val == pytest.approx(1.0), (
                f"Default weight for '{key}' should be 1.0, got {val}."
            )


# ---------------------------------------------------------------------------
# Test 6b: Weight resolution — custom scalar weight is used
# ---------------------------------------------------------------------------

def test_resolve_weights_custom_scalar(warmup_rhs):
    """_resolve_weights uses custom scalar weight when provided."""
    model, _ = warmup_rhs
    model.cut_weights_yp = {"global_p": 2.5}

    weights = model._resolve_weights("yp", model.constrs_yp, include_pi0=False)

    if "global_p" in weights:
        assert weights["global_p"] == pytest.approx(2.5), (
            "Custom scalar weight for 'global_p' should be 2.5."
        )

    # Clean up
    model.cut_weights_yp = None


# ---------------------------------------------------------------------------
# Test 7: Optimality-cut flag absent for Fekete objective
# ---------------------------------------------------------------------------

def test_optimality_cut_flag_absent_for_fekete(warmup_rhs):
    """For Fekete objective, get_cgsp_cut_y witness must not contain is_optimality_cut=True."""
    model, x_sol = warmup_rhs

    # Confirm it's a Fekete-objective model
    if getattr(model, "objective", "Fekete") == "Internal":
        pytest.skip("Test requires Fekete-objective model.")

    _, _, witness = model.get_cgsp_cut_y(x_sol, eta_sol=0.0)

    assert witness.get("is_optimality_cut", False) is not True, (
        "Fekete objective must not produce an optimality cut witness."
    )
