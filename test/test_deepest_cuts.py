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
    pytest.skip("No suitable small instance file found (us-night-0000008.instance or us-night-0000010.instance).")


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
    assert "l1_norm" in constr_names, "CGSP (Y') must include the L₁ normalisation constraint 'l1_norm'."
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
    assert "l1_norm" in constr_names, "CGSP (Y) must include the L₁ normalisation constraint 'l1_norm'."
    # Fekete objective → no π₀
    assert pi0_var is None, "For Fekete objective, _build_cgsp_y must return pi0_var=None."


# ---------------------------------------------------------------------------
# Test 3: get_cgsp_cut_yp returns correct 3-tuple types
# ---------------------------------------------------------------------------


def test_get_cgsp_cut_yp_structure(warmup_rhs):
    """get_cgsp_cut_yp must return (LinExpr, float, dict)."""
    model, x_sol = warmup_rhs

    import gurobipy as gp

    result = model.get_cgsp_cut_yp(x_sol)

    assert isinstance(result, tuple) and len(result) == 3, "get_cgsp_cut_yp must return a 3-tuple."
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

    assert isinstance(result, tuple) and len(result) == 3, "get_cgsp_cut_y must return a 3-tuple."
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

    assert model.use_deepest_cuts is False, "use_deepest_cuts must default to False for backward compatibility."


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
    assert obj_benders is not None, "Benders (use_deepest_cuts=False) must find a solution."
    assert obj_compact == pytest.approx(obj_benders, rel=1e-3), (
        f"Legacy Benders obj={obj_benders:.4f} should match compact obj={obj_compact:.4f}."
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
                assert w == pytest.approx(1.0), f"Default weight for {key}[{arc_key}] should be 1.0, got {w}."
        else:
            assert val == pytest.approx(1.0), f"Default weight for '{key}' should be 1.0, got {val}."


# ---------------------------------------------------------------------------
# Test 6b: Weight resolution — custom scalar weight is used
# ---------------------------------------------------------------------------


def test_resolve_weights_custom_scalar(warmup_rhs):
    """_resolve_weights uses custom scalar weight when provided."""
    model, _ = warmup_rhs
    model.cut_weights_yp = {"global_p": 2.5}

    weights = model._resolve_weights("yp", model.constrs_yp, include_pi0=False)

    if "global_p" in weights:
        assert weights["global_p"] == pytest.approx(2.5), "Custom scalar weight for 'global_p' should be 2.5."

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


# ---------------------------------------------------------------------------
# Test 8a: _compute_relaxed_l1_weights — structure mirrors constrs_yp
# ---------------------------------------------------------------------------


def test_compute_relaxed_l1_weights_structure(built_benders_farkas):
    """_compute_relaxed_l1_weights returns a dict with the same nested structure
    as constrs_yp: dict-valued groups map every arc key to a float, scalar
    groups map to a single float."""
    model = built_benders_farkas

    weights_yp = model._compute_relaxed_l1_weights("yp")
    weights_y = model._compute_relaxed_l1_weights("y")

    for which, weights, constrs in (
        ("yp", weights_yp, model.constrs_yp),
        ("y", weights_y, model.constrs_y),
    ):
        assert set(weights.keys()) == set(constrs.keys()), (
            f"_compute_relaxed_l1_weights('{which}') keys must match constrs_{which} keys. "
            f"Got {set(weights.keys())}, expected {set(constrs.keys())}."
        )
        for key, val in constrs.items():
            w = weights[key]
            if isinstance(val, dict):
                assert isinstance(w, dict), (
                    f"Group '{key}' is arc-indexed in constrs_{which} — weight must also be a dict, got {type(w)}."
                )
                assert set(w.keys()) == set(val.keys()), (
                    f"Arc keys for group '{key}' differ: got {set(w.keys())}, expected {set(val.keys())}."
                )
                for arc_key, wv in w.items():
                    assert isinstance(wv, float), f"Weight for {key}[{arc_key}] must be float, got {type(wv)}."
            else:
                assert isinstance(w, float), f"Scalar group '{key}' weight must be float, got {type(w)}."


# ---------------------------------------------------------------------------
# Test 8b: _compute_relaxed_l1_weights — group-level values are correct
# ---------------------------------------------------------------------------


def test_compute_relaxed_l1_weights_group_values(built_benders_farkas):
    """Structural weights follow the Relaxed-ℓ₁ table:
        alpha/gamma/delta → 1.0, beta/r3 → 2.0, global/r1/r2 → 0.0.
    Groups absent from the model are simply skipped."""
    model = built_benders_farkas

    # Expected structural weights per group name
    _EXPECTED: dict[str, float] = {
        "alpha": 1.0,
        "beta": 2.0,
        "gamma": 1.0,
        "delta": 1.0,
        "global": 0.0,
        "r1": 0.0,
        "r2": 0.0,
        "r3": 2.0,
        "alpha_p": 1.0,
        "beta_p": 2.0,
        "gamma_p": 1.0,
        "delta_p": 1.0,
        "global_p": 0.0,
        "r1_p": 0.0,
        "r2_p": 0.0,
        "r3_p": 2.0,
    }

    for which in ("y", "yp"):
        weights = model._compute_relaxed_l1_weights(which)
        for key, val in weights.items():
            if key not in _EXPECTED:
                continue  # unknown group — skip assertion
            expected_w = _EXPECTED[key]
            if isinstance(val, dict):
                for arc_key, wv in val.items():
                    assert wv == pytest.approx(expected_w), (
                        f"Weight for {key}[{arc_key}] (which='{which}'): expected {expected_w}, got {wv}."
                    )
            else:
                assert val == pytest.approx(expected_w), (
                    f"Scalar weight for '{key}' (which='{which}'): expected {expected_w}, got {val}."
                )


# ---------------------------------------------------------------------------
# Test 8c: build(use_deepest_cuts=True, cgsp_norm="relaxed_l1") does not crash
# ---------------------------------------------------------------------------


def test_build_with_use_deepest_cuts_relaxed_l1(small_instance):
    """build(use_deepest_cuts=True, cgsp_norm='relaxed_l1') must not raise
    AttributeError and must populate cut_weights_y / cut_weights_yp."""
    points, triangles, _ = small_instance
    from models.OAPBendersModel import OAPBendersModel

    model = OAPBendersModel(points, triangles, name="test_rl1_build")
    # This previously crashed with:
    #   AttributeError: 'OAPBendersModel' object has no attribute
    #   '_compute_relaxed_l1_weights'
    model.build(
        objective="Fekete",
        maximize=False,
        benders_method="farkas",
        use_deepest_cuts=True,
        cgsp_norm="relaxed_l1",
    )

    assert model.cut_weights_y is not None, (
        "cut_weights_y must be a dict after build(use_deepest_cuts=True, cgsp_norm='relaxed_l1')."
    )
    assert model.cut_weights_yp is not None, (
        "cut_weights_yp must be a dict after build(use_deepest_cuts=True, cgsp_norm='relaxed_l1')."
    )
    assert len(model.cut_weights_y) > 0, "cut_weights_y must not be empty."
    assert len(model.cut_weights_yp) > 0, "cut_weights_yp must not be empty."


# ---------------------------------------------------------------------------
# Test 9: invalidate_cgsp_cache resets both cache attributes to None
# ---------------------------------------------------------------------------


def test_invalidate_cgsp_cache(warmup_rhs):
    """invalidate_cgsp_cache must set _cgsp_yp_cache and _cgsp_y_cache to None."""
    model, x_sol = warmup_rhs

    # Populate the caches by triggering one CGSP build
    model._get_or_build_cgsp_yp(x_sol)
    model._get_or_build_cgsp_y(x_sol, eta_sol=0.0)

    assert model._cgsp_yp_cache is not None, "Cache should be populated after _get_or_build_cgsp_yp."
    assert model._cgsp_y_cache is not None, "Cache should be populated after _get_or_build_cgsp_y."

    # Invalidate
    model.invalidate_cgsp_cache()

    assert model._cgsp_yp_cache is None, "_cgsp_yp_cache must be None after invalidate_cgsp_cache()."
    assert model._cgsp_y_cache is None, "_cgsp_y_cache must be None after invalidate_cgsp_cache()."
