# test/test_benders_semiplane.py
"""
Unit tests for V1 semiplane constraints in the Benders master.

Design reference: .claude/context/designs/2026-04-30-benders-semiplane-master.md
Plan reference:   .claude/context/plans/2026-04-30-benders-semiplane-and-benchmark.md

Tests:
  1. Semiplane=1 strictly increases master NumConstrs vs semiplane=0.
  2. Subproblems (Y, Y') are structurally identical regardless of master semiplane.
  3. IP optimum is invariant under semiplane (validity regression).
  4. No KeyError on instances with aggressive CH-cleanup (stars family).

All tests auto-skip when instance files are absent (mirrors test_benders.py convention).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from models.OAPBendersModel import OAPBendersModel
from utils.utils import compute_triangles, read_indexed_instance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INSTANCE_US_NIGHT_8 = "instance/us-night-0000008.instance"
INSTANCE_US_NIGHT_10 = "instance/us-night-0000010.instance"
INSTANCE_STARS_10 = "instance/stars-0000010.instance"


@pytest.fixture(scope="module")
def us_night_8():
    """Load us-night-0000008 once for all tests that need it."""
    path = Path(INSTANCE_US_NIGHT_8)
    if not path.exists():
        pytest.skip(f"Instance not found: {path}")
    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return points, triangles


@pytest.fixture(scope="module")
def us_night_10():
    """Load us-night-0000010 once for all tests that need it."""
    path = Path(INSTANCE_US_NIGHT_10)
    if not path.exists():
        pytest.skip(f"Instance not found: {path}")
    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return points, triangles


@pytest.fixture(scope="module")
def stars_10():
    """Load stars-0000010 once for CH-cleanup stress test."""
    path = Path(INSTANCE_STARS_10)
    if not path.exists():
        pytest.skip(f"Instance not found: {path}")
    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return points, triangles


# ---------------------------------------------------------------------------
# Test 1 — Constraint count delta
# ---------------------------------------------------------------------------


@pytest.mark.gurobi
def test_semiplane_master_adds_constraints(us_night_8):
    """Semiplane=1 strictly increases master NumConstrs vs semiplane=0."""
    points, triangles = us_night_8

    m0 = OAPBendersModel(points, triangles, name="sp_test_m0")
    m0.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=0)
    n0 = m0.model.NumConstrs

    m1 = OAPBendersModel(points, triangles, name="sp_test_m1")
    m1.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=1)
    n1 = m1.model.NumConstrs

    logger.info("NumConstrs semiplane=0: %d, semiplane=1: %d (delta=%d)", n0, n1, n1 - n0)
    assert n1 > n0, (
        f"semiplane=1 added no constraints (n0={n0}, n1={n1}). Check _add_semiplane_master for us-night-0000008."
    )


# ---------------------------------------------------------------------------
# Test 2 — Subproblem structural invariance
# ---------------------------------------------------------------------------


@pytest.mark.gurobi
def test_semiplane_master_preserves_subproblems(us_night_8):
    """Subproblems Y and Y' are structurally identical regardless of master semiplane."""
    points, triangles = us_night_8

    m0 = OAPBendersModel(points, triangles, name="sp_sub_m0")
    m0.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=0)

    m1 = OAPBendersModel(points, triangles, name="sp_sub_m1")
    m1.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=1)

    # Y subproblem
    assert m0.sub_y.NumConstrs == m1.sub_y.NumConstrs, (
        f"sub_y NumConstrs differ: semiplane=0 has {m0.sub_y.NumConstrs}, semiplane=1 has {m1.sub_y.NumConstrs}"
    )
    assert m0.sub_y.NumVars == m1.sub_y.NumVars, f"sub_y NumVars differ: {m0.sub_y.NumVars} vs {m1.sub_y.NumVars}"

    # Y' subproblem
    assert m0.sub_yp.NumConstrs == m1.sub_yp.NumConstrs, (
        f"sub_yp NumConstrs differ: semiplane=0 has {m0.sub_yp.NumConstrs}, semiplane=1 has {m1.sub_yp.NumConstrs}"
    )
    assert m0.sub_yp.NumVars == m1.sub_yp.NumVars, f"sub_yp NumVars differ: {m0.sub_yp.NumVars} vs {m1.sub_yp.NumVars}"


# ---------------------------------------------------------------------------
# Test 3 — IP optimum invariance (validity regression)
# ---------------------------------------------------------------------------


@pytest.mark.gurobi
@pytest.mark.slow
def test_semiplane_master_preserves_optimum(us_night_10):
    """IP optimum is identical with and without master semiplane (validity check)."""
    points, triangles = us_night_10

    m0 = OAPBendersModel(points, triangles, name="sp_ip_m0")
    m0.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=0)
    m0.solve(time_limit=300, verbose=False)
    _, _, ip0, _, _ = m0.get_model_stats()

    m1 = OAPBendersModel(points, triangles, name="sp_ip_m1")
    m1.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=1)
    m1.solve(time_limit=300, verbose=False)
    _, _, ip1, _, _ = m1.get_model_stats()

    assert ip0 is not None, "semiplane=0 model returned no IP value."
    assert ip1 is not None, "semiplane=1 model returned no IP value."

    logger.info("IP semiplane=0: %.6f, semiplane=1: %.6f", ip0, ip1)
    assert abs(ip0 - ip1) < 1e-2, (
        f"IP optimum diverged under semiplane: ip0={ip0:.6f}, ip1={ip1:.6f}. "
        "Semiplane V1 must be a valid constraint (no relaxation of the feasible set)."
    )


# ---------------------------------------------------------------------------
# Test 4 — No KeyError on CH-cleanup instances
# ---------------------------------------------------------------------------


@pytest.mark.gurobi
def test_semiplane_master_no_keyerror_on_ch_cleanup(stars_10):
    """Build with semiplane=1 must not raise KeyError on any instance family."""
    points, triangles = stars_10

    # Build should complete without raising; constraint count may be 0
    try:
        m = OAPBendersModel(points, triangles, name="sp_stars_guard")
        m.build(objective="Fekete", maximize=False, benders_method="farkas", semiplane=1)
    except KeyError as exc:
        pytest.fail(
            f"_add_semiplane_master raised KeyError on stars-0000010: {exc}. Check (j, j_next) not in self.x guard."
        )

    n = m.model.NumConstrs
    logger.info("stars-0000010 with semiplane=1: %d master constraints", n)
    # Constraint count may legitimately be 0 if no eligible arcs survive CH-cleanup
    assert n >= 0
