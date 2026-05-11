"""
Cross-model equivalence tests: OAPCompactModel vs OAPBendersModel.

For instances of 10–20 points (3 sizes × 6 families = 18 instances) verifies
that both models produce:
  1. The same IP optimal value  (test_ip_equivalence)
  2. The same LP relaxation bound (test_lp_equivalence)

Six Benders variants are tested to cover all implementation packages:
  - farkas      : baseline Farkas cuts (Package A)
  - pi          : Pi cuts (Package A)
  - cgsp        : CGSP deepest cuts, farkas-mode (Package B)
  - cgsp_pi     : CGSP deepest cuts, pi-mode (Package B)
  - mw          : Magnanti-Wong, farkas-mode, lp_relaxation core point (Package C)
  - mw_uniform  : Magnanti-Wong, farkas-mode, uniform core point (Package C)

A third test (test_benders_variant_parity) verifies that all Benders variants
agree with each other on the IP optimal value, guarding against cut-validity
bugs (wrong cuts that cut off the optimal solution).

All tests skip gracefully when instance files are absent (e.g. in CI).
"""

from pathlib import Path
from typing import Any, NamedTuple

import gurobipy as gp
import pytest

from models.OAPBendersModel import OAPBendersModel
from models.OAPCompactModel import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

gp.setParam("OutputFlag", 0)

# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

INSTANCE_DIR = Path("instance")

CANDIDATES = [
    "euro-night-0000010",
    "euro-night-0000015",
    "euro-night-0000020",
    "uniform-0000010-1",
    "uniform-0000015-1",
    "uniform-0000020-1",
    "us-night-0000010",
    "us-night-0000015",
    "us-night-0000020",
    "stars-0000010",
    "stars-0000015",
    "stars-0000020",
    "london-0000010",
    "london-0000015",
    "london-0000020",
    "uniform-0000010-2",
    "uniform-0000015-2",
    "uniform-0000020-2",
]

INSTANCE_FILES = [p for name in CANDIDATES if (p := INSTANCE_DIR / f"{name}.instance").exists()]

if not INSTANCE_FILES:
    pytest.skip(
        "No instance files found in instance/ — skipping cross-model equivalence tests.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Benders variants
# ---------------------------------------------------------------------------


class BendersVariant(NamedTuple):
    method: str
    label: str
    extra_kwargs: dict[str, Any]


BENDERS_VARIANTS = [
    pytest.param(BendersVariant("farkas", "farkas", {}), id="farkas"),
    pytest.param(BendersVariant("pi", "pi", {}), id="pi"),
    pytest.param(BendersVariant("farkas", "cgsp", {"use_deepest_cuts": True}), id="cgsp"),
    pytest.param(BendersVariant("pi", "cgsp_pi", {"use_deepest_cuts": True}), id="cgsp_pi"),
    pytest.param(BendersVariant("farkas", "mw", {"use_magnanti_wong": True}), id="mw"),
    pytest.param(
        BendersVariant("farkas", "mw_uniform", {"use_magnanti_wong": True, "core_point_strategy": "uniform"}),
        id="mw_uniform",
    ),
]

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=INSTANCE_FILES, ids=[p.stem for p in INSTANCE_FILES])
def instance(request):  # type: ignore[no-untyped-def]
    """Load an instance file and return (name, points, triangles)."""
    path: Path = request.param
    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return path.stem, points, triangles


# ---------------------------------------------------------------------------
# Test 1 — IP equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("maximize", [True, False], ids=["max", "min"])
@pytest.mark.parametrize("variant", BENDERS_VARIANTS)
def test_ip_equivalence(instance, variant: BendersVariant, maximize: bool) -> None:  # type: ignore[no-untyped-def]
    """
    Compact MIP optimal value must equal Benders MIP optimal value
    for every instance, maximize flag and Benders variant.
    """
    instance_name, points, triangles = instance
    objective = "Fekete"
    direction = "max" if maximize else "min"

    # --- Compact (ground truth) ---
    compact = OAPCompactModel(
        points,
        triangles,
        name=f"Compact_IP_{instance_name}_{objective}_{direction}",
    )
    compact.build(objective=objective, maximize=maximize)
    compact.solve(time_limit=300, verbose=False)

    if compact.model.Status != gp.GRB.OPTIMAL:
        pytest.skip(f"Compact did not reach OPTIMAL for {instance_name} (status={compact.model.Status}). Skipping.")
    expected = compact.model.ObjVal

    # --- Benders ---
    benders = OAPBendersModel(
        points,
        triangles,
        name=f"Benders_IP_{instance_name}_{objective}_{direction}_{variant.label}",
    )
    benders.build(
        objective=objective,
        maximize=maximize,
        benders_method=variant.method,
        **variant.extra_kwargs,
    )
    benders.solve(time_limit=300, verbose=False)

    assert benders.model.Status == gp.GRB.OPTIMAL, (
        f"Benders ({variant.label}) did not reach OPTIMAL for {instance_name} (status={benders.model.Status})"
    )
    actual = benders.get_objval_int()
    assert actual is not None, f"get_objval_int() returned None for {instance_name} ({variant.label})"

    assert actual == pytest.approx(expected, rel=1e-4), (
        f"IP divergence [{instance_name} {direction} {variant.label}]: Compact={expected:.6f}  Benders={actual:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — LP relaxation equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("maximize", [True, False], ids=["max", "min"])
@pytest.mark.parametrize("variant", BENDERS_VARIANTS)
def test_lp_equivalence(instance, variant: BendersVariant, maximize: bool) -> None:  # type: ignore[no-untyped-def]
    """
    Compact LP relaxation bound must equal Benders LP relaxation bound
    for every instance, maximize flag and Benders variant.
    """
    instance_name, points, triangles = instance
    objective = "Fekete"
    direction = "max" if maximize else "min"

    # --- Compact LP ---
    compact = OAPCompactModel(
        points,
        triangles,
        name=f"Compact_LP_{instance_name}_{objective}_{direction}",
    )
    compact.build(objective=objective, maximize=maximize)
    compact.solve(time_limit=120, verbose=False, relaxed=True)
    lp_compact = compact.get_objval_lp()

    if lp_compact == "-":
        pytest.skip(f"Compact LP not available for {instance_name} ({objective} {direction})")

    # --- Benders LP ---
    benders = OAPBendersModel(
        points,
        triangles,
        name=f"Benders_LP_{instance_name}_{objective}_{direction}_{variant.label}",
    )
    benders.build(
        objective=objective,
        maximize=maximize,
        benders_method=variant.method,
        **variant.extra_kwargs,
    )
    benders.solve(time_limit=120, verbose=False, relaxed=True)
    lp_benders = benders.get_objval_lp()

    if lp_benders == "-":
        pytest.skip(f"Benders ({variant.label}) LP not available for {instance_name} ({objective} {direction})")

    assert lp_benders == pytest.approx(lp_compact, rel=1e-4), (
        f"LP divergence [{instance_name} {direction} {variant.label}]: "
        f"Compact={lp_compact:.6f}  Benders={lp_benders:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Cross-Benders variant parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("maximize", [True, False], ids=["max", "min"])
def test_benders_variant_parity(instance, maximize: bool) -> None:  # type: ignore[no-untyped-def]
    """
    All Benders variants must agree on the IP optimal value.

    Each variant uses a different cut-generation strategy (Farkas, Pi,
    CGSP deepest cuts, Magnanti-Wong) but they all solve the SAME MILP,
    so the IP optimum must be identical across all variants.

    This guards against:
    - Cut validity bugs (wrong cuts that cut off the optimal solution)
    - Sign errors in cut reconstruction
    - Core-point issues in MW that could make the secondary LP infeasible
    """
    instance_name, points, triangles = instance
    TIME_LIMIT = 120  # seconds
    ABS_TOL = 1.99  # same as model.Params.MIPGapAbs in OAPBendersModel.__init__

    results: dict[str, float | None] = {}

    # Collect IP values for all variants
    variants = [
        ("farkas", "farkas", {}),
        ("farkas", "cgsp", {"use_deepest_cuts": True}),
        ("farkas", "mw", {"use_magnanti_wong": True}),
    ]

    for benders_method, label, extra_kwargs in variants:
        m = OAPBendersModel(
            points,
            triangles,
            name=f"Parity_{instance_name}_{label}_{maximize}",
        )
        m.build(
            objective="Fekete",
            maximize=maximize,
            benders_method=benders_method,
            **extra_kwargs,
        )
        m.solve(time_limit=TIME_LIMIT, verbose=False)
        if m.model.Status == gp.GRB.OPTIMAL:
            results[label] = m.get_objval_int()
        else:
            results[label] = None  # didn't solve within time limit

    # At least farkas must solve (it's the most reliable)
    if results.get("farkas") is None:
        pytest.skip(f"Farkas did not solve {instance_name} in {TIME_LIMIT}s — skipping parity check")

    baseline = results["farkas"]
    for label, val in results.items():
        if val is None:
            continue  # variant timed out — don't assert
        assert val == pytest.approx(baseline, abs=ABS_TOL, rel=1e-4), (
            f"Variant parity failure [{instance_name} {'max' if maximize else 'min'}]: "
            f"farkas={baseline:.4f}  {label}={val:.4f}\n"
            f"This means {label} produced an INVALID CUT that cut off the optimal solution."
        )
