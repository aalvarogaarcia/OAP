# test/test_inverse_benders_vs_compact.py
"""Regression test: OAPInverseBendersModel vs OAPCompactModel.

Parametrised over a small set of instance names so the test list is always
visible (even when instance files are absent).  Each test auto-skips if the
instance file does not exist locally — identical behaviour to
``test_compact_maximize.py``.

The test checks that the optimal Internal-area objective value reported by
``OAPInverseBendersModel`` matches the value from ``OAPCompactModel``
within a relative tolerance of 0.1 %.

No TSV reference file is required: the compact model is the ground truth.

CI note
-------
This test requires a valid Gurobi licence.  It is skipped automatically when
``gurobipy`` cannot create a model (licence error is caught as a
``gurobipy.GurobiError``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from models import OAPCompactModel, OAPInverseBendersModel
from utils.utils import compute_triangles, read_indexed_instance

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Parametrisation — add instance names here as new test cases
# ---------------------------------------------------------------------------
INSTANCE_NAMES: list[str] = [
    "us-night-0000005",
    "us-night-0000006",
    "us-night-0000007",
    "us-night-0000008",
    "us-night-0000010",
]

# Side-effect: ensure output directory exists at import time (mirrors other tests)
Path("outputs/Others/Testing").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(instance_name: str):
    """Load points and triangles, skip if the instance file is absent."""
    path = BASE_DIR / "instance" / f"{instance_name}.instance"
    if not path.exists():
        pytest.skip(f"Instance file not found: {path}")
    points    = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return points, triangles


def _compact_internal_obj(instance_name: str) -> float:
    """Solve the compact model and return the Internal-area optimal value."""
    points, triangles = _load(instance_name)
    model = OAPCompactModel(points, triangles, name=f"{instance_name}_compact_ref")
    model.build(
        objective="Internal",
        mode=0,
        maximize=True,
        subtour="SCF",
        sum_constrain=True,
    )
    model.solve(time_limit=120, verbose=False)
    _, _, ip_val, _, _ = model.get_model_stats()
    if ip_val == "-":
        pytest.skip(f"Compact model did not find a solution for {instance_name}.")
    return float(ip_val)


def _inv_benders_obj(instance_name: str) -> float:
    """Solve OAPInverseBendersModel and return the optimal objective value."""
    import gurobipy as gp
    points, triangles = _load(instance_name)
    try:
        model = OAPInverseBendersModel(
            points, triangles, name=f"{instance_name}_inv_benders"
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi licence error: {exc}")

    model.build(objective="Internal", maximize=True, sum_constrain=True)
    model.solve(time_limit=120, verbose=False, save_cuts=False)

    if model.model.SolCount == 0:
        pytest.skip(
            f"OAPInverseBendersModel found no solution for {instance_name} "
            f"within the time limit."
        )
    return float(model.model.ObjVal)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("instance_name", INSTANCE_NAMES)
def test_inv_benders_matches_compact(instance_name: str) -> None:
    """OAPInverseBendersModel objective == OAPCompactModel (Internal, max)."""
    import gurobipy as gp

    # Skip early if Gurobi is not available (no licence)
    try:
        _probe = gp.Model("_licence_probe")
        _probe.dispose()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi not available: {exc}")

    compact_val = _compact_internal_obj(instance_name)
    inv_val     = _inv_benders_obj(instance_name)

    logger.info(
        f"{instance_name}: compact={compact_val:.6f}  inv_benders={inv_val:.6f}"
    )

    assert inv_val == pytest.approx(compact_val, rel=1e-3), (
        f"Objective mismatch for {instance_name}: "
        f"InvBenders={inv_val:.6f} vs Compact={compact_val:.6f}"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
