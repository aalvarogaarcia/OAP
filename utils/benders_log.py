"""Benders cut logging and JSONL serialisation for OAP.

Functions in this module write and read structured cut-log files in JSON Lines
format.  No Gurobi model is constructed here; the only Gurobi dependency is
reading coefficient data from ``gp.LinExpr``.
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

import gurobipy as gp

from utils.geometry import Arc

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SerializedCoeffMap = dict[str, float]
SerializedExpr = dict[str, SerializedCoeffMap | float]
SerializedRayData = dict[str, SerializedCoeffMap | float]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def serialize_expr(expr: gp.LinExpr | None) -> SerializedExpr | None:
    """Convert a Gurobi ``LinExpr`` to a JSON-serialisable dict.

    Returns ``None`` if *expr* is ``None``.

    The returned dict has the shape::

        {"coeffs": {"x_i_j": coeff, ...}, "constant": float}
    """
    if expr is None:
        return None
    coeffs: dict[str, float] = {}
    for i in range(expr.size()):
        var = expr.getVar(i)
        coeff = expr.getCoeff(i)
        if abs(coeff) > 1e-6:
            coeffs[var.VarName] = round(coeff, 4)
    return {"coeffs": coeffs, "constant": round(expr.getConstant(), 4)}


def format_cut_string(cut_expr: SerializedExpr, sense: str | None) -> str:
    """Render a serialised cut expression as a human-readable string.

    Example output: ``"Corte Lógico: 1.0 <= x_0_1 + x_2_1"``
    """
    const = cast(float, cut_expr.get("constant", 0.0))
    coeffs = cast(SerializedCoeffMap, cut_expr.get("coeffs", {}))

    parts: list[str] = []
    for var, val in coeffs.items():
        if val < 0:
            parts.append(f"{abs(val)}*{var}" if abs(val) != 1 else var)
        elif val > 0:
            parts.append(f"- {val}*{var}" if val != 1 else f"- {var}")

    formula = " + ".join(parts).replace("+ -", "- ")

    if sense == "<=":
        return f"Corte Lógico: {const} <= {formula}"
    if sense == ">=":
        return f"Corte Lógico: {const} >= {formula}"
    raise ValueError(f"format_cut_string: unknown cut sense {sense!r}")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def log_benders_cut(
    filepath: str,
    iteration: int,
    node_depth: int,
    subproblem_type: str,
    x_sol: dict[tuple[int, int], float],
    v_components: dict[str, Any],
    cut_value: float,
    tolerance: float = 1e-5,
    cut_expr: gp.LinExpr | None = None,
    sense: str | None = None,
) -> None:
    """Append a Benders cut record to a JSONL log file.

    Args:
        filepath: Path to the ``.jsonl`` log file (created if absent).
        iteration: Cut counter or callback iteration number.
        node_depth: B&B tree depth (0 = root / integer solution).
        subproblem_type: ``'Y'`` (external feasibility), ``'Y_prime'``
            (internal feasibility), or ``'Y_OPT'`` (optimality).
        x_sol: Current master solution ``{(i, j): value}``.
        v_components: Dual-ray / dual-vertex components
            ``{name: {arc: coeff} | scalar}``.
        cut_value: Cut violation (feasibility) or triangulation cost (optimality).
        tolerance: Threshold below which arc values are treated as zero.
        cut_expr: Optional Gurobi expression for the generated cut.
        sense: Cut sense (``'<='``, ``'>='``, or ``None``).
    """
    active_x: SerializedCoeffMap = {f"{i}_{j}": round(val, 4) for (i, j), val in x_sol.items() if abs(val) > tolerance}

    ray_data: SerializedRayData = {}
    for comp_name, values in v_components.items():
        if values:
            if isinstance(values, dict):
                typed = cast(dict[Arc, float], values)
                ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in typed.items()}
            else:
                ray_data[comp_name] = round(cast(float, values), 4)

    record: dict[str, Any] = {
        "iteration": iteration,
        "node_depth": node_depth,
        "subproblem": subproblem_type,
        "cut_value": round(cut_value, 6),
        "sense": sense,
        "active_x": active_x,
        "dual_components": ray_data,
        "cut_expr": serialize_expr(cut_expr),
    }

    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def log_inv_benders_cut(
    filepath: str,
    iteration: int,
    node_depth: int,
    y_sol: dict[int, float],
    yp_sol: dict[int, float],
    v_components: dict[str, Any],
    cut_value: float,
    tolerance: float = 1e-5,
    cut_expr: gp.LinExpr | None = None,
    sense: str | None = None,
) -> None:
    """Append an inverted-Benders cut record to a JSONL log file.

    This is the counterpart of :func:`log_benders_cut` for
    ``OAPInverseBendersModel``.  The master holds triangle-assignment
    variables ``y`` / ``yp`` (not tour arcs), so the active-solution fields
    are triangle-indexed rather than arc-indexed.

    Args:
        filepath: Path to the ``.jsonl`` log file (created if absent).
        iteration: Cut counter or callback iteration number.
        node_depth: B&B tree depth (0 = root / integer solution).
        y_sol: Master triangle assignment ``{triangle_id: value}`` for ``y``.
        yp_sol: Master triangle assignment ``{triangle_id: value}`` for ``yp``.
        v_components: Farkas ray components, arc-keyed
            ``{name: {(i,j): coeff} | scalar}`` — same shape as
            :func:`log_benders_cut` so that ``plot_farkas_ray_network`` can
            consume the ``dual_components`` field without modification.
        cut_value: Farkas violation value (``v^T b(y*, yp*)``).
        tolerance: Threshold below which triangle values are treated as zero.
        cut_expr: Optional Gurobi ``LinExpr`` for the generated cut
            (in master variables ``y`` / ``yp``).
        sense: Cut sense (``'<='``, ``'>='``, or ``None``).
    """
    active_y: SerializedCoeffMap = {str(t): round(val, 4) for t, val in y_sol.items() if abs(val) > tolerance}
    active_yp: SerializedCoeffMap = {str(t): round(val, 4) for t, val in yp_sol.items() if abs(val) > tolerance}

    ray_data: SerializedRayData = {}
    for comp_name, values in v_components.items():
        if values:
            if isinstance(values, dict):
                typed = cast(dict[Arc, float], values)
                ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in typed.items()}
            else:
                ray_data[comp_name] = round(cast(float, values), 4)

    record: dict[str, Any] = {
        "iteration": iteration,
        "node_depth": node_depth,
        "subproblem": "X",
        "cut_value": round(cut_value, 6),
        "sense": sense,
        "active_y": active_y,
        "active_yp": active_yp,
        "dual_components": ray_data,
        "cut_expr": serialize_expr(cut_expr),
    }

    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def load_farkas_logs(filepath: str) -> list[dict[str, Any]]:
    """Load Benders cut records from a JSONL file.

    Returns an empty list (with a warning) if the file does not exist.
    """
    logs: list[dict[str, Any]] = []
    if not os.path.exists(filepath):
        print(f"load_farkas_logs: file not found: {filepath}")
        return logs
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                logs.append(cast(dict[str, Any], json.loads(stripped)))
    return logs


def parse_edge(edge_str: str) -> tuple[int, int]:
    """Convert a serialised edge string ``'i_j'`` to an integer tuple ``(i, j)``."""
    i_str, j_str = edge_str.split("_")
    return int(i_str), int(j_str)
