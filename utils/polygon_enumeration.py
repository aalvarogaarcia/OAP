"""Utilities for enumerating feasible polygons from Gurobi solution pool."""

import json
import os
from typing import Any

import cdd
import gurobipy as gp
import networkx as nx
import numpy as np
from numpy.typing import NDArray


def extract_polygon_from_solution(
    model: gp.Model,
    x_vars: dict[tuple[int, int], gp.Var],
    points: NDArray[np.int64],
    sol_index: int,
) -> list[int] | None:
    """Extract polygon vertices in order from a solution's x-variables.

    Parameters
    ----------
    model : gp.Model
        Gurobi model with solutions in pool
    x_vars : dict[tuple[int, int], gp.Var]
        Arc variable mapping (i, j) -> Var
    points : NDArray[np.int64]
        Point coordinates
    sol_index : int
        Solution index in the pool (0-based)

    Returns
    -------
    list[int] | None
        Ordered list of vertex indices forming the polygon, or None if invalid.
    """
    # Build adjacency from x-variables
    edges: set[tuple[int, int]] = set()

    try:
        # Set which solution to extract
        model.Params.SolutionNumber = sol_index

        for (i, j), var in x_vars.items():
            # After setting SolutionNumber, var.X gives the value from that solution
            val = var.X
            if val > 0.5:  # binary variable, treat as 1
                edges.add((i, j))
    except (AttributeError, IndexError, TypeError):
        # Solution not available or invalid index
        return None

    if not edges:
        return None

    # Reconstruct polygon by following edges
    adj: dict[int, list[int]] = {}
    for i, j in edges:
        if i not in adj:
            adj[i] = []
        adj[i].append(j)

    # Start from first vertex and traverse
    start = next(iter(adj)) if adj else 0
    polygon = [start]
    current = start

    while True:
        if current not in adj or not adj[current]:
            return None  # Invalid: dead end

        next_vertex = adj[current][0]
        if next_vertex == start:
            break  # Cycle complete

        polygon.append(next_vertex)
        current = next_vertex

        if len(polygon) > len(points):
            return None  # Invalid: too many vertices

    return polygon if len(polygon) >= 3 else None


def get_solution_objective_value(model: gp.Model, sol_index: int) -> float | None:
    """Get the objective value of a solution from the pool.

    Parameters
    ----------
    model : gp.Model
        Gurobi model with solutions in pool
    sol_index : int
        Solution index (0-based)

    Returns
    -------
    float | None
        Objective value, or None if unavailable
    """
    try:
        # Set which solution to extract, then compute its objective
        model.Params.SolutionNumber = sol_index
        obj_val = model.ObjVal  # This gets the objective value of the current solution
        return obj_val
    except (AttributeError, IndexError, TypeError):
        return None


def enumerate_all_polygons(
    model: gp.Model,
    x_vars: dict[tuple[int, int], gp.Var],
    points: NDArray[np.int64],
) -> list[dict[str, Any]]:
    """Extract all polygons from the solution pool.

    Parameters
    ----------
    model : gp.Model
        Solved Gurobi model
    x_vars : dict[tuple[int, int], gp.Var]
        Arc variable mapping
    points : NDArray[np.int64]
        Point coordinates

    Returns
    -------
    list[dict[str, Any]]
        List of polygons, each with 'vertices' and 'objective_value'
    """
    try:
        num_solutions = model.SolCount
    except AttributeError:
        return []

    polygons: list[dict[str, Any]] = []
    failed_count = 0

    for sol_idx in range(num_solutions):
        polygon = extract_polygon_from_solution(model, x_vars, points, sol_idx)
        if polygon:
            obj_val = get_solution_objective_value(model, sol_idx)
            polygons.append(
                {
                    "vertices": polygon,
                    "objective_value": obj_val,
                    "num_vertices": len(polygon),
                }
            )
        else:
            failed_count += 1

    # Sort by objective value (descending, assuming maximization by default)
    polygons.sort(key=lambda p: p["objective_value"] if p["objective_value"] is not None else float("-inf"), reverse=True)

    return polygons


def save_polygons_to_json(polygons: list[dict[str, Any]], filepath: str) -> None:
    """Save enumerated polygons to JSON file.

    Parameters
    ----------
    polygons : list[dict[str, Any]]
        List of polygon dictionaries
    filepath : str
        Output file path
    """
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    output = {
        "num_polygons": len(polygons),
        "polygons": polygons,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def display_polygons_in_terminal(
    polygons: list[dict[str, Any]],
    points: NDArray[np.int64],
    max_display: int = 50,
) -> None:
    """Display enumerated polygons in terminal (limited to max_display).

    Parameters
    ----------
    polygons : list[dict[str, Any]]
        List of polygon dictionaries
    points : NDArray[np.int64]
        Point coordinates
    max_display : int
        Maximum polygons to display (default 50)
    """
    num_to_display = min(len(polygons), max_display)

    print(f"\n{'='*70}")
    print(f"[*] ENUMERATED POLYGONS ({num_to_display} of {len(polygons)} shown)")
    print(f"{'='*70}\n")

    for idx, polygon in enumerate(polygons[:num_to_display]):
        vertices = polygon["vertices"]
        obj_val = polygon["objective_value"]
        vertex_coords = [f"({points[v][0]}, {points[v][1]})" for v in vertices]

        print(f"[Polygon {idx + 1}]")
        print(f"  Vertices (indices): {vertices}")
        print(f"  Vertex coordinates: {', '.join(vertex_coords)}")
        print(f"  Objective value: {obj_val}")
        print()

    if len(polygons) > max_display:
        print(f"... and {len(polygons) - max_display} more polygons (see file for all).\n")

    print(f"{'='*70}\n")


def extract_facets_from_enumerated_polygons(
    polygons: list[dict[str, Any]],
    x_vars: dict[tuple[int, int], gp.Var],
    verbose: bool = False,
) -> tuple[list[str], list[list[float]], set[int]]:
    """Extract facets of the convex hull of enumerated polygon indicator vectors.

    Builds the V-representation of the polytope whose vertices are the 0/1
    arc-indicator vectors of every enumerated polygon, then uses cdd to convert
    to the minimal H-representation (facet inequalities).

    Each facet row follows the cdd convention: ``[b, -a_1, ..., -a_m]``
    corresponding to ``a · x <= b``.

    Parameters
    ----------
    polygons:
        Output of :func:`enumerate_all_polygons` — each entry must have a
        ``"vertices"`` key with an ordered list of point indices.
    x_vars:
        Arc variable mapping ``(i, j) -> gp.Var`` that defines the full arc
        space and supplies variable names.
    verbose:
        Print progress information to stdout.

    Returns
    -------
    arc_names:
        Variable names in the same column order used by *facets*.
    facets:
        List of H-representation rows ``[b, -a_1, ..., -a_m]``.
    lin_set:
        Row indices that are equalities (``a · x == b``).
    """
    if not polygons or not x_vars:
        return [], [], set()

    # Stable arc ordering so column indices are reproducible.
    arc_list = sorted(x_vars.keys())
    arc_to_idx: dict[tuple[int, int], int] = {arc: k for k, arc in enumerate(arc_list)}
    arc_names = [x_vars[arc].VarName for arc in arc_list]
    n_arcs = len(arc_list)

    # Build V-representation: cdd row format is [1, x_1, ..., x_m] for a vertex.
    v_rows: list[list[float]] = []
    for poly in polygons:
        vertices: list[int] = poly["vertices"]
        vec = [0.0] * n_arcs
        for k in range(len(vertices)):
            arc = (vertices[k], vertices[(k + 1) % len(vertices)])
            if arc in arc_to_idx:
                vec[arc_to_idx[arc]] = 1.0
        v_rows.append([1.0] + vec)

    if verbose:
        print(f"[cdd] V-rep: {len(v_rows)} vertices, {n_arcs} arc variables")

    mat = cdd.Matrix(v_rows, number_type="float")
    mat.rep_type = cdd.RepType.GENERATOR

    ineqs = cdd.Polyhedron(mat).get_inequalities()
    ineqs.canonicalize()

    lin_set = set(ineqs.lin_set)
    facets = [list(ineqs[i]) for i in range(ineqs.row_size)]

    if verbose:
        print(f"[cdd] H-rep: {len(facets)} facet inequalities ({len(lin_set)} equalities)")

    return arc_names, facets, lin_set
