"""Pure geometry helpers, instance I/O, and shared type aliases for OAP.

This module contains no Gurobi dependency and no side-effects at import time.
All heavy geometry computation (triangulation, crossing-edge detection, etc.)
lives here so it can be tested and reused without a solver licence.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Arc = tuple[int, int]
PointLookup = dict[int, tuple[float, float]] | NDArray[np.int64]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_data(file_path: str) -> list[str]:
    """Read a file and return its lines as a stripped list."""
    with open(file_path, "r") as fh:
        data = fh.readlines()
    return [line.strip() for line in data]


def read_indexed_instance(filepath: str) -> NDArray[np.int64]:
    """Read a ``.instance`` or ``.pre`` file and return point coordinates.

    Returns an ``(n, 2)`` array of integer (x, y) coordinates.
    """
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # Detect .pre format (has a POINTS section header)
    points_section = any(line.strip() == "POINTS" for line in lines)

    if points_section:
        coords_list: list[list[int]] = []
        in_points = False
        for line in lines:
            stripped = line.strip()
            if stripped == "POINTS":
                in_points = True
                continue
            if in_points:
                if stripped in {
                    "CONVEX_HULL",
                    "TRIANGLES",
                    "CROSSING_SEGMENTS",
                    "INCOMPATIBLE_TRIANGLES",
                }:
                    break
                if stripped == "" or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 3:
                    coords_list.append([int(float(parts[1])), int(float(parts[2]))])

        if not coords_list:
            raise ValueError(f"No points found in POINTS section of {filepath}")
        return np.array(coords_list, dtype=np.int64)

    # Simple .instance format: index  x  y  (one point per row)
    data = np.loadtxt(filepath, comments="#", dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three integer columns in {filepath}")
    return data[:, 1:3].astype(np.int64)


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------


def compute_convex_hull(points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the indices of convex-hull vertices (scipy ordering)."""
    hull = ConvexHull(points)
    return hull.vertices


def compute_convex_hull_area(points: NDArray[np.int64]) -> float:
    """Convex-hull area via the shoelace formula."""
    if len(points) < 3:
        return 0.0
    hull_idx = compute_convex_hull(points)
    hull_pts = points[hull_idx]
    x = hull_pts[:, 0]
    y = hull_pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------


def orientation_2d(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
) -> int:
    """Orientation sign for three (x, y) points.

    Returns ``0`` for collinear, ``1`` for clockwise, ``-1`` for CCW.
    """
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if det == 0:
        return 0
    return -1 if det > 0 else 1


def contains_proper(
    p1: tuple[int, int],
    q1: tuple[int, int],
    p2: tuple[int, int],
    q2: tuple[int, int],
) -> bool:
    """Return ``True`` if segments *p1q1* and *p2q2* intersect properly.

    Excludes endpoint-only touches and collinear/overlapping cases.
    """
    o1 = orientation_2d(p1, q1, p2)
    o2 = orientation_2d(p1, q1, q2)
    o3 = orientation_2d(p2, q2, p1)
    o4 = orientation_2d(p2, q2, q1)
    return o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and o1 != o2 and o3 != o4


def point_in_triangle(
    pt: NDArray[np.int64],
    v1: NDArray[np.int64],
    v2: NDArray[np.int64],
    v3: NDArray[np.int64],
) -> bool:
    """Return ``True`` if *pt* lies inside (or on the boundary of) triangle *v1 v2 v3*."""
    d1 = (pt[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (pt[1] - v2[1])
    d2 = (pt[0] - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[0]) * (pt[1] - v3[1])
    d3 = (pt[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (pt[1] - v1[1])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def signed_area(
    p1: NDArray[np.int64],
    p2: NDArray[np.int64],
    p3: NDArray[np.int64],
) -> float:
    """Signed area of triangle *p1 p2 p3* (positive = CCW)."""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def is_colineal(
    p1: NDArray[np.int64],
    p2: NDArray[np.int64],
    p3: NDArray[np.int64],
) -> bool:
    """Return ``True`` if *p1*, *p2*, *p3* are collinear."""
    return (p2[1] - p1[1]) * (p3[0] - p2[0]) == (p3[1] - p2[1]) * (p2[0] - p1[0])


def segments_intersect(
    p1: NDArray[np.int64],
    p2: NDArray[np.int64],
    p3: NDArray[np.int64],
    p4: NDArray[np.int64],
) -> bool:
    """Return ``True`` if segments *p1-p2* and *p3-p4* properly intersect.

    Endpoint-only intersections are excluded.
    """

    def _ccw(
        A: NDArray[np.int64],
        B: NDArray[np.int64],
        C: NDArray[np.int64],
    ) -> np.bool_:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    if np.array_equal(p1, p3) or np.array_equal(p1, p4) or np.array_equal(p2, p3) or np.array_equal(p2, p4):
        return False
    d1 = _ccw(p1, p3, p4) != _ccw(p2, p3, p4)
    d2 = _ccw(p1, p2, p3) != _ccw(p1, p2, p4)
    return bool(d1 and d2)


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def compute_triangles(points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Compute all valid (empty, non-degenerate) triangles for a point set.

    Returns an ``(T, 3)`` array of vertex-index triples.
    """
    n = len(points)
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)

    hull = compute_convex_hull(points)
    hull_edges = np.sort(
        np.array(
            [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))],
            dtype=np.int64,
        ),
        axis=1,
    )
    hull_mask = np.zeros(n, dtype=bool)
    hull_mask[hull] = True

    valid_triangles: list[NDArray[np.int64]] = []

    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product == 0:  # degenerate / collinear
            continue

        tri_idx = np.array([i, j, k], dtype=np.int64)

        # Ensure all hull edges appear CCW in the triangle
        for h_idx in range(len(hull)):
            hull_edge_cw = (hull[(h_idx + 1) % len(hull)], hull[h_idx])
            tri_edges_directed = [
                (int(tri_idx[0]), int(tri_idx[1])),
                (int(tri_idx[1]), int(tri_idx[2])),
                (int(tri_idx[2]), int(tri_idx[0])),
            ]
            for _edge_pos, (va, vb) in enumerate(tri_edges_directed):
                if (va, vb) == hull_edge_cw:
                    tri_idx = np.array([tri_idx[0], tri_idx[2], tri_idx[1]], dtype=np.int64)
                    break

        # Reject triangles with chord edges (hull–hull edges that are not hull edges)
        tri_edges = np.sort(
            np.array(
                [
                    [tri_idx[0], tri_idx[1]],
                    [tri_idx[1], tri_idx[2]],
                    [tri_idx[2], tri_idx[0]],
                ],
                dtype=np.int64,
            ),
            axis=1,
        )
        hull_hull = hull_mask[tri_edges].all(axis=1)
        if hull_hull.any():
            matches = (tri_edges[:, None, :] == hull_edges[None, :, :]).all(axis=2).any(axis=1)
            if (hull_hull & ~matches).any():
                continue

        # Reject triangles that contain another point
        contains_other = False
        for m in range(n):
            if m in (i, j, k):
                continue
            if point_in_triangle(points[m], points[tri_idx[0]], points[tri_idx[1]], points[tri_idx[2]]):
                contains_other = True
                break

        if not contains_other:
            valid_triangles.append(tri_idx)

    if valid_triangles:
        return np.array(valid_triangles, dtype=np.int64)
    return np.empty((0, 3), dtype=np.int64)


def are_triangles_incompatible(
    t1_coords: tuple[int, int, int],
    t2_coords: tuple[int, int, int],
    points: NDArray[np.int64],
) -> bool:
    """Return ``True`` if two triangles have a proper edge intersection."""
    t1_arr = np.array([points[idx] for idx in t1_coords], dtype=np.int64)
    t2_arr = np.array([points[idx] for idx in t2_coords], dtype=np.int64)

    # AABB quick-reject
    t1x, t1y = t1_arr[:, 0], t1_arr[:, 1]
    t2x, t2y = t2_arr[:, 0], t2_arr[:, 1]
    if np.max(t1x) < np.min(t2x) or np.min(t1x) > np.max(t2x) or np.max(t1y) < np.min(t2y) or np.min(t1y) > np.max(t2y):
        return False

    edges1 = [(t1_arr[0], t1_arr[1]), (t1_arr[1], t1_arr[2]), (t1_arr[2], t1_arr[0])]
    edges2 = [(t2_arr[0], t2_arr[1]), (t2_arr[1], t2_arr[2]), (t2_arr[2], t2_arr[0])]
    for e1 in edges1:
        for e2 in edges2:
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False


def incompatible_triangles(
    triangles: NDArray[np.int64],
    points: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return an ``(P, 2)`` array of index pairs of incompatible triangles."""
    incompatible_pairs: list[list[int]] = []
    n = len(triangles)
    tri_tuples = [tuple(tri) for tri in triangles]
    for i in range(n):
        for j in range(i + 1, n):
            if are_triangles_incompatible(
                tri_tuples[i],
                tri_tuples[j],
                points,
            ):
                incompatible_pairs.append([i, j])
    if incompatible_pairs:
        return np.array(incompatible_pairs, dtype=np.int64)
    return np.empty((0, 2), dtype=np.int64)


def compute_crossing_edges(
    triangles: NDArray[np.int64],
    points: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return an ``(N, 4)`` array of crossing triangle-edge pairs.

    Each row ``[u, v, w, z]`` encodes two crossing edges where ``u > v``,
    ``w > z``, and the pair is ordered lexicographically by first index.
    """
    if len(triangles) == 0:
        return np.empty((0, 4), dtype=np.int64)

    edges_data: list[dict[str, Any]] = []
    edge_set: set[tuple[int, int]] = set()

    for tri in triangles:
        tri_indices = [int(tri[0]), int(tri[1]), int(tri[2])]
        tri_coords = [points[idx] for idx in tri_indices]
        for idx in range(3):
            v1_idx = tri_indices[idx]
            v2_idx = tri_indices[(idx + 1) % 3]
            edge_key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            c1 = tuple(int(x) for x in tri_coords[idx])
            c2 = tuple(int(x) for x in tri_coords[(idx + 1) % 3])
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges_data.append({"indices": edge_key, "coords": (c1, c2)})

    crossing_segments: list[list[int]] = []
    m = len(edges_data)
    for i in range(m):
        for j in range(i + 1, m):
            e1 = edges_data[i]
            e2 = edges_data[j]
            c1_a, c1_b = e1["coords"]
            c2_a, c2_b = e2["coords"]
            p1 = np.array(c1_a, dtype=np.int64)
            p2 = np.array(c1_b, dtype=np.int64)
            p3 = np.array(c2_a, dtype=np.int64)
            p4 = np.array(c2_b, dtype=np.int64)
            if segments_intersect(p1, p2, p3, p4):
                u_raw, v_raw = e1["indices"]
                w_raw, z_raw = e2["indices"]
                seg1 = (max(u_raw, v_raw), min(u_raw, v_raw))
                seg2 = (max(w_raw, z_raw), min(w_raw, z_raw))
                if seg1[0] < seg2[0]:
                    row = [seg1[0], seg1[1], seg2[0], seg2[1]]
                elif seg2[0] < seg1[0]:
                    row = [seg2[0], seg2[1], seg1[0], seg1[1]]
                elif seg1[1] <= seg2[1]:
                    row = [seg1[0], seg1[1], seg2[0], seg2[1]]
                else:
                    row = [seg2[0], seg2[1], seg1[0], seg1[1]]
                crossing_segments.append(row)

    if not crossing_segments:
        return np.empty((0, 4), dtype=np.int64)

    arr = np.array(crossing_segments, dtype=np.int64)
    ind = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[ind]


def iter_directed_crossing_pairs(
    crossing_edges: NDArray[np.int64],
) -> Iterable[tuple[Arc, Arc]]:
    """Expand each undirected crossing pair into four directed arc combos.

    Preconditions
    -------------
    - ``crossing_edges`` has shape ``(M, 4)`` and dtype ``np.int64``.
    - Each row ``(a, b, c, d)`` is in canonical lex order as produced by
      :func:`compute_crossing_edges`: ``a >= b``, ``c >= d``, ``a >= c``,
      and if ``a == c`` then ``b <= d``.

    Postconditions
    --------------
    For each row ``(a, b, c, d)`` the generator yields exactly 4 tuples in
    the following order:

    1. ``((a, b), (c, d))``   — forward / forward   (F/F)
    2. ``((b, a), (d, c))``   — reverse / reverse   (R/R)
    3. ``((a, b), (d, c))``   — forward / reverse   (F/R)
    4. ``((b, a), (c, d))``   — reverse / forward   (R/F)

    If ``crossing_edges`` is empty (shape ``(0, 4)``), nothing is yielded.

    The caller is responsible for filtering out arcs that are absent from the
    model's ``x`` dictionary — this function is a pure geometry helper with no
    Gurobi dependency.

    Yields
    ------
    tuple[Arc, Arc]
        Pairs of directed arcs ``((u, v), (w, z))`` representing a directed
        crossing pair candidate.
    """
    for row in crossing_edges:
        a, b, c, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        yield (a, b), (c, d)  # F/F
        yield (b, a), (d, c)  # R/R
        yield (a, b), (d, c)  # F/R
        yield (b, a), (c, d)  # R/F


# ---------------------------------------------------------------------------
# Triangle metrics
# ---------------------------------------------------------------------------


def cost_function_area(
    points: NDArray[np.int64],
    x_array: Iterable[tuple[int, int]],
    mode: int = 0,
) -> dict[tuple[int, int], float]:
    """Return edge cost coefficients for the area objective.

    *mode* selects the shoelace variant (0–3).
    """
    c: dict[tuple[int, int], float] = {}
    if mode == 0:
        max_val = int(np.max(points)) if len(points) > 0 else 0
        r = np.array([max_val + 1, max_val + 1])
        for i, j in x_array:
            c[(i, j)] = signed_area(r, points[i], points[j])
    elif mode == 1:
        for i, j in x_array:
            p1, p2 = points[i], points[j]
            c[(i, j)] = (p1[0] * p2[1] - p2[0] * p1[1]) / 2
    elif mode == 2:
        for i, j in x_array:
            p1, p2 = points[i], points[j]
            c[(i, j)] = ((p1[0] + p2[0]) * (p1[1] - p2[1])) / 2
    elif mode == 3:
        for i, j in x_array:
            p1, p2 = points[i], points[j]
            c[(i, j)] = ((p1[1] + p2[1]) * (p1[0] - p2[0])) / 2
    else:
        raise ValueError(f"cost_function_area: mode {mode!r} is not implemented (valid: 0–3)")
    return c


def triangles_area(
    triangles: NDArray[np.int64],
    points: NDArray[np.int64],
) -> list[float]:
    """Return the signed area of each triangle."""
    return [signed_area(points[tri[0]], points[tri[1]], points[tri[2]]) for tri in triangles]


def triangles_adjacency_list(
    triangles: NDArray[np.int64],
    p: NDArray[np.int64],
) -> list[list[list[int]]]:
    """Build a 2-D adjacency list: ``ta_arc[i][j]`` = list of triangle IDs on arc (i→j)."""
    n = len(p)
    ta_arc: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(n)]
    for tid, t in enumerate(triangles):
        i, j, k = int(t[0]), int(t[1]), int(t[2])
        if signed_area(p[i], p[j], p[k]) > 0:
            ta_arc[i][j].append(tid)
            ta_arc[j][k].append(tid)
            ta_arc[k][i].append(tid)
        else:
            ta_arc[j][i].append(tid)
            ta_arc[k][j].append(tid)
            ta_arc[i][k].append(tid)
    return ta_arc


def minimal_triangle_adjency_list(
    triangles: NDArray[np.int64],
    p: NDArray[np.int64],
) -> dict[tuple[int, int], list[int]]:
    """Build a sparse adjacency dict: ``ta_arc[(i, j)]`` = list of triangle IDs on arc (i→j)."""
    ta_arc: dict[tuple[int, int], list[int]] = {}
    for tid, t in enumerate(triangles):
        i, j, k = int(t[0]), int(t[1]), int(t[2])
        if signed_area(p[i], p[j], p[k]) > 0:
            edges: list[tuple[int, int]] = [(i, j), (j, k), (k, i)]
        else:
            edges = [(j, i), (k, j), (i, k)]
        for edge in edges:
            ta_arc.setdefault(edge, []).append(tid)
    return ta_arc


# ---------------------------------------------------------------------------
# File I/O — pre-processing file
# ---------------------------------------------------------------------------


def write_prefile(file_path: str) -> None:
    """Compute geometric structures for an instance and write a ``.pre`` file."""
    points = read_indexed_instance(file_path)
    n_points = len(points)

    hull = compute_convex_hull(points)
    triangles = compute_triangles(points)
    crossing = compute_crossing_edges(triangles, points)
    incompatible = incompatible_triangles(triangles, points)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join("outputs", "Pre-files", f"{base_name.split('.')[0]}.pre")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w") as fh:
        fh.write(f"# Preprocesing information of intance {base_name}\n")
        fh.write("# Points\tConvex_Hull\tTriangles\tCrossing_segments\tIncompatible_Triangles\tTime\n")
        fh.write(f"{n_points}\t{len(hull)}\t{len(triangles)}\t{len(crossing)}\t{len(incompatible)}\t0.0\t0.0\n")
        fh.write("POINTS\n")
        for idx, pt in enumerate(points):
            fh.write(f"{idx}\t{float(pt[0])}\t{float(pt[1])}\n")
        fh.write("CONVEX_HULL\n")
        for idx, v in enumerate(hull):
            fh.write(f"{idx}\t{v}\n")
        fh.write("TRIANGLES\n")
        for idx, tri in enumerate(triangles):
            area = signed_area(points[tri[0]], points[tri[1]], points[tri[2]])
            fh.write(f"{idx}\t{tri[0]}\t{tri[1]}\t{tri[2]}\t{area}\n")
        fh.write("CROSSING_SEGMENTS\n")
        for idx, seg in enumerate(crossing):
            fh.write(f"{idx}\t{seg[0]}\t{seg[1]}\t{seg[2]}\t{seg[3]}\n")
        fh.write("INCOMPATIBLE_TRIANGLES\n")
        for idx, pair in enumerate(incompatible):
            fh.write(f"{idx}\t{pair[0]}\t{pair[1]}\n")
        fh.write(
            "Objetivo\tSolver\tModelo\tNombre\tn\tseg_cruzan_fort\t"
            "rest_u_fort\tpos_u_fort\tfijar_var_0\tstatus\tTiempo\tArea\n"
        )


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------


def extract_metric_to_csv(
    input_tsv: str,
    output_csv: str,
    metric: str = "LPvalue",
) -> None:
    """Extract MIN/MAX columns for *metric* from a TSV and write a tidy CSV."""
    min_col = f"MIN_{metric}"
    max_col = f"MAX_{metric}"

    try:
        with (
            open(input_tsv, mode="r", encoding="utf-8") as tsv_fh,
            open(output_csv, mode="w", newline="", encoding="utf-8") as csv_fh,
        ):
            reader = csv.DictReader(tsv_fh, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"Empty TSV file: {input_tsv}")
            if min_col not in reader.fieldnames or max_col not in reader.fieldnames:
                raise ValueError(f"Columns '{min_col}' and/or '{max_col}' not found in {input_tsv}.")
            writer = csv.writer(csv_fh)
            writer.writerow(["instance", f"{metric.lower()}_min", f"{metric.lower()}_max"])
            for row in reader:
                writer.writerow([row["instance"], row[min_col], row[max_col]])
        print(f"Archivo '{output_csv}' generado correctamente con la métrica '{metric}'.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_tsv}'.")
    except Exception as exc:
        print(f"Error inesperado: {exc}")
