# SIMPLIFIED OVER SIMPLIFIED PROBLEM
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import gurobipy as gp
import os
import csv
from collections.abc import Iterable
from typing import Any, cast
import json
import seaborn as sns
from matplotlib.lines import Line2D
import plotly.graph_objects as go

Arc = tuple[int, int]
SerializedCoeffMap = dict[str, float]
SerializedExpr = dict[str, SerializedCoeffMap | float]
SerializedRayData = dict[str, SerializedCoeffMap | float]
PointLookup = dict[int, tuple[float, float]] | NDArray[np.int64]
ArcConstraintMap = dict[Arc, gp.Constr]


def read_data(file_path: str) -> list[str]:
    """Reads data from a file and returns it as a list of lines."""
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def read_indexed_instance(filepath: str) ->NDArray[np.int64]:
    """Lee el archivo .instance o .pre y devuelve las coordenadas de los puntos."""
    # Read file to detect format
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if it's a .pre file with POINTS section
    points_section = False
    for line in lines:
        if line.strip() == "POINTS":
            points_section = True
            break
    
    if points_section:
        # Handle .pre format with sections
        coords_list = []
        in_points = False
        for line in lines:
            stripped = line.strip()
            if stripped == "POINTS":
                in_points = True
                continue
            elif in_points:
                # Check if we've reached the next section
                if stripped in ["CONVEX_HULL", "TRIANGLES", "CROSSING_SEGMENTS", "INCOMPATIBLE_TRIANGLES"] or stripped == "":
                    if stripped != "":  # Stop at next section
                        break
                    continue  # Skip empty lines
                # Parse point data (index, x, y)
                if not stripped.startswith("#"):
                    parts = stripped.split()
                    if len(parts) >= 3:
                        # Convert to int64 (may be floats in file)
                        x = int(float(parts[1]))
                        y = int(float(parts[2]))
                        coords_list.append([x, y])
        
        if not coords_list:
            raise ValueError(f"No points found in POINTS section of {filepath}")
        
        return np.array(coords_list, dtype=np.int64)
    else:
        # Handle simple .instance format
        data = np.loadtxt(filepath, comments="#", dtype=np.int64)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 3:
            raise ValueError(f"Expected at least three integer columns in {filepath}")

        coords = data[:, 1:3].astype(np.int64)

        return coords

def compute_convex_hull(points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Computes the convex hull of a set of points."""
    hull = ConvexHull(points)
    return hull.vertices

def compute_convex_hull_area(points: NDArray[np.int64]) -> float:
    """Area of the convex hull using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    hull_idx = compute_convex_hull(points)
    hull_pts = points[hull_idx]

    x = hull_pts[:, 0]
    y = hull_pts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)

def orientation_2d(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    """Orientation sign for coordinates (x,y) tuples.
    Returns 0 for collinear, 1 for clockwise, -1 for counterclockwise.
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
    """Return True if segments p1q1 and p2q2 intersect properly (strictly in interiors).
    Excludes cases where they only touch at endpoints or are collinear/overlapping.
    Points are (x, y) tuples.
    """
    o1 = orientation_2d(p1, q1, p2)
    o2 = orientation_2d(p1, q1, q2)
    o3 = orientation_2d(p2, q2, p1)
    o4 = orientation_2d(p2, q2, q1)
    # Proper intersection requires strict opposite orientations on both segments
    return (o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and (o1 != o2) and (o3 != o4))


def point_in_triangle(pt: NDArray[np.int64], v1: NDArray[np.int64], v2: NDArray[np.int64], v3: NDArray[np.int64]) -> bool:
    """Checks if a point is inside a triangle defined by three vertices."""
    d1 = (pt[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (pt[1] - v2[1])
    d2 = (pt[0] - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[0]) * (pt[1] - v3[1])
    d3 = (pt[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (pt[1] - v1[1])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def compute_triangles(points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Computes valid triangles from a set of points."""
    n = len(points)

    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    
    hull = compute_convex_hull(points)
    hull_edges = np.sort(np.array([(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))], dtype=np.int64), axis=1)
    hull_mask = np.zeros(n, dtype=bool)
    hull_mask[hull] = True

    valid_triangles: list[NDArray[np.int64]] = []

    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]        
        if cross_product == 0:  # Colinear
            continue

        """        # Ensure CCW ordering of indices
        if cross_product > 0:
            tri_idx = np.array([i, j, k], dtype=np.int64)
        else:
            tri_idx = np.array([i, k, j], dtype=np.int64)"""

        tri_idx = np.array([i, j, k], dtype=np.int64)
        
        # Check if triangle has a convex hull edge oriented clockwise and adjust
        # Hull edges are oriented CCW: hull[i] -> hull[(i+1) % len(hull)]
        for h_idx in range(len(hull)):
            hull_edge_cw = (hull[(h_idx + 1) % len(hull)], hull[h_idx])
            
            # Check all three edges of the triangle
            tri_edges_directed = [
                (tri_idx[0], tri_idx[1]),
                (tri_idx[1], tri_idx[2]),
                (tri_idx[2], tri_idx[0])
            ]
            
            for edge_pos, (v1, v2) in enumerate(tri_edges_directed):
                if (v1, v2) == hull_edge_cw:
                    # Found a hull edge oriented clockwise in the triangle - flip triangle
                    tri_idx = np.array([tri_idx[0], tri_idx[2], tri_idx[1]], dtype=np.int64)
                    break

        # Build triangle edges (undirected, sorted)
        tri_edges = np.array([
            [tri_idx[0], tri_idx[1]],
            [tri_idx[1], tri_idx[2]],
            [tri_idx[2], tri_idx[0]],
        ], dtype=np.int64)
        tri_edges = np.sort(tri_edges, axis=1)

        # Reject if any edge connects two hull vertices but is not a hull edge
        hull_hull = hull_mask[tri_edges].all(axis=1)
        if hull_hull.any():
            # membership test via broadcasting
            # For each tri edge, check if it matches any hull edge
            matches = (tri_edges[:, None, :] == hull_edges[None, :, :]).all(axis=2).any(axis=1)
            if (hull_hull & ~matches).any():
                continue

        # Check containment of other points (boundary inclusive)
        contains_other_point = False
        for m in range(n):
            if m in (i, j, k):
                continue
            if point_in_triangle(points[m], points[tri_idx[0]], points[tri_idx[1]], points[tri_idx[2]]):
                contains_other_point = True
                break

        if not contains_other_point:
            valid_triangles.append(tri_idx)

    tris = np.array(valid_triangles, dtype=np.int64) if valid_triangles else np.empty((0, 3), dtype=np.int64)
    return tris


        
def segments_intersect(p1: NDArray[np.int64], p2: NDArray[np.int64], p3: NDArray[np.int64], p4: NDArray[np.int64]) -> bool:
    """Checks if two line segments (p1-p2) and (p3-p4) intersect, excluding endpoint-only intersections."""
    def ccw(A: NDArray[np.int64], B: NDArray[np.int64], C: NDArray[np.int64]) -> np.bool_:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # Check if segments share any endpoints
    if np.array_equal(p1, p3) or np.array_equal(p1, p4) or np.array_equal(p2, p3) or np.array_equal(p2, p4):
        return False
    
    # Check if segments properly intersect
    d1 = ccw(p1, p3, p4) != ccw(p2, p3, p4)
    d2 = ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    return d1 and d2

def compute_crossing_edges(triangles: NDArray[np.int64], points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Computes pairs of edges from triangles that cross each other.
    
    Returns array of shape (N, 4) where each row is [u, v, w, z].
    Normalized so u > v, w > z, and the segment (u,v) is lexicographically 
    smaller than (w,z) based on the first index.
    """
    if len(triangles) == 0:
        return np.empty((0, 4), dtype=np.int64)
    
    # 1. Collect all unique triangle edges
    edges_data = []
    edge_set = set()
    
    for tri in triangles:
        # Usamos los índices directos
        tri_indices = [tri[0], tri[1], tri[2]]
        tri_coords = [points[idx] for idx in tri_indices]
        
        for i in range(3):
            v1_idx = tri_indices[i]
            v2_idx = tri_indices[(i + 1) % 3]
            
            # Guardamos internamente como (min, max) para el set (evitar duplicados)
            # Esto está bien para la lógica de unicidad.
            edge_idx = tuple(sorted([v1_idx, v2_idx]))
            
            c1 = tuple(int(x) for x in tri_coords[i])
            c2 = tuple(int(x) for x in tri_coords[(i + 1) % 3])
            edge_coords = (c1, c2)
            
            if edge_idx not in edge_set:
                edge_set.add(edge_idx)
                edges_data.append({
                    'indices': edge_idx, # (min, max)
                    'coords': edge_coords
                })
    
    # 2. Find crossing pairs
    crossing_segments = []
    n = len(edges_data)
    
    for i in range(n):
        for j in range(i + 1, n):
            edge1 = edges_data[i]
            edge2 = edges_data[j]
            
            # Intersection test geometry
            c1_a, c1_b = edge1['coords']
            c2_a, c2_b = edge2['coords']
            
            p1 = np.array(c1_a, dtype=np.int64)
            p2 = np.array(c1_b, dtype=np.int64)
            p3 = np.array(c2_a, dtype=np.int64)
            p4 = np.array(c2_b, dtype=np.int64)
            
            # Suponemos que segments_intersect devuelve True SOLO si cruzan
            # propiamente (no si comparten un vertice final)
            if segments_intersect(p1, p2, p3, p4):
                
                # --- AQUÍ ESTÁ EL CAMBIO CLAVE (Normalización al vuelo) ---
                
                # 1. Recuperar índices
                u_raw, v_raw = edge1['indices']
                w_raw, z_raw = edge2['indices']
                
                # 2. Aplicar regla "Mayor > Menor" (u > v)
                # Tu código original guardaba (min, max), aquí lo invertimos
                seg1 = (max(u_raw, v_raw), min(u_raw, v_raw))
                seg2 = (max(w_raw, z_raw), min(w_raw, z_raw))
                
                # 3. Aplicar regla de orden entre segmentos
                # El segmento que empieza con el número más bajo va primero a la izquierda
                if seg1[0] < seg2[0]:
                    row = [seg1[0], seg1[1], seg2[0], seg2[1]]
                elif seg2[0] < seg1[0]:
                    row = [seg2[0], seg2[1], seg1[0], seg1[1]]
                else:
                    # Si los primeros puntos son iguales (raro en cruces propios, 
                    # pero posible geométricamente), desempatar con el segundo punto.
                    if seg1[1] < seg2[1]:
                        row = [seg1[0], seg1[1], seg2[0], seg2[1]]
                    else:
                        row = [seg2[0], seg2[1], seg1[0], seg1[1]]
                
                crossing_segments.append(row)
    
    # 3. Convert to numpy array and sort
    if crossing_segments:
        crossing_array = np.array(crossing_segments, dtype=np.int64)
        
        # Ahora el lexsort funcionará perfecto porque los datos ya tienen la estructura correcta
        # Orden de prioridad (de abajo hacia arriba): Col 0, Col 1, Col 2, Col 3
        ind = np.lexsort((
            crossing_array[:, 3], 
            crossing_array[:, 2], 
            crossing_array[:, 1], 
            crossing_array[:, 0]
        ))
        return crossing_array[ind]
    else:
        return np.empty((0, 4), dtype=np.int64)

    
def are_triangles_incompatible(
    t1_coords: tuple[int, int, int],
    t2_coords: tuple[int, int, int],
    points: NDArray[np.int64],
) -> bool:
    """Coordinate-based incompatibility test between two triangles.
    
    Two triangles are incompatible ONLY if their edges properly intersect
    (not just touching at vertices/endpoints).
    This is a stricter definition focusing on geometric conflicts.
    """
    # Convert indices to arrays for geometric tests
    t1_arr = np.array([points[idx] for idx in t1_coords], dtype=np.int64)
    t2_arr = np.array([points[idx] for idx in t2_coords], dtype=np.int64)
    
    # 1) Axis-aligned bounding box disjointness (quick reject)
    t1x = t1_arr[:, 0]
    t1y = t1_arr[:, 1]
    t2x = t2_arr[:, 0]
    t2y = t2_arr[:, 1]
    
    if (np.max(t1x) < np.min(t2x) or np.min(t1x) > np.max(t2x) or
        np.max(t1y) < np.min(t2y) or np.min(t1y) > np.max(t2y)):
        return False
    
    # 2) Proper edge intersection is the ONLY condition for incompatibility
    edges1 = [
        (t1_arr[0], t1_arr[1]),
        (t1_arr[1], t1_arr[2]),
        (t1_arr[2], t1_arr[0])
    ]
    edges2 = [
        (t2_arr[0], t2_arr[1]),
        (t2_arr[1], t2_arr[2]),
        (t2_arr[2], t2_arr[0])
    ]
    
    for e1 in edges1:
        for e2 in edges2:
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    
    return False

def incompatible_triangles(triangles: NDArray[np.int64], points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Finds pairs of triangles that are incompatible.
    
    Two triangles are incompatible if their interiors overlap (not disjoint).
    Uses coordinate-based predicates to distinguish proper intersections and
    strict containment.
    """
    incompatible_pairs = []
    n = len(triangles)
    
    # Convert triangles to tuples for comparison
    tri_tuples = [tuple(tri) for tri in triangles]
    
    for i in range(n):
        for j in range(i + 1, n):
            if are_triangles_incompatible(tri_tuples[i], tri_tuples[j], points):
                incompatible_pairs.append([i, j])
    
    return np.array(incompatible_pairs, dtype=np.int64) if incompatible_pairs else np.empty((0, 2), dtype=np.int64)

def signed_area(p1: NDArray[np.int64], p2: NDArray[np.int64], p3: NDArray[np.int64]) -> float:
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def cost_function_area(
    points: NDArray[np.int64],
    x_array: Iterable[tuple[int, int]],
    mode: int = 0,
) -> dict[tuple[int, int], float]:
    """Calculates the cost function based on points and possible selected edges."""
    c = {}
    if mode == 0:
        max_val = np.max(points) if len(points) > 0 else 0
        r = np.array([max_val + 1, max_val + 1])
        for i, j in x_array:
            p1 = points[i]
            p2 = points[j]
            area = (signed_area(r, p1, p2))
            c[(i, j)] = area

    elif mode == 1:
        for i, j in x_array:
            p1 = points[i]
            p2 = points[j]
            area = (p1[0]*p2[1] - p2[0]*p1[1])/2
            c[(i, j)] = area
    
    elif mode == 2:
        for i, j in x_array:
            p1 = points[i]
            p2 = points[j]
            area = ((p1[0]+p2[0]) * (p1[1]-p2[1]))/2
            c[(i, j)] = area

    elif mode == 3:
        for i, j in x_array:
            p1 = points[i]
            p2 = points[j]
            area =  ((p1[1]+p2[1]) * (p1[0]-p2[0]))/2
            c[(i, j)] = area
        
    else:
        print("Cost function mode not implemented")
        quit()

    return c

def triangles_area(triangles: NDArray[np.int64], points: NDArray[np.int64]) -> list[float]:
    # Calculate signed area for each triangle
    triangle_areas = []
    for tri in triangles:
        triangle_areas.append(signed_area(points[tri[0]], points[tri[1]], points[tri[2]]))
    
    return triangle_areas


def triangles_adjacency_list(triangles: NDArray[np.int64], p: NDArray[np.int64]) -> list[list[list[int]]]:
    ta_arc = []

    for i in range(len(p)):
        ta_arc.append([])
        for j in range(len(p)):
            ta_arc[i].append([])

    for id, t in enumerate(triangles):
        i, j, k = t[0], t[1], t[2]
        if signed_area(p[i], p[j], p[k]) > 0:
            ta_arc[i][j].append(id)
            ta_arc[j][k].append(id)
            ta_arc[k][i].append(id)
        else:
            ta_arc[j][i].append(id)
            ta_arc[k][j].append(id)
            ta_arc[i][k].append(id)
    return ta_arc

def minimal_triangle_adjency_list(
    triangles: NDArray[np.int64],
    p: NDArray[np.int64],
) -> dict[tuple[int, int], list[int]]:
    ta_arc = {}

    for id, t in enumerate(triangles):
        i, j, k = t[0], t[1], t[2]
        if signed_area(p[i], p[j], p[k]) > 0:
            edge1 = (i, j)
            edge2 = (j, k)
            edge3 = (k, i)
        else:
            edge1 = (j, i)
            edge2 = (k, j)
            edge3 = (i, k)
        
        for edge in [edge1, edge2, edge3]:
            if edge not in ta_arc:
                ta_arc[edge] = []
            ta_arc[edge].append(id)
    
    return ta_arc


def write_prefile(file_path: str) -> None:
    """Writes the computed geometric structures to a .pre file.
    
    Reads an instance file, computes convex hull, triangles, crossing segments,
    and incompatible triangle pairs, then writes results to a .pre file.
    """
    
    # Read instance file
    points = read_indexed_instance(file_path)
    n_points = len(points)
    
    # Compute geometric structures
    hull = compute_convex_hull(points)
    triangles = compute_triangles(points)
    crossing = compute_crossing_edges(triangles, points)
    incompatible = incompatible_triangles(triangles, points)
    
    
    # Generate output filename: nXsY.pre where X=num points, Y=seed from input filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    out_file = os.path.join("outputs", "Pre-files", f"{base_name.split('.')[0]}.pre")
    
    # Write .pre file
    with open(out_file, 'w') as f:
        # Header with statistics
        f.write(f"# Preprocesing information of intance {base_name}\n")
        f.write("# Points\tConvex_Hull\tTriangles\tCrossing_segments\tIncompatible_Triangles\tTime\n")
        f.write(f"{n_points}\t{len(hull)}\t{len(triangles)}\t{len(crossing)}\t{len(incompatible)}\t0.0\t0.0\n")
        
        # POINTS section
        f.write("POINTS\n")
        for i, pt in enumerate(points):
            f.write(f"{i}\t{float(pt[0])}\t{float(pt[1])}\n")
        
        # CONVEX_HULL section
        f.write("CONVEX_HULL\n")
        for i, v in enumerate(hull):
            f.write(f"{i}\t{v}\n")
        
        # TRIANGLES section
        f.write("TRIANGLES\n")
        for i, tri in enumerate(triangles):
            f.write(f"{i}\t{tri[0]}\t{tri[1]}\t{tri[2]}\t{signed_area(points[tri[0]], points[tri[1]], points[tri[2]])}\n")
        
        # CROSSING_SEGMENTS section
        f.write("CROSSING_SEGMENTS\n")
        for i, seg in enumerate(crossing):
            f.write(f"{i}\t{seg[0]}\t{seg[1]}\t{seg[2]}\t{seg[3]}\n")
        
        # INCOMPATIBLE_TRIANGLES section
        f.write("INCOMPATIBLE_TRIANGLES\n")
        for i, pair in enumerate(incompatible):
            f.write(f"{i}\t{pair[0]}\t{pair[1]}\n")
        
        # Footer
        f.write("Objetivo\tSolver\tModelo\tNombre\tn\tseg_cruzan_fort\trest_u_fort\tpos_u_fort\tfijar_var_0\tstatus\tTiempo\tArea\n")






def is_colineal(p1: NDArray[np.int64], p2: NDArray[np.int64], p3: NDArray[np.int64]) -> bool:
    """Checks if three points are collinear."""
    return (p2[1] - p1[1]) * (p3[0] - p2[0]) == (p3[1] - p2[1]) * (p2[0] - p1[0])





def plot_solution(model: gp.Model, title: str = "Solution") -> None:
    """Plots the solution of the model by drawing the tour."""

    x = model._x_results
    points = model._points_

    G = nx.DiGraph()
    G.add_edges_from(x)
    
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    
    # Draw edges of the tour
    for edge in G.edges():
        pt1 = points[edge[0]]
        pt2 = points[edge[1]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.7)

    #Show convex hull
    hull = compute_convex_hull(points)
    hull_set = set(hull)
    for i in range(len(hull)):
        pt1 = points[hull[i]]
        pt2 = points[hull[(i + 1) % len(hull)]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-.', alpha=0.5)
    
    # Label all points with their indices
    for i, pt in enumerate(points):
        if i in hull_set:
            # Convex hull points in red and bold
            plt.annotate(str(i), (pt[0], pt[1]), 
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=10, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            # Regular points in black
            plt.annotate(str(i), (pt[0], pt[1]), 
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))    
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()



def extract_metric_to_csv(input_tsv: str, output_csv: str, metric: str = "LPvalue") -> None:
    """
    Lee un archivo TSV y crea un CSV con la instancia y los valores MIN/MAX de una métrica.
    
    Parámetros:
    - input_tsv: Ruta del archivo TSV de entrada.
    - output_csv: Ruta del archivo CSV a generar.
    - metric: El nombre de la métrica a extraer (ej: 'LPvalue', 'IPvalue', 'IPtime').
    """
    
    # Construimos los nombres exactos de las columnas como aparecen en el TSV
    min_col = f"MIN_{metric}"
    max_col = f"MAX_{metric}"
    
    try:
        with open(input_tsv, mode='r', encoding='utf-8') as tsv_file, \
             open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            
            # DictReader mapea la información a un diccionario usando la primera fila como claves
            reader = csv.DictReader(tsv_file, delimiter='\t')
            
            # Verificamos que las columnas solicitadas realmente existen en el TSV
            if min_col not in reader.fieldnames or max_col not in reader.fieldnames:
                raise ValueError(f"No se encontraron las columnas '{min_col}' y/o '{max_col}' en el TSV.")
            
            writer = csv.writer(csv_file)
            
            # Escribir la cabecera (ej: instance, lpvalue_min, lpvalue_max)
            writer.writerow(['instance', f"{metric.lower()}_min", f"{metric.lower()}_max"])
            
            # Escribir los datos fila por fila
            for row in reader:
                writer.writerow([
                    row['instance'], 
                    row[min_col], 
                    row[max_col]
                ])
                
        print(f"✅ Archivo '{output_csv}' generado correctamente con la métrica '{metric}'.")

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{input_tsv}'.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")


def restricciones_semiplano(
    model: gp.Model,
    points: NDArray[np.int64],
    CH: NDArray[np.int64],
) -> tuple[gp.Model, list[gp.Constr]]:
    """Agrega restricciones de semiplano relacionado a CH al modelo para cada arista."""
    
    # 1. FORMA SEGURA: Filtrar índices, no coordenadas.
    A_pp = [i for i in range(len(points)) if i not in CH]
    
    constrains = []
    for i in A_pp:
        for j in CH: 
            # 2. CHECK DE EXISTENCIA OPTIMIZADO: Hacerlo antes de iterar k.
            # En Gurobi, es infinitamente más rápido buscar la tupla en el diccionario de variables
            if (i, j) not in model._x:
                continue
                
            semiplano_izquierdo_limpio = True

            for k in A_pp:
                if k == i:
                    continue
                
                # Calcular el determinante geométrico
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                
                D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                
                if D > 0: # El nodo interno k está a la izquierda
                    semiplano_izquierdo_limpio = False
                    break # 3. EARLY EXIT: ¡Ya falló! No necesitamos revisar más puntos.
            
            # (Eliminado el bloque "if k == i:" problemático que estaba aquí)

            if semiplano_izquierdo_limpio:
                index = np.where(CH == j)[0][0]
                j_siguiente = CH[(index + 1) % len(CH)]
                
                # Buena práctica: comprobar también que la variable destino exista
                if (j, j_siguiente) in model._x:
                    constrains.append(
                        model.addConstr(
                            model._x[i,j] <= model._x[j, j_siguiente],
                            name=f"semiplano_{i}_{j}"
                    ))
    
    return model, constrains


def restricciones_semiplanoV2(
    model: gp.Model,
    points: NDArray[np.int64],
    CH: NDArray[np.int64],
) -> tuple[gp.Model, list[gp.Constr]]:

    # 1. FORMA SEGURA: Filtrar índices, no coordenadas.
    A_pp = [i for i in range(len(points)) if i not in CH]
    
    constrains = []
    for i in A_pp:

        for j in CH:    
            if (i, j) not in model._x:
                continue
            
            # 1. Recopilar TODOS los puntos internos que caen en el semiplano izquierdo
            S_left = []
            for k in A_pp:
                if k == i:
                    continue

                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]

                D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                if D_k > 0: # El punto k está atrapado en el semiplano izquierdo
                    S_left.append(k)

            # Encontrar el siguiente nodo en la Envolvente Convexa
            idx_j = np.where(CH == j)[0][0]
            j_siguiente = CH[(idx_j + 1) % len(CH)]


            # 2. BIFURCACIÓN DE LA LÓGICA

            if len(S_left) == 0:
                # 1. Encontramos la posición del nodo j en la Envolvente Convexa
                idx_j = np.where(CH == j)[0][0]
                nodo_actual_ch = j
                
                # 2. Recorremos la CH en orden antihorario a partir de j
                for step in range(1, len(CH)):
                    idx_siguiente = (idx_j + step) % len(CH)
                    nodo_siguiente_ch = CH[idx_siguiente]
                    
                    # 3. Comprobamos si 'nodo_siguiente_ch' SIGUE estando a la izquierda de la línea i->j infinita
                    x_i, y_i = points[i]
                    x_j, y_j = points[j]
                    x_sig, y_sig = points[nodo_siguiente_ch]
                    
                    D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                    
                    if D_sig > 0: # El nodo de la CH sigue estando en el semiplano izquierdo
                        
                        if (nodo_actual_ch, nodo_siguiente_ch) in model._x:
                            constrains.append(
                                model.addConstr(
                                    model._x[i, j] <= model._x[nodo_actual_ch, nodo_siguiente_ch],
                                    name=f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual_ch}_{nodo_siguiente_ch}"
                                )
                            )
                        # Avanzamos en la cadena
                        nodo_actual_ch = nodo_siguiente_ch
                    
                    else:
                        # En cuanto un nodo de la CH cruza al semiplano derecho, 
                        # el polígono ya tiene libertad geométrica. Rompemos la cadena.
                        break
            else:
                # --- CASO B: PSEUDO-VACÍO (Bolsillo con puntos) ---
                # La obligación se reparte: o vas por el perímetro, o visitas a los puntos atrapados
                expr_escape = gp.LinExpr()

                # Vía de escape 1: El perímetro
                if (j, j_siguiente) in model._x:
                    expr_escape.addTerms(1.0, model._x[j, j_siguiente])

                # Vía de escape 2: Rescatar a los puntos internos atrapados
                for k in S_left:
                    if (j, k) in model._x:
                        expr_escape.addTerms(1.0, model._x[j, k])

                # Añadir la restricción relajada
                constrains.append(
                    model.addConstr(
                        model._x[i, j] <= expr_escape,
                        name=f"bolsillo_{i}_{j}_soporta_{len(S_left)}_puntos"
                    )
                )

    return model, constrains

def inyectar_cortes_knapsack_locales(
    model: gp.Model,
    points: NDArray[np.int64],
    pesos_obj: dict[tuple[int, int], float],
) -> gp.Model:
    """
    Inyecta restricciones de mochila que limitan la contribución fraccionaria máxima
    de las aristas incidentes a cada nodo.
    """
    num_nodos = len(points)
    cortes_añadidos = 0
    
    for i in range(num_nodos):
        
        # 1. Encontrar la mejor PAREJA de aristas VÁLIDA para el nodo i
        max_beneficio_real = 0.0
        
        # Evaluamos todas las posibles parejas de vecinos j1 y j2
        for j1 in range(num_nodos):
            if j1 == i or (i, j1) not in model._x:
                continue
                
            for j2 in range(j1 + 1, num_nodos):
                if j2 == i or (i, j2) not in model._x:
                    continue
                    
                # 2. Condición Geométrica: ¿Es esta pareja legal?
                # Usamos tu lógica de triángulo vacío. Si el triángulo (j1, i, j2) 
                # contiene algún punto 'k', esta pareja es INVÁLIDA y la ignoramos.
                es_pareja_legal = True
                for k in range(num_nodos):
                    if k in [i, j1, j2]:
                        continue
                    if point_in_triangle(points[k], points[j1], points[i], points[j2]):
                        es_pareja_legal = False
                        break
                
                # 3. Si es legal, calculamos su beneficio conjunto
                if es_pareja_legal:
                    # El beneficio es la suma de los coeficientes de ambas aristas
                    beneficio_pareja = pesos_obj[i, j1] + pesos_obj[i, j2]
                    if beneficio_pareja > max_beneficio_real:
                        max_beneficio_real = beneficio_pareja
                        
        # 4. CREAR LA RESTRICCIÓN KNAPSACK PARA EL NODO 'i'
        # La suma de las fracciones elegidas multiplicadas por su peso no puede superar el tope
        expr_knapsack = gp.LinExpr()
        
        for j in range(num_nodos):
            if j != i and (i, j) in model._x:
                # Sumamos W_ij * x_ij
                expr_knapsack.addTerms(pesos_obj[i, j], model._x[i, j])
                
        # Añadir al modelo
        if expr_knapsack.size() > 0:
            model.addConstr(
                expr_knapsack <= max_beneficio_real,
                name=f"knapsack_local_nodo_{i}"
            )
            cortes_añadidos += 1
            
    print(f"Inyectados {cortes_añadidos} Cortes Knapsack Locales.")
    return model


def inyectar_cliques_de_cruce(model: gp.Model, points: NDArray[np.int64]) -> gp.Model:
    """
    Busca grupos de arcos que se cruzan TODOS entre sí (Cliques) 
    y restringe que como máximo 1 de ellos puede estar activo.
    """
    print("Construyendo grafo de intersecciones para Cliques...")
    
    # 1. Recopilar todas las variables (aristas) válidas del modelo
    aristas = []
    for i in range(len(points)):
        for j in range(len(points)):
            if (i, j) in model._x:
                # Solo guardamos una dirección i < j para evitar duplicados en el cruce
                if i < j: 
                    aristas.append((i, j))

    # 2. Construir el Grafo de Intersecciones
    G_cruces = nx.Graph()
    G_cruces.add_nodes_from(aristas)
    
    # Evaluar qué pares de aristas se cruzan
    # (Asume que tienes una función 'se_cruzan(p1, p2, p3, p4)' que devuelve True/False)
    for idx, e1 in enumerate(aristas):
        for e2 in aristas[idx+1:]:
            # Si comparten un nodo, no se "cruzan" en el sentido estricto (forman un ángulo)
            if e1[0] in e2 or e1[1] in e2:
                continue
                
            p1, p2 = points[e1[0]], points[e1[1]]
            p3, p4 = points[e2[0]], points[e2[1]]
            
            if segments_intersect(p1, p2, p3, p4):
                G_cruces.add_edge(e1, e2)
                
    # 3. Encontrar Cliques Maximales
    # find_cliques encuentra grupos maximales donde todos los nodos están conectados
    cliques = list(nx.find_cliques(G_cruces))
    
    cortes_añadidos = 0
    # 4. Añadir las restricciones al modelo
    for clique in cliques:
        # Solo nos interesan los cliques de tamaño 3 o más. 
        # (Los de tamaño 2 ya están cubiertos por tu modelo base xA + xB <= 1)
        if len(clique) >= 3:
            expr_clique = gp.LinExpr()
            
            for e in clique:
                # Añadir la variable en ambas direcciones si tu modelo es dirigido
                if e in model._x:
                    expr_clique.addTerms(1.0, model._x[e[0], e[1]])
                if (e[1], e[0]) in model._x:
                    expr_clique.addTerms(1.0, model._x[e[1], e[0]])
                
            model.addConstr(expr_clique <= 1, name=f"clique_cruce_{cortes_añadidos}")
            cortes_añadidos += 1
            
    print(f"¡Inyectados {cortes_añadidos} Cortes de Clique de Cruces!")
    return model




def log_farkas_ray(filepath: str, iteration: int, node_depth: int, subproblem_type: str, 
                   x_sol: dict[tuple[int, int], float], v_components: dict[str, Any], violation_value: float, tolerance: float = 1e-5, cut_expr: gp.LinExpr | None = None) -> None:
    """
    Registra la información de un rayo de Farkas y la solución candidata en un archivo JSONL.
    
    Args:
        filepath: Ruta del archivo .jsonl donde se guardará el log.
        iteration: Número de iteración o contador de cortes.
        node_depth: Profundidad del árbol de exploración (0 para nodo raíz/soluciones enteras).
        subproblem_type: 'Y' o 'Y_prime'.
        x_sol: Diccionario con los valores de la solución maestra actual.
        v_components: Diccionario con los componentes del rayo (alpha, beta, etc.).
        violation_value: El valor numérico de la violación del corte.
        tolerance: Valor por debajo del cual se considera que una variable es 0.
        cut_expr: Expresión del corte generado (para debugging o análisis).
    """
    # 1. Filtrar x_sol: Guardar solo los arcos activos para no saturar el log
    active_x: SerializedCoeffMap = {
        f"{i}_{j}": round(val, 4)
        for (i, j), val in x_sol.items()
        if abs(val) > tolerance
    }

    # 2. Estructurar los componentes del rayo
    ray_data: SerializedRayData = {}
    for comp_name, values in v_components.items():
        if values:
            if isinstance(values, dict):
                typed_values = cast(dict[Arc, float], values)
                ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in typed_values.items()}
            else:
                ray_data[comp_name] = round(cast(float, values), 4)

    serialized_cut = serialize_expr(cut_expr) if cut_expr else None

    # 3. Crear el registro consolidado
    registro: dict[str, Any] = {
        "iteration": iteration,
        "node_depth": node_depth,
        "subproblem": subproblem_type,
        "violation": round(violation_value, 6),
        "active_x": active_x,
        "ray_components": ray_data,
        "cut_expr": serialized_cut,
    }
    
    # 4. Asegurar que el directorio existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 5. Escribir en formato JSON Lines (append mode)
    with open(filepath, 'a') as f:
        f.write(json.dumps(registro) + '\n')




def serialize_expr(expr: gp.LinExpr | None) -> SerializedExpr | None:
    """Convierte una expresión lineal de Gurobi en un diccionario serializable."""
    if expr is None:
        return None
    
    # Creamos un diccionario: {"nombre_variable": coeficiente}
    # Por ejemplo: {"x_0_1": 1.0, "x_2_3": -1.0}
    coeffs: dict[str, float] = {}
    for i in range(expr.size()):
        var = expr.getVar(i)
        coeff = expr.getCoeff(i)
        if abs(coeff) > 1e-6:
            coeffs[var.VarName] = round(coeff, 4)
    
    return {
        "coeffs": coeffs,
        "constant": round(expr.getConstant(), 4)
    }


def format_cut_string(cut_expr: SerializedExpr) -> str:
    """Convierte el log entry en algo como '1.0 <= x_0_1 + x_2_1'"""
    const = cast(float, cut_expr.get('constant', 0.0))
    coeffs = cast(SerializedCoeffMap, cut_expr.get('coeffs', {}))
    
    # Siguiendo la lógica: const <= sum(-coeff * x)
    parts: list[str] = []
    for var, val in coeffs.items():
        if val < 0:
            parts.append(f"{abs(val)}*{var}" if abs(val) != 1 else var)
        elif val > 0:
            parts.append(f"- {val}*{var}" if val != 1 else f"- {var}")
            
    formula = " + ".join(parts).replace("+ -", "- ")
    return f"Corte Lógico: {const} <= {formula}"


def load_farkas_logs(filepath: str) -> list[dict[str, Any]]:
    """Carga el historial de rayos de Farkas desde un archivo .jsonl."""
    logs: list[dict[str, Any]] = []
    if not os.path.exists(filepath):
        print(f"El archivo {filepath} no existe.")
        return logs
        
    with open(filepath, 'r') as f:
        for line in f:
            logs.append(cast(dict[str, Any], json.loads(line.strip())))
    return logs

def parse_edge(edge_str: str) -> tuple[int, int]:
    """Convierte un string 'i_j' en una tupla de enteros (i, j)."""
    i, j = edge_str.split('_')
    return int(i), int(j)





def plot_farkas_ray_network(
    log_entry: dict[str, Any],
    points: PointLookup | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    Dibuja la red mostrando la solución candidata x_bar y los componentes del rayo de Farkas.
    
    Args:
        log_entry: Un diccionario extraído del archivo .jsonl.
        points: Diccionario con las coordenadas de los nodos {id: (x, y)}. Si es None, se autogeneran.
        save_path: Ruta donde guardar la imagen. Si es None, se muestra en pantalla.
    """
    nx_api: Any = nx
    plt_api: Any = plt
    G: Any = nx_api.DiGraph()
    
    # Extraer datos del log
    iteration = log_entry.get("iteration", "N/A")
    subproblem = log_entry.get("subproblem", "Unknown")
    violation = log_entry.get("violation", 0.0)
    active_x_strs = cast(SerializedCoeffMap, log_entry.get("active_x", {}))
    ray_components = cast(SerializedRayData, log_entry.get("ray_components", {}))

    # 1. Añadir aristas de la solución candidata (x_bar)
    active_edges: list[Arc] = [parse_edge(e) for e in active_x_strs.keys()]
    G.add_edges_from(active_edges)

    # 2. Configurar layout (posiciones de los nodos)
    pos: dict[int, tuple[float, float]] | dict[int, NDArray[np.float64]]
    if points is not None:
        if isinstance(points, np.ndarray):
            pos = {int(i): (float(points[i][0]), float(points[i][1])) for i in G.nodes()}
        else:
            pos = {int(i): points[int(i)] for i in G.nodes()}
    else:
        pos = cast(dict[int, NDArray[np.float64]], nx_api.spring_layout(G, seed=42))

    plt_api.figure(figsize=(10, 8))
    nx_api.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300, edgecolors='black')
    nx_api.draw_networkx_labels(G, pos, font_size=10)
    nx_api.draw_networkx_edges(
        G,
        pos,
        edgelist=active_edges,
        edge_color='gray',
        style='dashed',
        alpha=0.6,
        width=2,
        arrows=True,
    )

    color_map: dict[str, str] = {
        'alpha': 'red', 'alpha_p': 'darkred',
        'beta': 'blue', 'beta_p': 'darkblue',
        'gamma': 'green', 'gamma_p': 'darkgreen',
        'delta': 'orange', 'delta_p': 'darkorange',
    }

    legend_handles: list[Line2D] = []
    for comp_name, edges_dict in ray_components.items():
        if not edges_dict or not isinstance(edges_dict, dict):
            continue

        color = color_map.get(comp_name, 'purple')
        farkas_edges: list[Arc] = [parse_edge(edge_name) for edge_name in edges_dict.keys()]

        G.add_edges_from(farkas_edges)
        if points is None:
            pos = cast(dict[int, NDArray[np.float64]], nx_api.spring_layout(G, pos=pos, fixed=list(pos.keys()), seed=42))

        nx_api.draw_networkx_edges(
            G,
            pos,
            edgelist=farkas_edges,
            edge_color=color,
            width=2.5,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f'Rayo {comp_name}'))

    legend_handles.append(Line2D([0], [0], color='gray', linestyle='dashed', lw=2, label=r'Solución candidata $\bar{x}$'))

    plt_api.title(f"Iteración: {iteration} | Subproblema: {subproblem}\nViolación: {violation}", fontsize=14)
    plt_api.legend(handles=legend_handles, loc='best')
    plt_api.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()





def plot_cut_heatmap(
    log_entry: dict[str, Any],
    num_nodes: int,
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    plt_api: Any = plt
    sns_api: Any = sns
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data:
        return

    adj_matrix: NDArray[np.float64] = np.zeros((num_nodes, num_nodes))
    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    
    for var_name, coeff in coeffs.items():
        try:
            parts = var_name.split('_')
            i, j = int(parts[1]), int(parts[2])
            adj_matrix[i, j] = coeff
        except Exception:
            continue

    plt_api.figure(figsize=(6, 5))
    sns_api.heatmap(adj_matrix, annot=True, cmap="RdBu", center=0)
    plt_api.title(f"Estructura del Corte - Iteración {log_entry['iteration']}")
    plt_api.xlabel("Nodo Destino (j)")
    plt_api.ylabel("Nodo Origen (i)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()


def plot_cut_weights(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Dibuja un gráfico de barras con los pesos (coeficientes) del corte."""
    plt_api: Any = plt
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data or not cut_data["coeffs"]:
        print("No hay datos de expresión en este registro.")
        return

    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    sorted_coeffs: SerializedCoeffMap = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
    
    names: list[str] = list(sorted_coeffs.keys())
    values: list[float] = list(sorted_coeffs.values())

    plt_api.figure(figsize=(10, 5))
    colors = ['red' if v < 0 else 'blue' for v in values]
    plt_api.bar(names, values, color=colors)
    plt_api.xticks(rotation=45, ha='right')
    plt_api.title(f"Pesos del Corte - Iteración {log_entry['iteration']} ({log_entry['subproblem']})")
    plt_api.ylabel("Coeficiente en el Maestro")
    plt_api.grid(axis='y', linestyle='--', alpha=0.7)
    plt_api.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()









def plot_sankey_traceability(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    plotly_api: Any = go
    trace = cast(dict[str, dict[str, Any]] | None, log_entry.get("traceability"))
    if not trace:
        return
    
    sources: list[str] = []
    targets: list[str] = []
    values: list[float] = []
    
    for x_var, info in trace.items():
        sources.append(cast(str, info["source_component"]))
        targets.append(x_var)
        values.append(abs(float(cast(float, info["dual_value"]))))
        
    nodes = list(set(sources + targets))
    source_idx = [nodes.index(s) for s in sources]
    target_idx = [nodes.index(t) for t in targets]
    
    fig: Any = plotly_api.Figure(data=[plotly_api.Sankey(
        node=dict(label=nodes, pad=15, thickness=20),
        link=dict(source=source_idx, target=target_idx, value=values),
    )])
    
    fig.update_layout(title_text="Trazabilidad: Del Subproblema al Maestro", font_size=12)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    elif show_plot:
        fig.show()


