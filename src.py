# SIMPLIFIED OVER SIMPLIFIED PROBLEM
from xml.parsers.expat import model
from pyexpat import model
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from itertools import combinations
import time
import glob
import gurobipy as gp
from gurobipy import GRB
from docplex.mp.model import Model

def read_data(file_path):
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

def orientation_2d(a: tuple, b: tuple, c: tuple) -> int:
    """Orientation sign for coordinates (x,y) tuples.
    Returns 0 for collinear, 1 for clockwise, -1 for counterclockwise.
    """
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if det == 0:
        return 0
    return -1 if det > 0 else 1

def contains_proper(p1: tuple, q1: tuple, p2: tuple, q2: tuple) -> bool:
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

    if n < 3: return np.empty((0, 3), dtype=np.int64)
    
    hull = compute_convex_hull(points)
    hull_edges = np.sort(np.array([(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))], dtype=np.int64), axis=1)
    hull_mask = np.zeros(n, dtype=bool)
    hull_mask[hull] = True

    valid_triangles: list[NDArray[np.int64]] = []

    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross_product = np.cross(v1, v2)
        if cross_product == 0:  # Colinear
            continue

        """        # Ensure CCW ordering of indices
        if cross_product > 0:
            tri_idx = np.array([i, j, k], dtype=np.int64)
        else:
            tri_idx = np.array([i, k, j], dtype=np.int64)"""

        tri_idx = np.array([i, j, k], dtype=np.int64)

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
    def ccw(A, B, C):
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

    
def are_triangles_incompatible(t1_coords: tuple, t2_coords: tuple, points: NDArray[np.int64]) -> bool:
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

def signed_area(p1, p2, p3):
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def triangles_area(triangles: NDArray[np.int64], points) -> float:
    # Calculate signed area for each triangle
    triangle_areas = []
    for tri in triangles:
        triangle_areas.append(signed_area(points[tri[0]], points[tri[1]], points[tri[2]]))
    
    return triangle_areas


def triangles_adjacency_list(triangles: NDArray[np.int64], p: NDArray) -> dict[int, list[int]]:
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


def write_prefile(file_path:str):
    """Writes the computed geometric structures to a .pre file.
    
    Reads an instance file, computes convex hull, triangles, crossing segments,
    and incompatible triangle pairs, then writes results to a .pre file.
    """
    import os
    
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
    
    # Extract seed from filename (last number before .instance)
    # Format expected: something-<seed>.instance -> extract <seed>
    parts = base_name.split('-')
    seed = parts[-1] if parts else '0'
    
    out_file = os.path.join(os.path.dirname(file_path), f"n{n_points}s{seed}.pre")
    
    # Write .pre file
    with open(out_file, 'w') as f:
        # Header with statistics
        f.write(f"# Preprocesing information of intance {base_name}\n")
        f.write(f"# Points\tConvex_Hull\tTriangles\tCrossing_segments\tIncompatible_Triangles\tTime\n")
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


def insertion_heuristic(points: NDArray[np.int64], CH: NDArray[np.int64], V: NDArray[np.int64], sense: bool) -> list[int]:
    """
    Genera un polígono simple comenzando con la Envolvente Convexa (CH)
    e insertando puntos restantes según un criterio (ej. mayor área añadida).
    """
    # Conjunto de puntos disponibles para insertar (los que no están en el CH)
    N = CH
    Nprime = np.array([i for i in range(len(points)) if i not in CH], dtype=np.int64)
    Wa = np.abs(triangles_area(V, points))
    ta = triangles_adjacency_list(V, points)
    Vprime = [ta[N[i]][N[(i + 1) % len(N)]] for i in range(len(N))]
    IV = incompatible_triangles(V, points)
    Weight_Vij = [[Wa[t] for t in Vprime[i]] for i in range(len(N))]

    
    

    






    while len(Nprime) > 0:
        mejor_ganancia = -float('inf')
        mejor_movimiento = None # Tupla: (indice_en_ruta, punto_a_insertar)
        
        

            
    return N

def solve_benders_decomposition(points: NDArray[np.int64], ta, time_limit: int = 7200000):
    # --- MODELO MAESTRO ---
    m_master = gp.Model("Benders_Master")

    x = { (i, j): m_master.addVar(vtype=GRB.BINARY, lb = 0, ub = 1, name=f"x_{i}_{j}") for i in N for j in N  if i != j }
    f = {(i,j) : m_master.addVar(vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}") for i in N for j in N if i != j }
    z = {(i): m_master.addVar(vtype=GRB.CONTINUOUS, name=f"z_{i}") for i in V }
    m_master.ModelSense = GRB.MAXIMIZE

    # AÑADIR LAS RESTRICCIONES INICIALES DEL MAESTRO AQUÍ

    # Restricciones de flujo


    # --- MODELO SUBPROBLEMA ---
    m_sub = gp.Model("Benders_Subproblem")
    m_sub.setParam('OutputFlag', 0)
    y = { (i) : m_sub.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"y_{i}") for i in V}
    yp = { (i) : m_sub.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"yp_{i}") for i in V}






def build_and_solve_model(instance_path: str, verbose: bool = False, plot: bool = False, time_limit: int = 7200000, 
                          maximize: bool = True, sum_constrain: bool = True, benders: bool = False) -> gp.Model:
    points = read_indexed_instance(instance_path)
    N = range(len(points))
    CH = compute_convex_hull(points)
    triangles = compute_triangles(points)
    V = range(len(triangles))
    sc = compute_crossing_edges(triangles, points)
    it = incompatible_triangles(triangles, points)
    ta = triangles_adjacency_list(triangles, points)

    if verbose:
        print(f"Instance: {instance_path}")

    model = gp.Model("OAP_Simple")
    model._convex_hull_area = compute_convex_hull_area(points)
    # Model building logic goes here
    if verbose:
        print("Building model... \nDefining variables...")

    # Consideramos xij la variable de arcos dirigidos
    x = { (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in N for j in N  if i != j }

    # Consideramos f_ij la variable de subciclos
    f = {(i,j) : model.addVar(vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}") for i in N for j in N if i != j }

    # Definimos y e yp variables para los triangulos
    y = { (i) : model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"y_{i}") for i in V}
    yp = { (i) : model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"yp_{i}") for i in V}
    
    
    
    # Objective: Maximize total area of selected triangles
    at = [np.abs(signed_area(points[tri[0]], points[tri[1]], points[tri[2]])) for tri in triangles]
    if maximize:
        model.setObjective(gp.quicksum(y[i] * at[i] for i in V), GRB.MAXIMIZE)
    else:
        model.setObjective(model._convex_hull_area - gp.quicksum(yp[i] * at[i] for i in V), GRB.MINIMIZE)

    if verbose:
        print("Variables defined. \nAdding constraints...")

    # ----- CONSTRAINS -----

    
    # Resticiones de triangulos
    if sum_constrain is not None:
        model.addConstr(gp.quicksum(y[i] for i in V) == len(N)-2 , name="triangulos_internos_totales") #NONE
        model.addConstr(gp.quicksum(yp[i] for i in V) == len(N)-len(CH) , name="triangulos_externos_totales") #NONE
    


    # Restriciones de arcos
    for i in N:
        model.addConstr((gp.quicksum(x[i,j] for j in N if j != i) == 1 ), name=f"Grado_salida_{i}") #(2)
        model.addConstr((gp.quicksum(x[j,i] for j in N if j != i) == 1 ), name=f"Grado_entrada_{i}")


    #      Restricciones de cruces
    for k in range(len(sc)):
        model.addConstr(x[sc[k][0], sc[k][1]] + x[sc[k][2], sc[k][3]] <= 1 , name=f"cruce_arcos_{k}") #(3)
        model.addConstr(gp.quicksum(y[it] + yp[it] for it in ta[sc[k][0]][sc[k][1]]) + 
                        gp.quicksum(y[it] + yp[it] for it in ta[sc[k][2]][sc[k][3]]) <= 1 , name=f"cruce_tris_{k}_y")


    # --------------------------------------------------- #
    # RESTRICCIONES DE PRUEBA - COMPROBANDO FUNCIONALIDAD #
    # --------------------------------------------------- #


    model.addConstr(gp.quicksum(np.abs(signed_area(points[triangles[t][0]], points[triangles[t][1]], points[triangles[t][2]])) * (y[t] + yp[t]) for t in V)
                        == model._convex_hull_area , name=f"area_positiva") #(22)
        
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(gp.quicksum(y[t] + yp[t] for t in ta[i][j]) <= 1 , name=f"triangles_compatibility_{i}_{j}") #(4)

    # --------------------------------------------------- #







    # Restricciones de flujo \sim 3
    for i in N:
        if i != 0:
            model.addConstr(gp.quicksum(f[j,i] for j in N if j != i) - gp.quicksum(f[i,j] for j in N if j != i) == 1, name=f"flujo_nodos_{i}") 
        for j in N:
            if i != j:
                model.addConstr(f[i,j] <= (len(N)-1) * x[i,j] , name=f"flujo_arcos_{i}_{j}") 

    
    #   Restricciones de incompatibilidad de triangulos             (A PROBAR)
    #[model.addConstrs( y[it[k,0]] + y[it[k,1]] <= 1 , name=f"incompatibilidad_tris_{k}") for k in range(len(it))]


    # Restricciones de relacion entre triangulos y arcos envolvente convexa
    for i in range(len(CH)):
        model.addConstr(gp.quicksum(y[t] for t in ta[CH[i]][CH[(i + 1) % len(CH)]]) <= x[CH[i], CH[(i + 1) % len(CH)]] , name=f"CH_arcos_internos_{i}_{(i + 1) % len(CH)}")  #(19)
        model.addConstr(gp.quicksum(yp[t] for t in ta[CH[i]][CH[(i + 1) % len(CH)]]) <= 1 - x[CH[(i + 1) % len(CH)], CH[i]] , name=f"CH_arcos_externos_{i}_{(i + 1) % len(CH)}") #(23)


    #RESTRICCIONES DE RELACION ENTRE VARIABLES
    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                model.addConstr( gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) == x[i,j] - x[j,i] ) #(20)
                model.addConstr( gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) == x[j,i] - x[i,j] ) #(24)



    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                model.addConstr( x[i,j] <= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)
                model.addConstr( 1-x[j,i] >= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)

    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                model.addConstr( x[j,i] <= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)
                model.addConstr( 1 - x[i,j] >= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)

    if verbose:
        print("Constraints added. \nOptimizing model...")
    
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.optimize()
    return model






def build_and_solve_model_cplex(instance_path: str, verbose: bool = False, plot: bool = False, time_limit: int = 7200000, 
                                maximize: bool = True, sum_constrain: bool = True, benders: bool = False):
    
    # --- Lectura de datos (Igual que tu código original) ---
    points = read_indexed_instance(instance_path)
    N = range(len(points))
    CH = compute_convex_hull(points)
    triangles = compute_triangles(points)
    V = range(len(triangles))
    sc = compute_crossing_edges(triangles, points)
    it = incompatible_triangles(triangles, points)
    ta = triangles_adjacency_list(triangles, points)

    if verbose:
        print(f"Instance: {instance_path}")

    # Inicializar Modelo CPLEX
    mdl = Model(name="OAP_Simple_CPLEX")
    
    # Guardamos el area igual que en tu objeto Gurobi (monkey-patching para mantener lógica)
    mdl._convex_hull_area = compute_convex_hull_area(points)

    if verbose:
        print("Building model... \nDefining variables...")

    # --- VARIABLES ---

    # Consideramos xij la variable de arcos dirigidos
    # Nota: CPLEX usa binary_var. Creamos el diccionario manualmente para replicar tu estructura exacta.
    x = { (i, j): mdl.binary_var(name=f"x_{i}_{j}") for i in N for j in N if i != j }
    # Consideramos f_ij la variable de subciclos
    f = {(i,j) : mdl.continuous_var(name=f"f_{i}_{j}") for i in N for j in N if i != j }

    # Definimos y e yp variables para los triangulos
    # En docplex lb=0 es por defecto, pero especificamos ub=1
    y = { (i) : mdl.continuous_var(lb=0, ub=1, name=f"y_{i}") for i in V}
    yp = { (i) : mdl.continuous_var(lb=0, ub=1, name=f"yp_{i}") for i in V}

    
    # --- OBJETIVO ---
    
    at = [np.abs(signed_area(points[tri[0]], points[tri[1]], points[tri[2]])) for tri in triangles]
    
    # mdl.sum es el equivalente a gp.quicksum
    objective_expr = mdl.sum(y[i] * at[i] for i in V)

    if maximize:
        mdl.maximize(objective_expr)
    else:
        mdl.minimize(mdl._convex_hull_area - mdl.sum(yp[i] * at[i] for i in V))

    if verbose:
        print("Variables defined. \nAdding constraints...")

    # ----- CONSTRAINTS -----

    # Restricciones de triangulos
    if sum_constrain is not None:
        mdl.add_constraint(mdl.sum(y[i] for i in V) == len(N)-2, ctname="triangulos_internos_totales")
        mdl.add_constraint(mdl.sum(yp[i] for i in V) == len(N)-len(CH), ctname="triangulos_externos_totales")

    # Restricciones de arcos
    for i in N:
        mdl.add_constraint(mdl.sum(x[i,j] for j in N if j != i) == 1, ctname=f"Grado_salida_{i}")
        mdl.add_constraint(mdl.sum(x[j,i] for j in N if j != i) == 1, ctname=f"Grado_entrada_{i}")

    # Restricciones de cruces
    for k in range(len(sc)):
        mdl.add_constraint(x[sc[k][0], sc[k][1]] + x[sc[k][2], sc[k][3]] <= 1, ctname=f"cruce_arcos_{k}")
        
        mdl.add_constraint(
            mdl.sum(y[it] + yp[it] for it in ta[sc[k][0]][sc[k][1]]) + 
            mdl.sum(y[it] + yp[it] for it in ta[sc[k][2]][sc[k][3]]) <= 1, 
            ctname=f"cruce_tris_{k}_y"
        )

    """
    # --------------------------------------------------- #
    # RESTRICCIONES DE PRUEBA - COMPROBANDO FUNCIONALIDAD #
    # --------------------------------------------------- #

    mdl.add_constraint(
        mdl.sum(np.abs(signed_area(points[triangles[t][0]], points[triangles[t][1]], points[triangles[t][2]])) * (y[t] + yp[t]) for t in V)
        == mdl._convex_hull_area, 
        ctname="area_positiva"
    )
        
    for i in N:
        for j in N:
            if i != j:
                mdl.add_constraint(mdl.sum(y[t] + yp[t] for t in ta[i][j]) <= 1, ctname=f"triangles_compatibility_{i}_{j}")
    """
    # --------------------------------------------------- #

    # Restricciones de flujo
    for i in N:
        if i != 0:
            mdl.add_constraint(
                mdl.sum(f[j,i] for j in N if j != i) - mdl.sum(f[i,j] for j in N if j != i) == 1, 
                ctname=f"flujo_nodos_{i}"
            )
        for j in N:
            if i != j:
                mdl.add_constraint(f[i,j] <= (len(N)-1) * x[i,j], ctname=f"flujo_arcos_{i}_{j}")

    # Restricciones de relacion entre triangulos y arcos envolvente convexa
    for i in range(len(CH)):
        idx_next = (i + 1) % len(CH)
        u, v = CH[i], CH[idx_next]
        
        mdl.add_constraint(mdl.sum(y[t] for t in ta[u][v]) <= x[u, v], ctname=f"CH_arcos_internos_{i}_{idx_next}")
        mdl.add_constraint(mdl.sum(yp[t] for t in ta[u][v]) <= 1 - x[v, u], ctname=f"CH_arcos_externos_{i}_{idx_next}")

    # RESTRICCIONES DE RELACION ENTRE VARIABLES
    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                # Ecuacion (20) y (24)
                mdl.add_constraint(
                    mdl.sum(y[t] for t in ta[i][j]) - mdl.sum(y[t] for t in ta[j][i]) == x[i,j] - x[j,i]
                )
                mdl.add_constraint(
                    mdl.sum(yp[t] for t in ta[i][j]) - mdl.sum(yp[t] for t in ta[j][i]) == x[j,i] - x[i,j]
                )

    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                # Ecuaciones (21) y (25)
                mdl.add_constraint(x[i,j] <= mdl.sum(y[t] for t in ta[i][j]))
                mdl.add_constraint(1-x[j,i] >= mdl.sum(y[t] for t in ta[i][j]))

    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i!= j:
                # Ecuaciones (25) duplicadas (según tu código original)
                mdl.add_constraint(x[j,i] <= mdl.sum(yp[t] for t in ta[i][j]))
                mdl.add_constraint(1 - x[i,j] >= mdl.sum(yp[t] for t in ta[i][j]))

    if verbose:
        print("Constraints added. \nOptimizing model...")
    
    # Parametros de CPLEX
    # CPLEX usa segundos por defecto. 7200000ms = 7200s
    mdl.set_time_limit(time_limit) 
    
    # Control de log
    mdl.context.solver.log_output = True if verbose else False
    
    solution = mdl.solve()
    
    if verbose:
        if solution:
            print("Solution found!")
            print(f"Objective value: {solution.objective_value}")
        else:
            print("No solution found within time limit.")

    return mdl























if __name__ == "__main__":
    for file in glob.glob("*/*.instance"):
        start_time = time.time()
        mod = build_and_solve_model(
            instance_path=file,
            verbose=True,
            plot=False,
            time_limit=7200000,
            maximize=True
        )
        end_time = time.time()
        print(f"Processed {file} in {end_time - start_time:.2f} seconds.")
        print("--------------------------------------------------\n")
        print("Results Summary:\n")
        print(f"Instance: {file}")
        print(f"Objective Value: {mod.ObjVal}")
        print(f"Status: {mod.Status}")
        print("--------------------------------------------------\n")
        mod.write(f"{file}_solution.lp")
        model_lp = mod.relax()
        model_lp.setParam('OutputFlag', 1 if False else 0)
        model_lp.optimize()
        print("--------------------------------------------------\n")
        print("LP Relaxation Results Summary:\n")
        print(f"LP Status: {model_lp.Status}")
        print(f"LP Relaxed Objective Value: {model_lp.ObjVal}")
        print("--------------------------------------------------\n")
    

