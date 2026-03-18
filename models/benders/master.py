from gurobipy import GRB
import gurobipy as gp
import numpy as np
from numpy.typing import NDArray
from typing import Literal, cast

from models.benders.farkas import build_farkas_subproblems
from models.benders.pi import build_pi_subproblems
from utils.utils import (
    compute_convex_hull,
    compute_convex_hull_area,
    compute_crossing_edges,
    compute_triangles,
    cost_function_area,
    read_indexed_instance,
)

Arc = tuple[int, int]
CrossingArc = tuple[int, int, int, int]
ArcVarMap = dict[Arc, gp.Var]
ArcConstrMap = dict[str, dict[Arc, gp.Constr]]
BendersSubproblemData = tuple[gp.Model, gp.Model, ArcConstrMap, ArcConstrMap]

def build_master_problem(
    instance_path: str,
    verbose: bool = False,
    plot: bool = False,
    time_limit: int = 7200,
    maximize: bool = True,
    save_cuts: bool = False,
    crosses_constrain: bool = False,
    benders_method: Literal["farkas", "pi"] = "farkas",
    sum_constrain: bool = True,
) -> gp.Model:
    
        # Lectura de datos
    points: NDArray[np.int64] = read_indexed_instance(instance_path)
    N = range(len(points))
    CH: NDArray[np.int64] = compute_convex_hull(points)
    triangles: NDArray[np.int64] = compute_triangles(points)
    crossing: NDArray[np.int64] = compute_crossing_edges(triangles, points)

    model = gp.Model("Master Problem - Benders Decomposition")
    model.setParam('TimeLimit', time_limit)
    model.params.OutputFlag = 1 if verbose else 0
    model.params.LazyConstraints = 1                            # Habilitar cortes Lazy (Callback)
    #model.params.UserCuts = 1                                   # Permitir cortes de usuario en el callback   
    model.params.PreCrush = 1


    # Guardar datos en el modelo para acceso desde el callback
    model._points_ = points
    model._N = N
    model._CH = CH
    model._convex_hull_area = compute_convex_hull_area(points)
    model._save_cuts = save_cuts
    if model._save_cuts:
        model._instance_name = instance_path.split('/')[-1].replace('.instance', '')
        model._iteration = 0  # Contador de iteraciones para nombrado de cortes
    
    # Save datos para análisis post-mortem de cortes
    model._instance_path = instance_path
    model._verbose = verbose
    model._plot = plot
    model._time_limit = time_limit
    model._maximize = maximize
    model._crosses_constrain = crosses_constrain
    model._benders_method = benders_method

    # Consideramos xij la variable de arcos dirigidos
    x: ArcVarMap = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in N for j in N if i != j}
    
    # Consideramos f_ij la variable de subciclos
    f: ArcVarMap = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}") for i in N for j in N if i != j}
    
    for i in range(len(CH)):
        for j in range(i+2, len(CH)):
            if (i == 0 and j == len(CH) - 1):
                continue
            model.remove(x[CH[i], CH[j]])
            model.remove(x[CH[j], CH[i]])
            model.remove(f[CH[i], CH[j]])
            model.remove(f[CH[j], CH[i]])
            f.pop((CH[i], CH[j]))
            f.pop((CH[j], CH[i]))
            x.pop((CH[i], CH[j]))
            x.pop((CH[j], CH[i]))

    
    # Remove clockwise oriented convex hull edges
    for i in range(len(CH)):
        j = (i + 1) % len(CH)
        #print(f"Removing clockwise CH edge: ({CH[j]}, {CH[i]})")
        if (CH[j], CH[i]) in x.keys():
            model.remove(x[CH[j], CH[i]])
            model.remove(f[CH[j], CH[i]])
            f.pop((CH[j], CH[i]))
            x.pop((CH[j], CH[i]))

    model._x = x
    model._f = f
    model.update()

    # --- Función Objetivo ---
    if maximize:
        optimizer = GRB.MAXIMIZE
    else:
        optimizer = GRB.MINIMIZE

    # Coeficientes de área ca
    c: dict[Arc, float] = cost_function_area(points, x.keys(), mode=3)
    model.setObjective(gp.quicksum(c[i] * x[i] for i in x.keys()), optimizer)

    # --- Restricciones del Maestro (Bloque A0) ---

    # 1. Restricciones de Grado (Eq 2-3 implícitas): Entrante = 1, Saliente = 1
    # Asegura que se forme una colección de ciclos
    for i in N:
        model.addConstr(gp.quicksum(x[i, j] for j in N if (i, j) in x.keys()) == 1, name=f"deg_out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in N if (j, i) in x.keys()) == 1, name=f"deg_in_{i}")

    # 2. Eliminación de Subciclos (Flujo f)
    # Restricciones de balance de flujo para f, forzando un único ciclo conectado
    # Nodo 0 actúa como fuente/sumidero
    for i in N:
        if i != 0:
            # Flujo neto en nodos intermedios = 1 (consumo de 1 unidad por nodo)
            model.addConstr(
                gp.quicksum(f[j, i] for j in N if (j, i) in x.keys()) - 
                gp.quicksum(f[i, j] for j in N if (i, j) in x.keys()) == 1, 
                name=f"flow_balance_{i}"
            )
    
    # Capacidad de flujo vinculada a x (Variable Coupling)
    M = len(N) - 1
    for i, j in x.keys():
        model.addConstr(f[i, j] <= M * x[i, j], name=f"flow_capacity_{i}_{j}")

    # 3. Restricciones de Cruce (Bloque A1)
    # Evitan que se crucen aristas en la solución
    if crosses_constrain:
        for cross in crossing:
            i, j, k, m = cross
            
            exists_edge_1 = (i, j) in x and (j, i) in x
            exists_edge_2 = (k, m) in x and (m, k) in x
            
            if exists_edge_1 and exists_edge_2:
                model.addConstr(
                    x[i, j] + x[j, i] + x[k, m] + x[m, k] <= 1, 
                    name=f"crossing_{i}_{j}_{k}_{m}"
                )

    # --- Construcción del Subproblema ---
    if benders_method == "farkas":
        sub_y, sub_yp, constrs_y, constrs_yp = cast(
            BendersSubproblemData,
            build_farkas_subproblems(points, N, CH, x.keys(), sum_constrain=sum_constrain),
        )
    elif benders_method == "pi":
        sub_y, sub_yp, constrs_y, constrs_yp = cast(
            BendersSubproblemData,
            build_pi_subproblems(points, N, CH, x.keys(), sum_constrain=sum_constrain),
        )
    else:
        raise ValueError(f"Método de Benders desconocido: {benders_method}. Elige 'farkas' o 'pi'.")
    
    model._x = x  
    model._sub_y = sub_y
    model._sub_yp = sub_yp
    model._constrs_y = constrs_y
    model._constrs_yp = constrs_yp

    model._benders_ = True

    return model


