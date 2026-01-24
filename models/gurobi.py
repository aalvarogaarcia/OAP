
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from utils.utils import *

def build_and_solve_model(instance_path: str, verbose: bool = False, plot: bool = False, time_limit: int = 7200000, 
                          maximize: bool = True, sum_constrain: bool = True, benders: bool = False, relaxed: bool = False) -> gp.Model:
    points = read_indexed_instance(instance_path)
    N = range(len(points))
    CH = compute_convex_hull(points)
    triangles = compute_triangles(points)
    V = range(len(triangles))
    sc = compute_crossing_edges(triangles, points)
    it = incompatible_triangles(triangles, points)
    ta = triangles_adjacency_list(triangles, points)
    
    for i in range(len(CH)):
        p1 = points[CH[i]]
        p2 = points[CH[(i + 1) % len(CH)]]
        for j in range(len(N)):
            if j not in CH:
                p3 = points[j]
                if is_colineal(p1, p2, p3):
                    print(f"Point {j} is collinear with CH edge ({CH[i]}, {CH[(i + 1) % len(CH)]}) in file {instance_path}")
                    quit()
       

    if verbose:
        print(f"Instance: {instance_path}")

    model = gp.Model("OAP_Simple")
    model._convex_hull_area = compute_convex_hull_area(points)
    # Model building logic goes here
    if verbose:
        print("Building model... \nDefining variables...")

    # Consideramos xij la variable de arcos dirigidos
    x = { (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in N for j in N  if i != j }


    for i in range(len(CH)):
        for j in range(i+2, len(CH)):
            if i == 0 and j == len(CH) - 1:
                continue
            model.remove(x[CH[i], CH[j]])
            model.remove(x[CH[j], CH[i]])
            x.pop((CH[i], CH[j]))
            x.pop((CH[j], CH[i]))

    model.update()
    # Consideramos f_ij la variable de subciclos
    f = {(i,j) : model.addVar(vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}") for i in N for j in N if i != j }

    # Definimos y e yp variables para los triangulos
    y = { (i) : model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}") for i in V}
    yp = { (i) : model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yp_{i}") for i in V}
    
    
    
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
        model.addConstr((gp.quicksum(x[i,j] for j in N if j != i and (i,j) in x.keys()) == 1 ), name=f"Grado_salida_{i}") #(2)
        model.addConstr((gp.quicksum(x[j,i] for j in N if j != i and (j,i) in x.keys()) == 1 ), name=f"Grado_entrada_{i}")


    """ #      Restricciones de cruces
    for k in range(len(sc)):
        model.addConstr(x[sc[k][0], sc[k][1]] + x[sc[k][2], sc[k][3]] <= 1 , name=f"cruce_arcos_{k}") #(3)"""
    """model.addConstr(gp.quicksum(y[it] + yp[it] for it in ta[sc[k][0]][sc[k][1]]) + 
                        gp.quicksum(y[it] + yp[it] for it in ta[sc[k][2]][sc[k][3]]) <= 1 , name=f"cruce_tris_{k}_y")"""


    # --------------------------------------------------- #
    # RESTRICCIONES DE PRUEBA - COMPROBANDO FUNCIONALIDAD #
    # --------------------------------------------------- #

    """if sum_constrain:
    model.addConstr(gp.quicksum(np.abs(signed_area(points[triangles[t][0]], points[triangles[t][1]], points[triangles[t][2]])) * (y[t] + yp[t]) for t in V)
                        == model._convex_hull_area , name=f"area_positiva") #(22)
        
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(gp.quicksum(y[t] + yp[t] for t in ta[i][j]) <= 1 , name=f"triangles_compatibility_{i}_{j}") #(4)"""

    # --------------------------------------------------- #







    # Restricciones de flujo \sim 3
    for i in N:
        if i != 0:
            model.addConstr(gp.quicksum(f[j,i] for j in N if j != i) - gp.quicksum(f[i,j] for j in N if j != i) == 1, name=f"flujo_nodos_{i}") 
        for j in N:
            if (i,j) in x.keys():
                model.addConstr(f[i,j] <= (len(N)-1) * x[i,j] , name=f"flujo_arcos_{i}_{j}") 

    
    #   Restricciones de incompatibilidad de triangulos             (A PROBAR)
    #[model.addConstrs( y[it[k,0]] + y[it[k,1]] <= 1 , name=f"incompatibilidad_tris_{k}") for k in range(len(it))]


    # Restricciones de relacion entre triangulos y arcos envolvente convexa
    for i in range(len(CH)):
        if (CH[i], CH[(i + 1) % len(CH)]) in x.keys():
            model.addConstr(gp.quicksum(y[t] for t in ta[CH[i]][CH[(i + 1) % len(CH)]]) <= x[CH[i], CH[(i + 1) % len(CH)]] , name=f"CH_arcos_internos_{i}_{(i + 1) % len(CH)}")  #(19)
            model.addConstr(gp.quicksum(yp[t] for t in ta[CH[i]][CH[(i + 1) % len(CH)]]) <= 1 - x[CH[(i + 1) % len(CH)], CH[i]] , name=f"CH_arcos_externos_{i}_{(i + 1) % len(CH)}") #(23)


    #RESTRICCIONES DE RELACION ENTRE VARIABLES
    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i < j:
                model.addConstr( gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) == x[i,j] - x[j,i] ) #(20)
                model.addConstr( gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) == x[j,i] - x[i,j] ) #(24)



    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i < j:
                model.addConstr( x[i,j] <= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)
                model.addConstr( 1-x[j,i] >= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)

    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i < j:
                model.addConstr( x[j,i] <= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)
                model.addConstr( 1 - x[i,j] >= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)

    if verbose:
        print("Constraints added. \nOptimizing model...")
    
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    
    if relaxed:
        model.relax()
    model.optimize()

    model._x_results = []
    
    for x_vars in model.getAttr('X', x).keys():
         if model.getAttr('X', x)[x_vars] == 1:
            model._x_results.append(x_vars)

    model._points_ = points

    if plot:
        plot_solution(model, title="Optimal Tour" if model.Status == GRB.OPTIMAL else "Best Found Tour")

    return model


