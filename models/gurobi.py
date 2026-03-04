
from xml.parsers.expat import model
from xml.parsers.expat import model
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import os
from utils.utils import *

def build_and_solve_model(instance_path: str, verbose: bool = False, plot: bool = False, time_limit: int = 7200000, 
                          maximize: bool = True, sum_constrain: bool = True, relaxed: bool = False,
                          obj: int = 2, subtour: int = 0,**kwargs) -> gp.Model:
    """
    Docstring for build_and_solve_model
    
    :param instance_path: Where we can find the instance
    :type instance_path: str
    :param verbose: If we want to print information during the execution
    :type verbose: bool
    :param plot: If we want to plot the solution at the end
    :type plot: bool
    :param time_limit: Set a time limit for the optimization process
    :type time_limit: int
    :param maximize: If we want to maximize the objective function
    :type maximize: bool
    :param sum_constrain: If we want to add the sum of triangles constrain
    :type sum_constrain: bool
    :param relaxed: If we want to relax the integrality constraints
    :type relaxed: bool
    :param obj: Which objective function to use: 0 for Fekete, 1 for Internal Area, 2 for External Area, 3 for Diagonals
    :type obj: int
    :param subtour: Type of subtour elimination constraints to use (0: None, 1: Miller-Tucker-Zemlin, 2: Multi-commodity flow)
    :type subtour: int
    :type kwargs: dict


    :return: Model built and solved
    :rtype: gp.Model
    
    
    """

    points = read_indexed_instance(instance_path)
    N = range(len(points))
    CH = compute_convex_hull(points)
    triangles = compute_triangles(points)
    V = range(len(triangles))
    ta = triangles_adjacency_list(triangles, points)
    mode= kwargs.get('mode', 0)


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
    # Persist configuration metadata for downstream reporting
    model._instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    model._sum_constrain = sum_constrain
    model._obj = obj
    model._maximize_flag = maximize
    model._objective_desc = (
        "Fekete (0)" if obj == 0 and mode == 0 else
        "Fekete (1)" if obj == 0 and mode == 1 else
        "Fekete (2)" if obj == 0 and mode == 2 else
        "Fekete (3)" if obj == 0 and mode == 3 else
        "Internal Area" if obj == 1 else
        "External Area" if obj == 2 else
        "Diagonals"
    )


    # Model building logic goes here
    if verbose:
        print("Building model... \nDefining variables...")

    # Consideramos xij la variable de arcos dirigidos
    x = { (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in N for j in N  if i != j }

    

    
    
    # --- VARIABLES DE ELIMINACIÓN DE SUBTOURS ---
    if subtour == 0:
        # 0: Single-commodity flow (Flujo de un solo producto)
        f = {(i,j) : model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"f_{i}_{j}") for i in N for j in N if i != j }
    
    elif subtour == 1:
        # 1: Miller-Tucker-Zemlin (MTZ)
        u = {i : model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=len(N)-1, name=f"u_{i}") for i in N if i != 0}
        
    elif subtour == 2:
        # 2: Multi-commodity flow (Flujo multiproducto)
        # f_mcf[k, i, j] es el flujo del commodity k que viaja por el arco (i,j)
        f_mcf = {(k, i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"fmcf_{k}_{i}_{j}") 
                 for k in N if k != 0 for i in N for j in N if i != j}

    # --- LIMPIEZA DE ARCOS EN LA ENVOLVENTE CONVEXA (CH) ---
    for i in range(len(CH)):
        for j in range(i+2, len(CH)):
            if (i == 0 and j == len(CH) - 1):
                continue
            model.remove(x[CH[i], CH[j]])
            model.remove(x[CH[j], CH[i]])
            x.pop((CH[i], CH[j]))
            x.pop((CH[j], CH[i]))
            
            # Limpieza condicional de las variables SEC
            if subtour == 0:
                model.remove(f[CH[i], CH[j]])
                model.remove(f[CH[j], CH[i]])
                f.pop((CH[i], CH[j]))
                f.pop((CH[j], CH[i]))

    # Remover arcos en sentido horario de la envolvente convexa
    for i in range(len(CH)):
        j = (i + 1) % len(CH)
        if (CH[j], CH[i]) in x.keys():
            model.remove(x[CH[j], CH[i]])
            x.pop((CH[j], CH[i]))
            if subtour == 0:
                model.remove(f[CH[j], CH[i]])
                f.pop((CH[j], CH[i]))




    
    # Definimos y e yp variables para los triangulos
    y = { (i) : model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"y_{i}") for i in V}
    yp = { (i) : model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"yp_{i}") for i in V}
    
    
    # Definimos zij variables para diagonales
    if obj == 3:

        if mode == 0:
            z = { (i,j) : model.addVar(vtype=GRB.CONTINUOUS, name=f"z_{i}_{j}") for i in N for j in N  if i != j and ((i not in CH) or (j not in CH)) }
            for i,j in z.keys():
                if i < j and (j,i) in z.keys():
                    model.addConstr( z[i,j] == z[j,i] , name=f"diagonal_selection_{i}_{j}")

            for i,j in z.keys():
                model.addConstr(z[i,j] == gp.quicksum(y[t] for t in ta[i][j]) - x[i,j] , name=f"diagonal_triangle_relation_{i}_{j}") #Equivalente a (20)

        elif mode == 1:
            zp = { (i,j) : model.addVar(vtype=GRB.CONTINUOUS, name=f"zp_{i}_{j}") for i in N for j in N  if i != j and ((i not in CH) or (j not in CH)) }
            for i,j in zp.keys():
                if i < j and (j,i) in zp.keys():
                    model.addConstr( zp[i,j] == zp[j,i] , name=f"diagonal_selection_{i}_{j}")

            for i,j in zp.keys():
                model.addConstr(zp[i,j] == gp.quicksum(yp[t] for t in ta[i][j]) - x[i,j] , name=f"diagonal_triangle_relation_{i}_{j}") #Equivalente a (24)


    

    at = [np.abs(signed_area(points[tri[0]], points[tri[1]], points[tri[2]])) for tri in triangles]

    # ----- OBJECTIVE FUNCTION -----
    
    if maximize:
        optimizer = GRB.MAXIMIZE
    else:
        optimizer = GRB.MINIMIZE
    



    
    if obj == 0:
        if verbose:
            print(f"Using Fekete objective function with mode {mode}")
        c = cost_function_area(points, x.keys(), mode=mode)

        model.setObjective(gp.quicksum(c[i] * x[i] for i in x.keys()), optimizer)
        
        
    elif obj == 1:
        model.setObjective(gp.quicksum(y[i] * at[i] for i in V), optimizer)

    elif obj == 2:
        model.setObjective(model._convex_hull_area - gp.quicksum(yp[i] * at[i] for i in V), optimizer)
        
        
    elif obj == 3:
        if verbose:
            print("Using Diagonals objective function")
        
        signed_at = [(signed_area(points[tri[0]], points[tri[1]], points[tri[2]])) for tri in triangles]
        d = {(i,j):np.min([(signed_at[t]) for t in ta[i][j]]) for i,j in x.keys()  }
        
        td = {}
        for idx, tri in enumerate(triangles):
            print(tri)
            td[tuple(tri)] = 3*signed_at[idx] - d[(tri[0]), (tri[1])] - d[(tri[1]), (tri[2])] - d[(tri[2]), (tri[0])]

        if mode == 0:
            sum_x = gp.quicksum(d[i,j] * x[i,j] for i, j in x.keys())
            sum_z = gp.quicksum(d[i,j] * z[i,j] for i, j in z.keys())
        elif mode == 1:
            sum_z = gp.quicksum(d[i,j] * (x[i,j] + zp[i,j]) for i, j in zp.keys())
        else:
            print("Mode not implemented for Diagonals objective") 
        sum_td = 3 * gp.quicksum(td[i, j, k] for i, j, k in td.keys())

        model.setObjective(1/3*(sum_x + sum_z + sum_td) , optimizer)
            
            
            
            

    else:
        print("Objective function not implemented")
        quit()


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


# --- RESTRICCIONES DE ELIMINACIÓN DE SUBTOURS (SEC) ---
    if subtour == 0:
        # Flujo de un solo producto (Tu formulación original)
        for i in N:
            if i != 0:
                model.addConstr(gp.quicksum(f[j,i] for j in N if j != i and (j,i) in f.keys()) - 
                                gp.quicksum(f[i,j] for j in N if j != i and (i,j) in f.keys()) == 1, 
                                name=f"flujo_nodos_{i}") 
            for j in N:
                if i != j and (i,j) in f.keys():
                    model.addConstr(f[i,j] <= (len(N)-1) * x[i,j] , name=f"flujo_arcos_{i}_{j}") 
                    
    elif subtour == 1:
        # MTZ (Miller-Tucker-Zemlin)
        for i in N:
            for j in N:
                if i != 0 and j != 0 and i != j and (i,j) in x.keys():
                    # Formulación clásica: u_i - u_j + (n-1)*x_ij <= n-2
                    model.addConstr(u[i] - u[j] + (len(N)-1) * x[i,j] <= len(N) - 2, name=f"MTZ_{i}_{j}")
                    
    elif subtour == 2:
        # Flujo Multiproducto (Multi-commodity flow)
        for k in N:
            if k != 0:
                for i in N:
                    in_flow = gp.quicksum(f_mcf[k, j, i] for j in N if j != i and (j,i) in x.keys())
                    out_flow = gp.quicksum(f_mcf[k, i, j] for j in N if j != i and (i,j) in x.keys())
                    
                    if i == 0:
                        model.addConstr(out_flow - in_flow == 1, name=f"mcf_origen_{k}_{i}")
                    elif i == k:
                        model.addConstr(in_flow - out_flow == 1, name=f"mcf_destino_{k}_{i}")
                    else:
                        model.addConstr(in_flow - out_flow == 0, name=f"mcf_transito_{k}_{i}")
                
                for i in N:
                    for j in N:
                        if i != j and (i,j) in x.keys():
                            model.addConstr(f_mcf[k, i, j] <= x[i,j], name=f"mcf_cap_{k}_{i}_{j}")


    # RESTRICCION RAMOS ET AL.(2022B) )(8)
#    for i in N:
#        for j in ta[i]:
#            model.addConstr(gp.quicksum(y[t] for t in j) >= 1 , name=f"Ramos_et_al_{i}")

    # Restricciones de relacion entre triangulos y arcos envolvente convexa
    for i in range(len(CH)):
        a = CH[i]
        b = CH[(i + 1) % len(CH)]
        #print(a,b)
        if (a,b) in x.keys():
            model.addConstr(gp.quicksum(y[t] for t in ta[a][b]) == x[a, b] , name=f"CH_arcos_internos_{a}_{b}")  #(19)
            model.addConstr(gp.quicksum(yp[t] for t in ta[a][b]) == 1 - x[a, b] , name=f"CH_arcos_externos_{a}_{b}") #(23)


    #RESTRICCIONES DE RELACION ENTRE VARIABLES
    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i < j :
                if obj == 3 and mode == 0:
                    model.addConstr( gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) == x[j,i] - x[i,j] ) #(24)
                elif obj == 3 and mode == 1:
                    model.addConstr( gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) == x[i,j] - x[j,i] ) #(20)
                else:
                    model.addConstr( gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) == x[i,j] - x[j,i] ) #(20)
                    model.addConstr( gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) == x[j,i] - x[i,j] ) #(24)



    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i != j and (i,j) in x.keys():
                model.addConstr( x[i,j] <= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)
                model.addConstr( 1-x[j,i] >= gp.quicksum(y[t] for t in ta[i][j]) ) #(21)



    for i in N:
        for j in N:
            if ((i not in CH) or (j not in CH)) and i !=j and (i,j) in x.keys():
                model.addConstr( x[j,i] <= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)
                model.addConstr( 1 - x[i,j] >= gp.quicksum(yp[t] for t in ta[i][j]) ) #(25)

    if verbose:
        print("Constraints added. \nOptimizing model...")
    
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    model.update()
    if relaxed:
        model.relax()

    model.Params.MIPGap = 0.00001  # Establece un gap del 0.001% para la solución óptima
    model.Params.NodeLimit = GRB.INFINITY  # Limita el número de nodos explorados
    model.Params.SolutionLimit = GRB.MAXINT  # Limita el número de soluciones enteras encontradas

    model.optimize()


    model._x_results = []
    if model.SolCount > 0:
        for x_vars in model.getAttr('X', x).keys():
            if model.getAttr('X', x)[x_vars] > 0.5:
                model._x_results.append(x_vars)

    print(model._x_results)
    model._points_ = points

    if plot:
        plot_solution(model, title="Optimal Tour" if model.Status == GRB.OPTIMAL else "Best Found Tour")

    return model