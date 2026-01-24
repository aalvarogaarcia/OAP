
from docplex.mp.model import Model
import numpy as np
from utils.utils import *

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
    
    if benders:
        mdl.parameters.benders.strategy = 1
        if verbose:
            print("Configuring Benders decomposition strategy...")
        
        #Variables del master (x, f) anotadas con 0
        for var in x.values():
            mdl.set_benders_annotation(var, 0)
        for var in f.values():
            mdl.set_benders_annotation(var, 0)

        #Variables del subproblema (y, yp) anotadas con 1
        for var in y.values():
            mdl.set_benders_annotation(var, 1)
        for var in yp.values():
            mdl.set_benders_annotation(var, 1)
        
    
    
    solution = mdl.solve()
        
    if verbose:
        if solution:
            print("Solution found!")
            print(f"Objective value: {solution.objective_value}")
        else:
            print("No solution found within time limit.")

    return mdl



def create_relaxed_model_cplex(model):
    """
    Crea una copia relajada (LP) del modelo.
    Versión corregida para asegurar que las binarias se convierten a continuas.
    """
    print("Creando relajación del modelo...")
    
    # 1. Clonar el modelo
    model_lp = model.clone()
    model_lp.name = f"{model.name}_Relaxed"
    
    # 2. Identificar explícitamente las variables que NO son continuas
    # IMPORTANTE: Usamos [...] para crear una lista real, no un generador
    vars_to_relax = [v for v in model_lp.iter_variables() if v.is_discrete()]
    
    print(f"  - Se han encontrado {len(vars_to_relax)} variables discretas para relajar.")

    # 3. Cambiarlas a continuas
    if vars_to_relax:
        # docplex necesita una lista concreta, no un iterador
        model_lp.change_var_types(vars_to_relax, model_lp.continuous_vartype)
        print("  - Conversión a continuas completada.")
    else:
        print("  - No había variables discretas.")

    model_lp.parameters.benders.strategy = 0
    return model_lp
