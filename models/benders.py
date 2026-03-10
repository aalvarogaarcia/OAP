from gurobipy import GRB
import gurobipy as gp
import os
from utils.utils import *


def build_subproblems(points, N, CH, master_x_keys):
    """
    Construye DOS Subproblemas (SP_Y y SP_YP) de factibilidad para la triangulación.
    Están separados para evitar enmascaramiento de rayos de Farkas si ambos son infactibles.
    """
    
    # Pre-cálculo de triángulos y adyacencias
    triangles = compute_triangles(points) # Asumo que esta función existe en tu entorno
    V = range(len(triangles))
    ta = triangles_adjacency_list(triangles, points) # Asumo que esta función existe
    
    
    #--------------------------------------------------------------
    #---------               MODELO Y                --------------
    #--------------------------------------------------------------
    
    sub_y = gp.Model("Subproblem_Y")
    sub_y.Params.OutputFlag = 0
#    sub_y.Params.InfUnbdInfo = 1
    sub_y.Params.DualReductions = 0

    y = sub_y.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="y")
    # Variables artificiales para garantizar factibilidad y obtener rayos de Farkas en caso de infactibilidad
    art_y_alpha_p = sub_y.addVars(master_x_keys, lb=0, name="art_alpha_p")
    art_y_alpha_n = sub_y.addVars(master_x_keys, lb=0, name="art_alpha_n")
    art_y_beta_p = sub_y.addVars(master_x_keys, lb=0, name="art_beta_p")
    art_y_beta_n = sub_y.addVars(master_x_keys, lb=0, name="art_beta_n")
    art_y_gamma_p = sub_y.addVars(master_x_keys, lb=0, name="art_gamma_p")
    art_y_gamma_n = sub_y.addVars(master_x_keys, lb=0, name="art_gamma_n")
    art_y_delta_n = sub_y.addVars(master_x_keys, lb=0, name="art_delta_n")
    art_y_delta_p = sub_y.addVars(master_x_keys, lb=0, name="art_delta_p")


    sub_y.setObjective(
        gp.quicksum(art_y_alpha_n) + gp.quicksum(art_y_alpha_p) + 
        gp.quicksum(art_y_beta_n) + gp.quicksum(art_y_beta_p) + 
        gp.quicksum(art_y_gamma_n) + gp.quicksum(art_y_gamma_p) + 
        gp.quicksum(art_y_delta_n) + gp.quicksum(art_y_delta_p), GRB.MINIMIZE
    )

    sub_y.Params.Presolve = 1
    
    #--------------------------------------------------------------
    #---------               MODELO Y'               --------------
    #--------------------------------------------------------------
    
    sub_yp = gp.Model("Subproblem_YP")
    sub_yp.Params.OutputFlag = 0
#    sub_yp.Params.InfUnbdInfo = 1
    sub_yp.Params.DualReductions = 0
    yp = sub_yp.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="yp")

    #Variables artificiales para garantizar factibilidad y obtener rayos de Farkas en caso de infactibilidad
    art_yp_alpha_p = sub_yp.addVars(master_x_keys, lb=0, name="art_alpha_p")
    art_yp_alpha_n = sub_yp.addVars(master_x_keys, lb=0, name="art_alpha_n")
    art_yp_beta_p = sub_yp.addVars(master_x_keys, lb=0, name="art_beta_p")
    art_yp_beta_n = sub_yp.addVars(master_x_keys, lb=0, name="art_beta_n")
    art_yp_gamma_p = sub_yp.addVars(master_x_keys, lb=0, name="art_gamma_p")
    art_yp_gamma_n = sub_yp.addVars(master_x_keys, lb=0, name="art_gamma_n")
    art_yp_delta_n = sub_yp.addVars(master_x_keys, lb=0, name="art_delta_n")
    art_yp_delta_p = sub_yp.addVars(master_x_keys, lb=0, name="art_delta_p")


    sub_yp.setObjective(
        gp.quicksum(art_yp_alpha_n) + gp.quicksum(art_yp_alpha_p) + 
        gp.quicksum(art_yp_beta_n) + gp.quicksum(art_yp_beta_p) + 
        gp.quicksum(art_yp_gamma_n) + gp.quicksum(art_yp_gamma_p) + 
        gp.quicksum(art_yp_delta_n) + gp.quicksum(art_yp_delta_p), GRB.MINIMIZE
    )



    sub_yp.Params.Presolve = 1
    # Diccionarios para almacenar las restricciones (agrupadas por su variable dual)
    # RHS se inicializa en 0 o 1, y deberá actualizarse en el callback con los valores de x_bar
    constrs_y = {'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}, 'global':{}}
    constrs_yp = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}, 'global':{}}
    
    
    rhs_y = len(N) - 2
    constrs_y['global'] = sub_y.addConstr(
        gp.quicksum(y[t] for t in V) == rhs_y, 
        name="triangulos_internos_totales"
    )
    
    
    rhs_yp = len(N) - len(CH)  
    constrs_yp['global'] = sub_yp.addConstr(
        gp.quicksum(yp[t] for t in V) == rhs_yp,
        name="triangulos_externos_totales"
    )

    # 1. Conjunto A': Arcos dirigidos en la frontera del Convex Hull
    A_prime = []
    for i in range(len(CH)):
        u, v = CH[i], CH[(i + 1) % len(CH)]
        if (u, v) in master_x_keys:
            A_prime.append((u, v))

    for (i, j) in A_prime:
        # Eq (sp:6): sum(y) = x_bar_ij -> Placeholder RHS = 0
        constrs_y['alpha'][i, j] = sub_y.addConstr(
            gp.quicksum(y[t] for t in ta[i][j]) + art_y_alpha_p[i, j] - art_y_alpha_n[i, j] == 0, name=f"alpha_{i}_{j}"
        )
        # Eq (sp:11): sum(y') = 1 - x_bar_ij -> Placeholder RHS = 0
        constrs_yp['alpha_p'][i, j] = sub_yp.addConstr(
            gp.quicksum(yp[t] for t in ta[i][j]) + art_yp_alpha_p[i, j] - art_yp_alpha_n[i, j] == 0, name=f"alpha_p_{i}_{j}"
        )

    # 2. Conjuntos E'' (Aristas no dirigidas) y A'' (Arcos dirigidos) no pertenecientes al CH
    for i in N:
        for j in N:
            # Condición para no pertenecer a las aristas explícitas del Convex Hull
            is_non_ch = ((i not in CH) or (j not in CH))
            
            # E'': Pares ordenados i < j para restricciones de balance (restricción por arista)
            if is_non_ch and i < j:
                # Eq (sp:7): sum(y_ij) - sum(y_ji) = x_bar_ij - x_bar_ji
                constrs_y['beta'][i, j] = sub_y.addConstr(
                    gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) + art_y_beta_p[i, j] - art_y_beta_n[i, j] == 0,
                    name=f"beta_{i}_{j}"
                )
                # Eq (sp:12): sum(yp_ij) - sum(yp_ji) = x_bar_ji - x_bar_ij
                constrs_yp['beta_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) + art_yp_beta_p[i, j] - art_yp_beta_n[i, j] == 0,
                    name=f"beta_p_{i}_{j}"
                )

            # A'': Todos los arcos dirigidos (i != j) para cotas superior e inferior
            if is_non_ch and i != j:
                # Eq (sp:8): sum(y) >= x_bar_ij
                constrs_y['gamma'][i, j] = sub_y.addConstr(
                    gp.quicksum(y[t] for t in ta[i][j]) + art_y_gamma_p[i, j] - art_y_gamma_n[i, j] >= 0, name=f"gamma_{i}_{j}"
                )
                # Eq (sp:9): sum(y) <= 1 - x_bar_ji
                constrs_y['delta'][i, j] = sub_y.addConstr(
                    gp.quicksum(y[t] for t in ta[i][j]) + art_y_delta_p[i, j] - art_y_delta_n[i, j] <= 0, name=f"delta_{i}_{j}"
                )
                
                # Eq (sp:13): sum(yp) >= x_bar_ji
                constrs_yp['gamma_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j]) + art_yp_gamma_p[i, j] - art_yp_gamma_n[i, j] >= 0, name=f"gamma_p_{i}_{j}"
                )
                # Eq (sp:14): sum(yp) <= 1 - x_bar_ij
                constrs_yp['delta_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j]) + art_yp_delta_p[i, j] - art_yp_delta_n[i, j] <= 0, name=f"delta_p_{i}_{j}"
                )

    sub_y.update()
    sub_yp.update()
    
    return sub_y, sub_yp, constrs_y, constrs_yp

def update_subproblem_rhs(model, x_sol):
        constrs_y = model._constrs_y
        constrs_yp = model._constrs_yp

        if hasattr(model, '_iteration'):
            model._iteration += 1

        # ==========================================
        # 2. ACTUALIZAR RHS PARA EL SUBPROBLEMA Y
        # ==========================================
        # Ec. (6): alpha_ij -> RHS = x_bar_ij
        for (i, j), constr in constrs_y['alpha'].items():
            constr.RHS = x_sol[i, j]
            
        # Ec. (7): beta_ij -> RHS = x_bar_ij - x_bar_ji
        for (i, j), constr in constrs_y['beta'].items():
            constr.RHS = x_sol[i, j] - x_sol[j, i]
            
        # Ec. (8): gamma_ij -> RHS = x_bar_ij
        for (i, j), constr in constrs_y['gamma'].items():
            constr.RHS = x_sol[i, j]
            
        # Ec. (9): delta_ij -> RHS = 1 - x_bar_ji
        for (i, j), constr in constrs_y['delta'].items():
            constr.RHS = 1 - x_sol[j, i]

        # ==========================================
        # 3. ACTUALIZAR RHS PARA EL SUBPROBLEMA Y'
        # ==========================================
        # Ec. (11): alpha'_ij -> RHS = 1 - x_bar_ij
        for (i, j), constr in constrs_yp['alpha_p'].items():
            constr.RHS = 1 - x_sol[i, j]
            
        # Ec. (12): beta'_ij -> RHS = x_bar_ji - x_bar_ij
        for (i, j), constr in constrs_yp['beta_p'].items():
            constr.RHS = x_sol[j, i] - x_sol[i, j]
            
        # Ec. (13): gamma'_ij -> RHS = x_bar_ji
        for (i, j), constr in constrs_yp['gamma_p'].items():
            constr.RHS = x_sol[j, i]
            
        # Ec. (14): delta'_ij -> RHS = 1 - x_bar_ij
        for (i, j), constr in constrs_yp['delta_p'].items():
            constr.RHS = 1 - x_sol[i, j]

def generate_farkas_cut_y(constrs_y, x_sol, model, TOL = 1e-10):
    """
    Genera un corte de Benders a partir de los rayos de Farkas obtenidos de ambos subproblemas.
    Analiza las componentes del vector v_ij para entender la naturaleza de la violación.
    """
    # Este método se llamará desde el callback después de detectar infactibilidad en alguno de los subproblemas
    # y después de actualizar los RHS con update_subproblem_rhs(model, x_sol)
    
    # Aquí se implementaría la lógica para extraer los rayos de Farkas, construir el corte y analizarlo.
    # Dado que esta función es bastante extensa, la dejaremos como un placeholder por ahora.

    # =========================================================
    # 5. ANÁLISIS DEL VECTOR v_ij Y GENERACIÓN DE CORTES PARA Y
    # =========================================================
    cut_y_expr = gp.LinExpr() # Expresión simbólica para Gurobi
    cut_y_val = 0.0           # Valor numérico para comprobar el signo
    
    # Diccionario para almacenar y analizar las componentes de v_ij
    v_components_y = {'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}}
    
    # --- Extracción de Farkas Duales ---
    for (i, j), constr in constrs_y['alpha'].items():
        farkas = constr.FarkasDual
        if abs(farkas) > TOL:
            v_components_y['alpha'][i, j] = farkas
            cut_y_expr.add(farkas * model._x[i, j])
            cut_y_val += farkas * x_sol[i, j]
            
    for (i, j), constr in constrs_y['beta'].items():
        farkas = constr.FarkasDual
        if abs(farkas) > TOL:
            v_components_y['beta'][i, j] = farkas
            cut_y_expr.add(farkas * (model._x[i, j] - model._x[j, i]))
            cut_y_val += farkas * (x_sol[i, j] - x_sol[j, i])
            
    for (i, j), constr in constrs_y['gamma'].items():
        farkas = constr.FarkasDual
        if abs(farkas) > TOL:
            v_components_y['gamma'][i, j] = farkas
            cut_y_expr.add(farkas * model._x[i, j])
            cut_y_val += farkas * x_sol[i, j]
            
    for (i, j), constr in constrs_y['delta'].items():
        farkas = constr.FarkasDual
        if abs(farkas) > TOL:
            v_components_y['delta'][i, j] = farkas
            cut_y_expr.add(farkas * (1 - model._x[j, i]))
            cut_y_val += farkas * (1 - x_sol[j, i])
    
    # --- Análisis por consola ---
    if model._save_cuts:
        print("\n" + "-"*30)
        print("RAYO DE FARKAS DETECTADO EN SUBPROBLEMA Y")
        print("Valor numérico de la violación (v^T * b(x_bar)): ", cut_y_val)
        for comp, values in v_components_y.items():
            if values: # Solo imprime si hay valores no nulos
                print(f"Componente {comp}:")
                for k, v in values.items():
                    print(f"  {k}: {v:.4f}")
        print("-"*30 + "\n")
   
    # --- NUEVO: Guardar en el log estructurado ---
    if getattr(model, '_save_cuts', False):
        # 1. Obtener trazabilidad detallada        
        log_farkas_ray(
            filepath=model._farkas_log_path,
            iteration=model._iteration,
            node_depth=0, # 0 porque estamos en MIPSOL
            subproblem_type='Y',
            x_sol=x_sol,
            v_components=v_components_y,
            violation_value=cut_y_val,
            tolerance=TOL,
            cut_expr = cut_y_expr,
        )
    
    return cut_y_expr, cut_y_val, v_components_y
    # --- Añadir el corte al maestro de forma segura ---
    # Si el valor evaluado es positivo, el hiperplano debe forzarse hacia <= 0
   

def generate_farkas_cut_yp(constrs_yp, x_sol, model, TOL = 1e-10):
    # =========================================================
    # 6. ANÁLISIS DEL VECTOR v'_ij Y GENERACIÓN DE CORTES PARA Y'
    # =========================================================
        cut_yp_expr = gp.LinExpr()
        cut_yp_val = 0.0
        
        v_components_yp = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}}
        
        # --- Extracción de Farkas Duales ---
        for (i, j), constr in constrs_yp['alpha_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_components_yp['alpha_p'][i, j] = farkas
                cut_yp_expr.add(farkas * (1 - model._x[i, j]))
                cut_yp_val += farkas * (1 - x_sol[i, j])
                
        for (i, j), constr in constrs_yp['beta_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_components_yp['beta_p'][i, j] = farkas
                cut_yp_expr.add(farkas * (model._x[j, i] - model._x[i, j]))
                cut_yp_val += farkas * (x_sol[j, i] - x_sol[i, j])
                
        for (i, j), constr in constrs_yp['gamma_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_components_yp['gamma_p'][i, j] = farkas
                cut_yp_expr.add(farkas * model._x[j, i])
                cut_yp_val += farkas * x_sol[j, i]
                
        for (i, j), constr in constrs_yp['delta_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_components_yp['delta_p'][i, j] = farkas
                cut_yp_expr.add(farkas * (1 - model._x[i, j]))
                cut_yp_val += farkas * (1 - x_sol[i, j])
        
        # --- Análisis por consola ---
        if model._save_cuts:
            print("\n" + "="*50)
            print("RAYO DE FARKAS DETECTADO EN SUBPROBLEMA Y'")
            print("Valor numérico de la violación (v'^T * b(x_bar)): ", cut_yp_val)
            for comp, values in v_components_yp.items():
                if values:
                    print(f"Componente {comp}:")
                    for k, v in values.items():
                        print(f"  {k}: {v:.4f}")
            print("="*50 + "\n")
        # --- NUEVO: Guardar en el log estructurado ---
        if getattr(model, '_save_cuts', False):
            log_farkas_ray(
                filepath=model._farkas_log_path,
                iteration=model._iteration,
                node_depth=0, # 0 porque estamos en MIPSOL
                subproblem_type='Y_prime',
                x_sol=x_sol,
                v_components=v_components_yp,
                violation_value=cut_yp_val,
                tolerance=TOL,
                cut_expr = cut_yp_expr
            )
        # --- Añadir el corte al maestro de forma segura ---
        # --- Añadir el corte al maestro de forma segura ---
        return cut_yp_expr, cut_yp_val, v_components_yp

import gurobipy as gp

def generate_pi_cut_y(constrs_y, x_sol, model, TOL):
    """
    Genera el corte de Benders usando variables duales (.Pi) de la Fase 1
    para el subproblema Y.
    """
    cut_expr = gp.LinExpr()
    cut_val = 0.0
    v_components = {'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}}

    # --- Alpha (RHS = x_ij) ---
    for (i, j), constr in constrs_y['alpha'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['alpha'][i, j] = pi_val
            cut_expr.add(pi_val * model._x[i, j])
            cut_val += pi_val * x_sol[i, j]

    # --- Beta (RHS = x_ij - x_ji) ---
    for (i, j), constr in constrs_y['beta'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['beta'][i, j] = pi_val
            cut_expr.add(pi_val * (model._x[i, j] - model._x[j, i]))
            cut_val += pi_val * (x_sol[i, j] - x_sol[j, i])

    # --- Gamma (RHS = x_ij) ---
    for (i, j), constr in constrs_y['gamma'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['gamma'][i, j] = pi_val
            cut_expr.add(pi_val * model._x[i, j])
            cut_val += pi_val * x_sol[i, j]

    # --- Delta (RHS = 1 - x_ji) ---
    for (i, j), constr in constrs_y['delta'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['delta'][i, j] = pi_val
            cut_expr.add(pi_val * (1 - model._x[j, i]))
            cut_val += pi_val * (1 - x_sol[j, i])

    constr_global = constrs_y['global']
    pi_global = constr_global.Pi
    if abs(pi_global) > TOL:
        rhs_global = constr_global.RHS
        v_components['global'] = pi_global

        cut_expr.add(pi_global * rhs_global)
        cut_val += pi_global * rhs_global
        
    
    return cut_expr, cut_val, v_components



def generate_pi_cut_yp(constrs_yp, x_sol, model, TOL):
    """
    Genera el corte de Benders usando variables duales (.Pi) de la Fase 1
    para el subproblema Y'.
    """
    cut_expr = gp.LinExpr()
    cut_val = 0.0
    v_components = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}}

    # --- Alpha_p (RHS = 1 - x_ij) ---
    for (i, j), constr in constrs_yp['alpha_p'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['alpha_p'][i, j] = pi_val
            cut_expr.add(pi_val * (1 - model._x[i, j]))
            cut_val += pi_val * (1 - x_sol[i, j])

    # --- Beta_p (RHS = x_ji - x_ij) ---
    for (i, j), constr in constrs_yp['beta_p'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['beta_p'][i, j] = pi_val
            cut_expr.add(pi_val * (model._x[j, i] - model._x[i, j]))
            cut_val += pi_val * (x_sol[j, i] - x_sol[i, j])

    # --- Gamma_p (RHS = x_ji) ---
    for (i, j), constr in constrs_yp['gamma_p'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['gamma_p'][i, j] = pi_val
            cut_expr.add(pi_val * model._x[j, i])
            cut_val += pi_val * x_sol[j, i]

    # --- Delta_p (RHS = 1 - x_ij) ---
    for (i, j), constr in constrs_yp['delta_p'].items():
        pi_val = constr.Pi
        if abs(pi_val) > TOL:
            v_components['delta_p'][i, j] = pi_val
            cut_expr.add(pi_val * (1 - model._x[i, j]))
            cut_val += pi_val * (1 - x_sol[i, j])
    
    
    constr_global = constrs_yp['global']
    pi_global = constr_global.Pi
    
    
    if abs(pi_global) > TOL:
        rhs_global = constr_global.RHS
        v_components['global'] = pi_global

        cut_expr.add(pi_global * rhs_global)
        cut_val += pi_global * rhs_global



    return cut_expr, cut_val, v_components






def generate_farkas_cut(sub_y, sub_yp, constrs_y, constrs_yp, x_sol, model, TOL = 1e-10):
    # Lógica para decidir cuál corte generar (si ambos subproblemas son infactibles, se pueden generar ambos cortes)
    if sub_y.Status == GRB.INFEASIBLE:
        cut_y_expr, cut_y_val, _ = generate_farkas_cut_y(constrs_y, x_sol, model, TOL)
        
        if cut_y_val > TOL:
            model.cbLazy(cut_y_expr <= 0)
        # Si el valor evaluado es negativo, el hiperplano debe forzarse hacia >= 0
        elif cut_y_val < -TOL:
            model.cbLazy(cut_y_expr >= 0)
    
    
    if sub_yp.Status == GRB.INFEASIBLE:
        cut_yp_expr, cut_yp_val, _ = generate_farkas_cut_yp(constrs_yp, x_sol, model, TOL)
       
        if cut_yp_val > TOL:
            model.cbLazy(cut_yp_expr <= 0)
        # Si el valor evaluado es negativo, el hiperplano debe forzarse hacia >= 0
        elif cut_yp_val < -TOL:
            model.cbLazy(cut_yp_expr >= 0)



def benders_callback(model, where):
    """
    Callback de Benders. Se ejecuta cuando el Maestro encuentra una solución entera (MIPSOL).
    Verifica la factibilidad en los DOS subproblemas independientes y añade cortes Lazy si es necesario.
    """
    if where == GRB.Callback.MIPSOL:
        # 1. Obtener la solución actual del Maestro (x barra)
        x_sol = model.cbGetSolution(model._x)

        update_subproblem_rhs(model, x_sol)

        model._sub_y.optimize()
        model._sub_yp.optimize()
        
        TOL = 1e-10
        
        if model._sub_y.ObjVal > TOL:
            cut_y_expr, cut_y_val, _ = generate_pi_cut_y(model._constrs_y, x_sol, model, TOL)
            if cut_y_val > TOL:
                model.cbLazy(cut_y_expr <= 0)

        if model._sub_yp.ObjVal > TOL:
            cut_yp_expr, cut_yp_val, _ = generate_pi_cut_yp(model._constrs_yp, x_sol, model, TOL)
            if cut_yp_val > TOL:
                model.cbLazy(cut_yp_expr <= 0)

        
        #generate_farkas_cut(model._sub_y, model._sub_yp, model._constrs_y, model._constrs_yp, x_sol, model, TOL)


    if where == GRB.Callback.MIPNODE:
        
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x_sol = model.cbGetNodeRel(model._x)

            update_subproblem_rhs(model, x_sol)

            model._sub_y.optimize()
            model._sub_yp.optimize()

            TOL = 1e-10

            if model._sub_y.Status == GRB.INFEASIBLE:
                cut_y_expr, cut_y_val, _ = generate_farkas_cut_y(model._constrs_y, x_sol, model, TOL)

                if cut_y_val > TOL:
                    model.cbCut(cut_y_expr <= 0)
                elif cut_y_val < -TOL:
                    model.cbCut(cut_y_expr >= 0)
        
            if model._sub_yp.Status == GRB.INFEASIBLE:
                cut_yp_expr, cut_yp_val, _ = generate_farkas_cut_yp(model._constrs_yp, x_sol, model, TOL)

                if cut_yp_val > TOL:
                    model.cbCut(cut_yp_expr <= 0) 
                elif cut_yp_val < -TOL:
                    model.cbCut(cut_yp_expr >= 0)






def build_master_problem(instance_path: str, verbose: bool = False, plot: bool = False, 
                         time_limit: int = 7200, maximize: bool = True, save_cuts: bool = False,
                         crosses_constrain: bool = False) -> gp.Model:
    
        # Lectura de datos
    points = read_indexed_instance(instance_path)
    N = range(len(points))
    CH = compute_convex_hull(points)
    triangles = compute_triangles(points)
    crossing = compute_crossing_edges(triangles, points)

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

    # Consideramos xij la variable de arcos dirigidos
    x = { (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in N for j in N  if i != j }
    
    # Consideramos f_ij la variable de subciclos
    f = {(i,j) : model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"f_{i}_{j}") for i in N for j in N if i != j }
    
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
    c = cost_function_area(points, x.keys(), mode=3)
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
            i, j, k, l = cross
            
            exists_edge_1 = (i, j) in x and (j, i) in x
            exists_edge_2 = (k, l) in x and (l, k) in x
            
            if exists_edge_1 and exists_edge_2:
                model.addConstr(
                    x[i, j] + x[j, i] + x[k, l] + x[l, k] <= 1, 
                    name=f"crossing_{i}_{j}_{k}_{l}"
                )

    # --- Construcción del Subproblema ---
    sub_y, sub_yp, constrs_y, constrs_yp = build_subproblems(points, N, CH, x.keys())
    model._x = x  
    model._sub_y = sub_y
    model._sub_yp = sub_yp
    model._constrs_y = constrs_y
    model._constrs_yp = constrs_yp

    model._benders_ = True

    return model



def optimize_master_MILP(instance_path: str, verbose: bool = False, plot: bool = False, 
                         time_limit: int = 7200, maximize: bool = True, save_cuts: bool = False,
                         crosses_constrain: bool = False) -> gp.Model:
    """
    Construye y resuelve el Problema Maestro (PM) usando Descomposición de Benders.
    """

    model = build_master_problem(
        instance_path,
        verbose=verbose,
        plot=plot,
        time_limit=time_limit,
        maximize=maximize,
        save_cuts=save_cuts,
        crosses_constrain=crosses_constrain
    )

    # --- Optimización ---
    if verbose:
        print("Starting optimization with Benders decomposition...")
    
    if model._save_cuts:
        model._instance_name = instance_path.split('/')[-1].replace('.instance', '')
        model._iteration = 0
        
        # NUEVO: Ruta para el log de rayos de Farkas
        if crosses_constrain:
            model._farkas_log_path = f"outputs/Others/Benders/{model._instance_name}-Crosses/farkas_log.jsonl"
        else:
            model._farkas_log_path = f"outputs/Others/Benders/{model._instance_name}/farkas_log.jsonl"
        # Limpiar el archivo si ya existe de una corrida anterior
        if os.path.exists(model._farkas_log_path):
            os.remove(model._farkas_log_path)

    model.optimize(benders_callback)
    x = model._x


    # --- Resultados ---
    model._x_results = []
    if model.SolCount > 0:
        for k, v in x.items():
            if v.X > 0.5:
                model._x_results.append(k)
        
        if plot:
            # Asumiendo que existe una función plot_solution en utils
            plot_solution(model, title="Optimal Tour" if model.Status == GRB.OPTIMAL else "Best Found")
    
    model._instance_name = instance_path.split('/')[-1].replace('.instance', '')
    model.write(f"outputs/Others/Benders/{model._instance_name}.lp")

    return model


def optimize_master_LP(instance_path: str, verbose: bool = False, plot: bool = False, 
                         time_limit: int = 7200, maximize: bool = True, save_cuts: bool = False,
                         crosses_constrain: bool = False) -> gp.Model:
    """
    Construye y resuelve la relajación LP del Problema Maestro (PM) usando Descomposición de Benders.
    """
    model = build_master_problem(
        instance_path,
        verbose=verbose,
        plot=plot,
        time_limit=time_limit,
        maximize=maximize,
        save_cuts=save_cuts,
        crosses_constrain=crosses_constrain
    )

    model.params.OutputFlag = 0
    model.params.Presolve = 1
    model.params.LazyConstraints = 0
    model.params.Presolve = 1


    # Cambiamos manualmente el tipo de todas las variables a continuas
    for v in model.getVars():
        if v.VType != GRB.CONTINUOUS:
            v.VType = GRB.CONTINUOUS
    model.update()
    
    sub_y = model._sub_y
    sub_yp = model._sub_yp

    constrs_y = model._constrs_y
    constrs_yp = model._constrs_yp

    model._iteration = 0
    TOL = 1e-10

    converged = False
    converged_sub_y = False
    converged_sub_yp = False

    if verbose:
        print("Starting optimization of LP relaxation with Benders decomposition...")
    

    while not converged:

        if verbose:
                print("\n=== Iteración: {} ===".format(model._iteration))


        model.optimize()
        x_sol = {k: v.X for k, v in model._x.items()}

        update_subproblem_rhs(model, x_sol)

        sub_y.optimize()
        sub_yp.optimize()


        # --- Evaluación Subproblema Y ---
        if sub_y.ObjVal <= TOL:
            converged_sub_y = True
        else:
            converged_sub_y = False
            cut_y_expr, cut_y_val, _ = generate_pi_cut_y(constrs_y, x_sol, model, TOL)

            # Como el Subproblema Fase 1 siempre busca minimizar una función objetivo positiva
            # (las variables artificiales >= 0), la violación (cut_y_val) siempre será positiva.
            # Por tanto, forzamos a que esa combinación valga <= 0.
            if cut_y_val > TOL:
                model.addConstr(cut_y_expr <= 0, name=f"Phase1_Cut_Y_iter_{model._iteration}")

        # --- Evaluación Subproblema Y' ---
        if sub_yp.ObjVal <= TOL:
            converged_sub_yp = True
        else:
            converged_sub_yp = False
            cut_yp_expr, cut_yp_val, _ = generate_pi_cut_yp(constrs_yp, x_sol, model, TOL)

            if cut_yp_val > TOL:
                model.addConstr(cut_yp_expr <= 0, name=f"Phase1_Cut_YP_iter_{model._iteration}")




        """if sub_y.Status == GRB.OPTIMAL:
            converged_sub_y = True # El maestro aproximó bien, terminamos.
                
        elif sub_y.Status == GRB.INFEASIBLE:
            converged_sub_y = False
            cut_y_expr, cut_y_val, _ = generate_farkas_cut_y(constrs_y, x_sol, model, TOL)

            if cut_y_val > TOL:
                model.addConstr(cut_y_expr <= 0, name=f"benders_cut_y_{model._iteration}")
            elif cut_y_val < -TOL:
                model.addConstr(cut_y_expr >= 0, name=f"benders_cut_y_{model._iteration}")
        else:
            converged_sub_y = False



        if sub_yp.Status == GRB.OPTIMAL:
            # Evaluar si el costo real del SP es mayor que la aproximación theta
            converged_sub_yp = True # El maestro aproximó bien, terminamos.
                
        elif sub_yp.Status == GRB.INFEASIBLE:
            converged_sub_yp = False
            cut_yp_expr, cut_yp_val, _ = generate_farkas_cut_yp(constrs_yp, x_sol, model, TOL)

            if cut_yp_val > TOL:
                model.addConstr(cut_yp_expr <= 0, name=f"benders_cut_yp_{model._iteration}")
            elif cut_yp_val < -TOL:
                model.addConstr(cut_yp_expr >= 0, name=f"benders_cut_yp_{model._iteration}")

        else:
            converged_sub_yp = False
        """
        model._iteration += 1   

        if converged_sub_y and converged_sub_yp:
            converged = True
        

    print("LP Relaxation converged after {} iterations.".format(model._iteration))

    return model