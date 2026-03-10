import gurobipy as gp
from gurobipy import GRB
from models.benders.optimize import log_farkas_ray
from utils.utils import compute_triangles, triangles_adjacency_list
from models.benders.utils import log_farkas_ray, load_farkas_logs


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



def build_farkas_subproblems(points, N, CH, master_x_keys):
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
    sub_y.Params.InfUnbdInfo = 1
    sub_y.Params.DualReductions = 0

    y = sub_y.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="y")
    sub_y.Params.Presolve = 1
    
    #--------------------------------------------------------------
    #---------               MODELO Y'               --------------
    #--------------------------------------------------------------
    
    sub_yp = gp.Model("Subproblem_YP")
    sub_yp.Params.OutputFlag = 0
    sub_yp.Params.InfUnbdInfo = 1
    sub_yp.Params.DualReductions = 0
    
    yp = sub_yp.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="yp")
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
            gp.quicksum(y[t] for t in ta[i][j])  == 0, name=f"alpha_{i}_{j}"
        )
        # Eq (sp:11): sum(y') = 1 - x_bar_ij -> Placeholder RHS = 0
        constrs_yp['alpha_p'][i, j] = sub_yp.addConstr(
            gp.quicksum(yp[t] for t in ta[i][j]) == 0, name=f"alpha_p_{i}_{j}"
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
                    gp.quicksum(y[t] for t in ta[i][j]) - gp.quicksum(y[t] for t in ta[j][i]) == 0,
                    name=f"beta_{i}_{j}"
                )
                # Eq (sp:12): sum(yp_ij) - sum(yp_ji) = x_bar_ji - x_bar_ij
                constrs_yp['beta_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j]) - gp.quicksum(yp[t] for t in ta[j][i]) == 0,
                    name=f"beta_p_{i}_{j}"
                )

            # A'': Todos los arcos dirigidos (i != j) para cotas superior e inferior
            if is_non_ch and i != j:
                # Eq (sp:8): sum(y) >= x_bar_ij
                constrs_y['gamma'][i, j] = sub_y.addConstr(
                    gp.quicksum(y[t] for t in ta[i][j]) >= 0, name=f"gamma_{i}_{j}"
                )
                # Eq (sp:9): sum(y) <= 1 - x_bar_ji
                constrs_y['delta'][i, j] = sub_y.addConstr(
                    gp.quicksum(y[t] for t in ta[i][j]) <= 0, name=f"delta_{i}_{j}"
                )
                
                # Eq (sp:13): sum(yp) >= x_bar_ji
                constrs_yp['gamma_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j])  >= 0, name=f"gamma_p_{i}_{j}"
                )
                # Eq (sp:14): sum(yp) <= 1 - x_bar_ij
                constrs_yp['delta_p'][i, j] = sub_yp.addConstr(
                    gp.quicksum(yp[t] for t in ta[i][j]) <= 0, name=f"delta_p_{i}_{j}"
                )

    sub_y.update()
    sub_yp.update()
    
    return sub_y, sub_yp, constrs_y, constrs_yp
