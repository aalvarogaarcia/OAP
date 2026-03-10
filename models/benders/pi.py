
import gurobipy as gp
from gurobipy import GRB
from utils.utils import compute_triangles, triangles_adjacency_list


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

def generate_pi_cut(constrs_y, constrs_yp, x_sol, model, TOL = 1e-10):
    if model._sub_y.ObjVal > TOL:
        cut_y_expr, cut_y_val, _ = generate_pi_cut_y(constrs_y, x_sol, model, TOL)
        if cut_y_val > TOL:
            model.cbLazy(cut_y_expr <= 0)

    if model._sub_yp.ObjVal > TOL:
        cut_yp_expr, cut_yp_val, _ = generate_pi_cut_yp(constrs_yp, x_sol, model, TOL)
        if cut_yp_val > TOL:
            model.cbLazy(cut_yp_expr <= 0)

def build_pi_subproblems(points, N, CH, master_x_keys):
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


