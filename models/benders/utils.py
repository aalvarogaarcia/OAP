import os
import json

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


def log_farkas_ray(filepath: str, iteration: int, node_depth: int, subproblem_type: str, 
                   x_sol: dict, v_components: dict, violation_value: float, tolerance: float = 1e-5, cut_expr: str = None):
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
    active_x = {
        f"{i}_{j}": round(val, 4) 
        for (i, j), val in x_sol.items() 
        if abs(val) > tolerance
    }
    
    # 2. Estructurar los componentes del rayo
    ray_data = {}
    for comp_name, values in v_components.items():
        if values:  # Solo procesamos si el diccionario no está vacío
            # Convertimos las tuplas (i, j) a strings "i_j" porque JSON no soporta tuplas como keys
            ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in values.items()}
            
    serialized_cut = serialize_expr(cut_expr) if cut_expr else None

    # 3. Crear el registro consolidado
    registro = {
        "iteration": iteration,
        "node_depth": node_depth,
        "subproblem": subproblem_type,
        "violation": round(violation_value, 6),
        "active_x": active_x,
        "ray_components": ray_data,
        "cut_expr": serialized_cut
    }
    
    # 4. Asegurar que el directorio existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 5. Escribir en formato JSON Lines (append mode)
    with open(filepath, 'a') as f:
        f.write(json.dumps(registro) + '\n')




def serialize_expr(expr):
    """Convierte una expresión lineal de Gurobi en un diccionario serializable."""
    if expr is None:
        return None
    
    # Creamos un diccionario: {"nombre_variable": coeficiente}
    # Por ejemplo: {"x_0_1": 1.0, "x_2_3": -1.0}
    coeffs = {}
    for i in range(expr.size()):
        var = expr.getVar(i)
        coeff = expr.getCoeff(i)
        if abs(coeff) > 1e-6:
            coeffs[var.VarName] = round(coeff, 4)
    
    return {
        "coeffs": coeffs,
        "constant": round(expr.getConstant(), 4)
    }


def format_cut_string(cut_expr):
    """Convierte el log entry en algo como '1.0 <= x_0_1 + x_2_1'"""
    const = cut_expr.get('constant', 0.0)
    coeffs = cut_expr.get('coeffs', {})
    
    # Siguiendo la lógica: const <= sum(-coeff * x)
    parts = []
    for var, val in coeffs.items():
        if val < 0:
            parts.append(f"{abs(val)}*{var}" if abs(val) != 1 else var)
        elif val > 0:
            parts.append(f"- {val}*{var}" if val != 1 else f"- {var}")
            
    formula = " + ".join(parts).replace("+ -", "- ")
    return f"Corte Lógico: {const} <= {formula}"
