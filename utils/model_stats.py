from gurobipy import GRB
import gurobipy as gp
from docplex.mp.model import Model
import networkx as nx

def get_ObjVal_int(model):
    """
    Retorna el valor objetivo entero del modelo si existe solución.
    Si no hay solución, retorna None.
    """
    if model and model.SolCount > 0:
        x = model._x_results
        G = nx.DiGraph()
        G.add_edges_from(x)
        obj_val = 0
        for a in x:
            i = model._points_[a[0]]
            j = model._points_[a[1]]

            obj_val += (i[0]*j[1] - j[0]*i[1])/2
        
        return obj_val





def get_model_stats(model, relaxed_model):
    """
    Extrae estadísticas clave del modelo y su relajación.
    Retorna: (LP_Val, Gap, IP_Val, Time, Nodes)
    """

    if model and model.SolCount > 0:
        ip_val = get_ObjVal_int(model)
        time_s = model.Runtime
        nodes = model.NodeCount
    else:
        return "-", "-", "-", "-", "-"

    if relaxed_model and relaxed_model.SolCount > 0:
        lp_val = relaxed_model.ObjVal
    else:
        lp_val = 0 # O manejar error

    # Calcular Gap: (IP - LP )/ IP * 100 if MinArea (evitando división por cero)
    # Si MaxArea, el gap es (LP - IP) / (Area(CH)-IP) * 100
    gap = 0.0
    if ip_val != 0 and model.ModelSense == GRB.MINIMIZE:
        gap = (ip_val - lp_val) / ip_val * 100
    
    elif ip_val != 0 and model.ModelSense == GRB.MAXIMIZE:
        # Asumimos que el área del casco convexo es accesible como atributo
        area_ch = model._convex_hull_area if hasattr(model, '_convex_hull_area') else None
        #print(f"Convex Hull Area for gap calculation: {area_ch}")
        if area_ch is not None and (area_ch - ip_val) != 0:
            gap = (lp_val - ip_val) / (area_ch - ip_val) * 100
    
    return lp_val, gap, ip_val, time_s, nodes


def get_model_stats_cplex(model: Model, relaxed_model: Model):
    """
    Extrae estadísticas clave del modelo CPLEX y su relajación.
    Retorna: (LP_Val, Gap, IP_Val, Time, Nodes)
    """

    # 1. Validar si el modelo principal (IP) tiene solución
    # En docplex, 'model.solution' es None si no se encontró solución.
    if model and model.solution:
        ip_val = model.objective_value
        
        # El tiempo y los nodos se extraen de los detalles de la resolución
        solve_details = model.get_solve_details()
        time_s = solve_details.time  # Tiempo en segundos
        nodes = solve_details.nb_nodes_processed
    else:
        return "-", "-", "-", "-", "-"

    # 2. Validar si el modelo relajado (LP) tiene solución
    if relaxed_model and relaxed_model.solution:
        lp_val = relaxed_model.objective_value
    else:
        lp_val = 0 # O manejar error de relajación infactible

    # 3. Calcular Gap
    # (IP - LP )/ IP * 100 if MinArea (evitando división por cero)
    # Si MaxArea, el gap es (LP - IP) / (Area(CH)-IP) * 100
    gap = 0.0
    
    # Detectar sentido de optimización (Minimizar vs Maximizar)
    is_minimization = model.objective_sense.is_minimize() 
    
    if ip_val != 0 and is_minimization:
        # Gap estándar para minimización
        gap = (ip_val - lp_val) / ip_val * 100
    
    elif ip_val != 0 and not is_minimization: # Maximización
        # Asumimos que el área del casco convexo fue guardada en el atributo _convex_hull_area
        area_ch = model._convex_hull_area if hasattr(model, '_convex_hull_area') else None
        
        # print(f"Convex Hull Area for gap calculation: {area_ch}")
        
        if area_ch is not None and (area_ch - ip_val) != 0:
            # Fórmula específica para OAP (Area Poligonización)
            gap = (lp_val - ip_val) / (area_ch - ip_val) * 100
        else:
            # Fallback a cálculo estándar si no hay Convex Hull info
            gap = (lp_val - ip_val) / abs(ip_val) * 100
    
    return lp_val, gap, ip_val, time_s, nodes