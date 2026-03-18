import os
import json
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


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
        if values:  # Solo procesamos si el diccionario no está vacío (o si el float no es 0)
            if isinstance(values, dict):
                # Convertimos las tuplas (i, j) a strings "i_j" porque JSON no soporta tuplas como keys
                ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in values.items()}
            else:
                # Es un valor escalar (ej. la restricción global)
                ray_data[comp_name] = round(values, 4)

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
    """Convierte el log entry en algo como '1.0 >= x_0_1 + x_2_1'"""
    const = cut_expr.get('constant', 0.0)
    coeffs = cut_expr.get('coeffs', {})
    
    # Siguiendo la lógica: const >= sum(-coeff * x)
    parts = []
    for var, val in coeffs.items():
        if val < 0:
            parts.append(f"{abs(val)}*{var}" if abs(val) != 1 else var)
        elif val > 0:
            parts.append(f"- {val}*{var}" if val != 1 else f"- {var}")
            
    formula = " + ".join(parts).replace("+ -", "- ")
    return f"Corte Lógico: {const} >= {formula}"


def load_farkas_logs(filepath: str):
    """Carga el historial de rayos de Farkas desde un archivo .jsonl."""
    logs = []
    if not os.path.exists(filepath):
        print(f"El archivo {filepath} no existe.")
        return logs
        
    with open(filepath, 'r') as f:
        for line in f:
            logs.append(json.loads(line.strip()))
    return logs

def parse_edge(edge_str: str):
    """Convierte un string 'i_j' en una tupla de enteros (i, j)."""
    i, j = edge_str.split('_')
    return int(i), int(j)





def plot_farkas_ray_network(log_entry: dict, points: dict = None, save_path: str = None, show_plot: bool = True):
    """
    Dibuja la red mostrando la solución candidata x_bar y los componentes del rayo de Farkas.
    
    Args:
        log_entry: Un diccionario extraído del archivo .jsonl.
        points: Diccionario con las coordenadas de los nodos {id: (x, y)}. Si es None, se autogeneran.
        save_path: Ruta donde guardar la imagen. Si es None, se muestra en pantalla.
    """
    G = nx.DiGraph()
    
    # Extraer datos del log
    iteration = log_entry.get("iteration", "N/A")
    subproblem = log_entry.get("subproblem", "Unknown")
    violation = log_entry.get("violation", 0.0)
    active_x_strs = log_entry.get("active_x", {})
    ray_components = log_entry.get("ray_components", {})

    # 1. Añadir aristas de la solución candidata (x_bar)
    active_edges = [parse_edge(e) for e in active_x_strs.keys()]
    G.add_edges_from(active_edges)

    # 2. Configurar layout (posiciones de los nodos)
    if points is not None:
        pos = {i: points[i] for i in G.nodes()}
    else:
        # Layout por defecto si no se proveen coordenadas reales
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # 3. Dibujar la solución candidata (líneas grises discontinuas)
    nx.draw_networkx_edges(G, pos, edgelist=active_edges, 
                           edge_color='gray', style='dashed', alpha=0.6, width=2, arrows=True)

    # 4. Colores para los diferentes componentes del rayo de Farkas
    # Ajusta estos colores según tus preferencias y los nombres de tus restricciones
    color_map = {
        'alpha': 'red', 'alpha_p': 'darkred',
        'beta': 'blue', 'beta_p': 'darkblue',
        'gamma': 'green', 'gamma_p': 'darkgreen',
        'delta': 'orange', 'delta_p': 'darkorange'
    }

    # 5. Dibujar las aristas involucradas en el rayo de Farkas
    legend_handles = []
    for comp_name, edges_dict in ray_components.items():
        # Si está vacío o NO es un diccionario (es el offset global), nos lo saltamos para el grafo
        if not edges_dict or not isinstance(edges_dict, dict):
            continue
            
        color = color_map.get(comp_name, 'purple') # Púrpura por defecto si no está en el mapa
        farkas_edges = [parse_edge(e) for e in edges_dict.keys()]
        
        # Asegurarnos de que los nodos del rayo existan en el grafo para dibujarlos
        G.add_edges_from(farkas_edges)
        if points is None: # Actualizar posiciones si añadimos nodos nuevos
            pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), seed=42)
            
        nx.draw_networkx_edges(G, pos, edgelist=farkas_edges, 
                               edge_color=color, width=2.5, arrows=True, connectionstyle="arc3,rad=0.1")
        
        # Etiqueta para la leyenda
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.5, label=f'Rayo {comp_name}'))

        
    # Línea representativa de x_bar para la leyenda
    legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle='dashed', lw=2, label=r'Solución candidata $\bar{x}$'))

    # Título y Leyenda
    plt.title(f"Iteración: {iteration} | Subproblema: {subproblem}\nViolación: {violation}", fontsize=14)
    plt.legend(handles=legend_handles, loc='best')
    plt.axis('off')

    # Guardar o mostrar
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    elif show_plot:
        plt.show()





def plot_cut_heatmap(log_entry: dict, num_nodes: int, save_path: str = None, show_plot: bool = False):
    cut_data = log_entry.get("cut_expr")
    if not cut_data: return

    # Crear matriz vacía
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for var_name, coeff in cut_data["coeffs"].items():
        # Extraer i, j del nombre de la variable (ej: x_0_1)
        try:
            parts = var_name.split('_')
            i, j = int(parts[1]), int(parts[2])
            adj_matrix[i, j] = coeff
        except: continue

    plt.figure(figsize=(6, 5))
    sns.heatmap(adj_matrix, annot=True, cmap="RdBu", center=0)
    plt.title(f"Estructura del Corte - Iteración {log_entry['iteration']}")
    plt.xlabel("Nodo Destino (j)")
    plt.ylabel("Nodo Origen (i)")
    # Guardar o mostrar

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    elif show_plot: 
        plt.show()


def plot_cut_weights(log_entry: dict, save_path: str = None, show_plot: bool = False):
    """Dibuja un gráfico de barras con los pesos (coeficientes) del corte."""
    cut_data = log_entry.get("cut_expr")
    if not cut_data or not cut_data["coeffs"]:
        print("No hay datos de expresión en este registro.")
        return

    coeffs = cut_data["coeffs"]
    # Ordenar por magnitud del coeficiente
    sorted_coeffs = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
    
    names = list(sorted_coeffs.keys())
    values = list(sorted_coeffs.values())

    plt.figure(figsize=(10, 5))
    colors = ['red' if v < 0 else 'blue' for v in values]
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Pesos del Corte - Iteración {log_entry['iteration']} ({log_entry['subproblem']})")
    plt.ylabel("Coeficiente en el Maestro")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    elif show_plot:  
        plt.show()









def plot_sankey_traceability(log_entry: dict, save_path: str = None, show_plot: bool = False):
    trace = log_entry.get("traceability")
    
    sources = [] # Componentes Duales
    targets = [] # Variables x_ij
    values = []  # Pesos
    
    for x_var, info in trace.items():
        sources.append(info["source_component"])
        targets.append(x_var)
        values.append(abs(info["dual_value"]))
        
    # Plotly necesita índices numéricos para el Sankey
    nodes = list(set(sources + targets))
    source_idx = [nodes.index(s) for s in sources]
    target_idx = [nodes.index(t) for t in targets]
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(label = nodes, pad=15, thickness=20),
        link = dict(source = source_idx, target = target_idx, value = values)
    )])
    
    fig.update_layout(title_text="Trazabilidad: Del Subproblema al Maestro", font_size=12)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    elif show_plot:
        fig.show()


