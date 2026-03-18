import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import gurobipy as gp
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from typing import Any, cast


Arc = tuple[int, int]
SerializedCoeffMap = dict[str, float]
SerializedExpr = dict[str, SerializedCoeffMap | float]
SerializedRayData = dict[str, SerializedCoeffMap | float]
PointLookup = dict[int, tuple[float, float]] | NDArray[np.int64]
ArcConstraintMap = dict[Arc, gp.Constr]


def update_subproblem_rhs(model: gp.Model, x_sol: dict[Arc, float]) -> None:
        constrs_y = cast(dict[str, ArcConstraintMap], model._constrs_y)
        constrs_yp = cast(dict[str, ArcConstraintMap], model._constrs_yp)

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
                   x_sol: dict[tuple[int, int], float], v_components: dict[str, Any], violation_value: float, tolerance: float = 1e-5, cut_expr: gp.LinExpr | None = None) -> None:
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
    active_x: SerializedCoeffMap = {
        f"{i}_{j}": round(val, 4)
        for (i, j), val in x_sol.items()
        if abs(val) > tolerance
    }

    # 2. Estructurar los componentes del rayo
    ray_data: SerializedRayData = {}
    for comp_name, values in v_components.items():
        if values:
            if isinstance(values, dict):
                typed_values = cast(dict[Arc, float], values)
                ray_data[comp_name] = {f"{k[0]}_{k[1]}": round(v, 4) for k, v in typed_values.items()}
            else:
                ray_data[comp_name] = round(cast(float, values), 4)

    serialized_cut = serialize_expr(cut_expr) if cut_expr else None

    # 3. Crear el registro consolidado
    registro: dict[str, Any] = {
        "iteration": iteration,
        "node_depth": node_depth,
        "subproblem": subproblem_type,
        "violation": round(violation_value, 6),
        "active_x": active_x,
        "ray_components": ray_data,
        "cut_expr": serialized_cut,
    }
    
    # 4. Asegurar que el directorio existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 5. Escribir en formato JSON Lines (append mode)
    with open(filepath, 'a') as f:
        f.write(json.dumps(registro) + '\n')




def serialize_expr(expr: gp.LinExpr | None) -> SerializedExpr | None:
    """Convierte una expresión lineal de Gurobi en un diccionario serializable."""
    if expr is None:
        return None
    
    # Creamos un diccionario: {"nombre_variable": coeficiente}
    # Por ejemplo: {"x_0_1": 1.0, "x_2_3": -1.0}
    coeffs: dict[str, float] = {}
    for i in range(expr.size()):
        var = expr.getVar(i)
        coeff = expr.getCoeff(i)
        if abs(coeff) > 1e-6:
            coeffs[var.VarName] = round(coeff, 4)
    
    return {
        "coeffs": coeffs,
        "constant": round(expr.getConstant(), 4)
    }


def format_cut_string(cut_expr: SerializedExpr) -> str:
    """Convierte el log entry en algo como '1.0 <= x_0_1 + x_2_1'"""
    const = cast(float, cut_expr.get('constant', 0.0))
    coeffs = cast(SerializedCoeffMap, cut_expr.get('coeffs', {}))
    
    # Siguiendo la lógica: const <= sum(-coeff * x)
    parts: list[str] = []
    for var, val in coeffs.items():
        if val < 0:
            parts.append(f"{abs(val)}*{var}" if abs(val) != 1 else var)
        elif val > 0:
            parts.append(f"- {val}*{var}" if val != 1 else f"- {var}")
            
    formula = " + ".join(parts).replace("+ -", "- ")
    return f"Corte Lógico: {const} <= {formula}"


def load_farkas_logs(filepath: str) -> list[dict[str, Any]]:
    """Carga el historial de rayos de Farkas desde un archivo .jsonl."""
    logs: list[dict[str, Any]] = []
    if not os.path.exists(filepath):
        print(f"El archivo {filepath} no existe.")
        return logs
        
    with open(filepath, 'r') as f:
        for line in f:
            logs.append(cast(dict[str, Any], json.loads(line.strip())))
    return logs

def parse_edge(edge_str: str) -> tuple[int, int]:
    """Convierte un string 'i_j' en una tupla de enteros (i, j)."""
    i, j = edge_str.split('_')
    return int(i), int(j)





def plot_farkas_ray_network(
    log_entry: dict[str, Any],
    points: PointLookup | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    Dibuja la red mostrando la solución candidata x_bar y los componentes del rayo de Farkas.
    
    Args:
        log_entry: Un diccionario extraído del archivo .jsonl.
        points: Diccionario con las coordenadas de los nodos {id: (x, y)}. Si es None, se autogeneran.
        save_path: Ruta donde guardar la imagen. Si es None, se muestra en pantalla.
    """
    nx_api: Any = nx
    plt_api: Any = plt
    G: Any = nx_api.DiGraph()
    
    # Extraer datos del log
    iteration = log_entry.get("iteration", "N/A")
    subproblem = log_entry.get("subproblem", "Unknown")
    violation = log_entry.get("violation", 0.0)
    active_x_strs = cast(SerializedCoeffMap, log_entry.get("active_x", {}))
    ray_components = cast(SerializedRayData, log_entry.get("ray_components", {}))

    # 1. Añadir aristas de la solución candidata (x_bar)
    active_edges: list[Arc] = [parse_edge(e) for e in active_x_strs.keys()]
    G.add_edges_from(active_edges)

    # 2. Configurar layout (posiciones de los nodos)
    pos: dict[int, tuple[float, float]] | dict[int, NDArray[np.float64]]
    if points is not None:
        if isinstance(points, np.ndarray):
            pos = {int(i): (float(points[i][0]), float(points[i][1])) for i in G.nodes()}
        else:
            pos = {int(i): points[int(i)] for i in G.nodes()}
    else:
        pos = cast(dict[int, NDArray[np.float64]], nx_api.spring_layout(G, seed=42))

    plt_api.figure(figsize=(10, 8))
    nx_api.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300, edgecolors='black')
    nx_api.draw_networkx_labels(G, pos, font_size=10)
    nx_api.draw_networkx_edges(
        G,
        pos,
        edgelist=active_edges,
        edge_color='gray',
        style='dashed',
        alpha=0.6,
        width=2,
        arrows=True,
    )

    color_map: dict[str, str] = {
        'alpha': 'red', 'alpha_p': 'darkred',
        'beta': 'blue', 'beta_p': 'darkblue',
        'gamma': 'green', 'gamma_p': 'darkgreen',
        'delta': 'orange', 'delta_p': 'darkorange',
    }

    legend_handles: list[Line2D] = []
    for comp_name, edges_dict in ray_components.items():
        if not edges_dict or not isinstance(edges_dict, dict):
            continue

        color = color_map.get(comp_name, 'purple')
        farkas_edges: list[Arc] = [parse_edge(edge_name) for edge_name in edges_dict.keys()]

        G.add_edges_from(farkas_edges)
        if points is None:
            pos = cast(dict[int, NDArray[np.float64]], nx_api.spring_layout(G, pos=pos, fixed=list(pos.keys()), seed=42))

        nx_api.draw_networkx_edges(
            G,
            pos,
            edgelist=farkas_edges,
            edge_color=color,
            width=2.5,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f'Rayo {comp_name}'))

    legend_handles.append(Line2D([0], [0], color='gray', linestyle='dashed', lw=2, label=r'Solución candidata $\bar{x}$'))

    plt_api.title(f"Iteración: {iteration} | Subproblema: {subproblem}\nViolación: {violation}", fontsize=14)
    plt_api.legend(handles=legend_handles, loc='best')
    plt_api.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()





def plot_cut_heatmap(
    log_entry: dict[str, Any],
    num_nodes: int,
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    plt_api: Any = plt
    sns_api: Any = sns
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data:
        return

    adj_matrix: NDArray[np.float64] = np.zeros((num_nodes, num_nodes))
    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    
    for var_name, coeff in coeffs.items():
        try:
            parts = var_name.split('_')
            i, j = int(parts[1]), int(parts[2])
            adj_matrix[i, j] = coeff
        except Exception:
            continue

    plt_api.figure(figsize=(6, 5))
    sns_api.heatmap(adj_matrix, annot=True, cmap="RdBu", center=0)
    plt_api.title(f"Estructura del Corte - Iteración {log_entry['iteration']}")
    plt_api.xlabel("Nodo Destino (j)")
    plt_api.ylabel("Nodo Origen (i)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()


def plot_cut_weights(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Dibuja un gráfico de barras con los pesos (coeficientes) del corte."""
    plt_api: Any = plt
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data or not cut_data["coeffs"]:
        print("No hay datos de expresión en este registro.")
        return

    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    sorted_coeffs: SerializedCoeffMap = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
    
    names: list[str] = list(sorted_coeffs.keys())
    values: list[float] = list(sorted_coeffs.values())

    plt_api.figure(figsize=(10, 5))
    colors = ['red' if v < 0 else 'blue' for v in values]
    plt_api.bar(names, values, color=colors)
    plt_api.xticks(rotation=45, ha='right')
    plt_api.title(f"Pesos del Corte - Iteración {log_entry['iteration']} ({log_entry['subproblem']})")
    plt_api.ylabel("Coeficiente en el Maestro")
    plt_api.grid(axis='y', linestyle='--', alpha=0.7)
    plt_api.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_api.savefig(save_path, bbox_inches='tight', dpi=300)
        plt_api.close()
    elif show_plot:
        plt_api.show()









def plot_sankey_traceability(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    plotly_api: Any = go
    trace = cast(dict[str, dict[str, Any]] | None, log_entry.get("traceability"))
    if not trace:
        return
    
    sources: list[str] = []
    targets: list[str] = []
    values: list[float] = []
    
    for x_var, info in trace.items():
        sources.append(cast(str, info["source_component"]))
        targets.append(x_var)
        values.append(abs(float(cast(float, info["dual_value"]))))
        
    nodes = list(set(sources + targets))
    source_idx = [nodes.index(s) for s in sources]
    target_idx = [nodes.index(t) for t in targets]
    
    fig: Any = plotly_api.Figure(data=[plotly_api.Sankey(
        node=dict(label=nodes, pad=15, thickness=20),
        link=dict(source=source_idx, target=target_idx, value=values),
    )])
    
    fig.update_layout(title_text="Trazabilidad: Del Subproblema al Maestro", font_size=12)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    elif show_plot:
        fig.show()


