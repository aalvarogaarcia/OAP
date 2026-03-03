import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import plotly.graph_objects as go



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
        if not edges_dict:
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


