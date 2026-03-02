import json
import networkx as nx
import matplotlib.pyplot as plt
import os

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
    elif show_plot:  # <--- NUEVO PARÁMETRO
        plt.show()