import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
from numpy.typing import NDArray
from utils.utils import segments_intersect

# --- 1. CONFIGURACIÓN DE ARCOS ---
arcs_left = [(1, 3), (2, 3), (2, 9)]
arcs_right = [(3, 2), (9, 2), (3, 4)]
all_arcs = arcs_left + arcs_right


def get_var_name(arc):
    return f"x_{arc[0]}_{arc[1]}"

vars_left = [get_var_name(a) for a in arcs_left]
vars_right = [get_var_name(a) for a in arcs_right]
all_vars = vars_left + vars_right


# --- 2. LECTURA DE INSTANCIA ---
def load_instance(filename):
    pos = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if len(parts) == 3:
                    # Guardamos las coordenadas como arreglos de numpy para la función de intersección
                    pos[int(parts[0])] = np.array([float(parts[1]), float(parts[2])], dtype=np.int64)
    except FileNotFoundError:
        print(f"⚠️ No se encontró {filename}. Usando posiciones de respaldo.")
        pos = {
            1: np.array([0, 10], dtype=np.int64), 
            2: np.array([5, 0], dtype=np.int64), 
            3: np.array([5, 10], dtype=np.int64), 
            4: np.array([10, 10], dtype=np.int64), 
            9: np.array([10, 0], dtype=np.int64)
        }
    return pos


# --- 4. LÓGICA CON FILTROS ---
def get_valid_cases(a_left, a_right, posiciones, constant: int = 0):
    anti_pairs = []
    for a in a_left:
        reverse = (a[1], a[0])
        if reverse in a_right:
            anti_pairs.append((get_var_name(a), get_var_name(reverse)))

    valid = []
    for comb in itertools.product([0, 1], repeat=len(all_vars)):
        
        if sum(comb) == 0:
            continue # Filtro trivial
        
        d = dict(zip(all_vars, comb))
        
        # Filtro de antisimetría
        if any(d[v1] + d[v2] > 1 for v1, v2 in anti_pairs):
            continue
            
        active_arcs = [a for a in all_arcs if d[get_var_name(a)] == 1]
        
        # NUEVO: Filtro de Grado (Máx 1 entrada y Máx 1 salida por nodo)
        in_degree = {}
        out_degree = {}
        grado_invalido = False
        
        for origen, destino in active_arcs:
            out_degree[origen] = out_degree.get(origen, 0) + 1
            in_degree[destino] = in_degree.get(destino, 0) + 1
            
            # Si algún nodo supera 1 salida o 1 entrada, descartamos
            if out_degree[origen] > 1 or in_degree[destino] > 1:
                grado_invalido = True
                break
                
        if grado_invalido:
            continue
            
        # Filtro de Clique de Intersecciones
        has_intersection = False
        for edge1, edge2 in itertools.combinations(active_arcs, 2):
            if segments_intersect(posiciones[edge1[0]], posiciones[edge1[1]], 
                                  posiciones[edge2[0]], posiciones[edge2[1]]):
                has_intersection = True
                break
                
        if has_intersection:
            continue 
            
        # Desigualdad principal
        sum_l = sum(d[v] for v in vars_left)
        sum_r = sum(d[v] for v in vars_right)
        
        if sum_r == 0:
            continue

        if sum_l - constant >= sum_r:
            valid.append((d, sum_l, sum_r))
            
    # Ordenar maximizando el lado derecho y luego el izquierdo
    valid.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
    return valid

# --- 5. VISUALIZACIÓN ---
def mostrar_casos_secuencialmente(casos, posiciones):
    print(f"Se encontraron {len(casos)} casos válidos (sin cruces ni triviales).")
    print("MIRA LA VENTANA EMERGENTE. Cierra la ventana del gráfico (X) para ver el siguiente caso.")
    print("-" * 50)
    
    for i, (case_dict, s_l, s_r) in enumerate(casos, 1):
        # Separar arcos
        active_edges = [a for a in all_arcs if case_dict[get_var_name(a)] == 1]
        inactive_edges = [a for a in all_arcs if case_dict[get_var_name(a)] == 0]

        # 1. Filtramos las tuplas activas de cada lado usando arcs_left y arcs_right
        activas_izq = [get_var_name(a) for a in active_edges if a in arcs_left]
        activas_der = [get_var_name(a) for a in active_edges if a in arcs_right]
        
        # 2. Las unimos con un " + " para que parezca una ecuación. Si está vacío, ponemos "0"
        str_izq = " + ".join(activas_izq) if activas_izq else "0"
        str_der = " + ".join(activas_der) if activas_der else "0"
        
        print(f"Mostrando caso {i} de {len(casos)}... (Lado Izq:[{str_izq}] {s_l} >= Lado Der:[{str_der}] {s_r})")
        
        G = nx.DiGraph()
        G.add_nodes_from(posiciones.keys())

        plt.figure(figsize=(9, 6))
        
        # Dibujar nodos
        nx.draw_networkx_nodes(G, posiciones, node_size=600, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(G, posiciones, font_size=11, font_weight='bold')


        
        # Dibujar arcos
        nx.draw_networkx_edges(G, posiciones, edgelist=active_edges, edge_color='green', width=3, arrowsize=20)
        nx.draw_networkx_edges(G, posiciones, edgelist=inactive_edges, edge_color='grey', width=1, style='dashed', alpha=0.3)
        

        # 3. Armamos el título de forma limpia
        plt.title(f"Caso {i}/{len(casos)} | Ordenado por MAX Lado Der\n"
                  f"[{str_izq}] ({s_l})  >=  [{str_der}] ({s_r})", 
                  fontsize=12)
        plt.axis('off')
        
        plt.show()

# --- EJECUCIÓN PRINCIPAL ---
posiciones = load_instance('instance/euro-night-0000010.instance')
casos_validos = get_valid_cases(arcs_left, arcs_right, posiciones)
mostrar_casos_secuencialmente(casos_validos, posiciones)
print("Análisis terminado.")