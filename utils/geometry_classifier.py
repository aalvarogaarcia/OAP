from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from utils.utils import read_indexed_instance

def compute_onion_layers(points):
    node_to_layer = {}
    remaining_indices = np.arange(len(points))
    current_layer = 0

    while len(remaining_indices) >= 3:  
        hull = ConvexHull(points[remaining_indices])

        layer_original_indices = remaining_indices[hull.vertices]
        
        for node in layer_original_indices:
            node_to_layer[int(node)] = current_layer 
    
        
        remaining_indices = np.delete(remaining_indices, hull.vertices)
        current_layer += 1
    
    if len(remaining_indices) > 0:
        for node in remaining_indices:
            node_to_layer[int(node)] = current_layer

    return node_to_layer

    

def compute_delauney(points):
    # 1. Extraer IDs y coordenadas manteniendo el orden
    
    node_ids = range(len(points))
    
    # 2. Calcular la triangulación de Delaunay
    tri = Delaunay(points)
    
    delaunay_edges = set()
    
    # 3. Extraer las aristas de cada triángulo (simplex)
    for simplex in tri.simplices:
        n0 = node_ids[simplex[0]]
        n1 = node_ids[simplex[1]]
        n2 = node_ids[simplex[2]]
        
        # Añadir las 3 aristas del triángulo (en ambas direcciones para facilitar búsquedas)
        edges = [
            (n0, n1), (n1, n0),
            (n1, n2), (n2, n1),
            (n2, n0), (n0, n2)
        ]
        
        delaunay_edges.update(edges)
        
    return delaunay_edges



points = read_indexed_instance("instance/Instancias/euro-night-0000010.instance")

edges = compute_delauney(points)
print(edges)