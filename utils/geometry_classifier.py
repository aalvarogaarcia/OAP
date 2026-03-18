from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from utils.utils import read_indexed_instance


def compute_oninon_layers(points):
    layers = []
    remaining_indices = np.arange(len(points))
    
    while len(remaining_indices) >= 3:  # ConvexHull requires at least 3 points
        hull = ConvexHull(points[remaining_indices])

        layer_original_indices = remaining_indices[hull.vertices]
        layers.append(layer_original_indices)

        remaining_indices = np.delete(remaining_indices, hull.vertices)  # Eliminar los puntos de la capa actual
    
    if len(remaining_indices) > 0:
        layers.append(remaining_indices)  # Agregar los puntos restantes como la última capa

    return layers

#points = read_indexed_instance("instance/euro-night-0000035.instance")
#print(points)
#levels = compute_oninon_layers(points)
#
#print(levels)