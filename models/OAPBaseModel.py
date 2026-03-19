import gurobipy as gp
from numpy.typing import NDArray
import numpy as np
from utils.utils import compute_convex_hull, compute_convex_hull_area, triangles_adjacency_list
from models.mixin.oap_stats_mixin import OAPStatsMixin
from models.typing_oap import NumericArray

class OAPBaseModel(OAPStatsMixin):
    def __init__(self, points: NumericArray, triangles: NDArray[np.int64], name: str):
        # Todo lo que es común a ambos modelos
        self.points = points
        self.triangles = triangles
        self.triangles_adj_list = triangles_adjacency_list(triangles, points)
        self.N_list = range(len(points))
        self.N = len(points)
        self.CH = compute_convex_hull(points)
        self.V_list = range(len(triangles))
        self.convex_hull_area = compute_convex_hull_area(points)
        
        # El modelo principal (Para Compacto es el modelo entero, para Benders es el Master)
        self.name = name
        self.model = gp.Model(name)