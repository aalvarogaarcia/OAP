from src import *
import glob
import os
import time

points = np.array([[0, 0], [1, 2], [2, 1], [3, 3], [4, 0], [2, 4], [0, 4], [4, 4]])
CH = compute_convex_hull(points)
V = compute_triangles(points)

insertion_heuristic(points, CH, V, sense=True)