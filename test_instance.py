from utils.model_stats import *
from utils.utils import *
from models import build_and_solve_model
import glob
import os
import time
import networkx as nx


instance = "instance/uniform-0000035-2.instance"

write_prefile(instance)


model =build_and_solve_model(instance, verbose=True, plot=True, maximize=False, time_limit=400,
                              sum_constrain=True, obj = 2, mode=0)


lp, gap, ip, elapsed_time, nodes = get_model_stats(model)

print("-" * 30)
print("--- Valores del modelo matemático ---")
print("-" * 30)
print(f"Columnas modelo original: {model.NumVars}")
print(f"Filas modelo original: {model.NumConstrs}")
print(f"Area de la envolvente convexa: {model._convex_hull_area}")


print("-" * 30)
print("--- Valores del modelo IP y Relajado ---")
print("-" * 30)

print(f"Instance: {instance}")
print(f"IP Objective Value: {ip:.2f}")
print(f"LP Objective Value: {lp:.2f}")
print(f"Optimality Gap: {gap:.2f}%")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print(f"Number of Nodes Explored: {nodes}")

print("-" * 30)

print("-" * 30)

print(f"Resultados: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}")