from main import get_model_stats
from src import *
import glob
import os
import time

instance = "instance/euro-night-0000015.instance"
model = build_and_solve_model(instance, verbose=True, plot=False, maximize=True, time_limit=600, sum_constrain=True, benders=True)
model_lp = model.relax()
model_lp.optimize()
lp, gap, ip, elapsed_time, nodes = get_model_stats(model, model_lp)
print(f"Resultados: LP={lp}, Gap={gap}%, IP={ip}, Time={elapsed_time}s, Nodes={nodes}")