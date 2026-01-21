from main import get_model_stats, get_model_stats_cplex
from src import *
import glob
import os
import time

instance = "instance/euro-night-0000015.instance"
model = build_and_solve_model_cplex(instance, verbose=True, plot=False, maximize=False, time_limit=600, sum_constrain=True, benders=False)
#model.write("Minimal_Proof.lp")
model_lp = model.clone()
model_lp.name = f"{model.name}_LP"
model_lp.change_var_types(model_lp.iter_binary_vars(), 'C')
model_lp.solve()
lp, gap, ip, elapsed_time, nodes = get_model_stats_cplex(model, model_lp)
print(f"Resultados: LP={lp}, Gap={gap}%, IP={ip}, Time={elapsed_time}s, Nodes={nodes}")