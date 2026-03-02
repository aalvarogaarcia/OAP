from utils.model_stats import get_model_stats, get_model_stats_cplex, get_ObjVal_int
from utils.utils import *
from models import build_and_solve_model
import glob
import os
import time
import networkx as nx

"""
instance = "instance/euro-night-0000010.instance"
#model.write("Minimal_Proof.lp")
model =build_and_solve_model(instance, verbose=True, plot=True, maximize=False, time_limit=600, sum_constrain=True, benders=False)

model_lp = model.relax()

model_lp.optimize()

print("--- Estadísticas Reales ---")
print(f"Total variables modelo original: {model.number_of_variables}")
print(f"  > Binarias: {model.number_of_binary_variables}")
print(f"  > Continuas: {model.number_of_continuous_variables}")

print("-" * 30)

print(f"Total variables modelo LP (Relajado): {model_lp.number_of_variables}")
print(f"  > Binarias: {model_lp.number_of_binary_variables}  <-- ESTO DEBE SER 0")
print(f"  > Continuas: {model_lp.number_of_continuous_variables}")
model_lp.solve()
lp, gap, ip, elapsed_time, nodes = get_model_stats(model, model_lp)
print(f"Resultados: LP={lp}, Gap={gap}%, IP={ip}, Time={elapsed_time}s, Nodes={nodes}")
"""


if __name__ == "__main__":
        files = ['instance/stars-0000015.instance']
        for file in files:
            write_prefile(file)
            start_time = time.time()
            mod = build_and_solve_model(
                instance_path=file,
                verbose=True,
                plot=False,
                time_limit=7200000,
                maximize=True,
                sum_constrain=False
            )
            end_time = time.time()
            print(f"Processed {file} in {end_time - start_time:.2f} seconds.")
            print("--------------------------------------------------\n")
            print("Results Summary:\n")
            print(f"Instance: {file}")
            print(f"Objective Value: {get_ObjVal_int(mod)}")
            print(f"Status: {mod.Status}")
            print(f"Number of Variables: {mod.NumVars}")
            print(f"Number of Constraints: {mod.NumConstrs}")
            print("--------------------------------------------------\n")
            print(f"Obtain tour result: {mod._x_results}")
            G = nx.DiGraph()
            G.add_edges_from(mod._x_results)
            cycles = list(nx.simple_cycles(G))
            
            print(f"Obtain ordered tour: {cycles}")
            model_lp = mod.relax()
            model_lp.setParam('OutputFlag', 1 if False else 0)
            model_lp.optimize()
            outputfile = 'outputs/Others/' + file.split('/')[-1].split('.')[0]
            mod.write(outputfile + '.lp')
            print("--------------------------------------------------\n")
            print("LP Relaxation Results Summary:\n")
            print(f"LP Status: {model_lp.Status}")
            print(f"LP Relaxed Objective Value: {model_lp.ObjVal}")
            print("--------------------------------------------------\n")

            lp, gap, ip, elapsed_time, nodes = get_model_stats(mod, model_lp)
            print(f"Resultados: LP={lp}, Gap={gap}%, IP={ip}, Time={elapsed_time}s, Nodes={nodes}")
        