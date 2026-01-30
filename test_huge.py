from utils.model_stats import *
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
        fn =  'instance/uniform-0000020-1.instance'
        files = glob.glob(fn)
        outputfile = 'outputs/Excel/' + fn.replace('instance/', '').replace('*', '').replace('/', '_').replace('.', '_') + '_results.xlsx'
        for file in files:
            
            write_prefile(file)
            start_time = time.time()
            for maximize in [True, False]:
                print(f"=== Iteration {"maximizing" if maximize else "minimizing"} for instance {file} ===")
                for i in range(4):
                    print(f"=== Iteration {i+1} for instance {file} ===")
                    if i == 0:
                        for j in range(4):
                            print(f"--- Sub-iteration {j+1} ---")
                            mod = build_and_solve_model(
                                instance_path=file,
                                verbose=True,
                                plot=False,
                                time_limit=7200000,
                                maximize=maximize,
                                sum_constrain=True,
                                obj=i,
                                mode=j
                            )
                            mod_lp = mod.relax()
                            mod_lp.setParam('OutputFlag', 1 if False else 0)
                            mod_lp.optimize()
                            save_results_excel(mod, mod_lp, outputfile)

                    else:
                        mod = build_and_solve_model(
                            instance_path=file,
                            verbose=True,
                            plot=False,
                            time_limit=7200000,
                            maximize=maximize,
                            sum_constrain=True,
                            obj=i,
                            mode=0
                        )
                        mod_lp = mod.relax()
                        mod_lp.setParam('OutputFlag', 1 if False else 0)
                        mod_lp.optimize()
                        save_results_excel(mod, mod_lp, outputfile)

            print(f"Total time for instance {file}: {time.time() - start_time} seconds")

        