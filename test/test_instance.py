from models.benders import optimize_master_MILP
from models.gurobi import build_and_solve_model
from utils.model_stats import *
from utils.utils import *
from utils.farkas_viz import *
from utils.analyze_benders import generate_post_mortem_report


instance = "instance/london-0000010.instance"


model_compact = build_and_solve_model(instance, verbose=False, plot=True, maximize=True, time_limit=300,
                                sum_constrain=True, obj = 1, mode = 0, subtour = 0)


#model = optimize_master_MILP(instance, verbose=True, plot=True, maximize=True, time_limit=300, 
#                             save_cuts=False, crosses_constrain=False)
#lp, gap, ip, elapsed_time, nodes = get_model_stats(model)
#
#print("-" * 30)
#print("--- Valores del modelo matemático (Benders) ---")
#print("-" * 30)
#print(f"Columnas modelo original: {model.NumVars}")
#print(f"Filas modelo original: {model.NumConstrs}")
#print(f"Area de la envolvente convexa: {model._convex_hull_area}")
#
#
#print("-" * 30)
#print("--- Valores del modelo IP y Relajado ---")
#print("-" * 30)
#
#print(f"Instance: {instance}")
#print(f"IP Objective Value: {ip:.2f}")
#print(f"LP Objective Value: {lp:.2f}")
#print(f"Optimality Gap: {gap:.2f}%")
#print(f"Elapsed Time: {elapsed_time:.2f} seconds")
#print(f"Number of Nodes Explored: {nodes}")
#
#print("-" * 30)
#print("--- Tour obtenido ---")
#tour = get_tour(model)
#print(tour)
#print("-" * 30)
#
#print(f"Resultados: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}")#


#farkas_log = load_farkas_logs(model._farkas_log_path)

#if "crosses" in model._farkas_log_path.lower():
#    PDF_PATH = f"outputs/Others/Benders/{instance.split('/')[-1].split('.')[0]}-Crosses/Farkas_Analysis_Report.pdf"
#else:
#    PDF_PATH = f"outputs/Others/Benders/{instance.split('/')[-1].split('.')[0]}/Farkas_Analysis_Report.pdf"
#
#generate_post_mortem_report(model._farkas_log_path, 
#                            PDF_PATH, 
#                            points=model._points_,
#                            n=len(model._points_))
#



lp, gap, ip, elapsed_time, nodes = get_model_stats(model_compact)

print("-" * 30)
print("--- Valores del modelo matemático compacto ---")
print("-" * 30)
print(f"Columnas modelo original: {model_compact.NumVars}")
print(f"Filas modelo original: {model_compact.NumConstrs}")
print(f"Area de la envolvente convexa: {model_compact._convex_hull_area}")


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
print("--- Tour obtenido ---")
tour = get_tour(model_compact)
print(tour)
print("-" * 30)

print(f"Resultados: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}")