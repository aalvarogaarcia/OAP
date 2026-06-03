import logging
import os

from gurobipy import GRB

from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance, write_prefile
from utils.constraints import noncrossing_bipartition

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")


if __name__ == "__main__":
    # Aquí puedes cambiar el nombre de la instancia que quieres ejecutar
    instance_name = "euro-night-0000010"

    # 1. Preparar datos
    points = read_indexed_instance(f"instance/{instance_name}.instance")
    triangles = compute_triangles(points)
    #write_prefile(f"instance/{instance_name}.instance")
    
    # 2. Instanciar el modelo
    compact = OAPCompactModel(points, triangles, name=instance_name)
    compact.build(
        objective="Internal", maximize=True, subtour="MCF", use_bipartition=True, sum_constrain=True
    )
    
    # Instanciar el modelo de Benders (si quieres probarlo también)
    # benders = OAPBendersModel(points, triangles, name=instance_name)
    # benders.build(objective="Fekete", maximize=True,
    #               #subtour="SCF"
    #              )

    # 3. Resolver (Obligatorio save_cuts=True para el análisis)
    # El mixin se encargará de guardar el JSON en outputs/Logs/benders_{name}.json
    compact.solve(time_limit=300, verbose=True)

    # Si el modelo acaba infactible, exporta el IIS para depuración.
    status = compact.model.Status
    if status == GRB.INF_OR_UNBD:
        compact.model.setParam("DualReductions", 0)
        compact.model.optimize()
        status = compact.model.Status

    if status == GRB.INFEASIBLE:
        os.makedirs("outputs/IIS", exist_ok=True)
        iis_path = f"outputs/IIS/{instance_name}_compact.ilp"
        compact.model.computeIIS()
        compact.model.write(iis_path)
        print(f"IIS exportado en: {iis_path}")

    # benders.solve(time_limit=300, verbose=False, save_cuts=True)

    print(compact)
    # print(benders)
