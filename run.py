import logging

from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')


if __name__ == "__main__":
    # Aquí puedes cambiar el nombre de la instancia que quieres ejecutar
    instance_name = "stars-0000030"
    
    # 1. Preparar datos
    points = read_indexed_instance(f"instance/{instance_name}.instance")
    triangles = compute_triangles(points)
    
    # 2. Instanciar el modelo
    compact = OAPCompactModel(points, triangles, name=instance_name)
    compact.build(objective="Internal", maximize=True, subtour="SCF",semiplane=2, use_knapsack=False, use_cliques=False)

    #Instanciar el modelo de Benders (si quieres probarlo también)
    #benders = OAPBendersModel(points, triangles, name=instance_name)
    #benders.build(objective="Fekete", maximize=True,
    #               #subtour="SCF"
    #              )
    

    # 3. Resolver (Obligatorio save_cuts=True para el análisis)
    # El mixin se encargará de guardar el JSON en outputs/Logs/benders_{name}.json
    compact.solve(time_limit=300, verbose=True)
    #benders.solve(time_limit=300, verbose=False, save_cuts=True)

    print(compact)
    #print(benders)



