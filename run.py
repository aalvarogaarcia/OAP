from models import OAPBendersModel, OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')


if __name__ == "__main__":
    # Aquí puedes cambiar el nombre de la instancia que quieres ejecutar
    instance_name = "euro-night-0000030"
    
    # 1. Preparar datos
    points = read_indexed_instance(f"instance/{instance_name}.instance")
    triangles = compute_triangles(points)
    
    # 2. Instanciar el modelo
    compact = OAPCompactModel(points, triangles, name=instance_name)
    compact.build(objective="Internal", maximize=True, subtour="SCF",semiplane=2, use_knapsack=True, use_cliques=False)
    
    # 3. Resolver (Obligatorio save_cuts=True para el análisis)
    # El mixin se encargará de guardar el JSON en outputs/Logs/benders_{name}.json
    compact.solve(time_limit=300, verbose=True)

    lp_val, gap, ip_val, time_s, nodes = compact.get_model_stats()
    print("\n"+"#"*50)
    print("#"*10 + "\tRESULTADOS FINALES\t" + "#"*10)
    print("#"*50 +"\n")
    print(f"✅ Ejecución de {instance_name} completada. Revisa los logs para detalles.")
    print(f"📊 Estadísticas del modelo:")
    print(f"   - Valor de la función objetivo: {lp_val:.2f}")
    print(f"   - Gap: {gap:.2f}%")
    print(f"   - Valor de la solución entera: {int(ip_val):.0f}")
    print(f"   - Tiempo de resolución: {time_s:.2f} segundos")
    print(f"   - Nodos explorados: {int(nodes)}\n")