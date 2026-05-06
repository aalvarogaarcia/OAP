import logging
from pathlib import Path

import gurobipy as gp
import pandas as pd
import pytest

# Ajusta los imports según la estructura real de tus carpetas
from models.OAPCompactModel import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
tsv_path = BASE_DIR / "test" / "data" / "TablaResultadosA4.tsv"
Path("outputs/Others/Testing").mkdir(parents=True, exist_ok=True)
# 1. Leer el archivo TSV y convertirlo en una lista de diccionarios
try:
    df_resultados = pd.read_csv(tsv_path, sep='\t')
    _all = df_resultados.to_dict('records')
    instancias_test = [r for r in _all if any(f"{n:07d}" in str(r['instance']) for n in [10, 15, 20, 25])]
except FileNotFoundError:
    logger.warning(f"ATENCIÓN: No se encontró el TSV en {tsv_path}")
    instancias_test = []




# 2. Inyectar los datos en la función de prueba
# 'ids' hace que en la consola veamos el nombre de la instancia en lugar de "test_1", "test_2"
@pytest.mark.parametrize("ref_data", instancias_test, ids=[str(r['instance']) for r in instancias_test])
def test_oap_compact_model_minimize(ref_data, caplog):
    """
    Test que comprueba si el modelo compacto genera la estructura y 
    los resultados óptimos correctos para el caso de Minimización.
    """
    caplog.set_level(logging.INFO)
    instance_name = ref_data['instance']
    
    # Construir la ruta real (Asegúrate de que apunta a tu carpeta de instancias)
    instance_path = BASE_DIR / "instance" / f"{instance_name}.instance"
    
    logger.info(f"Ruta base del proyecto: {BASE_DIR}")
    logger.info(f"Ruta del TSV: {tsv_path}")
    logger.info(f"Ruta de la instancia: {instance_path}")

    if not instance_path.exists():
        logger.error(f"❌ El archivo no existe en la ruta especificada: {instance_path}")
        pytest.skip(f"Archivo de instancia no encontrado localmente: {instance_path}")

    # --- SETUP (Preparar) ---
    puntos = read_indexed_instance(str(instance_path))
    triangulos = compute_triangles(puntos)

    # --- ACT (Actuar) ---
    modelo = OAPCompactModel(puntos, triangulos, name=instance_name)
    modelo.build(objective="External", mode=0, maximize=False, subtour="SCF", sum_constrain=True) 
    
    # Para los tests de CI/CD solemos poner un límite de tiempo más estricto.
    # Capturamos GurobiError de licencia (size-limited) y hacemos skip.
    try:
        modelo.solve(time_limit=300, verbose=False)
    except gp.GurobiError as exc:
        if "size-limited" in str(exc).lower() or "too large" in str(exc).lower():
            pytest.skip(
                f"Instance '{instance_name}' too large for Gurobi size-limited license: {exc}"
            )
        raise


    # --- ASSERT (Verificar) ---
    
    # 1. Verificar la estructura matemática (Filas y Columnas)
    #assert modelo.model.NumVars == ref_data['MILP_col'], f"Discrepancia en columnas para {instance_name} ({modelo.model.NumVars} vs {ref_data['MILP_col']})"
    #assert modelo.model.NumConstrs == ref_data['MILP_row'], f"Discrepancia en filas para {instance_name} ({modelo.model.NumConstrs} vs {ref_data['MILP_row']})"

    # 2. Extraer métricas de tu Mixin
    lp_val, gap, ip_val, time_s, nodes = modelo.get_model_stats()

     # 2. Verificar el valor óptimo entero (IP Value)
    if ip_val != "-":
        assert ip_val == pytest.approx(ref_data['MIN_IPvalue'], rel=1e-3), \
            f"ERROR GRAVE: El valor objetivo (IP) para {instance_name} no coincide. IP Calculado: {ip_val:.4f} vs TSV: {ref_data['MIN_IPvalue']}"
    
    # 3. Verificar el valor de la relajación lineal (LP Value)
    # Para minimización, LP es cota inferior: solo falla si el calculado es MAYOR que la referencia.
    if lp_val != "-":
        assert lp_val <= ref_data['MIN_LPvalue'] * (1 + 1e-4), \
            f"ERROR: La relajación lineal (LP) para {instance_name} es peor que la referencia. LP Calculado: {lp_val:.4f} vs TSV: {ref_data['MIN_LPvalue']}"

    # 4. (Opcional) Verificar explícitamente el Gap
    # Para minimización, un gap menor o igual que el de referencia (±0.5 pp) es aceptable.
    if gap != "-":
        assert gap <= ref_data['MIN_LPgap'] + 0.5, \
            f"AVISO: El Gap de optimalidad para {instance_name} es peor que la referencia. Gap Calculado: {gap:.2f}% vs TSV: {ref_data['MIN_LPgap']}%"


if __name__ == "__main__":
    pytest.main(["-v", "test_instance.py"])