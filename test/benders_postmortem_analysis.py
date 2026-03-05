import os
import glob
import traceback
from pathlib import Path

from models.benders import build_master_problem
from utils.analyze_benders import load_farkas_logs, generate_post_mortem_report
from utils.model_stats import get_model_stats, get_tour
# =============================================================================
# LÓGICA PRINCIPAL
# =============================================================================

def process_single_instance(instance_path: str | Path, time_limit: int = 300) -> bool:
    """
    Ejecuta el workflow completo de Benders para una sola instancia.
    Retorna True si fue exitoso, False si hubo algún error.
    """
    instance_str = str(instance_path)
    # Usamos pathlib para extraer el nombre base sin extensión (ej: stars-0000015)
    instance_name = Path(instance_path).stem 
    
    print("=" * 60)
    print(f"🚀 INICIANDO INSTANCIA: {instance_name}")
    print("=" * 60)

    try:
        # 1. Construir modelo
        model = build_master_problem(
            instance_str, 
            verbose=False, 
            plot=False, 
            maximize=True, 
            time_limit=time_limit, 
            save_cuts=True, 
            crosses_constrain=False
        )

        # 2. Extraer estadísticas
        lp, gap, ip, elapsed_time, nodes = get_model_stats(model)

        print("-" * 30)
        print("--- Valores del modelo matemático (Benders) ---")
        print("-" * 30)
        print(f"Columnas modelo original: {model.NumVars}")
        print(f"Filas modelo original: {model.NumConstrs}")
        print(f"Area de la envolvente convexa: {model._convex_hull_area}")

        print("-" * 30)
        print("--- Valores del modelo IP y Relajado ---")
        print("-" * 30)
        print(f"Instance: {instance_str}")
        print(f"IP Objective Value: {ip:.2f}")
        print(f"LP Objective Value: {lp:.2f}")
        print(f"Optimality Gap: {gap:.2f}%")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Number of Nodes Explored: {nodes}")

        print("-" * 30)
        print("--- Tour obtenido ---")
        tour = get_tour(model)
        print(tour)
        print("-" * 30)

        print(f"Resumen: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}")
        
        # 3. Análisis Farkas
        farkas_log = load_farkas_logs(model._farkas_log_path)

        # Definir la ruta de salida
        if "crosses" in model._farkas_log_path.lower():
            out_dir = Path(f"outputs/Others/Benders/{instance_name}-Crosses")
        else:
            out_dir = Path(f"outputs/Others/Benders/{instance_name}")

        # IMPORTANTE: Crear los directorios si no existen para evitar FileNotFoundError
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = out_dir / "Farkas_Analysis_Report.pdf"

        # 4. Generar reporte
        generate_post_mortem_report(
            model._farkas_log_path, 
            str(pdf_path), 
            points=model._points_,
            n=len(model._points_)
        )
        
        print(f"✅ Instancia {instance_name} completada exitosamente. Reporte: {pdf_path}\n")
        return True

    except Exception as e:
        print(f"❌ ERROR procesando la instancia {instance_name}:")
        print(traceback.format_exc()) # Imprime el error detallado para debuggear
        return False


def process_directory(directory_path: str, extension: str = "*.instance", time_limit: int = 300):
    """
    Busca todos los archivos con la extensión dada en un directorio y los procesa uno por uno.
    """
    # Buscar todos los archivos que coincidan con la extensión
    search_pattern = os.path.join(directory_path, extension)
    instances = glob.glob(search_pattern)
    
    if not instances:
        print(f"⚠️ No se encontraron archivos '{extension}' en el directorio: {directory_path}")
        return

    # Ordenar alfabéticamente para tener una ejecución predecible
    instances.sort()
    
    total = len(instances)
    exitosas = 0
    fallidas = []

    print(f"\n📁 Se encontraron {total} instancias para procesar.\n")

    for i, instance_path in enumerate(instances, 1):
        print(f"--- Procesando {i}/{total} ---")
        success = process_single_instance(instance_path, time_limit)
        
        if success:
            exitosas += 1
        else:
            fallidas.append(instance_path)

    # Resumen final
    print("=" * 60)
    print("📊 RESUMEN DE EJECUCIÓN")
    print("=" * 60)
    print(f"Total procesadas: {total}")
    print(f"Exitosas: {exitosas}")
    print(f"Fallidas: {len(fallidas)}")
    
    if fallidas:
        print("\nLas siguientes instancias fallaron:")
        for f in fallidas:
            print(f"  - {f}")


# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================
if __name__ == "__main__":
    # Cambia esto a la ruta real de tu carpeta de instancias
    TARGET_DIRECTORY = "instance/little-instances" 
    MAX_TIME_LIMIT = 300
    
    process_directory(
        directory_path=TARGET_DIRECTORY, 
        extension="*.instance", 
        time_limit=MAX_TIME_LIMIT
    )