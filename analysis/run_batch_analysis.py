import argparse
import logging
import os
from pathlib import Path

# Ajusta las rutas de importación a tu proyecto
from models.OAPBendersModel import OAPBendersModel
from utils.utils import read_indexed_instance, compute_triangles
from typing import Literal
# Configuración básica del logger para la consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_batch(
    instances_dir: str, 
    filter_texts: list[str], 
    time_limit: int, 
    benders_method: Literal['farkas', 'pi'],
    maximize: bool,
    lp: bool
):
    """
    Busca instancias que coincidan con 'filter_texts', las resuelve con Benders
    y genera un reporte en PDF de su comportamiento dual.
    """
    directory_path = Path(instances_dir)
    
    if not directory_path.exists():
        logger.error(f"El directorio de instancias {instances_dir} no existe.")
        return

    # Buscar todas las instancias que contengan el texto del filtro
    todas_instancias = list(directory_path.glob("*.instance"))
    if not filter_texts:  # Si la lista está vacía (no se usó el flag)
        instancias_filtradas = todas_instancias
    else:
        # Quédate con el archivo si ALGUNO (any) de los textos del filtro está en su nombre
        instancias_filtradas = [
            p for p in todas_instancias 
            if any(texto in p.name for texto in filter_texts)
        ]
    if not instancias_filtradas:
        logger.warning(f"No se encontraron instancias que contengan '{filter_texts}' en {directory_path}.")
        return

    # Ordenar alfabéticamente/por tamaño
    instancias_filtradas.sort()
    total = len(instancias_filtradas)
    
    logger.info("=" * 60)
    logger.info(f"📁 Se procesarán {total} instancias (Filtro: '{filter_texts}')")
    logger.info("=" * 60)

    exitosas = 0
    errores = 0
    for i, instance_path in enumerate(instancias_filtradas, 1):
        instance_name = instance_path.stem
        logger.info(f"\n--- Procesando Instancia {i}/{total}: {instance_name} ---")
        
        try:
            # 1. Preparar datos
            points = read_indexed_instance(str(instance_path))
            triangles = compute_triangles(points)
            
            # 2. Instanciar el modelo
            log_path = f"outputs/Logs/benders_{instance_name}_MILP.json"

            benders = OAPBendersModel(points, triangles, name=instance_name)
            benders.build(
                objective="Fekete", 
                maximize=maximize, 
                benders_method=benders_method, 
                sum_constrain=True,
            )
            
            benders.set_log_path(log_path)  # Configura la ruta del log para este modelo

            # 3. Resolver (Obligatorio save_cuts=True para el análisis)
            # El mixin se encargará de guardar el JSON en outputs/Logs/benders_{name}.json
            benders.solve(time_limit=time_limit, verbose=False, save_cuts=True)
            
            # 4. Generar Reporte PDF
            pdf_path = f"outputs/Analysis/MILP/Report/{"MAX" if maximize else "MIN"}_{benders_method}_{instance_name}.pdf"

            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            benders.generate_benders_report(output_pdf_path=pdf_path)

            pdf = pdf_path.replace("Report", "Cut_Analysis")
            benders.generate_combinatorial_report(output_pdf=pdf, max_vars = 10)

            if lp:
                pdf_path = pdf_path.replace("MILP", "LP")
                
                # El mixin se encargará de guardar el JSON en outputs/Logs/benders_{name}.json
                benders.set_log_path(f"outputs/Logs/benders_{instance_name}_lp_relaxation.json")
                benders.solve(time_limit=time_limit, verbose=False, save_cuts=True, relaxed=True)
                
                # 4. Generar Reporte PDF
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                benders.generate_benders_report(output_pdf_path=pdf_path)
    
                pdf = pdf_path.replace("Report", "Cut_Analysis")
                benders.generate_combinatorial_report(output_pdf=pdf, max_vars = 10)
    
            exitosas += 1
            
        except Exception as e:
            logger.error(f"❌ Fallo crítico al procesar {instance_name}: {e}")
            errores += 1
            continue

    logger.info( "=" * 60)
    logger.info(f"📊 RESUMEN: {exitosas} de {total} reportes generados exitosamente.")
    logger.info(f"❌ ERRORES: {errores} instancias fallaron.")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutor en lote para Análisis de Benders.")
    
    # Argumentos de terminal
    parser.add_argument("-f", "--filter", nargs="+", default=[], 
                        help="Fragmento de texto para filtrar archivos (ej. 'us-night' o '0000010'). Si se omite, corre todas.")
    parser.add_argument("-d", "--dir", type=str, default="instance/little-instances", 
                        help="Directorio donde se encuentran las instancias.")
    parser.add_argument("-t", "--time", type=int, default=300, 
                        help="Límite de tiempo en segundos para cada ejecución (default: 300).")
    parser.add_argument("-m", "--method", type=str, choices=['farkas', 'pi'], default="farkas", 
                        help="Método de subproblema a usar (farkas o pi).")
    parser.add_argument("--maximize", action="store_true", 
                        help="Si se incluye este flag, el modelo maximizará. Por defecto minimiza.")
    parser.add_argument("--lp", action="store_true",
                        help="Si se incluye este flag, también se resolverá la versión relajada (LP) de cada instancia para análisis comparativo.")

    args = parser.parse_args()

    run_batch(
        instances_dir=args.dir,
        filter_texts=args.filter,
        time_limit=args.time,
        benders_method=args.method,
        maximize=args.maximize,
        lp = args.lp
    )