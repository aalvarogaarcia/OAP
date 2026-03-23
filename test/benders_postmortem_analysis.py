import os
import traceback
from pathlib import Path

# Ajusta el import según cómo tengas expuesto tu módulo
from models.benders.optimize import optimize_master_LP, optimize_master_MILP
from utils.analyze_benders import generate_post_mortem_report
from utils.utils import read_indexed_instance

# =============================================================================
# LÓGICA PRINCIPAL
# =============================================================================

def process_single_instance(instance_path: str | Path, time_limit: int = 300, LP: bool = False) -> bool:
    """
    Ejecuta el workflow completo de Benders para una sola instancia,
    evaluando ambos casos: CON y SIN restricciones globales (sum_constrain).
    Genera dos reportes PDF separados en la misma carpeta.
    """
    instance_str = str(instance_path)
    instance_name = Path(instance_path).stem 
    
    print("=" * 60)
    print(f"🚀 INICIANDO INSTANCIA: {instance_name}")
    print("=" * 60)

    # DIRECTORIO BASE MÁS CORTO Y LIMPIO
    if LP:
        base_dir = f"Outputs/Benders/LP-{instance_name}"
    else:
        base_dir = f"Outputs/Benders/MILP-{instance_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Extraer los puntos reales de la instancia para pasárselos al graficador
    try:
        points = read_indexed_instance(instance_path)
    except Exception as e:
        print(f"⚠️ No se pudieron cargar los puntos para {instance_name}: {e}")
        points = None

    success_overall = True

    # Bucle para evaluar ambos casos
    for sum_constrain_flag in [True, False]:
        suffix = "con_techo" if sum_constrain_flag else "sin_techo"
        print(f"\n--- Evaluando {instance_name} | sum_constrain = {sum_constrain_flag} ({suffix}) ---")
        
        # Rutas de los archivos de salida cortas
        if LP:
            log_filepath = f"{base_dir}/farkas_log_LP_{suffix}.jsonl"
            pdf_filepath = f"{base_dir}/Farkas_Analysis_Report_LP_{suffix}.pdf"
        else:
            log_filepath = f"{base_dir}/farkas_log_MILP_{suffix}.jsonl"
            pdf_filepath = f"{base_dir}/Farkas_Analysis_Report_MILP_{suffix}.pdf"

        # Limpiar el PDF anterior si existe
        if os.path.exists(pdf_filepath): 
            os.remove(pdf_filepath)

        try:
            if LP:
                print("⚠️ Modo LP activado: Solo se resolverá la relajación y se generará el PDF. No se resolverá el IP.")
                # 1. Resolver la relajación LP (esto genera el .jsonl en base_dir)
                optimize_master_LP(
                    instance_str, 
                    verbose=False, 
                    plot=False, 
                    maximize=True, 
                    time_limit=time_limit, 
                    save_cuts=True, 
                    crosses_constrain=False,
                    benders_method="farkas",             
                    sum_constrain=sum_constrain_flag     
                )
            else:
                print("⚠️ Modo IP activado: Se resolverá el modelo completo y se generará el PDF con los datos del IP.")
                # 1. Resolver el modelo completo (esto genera el .jsonl en base_dir)
                optimize_master_MILP(
                    instance_str, 
                    verbose=False, 
                    plot=False, 
                    maximize=True, 
                    time_limit=time_limit, 
                    save_cuts=True, 
                    crosses_constrain=False,
                    benders_method="farkas",
                    sum_constrain=sum_constrain_flag     
                )


            # 2. Generar el reporte Post-Mortem en PDF leyendo el .jsonl
            if os.path.exists(log_filepath):
                print(f"Generando reporte PDF: {pdf_filepath} ...")
                generate_post_mortem_report(
                    log_filepath=log_filepath, 
                    output_pdf_path=pdf_filepath, 
                    points=points
                )
            else:
                print(f"⚠️ No se encontró el archivo de log en {log_filepath}. Saltando PDF.")

        except Exception as e:
            print(f"❌ Error procesando {instance_name} ({suffix}):\n{e}")
            traceback.print_exc()
            success_overall = False

    return success_overall

# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================
if __name__ == "__main__":
    instances_dir = "instance/little-instances" 
    extension = "*.instance"  
    time_limit = 300

    directory_path = Path(instances_dir)
    instances = list(directory_path.glob(extension))

    if not instances:
        print(f"⚠️ No se encontraron archivos '{extension}' en el directorio: {directory_path}")
    else:
        instances.sort()
        total = len(instances)
        exitosas = 0
        fallidas = []

        print(f"\n📁 Se encontraron {total} instancias para procesar.\n")

        for i, instance_path in enumerate(instances, 1):
            print(f"\n--- Procesando Instancia {i}/{total} ---")
            success = process_single_instance(instance_path, time_limit, LP=True)  # Cambia a True para solo generar PDF de la relajación
            
            if success:
                exitosas += 1
            else:
                fallidas.append(instance_path)

        print("\n" + "=" * 60)
        print("📊 RESUMEN DE EJECUCIÓN")
        print("=" * 60)
        print(f"Total procesadas: {total}")
        print(f"Exitosas (ambos reportes): {exitosas}")
        print(f"Fallidas (al menos un error): {len(fallidas)}")