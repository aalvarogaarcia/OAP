import os
import csv
import gurobipy as gp
from gurobipy import GRB  # Importamos GRB para evaluar los estados del modelo

# Asegúrate de importar tus funciones correctamente
from models.benders import build_master_problem
from models.gurobi import build_and_solve_model
from utils.model_stats import get_ObjVal_int

def cargar_resultados_esperados(csv_path):
    """Lee el CSV y devuelve un diccionario {instancia: (ip_min, ip_max)}"""
    resultados = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre = row['instance'].strip()
            # Convertir a entero o guardar como None si está vacío
            ip_min = int(row['ip_min']) if row['ip_min'] else None
            ip_max = int(row['ip_max']) if row['ip_max'] else None
            resultados[nombre] = (ip_min, ip_max)
    return resultados

def evaluar_modelo(instancia, obj_gur, obj_ben, expected, mod_gur, mod_ben, test_type):
    """Evalúa los resultados de ambos modelos y devuelve un diccionario con el registro."""
    status = "Desconocido"
    
    # Comprobar si alguno de los dos modelos alcanzó el límite de tiempo
    timeout_gur = (mod_gur.Status == GRB.TIME_LIMIT)
    timeout_ben = (mod_ben.Status == GRB.TIME_LIMIT)

    if timeout_gur or timeout_ben:
        status = "Timeout (Falta comprobar)"
        print(f"     ⏳ {test_type}: Límite de tiempo alcanzado. Registrado para comprobar luego.")
    # Si no hay timeout, comprobamos discrepancias
    elif obj_gur != obj_ben:
        status = f"Error: Gurobi != Benders"
        print(f"     ❌ {test_type} FALLO: Gurobi ({obj_gur}) != Benders ({obj_ben})")
    elif expected is not None and obj_gur is not None and int(obj_gur) != expected:
        status = f"Error: Calculado != PDF"
        print(f"     ❌ {test_type} FALLO: Calculado ({obj_gur}) != PDF ({expected})")
    else:
        status = "✅ OK"
        print(f"     ✅ {test_type}: Gurobi y Benders coinciden con el PDF.")

    return {
        "Instance": instancia,
        "Type": test_type,
        "Gurobi_IP": obj_gur,
        "Benders_IP": obj_ben,
        "Expected": expected,
        "Status": status
    }

def test_instancia(instance_path, expected_min, expected_max):
    """Ejecuta los modelos para una instancia y devuelve sus resultados."""
    nombre_instancia = os.path.basename(instance_path)
    print(f"\n[{nombre_instancia}] Iniciando pruebas...")
    
    resultados_instancia = []

    # --- PRUEBA MINAREA (maximize = False) ---
    if expected_min is not None:
        print("  -> Comprobando MINAREA ...")
        mod_gur_min = build_and_solve_model(instance_path, verbose=False, maximize=False, time_limit=300, obj=1, subtour=2)
        ip_gur_min = get_ObjVal_int(mod_gur_min)
        
        mod_ben_min = build_master_problem(instance_path, verbose=False, maximize=False, time_limit=300)
        ip_ben_min = get_ObjVal_int(mod_ben_min)
        
        res_min = evaluar_modelo(nombre_instancia, ip_gur_min, ip_ben_min, expected_min, mod_gur_min, mod_ben_min, "MINAREA")
        resultados_instancia.append(res_min)

    # --- PRUEBA MAXAREA (maximize = True) ---
    if expected_max is not None:
        print("  -> Comprobando MAXAREA ...")
        mod_gur_max = build_and_solve_model(instance_path, verbose=False, maximize=True, time_limit=300, obj=2, subtour=2)
        ip_gur_max = get_ObjVal_int(mod_gur_max)
        
        mod_ben_max = build_master_problem(instance_path, verbose=False, maximize=True, time_limit=300)
        ip_ben_max = get_ObjVal_int(mod_ben_max)
        
        res_max = evaluar_modelo(nombre_instancia, ip_gur_max, ip_ben_max, expected_max, mod_gur_max, mod_ben_max, "MAXAREA")
        resultados_instancia.append(res_max)

    return resultados_instancia

def guardar_resultados_csv(resultados, output_csv):
    """Guarda la lista de resultados en un archivo CSV."""
    if not resultados:
        return
    
    claves = ["Instance", "Type", "Gurobi_IP", "Benders_IP", "Expected", "Status"]
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=claves)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n📁 Resultados guardados exitosamente en: {output_csv}")

def ejecutar_bateria_tests(folder_path, csv_path, output_csv_path):
    """Recorre la carpeta, evalúa las instancias y exporta resultados."""
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo CSV base en {csv_path}")
        return
        
    resultados_esperados = cargar_resultados_esperados(csv_path)
    archivos_instancia = [f for f in os.listdir(folder_path) if f.endswith('.txt') or f.endswith('.instance')]
    
    if not archivos_instancia:
        print(f"No se encontraron instancias en la carpeta {folder_path}")
        return

    print(f"Se encontraron {len(archivos_instancia)} instancias en la carpeta. Comenzando evaluación...")
    
    todos_los_resultados = []
    instancias_evaluadas = 0

    for archivo in archivos_instancia:
        nombre_instancia = archivo.replace('.instance', '').replace('.txt', '')
        
        if nombre_instancia in resultados_esperados:
            ruta_completa = os.path.join(folder_path, archivo)
            ip_min, ip_max = resultados_esperados[nombre_instancia]
            
            try:
                resultados = test_instancia(ruta_completa, ip_min, ip_max)
                todos_los_resultados.extend(resultados)
                instancias_evaluadas += 1
            except Exception as e:
                print(f"     ⚠️ ERROR CRÍTICO de ejecución en {nombre_instancia}: {str(e)}")
                todos_los_resultados.append({
                    "Instance": archivo,
                    "Type": "ERROR",
                    "Gurobi_IP": "", "Benders_IP": "", "Expected": "",
                    "Status": f"Exception: {str(e)}"
                })
        else:
            print(f"[-] Saltando '{nombre_instancia}': No está en el CSV de resultados esperados.")
            
    print(f"\n--- Batería completada: {instancias_evaluadas} instancias procesadas ---")
    
    # Guardar todos los resultados de la batería en CSV
    guardar_resultados_csv(todos_los_resultados, output_csv_path)

if __name__ == "__main__":
    # Configura aquí tus rutas
    CARPETA_INSTANCIAS = "./instance/"  # Asegúrate de que esta carpeta contenga tus archivos de instancia
    ARCHIVO_CSV_ENTRADA = "test/resultados_esperados.csv"
    ARCHIVO_CSV_SALIDA = "outputs/reporte_tests.csv"  # El nuevo archivo donde se guardarán los logs
    
    ejecutar_bateria_tests(CARPETA_INSTANCIAS, ARCHIVO_CSV_ENTRADA, ARCHIVO_CSV_SALIDA)