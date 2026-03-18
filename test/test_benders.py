import os
import csv
import gurobipy as gp
from gurobipy import GRB  # Importamos GRB para evaluar los estados del modelo

# Asegúrate de importar tus funciones correctamente
from models.benders import optimize_master_LP
from models.gurobi import build_and_solve_model

ExpectedLpGap = tuple[float | None, float | None]
ResultValue = str | float | None
ResultRow = dict[str, ResultValue]

def cargar_resultados_esperados(csv_path: str) -> dict[str, ExpectedLpGap]:
    """Lee el CSV y devuelve un diccionario {instancia: (ip_min, ip_max)}"""
    resultados: dict[str, ExpectedLpGap] = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre = row['instance'].strip()
            # Convertir a entero o guardar como None si está vacío
            ip_min = float(row['lpgap_min']) if row['lpgap_min'] else None
            ip_max = float(row['lpgap_max']) if row['lpgap_max'] else None
            resultados[nombre] = (ip_min, ip_max)
    return resultados

def evaluar_modelo(
    instancia: str,
    obj_gur: float | None,
    obj_ben: float | None,
    expected: float | None,
    mod_gur: gp.Model,
    mod_ben: gp.Model,
    test_type: str,
) -> ResultRow:
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
        status = "Error: Gurobi != Benders"
        print(f"     ❌ {test_type} FALLO: Gurobi ({obj_gur}) != Benders ({obj_ben})")
    elif expected is not None and obj_gur is not None and int(obj_gur) != expected:
        status = "Error: Calculado != PDF"
        print(f"     ❌ {test_type} FALLO: Calculado ({obj_gur}) != PDF ({expected})")
    else:
        status = "✅ OK"
        print(f"     ✅ {test_type}: Gurobi y Benders coinciden con el PDF.")

    return {
        "Instance": instancia,
        "Type": test_type,
        "Gurobi_LP": obj_gur,
        "Benders_LP": obj_ben,
        "Expected": expected,
        "Status": status
    }

def test_instancia(
    instance_path: str,
    expected_min: float | None,
    expected_max: float | None,
) -> list[ResultRow]:
    """Ejecuta los modelos para una instancia y devuelve sus resultados."""
    nombre_instancia = os.path.basename(instance_path)
    print(f"\n[{nombre_instancia}] Iniciando pruebas...")
    
    resultados_instancia: list[ResultRow] = []

    # --- PRUEBA MINAREA (maximize = False) ---
    if expected_min is not None:
        print("  -> Comprobando MINAREA ...")
        mod_gur_min = build_and_solve_model(instance_path, verbose=False, maximize=False, time_limit=300, obj=0, mode = 3, subtour=2, relaxed=True)
        lp_gur_min = mod_gur_min.ObjVal if mod_gur_min.SolCount > 0 else None
        
        mod_ben_min = optimize_master_LP(instance_path, verbose=False, maximize=False, time_limit=300)
        lp_ben_min= mod_ben_min.ObjVal if mod_ben_min.SolCount > 0 else None
        
        res_min = evaluar_modelo(nombre_instancia, lp_gur_min, lp_ben_min, expected_min, mod_gur_min, mod_ben_min, "MINAREA")
        resultados_instancia.append(res_min)

    # --- PRUEBA MAXAREA (maximize = True) ---
    if expected_max is not None:
        print("  -> Comprobando MAXAREA ...")
        mod_gur_max = build_and_solve_model(instance_path, verbose=False, maximize=True, time_limit=300, obj=0, mode = 3, subtour=2, relaxed=True)
        lp_gur_max = mod_gur_max.ObjVal if mod_gur_max.SolCount > 0 else None
        
        mod_ben_max = optimize_master_LP(instance_path, verbose=False, maximize=True, time_limit=300)
        lp_ben_max = mod_ben_max.ObjVal if mod_ben_max.SolCount > 0 else None

        res_max = evaluar_modelo(nombre_instancia, lp_gur_max, lp_ben_max, expected_max, mod_gur_max, mod_ben_max, "MAXAREA")
        resultados_instancia.append(res_max)

    return resultados_instancia

def guardar_resultados_csv(resultados: list[ResultRow], output_csv: str) -> None:
    """Guarda la lista de resultados en un archivo CSV."""
    if not resultados:
        return
    
    claves = ["Instance", "Type", "Gurobi_LP", "Benders_LP", "Expected", "Status"]
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=claves)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n📁 Resultados guardados exitosamente en: {output_csv}")

def ejecutar_bateria_tests(folder_path: str, csv_path: str, output_csv_path: str) -> None:
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
    
    todos_los_resultados: list[ResultRow] = []
    instancias_evaluadas = 0

    for archivo in archivos_instancia:
        nombre_instancia = archivo.replace('.instance', '').replace('.txt', '')
        
        if nombre_instancia in resultados_esperados:
            ruta_completa = os.path.join(folder_path, archivo)
            lpgap_min, lpgap_max = resultados_esperados[nombre_instancia]
            
            try:
                resultados = test_instancia(ruta_completa, lpgap_min, lpgap_max)
                todos_los_resultados.extend(resultados)
                instancias_evaluadas += 1
            except Exception as e:
                print(f"     ⚠️ ERROR CRÍTICO de ejecución en {nombre_instancia}: {str(e)}")
                todos_los_resultados.append({
                    "Instance": archivo,
                    "Type": "ERROR",
                    "Gurobi_LP": "", "Benders_LP": "", "Expected": "",
                    "Status": f"Exception: {str(e)}"
                })
        else:
            print(f"[-] Saltando '{nombre_instancia}': No está en el CSV de resultados esperados.")
            
    print(f"\n--- Batería completada: {instancias_evaluadas} instancias procesadas ---")
    
    # Guardar todos los resultados de la batería en CSV
    guardar_resultados_csv(todos_los_resultados, output_csv_path)


def diagnose_benders_gap(instance_path: str) -> None:
    # 1. Obtén la solución fraccionaria de tu Maestro de Benders (132B)
    benders_lp = optimize_master_LP(instance_path, verbose=False, plot=False, maximize=True, time_limit=300, 
                             save_cuts=False, crosses_constrain=False) # Tu código de Benders actual
    
    x_benders: dict[tuple[int, int], float] = {k: v.X for k, v in benders_lp._x.items()}
    
    # 2. Construye el Modelo Compacto (el que da 127B y 0% gap)
    compact_model = build_and_solve_model(instance_path, relaxed=True,verbose=False, plot=False, maximize=True, time_limit=300,
                                sum_constrain=True, obj = 0, mode = 0, subtour = 0) # Tu código del modelo compacto actual
    
    # 3. Fuerza al Modelo Compacto a tragarse la solución de Benders
    x_compact = compact_model._x # (Ajusta según cómo guardes x)
    for (i, j), val in x_benders.items():
        if(i,j) not in x_compact:
            print(f"Advertencia: La variable x[{i},{j}] no existe en el modelo compacto. Verifica tus índices.")
            continue
        # Fijamos los límites de la variable para forzar el valor
        x_compact[i, j].LB = val
        x_compact[i, j].UB = val
        
    # 4. Intenta resolver el Modelo Compacto con ese x fijado
    compact_model.optimize()
    
    # 5. El momento de la verdad
    if compact_model.Status == GRB.INFEASIBLE:
        print("\n¡BINGO! El Modelo Compacto rechaza la solución de Benders.")
        print("Calculando qué restricciones está violando...")
        
        # Calcular el subsistema inconsistente
        compact_model.computeIIS()
        compact_model.write("diagnostico_gap.ilp")
        
        print("Revisa el archivo 'diagnostico_gap.ilp'.")
        print("Ahí verás EXACTAMENTE qué ecuaciones del modelo compacto")
        print("hacen que esta solución sea inválida. ¡Esas son las ecuaciones")
        print("que le faltan a tu Subproblema de Benders!")
        
    elif compact_model.Status == GRB.OPTIMAL:
        print("\nAlgo es inconsistente en las funciones objetivo.")
        print("Compacto ObjVal:", compact_model.ObjVal, "Benders ObjVal:", benders_lp.ObjVal)

if __name__ == "__main__":
    # Configura aquí tus rutas
 #   CARPETA_INSTANCIAS = "./instance/little-instances"  # Asegúrate de que esta carpeta contenga tus archivos de instancia
 #   ARCHIVO_CSV_ENTRADA = "test/resultados_lp.csv"
 #   ARCHIVO_CSV_SALIDA = "outputs/reporte_tests_lp.csv"  # El nuevo archivo donde se guardarán los logs
 #   
 #   ejecutar_bateria_tests(CARPETA_INSTANCIAS, ARCHIVO_CSV_ENTRADA, ARCHIVO_CSV_SALIDA)


    # Ejemplo de diagnóstico para una instancia específica
    instance = "instance/stars-0000010.instance" # Cambia esto por la ruta de la instancia que quieres diagnosticar
    diagnose_benders_gap(instance)