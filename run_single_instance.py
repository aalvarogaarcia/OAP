import time

import inquirer

from models import OAPBendersModel, OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance


def get_experiment_config():
    """
    Pregunta al usuario por la configuración de la instancia a ejecutar.
    """
    print("\n" + "=" * 50)
    print(" 🚀 EJECUTOR DE INSTANCIAS ÚNICAS - OAP")
    print("=" * 50 + "\n")

    # 1. Preguntas base
    base_questions = [
        inquirer.Text(
            "instance_dir",
            message="Instance directory",
            default="instance",
        ),
        inquirer.Text("instance_name", message="Nombre de la instancia (ej. london-0000020)", default="london-0000020"),
        inquirer.List(
            "model_type", message="¿Qué modelo deseas ejecutar?", choices=["Compacto", "Benders"], default="Benders"
        ),
        inquirer.Confirm(
            "maximize", message="¿Deseas maximizar la función objetivo? (Si no, se minimizará)", default=True
        ),
        inquirer.Confirm("relaxed", message="¿Deseas resolver la relajación lineal del modelo?", default=False),
        inquirer.Confirm(
            "polihedral",
            message="¿Deseas generar el log polihedral? (Solo para análisis, no recomendado para instancias grandes)",
            default=False,
        ),
    ]

    config = inquirer.prompt(base_questions)

    # Si el usuario cancela (Ctrl+C)
    if not config:
        return None

    # 2. Preguntas condicionales
    if config["model_type"] == "Benders":
        benders_questions = [
            inquirer.List(
                "benders_method",
                message="¿Qué método de Benders quieres utilizar?",
                choices=["farkas", "pi"],
                default="farkas",
            ),
            inquirer.Confirm(
                "save_cuts",
                message="¿Deseas guardar el log de cortes para análisis posterior (ej. UMAP/PDF)?",
                default=True,
            ),
            inquirer.Confirm(
                "sum_constrain",
                message="Enable triangle-sum constraints?",
                default=False,
            ),
            inquirer.Confirm(
                "crosses_constrain",
                message="Enable non-crossing arc constraints?",
                default=False,
            ),
            inquirer.Confirm(
                "modify_log_path", message="¿Deseas modificar la ruta predeterminada del log de cortes?", default=False
            ),
        ]
        benders_config = inquirer.prompt(benders_questions)
        if not benders_config:
            return None
        config.update(benders_config)
        config["mode"] = 0

    else:  # Modelo Compacto
        compact_questions = [
            inquirer.List(
                "objective",
                message="¿Qué función objetivo deseas usar?",
                choices=["Fekete", "Internal", "External", "Diagonals"],
                default="Fekete",
            ),
            inquirer.List(
                "subtour",
                message="¿Qué tipo de subtour elimination deseas usar?",
                choices=["SCF", "MTZ", "MCF"],
                default="SCF",
            ),
            inquirer.List(
                "mode",
                message="Objective mode (0-3)",
                choices=["0", "1", "2", "3"],
                default="0",
            ),
            inquirer.Confirm(
                "sum_constrain",
                message="Enable triangle-sum constraints?",
                default=False,
            ),
            inquirer.Confirm(
                "Extra_Constraints",
                message="¿Deseas agregar restricciones adicionales (ej. de corte, simetría, etc.)?",
                default=False,
            ),
        ]
        compact_config = inquirer.prompt(compact_questions)
        if not compact_config:
            return None
        config.update(compact_config)
        config["mode"] = int(config["mode"])
        config["crosses_constrain"] = False
        config["benders_method"] = None
        config["save_cuts"] = False
        config["modify_log_path"] = False

    return config


def main():
    # 1. Capturar la configuración del usuario
    config = get_experiment_config()

    if not config:
        print("\nEjecución cancelada por el usuario.")
        return

    instance_name = config["instance_name"]
    print(f"\n[!] Cargando datos para la instancia: {instance_name}...")

    # ---------------------------------------------------------
    # 2. CARGA DE DATOS
    # ---------------------------------------------------------
    instance_dir = config["instance_dir"].rstrip("/\\")
    filepath = f"{instance_dir}/{instance_name}.instance"
    try:
        points = read_indexed_instance(filepath)
        triangles = compute_triangles(points)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {filepath}")
        return
    # ---------------------------------------------------------

    start_time = time.time()

    # 3. INSTANCIAR Y EJECUTAR EL MODELO ELEGIDO
    if config["model_type"] == "Compacto":
        print("\n[!] Construyendo OAPCompactModel...")
        modelo = OAPCompactModel(points, triangles, name=instance_name)
        modelo.build(
            objective=config["objective"],
            maximize=config["maximize"],
            subtour=config["subtour"],
            sum_constrain=config["sum_constrain"],
            mode=config["mode"],
        )

        print("\n[!] Resolviendo...")
        modelo.solve(relaxed=config["relaxed"], verbose=True)

        # Bloque Polihedral corregido: Preguntamos primero, extraemos/guardamos después
        if config["polihedral"]:
            print("\n[!] Iniciando análisis polihedral...")
            var_prefixes_input = input("Prefijo de las variables para extraer el poliedro (ej. 'x', o 'x,y,yp'): ")
            var_prefixes = [prefix.strip() for prefix in var_prefixes_input.split(",")]

            log_poly_path = f"outputs/Logs/{instance_name}_compact_polyhedral.json"

            # log_facets ya hace la extracción y guarda el JSONL
            modelo.log_facets(filepath=log_poly_path, var_prefixes=var_prefixes, verbose=True)
            print(f"✅ Log polihedral guardado en: {log_poly_path}")

    elif config["model_type"] == "Benders":
        print(f"\n[!] Construyendo OAPBendersModel (Método: {config['benders_method']})...")
        modelo = OAPBendersModel(points, triangles, name=instance_name)
        modelo.build(
            benders_method=config["benders_method"],
            maximize=config["maximize"],
            sum_constrain=config["sum_constrain"],
            crosses_constrain=config["crosses_constrain"],
        )

        if config["modify_log_path"]:
            custom_log_path = input(
                "\nIngresa la ruta completa donde deseas guardar el log de cortes (ej. outputs/Logs/milog.json): "
            )
            modelo.set_log_path(custom_log_path)  # Asumo que tienes esta función implementada
            print(f"Ruta del log de cortes actualizada a: {custom_log_path}")

        print("\n[!] Resolviendo (Callback Benders Activado)...")
        # El log_facets de Benders se ejecutará internamente en el callback gracias al flag polihedral
        modelo.solve(
            relaxed=config["relaxed"], save_cuts=config["save_cuts"], verbose=True, polihedral=config["polihedral"]
        )

    # 4. RESULTADOS
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(" 🎉 EJECUCIÓN FINALIZADA")
    print(f" Tiempo total: {elapsed_time:.2f} segundos")

    if config["model_type"] == "Benders" and config["save_cuts"]:
        ruta_final = getattr(modelo, "log_path", f"outputs/Logs/benders_{instance_name}.json")
        print(f" 💾 Logs de cortes guardados en: {ruta_final}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
