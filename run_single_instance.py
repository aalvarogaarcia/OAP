import argparse
import sys
import time

import inquirer

from models import OAPBendersModel, OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CLI mode."""
    p = argparse.ArgumentParser(
        prog="run_single_instance.py",
        description="OAP single-instance solver (interactive or CLI mode)",
    )
    p.add_argument(
        "instance_name",
        nargs="?",
        default=None,
        help="Instance name without extension (e.g. london-0000020). "
        "Omit to use interactive prompts.",
    )
    p.add_argument("--instance-dir", default="instance", metavar="DIR")
    p.add_argument(
        "--model",
        dest="model_type",
        choices=["Compacto", "Benders"],
        default="Compacto",
    )
    p.add_argument("--time-limit", type=int, default=3600, metavar="SECONDS")
    p.add_argument("--maximize", action="store_true", default=True)
    p.add_argument("--no-maximize", action="store_false", dest="maximize")
    p.add_argument("--relaxed", action="store_true", default=False)
    p.add_argument("--polyhedral", action="store_true", default=False)
    p.add_argument(
        "--mode", type=int, choices=[0, 1, 2, 3], default=0, metavar="0-3"
    )

    # Compacto-only
    p.add_argument(
        "--objective",
        choices=["Fekete", "Internal", "External", "Diagonals"],
        default="Fekete",
    )
    p.add_argument(
        "--subtour", choices=["SCF", "MTZ", "MCF"], default="SCF"
    )
    p.add_argument("--sum-constrain", action="store_true", default=False)
    p.add_argument("--strengthen", action="store_true", default=True)
    p.add_argument("--no-strengthen", action="store_false", dest="strengthen")
    p.add_argument(
        "--semiplane", type=int, choices=[0, 1, 2], default=0, metavar="0-2"
    )
    p.add_argument("--use-knapsack", action="store_true", default=False)
    p.add_argument("--use-cliques", action="store_true", default=False)
    p.add_argument("--crossing-constrain", action="store_true", default=False)

    # Benders-only
    p.add_argument(
        "--benders-method", choices=["farkas", "pi"], default="farkas"
    )
    p.add_argument("--save-cuts", action="store_true", default=False)
    p.add_argument("--crosses-constrain", action="store_true", default=False)

    return p.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Build config dict from argparse Namespace."""
    # Validate: Compacto-only flags not used with Benders
    if args.model_type == "Benders":
        if (
            args.objective != "Fekete"
            or args.subtour != "SCF"
            or args.semiplane != 0
            or args.use_knapsack
            or args.use_cliques
            or args.crossing_constrain
        ):
            print(
                "ERROR: --objective, --subtour, --semiplane, --use-knapsack, "
                "--use-cliques, --crossing-constrain are Compacto-only flags. "
                "Remove them when --model Benders is set.",
                file=sys.stderr,
            )
            sys.exit(1)

    return {
        "instance_dir": args.instance_dir,
        "instance_name": args.instance_name,
        "model_type": args.model_type,
        "maximize": args.maximize,
        "relaxed": args.relaxed,
        "polihedral": args.polyhedral,
        # Compacto keys
        "objective": args.objective if args.model_type == "Compacto" else None,
        "subtour": args.subtour if args.model_type == "Compacto" else None,
        "mode": args.mode,
        "sum_constrain": args.sum_constrain,
        "strengthen": args.strengthen if args.model_type == "Compacto" else False,
        "semiplane": args.semiplane if args.model_type == "Compacto" else 0,
        "use_knapsack": (
            args.use_knapsack if args.model_type == "Compacto" else False
        ),
        "use_cliques": (
            args.use_cliques if args.model_type == "Compacto" else False
        ),
        "crossing_constrain": (
            args.crossing_constrain if args.model_type == "Compacto" else False
        ),
        # Benders keys
        "benders_method": (
            args.benders_method if args.model_type == "Benders" else None
        ),
        "save_cuts": args.save_cuts if args.model_type == "Benders" else False,
        "crosses_constrain": (
            args.crosses_constrain if args.model_type == "Benders" else False
        ),
        # Shared
        "time_limit": args.time_limit,
        # Interactive-only (not exposed in CLI)
        "modify_log_path": False,
        "Extra_Constraints": False,
    }


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
        config.setdefault("strengthen", True)
        config.setdefault("semiplane", 0)
        config.setdefault("use_knapsack", False)
        config.setdefault("use_cliques", False)
        config.setdefault("crossing_constrain", False)

    return config


def main() -> None:
    args = _parse_args()
    if args.instance_name is None:
        # Interactive mode
        config = get_experiment_config()
    else:
        # CLI mode
        config = _build_config_from_args(args)

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
            strengthen=config["strengthen"],
            mode=config["mode"],
            semiplane=config["semiplane"],
            use_knapsack=config["use_knapsack"],
            use_cliques=config["use_cliques"],
            crossing_constrain=config["crossing_constrain"],
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
