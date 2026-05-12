import argparse
import json
import sys
import time
from typing import Any

from models import OAPBendersModel, OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance


def _parse_weight_dict(s: str | None, flag: str) -> dict[str, Any] | None:
    """Parse a weight-dict string into a Python dict.

    Accepted formats:
    - JSON: ``'{"alpha": 1.5, "beta": 2.0}'``
    - Flat key=value pairs: ``'alpha=1.5,beta=2.0'``  (values coerced to float)

    Nested arc-keyed weights (e.g. ``{"alpha": {"[0,3]": 2.0}}``) **require** JSON.
    Unknown group keys are not rejected here; ``_resolve_weights`` handles them
    silently via its ``.get(..., 1.0)`` fallback.

    Parameters
    ----------
    s:
        Raw string from argparse, or ``None``.
    flag:
        CLI flag name used in error messages (e.g. ``"--cut-weights-y"``).

    Returns
    -------
    dict | None
        Parsed dict, or ``None`` when *s* is ``None`` or empty.
    """
    if not s:
        return None
    # JSON path (required for nested / arc-keyed weights)
    try:
        return json.loads(s)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    # Flat key=value fallback: "alpha=1.5,beta=2.0"
    try:
        return {k.strip(): float(v) for k, v in (pair.split("=", 1) for pair in s.split(","))}
    except ValueError:
        print(
            f"ERROR: {flag} is not valid JSON or key=value pairs. Got: {s!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog="run_single_instance.py",
        description="OAP single-instance solver (CLI mode)",
    )
    p.add_argument(
        "instance_name",
        help="Instance name without extension (e.g. london-0000020)",
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
    p.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=0, metavar="0-3")

    # Compacto-only
    p.add_argument(
        "--objective",
        choices=["Fekete", "Internal", "External", "Diagonals"],
        default="Fekete",
    )
    p.add_argument("--subtour", choices=["SCF", "MTZ", "MCF"], default="SCF")
    p.add_argument("--sum-constrain", action="store_true", default=False)
    p.add_argument("--strengthen", action="store_true", default=True)
    p.add_argument("--no-strengthen", action="store_false", dest="strengthen")
    p.add_argument("--semiplane", type=int, choices=[0, 1, 2], default=0, metavar="0-2")
    p.add_argument("--use-knapsack", action="store_true", default=False)
    p.add_argument("--use-cliques", action="store_true", default=False)
    p.add_argument("--crossing-constrain", action="store_true", default=False)

    # Benders-only
    p.add_argument("--benders-method", choices=["farkas", "pi"], default="farkas")
    p.add_argument("--save-cuts", action="store_true", default=False)
    p.add_argument("--crosses-constrain", action="store_true", default=False)
    p.add_argument(
        "--benders-semiplane",
        type=int,
        choices=[0, 1],
        default=0,
        metavar="0-1",
        help=(
            "V1 half-plane constraints for the Benders master (Benders-only). "
            "0=off (default), 1=V1 arc-ordering.  Does NOT affect subproblems."
        ),
    )
    p.add_argument(
        "--use-deepest-cuts",
        action="store_true",
        default=False,
        help="Enable Deepest Benders Cuts (CGSP) for Benders model",
    )
    p.add_argument(
        "--no-deepest-cuts",
        action="store_false",
        dest="use_deepest_cuts",
        help="Disable Deepest Benders Cuts (CGSP); this is the default",
    )
    p.add_argument(
        "--cut-weights-y",
        type=str,
        default=None,
        metavar="JSON_OR_PAIRS",
        help=(
            "L1 normalisation weights for Y-subproblem CGSP. "
            "Accepts JSON (required for nested arc keys, e.g. "
            "'{\"alpha\": 1.5}') or flat key=value pairs "
            "('alpha=1.5,beta=2.0'). Unknown keys fall back to 1.0 silently."
        ),
    )
    p.add_argument(
        "--cut-weights-yp",
        type=str,
        default=None,
        metavar="JSON_OR_PAIRS",
        help=("L1 normalisation weights for Y'-subproblem CGSP. Same format as --cut-weights-y."),
    )

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

    # Validate: Benders-only flags not used with Compacto
    if args.model_type == "Compacto":
        if args.benders_method != "farkas" or args.save_cuts or args.crosses_constrain:
            print(
                "ERROR: --benders-method, --save-cuts, --crosses-constrain "
                "are Benders-only flags. Remove them when --model Compacto is set.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.use_deepest_cuts or args.cut_weights_y or args.cut_weights_yp:
            print(
                "ERROR: --use-deepest-cuts, --cut-weights-y, --cut-weights-yp "
                "are Benders-only flags. Remove them when --model Compacto is set.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.benders_semiplane != 0:
            print(
                "ERROR: --benders-semiplane is a Benders-only flag. Remove it when --model Compacto is set.",
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
        "use_knapsack": (args.use_knapsack if args.model_type == "Compacto" else False),
        "use_cliques": (args.use_cliques if args.model_type == "Compacto" else False),
        "crossing_constrain": (args.crossing_constrain if args.model_type == "Compacto" else False),
        # Benders keys
        "benders_method": (args.benders_method if args.model_type == "Benders" else None),
        "save_cuts": args.save_cuts if args.model_type == "Benders" else False,
        "crosses_constrain": (args.crosses_constrain if args.model_type == "Benders" else False),
        # CGSP flags (Benders-only)
        "use_deepest_cuts": args.use_deepest_cuts if args.model_type == "Benders" else False,
        "cut_weights_y": (
            _parse_weight_dict(args.cut_weights_y, "--cut-weights-y") if args.model_type == "Benders" else None
        ),
        "cut_weights_yp": (
            _parse_weight_dict(args.cut_weights_yp, "--cut-weights-yp") if args.model_type == "Benders" else None
        ),
        # Benders semiplane (master-side V1, Benders-only)
        "benders_semiplane": args.benders_semiplane if args.model_type == "Benders" else 0,
        # Shared
        "time_limit": args.time_limit,
        # Kept for compatibility with downstream logic
        "modify_log_path": False,
        "Extra_Constraints": False,
    }


def main() -> None:
    args = _parse_args()
    config = _build_config_from_args(args)

    instance_name = config["instance_name"]
    print(f"\n[!] Cargando datos para la instancia: {instance_name}...")

    # ---------------------------------------------------------
    # 2. CARGA DE DATOS
    # ---------------------------------------------------------
    instance_dir = config["instance_dir"].rstrip("/\\")  # type: ignore[attr-defined]
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
        modelo = OAPCompactModel(points, triangles, name=instance_name)  # type: ignore[arg-type]
        modelo.build(
            objective=config["objective"],  # type: ignore[arg-type]
            maximize=config["maximize"],  # type: ignore[arg-type]
            subtour=config["subtour"],  # type: ignore[arg-type]
            sum_constrain=config["sum_constrain"],  # type: ignore[arg-type]
            strengthen=config["strengthen"],  # type: ignore[arg-type]
            mode=config["mode"],  # type: ignore[arg-type]
            semiplane=config["semiplane"],  # type: ignore[arg-type]
            use_knapsack=config["use_knapsack"],  # type: ignore[arg-type]
            use_cliques=config["use_cliques"],  # type: ignore[arg-type]
            crossing_constrain=config["crossing_constrain"],  # type: ignore[arg-type]
        )

        print("\n[!] Resolviendo...")
        modelo.solve(relaxed=config["relaxed"], verbose=True)  # type: ignore[arg-type]

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
        modelo = OAPBendersModel(points, triangles, name=instance_name)  # type: ignore[arg-type, assignment]
        modelo.build(  # type: ignore[call-arg]
            benders_method=config["benders_method"],
            maximize=config["maximize"],  # type: ignore[arg-type]
            sum_constrain=config["sum_constrain"],  # type: ignore[arg-type]
            crosses_constrain=config["crosses_constrain"],
            use_deepest_cuts=config["use_deepest_cuts"],
            cut_weights_y=config["cut_weights_y"],
            cut_weights_yp=config["cut_weights_yp"],
            semiplane=config["benders_semiplane"],  # type: ignore[arg-type]
        )

        if config["modify_log_path"]:
            custom_log_path = input(
                "\nIngresa la ruta completa donde deseas guardar el log de cortes (ej. outputs/Logs/milog.json): "
            )
            modelo.set_log_path(custom_log_path)  # type: ignore[attr-defined]  # método solo Benders
            print(f"Ruta del log de cortes actualizada a: {custom_log_path}")

        print("\n[!] Resolviendo (Callback Benders Activado)...")
        # El log_facets de Benders se ejecutará internamente en el callback gracias al flag polihedral
        modelo.solve(  # type: ignore[call-arg]
            relaxed=config["relaxed"],  # type: ignore[arg-type]
            save_cuts=config["save_cuts"],
            verbose=True,
            polihedral=config["polihedral"],  # type: ignore[arg-type, unused-ignore]
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

    print("✅ Resumen de Resultados:")
    print(modelo)


if __name__ == "__main__":
    main()
