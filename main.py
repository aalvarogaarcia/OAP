from __future__ import annotations

import argparse
import glob
import os
from numbers import Real
from typing import Any

import pandas as pd

from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ResultRow = dict[str, Any]

# ---------------------------------------------------------------------------
# Constants — integer CLI codes mapped to the string keys the model expects
# ---------------------------------------------------------------------------
_SUBTOUR_MAP: dict[int, str] = {0: "SCF", 1: "MTZ", 2: "MCF"}
_OBJECTIVE_MAP: dict[int, str] = {
    0: "Fekete",
    1: "Internal",
    2: "External",
    3: "Diagonals",
}

# Default objective indices used when --obj is not supplied.
_DEFAULT_OBJ_MAX: int = 1  # Internal (maximise internal area)
_DEFAULT_OBJ_MIN: int = 2  # External (minimise external area)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, spec: str = ".2f") -> str:
    """Format *value* with *spec* when it is numeric; fall back to str() for
    sentinel values such as '-' returned by get_model_stats()."""
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def build_and_solve(
    instance_path: str,
    maximize: bool,
    subtour: int,
    time_limit: int,
    sum_constrain: bool,
    obj: int,
    mode: int,
) -> OAPCompactModel:
    """Construct, build, and solve an OAPCompactModel for *instance_path*.

    Args:
        instance_path: Path to a ``.instance`` or ``.txt`` file.
        maximize: ``True`` to maximise the objective, ``False`` to minimise.
        subtour: Integer code for the subtour-elimination method (see
            ``_SUBTOUR_MAP``).
        time_limit: Solver time limit in **seconds**.
        sum_constrain: Whether to add triangle-sum constraints.
        obj: Integer code for the objective function (see ``_OBJECTIVE_MAP``).
        mode: Sub-mode for the chosen objective function.

    Returns:
        A solved (or timed-out) ``OAPCompactModel`` instance.
    """
    instance_name = (
        os.path.basename(instance_path)
        .replace(".instance", "")
        .replace(".txt", "")
    )
    points = read_indexed_instance(instance_path)
    triangles = compute_triangles(points)
    model = OAPCompactModel(points, triangles, name=instance_name)
    model.build(
        objective=_OBJECTIVE_MAP[obj],
        mode=mode,
        maximize=maximize,
        subtour=_SUBTOUR_MAP[subtour],
        sum_constrain=sum_constrain,
    )
    model.solve(time_limit=time_limit)
    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    dir_path: str = "data",
    ext: str = "*.txt",
    sum_constrain: bool = True,
    time_limit: int = 7200,
    obj: int | None = None,
    mode: int = 0,
    tsv_ref: str = "data/instances_reference.tsv",
) -> None:
    files = glob.glob(os.path.join(dir_path, ext))
    files.sort()

    all_results_for_excel: list[ResultRow] = []
    subtour_methods = [
        0,
    #    1,
        2]  # Los métodos que quieres comparar
    print(f"Subtour methods to compare: {subtour_methods}")

    for st_method in subtour_methods:
        print(f"\n>>> EMPEZANDO COMPARATIVA: Subtour Method {st_method} <<<")

        st_suffix = f"_subtour{st_method}"
        output_filename = f"outputs/LaTex/resultados_tabla{st_suffix}.tex"
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data_list: list[ResultRow] = []
        
        with open(output_filename, "w", encoding="utf-8") as f:

            # 1. Escribir Cabecera de LaTeX en el archivo
            f.write(r"""
\documentclass{beamer}					% Document class\usepackage[english]{babel}				% Set language
\usepackage[utf8x]{inputenc}			% Set encoding
\usepackage{multicol}
\usepackage{multirow}
\mode<presentation>						% Set options
{
\usetheme{default}					% Set theme
\usecolortheme{default} 				% Set colors
\usefonttheme{default}  				% Set font theme
\setbeamertemplate{caption}[numbered]	% Set caption to be numbered
}% Uncomment this to have the outline at the beginning of each section highlighted.
%\AtBeginSection[]
%{
%  \begin{frame}{Outline}
%    \tableofcontents[currentsection]
%  \end{frame}
%}\usepackage{graphicx}					% For including figures
\usepackage{booktabs}					% For table rules
\usepackage{hyperref}					% For cross-referencing\title{Title for a minimal beamer presentation}	% Presentation title
\author{Author One}								% Presentation author
\institute{Name of institution}					% Author affiliation
\date{\today}									% Today's date	\begin{document}% Title page
% This page includes the informations defined earlier including title, author/s, affiliation/s and the date
\begin{frame}\begin{table}[htbp]
\centering
\setbeamerfont{caption}{size=\tiny}
\caption{Resultados computacionales: Comparación de Áreas (Generado automáticamente)}
\label{tab:resultados_optimizacion}
                
\setlength{\tabcolsep}{1.5pt}
                \resizebox{\textwidth}{!}{%
\tiny
\begin{tabular}{lrrrr|rrrrr|rrrrr}
\toprule
\multirow{2}{*}{\textbf{Instance}} & \multirow{2}{*}{\textbf{Size}} & \multirow{2}{*}{\textbf{Conv.Hull}} & \multirow{2}{*}{\textbf{Cols}} & \multirow{2}{*}{\textbf{Rows}} & \multicolumn{5}{c|}{\textbf{MINAREA}} & \multicolumn{5}{c}{\textbf{MAXAREA}} \\
\cmidrule(lr){6-10} \cmidrule(lr){11-15}
& & & & & \textbf{LP Val} & \textbf{Gap (\%)} & \textbf{IP Val} & \textbf{Time} & \textbf{Nodes} & \textbf{LP Val} & \textbf{Gap (\%)} & \textbf{IP Val} & \textbf{Time} & \textbf{Nodes} \\
\midrule
    """)
            f.write("\n")

            for file in files:
                filename = os.path.basename(file)
                print(f"Procesando archivo: {filename} ...")

                instance_name = (
                    filename
                    .replace(".txt", "")
                    .replace("_", "-")
                    .replace(".instance", "")
                )
                print(f"[{st_method}] Procesando: {filename}...")

                # Resolve objective index, guarding against obj=0 (Fekete) being falsy.
                obj_max: int = obj if obj is not None else _DEFAULT_OBJ_MAX
                obj_min: int = obj if obj is not None else _DEFAULT_OBJ_MIN

                # --- Lógica de resolución (MAX y MIN) ---
                modMax = build_and_solve(
                    instance_path=file,
                    maximize=True,
                    subtour=st_method,
                    time_limit=time_limit,
                    sum_constrain=sum_constrain,
                    obj=obj_max,
                    mode=mode,
                )
                max_lp, max_gap, max_ip, max_time, max_nodes = modMax.get_model_stats()

                modMin = build_and_solve(
                    instance_path=file,
                    maximize=False,
                    subtour=st_method,
                    time_limit=time_limit,
                    sum_constrain=sum_constrain,
                    obj=obj_min,
                    mode=mode,
                )
                min_lp, min_gap, min_ip, min_time, min_nodes = modMin.get_model_stats()

                # Use each model's own SolCount — do NOT cross-reference min with max.
                min_ip_str = (
                    f"{min_ip:.0f}"
                    if isinstance(min_ip, Real) and modMin.model.SolCount > 0
                    else str(min_ip)
                )
                max_ip_str = (
                    f"{max_ip:.0f}"
                    if isinstance(max_ip, Real) and modMax.model.SolCount > 0
                    else str(max_ip)
                )
                min_time_str = (
                    f"{min_time:.2f}"
                    if isinstance(min_time, Real) and min_time < time_limit
                    else "Timeout"
                )
                max_time_str = (
                    f"{max_time:.2f}"
                    if isinstance(max_time, Real) and max_time < time_limit
                    else "Timeout"
                )

                size_n: int = len(modMax.points)
                conv_hull_area_str: str = f"{modMax.convex_hull_area:.2f}"
                cols: int = modMax.model.NumVars
                rows: int = modMax.model.NumConstrs

                # Guardar datos para el Excel global incluyendo el método
                res_row: ResultRow = {
                    "Instance": instance_name,
                    "Subtour_Method": st_method,
                    "N": len(modMax.points),
                    "Convex_Hull_Area": modMax.convex_hull_area,
                    "Cols": modMax.model.NumVars,
                    "Rows": modMax.model.NumConstrs,
                    "Min_LP": min_lp, "Min_Gap": min_gap,
                    "Min_IP": min_ip, "Min_Time": min_time, "Min_Nodes": min_nodes,
                    "Max_LP": max_lp, "Max_Gap": max_gap,
                    "Max_IP": max_ip, "Max_Time": max_time, "Max_Nodes": max_nodes,
                }
                data_list.append(res_row)
                all_results_for_excel.append(res_row)

                # --- Escribir fila en el .tex actual ---
                # _fmt() guards against formatting "-" sentinel values with :.2f
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"{_fmt(min_lp)} & {_fmt(min_gap)} & {min_ip_str} & {min_time_str} & {min_nodes} & "
                    f"{_fmt(max_lp)} & {_fmt(max_gap)} & {max_ip_str} & {max_time_str} & {max_nodes} \\\\"
                )
                f.write(row_str + "\n")

            f.write(r"\bottomrule \end{tabular} } \end{table} \end{frame} \end{document}")

    # --- GENERACIÓN DEL EXCEL COMPARATIVO FINAL ---
    df_new = pd.DataFrame(all_results_for_excel)

    # 1. Pivotamos para tener los métodos lado a lado (opcional pero recomendado)
    df_pivot = df_new.pivot(
        index="Instance",
        columns="Subtour_Method",
        values=["Min_Time", "Max_Time", "Min_IP", "Max_IP"],
    )

    # 2. Intentamos cargar el TSV que generamos antes
    try:
        df_tsv = pd.read_csv(tsv_ref, sep="\t")

        df_final = pd.merge(
            df_tsv, df_pivot, left_on="instance", right_on="Instance", how="left"
        )

        # --- CÁLCULO DE COMPARATIVAS ---
        if 0 in subtour_methods:
            df_final["Diff_Cols_ST0"] = df_final[("Cols", 0)] - df_final["MILP_col"]
            df_final["Diff_Rows_ST0"] = df_final[("Rows", 0)] - df_final["MILP_row"]
            df_final["Diff_IP_Min_ST0"] = df_final[("Min_IP", 0)] - df_final["MIN_IPvalue"]
            df_final["Diff_IP_Max_ST0"] = df_final[("Max_IP", 0)] - df_final["MAX_IPvalue"]
            df_final["Gap_Improvement_Min_ST0"] = df_final["MIN_LPgap"] - df_final[("Min_Gap", 0)]
            df_final["Gap_Improvement_Max_ST0"] = df_final["MAX_LPgap"] - df_final[("Max_Gap", 0)]
            df_final["Diff_LP_Min_ST0"] = df_final[("Min_LP", 0)] - df_final["MIN_LPvalue"]
            df_final["Diff_LP_Max_ST0"] = df_final[("Max_LP", 0)] - df_final["MAX_LPvalue"]
            df_final["Time_Diff_Min_ST0"] = df_final[("Min_Time", 0)] - df_final["MIN_Time"]
            df_final["Time_Diff_Max_ST0"] = df_final[("Max_Time", 0)] - df_final["MAX_Time"]
            df_final["RelativeTime_Diff_Min_ST0"] = (
                (df_final[("Min_Time", 0)] - df_final["MIN_Time"]) / df_final["MIN_Time"] * 100
            )
            df_final["RelativeTime_Diff_Max_ST0"] = (
                (df_final[("Max_Time", 0)] - df_final["MAX_Time"]) / df_final["MAX_Time"] * 100
            )
            df_final["Nodes_Diff_Min_ST0"] = df_final[("Min_Nodes", 0)] - df_final["MIN_Nodes"]
            df_final["Nodes_Diff_Max_ST0"] = df_final[("Max_Nodes", 0)] - df_final["MAX_Nodes"]

        elif 1 in subtour_methods:
            df_final["Diff_Cols_ST1"] = df_final[("Cols", 1)] - df_final["MILP_col"]
            df_final["Diff_Rows_ST1"] = df_final[("Rows", 1)] - df_final["MILP_row"]
            df_final["Diff_IP_Min_ST1"] = df_final[("Min_IP", 1)] - df_final["MIN_IPvalue"]
            df_final["Diff_IP_Max_ST1"] = df_final[("Max_IP", 1)] - df_final["MAX_IPvalue"]
            df_final["Gap_Improvement_Min_ST1"] = df_final["MIN_LPgap"] - df_final[("Min_Gap", 1)]
            df_final["Gap_Improvement_Max_ST1"] = df_final["MAX_LPgap"] - df_final[("Max_Gap", 1)]
            df_final["Diff_LP_Min_ST1"] = df_final[("Min_LP", 1)] - df_final["MIN_LPvalue"]
            df_final["Diff_LP_Max_ST1"] = df_final[("Max_LP", 1)] - df_final["MAX_LPvalue"]
            df_final["Time_Diff_Min_ST1"] = df_final[("Min_Time", 1)] - df_final["MIN_Time"]
            df_final["Time_Diff_Max_ST1"] = df_final[("Max_Time", 1)] - df_final["MAX_Time"]
            df_final["RelativeTime_Diff_Min_ST1"] = (
                (df_final[("Min_Time", 1)] - df_final["MIN_Time"]) / df_final["MIN_Time"] * 100
            )
            df_final["RelativeTime_Diff_Max_ST1"] = (
                (df_final[("Max_Time", 1)] - df_final["MAX_Time"]) / df_final["MAX_Time"] * 100
            )
            df_final["Nodes_Diff_Min_ST1"] = df_final[("Min_Nodes", 1)] - df_final["MIN_Nodes"]
            df_final["Nodes_Diff_Max_ST1"] = df_final[("Max_Nodes", 1)] - df_final["MAX_Nodes"]

        elif 2 in subtour_methods:
            df_final["Diff_Cols_ST2"] = df_final[("Cols", 2)] - df_final["MILP_col"]
            df_final["Diff_Rows_ST2"] = df_final[("Rows", 2)] - df_final["MILP_row"]
            df_final["Diff_IP_Min_ST2"] = df_final[("Min_IP", 2)] - df_final["MIN_IPvalue"]
            df_final["Diff_IP_Max_ST2"] = df_final[("Max_IP", 2)] - df_final["MAX_IPvalue"]
            df_final["Gap_Improvement_Min_ST2"] = df_final["MIN_LPgap"] - df_final[("Min_Gap", 2)]
            df_final["Gap_Improvement_Max_ST2"] = df_final["MAX_LPgap"] - df_final[("Max_Gap", 2)]
            df_final["Diff_LP_Min_ST2"] = df_final[("Min_LP", 2)] - df_final["MIN_LPvalue"]
            df_final["Diff_LP_Max_ST2"] = df_final[("Max_LP", 2)] - df_final["MAX_LPvalue"]
            df_final["Time_Diff_Min_ST2"] = df_final[("Min_Time", 2)] - df_final["MIN_Time"]
            df_final["Time_Diff_Max_ST2"] = df_final[("Max_Time", 2)] - df_final["MAX_Time"]
            df_final["RelativeTime_Diff_Min_ST2"] = (
                (df_final[("Min_Time", 2)] - df_final["MIN_Time"]) / df_final["MIN_Time"] * 100
            )
            df_final["RelativeTime_Diff_Max_ST2"] = (
                (df_final[("Max_Time", 2)] - df_final["MAX_Time"]) / df_final["MAX_Time"] * 100
            )
            df_final["Nodes_Diff_Min_ST2"] = df_final[("Min_Nodes", 2)] - df_final["MIN_Nodes"]
            df_final["Nodes_Diff_Max_ST2"] = df_final[("Max_Nodes", 2)] - df_final["MAX_Nodes"]

        # --- LIMPIEZA DE FORMATO PARA EXCEL ---
        df_final.columns = [
            f"{col[0]}_ST{col[1]}" if isinstance(col, tuple) and col[1] != "" else col
            for col in df_final.columns
        ]

        output_excel = "outputs/Excel/Comparativa_Tecnica_Completa.xlsx"
        output_dir = os.path.dirname(output_excel)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            df_final.to_excel(writer, sheet_name="Comparativa", index=False)

            worksheet = writer.sheets["Comparativa"]
            for i, col in enumerate(df_final.columns):
                column_len = max(df_final[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)

        print(f"\n¡Éxito! Comparativa técnica guardada en {output_excel}")

    except FileNotFoundError:
        df_new.to_excel("outputs/Excel/Resultados_Subtours_Sin_TSV.xlsx")
        print(f"Error: No se encontró el archivo de referencia {tsv_ref}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimization models on instance files with configurable constraints and time limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py instance *.instance
  python main.py instance *.instance --sum
  python main.py instance *.instance --time-limit 3600
  python main.py instance *.instance --sum --time-limit 1800
        """,
    )

    # Positional arguments
    parser.add_argument(
        "dir_path",
        nargs="?",
        default="instance",
        help="Directory path containing instance files",
    )
    parser.add_argument(
        "ext",
        nargs="?",
        default="*.instance",
        help="File extension pattern (default: *.instance)",
    )

    # Optional flags
    parser.add_argument(
        "--sum",
        action="store_true",
        dest="sum_constrain",
        help="Enable sum constraints on triangles",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=7200,
        dest="time_limit",
        help="Time limit in seconds (default: 7200)",
    )
    parser.add_argument(
        "--obj",
        type=int,
        default=None,
        dest="obj",
        help=(
            "Objective function type: "
            "0=FEKETE, 1=INTERNALAREA, 2=EXTERNALAREA, 3=DIAGONAL "
            "(defaults to 1 for maximise, 2 for minimise)"
        ),
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        dest="mode",
        help=(
            "Mode for objective function (default: 0)\n"
            "FEKETE: 0=Standard, 1=Trapeze, 2=Polygons, 3=Polygons prime\n"
            "INTERNALAREA/EXTERNALAREA: 0=Standard\n"
            "DIAGONAL: 0=Internal, 1=External (Not implemented yet)"
        ),
    )

    args = parser.parse_args()

    main(
        dir_path=args.dir_path,
        ext=args.ext,
        sum_constrain=args.sum_constrain,
        time_limit=args.time_limit,
        obj=args.obj,
        mode=args.mode,
        tsv_ref="test/TablaResultadosA4.tsv",
    )
