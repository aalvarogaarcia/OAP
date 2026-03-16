from models.gurobi import build_and_solve_model
from utils.utils import *
from utils.model_stats import get_model_stats
import pandas as pd
import glob
import os
import time
from numbers import Real
import argparse


def main(dir_path="data", 
         ext="*.txt", 
         sum_constrain=True,
         time_limit=7200000,
         obj=None,
         mode=0,
         tsv_ref = "data/instances_reference.tsv"):
    
    files = glob.glob(os.path.join(dir_path, ext))
    files.sort() # Ordenar para que la tabla salga ordenada
    
    all_results_for_excel = []
    subtour_methods = [
    #    0,
    #    1, 
        2] # Los métodos que quieres comparar

    for st_method in subtour_methods:
        print(f"\n>>> EMPEZANDO COMPARATIVA: Subtour Method {st_method} <<<")
        
        # Suffix y nombre de archivo específico para este método de subtour
        st_suffix = f"_subtour{st_method}"
        output_filename = f"outputs/LaTex/resultados_tabla{st_suffix}.tex"

        data_list = []
            
        with open(output_filename, "w", encoding="utf-8") as f:
            
            # 1. Escribir Cabecera de LaTeX en el archivo
            # Usamos f.write() para escribir en el archivo
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
            f.write("\n") # Salto de línea después de la cabecera

            for file in files:
                filename = os.path.basename(file)
                # Feedback para el usuario en la CONSOLA (para saber que no se colgó)
                print(f"Procesando archivo: {filename} ...")
                
                # Limpieza del nombre para LaTeX
                instance_name = filename.replace(".txt", "").replace("_", "-").replace(".instance", "")
                print(f"[{st_method}] Procesando: {filename}...")

                # --- Lógica de resolución (MAX y MIN) ---
                # Importante: Pasar subtour=st_method en build_and_solve_model
                modMax = build_and_solve_model(instance_path=file, maximize=True, 
                                              subtour=st_method, time_limit=time_limit, 
                                              sum_constrain=sum_constrain, obj=obj or 1, mode=mode)
                
                max_lp, max_gap, max_ip, max_time, max_nodes = get_model_stats(modMax)

                modMin = build_and_solve_model(instance_path=file, maximize=False, 
                                              subtour=st_method, time_limit=time_limit, 
                                              sum_constrain=sum_constrain, obj=obj or 2, mode=mode)
                
                min_lp, min_gap, min_ip, min_time, min_nodes = get_model_stats(modMin)

                min_ip_str = f"{min_ip:.0f}" if isinstance(min_ip, Real) and modMax.SolCount > 0 else str(min_ip)
                max_ip_str = f"{max_ip:.0f}" if isinstance(max_ip, Real) and modMax.SolCount > 0 else str(max_ip)
                min_time_str = f"{min_time:.2f}" if isinstance(min_time, Real) and min_time < time_limit else "Timeout"
                max_time_str = f"{max_time:.2f}" if isinstance(max_time, Real) and max_time < time_limit else "Timeout"

                size_n = len(modMax._points_)
                conv_hull_area = getattr(modMax, '_convex_hull_area', None)
                conv_hull_area_str = f"{conv_hull_area:.2f}" if conv_hull_area is not None else "N/A"
                cols = modMax.NumVars
                rows = modMax.NumConstrs
                # Guardar datos para el Excel global incluyendo el método
                res_row = {
                    "Instance": instance_name,
                    "Subtour_Method": st_method,
                    "N": len(modMax._points_),
                    "Convex_Hull_Area": modMax._convex_hull_area,
                    "Cols": modMax.NumVars,
                    "Rows": modMax.NumConstrs,
                    "Min_LP": min_lp, "Min_Gap": min_gap,
                    "Min_IP": min_ip, "Min_Time": min_time, "Min_Nodes": min_nodes,
                    "Max_LP": max_lp, "Max_Gap": max_gap,
                    "Max_IP": max_ip, "Max_Time": max_time, "Max_Nodes": max_nodes
                }
                data_list.append(res_row)
                all_results_for_excel.append(res_row)

                # --- Escribir fila en el .tex actual ---
                row_str = (
                        f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                        f"{min_lp:.2f} & {min_gap:.2f} & {min_ip_str} & {min_time_str} & {min_nodes} & "
                        f"{max_lp:.2f} & {max_gap:.2f} & {max_ip_str} & {max_time_str} & {max_nodes} \\\\"
                        )

                f.write(row_str + "\n")

            f.write(r"\bottomrule \end{tabular} } \end{table} \end{frame} \end{document}")

    # --- GENERACIÓN DEL EXCEL COMPARATIVO FINAL ---
    df_new = pd.DataFrame(all_results_for_excel)
    
    # 1. Pivotamos para tener los métodos lado a lado (opcional pero recomendado)
    df_pivot = df_new.pivot(index="Instance", columns="Subtour_Method", 
                            values=["Min_Time", "Max_Time", "Min_IP", "Max_IP"])
    
    # 2. Intentamos cargar el TSV que generamos antes
    try:
        # Cargar el TSV original (usando los nombres de columnas del TSV proporcionado)
        df_tsv = pd.read_csv(tsv_ref, sep='\t')
        
        # Unir con los resultados de los nuevos experimentos (subtour 0, 1, 2)
        # df_pivot ya viene con niveles de columna (Variable, Subtour_Method)
        df_final = pd.merge(df_tsv, df_pivot, left_on="instance", right_on="Instance", how="left")

        # --- CÁLCULO DE COMPARATIVAS (Ejemplo con Subtour 0) ---
        # Comparamos el tamaño del MILP del TSV original vs el nuevo método 0
        if 0 in subtour_methods:
            # Diferencia en Columnas y Filas
            df_final['Diff_Cols_ST0'] = df_final[('Cols', 0)] - df_final['MILP_col']
            df_final['Diff_Rows_ST0'] = df_final[('Rows', 0)] - df_final['MILP_row']
            
            # Diferencia en IP Value (para asegurar que el óptimo es el mismo)
            df_final['Diff_IP_Min_ST0'] = df_final[('Min_IP', 0)] - df_final['MIN_IPvalue']
            df_final['Diff_IP_Max_ST0'] = df_final[('Max_IP', 0)] - df_final['MAX_IPvalue']
            
            # Comparación de LP Gap (Nuevo vs Original)
            df_final['Gap_Improvement_Min_ST0'] = df_final['MIN_LPgap'] - df_final[('Min_Gap', 0)]
            df_final['Gap_Improvement_Max_ST0'] = df_final['MAX_LPgap'] - df_final[('Max_Gap', 0)]

            df_final['Diff_LP_Min_ST0'] = df_final[('Min_LP', 0)] - df_final['MIN_LPvalue']
            df_final['Diff_LP_Max_ST0'] = df_final[('Max_LP', 0)] - df_final['MAX_LPvalue']

            df_final['Time_Diff_Min_ST0'] = df_final[('Min_Time', 0)] - df_final['MIN_Time']
            df_final['Time_Diff_Max_ST0'] = df_final[('Max_Time', 0)] - df_final['MAX_Time']

            df_final['RelativeTime_Diff_Min_ST0'] = (df_final[('Min_Time', 0)] - df_final['MIN_Time']) / df_final['MIN_Time'] * 100
            df_final['RelativeTime_Diff_Max_ST0'] = (df_final[('Max_Time', 0)] - df_final['MAX_Time']) / df_final['MAX_Time'] * 100

            df_final['Nodes_Diff_Min_ST0'] = df_final[('Min_Nodes', 0)] - df_final['MIN_Nodes']
            df_final['Nodes_Diff_Max_ST0'] = df_final[('Max_Nodes', 0)] - df_final['MAX_Nodes']

        elif 1 in subtour_methods:
            df_final['Diff_Cols_ST1'] = df_final[('Cols', 1)] - df_final['MILP_col']
            df_final['Diff_Rows_ST1'] = df_final[('Rows', 1)] - df_final['MILP_row']

            df_final['Diff_IP_Min_ST1'] = df_final[('Min_IP', 1)] - df_final['MIN_IPvalue']
            df_final['Diff_IP_Max_ST1'] = df_final[('Max_IP', 1)] - df_final['MAX_IPvalue']

            df_final['Gap_Improvement_Min_ST1'] = df_final['MIN_LPgap'] - df_final[('Min_Gap', 1)]
            df_final['Gap_Improvement_Max_ST1'] = df_final['MAX_LPgap'] - df_final[('Max_Gap', 1)]

            df_final['Diff_LP_Min_ST1'] = df_final[('Min_LP', 1)] - df_final['MIN_LPvalue']
            df_final['Diff_LP_Max_ST1'] = df_final[('Max_LP', 1)] - df_final['MAX_LPvalue']

            df_final['Time_Diff_Min_ST1'] = df_final[('Min_Time', 1)] - df_final['MIN_Time']
            df_final['Time_Diff_Max_ST1'] = df_final[('Max_Time', 1)] - df_final['MAX_Time']

            df_final['RelativeTime_Diff_Min_ST1'] = (df_final[('Min_Time', 1)] - df_final['MIN_Time']) / df_final['MIN_Time'] * 100
            df_final['RelativeTime_Diff_Max_ST1'] = (df_final[('Max_Time', 1)] - df_final['MAX_Time']) / df_final['MAX_Time'] * 100

            df_final['Nodes_Diff_Min_ST1'] = df_final[('Min_Nodes', 1)] - df_final['MIN_Nodes']
            df_final['Nodes_Diff_Max_ST1'] = df_final[('Max_Nodes', 1)] - df_final['MAX_Nodes']

        elif 2 in subtour_methods:
            df_final['Diff_Cols_ST2'] = df_final[('Cols', 2)] - df_final['MILP_col']
            df_final['Diff_Rows_ST2'] = df_final[('Rows', 2)] - df_final['MILP_row']

            df_final['Diff_IP_Min_ST2'] = df_final[('Min_IP', 2)] - df_final['MIN_IPvalue']
            df_final['Diff_IP_Max_ST2'] = df_final[('Max_IP', 2)] - df_final['MAX_IPvalue']

            df_final['Gap_Improvement_Min_ST2'] = df_final['MIN_LPgap'] - df_final[('Min_Gap', 2)]
            df_final['Gap_Improvement_Max_ST2'] = df_final['MAX_LPgap'] - df_final[('Max_Gap', 2)]

            df_final['Diff_LP_Min_ST2'] = df_final[('Min_LP', 2)] - df_final['MIN_LPvalue']
            df_final['Diff_LP_Max_ST2'] = df_final[('Max_LP', 2)] - df_final['MAX_LPvalue']

            df_final['Time_Diff_Min_ST2'] = df_final[('Min_Time', 2)] - df_final['MIN_Time']
            df_final['Time_Diff_Max_ST2'] = df_final[('Max_Time', 2)] - df_final['MAX_Time']

            df_final['RelativeTime_Diff_Min_ST2'] = (df_final[('Min_Time', 2)] - df_final['MIN_Time']) / df_final['MIN_Time'] * 100
            df_final['RelativeTime_Diff_Max_ST2'] = (df_final[('Max_Time', 2)] - df_final['MAX_Time']) / df_final['MAX_Time'] * 100

            df_final['Nodes_Diff_Min_ST2'] = df_final[('Min_Nodes', 2)] - df_final['MIN_Nodes']
            df_final['Nodes_Diff_Max_ST2'] = df_final[('Max_Nodes', 2)] - df_final['MAX_Nodes']

        
            

        # --- LIMPIEZA DE FORMATO PARA EXCEL ---
        # Flatten de las columnas multi-nivel para que el Excel sea legible
        df_final.columns = [
            f"{col[0]}_ST{col[1]}" if isinstance(col, tuple) and col[1] != "" else col 
            for col in df_final.columns
        ]

        output_excel = "outputs/Excel/Comparativa_Tecnica_Completa.xlsx"
        
        # Usamos un styler para resaltar diferencias negativas (opcional)
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, sheet_name='Comparativa', index=False)
            
            # Auto-ajustar columnas (opcional pero muy útil)
            worksheet = writer.sheets['Comparativa']
            for i, col in enumerate(df_final.columns):
                column_len = max(df_final[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)

        print(f"\n¡Éxito! Comparativa técnica guardada en {output_excel}")

    except FileNotFoundError:
        df_new.to_excel("outputs/Excel/Resultados_Subtours_Sin_TSV.xlsx")
        print(f"Error: No se encontró el archivo de referencia {tsv_ref}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run optimization models on instance files with configurable constraints and time limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multiple_run.py instance *.instance
  python multiple_run.py instance *.instance --sum
  python multiple_run.py instance *.instance --time-limit 3600000
  python multiple_run.py instance *.instance --sum  --time-limit 1800000
        """
    )
    
    # Positional arguments
    parser.add_argument('dir_path',
                        nargs='?',
                        default='instance',
                        help='Directory path containing instance files')
    parser.add_argument('ext', 
                       nargs='?',
                       default='*.instance',
                       help='File extension pattern (default: *.instance)')
    
    # Optional flags
    parser.add_argument('--sum', 
                       action='store_true',
                       dest='sum_constrain',
                       help='Enable sum constraints on triangles')
    
    parser.add_argument('--time-limit',
                       type=int,
                       default=7200,
                       dest='time_limit',
                       help='Time limit in seconds (default: 7200)')
    
    parser.add_argument('--obj',
                       type=int,
                       default= None,
                       dest='obj',
                       help='Objective function type (default: 2)' \
                       ': 0=FEKETE, 1=INTERNALAREA, 2=EXTERNALAREA, 3=DIAGONAL)')
    
    parser.add_argument('--mode',
                       type=int,
                       default=None,
                       dest='mode',
                       help="Mode for objective function (default: 0)"\
                        "\nFEKETE: 0=Standard, 1=Trapeze, 2=Polygons, 3 = Polygons prime"\
                        "\nINTERNALAREA/EXTERNALAREA: 0=Standard"\
                        "\nDIAGONAL: 0=Internal, 1=External (Not implemented yet)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(dir_path=args.dir_path,
         ext=args.ext,
         sum_constrain=args.sum_constrain,
         time_limit=args.time_limit,
         obj=args.obj,
         mode=args.mode,
         tsv_ref="test/TablaResultadosA4.tsv")
