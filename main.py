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
    subtour_methods = [0, 1, 2] # Los métodos que quieres comparar

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

                # Guardar datos para el Excel global incluyendo el método
                res_row = {
                    "Instance": instance_name,
                    "Subtour_Method": st_method,
                    "Cols": modMax.NumVars,
                    "Rows": modMax.NumConstrs,
                    "Min_IP": min_ip, "Min_Time": min_time,
                    "Max_IP": max_ip, "Max_Time": max_time
                }
                data_list.append(res_row)
                all_results_for_excel.append(res_row)

                # --- Escribir fila en el .tex actual ---
                f.write(f"{instance_name} & {min_ip:.2f} & {min_time:.2f} & {max_ip:.2f} & {max_time:.2f} \\\\\n")

            f.write(r"\bottomrule \end{tabular} } \end{table} \end{frame} \end{document}")

    # --- GENERACIÓN DEL EXCEL COMPARATIVO FINAL ---
    df_new = pd.DataFrame(all_results_for_excel)
    
    # 1. Pivotamos para tener los métodos lado a lado (opcional pero recomendado)
    df_pivot = df_new.pivot(index="Instance", columns="Subtour_Method", 
                            values=["Min_Time", "Max_Time", "Min_IP", "Max_IP"])
    
    # 2. Intentamos cargar el TSV que generamos antes
    try:
        df_tsv = pd.read_csv(tsv_reference_path, sep='\t')
        # Unimos los resultados nuevos con los del TSV usando la columna 'Instance' o 'instance'
        # Asegúrate de que los nombres coincidan
        df_final = pd.merge(df_tsv, df_pivot, left_on="instance", right_on="Instance", how="left")
        
        output_excel = "outputs/Excel/Comparativa_Completa_Subtours.xlsx"
        df_final.to_excel(output_excel)
        print(f"\n¡Éxito! Comparativa guardada en {output_excel}")
    except FileNotFoundError:
        df_new.to_excel("outputs/Excel/Resultados_Subtours_Sin_TSV.xlsx")
        print("No se encontró el TSV, se guardó solo el resultado de los modelos actuales.")

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
         mode=args.mode)
