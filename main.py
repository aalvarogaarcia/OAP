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
         mode=0):
    files = glob.glob(os.path.join(dir_path, ext))
    files.sort() # Ordenar para que la tabla salga ordenada
    
    # Generate filename based on flags
    flag_suffix = ""
    if not sum_constrain:
        flag_suffix += "_nosum"
    flag_suffix += f"_tl{time_limit}" if time_limit != 7200 else ""
    flag_suffix += ext.replace("*", "").split(".")[0] if ext != "*.txt" else ""
    if obj is not None:
        flag_suffix += f"_obj{obj}_mode{mode}"
    
    output_filename = f"outputs/LaTex/resultados_tabla{flag_suffix}.tex"
    output_excel_filename = f"outputs/Excel/resultados_tabla{flag_suffix}.xlsx"


    print(f"Iniciando proceso. Los resultados se guardarán en: {output_filename}")
    print(f"Flags: sum_constrain={sum_constrain}, time_limit={time_limit}")

    data_list = []
    # 'w' abre el archivo para escritura (write). encoding='utf-8' es importante para LaTeX.
    with open(output_filename, "w", encoding="utf-8") as f:
        
        # 1. Escribir Cabecera de LaTeX en el archivo
        # Usamos f.write() para escribir en el archivo
        f.write(r"""
\documentclass{beamer}					% Document class

\usepackage[english]{babel}				% Set language
\usepackage[utf8x]{inputenc}			% Set encoding
\usepackage{multicol}
\usepackage{multirow}
\mode<presentation>						% Set options
{
  \usetheme{default}					% Set theme
  \usecolortheme{default} 				% Set colors
  \usefonttheme{default}  				% Set font theme
  \setbeamertemplate{caption}[numbered]	% Set caption to be numbered
}

% Uncomment this to have the outline at the beginning of each section highlighted.
%\AtBeginSection[]
%{
%  \begin{frame}{Outline}
%    \tableofcontents[currentsection]
%  \end{frame}
%}

\usepackage{graphicx}					% For including figures
\usepackage{booktabs}					% For table rules
\usepackage{hyperref}					% For cross-referencing

\title{Title for a minimal beamer presentation}	% Presentation title
\author{Author One}								% Presentation author
\institute{Name of institution}					% Author affiliation
\date{\today}									% Today's date	

\begin{document}

% Title page
% This page includes the informations defined earlier including title, author/s, affiliation/s and the date
\begin{frame}

\begin{table}[htbp]
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
            instance_name = filename.replace(".txt", "").replace("_", "-")
            
            # Intento de extraer tamaño N del nombre
            try:
                size_n = ''.join(filter(str.isdigit, instance_name))
            except:
                size_n = "-"

            try:
                points = read_indexed_instance(file)
                conv_hull_area = compute_convex_hull_area(points)
            except Exception:
                conv_hull_area = "-"
            conv_hull_area_str = f"{conv_hull_area:.2f}" if isinstance(conv_hull_area, Real) else str(conv_hull_area)

            # --- MAXIMIZATION ---
            if obj is not None:
                modMax = build_and_solve_model(instance_path=file, 
                                              verbose=False, 
                                              plot=False, 
                                              time_limit=time_limit, 
                                              maximize=True,
                                              sum_constrain=sum_constrain,
                                              obj = obj,
                                              mode = mode)
            else:
                modMax = build_and_solve_model(instance_path=file, 
                                          verbose=False, 
                                          plot=False, 
                                          time_limit=time_limit, 
                                          maximize=True,
                                          sum_constrain=sum_constrain,
                                          obj = 1,
                                          mode = mode)

            max_lp, max_gap, max_ip, max_time, max_nodes = get_model_stats(modMax)
            print(f"  Resultados MAXAREA: LP={max_lp}, Gap={max_gap}%, IP={max_ip}, Time={max_time}s, Nodes={max_nodes}")
            # Datos estructurales
            cols = modMax.NumVars
            rows = modMax.NumConstrs
            
            # --- MINIMIZATION ---
            if obj is not None:
                modMin = build_and_solve_model(instance_path=file, 
                                              verbose=False, 
                                              plot=False, 
                                              time_limit=time_limit, 
                                              maximize=False,
                                              sum_constrain=sum_constrain,
                                              obj = obj,
                                              mode = mode)
            else:
                modMin = build_and_solve_model(instance_path=file, 
                                          verbose=False, 
                                          plot=False, 
                                          time_limit=time_limit, 
                                          maximize=False,
                                          sum_constrain=sum_constrain,
                                          obj = 2,
                                          mode = mode)

            min_lp, min_gap, min_ip, min_time, min_nodes = get_model_stats(modMin)
            print(f"  Resultados MINAREA: LP={min_lp}, Gap={min_gap}%, IP={min_ip}, Time={min_time}s, Nodes={min_nodes}")

            # --- CREAR LA LÍNEA DE DATOS ---
            # Usamos sintaxis LaTeX para separar celdas con & y terminar línea con \\
            if modMax.Status == 2 and modMin.Status == 2:
                
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"{min_lp:.2f} & {min_gap:.2f} & {min_ip:.2f} & {min_time:.2f} & {min_nodes} & "
                    f"{max_lp:.2f} & {max_gap:.2f} & {max_ip:.2f} & {max_time:.2f} & {max_nodes} \\\\"
                )
                
                # Escribir la línea en el archivo
                f.write(row_str + "\n")

                new_row = {
                    "Instance": instance_name,
                    "Objective function": obj,
                    "Size": size_n,
                    "Convex Hull": conv_hull_area,
                    "Cols": cols,
                    "Rows": rows,
                    "Min LP Val": min_lp,
                    "Min Gap (%)": min_gap,
                    "Min IP Val": min_ip,
                    "Min Time": min_time,
                    "Min Nodes": min_nodes,
                    "Max LP Val": max_lp,
                    "Max Gap (%)": max_gap,
                    "Max IP Val": max_ip,
                    "Max Time": max_time,
                    "Max Nodes": max_nodes
                }

            elif modMin.Status == 2 and modMax.Status != 2:
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"{min_lp:.2f} & {min_gap:.2f} & {min_ip:.2f} & {min_time:.2f} & {min_nodes} & "
                    f"- & - & - & {max_time:.2f} & {max_nodes} \\\\"
                )
                f.write(row_str + "\n")

                new_row = {
                    "Instance": instance_name,
                    "Objective function": obj,
                    "Size": size_n,
                    "Convex Hull": conv_hull_area,
                    "Cols": cols,
                    "Rows": rows,
                    "Min LP Val": min_lp,
                    "Min Gap (%)": min_gap,
                    "Min IP Val": min_ip,
                    "Min Time": min_time,
                    "Min Nodes": min_nodes,
                    "Max LP Val": '-',
                    "Max Gap (%)": '-',
                    "Max IP Val": '-',
                    "Max Time": max_time,
                    "Max Nodes": max_nodes
                }

            elif modMin.Status != 2 and modMax.Status == 2:
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"- & - & - & {min_time:.2f} & {min_nodes} & "
                    f"{max_lp:.2f} & {max_gap:.2f} & {max_ip:.2f} & {max_time:.2f} & {max_nodes} \\\\"
                )
                f.write(row_str + "\n")

                new_row = {
                    "Instance": instance_name,
                    "Objective function": obj,
                    "Size": size_n,
                    "Convex Hull": conv_hull_area,
                    "Cols": cols,
                    "Rows": rows,
                    "Min LP Val": '-',
                    "Min Gap (%)": '-',
                    "Min IP Val": '-',
                    "Min Time": min_time,
                    "Min Nodes": min_nodes,
                    "Max LP Val": max_lp,
                    "Max Gap (%)": max_gap,
                    "Max IP Val": max_ip,
                    "Max Time": max_time,
                    "Max Nodes": max_nodes
                }
                
                

            else:
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"- & - & - & {min_time:.2f} & {min_nodes} & "
                    f"- & - & - & {max_time:.2f} & {max_nodes} \\\\"
                )
                f.write(row_str + "\n")

                new_row = {
                    "Instance": instance_name,
                    "Objective function": obj,
                    "Size": size_n,
                    "Convex Hull": conv_hull_area,
                    "Cols": cols,
                    "Rows": rows,
                    "Min LP Val": '-',
                    "Min Gap (%)": '-',
                    "Min IP Val": '-',
                    "Min Time": min_time,
                    "Min Nodes": min_nodes,
                    "Max LP Val": '-',
                    "Max Gap (%)": '-',
                    "Max IP Val": '-',
                    "Max Time": max_time,
                    "Max Nodes": max_nodes
                }

            # 3. Añádelo a la lista
            data_list.append(new_row)
        # 3. Escribir Footer de LaTeX al terminar el bucle
        f.write(r"""
\bottomrule
\end{tabular}
}
\end{table}
                
\end{frame}

\end{document}          
""")
    # 4. Crear DataFrame y exportar a Excel
    df = pd.DataFrame(data_list)
    df.to_excel(output_excel_filename, index=False)
    print(f"\n¡Éxito! Archivo '{output_filename}' generado correctamente.")

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
                       help='Time limit in seconds (default: 7200 = 2 hours)')
    
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