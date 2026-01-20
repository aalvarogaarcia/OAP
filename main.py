from src import *
import glob
import os
import time
from numbers import Real
import argparse

def get_model_stats(model, relaxed_model):
    """
    Extrae estadísticas clave del modelo y su relajación.
    Retorna: (LP_Val, Gap, IP_Val, Time, Nodes)
    """

    if model and model.SolCount > 0:
        ip_val = model.ObjVal
        time_s = model.Runtime
        nodes = model.NodeCount
    else:
        return "-", "-", "-", "-", "-"

    if relaxed_model and relaxed_model.SolCount > 0:
        lp_val = relaxed_model.ObjVal
    else:
        lp_val = 0 # O manejar error

    # Calcular Gap: (IP - LP )/ IP * 100 if MinArea (evitando división por cero)
    # Si MaxArea, el gap es (LP - IP) / (Area(CH)-IP) * 100
    gap = 0.0
    if ip_val != 0 and model.ModelSense == GRB.MINIMIZE:
        gap = (ip_val - lp_val) / ip_val * 100
    
    elif ip_val != 0 and model.ModelSense == GRB.MAXIMIZE:
        # Asumimos que el área del casco convexo es accesible como atributo
        area_ch = model._convex_hull_area if hasattr(model, '_convex_hull_area') else None
        print(f"Convex Hull Area for gap calculation: {area_ch}")
        if area_ch is not None and (area_ch - ip_val) != 0:
            gap = (lp_val - ip_val) / (area_ch - ip_val) * 100
    
    return lp_val, gap, ip_val, time_s, nodes




def main(dir_path="data", 
         ext="*.txt", 
         sum_constrain=True,
         time_limit=7200000):
    files = glob.glob(os.path.join(dir_path, ext))
    files.sort() # Ordenar para que la tabla salga ordenada
    
    # Generate filename based on flags
    flag_suffix = ""
    if not sum_constrain:
        flag_suffix += "_nosum"
    flag_suffix += f"_tl{time_limit}" if time_limit != 7200000 else ""
    
    output_filename = f"outputs/LaTex/resultados_tabla{flag_suffix}.tex"
    
    print(f"Iniciando proceso. Los resultados se guardarán en: {output_filename}")
    print(f"Flags: sum_constrain={sum_constrain}, time_limit={time_limit}")

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
            modMax = build_and_solve_model(instance_path=file, 
                                          verbose=False, 
                                          plot=False, 
                                          time_limit=time_limit, 
                                          maximize=True,
                                          sum_constrain=sum_constrain)
            modMax_relax = modMax.relax()
            modMax_relax.optimize() 

            max_lp, max_gap, max_ip, max_time, max_nodes = get_model_stats(modMax, modMax_relax)
            print(f"  Resultados MAXAREA: LP={max_lp}, Gap={max_gap}%, IP={max_ip}, Time={max_time}s, Nodes={max_nodes}")
            # Datos estructurales
            cols = modMax.NumVars
            rows = modMax.NumConstrs
            
            # --- MINIMIZATION ---
            modMin = build_and_solve_model(instance_path=file, 
                                          verbose=False, 
                                          plot=False, 
                                          time_limit=time_limit, 
                                          maximize=False,
                                          sum_constrain=sum_constrain)
            
            modMin_relax = modMin.relax()
            modMin_relax.optimize()

            min_lp, min_gap, min_ip, min_time, min_nodes = get_model_stats(modMin, modMin_relax)
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

            else:
                row_str = (
                    f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                    f"- & - & - & {min_time:.2f} & {min_nodes} & "
                    f"- & - & - & {max_time:.2f} & {max_nodes} \\\\"
                )
                f.write(row_str + "\n")

        # 3. Escribir Footer de LaTeX al terminar el bucle
        f.write(r"""
\bottomrule
\end{tabular}
}
\end{table}
                
\end{frame}

\end{document}          
""")

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
                       default=7200000,
                       dest='time_limit',
                       help='Time limit in milliseconds (default: 7200000 = 2 hours)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(dir_path=args.dir_path,
         ext=args.ext,
         sum_constrain=args.sum_constrain,
         time_limit=args.time_limit)