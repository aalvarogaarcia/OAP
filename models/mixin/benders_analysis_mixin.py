import itertools
import json
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from utils.utils import (
    format_cut_string,
    load_farkas_logs,
    plot_farkas_ray_network,
)

logger = logging.getLogger(__name__)

class BendersAnalysisMixin:
    """
    Mixin que dota a los modelos Benders de la capacidad de auto-analizar
    sus logs de cortes integrando las funciones gráficas originales (Post-Mortem).
    """
    points: np.ndarray
    N: int
    name: str
    log_path: str | None

    def generate_benders_report(self, output_pdf_path: str | None = None) -> None:
        """
        Lee el archivo JSONL con el historial iterativo y genera un PDF multipágina
        usando las funciones originales (2 páginas por iteración: Grafo y Heatmap).
        """
        # 1. Comprobaciones de seguridad
        if not hasattr(self, 'log_path') or not self.log_path:
            logger.warning(f"❌ [Instancia: {self.name}] Imposible generar reporte. Faltó 'save_cuts=True'.")
            return

        if not os.path.exists(self.log_path):
            logger.warning(f"❌ [Instancia: {self.name}] El archivo de log no existe en {self.log_path}.")
            return

        # 2. Cargar logs
        logger.info(f"Cargando registros desde: {self.log_path}...")
        logs = load_farkas_logs(self.log_path)
        
        if not logs:
            logger.warning("No se encontraron registros o el archivo está vacío.")
            return

        logger.info(f"Se encontraron {len(logs)} iteraciones. Generando PDF...")

        if output_pdf_path is None:
            output_pdf_path = f"outputs/Analysis/PostMortem_{self.name}.pdf"

        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

        # 3. Bucle de generación (¡El tuyo original!)
        with PdfPages(output_pdf_path) as pdf:
            for i, log_entry in enumerate(logs):

                # ========================================================
                # PÁGINA 1: VISUALIZACIÓN DE RAYOS DE FARKAS (GRAFO)
                # ========================================================
                plt.figure(figsize=(10, 8))
                
                # Llamamos a TU función indicando que NO la muestre en pantalla
                # (Importante: usamos self.points)
                plot_farkas_ray_network(log_entry, points=self.points, save_path=None, show_plot=False)
                
                if "cut_expr" in log_entry:
                    # VISUALIZACIÓN DE PESOS DE CORTE (TEXTO)
                    texto_corte = format_cut_string(log_entry["cut_expr"], sense = log_entry["sense"])

                    # Colocamos el texto justo debajo del eje X
                    plt.text(0.5, -0.05, texto_corte, 
                            transform=plt.gca().transAxes, 
                            fontsize=10, 
                            ha='center', 
                            va='top',
                            style='italic',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

                # Guardamos y cerramos
                pdf.savefig(bbox_inches='tight')
                plt.close('all')

                # ========================================================
                # PÁGINA 2: VISUALIZACIÓN DE MATRIZ DE RESTRICCIONES (HEATMAP)
                # ========================================================
                #plt.figure(figsize=(10, 8))
                
                # Llamamos a TU función de Heatmap
                # (Importante: usamos self.N que viene del modelo)
                #plot_cut_heatmap(log_entry, num_nodes=self.N, save_path=None, show_plot=False)
                
                # Guardamos y cerramos
                #pdf.savefig(bbox_inches='tight')
                #plt.close()

                # Imprimir progreso
                if (i + 1) % 10 == 0 or (i + 1) == len(logs):
                    logger.info(f"Procesados {i + 1}/{len(logs)} cortes...")

        logger.info(f"✅ ¡Reporte generado con éxito en: {output_pdf_path}!")

    def generate_combinatorial_report(self, output_pdf: str | None = None, max_vars: int = 15, top_k_cases: int = 100) -> None:
        """
        Lee el log de cortes y, para cada uno, evalúa las combinaciones 0/1 válidas 
        (sin cruces, grado <= 1) que satisfacen la restricción. Guarda los mejores casos en PDF.
        """
        if not hasattr(self, 'log_path') or not self.log_path:
            print("❌ No hay log_path definido.")
            return

        output_pdf_path = output_pdf or f"outputs/analysis/Casos_Validos_{self.name}.pdf"
        posiciones = {i: pt for i, pt in enumerate(self.points)}
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        logs = []
        with open(self.log_path, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        print(f"🔍 Analizando combinaciones para {len(logs)} cortes...")

        with PdfPages(output_pdf_path) as pdf:
            for i, log in enumerate(logs):
                cut_expr = log.get("cut_expr", {})
                if not cut_expr:
                    continue
                
                coeffs = cut_expr.get("coeffs", {})
                constant = cut_expr.get("constant", 0.0)
                sense = log.get("sense", "<=")
                
                # Nombres de variables (ej. 'x_1_3' o '1_3') asegurando que funcionen
                all_vars = list(coeffs.keys())
                
                if len(all_vars) > max_vars:
                    print(f"⚠️ Corte {i+1} omitido: Tiene {len(all_vars)} variables (> límite {max_vars} para 2^n).")
                    continue
                
                all_arcs = []
                for v in all_vars:
                    partes = v.split('_')
                    all_arcs.append((int(partes[-2]), int(partes[-1])))

                # ==========================================================
                # 1. REFORMULACIÓN DE LA INECUACIÓN (Todo positivo: L >= R + C)
                # ==========================================================
                vars_left, vars_right = [], []
                weights_left, weights_right = {}, {}
                
                if sense == ">=":
                    constant_right = constant
                    for v, c in coeffs.items():
                        if c > 0:
                            vars_left.append(v)
                            weights_left[v] = c
                        elif c < 0:
                            vars_right.append(v)
                            weights_right[v] = abs(c)
                else: # sense == "<="
                    constant_right = -constant
                    for v, c in coeffs.items():
                        if c < 0:
                            vars_left.append(v)
                            weights_left[v] = abs(c)
                        elif c > 0:
                            vars_right.append(v)
                            weights_right[v] = c

                valid_cases = []
                
                # ==========================================================
                # 2. BÚSQUEDA COMBINATORIA
                # ==========================================================
                for comb in itertools.product([0, 1], repeat=len(all_vars)):
                    if sum(comb) == 0:
                        continue
                    
                    d = dict(zip(all_vars, comb, strict=False))
                    active_arcs = [all_arcs[idx] for idx, val in enumerate(comb) if val == 1]
                    
                    # Filtro Antisimetría
                    active_set = set(active_arcs)
                    if any((v, u) in active_set for u, v in active_arcs):
                        continue
                        
                    # Filtro Grado (Máx 1 in/out)
                    in_degree, out_degree = {}, {}
                    invalido = False
                    for u, v in active_arcs:
                        out_degree[u] = out_degree.get(u, 0) + 1
                        in_degree[v] = in_degree.get(v, 0) + 1
                        if out_degree[u] > 1 or in_degree[v] > 1:
                            invalido = True
                            break
                    if invalido:
                        continue
                        
                    # Filtro Cruces
                    from utils.utils import segments_intersect
                    has_intersect = False
                    for e1, e2 in itertools.combinations(active_arcs, 2):
                        if segments_intersect(self.points[e1[0]], self.points[e1[1]], 
                                              self.points[e2[0]], self.points[e2[1]]):
                            has_intersect = True
                            break
                    if has_intersect:
                        continue

                    # ==========================================================
                    # 3. FILTRO ESTRICTO DE LADOS Y EVALUACIÓN L >= R + C
                    # ==========================================================
                    activas_izq = [v for v in vars_left if d[v] == 1]
                    activas_der = [v for v in vars_right if d[v] == 1]
                    
                    # 🚨 LA CLAVE: Ignorar si no hay interacción (casos triviales o =0)
                    if not activas_izq or not activas_der:
                        continue
                        
                    sum_l = sum(weights_left[v] for v in activas_izq)
                    sum_r = sum(weights_right[v] for v in activas_der)
                    
                    # Evaluar desigualdad usando tolerancia para flotantes
                    if sum_l >= (sum_r + constant_right) - 1e-5:
                        valid_cases.append((d, active_arcs, sum_l, sum_r, activas_izq, activas_der))

                if not valid_cases:
                    continue
                
                # Ordenar priorizando maximizar el lado derecho y luego el izquierdo
                valid_cases.sort(key=lambda x: (x[3], x[2]), reverse=True)
                
                # ==========================================================
                # 4. DIBUJADO DE LOS TOP CASOS
                # ==========================================================
                for j, (case_dict, active_edges, s_l, s_r, act_izq, act_der) in enumerate(valid_cases[:top_k_cases], 1):
                    fig, ax = plt.subplots(figsize=(10, 7))
                    G = nx.DiGraph()
                    G.add_nodes_from(posiciones.keys())

                    inactive_edges = [all_arcs[idx] for idx, v in enumerate(all_vars) if case_dict[v] == 0]

                    nx.draw_networkx_nodes(G, posiciones, node_size=600, node_color='lightblue', edgecolors='black', ax=ax)
                    nx.draw_networkx_labels(G, posiciones, font_size=11, font_weight='bold', ax=ax)
                    
                    nx.draw_networkx_edges(G, posiciones, edgelist=active_edges, edge_color='green', width=3, arrowsize=20, ax=ax)
                    nx.draw_networkx_edges(G, posiciones, edgelist=inactive_edges, edge_color='grey', width=1, style='dashed', alpha=0.3, ax=ax)
                    
                    # Construir los strings de la ecuación para el título
                    str_izq = " + ".join([f"{weights_left[v]}*{v}" for v in act_izq])
                    str_der = " + ".join([f"{weights_right[v]}*{v}" for v in act_der])
                    str_const = f" + {constant_right:.2f}" if constant_right > 0 else (f" - {abs(constant_right):.2f}" if constant_right < 0 else "")
                    
                    titulo = (f"Corte {i+1} (Iter {log['iteration']}) | Caso {j}/{len(valid_cases)}\n"
                              f"[{str_izq}] >= [{str_der}]{str_const}\n"
                              f"Sumas: {s_l:.2f} >= {(s_r + constant_right):.2f}")
                    
                    ax.set_title(titulo, fontsize=10, fontweight='bold', pad=15)
                    plt.axis('off')
                    
                    pdf.savefig(fig)
                    plt.close(fig)