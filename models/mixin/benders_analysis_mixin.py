import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import numpy as np
import networkx as nx
import json

from utils.utils import (
    load_farkas_logs, 
    plot_cut_heatmap, 
    plot_farkas_ray_network, 
    format_cut_string
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

    def analyze_cuts_from_benders(self, output_pdf: str | None = None) -> None:
        """
        Lee el historial de cortes desde el log del modelo y genera un reporte PDF visual,
        utilizando las coordenadas (points) almacenadas directamente en el modelo.
        """
        # 1. Validaciones usando los atributos del modelo
        if not hasattr(self, 'log_path') or not self.log_path:
            print(f"❌ [Instancia: {self.name}] Imposible analizar cortes. No hay 'log_path' definido.")
            return

        # Generar nombre automático si no se provee
        if output_pdf is None:
            output_pdf = f"Analisis_Cortes_{self.name}.pdf"

        # 2. Extraer posiciones DIRECTAMENTE del modelo (sin leer el archivo .instance)
        # model.points es un numpy array, lo convertimos a diccionario para NetworkX
        posiciones = {i: pt for i, pt in enumerate(self.points)}

        # 3. Leer logs del archivo asociado al modelo
        logs = []
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        logs.append(json.loads(line))
        except FileNotFoundError:
            print(f"❌ Error: Archivo de log {self.log_path} no encontrado.")
            return

        if not logs:
            print("⚠️ El archivo de logs está vacío.")
            return

        print(f"Generando reporte para {self.name}... ({len(logs)} cortes detectados)")

        # 4. Generación del PDF (Misma lógica visual de antes)
        with PdfPages(output_pdf) as pdf:
            for log in logs:
                fig, ax = plt.subplots(figsize=(10, 8))
                G = nx.DiGraph()
                G.add_nodes_from(posiciones.keys())

                cut_expr = log.get("cut_expr", {})
                coeffs = cut_expr.get("coeffs", {}) if cut_expr else {}
                active_x = log.get("active_x", {})
                sense = log.get("sense", "<=")
                rhs = -cut_expr.get("constant", 0.0) if cut_expr else 0.0

                # 1. Extraer arcos del Maestro (Gris punteado)
                master_edges = []
                for var_name, val in active_x.items():
                    # Split por '_' y tomamos los dos últimos elementos
                    # Funciona tanto para "0_1" como para "x_0_1"
                    partes = var_name.split('_')
                    u, v = int(partes[-2]), int(partes[-1])
                    master_edges.append((u, v))
                    G.add_edge(u, v)
                
                if master_edges:
                    nx.draw_networkx_edges(G, posiciones, edgelist=master_edges, 
                                           edge_color='lightgray', style='dashed', alpha=0.5, ax=ax)
    
                # 2. Clasificar y dibujar arcos del corte
                edges_pos, edges_neg = [], []
                weights_pos, weights_neg = [], []
    
                for var_name, coeff in coeffs.items():
                    partes = var_name.split('_')
                    u, v = int(partes[-2]), int(partes[-1])
                    G.add_edge(u, v)
                
                # Clasificamos según el signo del coeficiente
                if coeff > 0:
                    edges_pos.append((u, v))
                    weights_pos.append(abs(coeff))
                elif coeff < 0:
                    edges_neg.append((u, v))
                    weights_neg.append(abs(coeff))

                nx.draw_networkx_nodes(G, posiciones, node_size=600, node_color='lightblue', edgecolors='black', ax=ax)
                nx.draw_networkx_labels(G, posiciones, font_size=11, font_weight='bold', ax=ax)

                if edges_pos:
                    nx.draw_networkx_edges(G, posiciones, edgelist=edges_pos, 
                                           edge_color='green', width=[w * 1.5 for w in weights_pos], 
                                           arrowsize=15, ax=ax)
                if edges_neg:
                    nx.draw_networkx_edges(G, posiciones, edgelist=edges_neg, 
                                           edge_color='red', width=[w * 1.5 for w in weights_neg], 
                                           arrowsize=15, ax=ax)

                # Títulos y metadatos
                ax.set_title(f"Iter: {log.get('iteration')} | Sub: {log.get('subproblem')} | Violación: {log.get('violation'):.4f}", fontweight='bold')

                leyenda = (
                    "Leyenda:\n"
                    "-- Gris punteado: x_bar (Maestro)\n"
                    "-> Verde: Coeff Positivo (<=)\n"
                    "-> Rojo: Coeff Negativo (>=)\n"
                    f"Restricción: {sense} {rhs}"
                )
                ax.text(0.02, 0.02, leyenda, transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)

        print(f"✅ ¡Reporte guardado en {output_pdf}!")