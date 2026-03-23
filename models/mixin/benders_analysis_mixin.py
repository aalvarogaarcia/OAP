import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import logging
import numpy as np

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
                    texto_corte = format_cut_string(log_entry["cut_expr"])

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
                plt.figure(figsize=(10, 8))
                
                # Llamamos a TU función de Heatmap
                # (Importante: usamos self.N que viene del modelo)
                plot_cut_heatmap(log_entry, num_nodes=self.N, save_path=None, show_plot=False)
                
                # Guardamos y cerramos
                pdf.savefig(bbox_inches='tight')
                plt.close()

                # Imprimir progreso
                if (i + 1) % 10 == 0 or (i + 1) == len(logs):
                    logger.info(f"Procesados {i + 1}/{len(logs)} cortes...")

        logger.info(f"✅ ¡Reporte generado con éxito en: {output_pdf_path}!")