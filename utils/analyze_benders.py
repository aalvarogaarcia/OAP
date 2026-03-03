import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Importamos las funciones desde tu módulo utils
from utils.farkas_viz import load_farkas_logs, plot_cut_heatmap, plot_cut_weights, plot_farkas_ray_network
from utils.utils import format_cut_string

def generate_post_mortem_report(log_filepath: str, output_pdf_path: str, points: dict = None, n:int = 10):
    """
    Lee un archivo JSONL con el historial de Farkas y genera un PDF multipágina.
    """
    print(f"Cargando registros desde: {log_filepath}...")
    logs = load_farkas_logs(log_filepath)
    
    if not logs:
        print("No se encontraron registros o el archivo no existe.")
        return

    print(f"Se encontraron {len(logs)} iteraciones infactibles. Generando PDF...")
    
    # Crear la carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Abrir el manejador de PDF multipágina
    with PdfPages(output_pdf_path) as pdf:
        for i, log_entry in enumerate(logs):

            # --- VISUALIZACIÓN DE RAYOS DE FARKAS (GRAFO)---

            # Creamos una figura nueva
            plt.figure(figsize=(10, 8))
            
            # Llamamos a la función de dibujo indicando que NO la muestre en pantalla
            plot_farkas_ray_network(log_entry, points=points, save_path=None, show_plot=False)
            
            if "cut_expr" in log_entry:
                # --- VISUALIZACIÓN DE PESOS DE CORTE (BARRAS) ---
                texto_corte = format_cut_string(log_entry["cut_expr"])

                # Colocamos el texto justo debajo del eje X (y = -0.1)
                plt.text(0.5, -0.05, texto_corte, 
                        transform=plt.gca().transAxes, 
                        fontsize=10, 
                        ha='center', 
                        va='top',
                        style='italic',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


            # Guardamos la figura actual como una nueva página en el PDF
            pdf.savefig(bbox_inches='tight')
            
            # Importante: cerramos la figura para liberar memoria RAM
            plt.close()
            


            
            # --- VISUALIZACIÓN DE MATRIZ DE RESTRICCIONES (HEATMAP) ---

            # Creamos una figura nueva
            plt.figure(figsize=(10, 8))
            
            # Llamamos a la función de dibujo indicando que NO la muestre en pantalla
            plot_cut_heatmap(log_entry, num_nodes= n, save_path=None, show_plot=False)  # Ajusta num_nodes según tu instancia
            
            # Guardamos la figura actual como una nueva página en el PDF
            pdf.savefig(bbox_inches='tight')
            
            # Importante: cerramos la figura para liberar memoria RAM
            plt.close()
            
            # Imprimir progreso
            if (i + 1) % 10 == 0 or (i + 1) == len(logs):
                print(f"Procesados {i + 1}/{len(logs)} cortes...")
                
    print(f"\n¡Reporte generado con éxito!\nPuedes revisarlo en: {output_pdf_path}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURACIÓN
    # ---------------------------------------------------------
    INSTANCE_NAME = "cut-yp_stars-0000010"  # Cambia esto al nombre de tu instancia
    
    LOG_PATH = f"outputs/Others/Benders/{INSTANCE_NAME}/farkas_log.jsonl"
    PDF_PATH = f"outputs/Others/Benders/{INSTANCE_NAME}/Farkas_Analysis_Report.pdf"
    
    # (OPCIONAL PERO MUY RECOMENDADO): Pasa las coordenadas reales de tus nodos.
    # Si las tienes, el grafo siempre mantendrá la misma forma y será más fácil ver
    # cómo cambian los caminos. Si es None, NetworkX inventará las posiciones.
    # Ejemplo: NODO_COORDENADAS = {0: (10.5, 20.1), 1: (15.0, 18.2), ...}
    NODO_COORDENADAS = None 
    
    # ---------------------------------------------------------
    # EJECUCIÓN
    # ---------------------------------------------------------
    generate_post_mortem_report(LOG_PATH, PDF_PATH, points=NODO_COORDENADAS)