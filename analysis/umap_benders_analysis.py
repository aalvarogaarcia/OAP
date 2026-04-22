import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import json

# Ajusta el import a tu estructura
from utils.utils import load_farkas_logs

logger = logging.getLogger(__name__)

def extract_cut_features(logs: list[dict]) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Convierte el historial de Benders en una matriz matemática X leyendo las claves 
    exactas del JSON (ray_components y subproblem).
    """
    features = []
    iterations = []
    cut_types = []

    # 1. Averiguar todos los posibles arcos/variables (nombres aplanados)
    all_keys = set()
    for log in logs:
        # CORRECCIÓN 1: Leer 'ray_components' en lugar de 'v_components'
        ray_comps = log.get('ray_components', {})
        for block_name, block_data in ray_comps.items():
            if isinstance(block_data, dict):
                for k in block_data.keys():
                    # Guardamos el nombre: ej. "alpha_17_5"
                    all_keys.add(f"{block_name}_{k}")
            else:
                all_keys.add(block_name)
    
    # 2. Construir la matriz X aplanada
    all_keys = sorted(list(all_keys)) # Fijamos un orden estricto para las columnas
    
    for log in logs:
        iterations.append(log.get('iteration', 0))
        # CORRECCIÓN 2: Leer 'subproblem' en lugar de 'subproblem_type'
        cut_types.append(log.get('subproblem', 'Unknown'))
        
        ray_comps = log.get('ray_components', {})
        
        flat_duals = {}
        for block_name, block_data in ray_comps.items():
            if isinstance(block_data, dict):
                for k, v in block_data.items():
                    flat_duals[f"{block_name}_{k}"] = v
            else:
                flat_duals[block_name] = block_data 

        # Extraemos el vector en el mismo orden exacto que all_keys
        vector = [flat_duals.get(k, 0.0) for k in all_keys]
        features.append(vector)

    return np.array(features), iterations, cut_types

def plot_dimensionality_reduction(log_path: str, output_path: str = None):
    """
    Aplica PCA y UMAP a los vectores de cortes para visualizar el comportamiento de Benders.
    """
    logs = load_farkas_logs(log_path)
    if not logs:
        print(f"No se pudieron cargar logs de {log_path}")
        return

    print(f"Extrayendo características de {len(logs)} iteraciones...")
    X, iterations, cut_types = extract_cut_features(logs)
    
    if len(X) < 5:
        print("Muy pocas iteraciones para que UMAP/PCA tengan sentido.")
        return

    # Estandarizamos los datos (Media 0, Varianza 1) fundamental para ML
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 1. PCA (Proyección Lineal) ---
    # PCA nos mostrará las "direcciones principales" reales de los cortes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- 2. UMAP (Proyección No Lineal / Topológica) ---
    # UMAP nos mostrará "familias" o clusters de cortes similares
    reducer = umap.UMAP(n_neighbors=min(15, len(X)-1), min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # --- PLOTEAR ---
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2)
    
    # Colormap basado en la progresión temporal (Iteración)
    cmap = plt.cm.viridis

    # Subplot 1: PCA
    ax1 = fig.add_subplot(gs[0])
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=iterations, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
    ax1.set_title(f"PCA de los Cortes (Varianza explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
    ax1.set_xlabel("Componente Principal 1")
    ax1.set_ylabel("Componente Principal 2")

    # Subplot 2: UMAP
    ax2 = fig.add_subplot(gs[1])
    sc2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=iterations, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
    ax2.set_title("UMAP de los Cortes (Clustering / Topología local)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    # Añadir barra de color que indica la iteración
    cbar = plt.colorbar(sc2, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Número de Iteración (Evolución Temporal)')

    plt.suptitle(f"Análisis del Espacio de Cortes Benders (Dimensiones Originales: {X.shape[1]})", fontsize=16)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Gráfico UMAP guardado en {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Prueba con tu instancia de Londres
    # Asegúrate de poner la ruta correcta al JSON que generó tu BendersAnalysisMixin
    
    log_file = "outputs/Logs/london-20.json" 
    np.random.seed(42)  # Para reproducibilidad de UMAP
    plot_dimensionality_reduction(log_file, "outputs/images/UMAP_london-0000020.png")