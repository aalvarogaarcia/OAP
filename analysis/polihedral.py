import numpy as np
import cdd
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import gurobipy as gp
from gurobipy import GRB
import logging

# Ajusta los imports a tu proyecto
from models.OAPCompactModel import OAPCompactModel
from utils.utils import compute_triangles

logger = logging.getLogger(__name__)

def generate_toy_instance(num_internal_points: int = 1) -> np.ndarray:
    """Genera una instancia minúscula: Un triángulo grande y N puntos internos."""
    points = [[0, 0], [100, 0], [50, 100]]
    if num_internal_points >= 1: points.append([50, 30])
    if num_internal_points >= 2: points.append([50, 60])
    if num_internal_points >= 3: points.append([50, 90])
    if num_internal_points >= 4: points.append([30, 50])
    if num_internal_points >= 5: points.append([70, 50])
    return np.array(points)

def extract_and_plot_3d_slice(
    model: gp.Model, 
    free_var_names: tuple[str, str, str], 
    ax_3d: plt.Axes, 
    ax_text: plt.Axes
) -> bool:
    """
    Calcula el poliedro 3D, lo dibuja en ax_3d y escribe las restricciones activas en ax_text.
    Devuelve True si se pudo dibujar algo, False si el espacio estaba vacío.
    """
    free_vars = []
    for var_name in free_var_names:
        v = model.getVarByName(var_name)
        if v is None: return False
        free_vars.append(v)

    cdd_matrix_data = []
    lin_set = set()
    row_idx = 0

    # 1. Convertir Restricciones de Gurobi
    for constr in model.getConstrs():
        expr = model.getRow(constr)
        sense = constr.Sense
        rhs = constr.RHS
        
        fixed_val = 0.0
        coeffs = [0.0, 0.0, 0.0]
        
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            
            if var.VarName == free_var_names[0]: coeffs[0] += coeff
            elif var.VarName == free_var_names[1]: coeffs[1] += coeff
            elif var.VarName == free_var_names[2]: coeffs[2] += coeff
            else: fixed_val += coeff * var.X
                
        b_prime = rhs - fixed_val
        
        # Formato CDD: b - Ax >= 0
        if sense == GRB.LESS_EQUAL:
            cdd_matrix_data.append([b_prime, -coeffs[0], -coeffs[1], -coeffs[2]])
            row_idx += 1
        elif sense == GRB.GREATER_EQUAL:
            cdd_matrix_data.append([-b_prime, coeffs[0], coeffs[1], coeffs[2]])
            row_idx += 1
        elif sense == GRB.EQUAL:
            cdd_matrix_data.append([b_prime, -coeffs[0], -coeffs[1], -coeffs[2]])
            lin_set.add(row_idx)
            row_idx += 1

    # 2. Convertir Límites (Bounds)
    for i, v in enumerate(free_vars):
        row_lb = [0.0, 0.0, 0.0, 0.0]
        row_lb[0] = -v.LB
        row_lb[i+1] = 1.0
        cdd_matrix_data.append(row_lb)
        row_idx += 1
        
        row_ub = [0.0, 0.0, 0.0, 0.0]
        row_ub[0] = v.UB
        row_ub[i+1] = -1.0
        cdd_matrix_data.append(row_ub)
        row_idx += 1

    # 3. Operaciones CDD
    mat = cdd.Matrix(cdd_matrix_data, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.lin_set = frozenset(lin_set)
    mat.canonicalize() # ¡CRUCIAL! Esto elimina las redundancias y deja solo las caras del poliedro

    poly = cdd.Polyhedron(mat)
    vertices_raw = poly.get_generators()
    
    vertices_3d = []
    for v in vertices_raw:
        if v[0] == 1.0: # Es un vértice
            vertices_3d.append([v[1], v[2], v[3]])
    vertices_3d = np.array(vertices_3d)

    # Si no hay espacio factible, abortamos
    if len(vertices_3d) == 0:
        return False

    # 4. Formatear y Mostrar las Restricciones (Matriz Canónica)
    restricciones_texto = f"Restricciones Activas (Caras del Poliedro):\n{'-'*45}\n"
    for i in range(mat.row_size):
        row = mat[i]
        b, a1, a2, a3 = row[0], -row[1], -row[2], -row[3]
        
        # Ignorar restricciones triviales (0 <= 0)
        if abs(a1) < 1e-5 and abs(a2) < 1e-5 and abs(a3) < 1e-5: continue
            
        signo = "==" if i in mat.lin_set else "<="
        
        eq = ""
        if abs(a1) > 1e-5: eq += f"{a1:+.2f}*x1 "
        if abs(a2) > 1e-5: eq += f"{a2:+.2f}*x2 "
        if abs(a3) > 1e-5: eq += f"{a3:+.2f}*x3 "
        
        restricciones_texto += f"{eq.strip()} {signo} {b:.2f}\n"

    ax_text.text(0.0, 1.0, restricciones_texto, fontsize=9, family='monospace', va='top')
    ax_text.axis('off')

    # 5. Dibujar el Poliedro 3D
    if len(vertices_3d) >= 4:
        try:
            hull = ConvexHull(vertices_3d)
            for s in hull.simplices:
                s = np.append(s, s[0])
                ax_3d.plot(vertices_3d[s, 0], vertices_3d[s, 1], vertices_3d[s, 2], "k-", alpha=0.8, linewidth=1.5)
            
            # Colorear caras (estas caras corresponden a las restricciones de la izquierda)
            faces = Poly3DCollection([vertices_3d[s] for s in hull.simplices], alpha=0.3, facecolors='cyan')
            ax_3d.add_collection3d(faces)
        except Exception:
            # Si son coplanarios (es un plano 2D flotando en 3D)
            ax_3d.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], color='blue', s=50)
    else:
        # Si es un segmento o un punto
        ax_3d.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], color='blue', s=50)

    # Dibujar el punto óptimo
    opt_val = [v.X for v in free_vars]
    ax_3d.scatter(*opt_val, color='red', s=100, label="Solución Óptima", zorder=10)

    ax_3d.set_xlabel(free_var_names[0])
    ax_3d.set_ylabel(free_var_names[1])
    ax_3d.set_zlabel(free_var_names[2])
    ax_3d.legend()

    return True

def generate_polyhedral_report(model: gp.Model, output_pdf: str):
    """
    Genera un PDF iterando sobre todas las combinaciones de 3 variables 'x'.
    """
    # 1. Filtramos para usar SOLO las variables 'x' (arcos de enrutamiento)
    # Si metemos variables auxiliares, el PDF tendrá miles de páginas
    x_vars = [v.VarName for v in model.getVars() if v.VarName.startswith('x')]
    
    # Generamos todas las combinaciones posibles de a 3
    combinaciones = list(itertools.combinations(x_vars, 3))
    
    print(f"Generando {len(combinaciones)} combinaciones posibles...")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    exitosos = 0
    with PdfPages(output_pdf) as pdf:
        for i, comb in enumerate(combinaciones):
            fig = plt.figure(figsize=(14, 8))
            gs = GridSpec(1, 2, width_ratios=[1, 2])
            
            ax_text = fig.add_subplot(gs[0])
            ax_3d = fig.add_subplot(gs[1], projection='3d')
            
            # Intentar extraer y dibujar
            valido = extract_and_plot_3d_slice(model, comb, ax_3d, ax_text)
            
            if valido:
                plt.suptitle(f"Espacio de Factibilidad LP para: {comb[0]}, {comb[1]}, {comb[2]}")
                pdf.savefig(fig, bbox_inches='tight')
                exitosos += 1
                
            plt.close(fig)
            
            if (i+1) % 20 == 0:
                print(f"Procesadas {i+1}/{len(combinaciones)} combinaciones...")

    print(f"✅ Reporte guardado en {output_pdf} ({exitosos} poliedros válidos generados)")

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    points = generate_toy_instance(num_internal_points=2)
    triangles = compute_triangles(points)
    
    modelo = OAPCompactModel(points, triangles, name="Toy_Polyhedron")
    modelo.build(objective="Fekete", mode=1, maximize=False, sum_constrain=True) 
    modelo.solve(verbose=False)
    
    output_path = "outputs/Analysis/Polihedral_All_Combinations.pdf"
    generate_polyhedral_report(modelo.model, output_path)