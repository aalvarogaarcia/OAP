from gurobipy import GRB
import cdd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from models.gurobi import build_and_solve_model

def extract_3d_slice_from_gurobi(model: gp.Model, free_var_names: list[str]) -> list[list[float]]:
    """
    Reduce un modelo de n dimensiones a un poliedro 3D, fijando todas las variables 
    (excepto las 3 elegidas) a sus valores óptimos de la solución de Gurobi.
    """
    model.update()
    vars = model.getVars()
    
    # Identificar cuáles son las 3 variables libres y guardar los valores del resto
    free_indices: list[int] = []
    fixed_dict: dict[int, float] = {}
    
    for i, v in enumerate(vars):
        if v.VarName in free_var_names:
            free_indices.append(i)
        else:
            try:
                fixed_dict[i] = v.X # Congelamos esta variable en su valor óptimo
            except AttributeError:
                raise ValueError("El modelo debe estar optimizado/resuelto antes de extraer el corte.")
            
    if len(free_indices) != 3:
        raise ValueError(f"Debes elegir exactamente 3 variables. Encontradas: {len(free_indices)}")
        
    A_sparse = model.getA()
    A_matrix = A_sparse.toarray()
    rhs = model.getAttr('RHS', model.getConstrs())
    senses = model.getAttr('Sense', model.getConstrs())
    
    cdd_matrix: list[list[float]] = []
    
    # 1. Procesar restricciones y colapsar la matriz a 3D
    for i in range(len(rhs)):
        b_val = rhs[i]
        a_row = A_matrix[i]
        
        # Extraer solo los coeficientes de las 3 variables libres
        a_free = [a_row[idx] for idx in free_indices]
        
        # Si la restricción no involucra a nuestras 3 variables, la saltamos
        if all(val == 0 for val in a_free):
            continue
            
        # Calcular el nuevo RHS restando el peso de las variables fijas
        fixed_sum = sum(a_row[idx] * fixed_dict[idx] for idx in fixed_dict)
        b_new = b_val - fixed_sum
        
        # Formato CDD: b - Ax >= 0
        if senses[i] == '<' or senses[i] == '=': 
            row = [b_new] + [-val for val in a_free]
            cdd_matrix.append(row)
        if senses[i] == '>' or senses[i] == '=': 
            row = [-b_new] + a_free
            cdd_matrix.append(row)
            
    # 2. Procesar los límites (bounds) SOLO para las 3 variables libres
    for local_idx, global_idx in enumerate(free_indices):
        v = vars[global_idx]
        lb, ub = v.LB, v.UB
        row_a = [0, 0, 0]
        row_a[local_idx] = 1.0
        
        if lb > -GRB.INFINITY: 
            cdd_matrix.append([-lb] + row_a)
        if ub < GRB.INFINITY:  
            cdd_matrix.append([ub] + [-val for val in row_a])

    return cdd_matrix


# --- PRUEBA DEL SCRIPT ---
if __name__ == "__main__":
    # Resuelves tu modelo
    model = build_and_solve_model("instance/uniform-0000010-2.instance", verbose=True, plot=True, maximize=True, time_limit=300)
    
    # ⚠️ AQUI DEBES PONER LOS NOMBRES EXACTOS DE 3 VARIABLES DE TU MODELO
    # Sugerencia para OAP: La X de un vértice, la Y de ese vértice, y el Área total.
    variables_a_visualizar = ["x_0_4", "y_0", "y_1"]  # Reemplaza con los nombres de tus variables
    
    # Extraer a formato CDD ya colapsado a 3D
    matriz_h = extract_3d_slice_from_gurobi(model, variables_a_visualizar)

    # El resto del código es exactamente igual
    mat = cdd.Matrix(matriz_h, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    
    # 1. Encontrar la representación mínima (eliminar redundancias en la matriz)
    mat.canonicalize()

    # 2. Convertir a poliedro usando la matriz ya limpia
    poly = cdd.Polyhedron(mat)

    vertices_raw = poly.get_generators() 
    
    # Limpiamos los vértices (ahora solo tienen 3 coordenadas, no 40)
    vertices_3d = []
    for v in vertices_raw:
        if v[0] == 1.0: 
            vertices_3d.append([v[1], v[2], v[3]]) 

    vertices_3d = np.array(vertices_3d)

    # --- VISUALIZACIÓN 3D CON MATPLOTLIB ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if len(vertices_3d) >= 4: # ConvexHull necesita al menos 4 puntos no coplanares
        hull = ConvexHull(vertices_3d)
        for s in hull.simplices:
            s = np.append(s, s[0])  
            ax.plot(vertices_3d[s, 0], vertices_3d[s, 1], vertices_3d[s, 2], "k-")

        faces = [vertices_3d[s] for s in hull.simplices]
        poly3d = Poly3DCollection(faces, alpha=0.5, facecolors='cyan', linewidths=1, edgecolors='blue')
        ax.add_collection3d(poly3d)

    ax.scatter(vertices_3d[:,0], vertices_3d[:,1], vertices_3d[:,2], color='red', s=50)

    ax.set_xlabel(variables_a_visualizar[0])
    ax.set_ylabel(variables_a_visualizar[1])
    ax.set_zlabel(variables_a_visualizar[2])
    ax.set_title("Corte 3D del Poliedro OAP")

    plt.show()