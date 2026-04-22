import json
import logging
import os

import cdd
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from models.mixin.oap_stats_mixin import OAPStatsMixin
from models.typing_oap import NumericArray
from utils.utils import compute_convex_hull, compute_convex_hull_area, triangles_adjacency_list



logger = logging.getLogger(__name__)

class OAPBaseModel(OAPStatsMixin):
    def __init__(self, points: NumericArray, triangles: NDArray[np.int64], name: str):
        # Todo lo que es común a ambos modelos
        self.points = points
        self.triangles = triangles
        self.triangles_adj_list = triangles_adjacency_list(triangles, points)
        self.N_list = range(len(points))
        self.N = len(points)
        self.CH = compute_convex_hull(points)
        self.V_list = range(len(triangles))
        self.convex_hull_area = compute_convex_hull_area(points)
        
        # El modelo principal (Para Compacto es el modelo entero, para Benders es el Master)
        self.name = name
        self.model = gp.Model(name)

    def extract_subspace_facets(self, var_prefixes: str | list[str] | None = None, verbose: bool = False) -> tuple[list[str], list[list[float]]]:
        """
        Extrae el poliedro N-dimensional correspondiente SOLO a las variables cuyos nombres
        empiezan por los prefijos indicados, fijando las demás variables a sus valores óptimos.
        """
        if var_prefixes is None:
            var_prefixes = ["x"]
        self.model.update()
        vars = self.model.getVars()

        # Permitir que el usuario pase un solo string ('x') o una lista (['x', 'y'])
        if isinstance(var_prefixes, str):
            var_prefixes = [var_prefixes]

        free_indices = []
        free_names = []
        fixed_dict = {}

        # 1. Separar las variables libres indicadas del resto
        for i, v in enumerate(vars):
            # Magia aquí: Comprueba si empieza por 'x' O por 'y'
            if any(v.VarName.startswith(prefix) for prefix in var_prefixes):
                free_indices.append(i)
                free_names.append(v.VarName)
            else:
                try:
                    fixed_dict[i] = v.X # Congelamos las otras variables
                except AttributeError as err:
                    raise ValueError(f"El modelo no tiene solución para la variable {v.VarName}.") from err

        if verbose:
            logger.info(f"Dimensión del sub-espacio: {len(free_indices)} variables ({', '.join(var_prefixes)}) libres.")
            logger.info(f"Variables congeladas: {len(fixed_dict)}")

        A_sparse = self.model.getA()
        A_matrix = A_sparse.toarray()
        rhs = self.model.getAttr('RHS', self.model.getConstrs())
        senses = self.model.getAttr('Sense', self.model.getConstrs())

        cdd_matrix: list[list[float]] = []

        # 2. Colapsar la matriz sobre el espacio de las variables libres
        for i in range(len(rhs)):
            b_val = rhs[i]
            a_row = A_matrix[i]

            a_free = [a_row[idx] for idx in free_indices]

            if all(abs(val) < 1e-6 for val in a_free):
                continue

            fixed_sum = sum(a_row[idx] * fixed_dict[idx] for idx in fixed_dict)
            b_new = b_val - fixed_sum

            # Formato CDD: b - Ax >= 0
            if senses[i] in ['<', '=']: 
                cdd_matrix.append([b_new] + [-val for val in a_free])
            if senses[i] in ['>', '=']: 
                cdd_matrix.append([-b_new] + a_free)

        # 3. Procesar los límites (bounds) solo para las variables libres
        for local_idx, global_idx in enumerate(free_indices):
            v = vars[global_idx]
            lb, ub = v.LB, v.UB
            row_a = [0.0] * len(free_indices)
            row_a[local_idx] = 1.0

            if lb > -GRB.INFINITY: 
                cdd_matrix.append([-lb] + row_a)
            if ub < GRB.INFINITY:  
                cdd_matrix.append([ub] + [-val for val in row_a])

        return free_names, cdd_matrix


    def extract_facets(self, var_prefixes: str | list[str] | None = None, verbose: bool = False) -> tuple[list[str], list[list[float]], set]:
        """
        Extrae las facetas mínimas del poliedro N-dimensional.
        Devuelve: (nombres_variables, matriz_filas, conjunto_igualdades)
        """
        freenames, cdd_matrix = self.extract_subspace_facets(var_prefixes=var_prefixes, verbose=verbose)
        mat = cdd.Matrix(cdd_matrix, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY

        if verbose:
            logger.info("\nCalculando facetas mínimas (eliminando redundancias)...")

        mat.canonicalize() 
        
        # ¡CORRECCIÓN! Guardamos el lin_set en un set de Python antes de destruir el objeto
        lin_set = set(mat.lin_set)

        if verbose:
            logger.info(f"\nEl poliedro tiene exactamente {mat.row_size} facetas (restricciones activas).")

        return freenames, [mat[i] for i in range(mat.row_size)], lin_set

    
    def log_facets(self, filepath: str, var_prefixes: str | list[str] | None = None, verbose: bool = False, 
                   freenames: list[str] | None = None, facets: list[list[float]] | None = None, lin_set: set | None = None) -> None:
        """
        Calcula las facetas y las guarda de forma estructurada (JSONL) en un archivo, 
        además de imprimirlas en el log si verbose=True.
        """
        if facets is None or freenames is None or lin_set is None:
            freenames, rows, lin_set = self.extract_facets(var_prefixes=var_prefixes, verbose=verbose)
        else:
            rows = facets
            freenames = freenames
            lin_set = lin_set

        iteracion_actual = self.iteration if hasattr(self, 'iteration') and self.iteration is not None else 0
        
        if verbose:
            logger.info(f"\n--- FACETAS EN ITERACIÓN {iteracion_actual} ---")
            
        facets_data = []
        
        for i, row in enumerate(rows):
            b = row[0]
            coefs = [-val for val in row[1:]] 

            ecuacion_str = ""
            coefs_dict = {}
            
            for j, coef in enumerate(coefs):
                if abs(coef) > 1e-5:
                    ecuacion_str += f"{coef:+.2f}*{freenames[j]} "
                    coefs_dict[freenames[j]] = coef # Guardamos estructurado para análisis futuro

            signo = "==" if i in lin_set else "<="
            ecuacion_completa = f"{ecuacion_str.strip()} {signo} {b:.2f}"
            
            if verbose:
                logger.info(ecuacion_completa)
                
            # Añadimos a la lista para el JSON
            facets_data.append({
                "equation_str": ecuacion_completa,
                "sense": signo,
                "rhs": b,
                "components": coefs_dict
            })

        # --- GUARDADO EN ARCHIVO JSONL ---
        log_entry = {
            "iteration": iteracion_actual,
            "num_facets": len(facets_data),
            "facets": facets_data
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')