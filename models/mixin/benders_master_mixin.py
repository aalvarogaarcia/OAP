# models/mixin/benders_master_mixin.py
import gurobipy as gp
from gurobipy import GRB
from utils.utils import cost_function_area, compute_crossing_edges
from models.typing_oap import NumericArray, IndexArray, TrianglesAdjList

class BendersMasterMixin:
    """Mixin exclusivo para construir el Problema Maestro de Benders."""
    
    # Pistas de tipado para el linter (sabemos que vendrán de OAPBaseModel)
    model: gp.Model
    N_list: range
    N: int
    CH: IndexArray
    points: NumericArray
    triangles_adj_list: TrianglesAdjList
    x: dict
    f: dict

    def build_master(
        self, 
        objective: str = "Fekete", 
        mode: int = 0, 
        maximize: bool = True, 
        crosses_constrain: bool = False
    ) -> None:
        """Construye las variables y restricciones del Problema Maestro."""

        self._add_variables_master()  # Método auxiliar para añadir variables al maestro
        
        # --- Función Objetivo ---
        self._add_function_objective_master(objective, mode, maximize)  # Método auxiliar para configurar la función objetivo del maestro

        # --- Restricciones de Grado ---
        self._add_degree_constraints_master()  # Método auxiliar para añadir restricciones de grado al maestro

        # --- Subtours (SCF) ---
        self._add_subtour_constraints_master()  # Método auxiliar para añadir restricciones de subtour (SCF)

        # --- Cortes de cruce ---
        if crosses_constrain:
            self._add_crossing_constraints_master()  # Método auxiliar para añadir restricciones de cruce

        self.model.update()

    def _add_variables_master(self):
        """Método auxiliar para añadir variables al maestro (si es necesario)."""
        self.x = {(i, j): self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") 
                  for i in self.N_list for j in self.N_list if i != j}
        
        self.f = {(i, j): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}") 
                  for i in self.N_list for j in self.N_list if i != j}

        # --- Limpieza de la CH ---
        for i in range(len(self.CH)):
            for j in range(i + 2, len(self.CH)):
                if i == 0 and j == len(self.CH) - 1: continue
                for var_dict in [self.x, self.f]:
                    if (self.CH[i], self.CH[j]) in var_dict:
                        self.model.remove(var_dict[self.CH[i], self.CH[j]])
                        self.model.remove(var_dict[self.CH[j], self.CH[i]])
                        var_dict.pop((self.CH[i], self.CH[j]), None)
                        var_dict.pop((self.CH[j], self.CH[i]), None)

            j = (i + 1) % len(self.CH)
            for var_dict in [self.x, self.f]:
                if (self.CH[j], self.CH[i]) in var_dict:
                    self.model.remove(var_dict[self.CH[j], self.CH[i]])
                    var_dict.pop((self.CH[j], self.CH[i]), None)


    def _add_function_objective_master(self, objective: str, mode: int, maximize: bool):
        """Método auxiliar para configurar la función objetivo del maestro."""
        if objective == "Fekete":
            c = cost_function_area(self.points, self.x.keys(), mode=mode)
            opt_sense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
            self.model.setObjective(gp.quicksum(c[i] * self.x[i] for i in self.x.keys()), opt_sense)

    def _add_degree_constraints_master(self):
        """Método auxiliar para añadir restricciones de grado al maestro."""
        for i in self.N_list:
            self.model.addConstr((gp.quicksum(self.x[i, j] for j in self.N_list if j != i and (i, j) in self.x) == 1), name=f"grado_salida_{i}")
            self.model.addConstr((gp.quicksum(self.x[j, i] for j in self.N_list if j != i and (j, i) in self.x) == 1), name=f"grado_entrada_{i}")

    def _add_subtour_constraints_master(self):
        """Método auxiliar para añadir restricciones de subtour (SCF) al maestro."""
        for i in self.N_list:
            if i != 0:
                self.model.addConstr(
                    gp.quicksum(self.f[j, i] for j in self.N_list if j != i and (j, i) in self.f) - 
                    gp.quicksum(self.f[i, j] for j in self.N_list if j != i and (i, j) in self.f) == 1, 
                    name=f"flujo_nodos_{i}"
                )
        M = self.N - 1
        for i, j in self.x.keys():
            self.model.addConstr(self.f[i, j] <= M * self.x[i, j], name=f"flujo_capacidad_{i}_{j}")

    def _add_crossing_constraints_master(self):
        """Método auxiliar para añadir restricciones de cruce al maestro."""
        crossing = compute_crossing_edges(self.points, self.x.keys())
        for cross in crossing:
            i, j, k, m = cross
            if (i, j) in self.x and (j, i) in self.x and (k, m) in self.x and (m, k) in self.x:
                self.model.addConstr(
                    self.x[i, j] + self.x[j, i] + self.x[k, m] + self.x[m, k] <= 1, 
                    name=f"crossing_{i}_{j}_{k}_{m}"
                )