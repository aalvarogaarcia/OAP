# oap_builder_mixin.py
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from utils.geometry import compute_crossing_edges
from utils.utils import cost_function_area, signed_area

Arc = tuple[int, int]
MCFArc = tuple[int, int, int]
Triangle = tuple[int, int, int]
AdjList = list[list[list[int]]]


class OAPBuilderMixin:
    """
    Clase Mixin que contiene exclusivamente los métodos de construcción 
    para el modelo OAPCompactModel.
    """
    # --- Pistas para el linter (Type Hints) ---
    # Le decimos al editor: "Confía en mí, quien herede esto tendrá estos atributos"
    model: gp.Model
    N: int
    N_list: range
    CH: NDArray[np.int_]
    V_list: range
    convex_hull_area: float
    points: NDArray[np.int64]
    triangles: NDArray[np.int64]
    triangles_adj_list: AdjList

    x: dict[Arc, gp.Var]
    y: dict[int, gp.Var]
    yp: dict[int, gp.Var]
    f: dict[Arc, gp.Var]
    u: dict[int, gp.Var]
    f_mcf: dict[MCFArc, gp.Var]
    z: dict[Arc, gp.Var]
    zp: dict[Arc, gp.Var]
    _abs_areas: list[float]

    def _create_variables(self, subtour: str, objective: str, mode: int):
        """Crea todas las variables y limpia los arcos inválidos de la envolvente convexa (CH)."""
        # Variables principales
        self.x = {(i, j): self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") 
                  for i in self.N_list for j in self.N_list if i != j}
        self.y = {i: self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}") for i in self.V_list}
        self.yp = {i: self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yp_{i}") for i in self.V_list}
        self.c = cost_function_area(self.points, self.x, 0)

        # Variables de Subtours
        if subtour == "SCF":
            self.f = {(i, j): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}") 
                      for i in self.N_list for j in self.N_list if i != j}
        elif subtour == "MTZ":
            self.u = {i: self.model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=self.N-1, name=f"u_{i}") 
                      for i in self.N_list if i != 0}
        elif subtour == "MCF":
            self.f_mcf = {(k, i, j): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"fmcf_{k}_{i}_{j}")
                          for k in self.N_list if k != 0 for i in self.N_list for j in self.N_list if i != j}

        # --- Limpieza de arcos en la Envolvente Convexa (CH) ---
        for i in range(len(self.CH)):
            for j in range(i + 2, len(self.CH)):
                if i == 0 and j == len(self.CH) - 1:
                    continue
                # Eliminar arcos que cruzan la CH
                for var_dict in [self.x, self.f]: # self.f estará vacío si subtour != "SCF"
                    if (self.CH[i], self.CH[j]) in var_dict:
                        self.model.remove(var_dict[self.CH[i], self.CH[j]])
                        self.model.remove(var_dict[self.CH[j], self.CH[i]])
                        var_dict.pop((self.CH[i], self.CH[j]), None)
                        var_dict.pop((self.CH[j], self.CH[i]), None)

        # Remover arcos en sentido horario de la CH
        for i in range(len(self.CH)):
            j = (i + 1) % len(self.CH)
            if (self.CH[j], self.CH[i]) in self.x:
                self.model.remove(self.x[self.CH[j], self.CH[i]])
                self.x.pop((self.CH[j], self.CH[i]), None)
                if subtour == "SCF" and (self.CH[j], self.CH[i]) in self.f:
                    self.model.remove(self.f[self.CH[j], self.CH[i]])
                    self.f.pop((self.CH[j], self.CH[i]), None)

        # Variables de Diagonales
        if objective == "Diagonals":
            if mode == 0:
                self.z = {(i, j): self.model.addVar(vtype=GRB.CONTINUOUS, name=f"z_{i}_{j}") 
                          for i in self.N_list for j in self.N_list if i != j and (i not in self.CH or j not in self.CH)}
            elif mode == 1:
                self.zp = {(i, j): self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zp_{i}_{j}") 
                           for i in self.N_list for j in self.N_list if i != j and (i not in self.CH or j not in self.CH)}

    def _set_objective(self, objective: str, mode: int, maximize: bool):
        """Define la función objetivo."""
        optimizer = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
        if not hasattr(self, "_abs_areas"):
            self._abs_areas = [
                np.abs(signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]))
                for tri in self.triangles
            ]
        at = self._abs_areas

        if objective == "Fekete":
            self.model.setObjective(gp.quicksum(self.c[i] * self.x[i] for i in self.x.keys()), optimizer)
        
        elif objective == "Internal":
            self.model.setObjective(gp.quicksum(self.y[i] * at[i] for i in self.V_list), optimizer)
            
        elif objective == "External":
            self.model.setObjective(self.convex_hull_area - gp.quicksum(self.yp[i] * at[i] for i in self.V_list), optimizer)
            
        elif objective == "Diagonals":
            signed_at = [signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]) for tri in self.triangles]
            d = {(i, j): np.min([signed_at[t] for t in self.triangles_adj_list[i][j]]) for i, j in self.x.keys()}
            td = {tuple(tri): 3*signed_at[idx] - d[tri[0], tri[1]] - d[tri[1], tri[2]] - d[tri[2], tri[0]] 
                  for idx, tri in enumerate(self.triangles)}

            sum_x = sum_z = 0
            if mode == 0:
                sum_x = gp.quicksum(d[i, j] * self.x[i, j] for i, j in self.x.keys())
                sum_z = gp.quicksum(d[i, j] * self.z[i, j] for i, j in self.z.keys())
            elif mode == 1:
                sum_z = gp.quicksum(d[i, j] * (self.x[i, j] + self.zp[i, j]) for i, j in self.zp.keys())
            
            sum_td = 3 * gp.quicksum(td[i, j, k] for i, j, k in td.keys())
            self.model.setObjective(1/3 * (sum_x + sum_z + sum_td), optimizer)

    def _add_degree_constraints(self):
        """Restricciones básicas de grado de entrada y salida."""
        for i in self.N_list:
            self.model.addConstr((gp.quicksum(self.x[i, j] for j in self.N_list if j != i and (i, j) in self.x) == 1), name=f"Grado_salida_{i}")
            self.model.addConstr((gp.quicksum(self.x[j, i] for j in self.N_list if j != i and (j, i) in self.x) == 1), name=f"Grado_entrada_{i}")

    def _add_subtour_constraints(self, subtour: str):
        """Restricciones de eliminación de subtours (SEC)."""
        if subtour == "SCF":
            for i in self.N_list:
                if i != 0:
                    self.model.addConstr(
                        gp.quicksum(self.f[j, i] for j in self.N_list if j != i and (j, i) in self.f) - 
                        gp.quicksum(self.f[i, j] for j in self.N_list if j != i and (i, j) in self.f) == 1, 
                        name=f"flujo_nodos_{i}"
                    )
                for j in self.N_list:
                    if i != j and (i, j) in self.f:
                        self.model.addConstr(self.f[i, j] <= (self.N - 1) * self.x[i, j], name=f"flujo_arcos_{i}_{j}")
        
        elif subtour == "MTZ":
            for i in self.N_list:
                for j in self.N_list:
                    if i != 0 and j != 0 and i != j and (i, j) in self.x:
                        self.model.addConstr(self.u[i] - self.u[j] + (self.N - 1) * self.x[i, j] <= self.N - 2, name=f"MTZ_{i}_{j}")
        
        elif subtour == "MCF":
            for k in self.N_list:
                if k != 0:
                    for i in self.N_list:
                        in_flow = gp.quicksum(self.f_mcf[k, j, i] for j in self.N_list if j != i and (j, i) in self.x)
                        out_flow = gp.quicksum(self.f_mcf[k, i, j] for j in self.N_list if j != i and (i, j) in self.x)
                        
                        if i == 0:
                            self.model.addConstr(out_flow - in_flow == 1, name=f"mcf_origen_{k}_{i}")
                        elif i == k:
                            self.model.addConstr(in_flow - out_flow == 1, name=f"mcf_destino_{k}_{i}")
                        else:
                            self.model.addConstr(in_flow - out_flow == 0, name=f"mcf_transito_{k}_{i}")
                    
                    for i in self.N_list:
                        for j in self.N_list:
                            if i != j and (i, j) in self.x:
                                self.model.addConstr(self.f_mcf[k, i, j] <= self.x[i, j], name=f"mcf_cap_{k}_{i}_{j}")


    def _add_sum_constraints(self):
        """Restricciones adicionales para asegurar que el número total de triángulos internos y externos sea correcto."""
        self.model.addConstr(gp.quicksum(self.y[i] for i in self.V_list) == self.N - 2, name="triangulos_internos_totales")
        self.model.addConstr(gp.quicksum(self.yp[i] for i in self.V_list) == self.N - len(self.CH), name="triangulos_externos_totales")


    def _add_triangle_ch_constraints(self):
        """Restricciones de los triángulos y su relación con la envolvente convexa."""

        for i in range(len(self.CH)):
            a = self.CH[i]
            b = self.CH[(i + 1) % len(self.CH)]
            if (a, b) in self.x:
                self.model.addConstr(gp.quicksum(self.y[t] for t in self.triangles_adj_list[a][b]) == self.x[a, b], name=f"CH_arcos_internos_{a}_{b}")
                self.model.addConstr(gp.quicksum(self.yp[t] for t in self.triangles_adj_list[a][b]) == 1 - self.x[a, b], name=f"CH_arcos_externos_{a}_{b}")

    def _add_variable_relation_constraints(self, objective: str, mode: int):
        """Restricciones lógicas entre x, y, yp y z (diagonales)."""
        # Diagonales
        if objective == "Diagonals":
            if mode == 0:
                for i, j in self.z.keys():
                    if i < j and (j, i) in self.z:
                        self.model.addConstr(self.z[i, j] == self.z[j, i], name=f"diagonal_selection_{i}_{j}")
                    self.model.addConstr(self.z[i, j] == gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) - self.x[i, j], name=f"diagonal_triangle_relation_{i}_{j}")
            elif mode == 1:
                for i, j in self.zp.keys():
                    if i < j and (j, i) in self.zp:
                        self.model.addConstr(self.zp[i, j] == self.zp[j, i], name=f"diagonal_selection_{i}_{j}")
                    self.model.addConstr(self.zp[i, j] == gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) - self.x[i, j], name=f"diagonal_triangle_relation_{i}_{j}")

        # Relación Y - YP - X
        for i in self.N_list:
            for j in self.N_list:
                if (i not in self.CH or j not in self.CH) and i < j:
                    if objective == "Diagonals" and mode == 0:
                        self.model.addConstr(gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.yp[t] for t in self.triangles_adj_list[j][i]) == self.x[j, i] - self.x[i, j])
                    elif objective == "Diagonals" and mode == 1:
                        self.model.addConstr(gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.y[t] for t in self.triangles_adj_list[j][i]) == self.x[i, j] - self.x[j, i])
                    else:
                        self.model.addConstr(gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.y[t] for t in self.triangles_adj_list[j][i]) == self.x[i, j] - self.x[j, i])
                        self.model.addConstr(gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.yp[t] for t in self.triangles_adj_list[j][i]) == self.x[j, i] - self.x[i, j])

                if (i not in self.CH or j not in self.CH) and i != j and (i, j) in self.x:
                    self.model.addConstr(self.x[i, j] <= gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]))
                    self.model.addConstr(1 - self.x[j, i] >= gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]))

                    self.model.addConstr(self.x[j, i] <= gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]))
                    self.model.addConstr(1 - self.x[i, j] >= gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]))

    def _add_strengthening_constraints(self) -> None:
        """
        LP-strengthening constraints (Sec. 5.4 of Hernández-Pérez et al., 2025).

        Redundant for the IP, but tighten the LP relaxation:

          R1 – area balance:
                Σ_t  c_t · (y_t + yp_t)  ==  area(CH)

          R2 – arc triangle-fan upper bound:
                Σ_{t ∈ V_ij}  (y_t + yp_t)  <=  1    ∀ (i,j) ∈ A

              Each directed arc can be covered by at most one triangle on each
              side, so the combined fan cannot exceed 1.  Skips arcs whose
              adjacency list is empty (would add a vacuous 0 <= 1 which is
              harmless but wasteful).

          R3 – mixed crossing clique:
                Σ_{t ∈ V_pq} (y_t + yp_t) + x[q,p] + x[r,s]
                + Σ_{t ∈ V_rs} (y_t + yp_t)  <=  1
                and its symmetric counterpart (R3b),
                for every pair of crossing undirected edges {p,q} × {r,s}.

              Couples triangle-fan activation with the routing decision;
              tighter than a pure x-variable crossing cut.
        """
        # --- R1: Area balance ---
        if not hasattr(self, "_abs_areas"):
            self._abs_areas = [
                abs(signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]))
                for tri in self.triangles
            ]
        at = self._abs_areas

        self.model.addConstr(
            gp.quicksum(at[t] * (self.y[t] + self.yp[t]) for t in self.V_list)
            == self.convex_hull_area,
            name="R1_area_balance_CH",
        )

        # --- R2: Arc triangle-fan upper bound ---
        for (i, j) in self.x.keys():
            tris = self.triangles_adj_list[i][j]
            if not tris:
                continue
            self.model.addConstr(
                gp.quicksum(self.y[t] + self.yp[t] for t in tris) <= 1,
                name=f"R2_arc_fan_{i}_{j}",
            )

        # --- R3: Mixed crossing clique ---
        # For each pair of crossing undirected edges {p,q} x {r,s}, add:
        #
        #   Σ_{t ∈ V_pq} (y_t + yp_t) + x[q,p] + x[r,s]
        #   + Σ_{t ∈ V_rs} (y_t + yp_t) <= 1
        #
        # and its symmetric counterpart with the two arcs swapped.
        # This couples the triangle-fan activation with the routing decision
        # and is tighter than a pure x-variable crossing cut alone.
        #
        # compute_crossing_edges already deduplicates pairs; each row
        # [a,b,c,d] encodes undirected edges {a,b} and {c,d}.
        crossing_rows = compute_crossing_edges(self.triangles, self.points)

        # Materialise self.x keys once into a set for O(1) membership tests.
        arc_set: set[Arc] = set(self.x.keys())

        for row in crossing_rows:
            p, q, r, s = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            tris_pq = self.triangles_adj_list[p][q]
            tris_qp = self.triangles_adj_list[q][p]
            tris_rs = self.triangles_adj_list[r][s]
            tris_sr = self.triangles_adj_list[s][r]

            # R3a: fan(p→q) + x[q,p] + x[r,s] + fan(s→r) <= 1
            if tris_pq and (q, p) in arc_set and (r, s) in arc_set and tris_rs:
                self.model.addConstr(
                    gp.quicksum(self.y[t] for t in tris_pq)
                    + self.x[q, p]
                    + self.x[r, s]
                    + gp.quicksum(self.y[t] for t in tris_sr)
                    <= 1,
                    name=f"R3a_{p}_{q}_{r}_{s}",
                )

            # R3b: fan(q→p) + x[p,q] + x[s,r] + fan(r→s) <= 1  (symmetric)
            if tris_qp and (p, q) in arc_set and (s, r) in arc_set and tris_sr:
                self.model.addConstr(
                    gp.quicksum(self.y[t] for t in tris_qp)
                    + self.x[p, q]
                    + self.x[s, r]
                    + gp.quicksum(self.y[t] for t in tris_rs)
                    <= 1,
                    name=f"R3b_{q}_{p}_{s}_{r}",
                )

            # R4a: fan(p→q) + x[p,q] + x[r,s] + fan(r→s) <= 1
            if tris_pq and (q, p) in arc_set and (r, s) in arc_set and tris_rs:
                self.model.addConstr(
                    gp.quicksum(self.y[t] for t in tris_pq)
                    + self.x[q, p]
                    + self.x[r, s]
                    + gp.quicksum(self.y[t] for t in tris_rs)
                    <= 1,
                    name=f"R3a_{p}_{q}_{r}_{s}",
                )

            # R4b: fan(q→p) + x[p,q] + x[s,r] + fan(s→r) <= 1  (symmetric)
            if tris_qp and (p, q) in arc_set and (s, r) in arc_set and tris_sr:
                self.model.addConstr(
                    gp.quicksum(self.y[t] for t in tris_qp)
                    + self.x[p, q]
                    + self.x[s, r]
                    + gp.quicksum(self.y[t] for t in tris_sr)
                    <= 1,
                    name=f"R3b_{q}_{p}_{s}_{r}",
                )
