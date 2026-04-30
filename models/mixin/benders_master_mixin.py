# models/mixin/benders_master_mixin.py
import logging
from typing import Literal

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from models.typing_oap import IndexArray, NumericArray, TrianglesAdjList
from utils.utils import compute_crossing_edges, cost_function_area

logger = logging.getLogger(__name__)


class BendersMasterMixin:
    """Mixin exclusivo para construir el Problema Maestro de Benders."""

    # Pistas de tipado para el linter (sabemos que vendrán de OAPBaseModel)
    model: gp.Model
    N_list: range
    N: int
    CH: IndexArray
    points: NumericArray
    triangles: IndexArray
    triangles_adj_list: TrianglesAdjList
    x: dict
    f: dict
    eta: gp.Var
    _cost_x: dict[tuple[int, int], float]  # per-arc Fekete area coefficients

    def build_master(
        self,
        objective: Literal["Fekete", "Internal"] = "Fekete",
        mode: int = 0,
        maximize: bool = True,
        crosses_constrain: bool = False,
        semiplane: Literal[0, 1] = 0,
    ) -> None:
        """Construye las variables y restricciones del Problema Maestro.

        Parameters
        ----------
        semiplane : Literal[0, 1], default 0
            When 1, adds V1 half-plane constraints to the master after the
            subtour and crossing constraints are in place, but before
            ``model.update()``.  See ``_add_semiplane_master`` for details.
        """

        self._add_variables_master(objective)  # Método auxiliar para añadir variables al maestro

        # --- Función Objetivo ---
        self._add_function_objective_master(
            objective, mode, maximize
        )  # Método auxiliar para configurar la función objetivo del maestro

        # --- Restricciones de Grado ---
        self._add_degree_constraints_master()  # Método auxiliar para añadir restricciones de grado al maestro

        # --- Subtours (SCF) ---
        self._add_subtour_constraints_master()  # Método auxiliar para añadir restricciones de subtour (SCF)

        # --- Cortes de cruce ---
        if crosses_constrain:
            self._add_crossing_constraints_master()  # Método auxiliar para añadir restricciones de cruce

        # --- Semiplane V1 (master-side, pure x-space) ---
        if semiplane == 1:
            self._add_semiplane_master()

        self.model.update()

    def _add_variables_master(self, objective: Literal["Fekete", "Internal"]):
        """Método auxiliar para añadir variables al maestro (si es necesario)."""
        self.x = {
            (i, j): self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            for i in self.N_list
            for j in self.N_list
            if i != j
        }

        self.f = {
            (i, j): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}")
            for i in self.N_list
            for j in self.N_list
            if i != j
        }

        # Instanciamos la variable artificial eta solo si usamos la descomposición guiada por triángulos
        if objective == "Internal":
            self.eta = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.convex_hull_area, name="eta")
            self.objective = objective  # Guardamos el objetivo para usarlo en callbacks o futuras referencias

        # --- Limpieza de la CH ---
        for i in range(len(self.CH)):
            for j in range(i + 2, len(self.CH)):
                if i == 0 and j == len(self.CH) - 1:
                    continue
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

    def _add_function_objective_master(self, objective: Literal["Fekete", "Internal"], mode: int, maximize: bool):
        """Método auxiliar para configurar la función objetivo del maestro."""
        opt_sense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
        # Always compute and cache the per-arc area coefficient even for
        # "Internal" — it is needed by the CGSP-Y optimality cut as f in
        # f^T x̄.
        self._cost_x = cost_function_area(self.points, self.x.keys(), mode=mode)
        if objective == "Fekete":
            self.model.setObjective(
                gp.quicksum(self._cost_x[i] * self.x[i] for i in self.x.keys()),
                opt_sense,
            )
        elif objective == "Internal":
            self.model.setObjective(self.eta, opt_sense)

    def _add_degree_constraints_master(self):
        """Método auxiliar para añadir restricciones de grado al maestro."""
        for i in self.N_list:
            self.model.addConstr(
                (gp.quicksum(self.x[i, j] for j in self.N_list if j != i and (i, j) in self.x) == 1),
                name=f"grado_salida_{i}",
            )
            self.model.addConstr(
                (gp.quicksum(self.x[j, i] for j in self.N_list if j != i and (j, i) in self.x) == 1),
                name=f"grado_entrada_{i}",
            )

    def _add_subtour_constraints_master(self):
        """Método auxiliar para añadir restricciones de subtour (SCF) al maestro."""
        for i in self.N_list:
            if i != 0:
                self.model.addConstr(
                    gp.quicksum(self.f[j, i] for j in self.N_list if j != i and (j, i) in self.f)
                    - gp.quicksum(self.f[i, j] for j in self.N_list if j != i and (i, j) in self.f)
                    == 1,
                    name=f"flujo_nodos_{i}",
                )
        M = self.N - 1
        for i, j in self.x.keys():
            self.model.addConstr(self.f[i, j] <= M * self.x[i, j], name=f"flujo_capacidad_{i}_{j}")

    def _compute_crossing_arc_pairs(self) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        if not hasattr(self, "_crossing_arc_pairs"):
            A_II = compute_crossing_edges(self.triangles, self.points)
            pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for p, q, r, s in A_II:
                arc_1 = (int(p), int(q))
                arc_2 = (int(r), int(s))
                for a in (arc_1, (arc_1[1], arc_1[0])):
                    for b in (arc_2, (arc_2[1], arc_2[0])):
                        pairs.add((a, b))
                        pairs.add((b, a))
            self._crossing_arc_pairs = pairs
        return self._crossing_arc_pairs

    def _add_crossing_constraints_master(self):
        """Método auxiliar para añadir restricciones de cruce al maestro."""
        crossing = compute_crossing_edges(self.triangles, self.points)
        for cross in crossing:
            i, j, k, m = cross
            if (i, j) in self.x and (j, i) in self.x and (k, m) in self.x and (m, k) in self.x:
                self.model.addConstr(
                    self.x[i, j] + self.x[j, i] + self.x[k, m] + self.x[m, k] <= 1,
                    name=f"crossing_{i}_{j}_{k}_{m}",
                )

    def _add_semiplane_master(self) -> None:
        """Add V1 half-plane (semiplane) constraints to the Benders master.

        Mirrors the V1 branch of ``OAPCompactModel.add_restricciones_semiplano``
        line-for-line, except it operates on ``self.x`` (the master-only arc
        variable dict) and routes logging through ``logger`` instead of print.

        For every interior node ``i`` (i.e. ``i not in self.CH``) and every hull
        node ``j in self.CH``, if **no other interior node lies strictly to the
        left** of the directed line ``i -> j`` (signed cross-product ``D > 0``),
        then the master is forced to follow ``(i, j)`` with the next CCW hull
        arc ``(j, j_next)``::

            x[i, j] <= x[j, j_next]   for every (i, j) satisfying the test
                                       AND with both arcs surviving CH-cleanup.

        Constraints are named ``semiplano_master_{i}_{j}`` to disambiguate from
        any Compact-side ``semiplano_{i}_{j}`` names in logged LP files.

        Preconditions
        -------------
        - ``self.x`` is populated (``_add_variables_master`` has run).
        - ``self.CH`` is the canonical CCW-ordered convex-hull index array.
        - ``self.points`` is the integer point matrix from ``OAPBaseModel``.

        Postconditions
        --------------
        - Zero or more new ``gp.Constr`` objects added to ``self.model``.
          Legitimately zero on degenerate instances where no interior->hull arc
          has an empty left half-plane.
        - ``self.x``, ``self.f``, subproblem models, and the objective are
          unchanged.
        - Does NOT call ``self.model.update()``; the caller does that once at
          the end of the build sequence.

        Notes
        -----
        Validity: the V1 inequality is proven valid for the MT3D polytope in
        Hernandez-Perez et al. (2025) §5.2.  Since it is a pure x-space
        constraint it projects unchanged into the Benders master's (x, f)
        polytope.  Decomposition invariance: Y / Y' subproblems are built
        *after* this method returns and are never touched here.
        """
        CH_set = set(int(v) for v in self.CH)
        A_pp = [i for i in range(self.N) if i not in CH_set]
        n_added = 0

        for i in A_pp:
            for j_raw in self.CH:
                j = int(j_raw)
                if (i, j) not in self.x:
                    continue  # arc removed by CH-cleanup or otherwise absent

                # Signed cross-product test: is the left half-plane of i->j
                # free of every *other* interior point?
                x_i, y_i = int(self.points[i][0]), int(self.points[i][1])
                x_j, y_j = int(self.points[j][0]), int(self.points[j][1])

                left_clear = True
                for k in A_pp:
                    if k == i:
                        continue
                    x_k, y_k = int(self.points[k][0]), int(self.points[k][1])
                    D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                    if D > 0:
                        left_clear = False
                        break

                if not left_clear:
                    continue

                idx_j = int(np.where(self.CH == j)[0][0])
                j_next = int(self.CH[(idx_j + 1) % len(self.CH)])
                if (j, j_next) not in self.x:
                    continue  # next-hull arc was cleaned away; constraint vacuous

                self.model.addConstr(
                    self.x[i, j] <= self.x[j, j_next],
                    name=f"semiplano_master_{i}_{j}",
                )
                n_added += 1

        logger.info("[semiplane V1] added %d master constraints.", n_added)
