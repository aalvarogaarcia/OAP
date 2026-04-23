# models/mixin/inv_benders_master_mixin.py
"""Master-problem builder for the Inverted Benders decomposition.

In the inverted decomposition the master holds the triangle-assignment
variables (y, yp) and the subproblem checks whether a consistent tour x
exists for a given assignment.
"""
from typing import Literal

import gurobipy as gp
from gurobipy import GRB

from models.typing_oap import IndexArray, NumericArray, TrianglesAdjList
from utils.utils import signed_area


class InvBendersMasterMixin:
    """Mixin that builds the Inverted-Benders master problem.

    Variables
    ---------
    y[t]  : binary — triangle t is *internal* (covered on the tour's left)
    yp[t] : binary — triangle t is *external* (covered on the tour's right)

    Objective
    ---------
    Maximise  Σ_t |area_t| · y_t   (Internal area)

    Optional sum constraints
    ------------------------
    Σ y_t  = N − 2         (exactly N−2 internal triangles in a triangulation)
    Σ yp_t = N − |CH|      (exactly N−|CH| external triangles)
    """

    # --- type hints for the linter (provided by OAPBaseModel) ---
    model: gp.Model
    N: int
    N_list: range
    V_list: range
    CH: IndexArray
    points: NumericArray
    triangles: IndexArray
    triangles_adj_list: TrianglesAdjList

    y:  dict[int, gp.Var]
    yp: dict[int, gp.Var]

    def build_inv_master(
        self,
        objective: Literal["Internal"] = "Internal",
        maximize: bool = True,
        sum_constrain: bool = True,
    ) -> None:
        """Build master variables, objective and (optionally) sum constraints."""
        self._add_variables_inv_master()
        self._add_objective_inv_master(maximize)
        if sum_constrain:
            self._add_sum_constraints_inv_master()
        self.model.update()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_variables_inv_master(self) -> None:
        """Create binary y and yp variables for every triangle."""
        self.y  = {t: self.model.addVar(vtype=GRB.BINARY, name=f"y_{t}")
                   for t in self.V_list}
        self.yp = {t: self.model.addVar(vtype=GRB.BINARY, name=f"yp_{t}")
                   for t in self.V_list}

    def _add_objective_inv_master(self, maximize: bool) -> None:
        """Set the Internal-area objective on the master."""
        sense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
        at = [
            abs(signed_area(
                self.points[tri[0]],
                self.points[tri[1]],
                self.points[tri[2]],
            ))
            for tri in self.triangles
        ]
        self.model.setObjective(
            gp.quicksum(at[t] * self.y[t] for t in self.V_list),
            sense,
        )

    def _add_sum_constraints_inv_master(self) -> None:
        """Add cardinality constraints on the total number of triangles."""
        self.model.addConstr(
            gp.quicksum(self.y[t]  for t in self.V_list) == self.N - 2,
            name="inv_sum_y",
        )
        self.model.addConstr(
            gp.quicksum(self.yp[t] for t in self.V_list) == self.N - len(self.CH),
            name="inv_sum_yp",
        )
