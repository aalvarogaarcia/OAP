# models/mixin/inv_benders_sub_mixin.py
"""Subproblem builder and Farkas-cut extractor for the Inverted Benders
decomposition.

Given a fixed binary assignment (y*, yp*) from the master, the subproblem LP
checks whether a consistent Hamiltonian tour x exists.  Infeasibility yields
a Farkas ray whose coefficients are translated back into a linear cut on
(y, yp) and injected into the master as a lazy constraint.
"""
from __future__ import annotations

import logging
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from utils.benders_log import log_inv_benders_cut

from models.typing_oap import IndexArray, NumericArray, TrianglesAdjList

logger = logging.getLogger(__name__)

Arc = tuple[int, int]
RayComponents = dict[str, dict[Arc, float] | float]


class InvBendersSubMixin:
    """Mixin that builds ``sub_x``, updates its RHS, and extracts Farkas cuts.

    Constraint naming mirrors the compact model:

    * **α**   CH CCW arc (a,b): ``x_{ab} = c_{ab}``              (from y*)
    * **α'**  CH CCW arc (a,b): ``x_{ab} = 1 − cp_{ab}``        (from yp*)
    * **β**   non-CH, i<j:      ``x_{ij}−x_{ji} = c_{ij}−c_{ji}`` (from y*)
    * **β'**  non-CH, i<j:      ``x_{ij}−x_{ji} = cp_{ji}−cp_{ij}`` (from yp*)
    * **γ**   non-CH, i≠j:      ``x_{ij} ≤ c_{ij}``             (from y*)
    * **δ**   non-CH, i≠j:      ``x_{ji} ≤ 1−c_{ij}``           (from y*)
    * **γ'**  non-CH, i≠j:      ``x_{ji} ≤ cp_{ij}``            (from yp*)
    * **δ'**  non-CH, i≠j:      ``x_{ij} ≤ 1−cp_{ij}``          (from yp*)
    * **degree / SCF** — no y/yp dependency.
    """

    # --- type hints (provided by OAPBaseModel) ---
    N: int
    N_list: range
    V_list: range
    CH: IndexArray
    points: NumericArray
    triangles: IndexArray
    triangles_adj_list: TrianglesAdjList

    # provided by InvBendersMasterMixin
    y:  dict[int, gp.Var]
    yp: dict[int, gp.Var]

    # provided by OAPInverseBendersModel.__init__
    sub_x:      gp.Model
    constrs_sub: dict[str, dict[Arc, gp.Constr]]
    iteration:  int
    log_path:   str
    save_cuts:  bool

    # sub_x internal variable dicts (not master vars)
    _sub_x_vars: dict[Arc, gp.Var]
    _sub_f_vars: dict[Arc, gp.Var]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_inv_sub(self) -> None:
        """Create the subproblem LP and store all parameterised constraints."""
        print("Building inverted Benders subproblem (sub_x LP)...")

        # 1. Gurobi params for Farkas extraction.
        #    Presolve MUST be 0: presolve transformations can suppress or
        #    alter the Farkas ray, producing zero-duals and silent infinite loops.
        self.sub_x.Params.InfUnbdInfo    = 1
        self.sub_x.Params.DualReductions = 0
        self.sub_x.Params.Presolve       = 0

        # 2. Create x and f variables (continuous, bounded [0,1] and [0,∞))
        self._sub_x_vars = {
            (i, j): self.sub_x.addVar(lb=0.0, ub=1.0, name=f"x_{i}_{j}")
            for i in self.N_list for j in self.N_list if i != j
        }
        self._sub_f_vars = {
            (i, j): self.sub_x.addVar(lb=0.0, name=f"f_{i}_{j}")
            for i in self.N_list for j in self.N_list if i != j
        }

        # 3. CH arc pruning — exact same logic as OAPBuilderMixin._create_variables
        for i in range(len(self.CH)):
            for j in range(i + 2, len(self.CH)):
                if i == 0 and j == len(self.CH) - 1:
                    continue
                for var_dict in [self._sub_x_vars, self._sub_f_vars]:
                    for key in [(self.CH[i], self.CH[j]),
                                (self.CH[j], self.CH[i])]:
                        if key in var_dict:
                            self.sub_x.remove(var_dict[key])
                            var_dict.pop(key)

        for i in range(len(self.CH)):
            j = (i + 1) % len(self.CH)
            for var_dict in [self._sub_x_vars, self._sub_f_vars]:
                key = (self.CH[j], self.CH[i])   # CW arc — forbidden
                if key in var_dict:
                    self.sub_x.remove(var_dict[key])
                    var_dict.pop(key)

        # 4. Initialise constraint storage
        self.constrs_sub = {
            'alpha':   {},
            'alpha_p': {},
            'beta':    {},
            'beta_p':  {},
            'gamma':   {},
            'delta':   {},
            'gamma_p': {},
            'delta_p': {},
        }

        # 5. Degree constraints (no y/yp dependency — RHS never changes)
        for i in self.N_list:
            out_arcs = [self._sub_x_vars[i, j]
                        for j in self.N_list if j != i and (i, j) in self._sub_x_vars]
            in_arcs  = [self._sub_x_vars[j, i]
                        for j in self.N_list if j != i and (j, i) in self._sub_x_vars]
            if out_arcs:
                self.sub_x.addConstr(gp.quicksum(out_arcs) == 1,
                                     name=f"sub_deg_out_{i}")
            if in_arcs:
                self.sub_x.addConstr(gp.quicksum(in_arcs) == 1,
                                     name=f"sub_deg_in_{i}")

        # 6. SCF subtour constraints (no y/yp dependency)
        M = self.N - 1
        for i in self.N_list:
            if i != 0:
                in_flow  = gp.quicksum(
                    self._sub_f_vars[j, i]
                    for j in self.N_list
                    if j != i and (j, i) in self._sub_f_vars
                )
                out_flow = gp.quicksum(
                    self._sub_f_vars[i, j]
                    for j in self.N_list
                    if j != i and (i, j) in self._sub_f_vars
                )
                self.sub_x.addConstr(in_flow - out_flow == 1,
                                     name=f"sub_flow_{i}")
        for (i, j) in self._sub_x_vars:
            self.sub_x.addConstr(
                self._sub_f_vars[i, j] <= M * self._sub_x_vars[i, j],
                name=f"sub_flow_cap_{i}_{j}",
            )

        # 7. Parameterised constraints (initial RHS = 0; updated before each solve)
        self._add_alpha_constrs()
        self._add_beta_constrs()
        self._add_bound_constrs()

        self.sub_x.update()
        print(f"  sub_x LP: {self.sub_x.NumVars} vars, "
              f"{self.sub_x.NumConstrs} constraints.")

    # ------------------------------------------------------------------
    # Constraint builders (called once from build_inv_sub)
    # ------------------------------------------------------------------

    def _add_alpha_constrs(self) -> None:
        """α and α' equality constraints for each CH CCW arc."""
        for i in range(len(self.CH)):
            a = self.CH[i]
            b = self.CH[(i + 1) % len(self.CH)]
            if (a, b) not in self._sub_x_vars:
                continue
            x_ab = self._sub_x_vars[a, b]
            # α : x_{ab} = c_{ab}       (RHS updated from y*)
            self.constrs_sub['alpha'][a, b] = self.sub_x.addConstr(
                x_ab == 0.0, name=f"sub_alpha_{a}_{b}"
            )
            # α': x_{ab} = 1 − cp_{ab}  (RHS updated from yp*)
            self.constrs_sub['alpha_p'][a, b] = self.sub_x.addConstr(
                x_ab == 1.0, name=f"sub_alpha_p_{a}_{b}"
            )

    def _add_beta_constrs(self) -> None:
        """β and β' balance constraints for non-CH arcs (i<j)."""
        ch_set = set(int(v) for v in self.CH)
        for i in self.N_list:
            for j in self.N_list:
                if j <= i:
                    continue
                if i in ch_set and j in ch_set:
                    continue
                if (i, j) not in self._sub_x_vars or (j, i) not in self._sub_x_vars:
                    continue
                diff = self._sub_x_vars[i, j] - self._sub_x_vars[j, i]
                # β : x_{ij}−x_{ji} = c_{ij}−c_{ji}   (from y*)
                self.constrs_sub['beta'][i, j] = self.sub_x.addConstr(
                    diff == 0.0, name=f"sub_beta_{i}_{j}"
                )
                # β': x_{ij}−x_{ji} = cp_{ji}−cp_{ij} (from yp*)
                self.constrs_sub['beta_p'][i, j] = self.sub_x.addConstr(
                    diff == 0.0, name=f"sub_beta_p_{i}_{j}"
                )

    def _add_bound_constrs(self) -> None:
        """γ, δ, γ', δ' upper-bound constraints for non-CH arcs (i≠j)."""
        ch_set = set(int(v) for v in self.CH)
        for i in self.N_list:
            for j in self.N_list:
                if i == j:
                    continue
                if i in ch_set and j in ch_set:
                    continue
                if (i, j) not in self._sub_x_vars:
                    continue
                x_ij = self._sub_x_vars[i, j]
                # γ : x_{ij} ≤ c_{ij}
                self.constrs_sub['gamma'][i, j] = self.sub_x.addConstr(
                    x_ij <= 0.0, name=f"sub_gamma_{i}_{j}"
                )
                # δ : x_{ji} ≤ 1 − c_{ij}   (bounds x_{ji} using y* for arc i→j)
                if (j, i) in self._sub_x_vars:
                    self.constrs_sub['delta'][i, j] = self.sub_x.addConstr(
                        self._sub_x_vars[j, i] <= 1.0, name=f"sub_delta_{i}_{j}"
                    )
                # γ': x_{ji} ≤ cp_{ij}
                if (j, i) in self._sub_x_vars:
                    self.constrs_sub['gamma_p'][i, j] = self.sub_x.addConstr(
                        self._sub_x_vars[j, i] <= 0.0, name=f"sub_gamma_p_{i}_{j}"
                    )
                # δ': x_{ij} ≤ 1 − cp_{ij}
                self.constrs_sub['delta_p'][i, j] = self.sub_x.addConstr(
                    x_ij <= 1.0, name=f"sub_delta_p_{i}_{j}"
                )

    # ------------------------------------------------------------------
    # RHS update
    # ------------------------------------------------------------------

    def update_sub_rhs(
        self,
        y_sol:  dict[int, float],
        yp_sol: dict[int, float],
    ) -> None:
        """Update every parameterised RHS using the current master solution.

        For each arc (i,j), compute::

            c_{ij}  = Σ_{t ∈ ta[i][j]} y*_t
            cp_{ij} = Σ_{t ∈ ta[i][j]} yp*_t

        then set the stored constraint RHS values accordingly.
        """
        ta = self.triangles_adj_list

        # Helper: coefficient for a directed arc
        def c(i: int, j: int) -> float:
            return sum(y_sol[t] for t in ta[i][j])

        def cp(i: int, j: int) -> float:
            return sum(yp_sol[t] for t in ta[i][j])

        # α
        for (a, b), constr in self.constrs_sub['alpha'].items():
            constr.RHS = c(a, b)

        # α'
        for (a, b), constr in self.constrs_sub['alpha_p'].items():
            constr.RHS = 1.0 - cp(a, b)

        # β
        for (i, j), constr in self.constrs_sub['beta'].items():
            constr.RHS = c(i, j) - c(j, i)

        # β'
        for (i, j), constr in self.constrs_sub['beta_p'].items():
            constr.RHS = cp(j, i) - cp(i, j)

        # γ
        for (i, j), constr in self.constrs_sub['gamma'].items():
            constr.RHS = c(i, j)

        # δ
        for (i, j), constr in self.constrs_sub['delta'].items():
            constr.RHS = 1.0 - c(i, j)

        # γ'
        for (i, j), constr in self.constrs_sub['gamma_p'].items():
            constr.RHS = cp(i, j)

        # δ'
        for (i, j), constr in self.constrs_sub['delta_p'].items():
            constr.RHS = 1.0 - cp(i, j)

    # ------------------------------------------------------------------
    # Farkas cut extraction
    # ------------------------------------------------------------------

    def get_farkas_cut(
        self,
        y_sol:  dict[int, float],
        yp_sol: dict[int, float],
        TOL: float = 1e-10,
    ) -> tuple[gp.LinExpr, float]:
        """Extract the Farkas ray from an infeasible ``sub_x`` and build a cut.

        The cut is a linear expression **in the master variables** (``self.y``,
        ``self.yp``) such that:

        * ``cut_val > TOL``  →  caller adds ``cbLazy(cut_expr <= 0)``
        * ``cut_val < -TOL`` →  caller adds ``cbLazy(cut_expr >= 0)``

        Returns
        -------
        cut_expr : gp.LinExpr
            Linear expression in the master's ``y`` and ``yp`` variables.
        cut_val : float
            Numerical value of the expression evaluated at ``(y*, yp*)``.
        """
        if self.sub_x.Status != GRB.INFEASIBLE:
            raise RuntimeError(
                f"get_farkas_cut called with sub_x.Status={self.sub_x.Status}. "
                "FarkasDual attributes are only populated after an INFEASIBLE solve."
            )

        ta = self.triangles_adj_list
        cut_expr = gp.LinExpr()
        cut_val  = 0.0

        v_comps: RayComponents = {
            'alpha':   {}, 'alpha_p': {},
            'beta':    {}, 'beta_p':  {},
            'gamma':   {}, 'delta':   {},
            'gamma_p': {}, 'delta_p': {},
        }

        # --- α : RHS = c_{ab} = Σ_t ta[a][b] · y_t ---
        for (a, b), constr in self.constrs_sub['alpha'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['alpha'][a, b] = lam
                for t in ta[a][b]:
                    cut_expr += lam * self.y[t]
                    cut_val  += lam * y_sol[t]

        # --- α': RHS = 1 − cp_{ab} = 1 − Σ_t ta[a][b] · yp_t ---
        for (a, b), constr in self.constrs_sub['alpha_p'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['alpha_p'][a, b] = lam
                cut_expr += lam               # constant +λ·1
                for t in ta[a][b]:
                    cut_expr -= lam * self.yp[t]
                    cut_val  -= lam * yp_sol[t]
                cut_val += lam

        # --- β : RHS = c_{ij}−c_{ji} ---
        for (i, j), constr in self.constrs_sub['beta'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['beta'][i, j] = lam
                for t in ta[i][j]:
                    cut_expr += lam * self.y[t]
                    cut_val  += lam * y_sol[t]
                for t in ta[j][i]:
                    cut_expr -= lam * self.y[t]
                    cut_val  -= lam * y_sol[t]

        # --- β': RHS = cp_{ji}−cp_{ij} ---
        for (i, j), constr in self.constrs_sub['beta_p'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['beta_p'][i, j] = lam
                for t in ta[j][i]:
                    cut_expr += lam * self.yp[t]
                    cut_val  += lam * yp_sol[t]
                for t in ta[i][j]:
                    cut_expr -= lam * self.yp[t]
                    cut_val  -= lam * yp_sol[t]

        # --- γ : RHS = c_{ij} = Σ_t ta[i][j] · y_t ---
        for (i, j), constr in self.constrs_sub['gamma'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['gamma'][i, j] = lam
                for t in ta[i][j]:
                    cut_expr += lam * self.y[t]
                    cut_val  += lam * y_sol[t]

        # --- δ : RHS = 1 − c_{ij} ---
        for (i, j), constr in self.constrs_sub['delta'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['delta'][i, j] = lam
                cut_expr += lam               # constant +λ·1
                for t in ta[i][j]:
                    cut_expr -= lam * self.y[t]
                    cut_val  -= lam * y_sol[t]
                cut_val += lam

        # --- γ': RHS = cp_{ij} = Σ_t ta[i][j] · yp_t ---
        for (i, j), constr in self.constrs_sub['gamma_p'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['gamma_p'][i, j] = lam
                for t in ta[i][j]:
                    cut_expr += lam * self.yp[t]
                    cut_val  += lam * yp_sol[t]

        # --- δ': RHS = 1 − cp_{ij} ---
        for (i, j), constr in self.constrs_sub['delta_p'].items():
            lam = constr.FarkasDual
            if abs(lam) > TOL:
                v_comps['delta_p'][i, j] = lam
                cut_expr += lam               # constant +λ·1
                for t in ta[i][j]:
                    cut_expr -= lam * self.yp[t]
                    cut_val  -= lam * yp_sol[t]
                cut_val += lam

        # Determine cut sense for logging
        sense: str | None = None
        if cut_val > TOL:
            sense = "<="
        elif cut_val < -TOL:
            sense = ">="

        self._log_and_print_inv_farkas(
            v_comps, cut_val, TOL, y_sol, yp_sol, cut_expr, sense
        )
        return cut_expr, cut_val

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_and_print_inv_farkas(
        self,
        v_components: RayComponents,
        cut_val:  float,
        TOL:      float,
        y_sol:    dict[int, float],
        yp_sol:   dict[int, float],
        cut_expr: gp.LinExpr,
        sense:    str | None,
    ) -> None:
        """Log a Farkas cut to the Python logger and (if save_cuts) to JSONL."""
        verbose   = getattr(self, 'verbose',   False)
        save_cuts = getattr(self, 'save_cuts', False)

        if verbose:
            log_lines = [
                f"\n{'='*55}",
                f"INV-BENDERS FARKAS CUT  (iter {self.iteration})",
                f"  cut_val = {cut_val:.6f}   sense = {sense}",
            ]
            for comp, vals in v_components.items():
                if vals and isinstance(vals, dict):
                    log_lines.append(f"  {comp}: { {str(k): round(v, 4) for k, v in vals.items()} }")
            log_lines.append('='*55)
            logger.info("\n".join(log_lines))

        if save_cuts and hasattr(self, 'log_path') and self.log_path:
            # Use 0.5 to filter active binary (y/yp) values, NOT the Farkas
            # dual tolerance — these are master solution values, not dual coefficients.
            log_inv_benders_cut(
                filepath       = self.log_path,
                iteration      = self.iteration,
                node_depth     = 0,
                y_sol          = y_sol,
                yp_sol         = yp_sol,
                v_components   = v_components,
                cut_value      = cut_val,
                tolerance      = 0.5,
                cut_expr       = cut_expr,
                sense          = sense,
            )
