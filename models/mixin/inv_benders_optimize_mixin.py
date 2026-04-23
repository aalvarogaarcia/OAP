# models/mixin/inv_benders_optimize_mixin.py
"""Callback orchestration and solve() for the Inverted Benders model."""
from __future__ import annotations

import logging
import os

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


class InvBendersOptimizeMixin:
    """Mixin that owns the MIPSOL callback and ``solve()`` for the inverted model.

    Attributes expected from sibling mixins / ``OAPBaseModel``:

    * ``model``         – master ``gp.Model`` (``y``/``yp`` binary)
    * ``sub_x``         – subproblem LP ``gp.Model``
    * ``y``, ``yp``     – ``dict[int, gp.Var]`` (master binary variables)
    * ``V_list``        – ``range`` over triangle indices
    * ``iteration``     – ``int`` cut counter
    * ``log_path``      – ``str`` JSONL log path
    * ``update_sub_rhs(y_sol, yp_sol)``   provided by ``InvBendersSubMixin``
    * ``get_farkas_cut(y_sol, yp_sol, TOL)`` provided by ``InvBendersSubMixin``
    """

    # Provided by OAPBaseModel
    model:     gp.Model
    V_list:    range
    # Provided by InvBendersMasterMixin
    iteration:  int
    save_cuts:  bool
    verbose:    bool
    log_path:   str
    y:   dict[int, gp.Var]
    yp:  dict[int, gp.Var]
    y_results:  list[int]
    yp_results: list[int]

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _inv_benders_callback(self, model: gp.Model, where: int) -> None:
        """Gurobi callback: intercept integer candidate solutions (MIPSOL)
        and reject any that yield an infeasible subproblem via a Farkas cut."""
        if where != GRB.Callback.MIPSOL:
            return

        # 1. Extract current master solution (y*, yp*)
        y_sol  = {t: model.cbGetSolution(self.y[t])  for t in self.V_list}
        yp_sol = {t: model.cbGetSolution(self.yp[t]) for t in self.V_list}

        # 2. Update subproblem RHS with the new y*/yp*
        self.iteration += 1
        self.update_sub_rhs(y_sol, yp_sol)

        # 3. Solve sub_x LP
        self.sub_x.optimize()

        TOL = 1e-6

        # 4. Infeasible → extract Farkas cut and inject as lazy constraint
        if self.sub_x.Status == GRB.INFEASIBLE:
            cut_expr, cut_val = self.get_farkas_cut(y_sol, yp_sol, TOL)
            if cut_val > TOL:
                model.cbLazy(cut_expr <= 0)
            elif cut_val < -TOL:
                model.cbLazy(cut_expr >= 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        time_limit: int  = 7200,
        verbose:    bool = False,
        save_cuts:  bool = False,
        relaxed:    bool = False,
    ) -> None:
        """Configure the master and run optimisation with the Benders callback.

        Args:
            time_limit: Wall-clock limit in seconds (default 7200).
            verbose:    Enable Gurobi console output and Python logger output.
            save_cuts:  Write each Farkas cut to ``self.log_path`` (JSONL).
                        The file is *truncated* at the start of each run.
            relaxed:    If True, solve only the LP relaxation via the manual
                        Benders loop (useful for debugging).
        """
        logger.info(
            f"Starting InvBenders solve — limit={time_limit}s  "
            f"verbose={verbose}  save_cuts={save_cuts}"
        )
        self.verbose   = verbose
        self.save_cuts = save_cuts

        # Truncate log at the start of each fresh run
        if save_cuts:
            parent = os.path.dirname(self.log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.log_path, "w"):
                pass

        # Mandatory Gurobi params
        self.model.Params.TimeLimit = time_limit
        if not verbose:
            self.model.Params.OutputFlag = 0

        if relaxed:
            self.solve_lp_relaxation(
                time_limit=time_limit,
                verbose=verbose,
                save_cuts=save_cuts,
            )
        else:
            # LazyConstraints required for cbLazy in the MIPSOL callback
            self.model.Params.LazyConstraints = 1
            self.model.optimize(self._inv_benders_callback)

        # Extract results if any integer solution was found
        if self.model.SolCount > 0:
            self.y_results  = [t for t, v in self.y.items()  if v.X > 0.5]
            self.yp_results = [t for t, v in self.yp.items() if v.X > 0.5]
            logger.info(
                f"Solution found: {len(self.y_results)} internal triangles, "
                f"{len(self.yp_results)} external triangles."
            )

        if self.model.Status == GRB.OPTIMAL:
            logger.info(f"Optimal objective: {self.model.ObjVal:.6f}")
        else:
            logger.warning(f"Solve ended with status: {self.model.Status}")

    def solve_lp_relaxation(
        self,
        time_limit: int  = 7200,
        verbose:    bool = False,
        save_cuts:  bool = False,
    ) -> None:
        """Solve the LP relaxation via a manual Benders cutting-plane loop.

        Gurobi does not fire ``MIPSOL`` callbacks for continuous models, so
        the integrality constraints are dropped and cuts are added directly
        as normal constraints until the subproblem reports feasible.
        """
        logger.info("Starting LP relaxation (manual Benders loop)...")
        self.verbose   = verbose
        self.save_cuts = save_cuts

        # Drop integrality
        for v in self.model.getVars():
            if v.VType != GRB.CONTINUOUS:
                v.VType = GRB.CONTINUOUS
        self.model.update()

        self.model.Params.Presolve        = 1
        self.model.Params.LazyConstraints = 0   # no callbacks in LP mode
        self.model.Params.TimeLimit       = time_limit
        if not verbose:
            self.model.Params.OutputFlag = 0

        TOL = 1e-6
        converged = False
        self.iteration = 0

        while not converged:
            self.iteration += 1
            if verbose:
                logger.info(f"=== LP iteration {self.iteration} ===")

            self.model.optimize()

            if self.model.Status == GRB.INFEASIBLE:
                logger.warning(
                    f"Master LP became INFEASIBLE at iteration {self.iteration}."
                )
                break

            y_sol  = {t: v.X for t, v in self.y.items()}
            yp_sol = {t: v.X for t, v in self.yp.items()}

            self.update_sub_rhs(y_sol, yp_sol)
            self.sub_x.optimize()

            if self.sub_x.Status == GRB.INFEASIBLE:
                cut_expr, cut_val = self.get_farkas_cut(y_sol, yp_sol, TOL)
                if cut_val > TOL:
                    self.model.addConstr(
                        cut_expr <= 0,
                        name=f"lp_inv_cut_{self.iteration}",
                    )
                elif cut_val < -TOL:
                    self.model.addConstr(
                        cut_expr >= 0,
                        name=f"lp_inv_cut_{self.iteration}",
                    )
            else:
                converged = True
                logger.info(
                    f"LP relaxation converged after {self.iteration} iterations."
                )
