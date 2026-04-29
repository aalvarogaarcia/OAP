# models/mixin/benders_cgsp_mixin.py
"""
BendersCGSPMixin — Deepest Benders Cuts via Cut-Generating Subproblems (CGSP).

For each candidate x_bar from the master, instead of reading FarkasDual or .Pi
from an infeasibility certificate, we solve a *maximisation* LP over the dual
space to find the most-violated cut (the "deepest" cut with respect to a
weighted L₁ norm).

Mathematical reference: Garcia-Munoz (2026) §3 / §4 — notation follows the
user's PDF. The CGSP objective is:

    Y' subproblem (always feasibility):
        max  π^T (b' - B' x_bar)
        s.t. A'^T π ≤ 0   (dual of primal ≥ 0 variables)
             Σ w[i] |π[i]| = 1   (L₁ normalisation)

    Y subproblem (feasibility):
        max  π^T (b - B x_bar)
        s.t. A^T π ≤ 0
             Σ w[i] |π[i]| = 1

    Y subproblem (optimality, objective = "Internal"):
        max  π^T (b - B x_bar) + π₀ (f^T x_bar − η_bar)
        s.t. A^T π ≤ 0,  π₀ free
             Σ w[i] |π[i]| + w₀ |π₀| = 1

The cut is:
    Feasibility:  π^T B x ≤ π^T b      (i.e.  0 ≥ π^T (b - B x))
    Optimality:   π₀ η ≥ π^T (b - B x) + π₀ f^T x

Implementation strategy
-----------------------
Rather than building a full dual LP from scratch (which would require
re-deriving all constraint matrices), the CGSP is built *on top of* the
existing Farkas-mode subproblems: the primal feasibility subproblem already
contains the constraints  A y = b(x_bar), y ≥ 0.  The dual of that LP is
exactly the CGSP.

We create a *new* Gurobi model for each CGSP call (light-weight: tiny LP)
using the constraint information stored in self.constrs_y / self.constrs_yp.

Design contract
---------------
This mixin assumes it is placed *before* BendersFarkasMixin and BendersPiMixin
in the MRO so that it can provide the CGSP entry points that the callback can
dispatch to when use_deepest_cuts=True.

Inherited attributes expected (provided by OAPBaseModel + BendersMasterMixin
+ BendersFarkasMixin/BendersPiMixin after build):
    self.model            : gp.Model  — master MILP
    self.sub_y            : gp.Model  — internal triangulation subproblem
    self.sub_yp           : gp.Model  — external triangulation subproblem
    self.x                : dict[Arc, gp.Var]
    self.constrs_y        : dict
    self.constrs_yp       : dict
    self.N                : int
    self.CH               : IndexArray
    self.V_list           : range
    self.convex_hull_area : float
    self.objective        : str  (only present for "Internal")
    self.eta              : gp.Var  (only present for "Internal")
    self.cut_weights_y    : dict | None
    self.cut_weights_yp   : dict | None
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    from models.typing_oap import IndexArray

logger = logging.getLogger(__name__)

Arc = tuple[int, int]

# Tolerance below which a dual value is treated as zero
_CGSP_TOL: float = 1e-10


class BendersCGSPMixin:
    """Mixin providing Deepest Benders Cuts via Cut-Generating Subproblems.

    Public entry points
    -------------------
    get_cgsp_cut_yp(x_sol)  -> (LinExpr, float, dict)
    get_cgsp_cut_y(x_sol, eta_sol)  -> (LinExpr, float, dict)

    Both return (cut_expr, cut_rhs, witness_dict) where:
        cut_expr  : gp.LinExpr over master x (and eta) variables
        cut_rhs   : float — the right-hand side
        witness   : dict of non-zero dual values (for logging / analysis)
    """

    # --- Type hints for inherited attributes (linter only) ---
    model: gp.Model
    sub_y: gp.Model
    sub_yp: gp.Model
    x: dict[Arc, gp.Var]
    constrs_y: dict
    constrs_yp: dict
    N: int
    CH: IndexArray
    V_list: range
    convex_hull_area: float
    # Optional — only when objective == "Internal"
    # eta: gp.Var
    # objective: str
    # cut_weights_y: dict | None
    # cut_weights_yp: dict | None

    # ------------------------------------------------------------------
    # Helper: extract pi vector from a solved CGSP LP
    # ------------------------------------------------------------------

    def _extract_cgsp_pi(
        self,
        pi_vars: dict[str, gp.Var | gp.tupledict],
        TOL: float = _CGSP_TOL,
    ) -> dict[str, float | dict]:
        """Extract non-zero dual values from a solved CGSP model.

        Parameters
        ----------
        pi_vars:
            Mapping from component name (e.g. 'alpha', 'beta', ...) to the
            corresponding Gurobi variable(s) in the CGSP LP.  Values may be
            either a single gp.Var (scalar) or a gp.tupledict of vars.
        TOL:
            Threshold below which a value is treated as zero.

        Returns
        -------
        A dict mirroring the structure of pi_vars but populated with the
        optimal values, filtered to |value| > TOL.
        """
        result: dict[str, float | dict] = {}
        for name, var_or_td in pi_vars.items():
            if isinstance(var_or_td, gp.Var):
                val = var_or_td.X
                if abs(val) > TOL:
                    result[name] = val
            else:
                # tupledict
                sub: dict = {}
                for k, v in var_or_td.items():
                    val = v.X
                    if abs(val) > TOL:
                        sub[k] = val
                if sub:
                    result[name] = sub
        return result

    # ------------------------------------------------------------------
    # Helper: resolve normalisation weights
    # ------------------------------------------------------------------

    def _resolve_weights(
        self,
        which: str,
        constrs: dict,
        include_pi0: bool = False,
    ) -> dict[str, float | dict[tuple, float]]:
        """Build the weight dictionary for the L₁ normalisation constraint.

        If the user supplied explicit weights via self.cut_weights_y /
        self.cut_weights_yp they are used directly (after validation).
        Otherwise all weights default to 1.0.

        Parameters
        ----------
        which : 'y' | 'yp'
            Selects self.cut_weights_y or self.cut_weights_yp.
        constrs : dict
            The constraint dictionary (self.constrs_y or self.constrs_yp)
            whose keys define the shape of the dual space.
        include_pi0 : bool
            If True, add a 'pi0' key with weight w₀ (defaults to 1.0).

        Returns
        -------
        A dict with the same nested structure as *constrs* (and optionally
        a 'pi0' key) mapping every dual variable to its weight.
        """
        user_weights: dict | None = (
            getattr(self, "cut_weights_y", None)
            if which == "y"
            else getattr(self, "cut_weights_yp", None)
        )

        weights: dict[str, float | dict[tuple, float]] = {}

        for key, val in constrs.items():
            if isinstance(val, dict):
                sub: dict[tuple, float] = {}
                for arc_key in val:
                    if user_weights and key in user_weights and isinstance(user_weights[key], dict):
                        sub[arc_key] = float(user_weights[key].get(arc_key, 1.0))
                    else:
                        sub[arc_key] = 1.0
                weights[key] = sub
            elif isinstance(val, gp.Constr):
                if user_weights and key in user_weights and isinstance(user_weights[key], (int, float)):
                    weights[key] = float(user_weights[key])
                else:
                    weights[key] = 1.0

        if include_pi0:
            if user_weights and "pi0" in user_weights:
                weights["pi0"] = float(user_weights["pi0"])
            else:
                weights["pi0"] = 1.0

        return weights

    # ------------------------------------------------------------------
    # Task 5: _build_cgsp_yp — dual LP for Y' (external triangulation)
    # ------------------------------------------------------------------

    def _build_cgsp_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, gp.Var | gp.tupledict]]:
        """Build the Cut-Generating Subproblem LP for Y' (external subproblem).

        The CGSP for Y' maximises the violation of the current x_sol:

            max  π^T (b'(x_sol) - 0)   [b' depends on x_sol through RHS updates]
            s.t. (structural dual constraints)
                 Σ w[i] |π[i]| = 1   (L₁ normalisation via split π = u - v, u,v ≥ 0)

        Because the primal subproblem Y' is a *feasibility* LP (no objective),
        its dual's only role is to certify infeasibility (Farkas lemma).  The
        CGSP finds the *deepest* such certificate.

        The normalisation is implemented by splitting each dual variable
        π = u - v  (u, v ≥ 0) and adding the constraint  Σ w (u + v) = 1,
        where the sum is over all components of the dual vector.

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current master solution (x_bar).
        TOL : float
            Numerical tolerance.

        Returns
        -------
        cgsp_model : gp.Model
            A ready-to-solve Gurobi LP.
        pi_vars : dict[str, gp.Var | gp.tupledict]
            Mapping from constraint group name to the corresponding π variables
            (already as u - v net variables, stored as gp.Var / gp.tupledict
            of the positive part u; the actual value is u.X - v.X after solve).
            The dict also contains 'u_<name>' and 'v_<name>' keys for the
            positive/negative parts if needed.
        """
        cgsp = gp.Model("cgsp_yp")
        cgsp.setParam("OutputFlag", 0)
        cgsp.setParam("InfUnbdInfo", 1)

        weights = self._resolve_weights("yp", self.constrs_yp, include_pi0=False)

        # Collect all constraint groups that have non-empty dicts or single constrs
        # We build one u/v pair per constraint.

        # We accumulate: objective terms and normalisation terms
        obj_expr = gp.LinExpr()
        norm_expr = gp.LinExpr()
        pi_vars: dict[str, gp.Var | gp.tupledict] = {}

        def _add_pi_scalar(
            group: str, rhs_val: float, weight: float
        ) -> tuple[gp.Var, gp.Var]:
            """Add a scalar dual variable π = u - v to cgsp."""
            u = cgsp.addVar(lb=0.0, name=f"u_{group}")
            v = cgsp.addVar(lb=0.0, name=f"v_{group}")
            obj_expr.addTerms([rhs_val, -rhs_val], [u, v])
            norm_expr.addTerms([weight, weight], [u, v])
            return u, v

        def _add_pi_indexed(
            group: str,
            keys: list,
            rhs_map: dict,
            weight_map: dict,
        ) -> tuple[gp.tupledict, gp.tupledict]:
            """Add indexed dual variables π[k] = u[k] - v[k] for each key."""
            u_td = cgsp.addVars(keys, lb=0.0, name=f"u_{group}")
            v_td = cgsp.addVars(keys, lb=0.0, name=f"v_{group}")
            for k in keys:
                rhs_k = rhs_map.get(k, 0.0)
                w_k = weight_map.get(k, 1.0) if isinstance(weight_map, dict) else weight_map
                obj_expr.addTerms([rhs_k, -rhs_k], [u_td[k], v_td[k]])
                norm_expr.addTerms([w_k, w_k], [u_td[k], v_td[k]])
            return u_td, v_td

        # ---- alpha_p: Σ_t yp[t, adj(i,j)] = 1 - x[i,j]  for (i,j) ∈ A' ----
        if "alpha_p" in self.constrs_yp and self.constrs_yp["alpha_p"]:
            keys_ap = list(self.constrs_yp["alpha_p"].keys())
            rhs_map_ap = {(i, j): 1.0 - x_sol.get((i, j), 0.0) for (i, j) in keys_ap}
            w_map_ap = weights.get("alpha_p", {})
            u_ap, v_ap = _add_pi_indexed("alpha_p", keys_ap, rhs_map_ap, w_map_ap)
            pi_vars["u_alpha_p"] = u_ap
            pi_vars["v_alpha_p"] = v_ap

        # ---- beta_p: flow balance = x[j,i] - x[i,j]  for (i,j) ∈ E'' ----
        if "beta_p" in self.constrs_yp and self.constrs_yp["beta_p"]:
            keys_bp = list(self.constrs_yp["beta_p"].keys())
            rhs_map_bp = {
                (i, j): x_sol.get((j, i), 0.0) - x_sol.get((i, j), 0.0)
                for (i, j) in keys_bp
            }
            w_map_bp = weights.get("beta_p", {})
            u_bp, v_bp = _add_pi_indexed("beta_p", keys_bp, rhs_map_bp, w_map_bp)
            pi_vars["u_beta_p"] = u_bp
            pi_vars["v_beta_p"] = v_bp

        # ---- gamma_p: Σ_t yp[t] ≥ x[j,i]  for (i,j) ∈ A'' ----
        if "gamma_p" in self.constrs_yp and self.constrs_yp["gamma_p"]:
            keys_gp = list(self.constrs_yp["gamma_p"].keys())
            rhs_map_gp = {(i, j): x_sol.get((j, i), 0.0) for (i, j) in keys_gp}
            w_map_gp = weights.get("gamma_p", {})
            u_gp, v_gp = _add_pi_indexed("gamma_p", keys_gp, rhs_map_gp, w_map_gp)
            pi_vars["u_gamma_p"] = u_gp
            pi_vars["v_gamma_p"] = v_gp

        # ---- delta_p: Σ_t yp[t] ≤ 1 - x[i,j]  for (i,j) ∈ A'' ----
        if "delta_p" in self.constrs_yp and self.constrs_yp["delta_p"]:
            keys_dp = list(self.constrs_yp["delta_p"].keys())
            rhs_map_dp = {(i, j): 1.0 - x_sol.get((i, j), 0.0) for (i, j) in keys_dp}
            w_map_dp = weights.get("delta_p", {})
            u_dp, v_dp = _add_pi_indexed("delta_p", keys_dp, rhs_map_dp, w_map_dp)
            pi_vars["u_delta_p"] = u_dp
            pi_vars["v_delta_p"] = v_dp

        # ---- global_p: Σ_t yp[t] = N - |CH| ----
        if "global_p" in self.constrs_yp and isinstance(self.constrs_yp["global_p"], gp.Constr):
            rhs_global = self.constrs_yp["global_p"].RHS
            w_global = float(weights.get("global_p", 1.0))
            u_g, v_g = _add_pi_scalar("global_p", rhs_global, w_global)
            pi_vars["u_global_p"] = u_g
            pi_vars["v_global_p"] = v_g

        # ---- r1_p: area balance (fixed RHS = CH_area) ----
        if "r1_p" in self.constrs_yp and isinstance(self.constrs_yp["r1_p"], gp.Constr):
            w_r1 = float(weights.get("r1_p", 1.0))
            u_r1, v_r1 = _add_pi_scalar("r1_p", self.convex_hull_area, w_r1)
            pi_vars["u_r1_p"] = u_r1
            pi_vars["v_r1_p"] = v_r1

        # ---- r2_p: arc coverage (fixed RHS = 1) ----
        if "r2_p" in self.constrs_yp and self.constrs_yp["r2_p"]:
            keys_r2 = list(self.constrs_yp["r2_p"].keys())
            rhs_map_r2 = {k: 1.0 for k in keys_r2}
            w_map_r2 = weights.get("r2_p", {})
            u_r2, v_r2 = _add_pi_indexed("r2_p", keys_r2, rhs_map_r2, w_map_r2)
            pi_vars["u_r2_p"] = u_r2
            pi_vars["v_r2_p"] = v_r2

        # ---- r3_p: crossing exclusion — RHS = 1 - x[i,j] - x[s,k] ----
        if "r3_p" in self.constrs_yp and self.constrs_yp["r3_p"]:
            keys_r3 = list(self.constrs_yp["r3_p"].keys())
            rhs_map_r3 = {
                (i, j, k, s): 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0)
                for (i, j, k, s) in keys_r3
            }
            w_map_r3 = weights.get("r3_p", {})
            u_r3, v_r3 = _add_pi_indexed("r3_p", keys_r3, rhs_map_r3, w_map_r3)
            pi_vars["u_r3_p"] = u_r3
            pi_vars["v_r3_p"] = v_r3

        # ---- L₁ normalisation: Σ w (u + v) = 1 ----
        # Only add if there are variables to normalise
        cgsp.update()
        all_vars = cgsp.getVars()
        if all_vars:
            cgsp.addConstr(norm_expr == 1.0, name="l1_norm")

        # ---- Objective: maximise violation ----
        cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
        cgsp.update()

        return cgsp, pi_vars

    # ------------------------------------------------------------------
    # Task 6: _build_cgsp_y — dual LP for Y (internal triangulation)
    # ------------------------------------------------------------------

    def _build_cgsp_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, gp.Var | gp.tupledict], gp.Var | None]:
        """Build the Cut-Generating Subproblem LP for Y (internal subproblem).

        For Fekete objective (no eta):
            max  π^T (b - B x_bar)
            s.t. normalisation

        For Internal objective (with eta):
            max  π^T (b - B x_bar) + π₀ (f^T x_bar − η_bar)
            s.t. normalisation including w₀ |π₀|

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current master solution.
        eta_sol : float
            Current value of η in the master solution (0.0 for Fekete objective).
        TOL : float
            Numerical tolerance.

        Returns
        -------
        cgsp_model : gp.Model
        pi_vars    : dict[str, gp.Var | gp.tupledict]
        pi0_var    : gp.Var | None — the π₀ dual variable (None for Fekete)
        """
        is_internal = (getattr(self, "objective", "Fekete") == "Internal")
        include_pi0 = is_internal

        cgsp = gp.Model("cgsp_y")
        cgsp.setParam("OutputFlag", 0)
        cgsp.setParam("InfUnbdInfo", 1)

        weights = self._resolve_weights("y", self.constrs_y, include_pi0=include_pi0)

        obj_expr = gp.LinExpr()
        norm_expr = gp.LinExpr()
        pi_vars: dict[str, gp.Var | gp.tupledict] = {}

        def _add_pi_scalar(
            group: str, rhs_val: float, weight: float
        ) -> tuple[gp.Var, gp.Var]:
            u = cgsp.addVar(lb=0.0, name=f"u_{group}")
            v = cgsp.addVar(lb=0.0, name=f"v_{group}")
            obj_expr.addTerms([rhs_val, -rhs_val], [u, v])
            norm_expr.addTerms([weight, weight], [u, v])
            return u, v

        def _add_pi_indexed(
            group: str,
            keys: list,
            rhs_map: dict,
            weight_map: dict,
        ) -> tuple[gp.tupledict, gp.tupledict]:
            u_td = cgsp.addVars(keys, lb=0.0, name=f"u_{group}")
            v_td = cgsp.addVars(keys, lb=0.0, name=f"v_{group}")
            for k in keys:
                rhs_k = rhs_map.get(k, 0.0)
                w_k = weight_map.get(k, 1.0) if isinstance(weight_map, dict) else weight_map
                obj_expr.addTerms([rhs_k, -rhs_k], [u_td[k], v_td[k]])
                norm_expr.addTerms([w_k, w_k], [u_td[k], v_td[k]])
            return u_td, v_td

        # ---- alpha: Σ_t y[t, adj(i,j)] = x[i,j]  for (i,j) ∈ A' ----
        if "alpha" in self.constrs_y and self.constrs_y["alpha"]:
            keys_a = list(self.constrs_y["alpha"].keys())
            rhs_map_a = {(i, j): x_sol.get((i, j), 0.0) for (i, j) in keys_a}
            w_map_a = weights.get("alpha", {})
            u_a, v_a = _add_pi_indexed("alpha", keys_a, rhs_map_a, w_map_a)
            pi_vars["u_alpha"] = u_a
            pi_vars["v_alpha"] = v_a

        # ---- beta: flow balance = x[i,j] - x[j,i]  for (i,j) ∈ E'' ----
        if "beta" in self.constrs_y and self.constrs_y["beta"]:
            keys_b = list(self.constrs_y["beta"].keys())
            rhs_map_b = {
                (i, j): x_sol.get((i, j), 0.0) - x_sol.get((j, i), 0.0)
                for (i, j) in keys_b
            }
            w_map_b = weights.get("beta", {})
            u_b, v_b = _add_pi_indexed("beta", keys_b, rhs_map_b, w_map_b)
            pi_vars["u_beta"] = u_b
            pi_vars["v_beta"] = v_b

        # ---- gamma: Σ_t y[t] ≥ x[i,j]  for (i,j) ∈ A'' ----
        if "gamma" in self.constrs_y and self.constrs_y["gamma"]:
            keys_g = list(self.constrs_y["gamma"].keys())
            rhs_map_g = {(i, j): x_sol.get((i, j), 0.0) for (i, j) in keys_g}
            w_map_g = weights.get("gamma", {})
            u_g, v_g = _add_pi_indexed("gamma", keys_g, rhs_map_g, w_map_g)
            pi_vars["u_gamma"] = u_g
            pi_vars["v_gamma"] = v_g

        # ---- delta: Σ_t y[t] ≤ 1 - x[j,i]  for (i,j) ∈ A'' ----
        if "delta" in self.constrs_y and self.constrs_y["delta"]:
            keys_d = list(self.constrs_y["delta"].keys())
            rhs_map_d = {(i, j): 1.0 - x_sol.get((j, i), 0.0) for (i, j) in keys_d}
            w_map_d = weights.get("delta", {})
            u_d, v_d = _add_pi_indexed("delta", keys_d, rhs_map_d, w_map_d)
            pi_vars["u_delta"] = u_d
            pi_vars["v_delta"] = v_d

        # ---- global: Σ_t y[t] = N - 2 ----
        if "global" in self.constrs_y and isinstance(self.constrs_y["global"], gp.Constr):
            rhs_global = self.constrs_y["global"].RHS
            w_global = float(weights.get("global", 1.0))
            u_gl, v_gl = _add_pi_scalar("global", rhs_global, w_global)
            pi_vars["u_global"] = u_gl
            pi_vars["v_global"] = v_gl

        # ---- r1: area balance (fixed RHS = CH_area) ----
        if "r1" in self.constrs_y and isinstance(self.constrs_y["r1"], gp.Constr):
            w_r1 = float(weights.get("r1", 1.0))
            u_r1, v_r1 = _add_pi_scalar("r1", self.convex_hull_area, w_r1)
            pi_vars["u_r1"] = u_r1
            pi_vars["v_r1"] = v_r1

        # ---- r2: arc coverage (fixed RHS = 1) ----
        if "r2" in self.constrs_y and self.constrs_y["r2"]:
            keys_r2 = list(self.constrs_y["r2"].keys())
            rhs_map_r2 = {k: 1.0 for k in keys_r2}
            w_map_r2 = weights.get("r2", {})
            u_r2, v_r2 = _add_pi_indexed("r2", keys_r2, rhs_map_r2, w_map_r2)
            pi_vars["u_r2"] = u_r2
            pi_vars["v_r2"] = v_r2

        # ---- r3: crossing exclusion — RHS = 1 - x[j,i] - x[k,s] ----
        if "r3" in self.constrs_y and self.constrs_y["r3"]:
            keys_r3 = list(self.constrs_y["r3"].keys())
            rhs_map_r3 = {
                (i, j, k, s): 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0)
                for (i, j, k, s) in keys_r3
            }
            w_map_r3 = weights.get("r3", {})
            u_r3, v_r3 = _add_pi_indexed("r3", keys_r3, rhs_map_r3, w_map_r3)
            pi_vars["u_r3"] = u_r3
            pi_vars["v_r3"] = v_r3

        # ---- π₀ for Internal objective ----
        pi0_var: gp.Var | None = None
        if include_pi0:
            # π₀ is free: split as π₀ = π₀_pos - π₀_neg
            pi0_pos = cgsp.addVar(lb=0.0, name="pi0_pos")
            pi0_neg = cgsp.addVar(lb=0.0, name="pi0_neg")
            # Objective term: π₀ * (f^T x_bar − η_bar)
            # f^T x_bar is the primal objective value of x_sol under the cost
            # function stored in self.model (not re-computed here); the subproblem
            # ObjVal already reflects this via eta_sol in the master.
            # We approximate: contribution is π₀ * (sub_y.ObjVal − η_bar) but
            # since the primal sub_y may not be solved yet, we use the sign from
            # the previously computed sub_y.ObjVal if available, else 0.
            sub_y_obj: float = 0.0
            if hasattr(self, "sub_y") and self.sub_y.Status == GRB.OPTIMAL:
                sub_y_obj = float(self.sub_y.ObjVal)
            pi0_contribution = sub_y_obj - eta_sol
            w_pi0 = float(weights.get("pi0", 1.0))
            obj_expr.addTerms([pi0_contribution, -pi0_contribution], [pi0_pos, pi0_neg])
            norm_expr.addTerms([w_pi0, w_pi0], [pi0_pos, pi0_neg])
            pi_vars["pi0_pos"] = pi0_pos
            pi_vars["pi0_neg"] = pi0_neg
            # Store net variable reference via a helper attribute name
            pi0_var = pi0_pos  # caller uses pi0_pos.X - pi0_neg.X for the net value

        # ---- L₁ normalisation ----
        cgsp.update()
        all_vars = cgsp.getVars()
        if all_vars:
            cgsp.addConstr(norm_expr == 1.0, name="l1_norm")

        # ---- Objective ----
        cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
        cgsp.update()

        return cgsp, pi_vars, pi0_var

    # ------------------------------------------------------------------
    # Task 8: Public entry points
    # ------------------------------------------------------------------

    def get_cgsp_cut_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict]:
        """Generate the deepest feasibility cut for Y' via CGSP.

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current master solution.
        TOL : float
            Numerical zero threshold.

        Returns
        -------
        cut_expr : gp.LinExpr
            Left-hand side of the cut over master variables x.
            The cut is:  cut_expr <= cut_rhs
        cut_rhs : float
            Right-hand side (0.0 for feasibility cuts).
        witness : dict
            Non-zero dual components (for logging and analysis).
        """
        cgsp, pi_vars = self._build_cgsp_yp(x_sol, TOL=TOL)
        cgsp.optimize()

        cut_expr = gp.LinExpr()
        cut_rhs = 0.0
        witness: dict = {}

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y') did not solve to optimality (status=%d).", cgsp.Status)
            return cut_expr, cut_rhs, witness

        if cgsp.ObjVal <= TOL:
            # No violated cut found
            return cut_expr, cut_rhs, witness

        # ---- Reconstruct the Benders cut in the master variable space ----
        # The cut is: Σ_i π_i * (b_i(x) - 0) ≤ 0  ↔  Σ_i π_i * b_i(x) ≤ 0
        # where b_i(x) is the i-th RHS expressed as a linear function of x.

        # Helper: retrieve net value (u - v) for a variable group
        def _net(u_key: str, v_key: str) -> dict:
            out: dict = {}
            u_vars = pi_vars.get(u_key)
            v_vars = pi_vars.get(v_key)
            if u_vars is None or v_vars is None:
                return out
            if isinstance(u_vars, gp.Var):
                val = u_vars.X - v_vars.X  # type: ignore[union-attr]
                if abs(val) > TOL:
                    out["scalar"] = val
            else:
                for k in u_vars.keys():
                    val = u_vars[k].X - v_vars[k].X  # type: ignore[index]
                    if abs(val) > TOL:
                        out[k] = val
            return out

        # alpha_p: RHS = 1 - x[i,j]  → contribution = π * (1 - x[i,j])
        ap_net = _net("u_alpha_p", "v_alpha_p")
        if ap_net:
            witness["alpha_p"] = ap_net
            for (i, j), pi_val in ap_net.items():
                cut_expr += pi_val * (1.0 - self.x[i, j])
                # cut_rhs stays 0 (accumulated in cut_expr)

        # beta_p: RHS = x[j,i] - x[i,j]  → contribution = π * (x[j,i] - x[i,j])
        bp_net = _net("u_beta_p", "v_beta_p")
        if bp_net:
            witness["beta_p"] = bp_net
            for (i, j), pi_val in bp_net.items():
                cut_expr += pi_val * (self.x[j, i] - self.x[i, j])

        # gamma_p: RHS = x[j,i]  → contribution = π * x[j,i]
        gp_net = _net("u_gamma_p", "v_gamma_p")
        if gp_net:
            witness["gamma_p"] = gp_net
            for (i, j), pi_val in gp_net.items():
                cut_expr += pi_val * self.x[j, i]

        # delta_p: RHS = 1 - x[i,j]  → contribution = π * (1 - x[i,j])
        dp_net = _net("u_delta_p", "v_delta_p")
        if dp_net:
            witness["delta_p"] = dp_net
            for (i, j), pi_val in dp_net.items():
                cut_expr += pi_val * (1.0 - self.x[i, j])

        # global_p: RHS = N - |CH|  (constant — move to right-hand side)
        gl_net = _net("u_global_p", "v_global_p")
        if gl_net and "scalar" in gl_net:
            pi_val = gl_net["scalar"]
            rhs_val = self.constrs_yp["global_p"].RHS
            cut_rhs += pi_val * rhs_val
            witness["global_p"] = pi_val

        # r1_p: area balance (constant RHS = CH_area)
        r1_net = _net("u_r1_p", "v_r1_p")
        if r1_net and "scalar" in r1_net:
            pi_val = r1_net["scalar"]
            cut_rhs += pi_val * self.convex_hull_area
            witness["r1_p"] = pi_val

        # r2_p: arc coverage (constant RHS = 1)
        r2_net = _net("u_r2_p", "v_r2_p")
        if r2_net:
            witness["r2_p"] = r2_net
            for _k, pi_val in r2_net.items():
                cut_rhs += pi_val * 1.0

        # r3_p: crossing exclusion — RHS = 1 - x[i,j] - x[s,k]
        r3_net = _net("u_r3_p", "v_r3_p")
        if r3_net:
            witness["r3_p"] = r3_net
            for (i, j, k, s), pi_val in r3_net.items():
                cut_expr += pi_val * (1.0 - self.x[i, j] - self.x[s, k])

        return cut_expr, cut_rhs, witness

    def get_cgsp_cut_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict]:
        """Generate the deepest feasibility (or optimality) cut for Y via CGSP.

        For Fekete objective: returns a feasibility cut  cut_expr <= cut_rhs.
        For Internal objective: if the subproblem optimal value exceeds eta_sol,
            returns an optimality cut; otherwise a feasibility cut.

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current master solution.
        eta_sol : float
            Current value of η (0.0 for Fekete objective).
        TOL : float
            Numerical zero threshold.

        Returns
        -------
        cut_expr : gp.LinExpr
            Left-hand side of the cut (over x and optionally eta).
        cut_rhs : float
            Right-hand side.
        witness : dict
            Non-zero dual components.
        """
        cgsp, pi_vars, pi0_var = self._build_cgsp_y(x_sol, eta_sol=eta_sol, TOL=TOL)
        cgsp.optimize()

        cut_expr = gp.LinExpr()
        cut_rhs = 0.0
        witness: dict = {}

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y) did not solve to optimality (status=%d).", cgsp.Status)
            return cut_expr, cut_rhs, witness

        if cgsp.ObjVal <= TOL:
            # No violated cut
            return cut_expr, cut_rhs, witness

        def _net(u_key: str, v_key: str) -> dict:
            out: dict = {}
            u_vars = pi_vars.get(u_key)
            v_vars = pi_vars.get(v_key)
            if u_vars is None or v_vars is None:
                return out
            if isinstance(u_vars, gp.Var):
                val = u_vars.X - v_vars.X  # type: ignore[union-attr]
                if abs(val) > TOL:
                    out["scalar"] = val
            else:
                for k in u_vars.keys():
                    val = u_vars[k].X - v_vars[k].X  # type: ignore[index]
                    if abs(val) > TOL:
                        out[k] = val
            return out

        is_internal = (getattr(self, "objective", "Fekete") == "Internal")

        # Determine if this is an optimality cut
        is_optimality_cut = False
        if is_internal and pi0_var is not None:
            pi0_neg_var = pi_vars.get("pi0_neg")
            pi0_net = pi0_var.X - (pi0_neg_var.X if pi0_neg_var is not None else 0.0)
            if abs(pi0_net) > TOL:
                is_optimality_cut = True
                witness["pi0"] = pi0_net
        else:
            pi0_net = 0.0

        # alpha: RHS = x[i,j]
        a_net = _net("u_alpha", "v_alpha")
        if a_net:
            witness["alpha"] = a_net
            for (i, j), pi_val in a_net.items():
                cut_expr += pi_val * self.x[i, j]

        # beta: RHS = x[i,j] - x[j,i]
        b_net = _net("u_beta", "v_beta")
        if b_net:
            witness["beta"] = b_net
            for (i, j), pi_val in b_net.items():
                cut_expr += pi_val * (self.x[i, j] - self.x[j, i])

        # gamma: RHS = x[i,j]
        g_net = _net("u_gamma", "v_gamma")
        if g_net:
            witness["gamma"] = g_net
            for (i, j), pi_val in g_net.items():
                cut_expr += pi_val * self.x[i, j]

        # delta: RHS = 1 - x[j,i]
        d_net = _net("u_delta", "v_delta")
        if d_net:
            witness["delta"] = d_net
            for (i, j), pi_val in d_net.items():
                cut_expr += pi_val * (1.0 - self.x[j, i])

        # global: RHS = N - 2 (constant)
        gl_net = _net("u_global", "v_global")
        if gl_net and "scalar" in gl_net:
            pi_val = gl_net["scalar"]
            rhs_val = self.constrs_y["global"].RHS
            cut_rhs += pi_val * rhs_val
            witness["global"] = pi_val

        # r1: area balance (constant RHS = CH_area)
        r1_net = _net("u_r1", "v_r1")
        if r1_net and "scalar" in r1_net:
            pi_val = r1_net["scalar"]
            cut_rhs += pi_val * self.convex_hull_area
            witness["r1"] = pi_val

        # r2: arc coverage (constant RHS = 1)
        r2_net = _net("u_r2", "v_r2")
        if r2_net:
            witness["r2"] = r2_net
            for _k, pi_val in r2_net.items():
                cut_rhs += pi_val * 1.0

        # r3: crossing exclusion — RHS = 1 - x[j,i] - x[k,s]
        r3_net = _net("u_r3", "v_r3")
        if r3_net:
            witness["r3"] = r3_net
            for (i, j, k, s), pi_val in r3_net.items():
                cut_expr += pi_val * (1.0 - self.x[j, i] - self.x[k, s])

        # For Internal optimality cuts: η ≥ cut_expr - cut_rhs_const
        # Cut form: π₀ η ≥ π^T b(x) - π^T B x  →  if π₀ > 0: η ≥ (π^T b - π^T B x) / π₀
        # We return the normalised form: cut_expr (involving x and eta) and cut_rhs
        if is_optimality_cut and is_internal:
            # Return as: eta >= (cut_expr - cut_rhs_const) / pi0_net
            # Or equivalently in linear form: pi0_net * eta >= cut_expr - cut_rhs_const
            # Caller handles via: model.addConstr(pi0_net * self.eta >= cut_expr - cut_rhs)
            if abs(pi0_net) > TOL:
                # Scale cut by 1/pi0_net to get standard: eta >= rhs_expr
                factor = 1.0 / pi0_net
                # multiply all coefficients
                new_cut_expr = gp.LinExpr()
                for i_var in range(cut_expr.size()):
                    new_cut_expr.addTerms(
                        cut_expr.getCoeff(i_var) * factor,
                        cut_expr.getVar(i_var),
                    )
                new_cut_rhs = cut_rhs * factor
                witness["is_optimality_cut"] = True
                # Add eta term: the cut is eta >= new_cut_expr (moved to LHS)
                # Return (new_cut_expr, new_cut_rhs, witness) where caller does:
                #   self.eta >= new_cut_expr OR new_cut_expr <= self.eta
                return new_cut_expr, new_cut_rhs, witness

        return cut_expr, cut_rhs, witness
