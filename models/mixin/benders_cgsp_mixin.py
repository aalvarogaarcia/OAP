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
from typing import TYPE_CHECKING, Any

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
    constrs_y: dict[str, Any]
    constrs_yp: dict[str, Any]
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
        pi_vars: dict[str, gp.Var | gp.tupledict[Any, gp.Var]],
        TOL: float = _CGSP_TOL,
    ) -> dict[str, float | dict[str, Any]]:
        """Extract non-zero dual values from a solved CGSP model.

        Parameters
        ----------
        pi_vars:
            Mapping from component name (e.g. 'alpha', 'beta', ...) to the
            corresponding Gurobi variable(s) in the CGSP LP.  Values may be
            either a single gp.Var (scalar) or a gp.tupledict[Any, gp.Var] of vars.
        TOL:
            Threshold below which a value is treated as zero.

        Returns
        -------
        A dict mirroring the structure of pi_vars but populated with the
        optimal values, filtered to |value| > TOL.
        """
        result: dict[str, float | dict[str, Any]] = {}
        for name, var_or_td in pi_vars.items():
            if isinstance(var_or_td, gp.Var):
                val = var_or_td.X
                if abs(val) > TOL:
                    result[name] = val
            else:
                # tupledict
                sub: dict[str, Any] = {}
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
        constrs: dict[str, Any],
        include_pi0: bool = False,
        x_sol: dict[Arc, float] | None = None,
    ) -> dict[str, float | dict[tuple[Any, ...], float]]:
        """Build the weight dictionary for the L₁ normalisation constraint.

        If the user supplied explicit weights via self.cut_weights_y /
        self.cut_weights_yp they are used directly (after validation).

        If self.cgsp_norm == "relaxed_l1" and x_sol is provided, weights are
        computed as  w[i] = 1 / (|b_i(x_bar)| + ε)  (Hosseini & Turner §4.2).
        This gives larger weights to dual variables with small RHS — the
        normalisation concentrates mass on the most informative directions.

        Otherwise all weights default to 1.0 (MISD / uniform).

        Parameters
        ----------
        which : 'y' | 'yp'
            Selects self.cut_weights_y or self.cut_weights_yp.
        constrs : dict
            The constraint dictionary (self.constrs_y or self.constrs_yp)
            whose keys define the shape of the dual space.
        include_pi0 : bool
            If True, add a 'pi0' key with weight w₀ (defaults to 1.0).
        x_sol : dict[Arc, float] | None
            Current master solution.  Required when cgsp_norm='relaxed_l1'.

        Returns
        -------
        A dict with the same nested structure as *constrs* (and optionally
        a 'pi0' key) mapping every dual variable to its weight.
        """
        _RL1_EPS = 1e-6  # ε for Relaxed-ℓ₁ denominator

        use_relaxed_l1 = getattr(self, "cgsp_norm", "misd") == "relaxed_l1" and x_sol is not None

        user_weights: dict[str, Any] | None = (
            getattr(self, "cut_weights_y", None) if which == "y" else getattr(self, "cut_weights_yp", None)
        )

        weights: dict[str, float | dict[tuple[Any, ...], float]] = {}

        for key, val in constrs.items():
            if isinstance(val, dict):
                sub: dict[tuple[Any, ...], float] = {}
                for arc_key in val:
                    if user_weights and key in user_weights and isinstance(user_weights[key], dict):
                        sub[arc_key] = float(user_weights[key].get(arc_key, 1.0))
                    elif use_relaxed_l1:
                        rhs_val = _rhs_for_key(key, arc_key, x_sol, self)  # type: ignore[arg-type]
                        sub[arc_key] = 1.0 / (abs(rhs_val) + _RL1_EPS)
                    else:
                        sub[arc_key] = 1.0
                weights[key] = sub
            elif isinstance(val, gp.Constr):
                if user_weights and key in user_weights and isinstance(user_weights[key], (int, float)):
                    weights[key] = float(user_weights[key])
                elif use_relaxed_l1:
                    rhs_val = float(val.RHS)
                    weights[key] = 1.0 / (abs(rhs_val) + _RL1_EPS)
                else:
                    weights[key] = 1.0

        if include_pi0:
            if user_weights and "pi0" in user_weights:
                weights["pi0"] = float(user_weights["pi0"])
            else:
                weights["pi0"] = 1.0

        return weights

    # ------------------------------------------------------------------
    # Helper: static Relaxed-ℓ₁ weights from constraint structure
    # ------------------------------------------------------------------

    def _compute_relaxed_l1_weights(
        self,
        which: str,
    ) -> dict[str, float | dict[tuple[Any, ...], float]]:
        """Return static Relaxed-ℓ₁ weights derived from the constraint structure.

        Called at build time (after ``build_farkas/pi_subproblems``) when
        ``use_deepest_cuts=True`` and ``cgsp_norm='relaxed_l1'`` and no explicit
        ``cut_weights_y/yp`` were supplied by the caller.

        Weights are based on the number of master x-variables that appear in
        each constraint's RHS  ``b_i(x̄)``  (Hosseini & Turner 2025, §3.3.2):

            w = 1   for groups whose RHS contains one x-variable
                    (alpha, gamma, delta and their ``_p`` counterparts)
            w = 2   for groups whose RHS contains two x-variables
                    (beta: ``x[j,i] - x[i,j]``; r3: ``1 - x[a] - x[b]``)
            w = 0   for constant-RHS groups whose RHS does not depend on x̄
                    (global, r1, r2 and their ``_p`` counterparts)

        This matches the weight scheme documented and used by
        ``BendersDDMAMixin._get_ddma_weights``, extended here to the full
        nested-dict format expected by ``_resolve_weights``.

        Parameters
        ----------
        which : 'y' | 'yp'
            Selects ``self.constrs_y`` (``which='y'``) or
            ``self.constrs_yp`` (``which='yp'``).

        Returns
        -------
        dict
            Mirrors the structure of ``constrs_y / constrs_yp``:

            * If a constraint-group value is a ``dict[Arc, gp.Constr]``,
              the returned entry is ``dict[Arc, float]`` with every arc
              mapped to the group's scalar weight.
            * If a constraint-group value is a scalar ``gp.Constr``,
              the returned entry is a single ``float``.

            Unknown group names default to weight ``1.0``.

        Notes
        -----
        These are *structural* (x̄-independent) approximations of the true
        Relaxed-ℓ₁ weights  ``1/(|b_i(x̄)| + ε)``.  The dynamic per-callback
        version is computed inside ``_resolve_weights`` when ``cgsp_norm='relaxed_l1'``
        and ``x_sol`` is provided.  Storing static weights here causes
        ``_resolve_weights`` to enter the ``user_weights`` branch, which is
        intentional: it pre-loads the normalisation with structural information
        before the first x̄ is available.
        """
        constrs = self.constrs_yp if which == "yp" else self.constrs_y
        suffix = "_p" if which == "yp" else ""

        # Group-level structural weights — matches BendersDDMAMixin._get_ddma_weights
        _group_weight: dict[str, float] = {
            f"alpha{suffix}": 1.0,  # RHS = 1 - x[i,j]          → 1 x-var
            f"beta{suffix}": 2.0,  # RHS = x[j,i] - x[i,j]     → 2 x-vars
            f"gamma{suffix}": 1.0,  # RHS = x[j,i]               → 1 x-var
            f"delta{suffix}": 1.0,  # RHS = 1 - x[i,j]           → 1 x-var
            f"global{suffix}": 0.0,  # constant RHS               → 0
            f"r1{suffix}": 0.0,  # constant RHS               → 0
            f"r2{suffix}": 0.0,  # constant RHS = 1           → 0
            f"r3{suffix}": 2.0,  # RHS = 1 - x[a] - x[b]     → 2 x-vars
        }

        result: dict[str, float | dict[tuple[Any, ...], float]] = {}
        for key, val in constrs.items():
            w = _group_weight.get(key, 1.0)  # unknown groups default to 1.0
            if isinstance(val, dict):
                result[key] = {arc_key: w for arc_key in val}
            else:
                # scalar gp.Constr
                result[key] = w

        return result

    # ------------------------------------------------------------------
    # B-1: Dual feasibility constraints  A^T π ≤ 0
    # ------------------------------------------------------------------

    def _build_dual_feasibility_constrs(
        self,
        cgsp: gp.Model,
        pi_vars: dict[str, Any],
        which: str,  # "y" or "yp"
    ) -> None:
        """Add A^T π ≤ 0 constraints to the CGSP model.

        One constraint per primal variable (triangle t ∈ V_list).
        Each constraint sums the dual contributions from every subproblem
        constraint that involves y[t] (or yp[t]), consistent with sign.

        Constraint type → dual sign
        ---------------------------
        alpha  (=)   → π_α free   → u_α - v_α
        beta   (=)   → π_β free   → u_β - v_β  (forward +1, backward -1)
        gamma  (≥)   → π_γ ≥ 0   → u_γ  only
        delta  (≤)   → π_δ ≤ 0   → -v_δ only
        global (=)   → π_g free   → u_g - v_g
        r1     (≤)   → π_r1 ≤ 0  → -v_r1 * area[t]
        r2     (≤)   → π_r2 ≤ 0  → -v_r2[arc]
        r3     (≤)   → π_r3 ≤ 0  → -v_r3[key]
        """
        suffix = "_p" if which == "yp" else ""
        adj: list[list[list[int]]] = getattr(self, "triangles_adj_list", [])
        constrs = self.constrs_yp if which == "yp" else self.constrs_y

        def _adj(i: int, j: int) -> list[int]:
            """Safe accessor for triangles_adj_list[i][j]."""
            try:
                return adj[i][j]
            except (IndexError, TypeError):
                return []

        for t in getattr(self, "V_list", []):
            expr = gp.LinExpr()

            # alpha[_p]: equality, adjacent to CH arcs
            for i, j in constrs.get(f"alpha{suffix}", {}):
                t_list = _adj(i, j)
                if t in t_list:
                    u = pi_vars.get(f"u_alpha{suffix}", {}).get((i, j))
                    v = pi_vars.get(f"v_alpha{suffix}", {}).get((i, j))
                    if u is not None:
                        expr += u
                    if v is not None:
                        expr -= v

            # beta[_p]: equality, non-CH edges
            for i, j in constrs.get(f"beta{suffix}", {}):
                u_key = f"u_beta{suffix}"
                v_key = f"v_beta{suffix}"
                if t in _adj(i, j):  # forward +1
                    u = pi_vars.get(u_key, {}).get((i, j))
                    v = pi_vars.get(v_key, {}).get((i, j))
                    if u is not None:
                        expr += u
                    if v is not None:
                        expr -= v
                if t in _adj(j, i):  # backward -1
                    u = pi_vars.get(u_key, {}).get((i, j))
                    v = pi_vars.get(v_key, {}).get((i, j))
                    if u is not None:
                        expr -= u
                    if v is not None:
                        expr += v

            # gamma[_p]: ≥ constraint, π_γ ≥ 0 → only u_gamma
            for i, j in constrs.get(f"gamma{suffix}", {}):
                if t in _adj(i, j):
                    u = pi_vars.get(f"u_gamma{suffix}", {}).get((i, j))
                    if u is not None:
                        expr += u

            # delta[_p]: ≤ constraint, π_δ ≤ 0 → only -v_delta
            for i, j in constrs.get(f"delta{suffix}", {}):
                if t in _adj(i, j):
                    v = pi_vars.get(f"v_delta{suffix}", {}).get((i, j))
                    if v is not None:
                        expr -= v

            # global[_p]: equality → free
            u_g = pi_vars.get(f"u_global{suffix}")
            v_g = pi_vars.get(f"v_global{suffix}")
            if isinstance(u_g, gp.Var):
                expr += u_g
            if isinstance(v_g, gp.Var):
                expr -= v_g

            # r1[_p]: ≤ → -v_r1 * area[t]
            v_r1 = pi_vars.get(f"v_r1{suffix}")
            if isinstance(v_r1, gp.Var) and hasattr(self, "_abs_areas"):
                expr -= self._abs_areas[t] * v_r1

            # r2[_p]: ≤ → -v_r2[arc]
            for i, j in constrs.get(f"r2{suffix}", {}):
                if t in _adj(i, j):
                    v = pi_vars.get(f"v_r2{suffix}", {}).get((i, j))
                    if v is not None:
                        expr -= v

            # r3[_p]: ≤ → -v_r3[key]
            for key in constrs.get(f"r3{suffix}", {}):
                i, j, k, s = key
                coef = (1 if t in _adj(i, j) else 0) + (1 if t in _adj(k, s) else 0)
                if coef:
                    v = pi_vars.get(f"v_r3{suffix}", {}).get(key)
                    if v is not None:
                        expr -= coef * v

            if expr.size() > 0:
                cgsp.addConstr(expr <= 0.0, name=f"dual_feas{suffix}_t{t}")

    # ------------------------------------------------------------------
    # Task 5: _build_cgsp_yp — dual LP for Y' (external triangulation)
    # ------------------------------------------------------------------

    def _build_cgsp_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, gp.Var | gp.tupledict[Any, gp.Var]]]:
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
        pi_vars : dict[str, gp.Var | gp.tupledict[Any, gp.Var]]
            Mapping from constraint group name to the corresponding π variables
            (already as u - v net variables, stored as gp.Var / gp.tupledict[Any, gp.Var]
            of the positive part u; the actual value is u.X - v.X after solve).
            The dict also contains 'u_<name>' and 'v_<name>' keys for the
            positive/negative parts if needed.
        """
        cgsp = gp.Model("cgsp_yp")
        cgsp.setParam("OutputFlag", 0)
        cgsp.setParam("InfUnbdInfo", 1)

        weights = self._resolve_weights("yp", self.constrs_yp, include_pi0=False, x_sol=x_sol)

        # Collect all constraint groups that have non-empty dicts or single constrs
        # We build one u/v pair per constraint.

        # We accumulate: objective terms and normalisation terms
        obj_expr = gp.LinExpr()
        norm_expr = gp.LinExpr()
        pi_vars: dict[str, gp.Var | gp.tupledict[Any, gp.Var]] = {}

        def _add_pi_scalar(group: str, rhs_val: float, weight: float) -> tuple[gp.Var, gp.Var]:
            """Add a scalar dual variable π = u - v to cgsp."""
            u = cgsp.addVar(lb=0.0, name=f"u_{group}")
            v = cgsp.addVar(lb=0.0, name=f"v_{group}")
            obj_expr.addTerms([rhs_val, -rhs_val], [u, v])
            norm_expr.addTerms([weight, weight], [u, v])
            return u, v

        def _add_pi_indexed(
            group: str,
            keys: list[Any],
            rhs_map: dict[str, Any],
            weight_map: dict[str, Any],
        ) -> tuple[gp.tupledict[Any, gp.Var], gp.tupledict[Any, gp.Var]]:
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
            u_ap, v_ap = _add_pi_indexed("alpha_p", keys_ap, rhs_map_ap, w_map_ap)  # type: ignore[arg-type]
            pi_vars["u_alpha_p"] = u_ap
            pi_vars["v_alpha_p"] = v_ap

        # ---- beta_p: flow balance = x[j,i] - x[i,j]  for (i,j) ∈ E'' ----
        if "beta_p" in self.constrs_yp and self.constrs_yp["beta_p"]:
            keys_bp = list(self.constrs_yp["beta_p"].keys())
            rhs_map_bp = {(i, j): x_sol.get((j, i), 0.0) - x_sol.get((i, j), 0.0) for (i, j) in keys_bp}
            w_map_bp = weights.get("beta_p", {})
            u_bp, v_bp = _add_pi_indexed("beta_p", keys_bp, rhs_map_bp, w_map_bp)  # type: ignore[arg-type]
            pi_vars["u_beta_p"] = u_bp
            pi_vars["v_beta_p"] = v_bp

        # ---- gamma_p: Σ_t yp[t] ≥ x[j,i]  for (i,j) ∈ A'' ----
        # π_γ ≥ 0 (≥ constraint) → only u_gamma_p, no v_gamma_p
        if "gamma_p" in self.constrs_yp and self.constrs_yp["gamma_p"]:
            keys_gp = list(self.constrs_yp["gamma_p"].keys())
            rhs_map_gp = {(i, j): x_sol.get((j, i), 0.0) for (i, j) in keys_gp}
            w_map_gp = weights.get("gamma_p", {})
            u_gp = cgsp.addVars(keys_gp, lb=0.0, name="u_gamma_p")
            for k in keys_gp:
                rhs_k = rhs_map_gp.get(k, 0.0)
                w_k = w_map_gp.get(k, 1.0) if isinstance(w_map_gp, dict) else 1.0
                obj_expr.addTerms([rhs_k], [u_gp[k]])
                norm_expr.addTerms([w_k], [u_gp[k]])
            pi_vars["u_gamma_p"] = u_gp
            # No v_gamma_p: π_γ ≥ 0

        # ---- delta_p: Σ_t yp[t] ≤ 1 - x[i,j]  for (i,j) ∈ A'' ----
        # π_δ ≤ 0 (≤ constraint) → only v_delta_p (π_δ = -v_δ), no u_delta_p
        if "delta_p" in self.constrs_yp and self.constrs_yp["delta_p"]:
            keys_dp = list(self.constrs_yp["delta_p"].keys())
            rhs_map_dp = {(i, j): 1.0 - x_sol.get((i, j), 0.0) for (i, j) in keys_dp}
            w_map_dp = weights.get("delta_p", {})
            v_dp = cgsp.addVars(keys_dp, lb=0.0, name="v_delta_p")
            for k in keys_dp:
                rhs_k = rhs_map_dp.get(k, 0.0)
                w_k = w_map_dp.get(k, 1.0) if isinstance(w_map_dp, dict) else 1.0
                obj_expr.addTerms([-rhs_k], [v_dp[k]])  # π_δ = -v_δ → contribution -rhs_k * v_δ
                norm_expr.addTerms([w_k], [v_dp[k]])
            pi_vars["v_delta_p"] = v_dp
            # No u_delta_p: π_δ ≤ 0

        # ---- global_p: Σ_t yp[t] = N - |CH| ----
        if "global_p" in self.constrs_yp and isinstance(self.constrs_yp["global_p"], gp.Constr):
            rhs_global = self.constrs_yp["global_p"].RHS
            w_global = float(weights.get("global_p", 1.0))  # type: ignore[arg-type]
            u_g, v_g = _add_pi_scalar("global_p", rhs_global, w_global)
            pi_vars["u_global_p"] = u_g
            pi_vars["v_global_p"] = v_g

        # ---- r1_p: area balance (fixed RHS = CH_area) ----
        if "r1_p" in self.constrs_yp and isinstance(self.constrs_yp["r1_p"], gp.Constr):
            w_r1 = float(weights.get("r1_p", 1.0))  # type: ignore[arg-type]
            u_r1, v_r1 = _add_pi_scalar("r1_p", self.convex_hull_area, w_r1)
            pi_vars["u_r1_p"] = u_r1
            pi_vars["v_r1_p"] = v_r1

        # ---- r2_p: arc coverage (fixed RHS = 1) ----
        if "r2_p" in self.constrs_yp and self.constrs_yp["r2_p"]:
            keys_r2 = list(self.constrs_yp["r2_p"].keys())
            rhs_map_r2 = {k: 1.0 for k in keys_r2}
            w_map_r2 = weights.get("r2_p", {})
            u_r2, v_r2 = _add_pi_indexed("r2_p", keys_r2, rhs_map_r2, w_map_r2)  # type: ignore[arg-type]
            pi_vars["u_r2_p"] = u_r2
            pi_vars["v_r2_p"] = v_r2

        # ---- r3_p: crossing exclusion — RHS = 1 - x[i,j] - x[s,k] ----
        if "r3_p" in self.constrs_yp and self.constrs_yp["r3_p"]:
            keys_r3 = list(self.constrs_yp["r3_p"].keys())
            rhs_map_r3 = {
                (i, j, k, s): 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0) for (i, j, k, s) in keys_r3
            }
            w_map_r3 = weights.get("r3_p", {})
            u_r3, v_r3 = _add_pi_indexed("r3_p", keys_r3, rhs_map_r3, w_map_r3)  # type: ignore[arg-type]
            pi_vars["u_r3_p"] = u_r3
            pi_vars["v_r3_p"] = v_r3

        # ---- Dual feasibility constraints: A^T π ≤ 0 (B-1) ----
        self._build_dual_feasibility_constrs(cgsp, pi_vars, which="yp")

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

    def _build_cgsp_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, gp.Var | gp.tupledict[Any, gp.Var]], gp.Var | None]:
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
        pi_vars    : dict[str, gp.Var | gp.tupledict[Any, gp.Var]]
        pi0_var    : gp.Var | None — the π₀ dual variable (None for Fekete)
        """
        is_internal = getattr(self, "objective", "Fekete") == "Internal"
        include_pi0 = is_internal

        cgsp = gp.Model("cgsp_y")
        cgsp.setParam("OutputFlag", 0)
        cgsp.setParam("InfUnbdInfo", 1)

        weights = self._resolve_weights("y", self.constrs_y, include_pi0=include_pi0, x_sol=x_sol)

        obj_expr = gp.LinExpr()
        norm_expr = gp.LinExpr()
        pi_vars: dict[str, gp.Var | gp.tupledict[Any, gp.Var]] = {}

        def _add_pi_scalar(group: str, rhs_val: float, weight: float) -> tuple[gp.Var, gp.Var]:
            u = cgsp.addVar(lb=0.0, name=f"u_{group}")
            v = cgsp.addVar(lb=0.0, name=f"v_{group}")
            obj_expr.addTerms([rhs_val, -rhs_val], [u, v])
            norm_expr.addTerms([weight, weight], [u, v])
            return u, v

        def _add_pi_indexed(
            group: str,
            keys: list[Any],
            rhs_map: dict[str, Any],
            weight_map: dict[str, Any],
        ) -> tuple[gp.tupledict[Any, gp.Var], gp.tupledict[Any, gp.Var]]:
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
            u_a, v_a = _add_pi_indexed("alpha", keys_a, rhs_map_a, w_map_a)  # type: ignore[arg-type]
            pi_vars["u_alpha"] = u_a
            pi_vars["v_alpha"] = v_a

        # ---- beta: flow balance = x[i,j] - x[j,i]  for (i,j) ∈ E'' ----
        if "beta" in self.constrs_y and self.constrs_y["beta"]:
            keys_b = list(self.constrs_y["beta"].keys())
            rhs_map_b = {(i, j): x_sol.get((i, j), 0.0) - x_sol.get((j, i), 0.0) for (i, j) in keys_b}
            w_map_b = weights.get("beta", {})
            u_b, v_b = _add_pi_indexed("beta", keys_b, rhs_map_b, w_map_b)  # type: ignore[arg-type]
            pi_vars["u_beta"] = u_b
            pi_vars["v_beta"] = v_b

        # ---- gamma: Σ_t y[t] ≥ x[i,j]  for (i,j) ∈ A'' ----
        # π_γ ≥ 0 (≥ constraint) → only u_gamma, no v_gamma
        if "gamma" in self.constrs_y and self.constrs_y["gamma"]:
            keys_g = list(self.constrs_y["gamma"].keys())
            rhs_map_g = {(i, j): x_sol.get((i, j), 0.0) for (i, j) in keys_g}
            w_map_g = weights.get("gamma", {})
            u_g2 = cgsp.addVars(keys_g, lb=0.0, name="u_gamma")
            for k in keys_g:
                rhs_k = rhs_map_g.get(k, 0.0)
                w_k = w_map_g.get(k, 1.0) if isinstance(w_map_g, dict) else 1.0
                obj_expr.addTerms([rhs_k], [u_g2[k]])
                norm_expr.addTerms([w_k], [u_g2[k]])
            pi_vars["u_gamma"] = u_g2
            # No v_gamma: π_γ ≥ 0

        # ---- delta: Σ_t y[t] ≤ 1 - x[j,i]  for (i,j) ∈ A'' ----
        # π_δ ≤ 0 (≤ constraint) → only v_delta (π_δ = -v_δ), no u_delta
        if "delta" in self.constrs_y and self.constrs_y["delta"]:
            keys_d = list(self.constrs_y["delta"].keys())
            rhs_map_d = {(i, j): 1.0 - x_sol.get((j, i), 0.0) for (i, j) in keys_d}
            w_map_d = weights.get("delta", {})
            v_d2 = cgsp.addVars(keys_d, lb=0.0, name="v_delta")
            for k in keys_d:
                rhs_k = rhs_map_d.get(k, 0.0)
                w_k = w_map_d.get(k, 1.0) if isinstance(w_map_d, dict) else 1.0
                obj_expr.addTerms([-rhs_k], [v_d2[k]])  # π_δ = -v_δ → contribution -rhs_k * v_δ
                norm_expr.addTerms([w_k], [v_d2[k]])
            pi_vars["v_delta"] = v_d2
            # No u_delta: π_δ ≤ 0

        # ---- global: Σ_t y[t] = N - 2 ----
        if "global" in self.constrs_y and isinstance(self.constrs_y["global"], gp.Constr):
            rhs_global = self.constrs_y["global"].RHS
            w_global = float(weights.get("global", 1.0))  # type: ignore[arg-type]
            u_gl, v_gl = _add_pi_scalar("global", rhs_global, w_global)
            pi_vars["u_global"] = u_gl
            pi_vars["v_global"] = v_gl

        # ---- r1: area balance (fixed RHS = CH_area) ----
        if "r1" in self.constrs_y and isinstance(self.constrs_y["r1"], gp.Constr):
            w_r1 = float(weights.get("r1", 1.0))  # type: ignore[arg-type]
            u_r1, v_r1 = _add_pi_scalar("r1", self.convex_hull_area, w_r1)
            pi_vars["u_r1"] = u_r1
            pi_vars["v_r1"] = v_r1

        # ---- r2: arc coverage (fixed RHS = 1) ----
        if "r2" in self.constrs_y and self.constrs_y["r2"]:
            keys_r2 = list(self.constrs_y["r2"].keys())
            rhs_map_r2 = {k: 1.0 for k in keys_r2}
            w_map_r2 = weights.get("r2", {})
            u_r2, v_r2 = _add_pi_indexed("r2", keys_r2, rhs_map_r2, w_map_r2)  # type: ignore[arg-type]
            pi_vars["u_r2"] = u_r2
            pi_vars["v_r2"] = v_r2

        # ---- r3: crossing exclusion — RHS = 1 - x[j,i] - x[k,s] ----
        if "r3" in self.constrs_y and self.constrs_y["r3"]:
            keys_r3 = list(self.constrs_y["r3"].keys())
            rhs_map_r3 = {
                (i, j, k, s): 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0) for (i, j, k, s) in keys_r3
            }
            w_map_r3 = weights.get("r3", {})
            u_r3, v_r3 = _add_pi_indexed("r3", keys_r3, rhs_map_r3, w_map_r3)  # type: ignore[arg-type]
            pi_vars["u_r3"] = u_r3
            pi_vars["v_r3"] = v_r3

        # ---- π₀ for Internal objective ----
        pi0_var: gp.Var | None = None
        if include_pi0:
            # π₀ is free: split as π₀ = π₀_pos - π₀_neg
            pi0_pos = cgsp.addVar(lb=0.0, name="pi0_pos")
            pi0_neg = cgsp.addVar(lb=0.0, name="pi0_neg")
            # Objective term: π₀ * (f^T x̄ − η̄)
            # f^T x̄ is computed directly from the cached master cost vector
            # self._cost_x (NEVER from self.sub_y.ObjVal, which is the
            # artificial-slack minimum, not the area).
            cost_x: dict[str, Any] = getattr(self, "_cost_x", {})
            f_dot_x = sum(cost_x.get(arc, 0.0) * x_sol.get(arc, 0.0) for arc in cost_x)  # type: ignore[call-overload]
            pi0_contribution = f_dot_x - eta_sol
            w_pi0 = float(weights.get("pi0", 1.0))  # type: ignore[arg-type]
            obj_expr.addTerms([pi0_contribution, -pi0_contribution], [pi0_pos, pi0_neg])
            norm_expr.addTerms([w_pi0, w_pi0], [pi0_pos, pi0_neg])
            pi_vars["pi0_pos"] = pi0_pos
            pi_vars["pi0_neg"] = pi0_neg
            # Store net variable reference via a helper attribute name
            pi0_var = pi0_pos  # caller uses pi0_pos.X - pi0_neg.X for the net value

        # ---- Dual feasibility constraints: A^T π ≤ 0 (B-1) ----
        self._build_dual_feasibility_constrs(cgsp, pi_vars, which="y")

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

    def _reconstruct_cut_from_pi(
        self,
        pi_vars: dict[str, gp.Var | gp.tupledict[Any, gp.Var]],
        which: str,  # "y" or "yp"
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]]:
        """Reconstruct a Benders cut from an already-solved pi_vars dict.

        This is the shared cut-reconstruction logic, separated so that both
        ``get_cgsp_cut_yp/y`` (which solves CGSP internally) and MW
        (which solves the secondary LP and needs to reconstruct from *that*
        LP's pi_vars) can reuse identical cut building code.

        Parameters
        ----------
        pi_vars : dict
            The u_*/v_* variable dict from a solved CGSP or MW secondary LP.
        which : 'y' | 'yp'
            Selects y or y' constraint RHS conventions.
        TOL : float
            Numerical zero threshold.

        Returns
        -------
        (cut_expr, cut_rhs, witness)
            The Benders cut  cut_expr <= cut_rhs  in master variable space.
        """
        suffix = "_p" if which == "yp" else ""

        def _net(u_key: str, v_key: str) -> dict[str, Any]:
            out: dict[str, Any] = {}
            u_vars = pi_vars.get(u_key)
            v_vars = pi_vars.get(v_key)
            if u_vars is None and v_vars is None:
                return out
            if isinstance(u_vars, gp.Var) or isinstance(v_vars, gp.Var):
                u_val = u_vars.X if isinstance(u_vars, gp.Var) else 0.0
                v_val = v_vars.X if isinstance(v_vars, gp.Var) else 0.0
                val = u_val - v_val
                if abs(val) > TOL:
                    out["scalar"] = val
            else:
                keys = u_vars.keys() if u_vars is not None else v_vars.keys()  # type: ignore[union-attr]
                for k in keys:
                    u_val = u_vars[k].X if u_vars is not None else 0.0
                    v_val = v_vars[k].X if v_vars is not None else 0.0
                    val = u_val - v_val
                    if abs(val) > TOL:
                        out[k] = val
            return out

        cut_expr = gp.LinExpr()
        cut_rhs = 0.0
        witness: dict[str, Any] = {}

        if which == "yp":
            constrs = self.constrs_yp

            ap_net = _net("u_alpha_p", "v_alpha_p")
            if ap_net:
                witness["alpha_p"] = ap_net
                for (i, j), pi_val in ap_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (1.0 - self.x[i, j])  # type: ignore[index]

            bp_net = _net("u_beta_p", "v_beta_p")
            if bp_net:
                witness["beta_p"] = bp_net
                for (i, j), pi_val in bp_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (self.x[j, i] - self.x[i, j])  # type: ignore[index]

            gp_net = _net("u_gamma_p", "v_gamma_p")
            if gp_net:
                witness["gamma_p"] = gp_net
                for (i, j), pi_val in gp_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * self.x[j, i]  # type: ignore[index]

            dp_net = _net("u_delta_p", "v_delta_p")
            if dp_net:
                witness["delta_p"] = dp_net
                for (i, j), pi_val in dp_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (1.0 - self.x[i, j])  # type: ignore[index]

            gl_net = _net("u_global_p", "v_global_p")
            if gl_net and "scalar" in gl_net:
                pi_val = gl_net["scalar"]
                rhs_val = constrs["global_p"].RHS
                cut_rhs -= pi_val * rhs_val
                witness["global_p"] = pi_val

            r1_net = _net("u_r1_p", "v_r1_p")
            if r1_net and "scalar" in r1_net:
                pi_val = r1_net["scalar"]
                cut_rhs -= pi_val * self.convex_hull_area
                witness["r1_p"] = pi_val

            r2_net = _net("u_r2_p", "v_r2_p")
            if r2_net:
                witness["r2_p"] = r2_net
                for _k, pi_val in r2_net.items():
                    cut_rhs -= pi_val * 1.0

            r3_net = _net("u_r3_p", "v_r3_p")
            if r3_net:
                witness["r3_p"] = r3_net
                for (i, j, k, s), pi_val in r3_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (1.0 - self.x[i, j] - self.x[s, k])  # type: ignore[index]

        else:  # "y"
            constrs = self.constrs_y

            a_net = _net("u_alpha", "v_alpha")
            if a_net:
                witness["alpha"] = a_net
                for (i, j), pi_val in a_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * self.x[i, j]  # type: ignore[index]

            b_net = _net("u_beta", "v_beta")
            if b_net:
                witness["beta"] = b_net
                for (i, j), pi_val in b_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (self.x[i, j] - self.x[j, i])  # type: ignore[index]

            g_net = _net("u_gamma", "v_gamma")
            if g_net:
                witness["gamma"] = g_net
                for (i, j), pi_val in g_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * self.x[i, j]  # type: ignore[index]

            d_net = _net("u_delta", "v_delta")
            if d_net:
                witness["delta"] = d_net
                for (i, j), pi_val in d_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (1.0 - self.x[j, i])  # type: ignore[index]

            gl_net = _net("u_global", "v_global")
            if gl_net and "scalar" in gl_net:
                pi_val = gl_net["scalar"]
                rhs_val = constrs["global"].RHS
                cut_rhs -= pi_val * rhs_val
                witness["global"] = pi_val

            r1_net = _net("u_r1", "v_r1")
            if r1_net and "scalar" in r1_net:
                pi_val = r1_net["scalar"]
                cut_rhs -= pi_val * self.convex_hull_area
                witness["r1"] = pi_val

            r2_net = _net("u_r2", "v_r2")
            if r2_net:
                witness["r2"] = r2_net
                for _k, pi_val in r2_net.items():
                    cut_rhs -= pi_val * 1.0

            r3_net = _net("u_r3", "v_r3")
            if r3_net:
                witness["r3"] = r3_net
                for (i, j, k, s), pi_val in r3_net.items():  # type: ignore[str-unpack]
                    cut_expr += pi_val * (1.0 - self.x[j, i] - self.x[k, s])  # type: ignore[index]

        return cut_expr, cut_rhs, witness

    # ------------------------------------------------------------------
    # Task 8.5: CGSP cache wrappers (F2.4)
    # ------------------------------------------------------------------

    def _cgsp_yp_obj_expr(
        self,
        pi_vars: dict[str, Any],
        x_sol: dict[Arc, float],
    ) -> gp.LinExpr:
        """Rebuild the CGSP Y' objective LinExpr from cached pi_vars + new x_sol.

        Only x-dependent RHS terms change between callbacks.  Fixed-RHS terms
        (global_p, r1_p, r2_p) use the constraint RHS stored in self.constrs_yp.
        """
        obj = gp.LinExpr()

        # alpha_p: RHS = 1 - x[i,j]
        u_ap = pi_vars.get("u_alpha_p")
        v_ap = pi_vars.get("v_alpha_p")
        if u_ap is not None:
            for k in u_ap.keys():
                rhs = 1.0 - x_sol.get(k, 0.0)
                obj.addTerms([rhs, -rhs], [u_ap[k], v_ap[k]])  # type: ignore[index]

        # beta_p: RHS = x[j,i] - x[i,j]
        u_bp = pi_vars.get("u_beta_p")
        v_bp = pi_vars.get("v_beta_p")
        if u_bp is not None:
            for k in u_bp.keys():
                i, j = k
                rhs = x_sol.get((j, i), 0.0) - x_sol.get((i, j), 0.0)
                obj.addTerms([rhs, -rhs], [u_bp[k], v_bp[k]])  # type: ignore[index]

        # gamma_p: RHS = x[j,i]  (only u, no v)
        u_gp = pi_vars.get("u_gamma_p")
        if u_gp is not None:
            for k in u_gp.keys():
                i, j = k
                rhs = x_sol.get((j, i), 0.0)
                obj.addTerms([rhs], [u_gp[k]])

        # delta_p: RHS = 1 - x[i,j]  (only v, π_δ = -v_δ → -rhs * v)
        v_dp = pi_vars.get("v_delta_p")
        if v_dp is not None:
            for k in v_dp.keys():
                rhs = 1.0 - x_sol.get(k, 0.0)
                obj.addTerms([-rhs], [v_dp[k]])

        # global_p: fixed RHS
        u_g = pi_vars.get("u_global_p")
        v_g = pi_vars.get("v_global_p")
        if u_g is not None and v_g is not None:
            rhs = self.constrs_yp["global_p"].RHS
            obj.addTerms([rhs, -rhs], [u_g, v_g])

        # r1_p: fixed RHS = convex_hull_area
        u_r1 = pi_vars.get("u_r1_p")
        v_r1 = pi_vars.get("v_r1_p")
        if u_r1 is not None and v_r1 is not None:
            rhs = self.convex_hull_area
            obj.addTerms([rhs, -rhs], [u_r1, v_r1])

        # r2_p: fixed RHS = 1
        u_r2 = pi_vars.get("u_r2_p")
        v_r2 = pi_vars.get("v_r2_p")
        if u_r2 is not None:
            for k in u_r2.keys():
                obj.addTerms([1.0, -1.0], [u_r2[k], v_r2[k]])  # type: ignore[index]

        # r3_p: RHS = 1 - x[i,j] - x[s,k]
        u_r3 = pi_vars.get("u_r3_p")
        v_r3 = pi_vars.get("v_r3_p")
        if u_r3 is not None:
            for k in u_r3.keys():
                i, j, ks, s = k
                rhs = 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, ks), 0.0)
                obj.addTerms([rhs, -rhs], [u_r3[k], v_r3[k]])  # type: ignore[index]

        return obj

    def invalidate_cgsp_cache(self) -> None:
        """Reset the CGSP LP caches so the next callback rebuilds them from scratch.

        Call this whenever the weights or constraint structure change at build
        time (e.g. after ``_compute_relaxed_l1_weights`` stores new weights into
        ``cut_weights_y / cut_weights_yp``).  Setting both cache attributes to
        ``None`` causes ``_get_or_build_cgsp_yp`` and ``_get_or_build_cgsp_y``
        to treat the next invocation as a first call and rebuild the full LP.
        """
        self._cgsp_yp_cache = None
        self._cgsp_y_cache = None

    def _get_or_build_cgsp_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, Any]]:
        """Return a cached (and updated) CGSP Y' model, building it on first call.

        On first call: delegates to _build_cgsp_yp and stores result.
        On subsequent calls: updates the objective in-place via setObjective,
        avoiding the overhead of rebuilding the full LP structure.
        """
        cache = getattr(self, "_cgsp_yp_cache", None)
        if cache is None:
            cgsp, pi_vars = self._build_cgsp_yp(x_sol, TOL=TOL)
            self._cgsp_yp_cache: tuple[gp.Model, dict[str, Any]] = (cgsp, pi_vars)  # type: ignore[assignment, no-redef]
            return cgsp, pi_vars
        cgsp, pi_vars = cache
        # Update objective for new x_sol
        new_obj = self._cgsp_yp_obj_expr(pi_vars, x_sol)
        cgsp.setObjective(new_obj, GRB.MAXIMIZE)
        cgsp.update()
        return cgsp, pi_vars

    def _get_or_build_cgsp_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.Model, dict[str, Any], gp.Var | None]:
        """Return a cached (and updated) CGSP Y model, building it on first call.

        On first call: delegates to _build_cgsp_y and stores result.
        On subsequent calls: rebuilds via _build_cgsp_y because the Y subproblem
        objective also depends on eta_sol which changes across callbacks.
        A full incremental obj-update helper for Y is deferred to a later task.
        """
        # NOTE: Y caching is simpler to implement as a fresh build for now,
        # since eta_sol also enters the objective (optimality cut mode).
        # The structural constraints don't change so we still avoid constraint
        # rebuild overhead by re-using the same model object with setObjective.
        cache = getattr(self, "_cgsp_y_cache", None)
        if cache is None:
            cgsp, pi_vars, pi0_var = self._build_cgsp_y(x_sol, TOL=TOL)
            self._cgsp_y_cache: tuple[gp.Model, dict[str, Any], gp.Var | None] = (cgsp, pi_vars, pi0_var)  # type: ignore[assignment, no-redef]
            return cgsp, pi_vars, pi0_var
        cgsp, pi_vars, pi0_var = cache
        # For Y, delegate full rebuild to avoid eta_sol complexity for now.
        # Re-use model object by rebuilding from scratch (structural constraints unchanged).
        new_cgsp, new_pi_vars, new_pi0_var = self._build_cgsp_y(x_sol, TOL=TOL)
        self._cgsp_y_cache = (new_cgsp, new_pi_vars, new_pi0_var)  # type: ignore[assignment]
        return new_cgsp, new_pi_vars, new_pi0_var

    def get_cgsp_cut_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
        """Generate the deepest feasibility cut for Y' via CGSP.

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current master solution.
        TOL : float
            Numerical zero threshold.

        Returns
        -------
        (cut_expr, cut_rhs, witness) when a violated cut is found:
            cut_expr : gp.LinExpr
                Left-hand side of the cut over master variables x.
                The cut is:  cut_expr <= cut_rhs
            cut_rhs : float
                Right-hand side (0.0 for feasibility cuts).
            witness : dict
                Non-zero dual components (for logging and analysis).
        (None, None, {'aborted': reason}) when no cut is emitted.
            Callers must check ``cut_expr is not None`` before adding the cut.
        """
        cgsp, pi_vars = self._get_or_build_cgsp_yp(x_sol, TOL=TOL)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y') did not solve to optimality (status=%d).", cgsp.Status)
            return None, None, {"aborted": "solve_failed", "status": cgsp.Status}

        if cgsp.ObjVal <= TOL:
            return None, None, {"aborted": "no_violation"}

        cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(pi_vars, which="yp", TOL=TOL)
        return cut_expr, cut_rhs, witness

    def get_cgsp_cut_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
        """Generate the deepest feasibility (or optimality) cut for Y via CGSP.

        For Fekete objective: returns a feasibility cut  cut_expr <= cut_rhs.
        For Internal objective: if the subproblem optimal value exceeds eta_sol,
            returns an optimality cut (scaled by 1/π₀, with η moved to LHS);
            otherwise a feasibility cut.

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
        (cut_expr, cut_rhs, witness) when a cut is found:
            For feasibility cuts: cut_expr <= cut_rhs (over x variables).
            For optimality cuts:  cut_expr <= cut_rhs where cut_expr already
                includes the -η term, i.e.  (π^T B x)/π₀ - η <= (π^T b)/π₀.
                The caller adds as ``model.cbLazy(cut_expr <= cut_rhs)``.
        (None, None, {'aborted': reason}) when no cut is emitted:
            - 'no_violation': CGSP objective <= TOL
            - 'solve_failed': CGSP did not solve to optimality
            - 'pi0_negative': π₀ < -TOL (numerical sign anomaly)
            Callers must check ``cut_expr is not None`` before adding the cut.
        """
        cgsp, pi_vars, pi0_var = self._get_or_build_cgsp_y(x_sol, eta_sol=eta_sol, TOL=TOL)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y) did not solve to optimality (status=%d).", cgsp.Status)
            return None, None, {"aborted": "solve_failed", "status": cgsp.Status}

        if cgsp.ObjVal <= TOL:
            return None, None, {"aborted": "no_violation"}

        is_internal = getattr(self, "objective", "Fekete") == "Internal"

        # Determine if this is an optimality cut
        pi0_net = 0.0
        is_optimality_cut = False
        if is_internal and pi0_var is not None:
            pi0_neg_var = pi_vars.get("pi0_neg")
            pi0_net = pi0_var.X - (pi0_neg_var.X if pi0_neg_var is not None else 0.0)
            if abs(pi0_net) > TOL:
                is_optimality_cut = True

        cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(pi_vars, which="y", TOL=TOL)

        if is_optimality_cut:
            witness["pi0"] = pi0_net

        # For Internal optimality cuts: scale by 1/π₀
        if is_optimality_cut and is_internal:
            if pi0_net < -TOL:
                logger.warning(
                    "CGSP-Y: pi0_net=%.3e < -TOL; skipping cut (numerical sign anomaly).",
                    pi0_net,
                )
                return None, None, {"aborted": "pi0_negative", "pi0_net": pi0_net}
            if pi0_net > TOL:
                scale = 1.0 / pi0_net
                scaled_expr = scale * cut_expr - self.eta  # type: ignore[attr-defined]
                scaled_rhs = cut_rhs * scale
                witness["is_optimality_cut"] = True
                return scaled_expr, scaled_rhs, witness

        return cut_expr, cut_rhs, witness


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _rhs_for_key(
    group: str,
    arc_key: tuple[Any, ...],
    x_sol: dict[tuple[Any, ...], float],
    model: object,
) -> float:
    """Return the RHS value b_i(x_bar) for a given constraint group and key.

    Used by _resolve_weights when cgsp_norm='relaxed_l1' to compute
    per-variable weights  w[i] = 1 / (|b_i(x_bar)| + ε).

    This mirrors the RHS computations in _build_cgsp_yp / _build_cgsp_y.
    Unknown groups default to 1.0 (uniform weight).
    """
    x = x_sol or {}

    # Y' groups
    if group == "alpha_p":
        i, j = arc_key
        return 1.0 - x.get((i, j), 0.0)
    if group == "beta_p":
        i, j = arc_key
        return x.get((j, i), 0.0) - x.get((i, j), 0.0)
    if group == "gamma_p":
        i, j = arc_key
        return x.get((j, i), 0.0)
    if group == "delta_p":
        i, j = arc_key
        return 1.0 - x.get((i, j), 0.0)
    if group == "r3_p":
        i, j, k, s = arc_key
        return 1.0 - x.get((i, j), 0.0) - x.get((s, k), 0.0)

    # Y groups
    if group == "alpha":
        i, j = arc_key
        return 1.0 - x.get((i, j), 0.0)
    if group == "beta":
        i, j = arc_key
        return x.get((j, i), 0.0) - x.get((i, j), 0.0)
    if group == "gamma":
        i, j = arc_key
        return x.get((j, i), 0.0)
    if group == "delta":
        i, j = arc_key
        return 1.0 - x.get((i, j), 0.0)
    if group == "r3":
        i, j, k, s = arc_key
        return 1.0 - x.get((i, j), 0.0) - x.get((s, k), 0.0)

    # Fixed-RHS groups: return 1.0 (weight = 1/(1+ε) ≈ 1.0 — effectively uniform)
    return 1.0
