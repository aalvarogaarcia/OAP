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
    # F2.5 — Relaxed-ℓ₁ weight computation (column sums of |B|)
    # ------------------------------------------------------------------

    def _compute_relaxed_l1_weights(self, which: str) -> dict:
        """Return Relaxed-ℓ₁ normalisation weights for CGSP cut generation.

        The weights correspond to the row sums of the |B| matrix, i.e. the
        number of x-variables that appear (with non-zero coefficient) in each
        RHS expression b(x) of the subproblem constraints.  Constraints whose
        RHS is constant (no x-dependence) receive weight 0.

        Parameters
        ----------
        which : str
            ``"y"``  → use ``self.constrs_y``   (Y-subproblem)
            ``"yp"`` → use ``self.constrs_yp``  (Y′-subproblem)

        Returns
        -------
        dict
            Nested weight dict with the same structure expected by
            ``_resolve_weights``.
        """
        if which == "yp":
            constrs = self.constrs_yp
            weights: dict = {
                "alpha_p":  {arc: 1.0 for arc in constrs.get("alpha_p",  {})},
                "beta_p":   {arc: 2.0 for arc in constrs.get("beta_p",   {})},
                "gamma_p":  {arc: 1.0 for arc in constrs.get("gamma_p",  {})},
                "delta_p":  {arc: 1.0 for arc in constrs.get("delta_p",  {})},
                "global_p": 0.0,
                "r1_p":     0.0,
                "r2_p":     {k: 0.0 for k in constrs.get("r2_p", {})},
                "r3_p":     {key: 2.0 for key in constrs.get("r3_p", {})},
            }
        else:
            constrs = self.constrs_y
            weights = {
                "alpha":  {arc: 1.0 for arc in constrs.get("alpha",  {})},
                "beta":   {arc: 2.0 for arc in constrs.get("beta",   {})},
                "gamma":  {arc: 1.0 for arc in constrs.get("gamma",  {})},
                "delta":  {arc: 1.0 for arc in constrs.get("delta",  {})},
                "global": 0.0,
                "r1":     0.0,
                "r2":     {k: 0.0 for k in constrs.get("r2", {})},
                "r3":     {key: 2.0 for key in constrs.get("r3", {})},
            }
        # Remove keys not present in constrs so _resolve_weights sees a clean dict
        present = set(constrs.keys())
        return {k: v for k, v in weights.items() if k in present or not isinstance(v, dict)}

    # ------------------------------------------------------------------
    # B-1: Dual feasibility constraints  A^T π ≤ 0
    # ------------------------------------------------------------------

    def _build_dual_feasibility_constrs(
        self,
        cgsp: gp.Model,
        pi_vars: dict,
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
            for (i, j) in constrs.get(f"alpha{suffix}", {}):
                t_list = _adj(i, j)
                if t in t_list:
                    u = pi_vars.get(f"u_alpha{suffix}", {}).get((i, j))
                    v = pi_vars.get(f"v_alpha{suffix}", {}).get((i, j))
                    if u is not None:
                        expr += u
                    if v is not None:
                        expr -= v

            # beta[_p]: equality, non-CH edges
            for (i, j) in constrs.get(f"beta{suffix}", {}):
                u_key = f"u_beta{suffix}"
                v_key = f"v_beta{suffix}"
                if t in _adj(i, j):   # forward +1
                    u = pi_vars.get(u_key, {}).get((i, j))
                    v = pi_vars.get(v_key, {}).get((i, j))
                    if u is not None:
                        expr += u
                    if v is not None:
                        expr -= v
                if t in _adj(j, i):   # backward -1
                    u = pi_vars.get(u_key, {}).get((i, j))
                    v = pi_vars.get(v_key, {}).get((i, j))
                    if u is not None:
                        expr -= u
                    if v is not None:
                        expr += v

            # gamma[_p]: ≥ constraint, π_γ ≥ 0 → only u_gamma
            for (i, j) in constrs.get(f"gamma{suffix}", {}):
                if t in _adj(i, j):
                    u = pi_vars.get(f"u_gamma{suffix}", {}).get((i, j))
                    if u is not None:
                        expr += u

            # delta[_p]: ≤ constraint, π_δ ≤ 0 → only -v_delta
            for (i, j) in constrs.get(f"delta{suffix}", {}):
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
                expr -= self._abs_areas[t] * v_r1  # type: ignore[index]

            # r2[_p]: ≤ → -v_r2[arc]
            for (i, j) in constrs.get(f"r2{suffix}", {}):
                if t in _adj(i, j):
                    v = pi_vars.get(f"v_r2{suffix}", {}).get((i, j))
                    if v is not None:
                        expr -= v

            # r3[_p]: ≤ → -v_r3[key]
            # F2.1 (audit §2.2): the primal r3 row sums y over BOTH triangle
            # fans (adj(i,j) and adj(k,s)).  When a triangle t belongs to both
            # fans the y_t coefficient in the row is 2, not 1.  The previous
            # `or` test contributed -v_r3 once even in that case, producing an
            # invalid dual feasibility constraint.  Count both indicators.
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
    # F2.4 — Objective recomputation (used by the CGSP-model cache)
    # ------------------------------------------------------------------

    def _compute_cgsp_objective(
        self,
        pi_vars: dict,
        x_sol: dict[Arc, float],
        which: str,  # "y" or "yp"
        eta_sol: float = 0.0,
    ) -> gp.LinExpr:
        """Rebuild the CGSP objective  Σ_i π_i b_i(x_sol)  from a cached model.

        Used both by the first-call build (so the same code path produces the
        objective in both cases) and by every subsequent callback once the
        CGSP skeleton is cached.  Skipping the variable/constraint rebuild
        is what brings CGSP from >100x slower than Farkas down toward parity
        — see audit §2.5 in
        ``.claude/context/reviews/2026-05-04-cgsp-paper-dissonance.md``.

        Notes
        -----
        Variables that exist only as ``u_*`` (e.g. gamma — the dual of a
        ≥ constraint) contribute ``+rhs * u``; ``v_*``-only (e.g. delta —
        the dual of a ≤ constraint) contribute ``-rhs * v``; both
        present ⇒ ``+rhs * u - rhs * v`` (free dual).
        """
        obj = gp.LinExpr()
        suffix = "_p" if which == "yp" else ""

        def _addto(u_key: str, v_key: str, rhs_map: dict) -> None:
            u_vars = pi_vars.get(u_key)
            v_vars = pi_vars.get(v_key)
            if u_vars is None and v_vars is None:
                return
            keys = (
                u_vars.keys() if u_vars is not None else v_vars.keys()  # type: ignore[union-attr]
            )
            for k in keys:
                rhs_k = rhs_map.get(k, 0.0)
                if u_vars is not None:
                    obj.addTerms([rhs_k], [u_vars[k]])  # type: ignore[index]
                if v_vars is not None:
                    obj.addTerms([-rhs_k], [v_vars[k]])  # type: ignore[index]

        # alpha[_p]:  yp → 1 - x[i,j];   y → x[i,j]
        a_keys = list(
            (pi_vars.get(f"u_alpha{suffix}") or pi_vars.get(f"v_alpha{suffix}") or {}).keys()
        )
        if which == "yp":
            a_rhs = {(i, j): 1.0 - x_sol.get((i, j), 0.0) for (i, j) in a_keys}
        else:
            a_rhs = {(i, j): x_sol.get((i, j), 0.0) for (i, j) in a_keys}
        _addto(f"u_alpha{suffix}", f"v_alpha{suffix}", a_rhs)

        # beta[_p]:  yp → x[j,i] - x[i,j];   y → x[i,j] - x[j,i]
        b_keys = list(
            (pi_vars.get(f"u_beta{suffix}") or pi_vars.get(f"v_beta{suffix}") or {}).keys()
        )
        if which == "yp":
            b_rhs = {
                (i, j): x_sol.get((j, i), 0.0) - x_sol.get((i, j), 0.0) for (i, j) in b_keys
            }
        else:
            b_rhs = {
                (i, j): x_sol.get((i, j), 0.0) - x_sol.get((j, i), 0.0) for (i, j) in b_keys
            }
        _addto(f"u_beta{suffix}", f"v_beta{suffix}", b_rhs)

        # gamma[_p]: u-only.  yp → x[j,i];   y → x[i,j]
        g_keys = list((pi_vars.get(f"u_gamma{suffix}") or {}).keys())
        if which == "yp":
            g_rhs = {(i, j): x_sol.get((j, i), 0.0) for (i, j) in g_keys}
        else:
            g_rhs = {(i, j): x_sol.get((i, j), 0.0) for (i, j) in g_keys}
        _addto(f"u_gamma{suffix}", f"v_gamma{suffix}", g_rhs)

        # delta[_p]: v-only.  yp → 1 - x[i,j];   y → 1 - x[j,i]
        d_keys = list((pi_vars.get(f"v_delta{suffix}") or {}).keys())
        if which == "yp":
            d_rhs = {(i, j): 1.0 - x_sol.get((i, j), 0.0) for (i, j) in d_keys}
        else:
            d_rhs = {(i, j): 1.0 - x_sol.get((j, i), 0.0) for (i, j) in d_keys}
        _addto(f"u_delta{suffix}", f"v_delta{suffix}", d_rhs)

        # global[_p]: scalar, RHS taken from the primal constraint
        constrs = self.constrs_yp if which == "yp" else self.constrs_y
        g_constr = constrs.get(f"global{suffix}")
        if g_constr is not None and hasattr(g_constr, "RHS"):
            rhs_val = float(g_constr.RHS)
            u_gl = pi_vars.get(f"u_global{suffix}")
            v_gl = pi_vars.get(f"v_global{suffix}")
            if isinstance(u_gl, gp.Var):
                obj.addTerms([rhs_val], [u_gl])
            if isinstance(v_gl, gp.Var):
                obj.addTerms([-rhs_val], [v_gl])

        # r1[_p]: scalar, constant RHS = convex_hull_area
        u_r1 = pi_vars.get(f"u_r1{suffix}")
        v_r1 = pi_vars.get(f"v_r1{suffix}")
        if isinstance(u_r1, gp.Var):
            obj.addTerms([float(self.convex_hull_area)], [u_r1])
        if isinstance(v_r1, gp.Var):
            obj.addTerms([-float(self.convex_hull_area)], [v_r1])

        # r2[_p]: indexed, constant RHS = 1
        r2_keys = list(
            (pi_vars.get(f"u_r2{suffix}") or pi_vars.get(f"v_r2{suffix}") or {}).keys()
        )
        r2_rhs = {k: 1.0 for k in r2_keys}
        _addto(f"u_r2{suffix}", f"v_r2{suffix}", r2_rhs)

        # r3[_p]: indexed, x-dependent RHS
        r3_keys = list(
            (pi_vars.get(f"u_r3{suffix}") or pi_vars.get(f"v_r3{suffix}") or {}).keys()
        )
        if which == "yp":
            r3_rhs = {
                (i, j, k, s): 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0)
                for (i, j, k, s) in r3_keys
            }
        else:
            r3_rhs = {
                (i, j, k, s): 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0)
                for (i, j, k, s) in r3_keys
            }
        _addto(f"u_r3{suffix}", f"v_r3{suffix}", r3_rhs)

        # π₀ for "Internal" objective on Y
        if which == "y" and getattr(self, "objective", "Fekete") == "Internal":
            pi0_pos = pi_vars.get("pi0_pos")
            pi0_neg = pi_vars.get("pi0_neg")
            cost_x: dict = getattr(self, "_cost_x", {})
            f_dot_x = sum(cost_x.get(arc, 0.0) * x_sol.get(arc, 0.0) for arc in cost_x)
            pi0_contribution = f_dot_x - eta_sol
            if isinstance(pi0_pos, gp.Var):
                obj.addTerms([pi0_contribution], [pi0_pos])
            if isinstance(pi0_neg, gp.Var):
                obj.addTerms([-pi0_contribution], [pi0_neg])

        return obj

    def invalidate_cgsp_cache(self) -> None:
        """Drop the cached CGSP skeletons so the next call rebuilds from scratch.

        Call this if the primal subproblem structure changes (e.g. dynamic
        addition of strengthening constraints during the solve, or rebuilding
        the master / subproblems with different parameters).  In normal
        Benders flow this is not needed because constraint families are
        fixed at build time.
        """
        self._cgsp_yp_cache = None
        self._cgsp_y_cache = None

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
        # F2.4 — fast path: reuse the cached skeleton (variables + dual
        # feasibility constraints + L₁ normalisation), only re-set the
        # objective.  All structure-only data (instance triangulation,
        # constraint RHS for constant rows) is fixed once the model is built,
        # so the only x_sol-dependent quantity is the objective coefficient
        # vector.
        cache = getattr(self, "_cgsp_yp_cache", None)
        if cache is not None:
            cgsp, pi_vars = cache
            obj_expr = self._compute_cgsp_objective(pi_vars, x_sol, "yp")
            cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
            cgsp.update()
            return cgsp, pi_vars

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
                obj_expr.addTerms([-rhs_k], [v_dp[k]])   # π_δ = -v_δ → contribution -rhs_k * v_δ
                norm_expr.addTerms([w_k], [v_dp[k]])
            pi_vars["v_delta_p"] = v_dp
            # No u_delta_p: π_δ ≤ 0

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

        # ---- Dual feasibility constraints: A^T π ≤ 0 (B-1) ----
        self._build_dual_feasibility_constrs(cgsp, pi_vars, which="yp")

        # ---- L₁ normalisation: Σ w (u + v) = 1 ----
        # Only add if there are variables to normalise
        cgsp.update()
        all_vars = cgsp.getVars()
        if all_vars:
            cgsp.addConstr(norm_expr == 1.0, name="l1_norm")

        # ---- Objective: maximise violation ----
        # F2.4 — replace the obj_expr we accumulated incrementally with the
        # canonical one produced by ``_compute_cgsp_objective``.  The two
        # expressions are mathematically identical for the first call, but
        # using the helper here means the cache-hit path on subsequent calls
        # produces a bit-identical objective and we don't have two parallel
        # implementations of the coefficient formulas.
        obj_expr = self._compute_cgsp_objective(pi_vars, x_sol, "yp")
        cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
        cgsp.update()

        # Cache the skeleton + pi_vars so subsequent callbacks reuse them.
        self._cgsp_yp_cache = (cgsp, pi_vars)

        return cgsp, pi_vars

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

        # F2.4 — cache fast path; symmetric to ``_build_cgsp_yp``.  The π₀
        # variable is recovered from pi_vars under the "pi0_pos" key (since
        # it is just the positive part of the free π₀).
        cache = getattr(self, "_cgsp_y_cache", None)
        if cache is not None:
            cgsp, pi_vars, pi0_var = cache
            obj_expr = self._compute_cgsp_objective(
                pi_vars, x_sol, "y", eta_sol=eta_sol,
            )
            cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
            cgsp.update()
            return cgsp, pi_vars, pi0_var

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
                obj_expr.addTerms([-rhs_k], [v_d2[k]])   # π_δ = -v_δ → contribution -rhs_k * v_δ
                norm_expr.addTerms([w_k], [v_d2[k]])
            pi_vars["v_delta"] = v_d2
            # No u_delta: π_δ ≤ 0

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
            # Objective term: π₀ * (f^T x̄ − η̄)
            # f^T x̄ is computed directly from the cached master cost vector
            # self._cost_x (NEVER from self.sub_y.ObjVal, which is the
            # artificial-slack minimum, not the area).
            cost_x: dict = getattr(self, "_cost_x", {})
            f_dot_x = sum(cost_x.get(arc, 0.0) * x_sol.get(arc, 0.0) for arc in cost_x)
            pi0_contribution = f_dot_x - eta_sol
            w_pi0 = float(weights.get("pi0", 1.0))
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

        # F2.4 — replace the obj_expr we accumulated incrementally with the
        # canonical one produced by ``_compute_cgsp_objective`` so that both
        # the first build and every cache-hit path produce a bit-identical
        # objective.
        obj_expr = self._compute_cgsp_objective(pi_vars, x_sol, "y", eta_sol=eta_sol)

        # ---- Objective ----
        cgsp.setObjective(obj_expr, GRB.MAXIMIZE)
        cgsp.update()

        # F2.4 — cache the skeleton + π₀ ref so subsequent callbacks reuse it.
        self._cgsp_y_cache = (cgsp, pi_vars, pi0_var)

        return cgsp, pi_vars, pi0_var

    # ------------------------------------------------------------------
    # F2.2 — Cut reconstruction helpers (shared between CGSP and MW)
    # ------------------------------------------------------------------

    @staticmethod
    def _net_pi(
        pi_vars: dict,
        u_key: str,
        v_key: str,
        TOL: float = _CGSP_TOL,
    ) -> dict:
        """Return the net dual values π = u - v for one constraint group.

        The sign convention for the split (π = u - v with u, v ≥ 0) is:
          • equality constraint ⇒ π is free  ⇒ both u and v exist.
          • ≥ constraint        ⇒ π ≥ 0      ⇒ only u exists.
          • ≤ constraint        ⇒ π ≤ 0      ⇒ only v exists, and π = -v.

        ``pi_vars`` may contain a single ``gp.Var`` (scalar dual) or a
        ``gp.tupledict`` keyed by the constraint index.

        Returns a dict whose keys are the constraint-group indices (or the
        sentinel ``"scalar"`` for a single-row group), filtered to entries
        with absolute value strictly above TOL.
        """
        out: dict = {}
        u_vars = pi_vars.get(u_key)
        v_vars = pi_vars.get(v_key)
        if u_vars is None and v_vars is None:
            return out
        if isinstance(u_vars, gp.Var) or isinstance(v_vars, gp.Var):
            u_val = u_vars.X if isinstance(u_vars, gp.Var) else 0.0  # type: ignore[union-attr]
            v_val = v_vars.X if isinstance(v_vars, gp.Var) else 0.0  # type: ignore[union-attr]
            val = u_val - v_val
            if abs(val) > TOL:
                out["scalar"] = val
        else:
            keys = (
                u_vars.keys() if u_vars is not None else v_vars.keys()  # type: ignore[union-attr]
            )
            for k in keys:
                u_val = u_vars[k].X if u_vars is not None else 0.0  # type: ignore[index]
                v_val = v_vars[k].X if v_vars is not None else 0.0  # type: ignore[index]
                val = u_val - v_val
                if abs(val) > TOL:
                    out[k] = val
        return out

    def _reconstruct_cut_from_pi(
        self,
        pi_vars: dict,
        x_sol: dict[Arc, float],
        which: str,  # "y" or "yp"
        eta_sol: float = 0.0,
        pi0_var: gp.Var | None = None,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]:
        """Reconstruct the Benders cut from already-solved CGSP / MW pi_vars.

        Mathematics
        -----------
        The CGSP guarantees that, at optimum,  Σ_i π_i b_i(x_bar) > 0,
        i.e. the master point  x_bar  violates the cut

            Σ_i π_i b_i(x) ≤ 0.                                       (★)

        Decompose each row's RHS as  b_i(x) = b_i^const + B_i · x.  Then
        (★) is equivalent to

            (Σ_i π_i B_i) · x   ≤   -Σ_i π_i b_i^const.

        We accumulate the *linear* part on ``cut_expr`` and the *constant*
        part on ``cut_rhs`` explicitly (F2.6 — previously the constants from
        groups with x-dependent RHS lived inside the LinExpr and relied on
        Gurobi's auto-move of LinExpr constants on ``addConstr``).

        For "Internal" objective with a non-zero π₀:
            π₀ · η  ≥  Σ_i π_i b_i(x)
        rearranges to (cut_expr / π₀) - η ≤ -(Σ π_i b_i^const) / π₀.

        Parameters
        ----------
        pi_vars : dict
            Mapping ``u_<group> | v_<group>`` → ``gp.Var | gp.tupledict``,
            populated by ``_build_cgsp_y(p)`` *after* the LP has been
            solved (so that ``.X`` is available).
        x_sol : dict[Arc, float]
            Current master solution (only used to populate the witness;
            the cut itself is in master variable *symbols* ``self.x``).
        which : "y" | "yp"
            Selects the constraint families and their RHS templates.
        eta_sol : float
            Current η value (ignored unless ``pi0_var`` is non-null).
        pi0_var : gp.Var | None
            The π₀_pos variable (the secondary LP returns it for the Y
            subproblem when ``objective == "Internal"``).
        TOL : float
            Numerical zero threshold.

        Returns
        -------
        Same triple contract as ``get_cgsp_cut_yp`` /
        ``get_cgsp_cut_y`` — see those functions' docstrings.
        """
        cut_expr = gp.LinExpr()
        cut_rhs = 0.0
        witness: dict = {}

        suffix = "_p" if which == "yp" else ""
        is_internal = (getattr(self, "objective", "Fekete") == "Internal")

        # Determine if this should be an optimality cut (Y, Internal, π₀ ≠ 0)
        is_optimality_cut = False
        pi0_net = 0.0
        if which == "y" and is_internal and pi0_var is not None:
            pi0_neg_var = pi_vars.get("pi0_neg")
            pi0_net = pi0_var.X - (pi0_neg_var.X if pi0_neg_var is not None else 0.0)
            if abs(pi0_net) > TOL:
                is_optimality_cut = True
                witness["pi0"] = pi0_net

        # ----------------------------------------------------------------
        # alpha[_p]:  b = ±x_arc + (0 or 1)
        #   y   →  b_α(x) = x[i,j]              (const=0, B = +1·x[i,j])
        #   yp  →  b_α'(x) = 1 - x[i,j]         (const=1, B = -1·x[i,j])
        # ----------------------------------------------------------------
        a_net = self._net_pi(pi_vars, f"u_alpha{suffix}", f"v_alpha{suffix}", TOL)
        if a_net:
            witness[f"alpha{suffix}"] = a_net
            for (i, j), pi_val in a_net.items():
                if which == "yp":
                    cut_expr += -pi_val * self.x[i, j]
                    cut_rhs += -pi_val * 1.0
                else:
                    cut_expr += pi_val * self.x[i, j]

        # ----------------------------------------------------------------
        # beta[_p]:  b is purely x-dependent (const = 0)
        #   y   →  b_β(x) = x[i,j] - x[j,i]
        #   yp  →  b_β'(x) = x[j,i] - x[i,j]
        # ----------------------------------------------------------------
        b_net = self._net_pi(pi_vars, f"u_beta{suffix}", f"v_beta{suffix}", TOL)
        if b_net:
            witness[f"beta{suffix}"] = b_net
            for (i, j), pi_val in b_net.items():
                if which == "yp":
                    cut_expr += pi_val * (self.x[j, i] - self.x[i, j])
                else:
                    cut_expr += pi_val * (self.x[i, j] - self.x[j, i])

        # ----------------------------------------------------------------
        # gamma[_p]:  b is purely x-dependent (const = 0)
        #   y   →  b_γ(x) = x[i,j]
        #   yp  →  b_γ'(x) = x[j,i]
        # γ is the dual of a ≥ constraint, so only u_γ exists in pi_vars.
        # ----------------------------------------------------------------
        g_net = self._net_pi(pi_vars, f"u_gamma{suffix}", f"v_gamma{suffix}", TOL)
        if g_net:
            witness[f"gamma{suffix}"] = g_net
            for (i, j), pi_val in g_net.items():
                if which == "yp":
                    cut_expr += pi_val * self.x[j, i]
                else:
                    cut_expr += pi_val * self.x[i, j]

        # ----------------------------------------------------------------
        # delta[_p]: const = 1, B = -1·x_arc
        #   y   →  b_δ(x)  = 1 - x[j,i]
        #   yp  →  b_δ'(x) = 1 - x[i,j]
        # δ is the dual of a ≤ constraint, so only v_δ exists; the helper
        # already maps v-only to π = -v.
        # ----------------------------------------------------------------
        d_net = self._net_pi(pi_vars, f"u_delta{suffix}", f"v_delta{suffix}", TOL)
        if d_net:
            witness[f"delta{suffix}"] = d_net
            for (i, j), pi_val in d_net.items():
                if which == "yp":
                    cut_expr += -pi_val * self.x[i, j]
                else:
                    cut_expr += -pi_val * self.x[j, i]
                cut_rhs += -pi_val * 1.0

        # ----------------------------------------------------------------
        # Constant-RHS groups: global, r1, r2 — pure cut_rhs contributions.
        # ----------------------------------------------------------------
        gl_net = self._net_pi(pi_vars, f"u_global{suffix}", f"v_global{suffix}", TOL)
        if gl_net and "scalar" in gl_net:
            pi_val = gl_net["scalar"]
            constrs = self.constrs_yp if which == "yp" else self.constrs_y
            rhs_val = constrs[f"global{suffix}"].RHS
            cut_rhs += -pi_val * rhs_val
            witness[f"global{suffix}"] = pi_val

        r1_net = self._net_pi(pi_vars, f"u_r1{suffix}", f"v_r1{suffix}", TOL)
        if r1_net and "scalar" in r1_net:
            pi_val = r1_net["scalar"]
            cut_rhs += -pi_val * self.convex_hull_area
            witness[f"r1{suffix}"] = pi_val

        r2_net = self._net_pi(pi_vars, f"u_r2{suffix}", f"v_r2{suffix}", TOL)
        if r2_net:
            witness[f"r2{suffix}"] = r2_net
            for _k, pi_val in r2_net.items():
                cut_rhs += -pi_val * 1.0

        # ----------------------------------------------------------------
        # r3[_p]: const = 1, B = -x_arc1 - x_arc2.
        #   y   →  b_r3(x)  = 1 - x[j,i] - x[k,s]
        #   yp  →  b_r3'(x) = 1 - x[i,j] - x[s,k]
        # ----------------------------------------------------------------
        r3_net = self._net_pi(pi_vars, f"u_r3{suffix}", f"v_r3{suffix}", TOL)
        if r3_net:
            witness[f"r3{suffix}"] = r3_net
            for (i, j, k, s), pi_val in r3_net.items():
                if which == "yp":
                    cut_expr += -pi_val * (self.x[i, j] + self.x[s, k])
                else:
                    cut_expr += -pi_val * (self.x[j, i] + self.x[k, s])
                cut_rhs += -pi_val * 1.0

        # ----------------------------------------------------------------
        # Optimality cut for Y under "Internal" objective.
        # ----------------------------------------------------------------
        if is_optimality_cut and which == "y" and is_internal:
            if pi0_net < -TOL:
                logger.warning(
                    "CGSP-Y: pi0_net=%.3e < -TOL; skipping cut "
                    "(numerical sign anomaly).",
                    pi0_net,
                )
                return None, None, {"aborted": "pi0_negative", "pi0_net": pi0_net}
            if pi0_net > TOL:
                # Cut form: π₀·η ≥ cut_expr - cut_rhs
                # → (cut_expr - cut_rhs) / π₀ ≤ η
                # → (cut_expr / π₀) - η ≤ cut_rhs / π₀
                scale = 1.0 / pi0_net
                scaled_expr = scale * cut_expr - self.eta
                scaled_rhs = cut_rhs * scale
                witness["is_optimality_cut"] = True
                return scaled_expr, scaled_rhs, witness
            # |pi0_net| ≤ TOL → fall through and emit a feasibility cut.

        return cut_expr, cut_rhs, witness

    # ------------------------------------------------------------------
    # Task 8: Public entry points
    # ------------------------------------------------------------------

    def get_cgsp_cut_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]:
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
        cgsp, pi_vars = self._build_cgsp_yp(x_sol, TOL=TOL)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y') did not solve to optimality (status=%d).", cgsp.Status)
            return None, None, {"aborted": "solve_failed", "status": cgsp.Status}

        if cgsp.ObjVal <= TOL:
            # No violated cut found
            return None, None, {"aborted": "no_violation"}

        # F2.2: cut reconstruction lives in _reconstruct_cut_from_pi so the
        # MW mixin can reuse it on the secondary LP's solution rather than
        # re-running CGSP from scratch.
        return self._reconstruct_cut_from_pi(
            pi_vars, x_sol, which="yp", TOL=TOL,
        )

    def get_cgsp_cut_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        TOL: float = _CGSP_TOL,
    ) -> tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]:
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
        cgsp, pi_vars, pi0_var = self._build_cgsp_y(x_sol, eta_sol=eta_sol, TOL=TOL)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.warning("CGSP (Y) did not solve to optimality (status=%d).", cgsp.Status)
            return None, None, {"aborted": "solve_failed", "status": cgsp.Status}

        if cgsp.ObjVal <= TOL:
            # No violated cut
            return None, None, {"aborted": "no_violation"}

        # F2.2: delegate to the shared reconstructor; π₀ handling lives there.
        return self._reconstruct_cut_from_pi(
            pi_vars,
            x_sol,
            which="y",
            eta_sol=eta_sol,
            pi0_var=pi0_var,
            TOL=TOL,
        )
