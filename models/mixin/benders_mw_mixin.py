"""Magnanti-Wong Pareto-optimal Benders cut mixin.

Provides `get_mw_cut_y` and `get_mw_cut_yp` — entry points that select the
deepest Pareto-optimal Benders cut for the current master solution.

Algorithm (Magnanti & Wong 1981, Papadakos 2008, Hosseini & Turner 2025 §3.3.3):
    1. Solve the primary CGSP to obtain q* = max normalised violation at x̄.
    2. If q* ≤ TOL → no cut exists, return None.
    3. Extract the CGSP cut as a fallback (always valid since q* > 0).
    4. Build the MW secondary LP by modifying the CGSP in-place:
           Pareto constraint : cgsp_obj_expr == q*   (uses cgsp.getObjective()
                                                       — complete, includes all groups)
           New objective     : max π^T b(x^0)         (via _recompute_obj_at_x)
    5. Re-solve.  If the secondary LP fails or gives no improvement → return
       the CGSP fallback cut (always valid).
    6. Otherwise reconstruct cut from MW's π★ and return.

This replaces the previous `_build_mw_secondary_lp` helper which had two bugs:
    Bug 1: used `sub_y.ObjVal` (primal, unnormalised) as q_opt instead of the
           normalised CGSP optimal value — making the Pareto constraint
           dimensionally inconsistent and the secondary LP typically infeasible.
    Bug 2: for infeasible subproblems, omitted the Pareto constraint entirely
           (add_pareto=False), allowing the secondary LP to return π with
           π^T b(x̄) ≤ 0 — a cut that does NOT cut off x̄, causing infinite LP
           loops and incorrect IP incumbents.

Reference: Magnanti & Wong (1981), "Accelerating Benders Decomposition:
Algorithmic Enhancement and Model Selection Criteria."

Design spec: .claude/context/designs/2026-05-04-pi-cgsp-mw-implementation.md
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    from utils.geometry import Arc

logger = logging.getLogger(__name__)

_MW_TOL = 1e-8


class BendersMagnantiWongMixin:
    """Pareto-optimal Benders cuts (Magnanti & Wong, 1981).

    Entry points
    ------------
    get_mw_cut_y(x_sol)   -> (LinExpr, float, dict) | (None, None, dict)
    get_mw_cut_yp(x_sol)  -> (LinExpr, float, dict) | (None, None, dict)

    All returned cuts have the form  cut_expr <= cut_rhs  (same contract as CGSP).

    Configuration attributes (set by build())
    ------------------------------------------
    use_magnanti_wong       : bool          — dispatch switch (default False)
    _core_point             : dict[Arc, float] | None — x^0
    _core_point_strategy    : str           — 'lp_relaxation' | 'uniform'
    """

    use_magnanti_wong: bool
    _core_point: dict[Arc, float] | None
    _core_point_strategy: str

    # ------------------------------------------------------------------
    # C-5: Core point computation
    # ------------------------------------------------------------------

    def _compute_core_point(self, strategy: str) -> dict[Arc, float]:
        """Return a point in relint(conv(X_M)).

        Strategies
        ----------
        'uniform':
            x^0[arc] = 1/N for every arc.  Zero cost.
        'lp_relaxation':
            Relax self.model, solve, use LP optimal as x^0.
            Falls back to 'uniform' on any failure.
        """
        if strategy == "uniform":
            n = getattr(self, "N", None)
            if n is None or n == 0:
                return {}
            return {arc: 1.0 / n for arc in getattr(self, "x", {})}

        # 'lp_relaxation'
        try:
            lp = self.model.relax()  # type: ignore[attr-defined]
            lp.setParam("OutputFlag", 0)
            lp.setParam("TimeLimit", 60)
            lp.optimize()
            if lp.Status == GRB.OPTIMAL:
                result: dict = {}
                for arc in getattr(self, "x", {}):
                    v = lp.getVarByName(f"x_{arc[0]}_{arc[1]}")
                    if v is not None:
                        result[arc] = v.X
                return result
            else:
                logger.warning(
                    "Core point LP did not solve to optimality (status=%d). "
                    "Falling back to uniform strategy.",
                    lp.Status,
                )
                return self._compute_core_point("uniform")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Core point LP failed (%s). Using uniform.", exc)
            return self._compute_core_point("uniform")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def get_mw_cut_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _MW_TOL,
    ) -> tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]:
        """Pareto-optimal cut for Y' subproblem.

        Returns (None, None, {'aborted': reason}) when:
        - no core point set
        - CGSP finds no violation (q* <= TOL)
        - secondary LP and CGSP fallback both fail
        """
        core_point = getattr(self, "_core_point", None)
        if not core_point:
            return None, None, {"aborted": "no_core_point"}

        # Step 1: build and solve the primary CGSP to get q* (max violation at x̄)
        try:
            cgsp, pi_vars = self._build_cgsp_yp(x_sol)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y'): CGSP build failed (%s).", exc)
            return None, None, {"aborted": "cgsp_build_failed"}

        cgsp.setParam("OutputFlag", 0)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug("MW (Y'): primary CGSP status=%d.", cgsp.Status)
            return None, None, {"aborted": "cgsp_failed", "status": cgsp.Status}

        q_opt = cgsp.ObjVal
        if q_opt <= TOL:
            return None, None, {"aborted": "no_violation"}

        # Step 2: extract CGSP cut as fallback (always valid since q* > 0)
        try:
            cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness = (
                self._reconstruct_cut_from_pi(pi_vars, which="yp", TOL=TOL)  # type: ignore[attr-defined]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y'): CGSP fallback reconstruction failed (%s).", exc)
            return None, None, {"aborted": "cgsp_recon_failed"}

        # Step 3: capture the complete Pareto expression from the CGSP objective
        # (cgsp.getObjective() is COMPLETE — includes r1, r2, r3 groups too)
        x_bar_expr = cgsp.getObjective()

        # Step 4: build the MW objective at x^0
        x0_expr = _recompute_obj_at_x(pi_vars, core_point, "yp", self)  # type: ignore[arg-type]
        if x0_expr is None:
            logger.debug("MW (Y'): x0_expr is None; using CGSP fallback.")
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        # Step 5: modify CGSP in-place → MW secondary LP
        try:
            cgsp.addConstr(x_bar_expr == q_opt, name="mw_pareto")
            cgsp.setObjective(x0_expr, GRB.MAXIMIZE)
            cgsp.update()
            cgsp.optimize()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y'): secondary LP solve failed (%s); using CGSP fallback.", exc)
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug(
                "MW (Y'): secondary LP status=%d; using CGSP fallback.", cgsp.Status
            )
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        if cgsp.ObjVal <= TOL:
            # MW found no improvement at core point; the CGSP cut is equally good
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        # Step 6: reconstruct cut from MW's π★
        try:
            cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(  # type: ignore[attr-defined]
                pi_vars, which="yp", TOL=TOL
            )
            witness["mw"] = True
            return cut_expr, cut_rhs, witness
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y'): MW cut reconstruction failed (%s); using CGSP fallback.", exc)
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

    def get_mw_cut_y(
        self,
        x_sol: dict[Arc, float],
        TOL: float = _MW_TOL,
    ) -> tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]:
        """Pareto-optimal cut for Y subproblem. Symmetric to get_mw_cut_yp."""
        core_point = getattr(self, "_core_point", None)
        if not core_point:
            return None, None, {"aborted": "no_core_point"}

        # Step 1: build and solve the primary CGSP to get q* (max violation at x̄)
        try:
            cgsp, pi_vars, _pi0_var = self._build_cgsp_y(x_sol)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y): CGSP build failed (%s).", exc)
            return None, None, {"aborted": "cgsp_build_failed"}

        cgsp.setParam("OutputFlag", 0)
        cgsp.optimize()

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug("MW (Y): primary CGSP status=%d.", cgsp.Status)
            return None, None, {"aborted": "cgsp_failed", "status": cgsp.Status}

        q_opt = cgsp.ObjVal
        if q_opt <= TOL:
            return None, None, {"aborted": "no_violation"}

        # Step 2: extract CGSP cut as fallback (always valid since q* > 0)
        try:
            cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness = (
                self._reconstruct_cut_from_pi(pi_vars, which="y", TOL=TOL)  # type: ignore[attr-defined]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y): CGSP fallback reconstruction failed (%s).", exc)
            return None, None, {"aborted": "cgsp_recon_failed"}

        # Step 3: capture the complete Pareto expression from the CGSP objective
        x_bar_expr = cgsp.getObjective()

        # Step 4: build the MW objective at x^0
        x0_expr = _recompute_obj_at_x(pi_vars, core_point, "y", self)  # type: ignore[arg-type]
        if x0_expr is None:
            logger.debug("MW (Y): x0_expr is None; using CGSP fallback.")
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        # Step 5: modify CGSP in-place → MW secondary LP
        try:
            cgsp.addConstr(x_bar_expr == q_opt, name="mw_pareto")
            cgsp.setObjective(x0_expr, GRB.MAXIMIZE)
            cgsp.update()
            cgsp.optimize()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y): secondary LP solve failed (%s); using CGSP fallback.", exc)
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        if cgsp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug(
                "MW (Y): secondary LP status=%d; using CGSP fallback.", cgsp.Status
            )
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        if cgsp.ObjVal <= TOL:
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness

        # Step 6: reconstruct cut from MW's π★
        try:
            cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(  # type: ignore[attr-defined]
                pi_vars, which="y", TOL=TOL
            )
            witness["mw"] = True
            return cut_expr, cut_rhs, witness
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y): MW cut reconstruction failed (%s); using CGSP fallback.", exc)
            return cgsp_cut_expr, cgsp_cut_rhs, cgsp_witness


# ------------------------------------------------------------------
# Module-level helpers (not methods — avoid polluting the mixin namespace)
# ------------------------------------------------------------------

def _recompute_obj_at_x(
    pi_vars: dict,
    x_sol: dict[Arc, float],
    which: str,
    model: object,
) -> gp.LinExpr | None:
    """Rebuild π^T b(x) as a LinExpr from pi_vars evaluated at x_sol.

    Used to build the MW objective at the core point x^0 (NOT the Pareto
    constraint — for that we use cgsp.getObjective() directly, which is
    complete and includes all constraint groups including r1/r2/r3).

    Covers: alpha, beta, gamma, delta, global, r1, r2, r3 (and _p variants).

    Returns None if pi_vars is empty or x_sol is empty.
    """
    if not pi_vars or not x_sol:
        return None

    suffix = "_p" if which == "yp" else ""
    expr = gp.LinExpr()

    # Helper: add contribution of a pi variable group with given RHS dict
    def _add_indexed(u_key: str, v_key: str, rhs_map: dict) -> None:
        u_vars = pi_vars.get(u_key)
        v_vars = pi_vars.get(v_key)
        if u_vars is None and v_vars is None:
            return
        keys = u_vars.keys() if u_vars is not None else v_vars.keys()  # type: ignore[union-attr]
        for k in keys:
            rhs_k = rhs_map.get(k, 0.0)
            if abs(rhs_k) < 1e-15:
                continue
            if u_vars is not None:
                expr.addTerms([rhs_k], [u_vars[k]])  # type: ignore[index]
            if v_vars is not None:
                expr.addTerms([-rhs_k], [v_vars[k]])  # type: ignore[index]

    # alpha[_p]: RHS = 1 - x[i,j]  (yp) or x[i,j] (y)
    if which == "yp":
        alpha_rhs = {k: 1.0 - x_sol.get(k, 0.0) for k in pi_vars.get(f"u_alpha{suffix}", {}).keys()  # type: ignore[union-attr]
                     or pi_vars.get(f"v_alpha{suffix}", {}).keys()}  # type: ignore[union-attr]
    else:
        alpha_rhs = {k: x_sol.get(k, 0.0) for k in pi_vars.get(f"u_alpha{suffix}", {}).keys()  # type: ignore[union-attr]
                     or pi_vars.get(f"v_alpha{suffix}", {}).keys()}  # type: ignore[union-attr]
    _add_indexed(f"u_alpha{suffix}", f"v_alpha{suffix}", alpha_rhs)

    # beta[_p]: RHS = x[j,i] - x[i,j]  (yp) or x[i,j] - x[j,i]  (y)
    beta_keys = list((pi_vars.get(f"u_beta{suffix}") or pi_vars.get(f"v_beta{suffix}") or {}).keys())
    if which == "yp":
        beta_rhs = {k: x_sol.get((k[1], k[0]), 0.0) - x_sol.get(k, 0.0) for k in beta_keys}
    else:
        beta_rhs = {k: x_sol.get(k, 0.0) - x_sol.get((k[1], k[0]), 0.0) for k in beta_keys}
    _add_indexed(f"u_beta{suffix}", f"v_beta{suffix}", beta_rhs)

    # gamma[_p]: RHS = x[j,i]  (yp) or x[i,j]  (y)  — only u_gamma
    gamma_keys = list((pi_vars.get(f"u_gamma{suffix}") or {}).keys())
    if which == "yp":
        gamma_rhs = {k: x_sol.get((k[1], k[0]), 0.0) for k in gamma_keys}
    else:
        gamma_rhs = {k: x_sol.get(k, 0.0) for k in gamma_keys}
    u_g = pi_vars.get(f"u_gamma{suffix}")
    if u_g is not None:
        for k in gamma_keys:
            rhs_k = gamma_rhs.get(k, 0.0)
            if abs(rhs_k) > 1e-15:
                expr.addTerms([rhs_k], [u_g[k]])  # type: ignore[index]

    # delta[_p]: RHS = 1 - x[i,j]  (yp) or 1 - x[j,i]  (y)  — only -v_delta
    delta_keys = list((pi_vars.get(f"v_delta{suffix}") or {}).keys())
    if which == "yp":
        delta_rhs = {k: 1.0 - x_sol.get(k, 0.0) for k in delta_keys}
    else:
        delta_rhs = {k: 1.0 - x_sol.get((k[1], k[0]), 0.0) for k in delta_keys}
    v_d = pi_vars.get(f"v_delta{suffix}")
    if v_d is not None:
        for k in delta_keys:
            rhs_k = delta_rhs.get(k, 0.0)
            if abs(rhs_k) > 1e-15:
                expr.addTerms([-rhs_k], [v_d[k]])  # type: ignore[index]

    # global[_p]: constant RHS — contributes to objective as π * RHS
    constrs = getattr(model, "constrs_yp" if which == "yp" else "constrs_y", {})
    global_rhs = constrs.get(f"global{suffix}")
    if global_rhs is not None and hasattr(global_rhs, "RHS"):
        rhs_val = global_rhs.RHS  # type: ignore[attr-defined]
        u_gl = pi_vars.get(f"u_global{suffix}")
        v_gl = pi_vars.get(f"v_global{suffix}")
        if isinstance(u_gl, gp.Var):
            expr.addTerms([rhs_val], [u_gl])
        if isinstance(v_gl, gp.Var):
            expr.addTerms([-rhs_val], [v_gl])

    # r1[_p]: constant RHS = convex_hull_area  (≤ constraint, π_r1 = -v_r1)
    v_r1 = pi_vars.get(f"v_r1{suffix}")
    if isinstance(v_r1, gp.Var) and hasattr(model, "convex_hull_area"):
        ch_area = model.convex_hull_area
        if abs(ch_area) > 1e-15:
            expr.addTerms([-ch_area], [v_r1])

    # r2[_p]: constant RHS = 1 per arc  (≤ constraint, π_r2 = -v_r2)
    v_r2 = pi_vars.get(f"v_r2{suffix}")
    if v_r2 is not None and not isinstance(v_r2, gp.Var):
        for k in v_r2.keys():  # type: ignore[union-attr]
            expr.addTerms([-1.0], [v_r2[k]])  # type: ignore[index]

    # r3[_p]: RHS = 1 - x[i,j] - x[k,s]  (≤ constraint, π_r3 = -v_r3)
    v_r3 = pi_vars.get(f"v_r3{suffix}")
    if v_r3 is not None and not isinstance(v_r3, gp.Var):
        for k in v_r3.keys():  # type: ignore[union-attr]
            if which == "yp":
                i, j, ks, s = k
                rhs_k = 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, ks), 0.0)
            else:
                i, j, k2, s = k
                rhs_k = 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k2, s), 0.0)
            if abs(rhs_k) > 1e-15:
                expr.addTerms([-rhs_k], [v_r3[k]])  # type: ignore[index]

    return expr
