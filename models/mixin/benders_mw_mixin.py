"""Magnanti-Wong Pareto-optimal Benders cut mixin.

Provides `get_mw_cut_y` and `get_mw_cut_yp` — entry points that solve a
secondary LP to find the deepest cut among all Pareto-optimal (non-dominated)
Benders cuts for the current master solution.

Reference: Magnanti & Wong (1981), "Accelerating Benders Decomposition:
Algorithmic Enhancement and Model Selection Criteria."

Design spec: .claude/context/designs/2026-05-04-pi-cgsp-mw-implementation.md
Packages C-1 and C-5.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

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
    _core_point: "dict[Arc, float] | None"
    _core_point_strategy: str

    # ------------------------------------------------------------------
    # C-5: Core point computation
    # ------------------------------------------------------------------

    def _compute_core_point(self, strategy: str) -> "dict[Arc, float]":
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
            # NOTE (F1.1): the master variable dict is `self.x` (no underscore).
            # An earlier draft used `self._x` which does not exist; the resulting
            # empty dict made every MW callback abort with "no_core_point" and
            # silently fall back to legacy Farkas — see audit
            # .claude/context/reviews/2026-05-04-cgsp-paper-dissonance.md §2.1.
            return {arc: 1.0 / n for arc in getattr(self, "x", {})}

        # 'lp_relaxation'
        try:
            lp = self.model.relax()  # type: ignore[attr-defined]
            lp.setParam("OutputFlag", 0)
            lp.setParam("TimeLimit", 60)
            lp.optimize()
            if lp.Status == GRB.OPTIMAL:
                result: dict = {}
                # F1.1: same correction as above — iterate `self.x`, not `self._x`.
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
    # C-1: Secondary LP construction
    # ------------------------------------------------------------------

    def _build_mw_secondary_lp(
        self,
        x_sol: "dict[Arc, float]",
        q_opt: "float | None",
        which: str,  # "y" or "yp"
        TOL: float = _MW_TOL,
    ) -> "tuple[gp.Model, dict, gp.LinExpr, gp.LinExpr | None] | tuple[None, None, None, None]":
        """Build the Magnanti-Wong secondary LP for subproblem Y or Y'.

        Secondary LP
        ------------
            max  π^T b(x^0)              [maximise at core point]
            s.t. A^T π ≤ 0               [dual feasibility — from CGSP B-1]
                 π^T b(x_bar) = q_opt    [Pareto constraint, if feasible]
                 Σ w|π| = 1              [L₁ normalisation]

        F2.3 (audit §4.3): when the original primal subproblem is infeasible
        ``q_opt`` is undefined (the dual value is +∞ in the Farkas sense).
        Pass ``q_opt = None`` and the Pareto constraint is dropped — this is
        the Papadakos relaxation (Hosseini & Turner 2025 §3.3.3 eq. 29).
        Setting ``q_opt = 0.0`` for an infeasible subproblem is wrong because
        it forces ``π^T b(x_bar) = 0``, which in general excludes every
        valid Farkas certificate.

        Returns
        -------
        (mw_model, pi_vars, obj_expr_x0, obj_expr_xbar)
            ``obj_expr_xbar`` is ``None`` when the Pareto constraint is
            dropped (Papadakos).  Returns the all-``None`` tuple on error.
        """
        core_point = getattr(self, "_core_point", None)
        if not core_point:
            logger.warning("MW: core point not set; cannot build secondary LP.")
            return None, None, None, None

        # Build the base CGSP model (variable structure + dual feasibility constrs)
        try:
            if which == "yp":
                mw, pi_vars = self._build_cgsp_yp(x_sol)  # type: ignore[attr-defined]
            else:
                mw, pi_vars, _pi0_var = self._build_cgsp_y(x_sol)  # type: ignore[attr-defined]

            x_bar_expr: "gp.LinExpr | None" = None
            if q_opt is not None:
                # Pareto constraint: π^T b(x_bar) = q_opt
                x_bar_expr = _recompute_obj_at_x(pi_vars, x_sol, which, self)  # type: ignore[arg-type]
                if x_bar_expr is None:
                    return None, None, None, None
                mw.addConstr(x_bar_expr == q_opt, name="mw_pareto")

            # Replace objective: maximise at core point x^0
            x0_expr = _recompute_obj_at_x(pi_vars, core_point, which, self)  # type: ignore[arg-type]
            if x0_expr is None:
                return None, None, None, None
            mw.setObjective(x0_expr, GRB.MAXIMIZE)
            mw.update()
            return mw, pi_vars, x0_expr, x_bar_expr
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW secondary LP build failed (%s).", exc)
            return None, None, None, None

    # ------------------------------------------------------------------
    # C-1: Public entry points
    # ------------------------------------------------------------------

    def get_mw_cut_yp(
        self,
        x_sol: "dict[Arc, float]",
        TOL: float = _MW_TOL,
    ) -> "tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]":
        """Pareto-optimal cut for Y' subproblem.

        Returns (None, None, {'aborted': reason}) when:
        - secondary LP infeasible or no violation
        - core point not set
        """
        if not getattr(self, "_core_point", None):
            return None, None, {"aborted": "no_core_point"}

        sub_yp = getattr(self, "sub_yp", None)
        if sub_yp is None:
            return None, None, {"aborted": "sub_yp_none"}

        yp_status = sub_yp.Status
        if yp_status == GRB.OPTIMAL:
            q_opt: "float | None" = sub_yp.ObjVal
        elif yp_status == GRB.INFEASIBLE:
            # F2.3 (audit §4.3): Papadakos — drop the Pareto constraint when
            # the primal is infeasible.  q_opt = 0.0 was wrong because it
            # forced π^T b(x_bar) = 0 which is generically inconsistent with
            # any valid Farkas certificate.
            q_opt = None
        else:
            return None, None, {"aborted": f"sub_yp_status_{yp_status}"}

        mw, pi_vars, _x0_expr, _xbar_expr = self._build_mw_secondary_lp(
            x_sol, q_opt, which="yp", TOL=TOL
        )
        if mw is None:
            return None, None, {"aborted": "build_failed"}

        mw.setParam("OutputFlag", 0)
        mw.optimize()

        if mw.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug("MW (Y') secondary LP status=%d.", mw.Status)
            return None, None, {"aborted": "secondary_lp_failed", "status": mw.Status}

        if mw.ObjVal <= TOL:
            return None, None, {"aborted": "no_violation"}

        # F2.2 (audit §4.2): reconstruct the cut from the secondary LP's own
        # π★, NOT by re-running CGSP from scratch.  The previous version
        # called self.get_cgsp_cut_yp again, which built a *new* CGSP model,
        # solved it, and used the resulting π — that π is generically Pareto-
        # dominated by the secondary LP's π★, so the emitted cut was not the
        # MW-optimal cut at all.
        try:
            cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(  # type: ignore[attr-defined]
                pi_vars, x_sol, which="yp", TOL=TOL
            )
            if cut_expr is None:
                return None, None, witness
            witness["mw"] = True
            witness["mw_papadakos"] = (q_opt is None)
            return cut_expr, cut_rhs, witness
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y') cut reconstruction failed (%s).", exc)
            return None, None, {"aborted": "cut_recon_failed"}

    def get_mw_cut_y(
        self,
        x_sol: "dict[Arc, float]",
        TOL: float = _MW_TOL,
    ) -> "tuple[gp.LinExpr, float, dict] | tuple[None, None, dict]":
        """Pareto-optimal cut for Y subproblem. Symmetric to get_mw_cut_yp."""
        if not getattr(self, "_core_point", None):
            return None, None, {"aborted": "no_core_point"}

        sub_y = getattr(self, "sub_y", None)
        if sub_y is None:
            return None, None, {"aborted": "sub_y_none"}

        y_status = sub_y.Status
        if y_status == GRB.OPTIMAL:
            q_opt: "float | None" = sub_y.ObjVal
        elif y_status == GRB.INFEASIBLE:
            # F2.3 — Papadakos: drop the Pareto constraint on infeasible.
            q_opt = None
        else:
            return None, None, {"aborted": f"sub_y_status_{y_status}"}

        mw, pi_vars, _x0_expr, _xbar_expr = self._build_mw_secondary_lp(
            x_sol, q_opt, which="y", TOL=TOL
        )
        if mw is None:
            return None, None, {"aborted": "build_failed"}

        mw.setParam("OutputFlag", 0)
        mw.optimize()

        if mw.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.debug("MW (Y) secondary LP status=%d.", mw.Status)
            return None, None, {"aborted": "secondary_lp_failed", "status": mw.Status}

        if mw.ObjVal <= TOL:
            return None, None, {"aborted": "no_violation"}

        # F2.2: reuse the shared reconstructor on the secondary LP's π★.
        # Note: pi0_var is None here because we did not separately build/store
        # the Y CGSP's π₀ for the MW path.  MW under the "Internal" objective
        # is therefore restricted to feasibility cuts; an extension to MW
        # optimality cuts (with a free π₀ in the secondary LP) is future work.
        try:
            cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi(  # type: ignore[attr-defined]
                pi_vars, x_sol, which="y", TOL=TOL
            )
            if cut_expr is None:
                return None, None, witness
            witness["mw"] = True
            witness["mw_papadakos"] = (q_opt is None)
            return cut_expr, cut_rhs, witness
        except Exception as exc:  # noqa: BLE001
            logger.warning("MW (Y) cut reconstruction failed (%s).", exc)
            return None, None, {"aborted": "cut_recon_failed"}


# ------------------------------------------------------------------
# Module-level helpers (not methods — avoid polluting the mixin namespace)
# ------------------------------------------------------------------

def _recompute_obj_at_x(
    pi_vars: dict,
    x_sol: "dict[Arc, float]",
    which: str,
    model: object,
) -> "gp.LinExpr | None":
    """Rebuild π^T b(x) as a LinExpr from pi_vars evaluated at x_sol.

    This mirrors the CGSP objective construction but uses a different x.
    Used to build both the Pareto constraint and the M-W objective.

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

    # F2.2 extension — strengthening duals (r1, r2, r3) must also enter the
    # Pareto / core-point objective so the MW LP is consistent with the cut
    # reconstruction.  These were missing from the previous version, which
    # silently dropped any contribution coming from the strengthening rows.
    convex_hull_area = float(getattr(model, "convex_hull_area", 0.0) or 0.0)

    # r1[_p]: scalar, constant RHS = convex_hull_area
    u_r1 = pi_vars.get(f"u_r1{suffix}")
    v_r1 = pi_vars.get(f"v_r1{suffix}")
    if isinstance(u_r1, gp.Var):
        expr.addTerms([convex_hull_area], [u_r1])
    if isinstance(v_r1, gp.Var):
        expr.addTerms([-convex_hull_area], [v_r1])

    # r2[_p]: indexed, constant RHS = 1
    r2_keys = list((pi_vars.get(f"u_r2{suffix}") or pi_vars.get(f"v_r2{suffix}") or {}).keys())
    r2_rhs = {k: 1.0 for k in r2_keys}
    _add_indexed(f"u_r2{suffix}", f"v_r2{suffix}", r2_rhs)

    # r3[_p]: indexed, x-dependent RHS = 1 - x[arc1] - x[arc2]
    r3_keys = list((pi_vars.get(f"u_r3{suffix}") or pi_vars.get(f"v_r3{suffix}") or {}).keys())
    if which == "yp":
        # b_r3'(x) = 1 - x[i,j] - x[s,k]  for key = (i, j, k, s)
        r3_rhs = {
            (i, j, k, s): 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0)
            for (i, j, k, s) in r3_keys
        }
    else:
        # b_r3(x)  = 1 - x[j,i] - x[k,s]
        r3_rhs = {
            (i, j, k, s): 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0)
            for (i, j, k, s) in r3_keys
        }
    _add_indexed(f"u_r3{suffix}", f"v_r3{suffix}", r3_rhs)

    return expr
