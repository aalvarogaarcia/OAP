# models/mixin/benders_ddma_mixin.py
"""
BendersDDMAMixin — Distance-based Depth-Maximisation Algorithm (DDMA).

Implements Algorithm 3 from:
    Hosseini, M. and Turner, J.
    "Deepest Cuts for Benders Decomposition"
    Operations Research 73(5):2591-2609, 2025.
    §4.1, p. 2602.

Overview
--------
DDMA avoids building a full normalised-LP separation problem (as CGSP does).
Instead it solves a sequence of *standard* Benders subproblems with a
progressively perturbed RHS:

    Step 0:  z ← 0.
    loop:
        Step 1 — solve sub with  RHS = b(x̄) − z · w  (perturbed)
                 if INFEASIBLE → π = Farkas ray
                 if OPTIMAL    → π = .Pi dual values
        Step 2 — depth update:
                 ẑ = (π^T (b − B x̄)) / (w^T π)
                 if |ẑ − z| < ε: return π  ← DONE
                 z ← ẑ
    end loop

The key advantages over CGSP-LP:

1. **No auxiliary LP** — each DDMA iteration is one classic Farkas/Pi
   subproblem solve, reusing the already-built ``sub_y`` / ``sub_yp``
   Gurobi models with warm-started LP bases.

2. **Early-termination safety** — Remark 6 in the paper guarantees that
   any iterate π already produces a *valid* Benders cut (feasibility or
   optimality), so the algorithm can be stopped early under a budget.

3. **~80% time reduction** vs CGSP-LP on CFLP/MCNDP/SNIP (§5.2 p. 2604).

Weight scheme
-------------
We use the Relaxed-ℓ₁ pseudonorm (paper §3.3.2, p. 2600), matching the
weights in ``BendersCGSPMixin._compute_relaxed_l1_weights``:

    w_α = w_γ = w_δ = 1   (single x-variable in RHS)
    w_β = w_{r3} = 2      (two x-variables)
    w_{global} = w_{r1} = w_{r2} = 0  (constant rows, no x-dependence)

Design contract
---------------
Assumes: BendersFarkasMixin / BendersPiMixin already built
``self.constrs_y``, ``self.constrs_yp``, ``self.sub_y``, ``self.sub_yp``.

Must appear before BendersFarkasMixin / BendersPiMixin in the MRO so
``OAPBendersModel`` resolves the DDMA entry points first.

``sub_y`` and ``sub_yp`` must have been built with
    sub_y.Params.InfUnbdInfo = 1
so that ``FarkasDual`` is available after INFEASIBLE solves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

Arc = tuple[int, int]

_DDMA_TOL: float = 1e-8
_DDMA_EPS: float = 1e-6
_DDMA_MAX_ITER: int = 20


class BendersDDMAMixin:
    """Mixin providing DDMA deepest cuts (Algorithm 3, Hosseini & Turner 2025).

    Public entry points
    -------------------
    get_ddma_cut_y(x_sol, eta_sol=0.0, max_iter=20, eps=1e-6)
    get_ddma_cut_yp(x_sol, max_iter=20, eps=1e-6)

    Both return (cut_expr, cut_rhs, witness) or (None, None, dict).
    """

    # --- Type hints (linter only) ---
    model: gp.Model
    sub_y: gp.Model
    sub_yp: gp.Model
    x: dict[str, Any]
    constrs_y: dict[str, Any]
    constrs_yp: dict[str, Any]
    N: int
    convex_hull_area: float

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _get_ddma_weights(self, which: str) -> dict[str, float]:
        """Return the Relaxed-ℓ₁ weight for every live constraint group.

        Returns a flat dict mapping group-name → scalar weight.  Groups
        absent from the constraint dicts are not included.

        w = 0 for constant-RHS rows (global, r1, r2) because they do not
        depend on x̄ — perturbing them would move the feasible region
        regardless of x̄.
        """
        constrs = self.constrs_yp if which == "yp" else self.constrs_y
        suffix = "_p" if which == "yp" else ""

        # group_name → Relaxed-ℓ₁ weight
        _template: dict[str, float] = {
            f"alpha{suffix}": 1.0,  # RHS = ±x[arc]           → 1 x-var
            f"beta{suffix}": 2.0,  # RHS = x[a] - x[b]       → 2 x-vars
            f"gamma{suffix}": 1.0,  # RHS = x[arc]             → 1 x-var
            f"delta{suffix}": 1.0,  # RHS = 1 - x[arc]         → 1 x-var
            f"global{suffix}": 0.0,  # constant RHS              → 0
            f"r1{suffix}": 0.0,  # constant RHS              → 0
            f"r2{suffix}": 0.0,  # constant RHS = 1          → 0
            f"r3{suffix}": 2.0,  # RHS = 1 - x[a] - x[b]   → 2 x-vars
        }
        return {k: v for k, v in _template.items() if k in constrs}

    # ------------------------------------------------------------------
    # RHS perturbation
    # ------------------------------------------------------------------

    def _apply_ddma_perturbation(
        self,
        z: float,
        which: str,
        weights: dict[str, float],
    ) -> None:
        """Subtract ``z * w_i`` from every constraint's current RHS.

        Called *after* ``_update_subproblem_rhs(x_sol)`` has already set
        ``constr.RHS = b_i(x̄)`` for each constraint.  The net result is
        ``RHS = b_i(x̄) − z * w_i``.

        Parameters
        ----------
        z : float
            Current depth parameter.
        which : "y" | "yp"
            Selects the subproblem.
        weights : dict
            Flat weight dict from ``_get_ddma_weights``.
        """
        if abs(z) < _DDMA_TOL:
            return  # nothing to do at z=0

        constrs = self.constrs_yp if which == "yp" else self.constrs_y

        for group_name, w in weights.items():
            if w == 0.0:
                continue
            group_val = constrs.get(group_name)
            if group_val is None:
                continue
            if isinstance(group_val, dict):
                for constr in group_val.values():
                    constr.RHS -= z * w
            elif isinstance(group_val, gp.Constr):
                group_val.RHS -= z * w

    # ------------------------------------------------------------------
    # π extraction from solved subproblem
    # ------------------------------------------------------------------

    def _extract_ddma_pi(
        self,
        which: str,
        TOL: float = _DDMA_TOL,
    ) -> dict[str, dict[str, Any] | float]:
        """Extract dual multipliers π from the most recent subproblem solve.

        Returns a nested dict matching the structure of ``constrs_y`` /
        ``constrs_yp``:  ``{group_name: {arc_key: float} | float}``.

        For INFEASIBLE subproblems: reads ``.FarkasDual`` (requires
        ``sub_y.Params.InfUnbdInfo = 1`` at build time).
        For OPTIMAL subproblems:   reads ``.Pi``.
        Returns ``{}`` for any other status.
        """
        sub = self.sub_yp if which == "yp" else self.sub_y
        constrs = self.constrs_yp if which == "yp" else self.constrs_y

        status = sub.Status
        if status not in (GRB.INFEASIBLE, GRB.OPTIMAL):
            return {}

        use_farkas = status == GRB.INFEASIBLE
        result: dict[str, dict[str, Any] | float] = {}

        for group_name, group_val in constrs.items():
            if isinstance(group_val, dict):
                pi_sub: dict[str, Any] = {}
                for key, constr in group_val.items():
                    try:
                        val = constr.FarkasDual if use_farkas else constr.Pi
                    except Exception:
                        val = 0.0
                    if abs(val) > TOL:
                        pi_sub[key] = val
                if pi_sub:
                    result[group_name] = pi_sub
            elif isinstance(group_val, gp.Constr):
                try:
                    val = group_val.FarkasDual if use_farkas else group_val.Pi
                except Exception:
                    val = 0.0
                if abs(val) > TOL:
                    result[group_name] = val

        return result

    # ------------------------------------------------------------------
    # Depth computation:  ẑ = π^T (b − B x̄) / (w^T π)
    # ------------------------------------------------------------------

    def _compute_ddma_depth(
        self,
        pi_dict: dict[str, dict[str, Any] | float],
        x_sol: dict[Arc, float],
        which: str,
        weights: dict[str, float],
        TOL: float = _DDMA_TOL,
    ) -> tuple[float, float]:
        """Compute depth ẑ = (π^T (b − B x̄)) / (w^T π) and the raw numerator.

        Returns
        -------
        (depth, numerator)
            depth     : float — normalised depth ẑ (used for convergence / π selection)
            numerator : float — unnormalised violation π^T (b − B x̄) (used for cut validity)

        The numerator π^T (b − B x̄) is the *unperturbed* RHS dot product
        (i.e. with z=0 in b − B x̄ − z w).  The denominator w^T π uses
        the absolute value |π_i| implicitly via the dual-cone sign rules:
        since γ duals are ≥ 0 and δ duals are ≤ 0, |π| = π * sign(w).

        To keep this simple and numerically robust we compute both the
        numerator and denominator directly.
        """
        constrs = self.constrs_yp if which == "yp" else self.constrs_y
        suffix = "_p" if which == "yp" else ""

        numerator = 0.0
        denominator = 0.0

        # Helper: contribution of each group to π^T b(x̄)
        def _rhs(group: str, key: tuple[Any, ...] | None = None) -> float:
            """Compute b_i(x̄) for a given constraint group and key."""
            if which == "yp":
                if group == f"alpha{suffix}":
                    i, j = key  # type: ignore[misc]
                    return 1.0 - x_sol.get((i, j), 0.0)
                if group == f"beta{suffix}":
                    i, j = key  # type: ignore[misc]
                    return x_sol.get((j, i), 0.0) - x_sol.get((i, j), 0.0)
                if group == f"gamma{suffix}":
                    i, j = key  # type: ignore[misc]
                    return x_sol.get((j, i), 0.0)
                if group == f"delta{suffix}":
                    i, j = key  # type: ignore[misc]
                    return 1.0 - x_sol.get((i, j), 0.0)
                if group == f"global{suffix}":
                    g = constrs.get(group)
                    return float(g.RHS) if isinstance(g, gp.Constr) else 0.0
                if group == f"r1{suffix}":
                    return float(self.convex_hull_area)
                if group == f"r2{suffix}":
                    return 1.0
                if group == f"r3{suffix}":
                    i, j, k, s = key  # type: ignore[misc]
                    return 1.0 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0)
            else:
                if group == "alpha":
                    i, j = key  # type: ignore[misc]
                    return x_sol.get((i, j), 0.0)
                if group == "beta":
                    i, j = key  # type: ignore[misc]
                    return x_sol.get((i, j), 0.0) - x_sol.get((j, i), 0.0)
                if group == "gamma":
                    i, j = key  # type: ignore[misc]
                    return x_sol.get((i, j), 0.0)
                if group == "delta":
                    i, j = key  # type: ignore[misc]
                    return 1.0 - x_sol.get((j, i), 0.0)
                if group == "global":
                    g = constrs.get(group)
                    return float(g.RHS) if isinstance(g, gp.Constr) else 0.0
                if group == "r1":
                    return float(self.convex_hull_area)
                if group == "r2":
                    return 1.0
                if group == "r3":
                    i, j, k, s = key  # type: ignore[misc]
                    return 1.0 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0)
            return 0.0

        for group, pi_val in pi_dict.items():
            w = weights.get(group, 0.0)
            if isinstance(pi_val, dict):
                for key, piv in pi_val.items():
                    rhs_k = _rhs(group, key)  # type: ignore[arg-type]
                    numerator += piv * rhs_k
                    denominator += w * abs(piv)
            else:
                rhs_k = _rhs(group, None)
                numerator += pi_val * rhs_k
                denominator += w * abs(pi_val)

        if abs(denominator) < TOL:
            # All non-zero π components sit on weight-0 groups (e.g. global constraint).
            # Use the numerator as a proxy depth; DDMA cannot improve the cut further via
            # depth maximisation (w^T|π| = 0 for all active duals).
            return numerator, numerator
        return numerator / denominator, numerator

    # ------------------------------------------------------------------
    # Cut reconstruction from flat π dict
    # ------------------------------------------------------------------

    def _reconstruct_cut_from_pi_values(
        self,
        pi_dict: dict[str, dict[str, Any] | float],
        x_sol: dict[Arc, float],
        which: str,
        TOL: float = _DDMA_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]]:
        """Build Benders cut  π^T B x ≤ −π^T b_const  from raw π floats.

        Mirrors ``BendersCGSPMixin._reconstruct_cut_from_pi`` but operates
        on scalar dual values (from ``.Pi`` / ``.FarkasDual``) rather than
        Gurobi LP variables.

        Returns
        -------
        (cut_expr, cut_rhs, witness)
            cut_expr : gp.LinExpr  — linear in master variables x (and η)
            cut_rhs  : float       — all constants on RHS explicitly
            witness  : dict        — non-zero π components for logging
        """
        cut_expr = gp.LinExpr()
        cut_rhs = 0.0
        witness: dict[str, Any] = {}
        suffix = "_p" if which == "yp" else ""

        def _add(group: str, pi_val: dict[str, Any] | float) -> None:
            nonlocal cut_rhs
            if isinstance(pi_val, dict):
                if not pi_val:
                    return
                witness[group] = pi_val
                for key, piv in pi_val.items():
                    if abs(piv) <= TOL:
                        continue
                    if which == "yp":
                        if group == f"alpha{suffix}":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([-piv], [self.x[i, j]])  # type: ignore[index]
                            cut_rhs += -piv * 1.0
                        elif group == f"beta{suffix}":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([piv, -piv], [self.x[j, i], self.x[i, j]])  # type: ignore[index]
                        elif group == f"gamma{suffix}":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([piv], [self.x[j, i]])  # type: ignore[index]
                        elif group == f"delta{suffix}":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([-piv], [self.x[i, j]])  # type: ignore[index]
                            cut_rhs += -piv * 1.0
                        elif group == f"r2{suffix}":
                            cut_rhs += -piv * 1.0
                        elif group == f"r3{suffix}":
                            i, j, k, s = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([-piv, -piv], [self.x[i, j], self.x[s, k]])  # type: ignore[index]
                            cut_rhs += -piv * 1.0
                    else:
                        if group == "alpha":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([piv], [self.x[i, j]])  # type: ignore[index]
                        elif group == "beta":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([piv, -piv], [self.x[i, j], self.x[j, i]])  # type: ignore[index]
                        elif group == "gamma":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([piv], [self.x[i, j]])  # type: ignore[index]
                        elif group == "delta":
                            i, j = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([-piv], [self.x[j, i]])  # type: ignore[index]
                            cut_rhs += -piv * 1.0
                        elif group == "r2":
                            cut_rhs += -piv * 1.0
                        elif group == "r3":
                            i, j, k, s = key  # type: ignore[str-unpack]
                            cut_expr.addTerms([-piv, -piv], [self.x[j, i], self.x[k, s]])  # type: ignore[index]
                            cut_rhs += -piv * 1.0
            else:
                piv = pi_val
                if abs(piv) <= TOL:
                    return
                witness[group] = piv
                # Scalar groups: global and r1 are constant-RHS
                if group in (f"global{suffix}", "global"):
                    constrs = self.constrs_yp if which == "yp" else self.constrs_y
                    g = constrs.get(group)
                    rhs_val = float(g.RHS) if isinstance(g, gp.Constr) else 0.0
                    cut_rhs += -piv * rhs_val
                elif group in (f"r1{suffix}", "r1"):
                    cut_rhs += -piv * float(self.convex_hull_area)

        for group, pi_val in pi_dict.items():
            _add(group, pi_val)

        return cut_expr, cut_rhs, witness

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def get_ddma_cut_yp(
        self,
        x_sol: dict[Arc, float],
        max_iter: int = _DDMA_MAX_ITER,
        eps: float = _DDMA_EPS,
        TOL: float = _DDMA_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
        """Generate the DDMA deepest feasibility cut for Y'.

        Returns (cut_expr, cut_rhs, witness) or (None, None, {'aborted': …}).
        """
        return self._run_ddma(x_sol, which="yp", eta_sol=0.0, max_iter=max_iter, eps=eps, TOL=TOL)

    def get_ddma_cut_y(
        self,
        x_sol: dict[Arc, float],
        eta_sol: float = 0.0,
        max_iter: int = _DDMA_MAX_ITER,
        eps: float = _DDMA_EPS,
        TOL: float = _DDMA_TOL,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
        """Generate the DDMA deepest feasibility (or optimality) cut for Y.

        Returns (cut_expr, cut_rhs, witness) or (None, None, {'aborted': …}).
        """
        return self._run_ddma(x_sol, which="y", eta_sol=eta_sol, max_iter=max_iter, eps=eps, TOL=TOL)

    # ------------------------------------------------------------------
    # Core DDMA loop
    # ------------------------------------------------------------------

    def _run_ddma(
        self,
        x_sol: dict[Arc, float],
        which: str,
        eta_sol: float,
        max_iter: int,
        eps: float,
        TOL: float,
    ) -> tuple[gp.LinExpr, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
        """Execute Algorithm 3 (DDMA) for subproblem ``which``.

        Algorithm
        ---------
        z ← 0
        for it = 1 … max_iter:
            perturb RHS by −z·w         (on top of the standard x̄-based RHS)
            solve sub
            restore RHS to b(x̄)
            extract π (Farkas if INFEASIBLE, Pi if OPTIMAL)
            ẑ = depth(π, x̄, w)
            if ẑ − z < eps: return cut(π)
            z ← ẑ
        # budget exhausted: return the last π (still a valid cut — Remark 6)
        return cut(π_last) if π_last else (None, None, {'aborted': 'budget'})
        """
        sub = self.sub_yp if which == "yp" else self.sub_y
        weights = self._get_ddma_weights(which)

        # Ensure FarkasDual is available after INFEASIBLE solves
        sub.setParam("InfUnbdInfo", 1)

        z_abs = 0.0
        best_pi: dict[str, Any] | None = None
        best_depth = 0.0  # signed depth of the selected π
        best_numerator = 0.0  # signed numerator of the selected π
        best_abs_numerator = -float("inf")  # selection criterion (|numerator|)
        best_status: int | None = None  # GRB.INFEASIBLE or GRB.OPTIMAL of best π

        for it in range(max_iter):
            # Step 1 — perturb RHS by −z_abs·w (always subtract = always tighten).
            # Tightening Dy ≤ b(x̄) to Dy ≤ b(x̄) − z_abs·w preserves infeasibility
            # and guarantees that any Farkas dual from the tightened system is also
            # valid for the original: since π ≤ 0 and w ≥ 0,  π^T w ≤ 0,  so
            # π^T(b − z·w) < 0  →  π^T b < z·π^T w ≤ 0  →  π^T b < 0.
            # This is GPA §5 (Algorithm 3), Hosseini & Turner 2025.
            z_signed = z_abs
            if abs(z_signed) > TOL:
                self._apply_ddma_perturbation(z_signed, which, weights)
            sub_to_solve = sub
            sub_to_solve.update()
            sub_to_solve.optimize()

            # Restore RHS to the standard b(x̄) (undo perturbation)
            if abs(z_signed) > TOL:
                self._apply_ddma_perturbation(-z_signed, which, weights)  # reverse

            status = sub_to_solve.Status

            if status not in (GRB.INFEASIBLE, GRB.OPTIMAL):
                logger.debug("DDMA(%s) iter %d: unexpected status %d", which, it, status)
                break

            # Extract π
            pi_dict = self._extract_ddma_pi(which, TOL=TOL)
            if not pi_dict:
                logger.debug("DDMA(%s) iter %d: empty π — stopping", which, it)
                break

            # depth = normalised violation (signed; for convergence / π-selection)
            # numerator = π^T (b − B x̄), the raw (unnormalised) violation (signed)
            depth, numerator = self._compute_ddma_depth(pi_dict, x_sol, which, weights, TOL=TOL)

            # Selection criterion is status-aware:
            # - INFEASIBLE (FarkasDual): Gurobi can return either sign convention;
            #   select by |numerator|.  A negative-numerator ray is handled by the
            #   flip block at the end (−π is also a valid Farkas certificate).
            # - OPTIMAL (.Pi): only accept numerator > TOL.  Negating a dual basis
            #   solution is NOT valid — .Pi values do not form a cone, so −π is not
            #   dual feasible and would produce an invalid cut.
            if status == GRB.INFEASIBLE:
                accept = abs(numerator) > TOL and abs(numerator) > best_abs_numerator
            else:  # OPTIMAL
                accept = numerator > TOL and numerator > best_abs_numerator
            if accept:
                best_abs_numerator = abs(numerator)
                best_depth = depth
                best_numerator = numerator
                best_pi = pi_dict
                best_status = status

            # Check for convergence on absolute depth increment
            if abs(depth) - z_abs < eps:
                logger.debug(
                    "DDMA(%s) converged at iter %d: z_abs=%.4e, |depth|=%.4e, numerator=%.4e",
                    which,
                    it,
                    z_abs,
                    abs(depth),
                    numerator,
                )
                break

            z_abs = abs(depth)

        if best_pi is None:
            return None, None, {"aborted": "no_pi"}

        # Validity check: |π^T (b − B x̄)| > TOL.  The unnormalised quantity is what the
        # Farkas-mixin uses (TOL = 1e-6 from the callback).  Using the normalised depth
        # would be incorrect because Gurobi's FarkasDuals are not scaled to ‖π‖₁ = 1
        # (CGSP normalises explicitly via its LP).
        if best_abs_numerator <= TOL:
            return (
                None,
                None,
                {
                    "aborted": "no_violation",
                    "depth": best_depth,
                    "numerator": best_numerator,
                },
            )

        cut_expr, cut_rhs, witness = self._reconstruct_cut_from_pi_values(best_pi, x_sol, which, TOL=TOL)
        witness["ddma_depth"] = best_depth
        witness["ddma_numerator"] = best_numerator
        witness["ddma_iters"] = it + 1

        # If the best π came from an INFEASIBLE subproblem and numerator < 0, the
        # Farkas certificate has the negative Gurobi sign convention (−π is the ray).
        # The callback injects `cut_expr ≤ cut_rhs`, so we negate both sides to flip
        # the inequality: `−cut_expr ≤ −cut_rhs` ≡ `cut_expr ≥ cut_rhs`.
        #
        # This flip is valid ONLY for Farkas rays (INFEASIBLE status): rays form a
        # cone so −λ is also a valid Farkas certificate.  For OPTIMAL (.Pi) duals,
        # −π is not dual feasible in general, so we must NEVER flip when best came
        # from an OPTIMAL solve.
        if best_numerator < 0 and best_status == GRB.INFEASIBLE:
            cut_expr = -cut_expr
            cut_rhs = -cut_rhs
            witness["ddma_flipped"] = True

        return cut_expr, cut_rhs, witness
