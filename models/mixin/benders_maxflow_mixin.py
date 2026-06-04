# models/mixin/benders_maxflow_mixin.py
"""
BendersMaxFlowMixin — combinatorial Benders cut generation via max-flow / min-cut.

Implements FR-7 (Combinatorial-Benders cut strengthening) and is the entry point
for M-4 (combinatorial cuts checkpoint).

Design reference: plan-maxflow-benders.md §3, own-benders-derivation.md §3.2.2.

**Hito 1** (implemented):
  ``build_maxflow_structures()`` — static forbidden-arc constraints + auxiliary
  data structures.

**Hito 2** (implemented):
  ``check_feasibility_integer`` / ``check_feasibility_integer_yp`` — O(N) check
  for integer x̄; ``_triangle_not_blocked`` / ``_triangle_not_blocked_yp``.

**Hito 3** (implemented):
  ``get_maxflow_cut_y`` / ``get_maxflow_cut_yp`` — for integer x̄: combinatorial
  blocking cut (§2.4, degenerate-D case); for fractional x̄: bipartite max-flow
  cut via ``networkx``.

  ⚠ OQ-A: LP-validity of the bipartite max-flow cut for fractional x̄ is an open
  question (plan-maxflow-benders.md §5).  The fractional path is guarded by the
  ``_mf_use_bipartite_fractional`` flag (default False until OQ-A is resolved).
  When the flag is False and x̄ is fractional, the methods return (None, 0.0, None)
  and the caller falls back to the existing Farkas/Pi path.
"""
from __future__ import annotations

import collections
import logging
from typing import Any

import gurobipy as gp
import networkx as nx

from models.typing_oap import IndexArray, TrianglesAdjList

logger = logging.getLogger(__name__)

Arc = tuple[int, int]


class BendersMaxFlowMixin:
    """
    Mixin for combinatorial Benders feasibility checks via max-flow / min-cut.

    Replaces the LP-based Farkas certificate at MIPSOL with an O(N) combinatorial
    oracle (Hito 2) and, for fractional x̄, with a bipartite max-flow solve whose
    min-cut directly produces the Benders cut (Hito 3, OQ-A caveat).

    This mixin is mutually exclusive with ``use_deepest_cuts``, ``use_magnanti_wong``,
    and ``use_ddma``.  Enforcement lives in ``OAPBendersModel.build()``.

    MRO position: after BendersMagnantiWongMixin, before BendersFarkasMixin.
    """

    # ------------------------------------------------------------------ #
    # Type stubs — declared here so mypy can see the attributes that are  #
    # provided by other mixins / OAPBaseModel in the actual MRO.          #
    # ------------------------------------------------------------------ #
    model: gp.Model
    x: dict[Arc, gp.Var]
    N: int
    N_list: range
    CH: IndexArray
    triangles: IndexArray
    triangles_adj_list: TrianglesAdjList
    # Populated by build():
    use_maxflow: bool
    # Populated by build_maxflow_structures():
    _mf_forbidden_arcs: set[Arc]
    """Set of arcs (i,j) with empty triangles_adj_list[i][j]; x[i,j] fixed to 0."""
    _mf_arc_to_triangles: dict[Arc, list[int]]
    """Snapshot of adj_list for arcs in self.x, used by Hito-2 O(N) check."""
    _mf_triangle_to_arcs: dict[int, list[Arc]]
    """Reverse mapping: triangle index → arcs (i,j) with t ∈ adj_list[i][j]."""
    _mf_use_bipartite_fractional: bool
    """
    When True, ``get_maxflow_cut_y/yp`` runs the bipartite max-flow for fractional
    x̄ at MIPNODE / LP-relaxation callbacks.  Defaults to False (OQ-A unresolved).
    Set to True explicitly only after mathematical-formalist validates LP-validity.
    """

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_maxflow_structures(self) -> None:
        """Precompute max-flow auxiliary structures and add static forbidden-arc constraints.

        Must be called from ``OAPBendersModel.build()`` after both master and
        subproblems are built (so that ``self.x`` and ``self.triangles_adj_list``
        are populated).

        **Static forbidden-arc constraints (Hito 1)**

        For every arc (i,j) ∈ self.x such that ``triangles_adj_list[i][j]`` is
        empty, the Y subproblem can never be satisfied when x̄[i,j]=1 (the
        gamma constraint becomes 0 ≥ 1, infeasible).  Adding x[i,j] == 0 to
        the master avoids generating such infeasible solutions altogether.

        The corresponding Benders cut is x[i,j] ≤ 0, which is the degenerate
        case D = {(i,j)}, B = ∅ of the bipartite min-cut (plan §2.4).

        **Reverse lookup for O(N) oracle (Hito 2)**

        Builds ``_mf_arc_to_triangles`` (arc → triangle list) and
        ``_mf_triangle_to_arcs`` (triangle → arc list) for fast feasibility
        checks without re-accessing triangles_adj_list inside the callback.
        """
        forbidden_count = 0
        self._mf_forbidden_arcs = set()
        self._mf_arc_to_triangles = {}
        self._mf_triangle_to_arcs = {}
        self._mf_use_bipartite_fractional = False  # OQ-A: default off

        for arc in self.x:
            i, j = arc
            triangles: list[int] = list(self.triangles_adj_list[i][j])
            self._mf_arc_to_triangles[arc] = triangles

            if not triangles:
                # Degenerate cut: x[i,j] must be 0 (Y infeasible for any x̄[i,j]=1)
                self._mf_forbidden_arcs.add(arc)
                self.model.addConstr(
                    self.x[arc] == 0,
                    name=f"mf_forbidden_{i}_{j}",
                )
                forbidden_count += 1
            else:
                # Build reverse mapping: triangle → arcs that cover it
                for t in triangles:
                    if t not in self._mf_triangle_to_arcs:
                        self._mf_triangle_to_arcs[t] = []
                    self._mf_triangle_to_arcs[t].append(arc)

        self.model.update()
        logger.info(
            "build_maxflow_structures: %d forbidden arcs fixed to 0 out of %d total arcs.",
            forbidden_count,
            len(self.x),
        )
        if forbidden_count:
            logger.debug("Forbidden arcs: %s", sorted(self._mf_forbidden_arcs))

    def check_feasibility_integer(
        self,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> tuple[bool, list[Arc]]:
        """O(N) feasibility oracle for Y(x̄) when x̄ ∈ {0,1}.

        For each active arc (i,j) (x̄[i,j] ≈ 1), checks whether at least one
        triangle in ``_mf_arc_to_triangles[(i,j)]`` is *not blocked* (i.e., has
        ``_triangle_not_blocked(t, x̄) == True``).

        Y(x̄) is **infeasible** when some active arc (i,j) has ALL its triangles
        blocked (no valid y-assignment satisfies the gamma constraint).

        Parameters
        ----------
        x_sol : dict[Arc, float]
            Current Gurobi MIPSOL solution (all arcs, including inactive ones).
        TOL : float, optional
            Integrality tolerance.  Default 1e-6.

        Returns
        -------
        feasible : bool
        blocking_arcs : list[Arc]
            Non-empty only when infeasible: the first arc for which no unblocked
            triangle exists, used to drive cut generation.
        """
        for arc in self.x:
            if x_sol.get(arc, 0.0) < 1.0 - TOL:
                continue
            # Forbidden arcs are already fixed to 0 by static constraints;
            # if one appears active (float noise), treat as infeasible so a
            # lazy cut is still generated (belt-and-suspenders).
            if arc in self._mf_forbidden_arcs:
                return False, [arc]
            triangles = self._mf_arc_to_triangles.get(arc, [])
            if not triangles:
                return False, [arc]
            if not any(self._triangle_not_blocked(t, x_sol, TOL) for t in triangles):
                return False, [arc]
        return True, []

    def check_feasibility_integer_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> tuple[bool, list[Arc]]:
        """O(N) feasibility oracle for Y'(x̄) when x̄ ∈ {0,1}.

        For Y' the demand on the group ``adj_list[i][j]`` comes from x̄[j,i]=1
        (gamma_p RHS = x̄[j,i]), and the blocking condition uses the *forward*
        arc: triangle t in adj_list[i][j] is blocked when x̄[i',j'] ≈ 1 for
        some (i',j') with t ∈ adj_list[i'][j'] (delta_p forces y'_t = 0).

        Parameters
        ----------
        x_sol : dict[Arc, float]
        TOL : float, optional

        Returns
        -------
        feasible : bool
        blocking_arcs : list[Arc]
        """
        for arc in self.x:
            i, j = arc
            # Demand for adj_list[i][j] comes from reverse arc (j,i) being active.
            if x_sol.get((j, i), 0.0) < 1.0 - TOL:
                continue
            if arc in self._mf_forbidden_arcs:
                return False, [arc]
            triangles = self._mf_arc_to_triangles.get(arc, [])
            if not triangles:
                return False, [arc]
            if not any(self._triangle_not_blocked_yp(t, x_sol, TOL) for t in triangles):
                return False, [arc]
        return True, []

    def get_maxflow_cut_y(
        self,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> tuple[gp.LinExpr | None, float, dict[str, Any] | None]:
        """Benders feasibility cut for Y subproblem.

        **Integer x̄ (MIPSOL)**: uses a combinatorial blocking cut (plan §2.4,
        no networkx dependency, provably valid for {0,1} x̄).

        **Fractional x̄ (MIPNODE / LP-relaxation)**: uses the bipartite
        max-flow / min-cut network from plan §2.3.  Only active when
        ``_mf_use_bipartite_fractional = True`` (default: False, OQ-A pending).
        Returns (None, 0.0, None) otherwise, triggering LP-subproblem fallback.

        Returns
        -------
        cut_expr : gp.LinExpr | None
        cut_rhs : float
        witness : dict | None
            Diagnostic info (infeasible arc, blockers); None when no cut.
        """
        is_integer = all(abs(v - round(v)) < TOL for v in x_sol.values())
        if is_integer:
            return self._blocking_cut(x_sol, which="y", TOL=TOL)
        if getattr(self, "_mf_use_bipartite_fractional", False):
            return self._bipartite_cut(x_sol, which="y", TOL=TOL)
        return None, 0.0, None

    def get_maxflow_cut_yp(
        self,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> tuple[gp.LinExpr | None, float, dict[str, Any] | None]:
        """Benders feasibility cut for Y' subproblem.  Mirror of ``get_maxflow_cut_y``."""
        is_integer = all(abs(v - round(v)) < TOL for v in x_sol.values())
        if is_integer:
            return self._blocking_cut(x_sol, which="yp", TOL=TOL)
        if getattr(self, "_mf_use_bipartite_fractional", False):
            return self._bipartite_cut(x_sol, which="yp", TOL=TOL)
        return None, 0.0, None

    # ------------------------------------------------------------------ #
    # Hito-2 helpers                                                      #
    # ------------------------------------------------------------------ #

    def _triangle_not_blocked(
        self,
        t: int,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> bool:
        """True if triangle t can have y_t = 1 under Y(x̄) (not blocked by delta).

        Triangle t is *blocked in Y* if there exists an arc (i',j') such that
        t ∈ adj_list[i'][j'] AND x̄[j',i'] ≈ 1.  Such a reverse-active arc
        makes delta(i',j') RHS = 1 − x̄[j',i'] = 0, forcing y_t = 0.

        Uses ``_mf_triangle_to_arcs`` for O(deg(t)) cost per call.
        """
        for ip, jp in self._mf_triangle_to_arcs.get(t, []):
            if x_sol.get((jp, ip), 0.0) >= 1.0 - TOL:
                return False
        return True

    def _triangle_not_blocked_yp(
        self,
        t: int,
        x_sol: dict[Arc, float],
        TOL: float = 1e-6,
    ) -> bool:
        """True if triangle t can have y'_t = 1 under Y'(x̄) (not blocked by delta_p).

        Triangle t is *blocked in Y'* if there exists an arc (i',j') such that
        t ∈ adj_list[i'][j'] AND x̄[i',j'] ≈ 1.  Such a forward-active arc
        makes delta_p(i',j') RHS = 1 − x̄[i',j'] = 0, forcing y'_t = 0.
        """
        for ip, jp in self._mf_triangle_to_arcs.get(t, []):
            if x_sol.get((ip, jp), 0.0) >= 1.0 - TOL:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Hito-3 cut generators                                               #
    # ------------------------------------------------------------------ #

    def _blocking_cut(
        self,
        x_sol: dict[Arc, float],
        which: str = "y",
        TOL: float = 1e-6,
    ) -> tuple[gp.LinExpr | None, float, dict[str, Any] | None]:
        """Combinatorial blocking cut for the first infeasible arc in integer x̄.

        For infeasible arc (i,j) with all its triangles blocked, collects the
        unique set B of *blocking variables* (one active-blocker arc per
        triangle) and emits the cut:

            x[demand_arc] + Σ_{b ∈ B} x[b] ≤ |B|

        where ``demand_arc`` is (i,j) for Y and (j,i) for Y'.

        **Validity** (integer case):
        At x̄ the LHS = 1 + |B| > |B| — violated ✓.
        At any master-feasible integer x: if demand_arc is active, at least one
        blocker in B must be inactive (otherwise the same blocking configuration
        recurs and Y / Y' is infeasible), so Σ x[b] ≤ |B| − 1 and the sum
        ≤ |B| ✓.

        Parameters
        ----------
        x_sol : dict[Arc, float]
        which : str
            "y" for Y subproblem; "yp" for Y'.
        TOL : float

        Returns
        -------
        cut_expr : gp.LinExpr | None
        cut_rhs : float
        witness : dict | None
        """
        is_yp = which == "yp"
        not_blocked = self._triangle_not_blocked_yp if is_yp else self._triangle_not_blocked

        for arc in self.x:
            i, j = arc
            # Determine which variable signals "demand" for this arc group.
            if is_yp:
                demand_arc: Arc = (j, i)
                if x_sol.get(demand_arc, 0.0) < 1.0 - TOL:
                    continue
            else:
                demand_arc = arc
                if x_sol.get(demand_arc, 0.0) < 1.0 - TOL:
                    continue

            if arc in self._mf_forbidden_arcs:
                # Degenerate: static constraint already covers this — emit
                # the cut anyway as a lazy guard (belt-and-suspenders).
                cut_expr = gp.LinExpr()
                cut_expr.addTerms(1.0, self.x[demand_arc])
                return cut_expr, 0.0, {"arc": demand_arc, "blockers": []}

            triangles = self._mf_arc_to_triangles.get(arc, [])
            if not triangles:
                cut_expr = gp.LinExpr()
                cut_expr.addTerms(1.0, self.x[demand_arc])
                return cut_expr, 0.0, {"arc": demand_arc, "blockers": []}

            # Check if all triangles are blocked.
            if any(not_blocked(t, x_sol, TOL) for t in triangles):
                continue  # this arc is satisfiable; not the blocking witness

            # Collect one representative blocker per triangle.
            blockers: set[Arc] = set()
            for t in triangles:
                for ip, jp in self._mf_triangle_to_arcs.get(t, []):
                    if is_yp:
                        blocker = (ip, jp)  # forward arc blocks in Y'
                    else:
                        blocker = (jp, ip)  # reverse arc blocks in Y
                    if x_sol.get(blocker, 0.0) >= 1.0 - TOL and blocker in self.x:
                        blockers.add(blocker)
                        break  # one per triangle is enough

            if not blockers:
                # No explicit blocker found (shouldn't happen after oracle confirms
                # infeasibility), fall back to forbidden-arc cut.
                cut_expr = gp.LinExpr()
                cut_expr.addTerms(1.0, self.x[demand_arc])
                return cut_expr, 0.0, {"arc": demand_arc, "blockers": []}

            # Build: x[demand_arc] + Σ_B x[b] ≤ |B|
            cut_expr = gp.LinExpr()
            cut_expr.addTerms(1.0, self.x[demand_arc])
            for b in blockers:
                cut_expr.addTerms(1.0, self.x[b])
            cut_rhs = float(len(blockers))

            logger.debug(
                "_blocking_cut (%s): demand=%s, |B|=%d, cut=%.0f <= %.0f at x̄",
                which,
                demand_arc,
                len(blockers),
                1.0 + len(blockers),
                cut_rhs,
            )
            return cut_expr, cut_rhs, {"arc": demand_arc, "blockers": sorted(blockers)}

        return None, 0.0, None

    def _bipartite_cut(
        self,
        x_sol: dict[Arc, float],
        which: str = "y",
        TOL: float = 1e-6,
    ) -> tuple[gp.LinExpr | None, float, dict[str, Any] | None]:
        """Bipartite max-flow / min-cut Benders cut for fractional x̄ (plan §2.3–2.4).

        ⚠ OQ-A caveat: LP-validity for fractional x̄ is an open question.
        Only called when ``_mf_use_bipartite_fractional = True``.

        Builds the bipartite network G_bp, solves max-flow S→T, and translates
        the min-cut partition (S_side, T_side) into:

            Σ_{(i,j) ∈ D} x[i,j]  ≤  Σ_{t ∈ B} Σ_{adj active arcs} x[k,l]

        where D = arc nodes in T_side, B = triangle nodes in S_side adjacent to D.

        Returns (None, 0.0, None) when the flow saturates (Y feasible).
        """
        G = self._build_bipartite_network(x_sol, which=which, TOL=TOL)

        total_demand = sum(
            x_sol.get(arc, 0.0)
            for arc in self.x
            if x_sol.get(arc, 0.0) > TOL and arc not in self._mf_forbidden_arcs
        )
        if total_demand <= TOL:
            return None, 0.0, None

        flow_value: float
        flow_dict: dict[Any, dict[Any, float]]
        flow_value, flow_dict = nx.maximum_flow(G, "S", "T", capacity="capacity")

        if flow_value >= total_demand - TOL:
            return None, 0.0, None

        S_side, T_side = self._min_cut_sides(G, flow_dict, TOL=TOL)
        return self._cut_from_partition(S_side, T_side, x_sol, which=which, TOL=TOL)

    def _build_bipartite_network(
        self,
        x_sol: dict[Arc, float],
        which: str = "y",
        TOL: float = 1e-6,
    ) -> nx.DiGraph:
        """Construct the bipartite network G_bp for Y or Y' (plan §2.3).

        Nodes
        -----
        "S"            — source
        ("arc", arc)   — one node per active arc (x̄[arc] > TOL, not forbidden)
        ("tri", t)     — one node per triangle reachable from active arcs
        "T"            — sink

        Edges
        -----
        S  → a_{ij}   cap = x̄[i,j]
        a_{ij} → t    cap = 1  (for each t ∈ adj_list[i][j])
        t → T         cap = t_capacity(t)
                        Y : max(0, 1 − Σ_{(i',j'): t adj, x̄[j',i']>0} x̄[j',i'])
                        Y': max(0, 1 − Σ_{(i',j'): t adj, x̄[i',j']>0} x̄[i',j'])

        The t→T capacity is the "unblocked" capacity of triangle t: the maximum
        y_t value allowed by the tightest delta (Y) or delta_p (Y') constraint.
        For integer x̄ this is 0 (fully blocked) or 1 (unblocked).
        """
        is_yp = which == "yp"
        G: nx.DiGraph = nx.DiGraph()
        G.add_node("S")
        G.add_node("T")

        # Precompute t → total blocking load for t→T capacity.
        # For Y:  blocking_load[t] = Σ_{(i',j'): t adj} x̄[j',i']
        # For Y': blocking_load[t] = Σ_{(i',j'): t adj} x̄[i',j']
        blocking_load: dict[int, float] = {}
        for arc in self.x:
            ip, jp = arc
            blocker_val = x_sol.get((jp, ip) if not is_yp else (ip, jp), 0.0)
            if blocker_val > TOL:
                for t in self._mf_arc_to_triangles.get(arc, []):
                    blocking_load[t] = blocking_load.get(t, 0.0) + blocker_val

        # Determine which arcs create "demand" in each subproblem.
        # Y:  demand_val = x̄[i,j];  arc group = adj_list[i][j]
        # Y': demand_val = x̄[j,i];  arc group = adj_list[i][j]
        for arc in self.x:
            i, j = arc
            demand_val = x_sol.get((j, i) if is_yp else arc, 0.0)
            if demand_val <= TOL or arc in self._mf_forbidden_arcs:
                continue

            arc_node: tuple[str, Arc] = ("arc", arc)
            G.add_edge("S", arc_node, capacity=demand_val)

            for t in self._mf_arc_to_triangles.get(arc, []):
                tri_node: tuple[str, int] = ("tri", t)
                G.add_edge(arc_node, tri_node, capacity=1.0)

                # Add t→T edge with unblocked capacity if not yet added.
                if not G.has_edge(tri_node, "T"):
                    t_cap = max(0.0, 1.0 - blocking_load.get(t, 0.0))
                    if t_cap > TOL:
                        G.add_edge(tri_node, "T", capacity=t_cap)

        return G

    def _min_cut_sides(
        self,
        G: nx.DiGraph,
        flow_dict: dict[Any, dict[Any, float]],
        TOL: float = 1e-6,
    ) -> tuple[set[Any], set[Any]]:
        """BFS on the residual graph from S; return (S_reachable, T_reachable)."""
        visited: set[Any] = {"S"}
        queue: collections.deque[Any] = collections.deque(["S"])

        while queue:
            u = queue.popleft()
            for v in G.successors(u):
                if v not in visited:
                    cap = G[u][v].get("capacity", 0.0)
                    flw = flow_dict.get(u, {}).get(v, 0.0)
                    if cap - flw > TOL:
                        visited.add(v)
                        queue.append(v)
            for v in G.predecessors(u):
                if v not in visited:
                    flw = flow_dict.get(v, {}).get(u, 0.0)
                    if flw > TOL:
                        visited.add(v)
                        queue.append(v)

        all_nodes: set[Any] = set(G.nodes)
        return visited, all_nodes - visited

    def _cut_from_partition(
        self,
        S_side: set[Any],
        T_side: set[Any],
        x_sol: dict[Arc, float],
        which: str = "y",
        TOL: float = 1e-6,
    ) -> tuple[gp.LinExpr | None, float, dict[str, Any] | None]:
        """Translate the min-cut partition into a Benders feasibility cut (plan §2.4).

        Cut formula:
            Σ_{(i,j) ∈ D} x[i,j]  ≤  Σ_{t ∈ B} Σ_{(k,l) adj to t, active} x[k,l]

        D = arc nodes in T_side (unmet demand).
        B = triangle nodes in S_side adjacent to at least one arc in D.

        The cut is violated at x̄ iff LHS > RHS (infeasible subproblem).
        """
        D: list[Arc] = [node[1] for node in T_side if isinstance(node, tuple) and node[0] == "arc"]
        if not D:
            return None, 0.0, None

        D_set: set[Arc] = set(D)

        # B = S_side triangle nodes adjacent to some arc in D.
        B: list[int] = []
        for node in S_side:
            if isinstance(node, tuple) and node[0] == "tri":
                t = node[1]
                for adj_arc in self._mf_triangle_to_arcs.get(t, []):
                    if adj_arc in D_set:
                        B.append(t)
                        break

        # LHS: Σ_{D} x[i,j]
        lhs = gp.LinExpr()
        for arc in D:
            lhs.addTerms(1.0, self.x[arc])

        # RHS: Σ_{B} Σ_{adj active arcs to t} x[k,l]
        rhs_expr = gp.LinExpr()
        rhs_arcs_seen: set[Arc] = set()
        for t in B:
            for adj_arc in self._mf_triangle_to_arcs.get(t, []):
                if adj_arc not in rhs_arcs_seen and x_sol.get(adj_arc, 0.0) > TOL:
                    rhs_arcs_seen.add(adj_arc)
                    rhs_expr.addTerms(1.0, self.x[adj_arc])

        # cut_expr = LHS − RHS ≤ 0
        cut_expr = lhs - rhs_expr
        cut_rhs = 0.0

        # Verify cut is violated at x̄ (sanity check before injecting).
        lhs_val = sum(x_sol.get(a, 0.0) for a in D)
        rhs_val = sum(x_sol.get(a, 0.0) for a in rhs_arcs_seen)
        if lhs_val - rhs_val <= TOL:
            logger.debug(
                "_cut_from_partition (%s): cut not violated at x̄ (LHS=%.4f, RHS=%.4f) — skipping.",
                which,
                lhs_val,
                rhs_val,
            )
            return None, 0.0, None

        return cut_expr, cut_rhs, {"D": list(D), "B": B, "lhs_val": lhs_val, "rhs_val": rhs_val}
