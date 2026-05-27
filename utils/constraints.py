"""Gurobi constraint injection helpers for OAP.

All functions in this module operate on a ``gp.Model`` instance and inject
geometric or combinatorial constraints directly into it.  They depend on
``utils.geometry`` for predicates but have no dependency on the logging or
visualisation submodules.
"""

from __future__ import annotations

from typing import NamedTuple

import gurobipy as gp
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from utils.geometry import Arc, point_in_triangle, segments_intersect

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArcConstraintMap = dict[Arc, gp.Constr]


# ---------------------------------------------------------------------------
# Half-plane constraints
# ---------------------------------------------------------------------------


def restricciones_semiplano(
    model: gp.Model,
    points: NDArray[np.int64],
    CH: NDArray[np.int64],
) -> tuple[gp.Model, list[gp.Constr]]:
    """Add empty-half-plane constraints to *model* for each interior→hull arc.

    For every interior node *i* and hull node *j* such that no other interior
    node lies strictly to the left of the directed line i→j, the arc (i, j)
    is forced to be followed by the next CCW hull arc (j, j_next).
    """
    A_pp = [i for i in range(len(points)) if i not in CH]
    constraints: list[gp.Constr] = []

    for i in A_pp:
        for j in CH:
            if (i, j) not in model._x:
                continue

            left_clear = True
            for k in A_pp:
                if k == i:
                    continue
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                if D > 0:
                    left_clear = False
                    break

            if left_clear:
                idx_j = int(np.where(CH == j)[0][0])
                j_next = int(CH[(idx_j + 1) % len(CH)])
                if (j, j_next) in model._x:
                    constraints.append(
                        model.addConstr(
                            model._x[i, j] <= model._x[j, j_next],
                            name=f"semiplano_{i}_{j}",
                        )
                    )

    return model, constraints


def restricciones_semiplanoV2(
    model: gp.Model,
    points: NDArray[np.int64],
    CH: NDArray[np.int64],
) -> tuple[gp.Model, list[gp.Constr]]:
    """Extended half-plane constraints that handle interior-point pockets.

    Variant of :func:`restricciones_semiplano` that distinguishes two cases:

    * **Empty pocket** – no interior node to the left: chain along hull arcs
      as long as they stay on the left side.
    * **Non-empty pocket** – at least one interior node to the left: the arc
      (i, j) is bounded by a sum of escape routes.
    """
    A_pp = [i for i in range(len(points)) if i not in CH]
    constraints: list[gp.Constr] = []

    for i in A_pp:
        for j in CH:
            if (i, j) not in model._x:
                continue

            S_left: list[int] = []
            for k in A_pp:
                if k == i:
                    continue
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                if D_k > 0:
                    S_left.append(k)

            idx_j = int(np.where(CH == j)[0][0])
            j_next = int(CH[(idx_j + 1) % len(CH)])

            if len(S_left) == 0:
                # Chain along hull while nodes remain on the left
                nodo_actual = j
                for step in range(1, len(CH)):
                    idx_sig = (idx_j + step) % len(CH)
                    nodo_sig = int(CH[idx_sig])
                    x_i, y_i = points[i]
                    x_j, y_j = points[j]
                    x_sig, y_sig = points[nodo_sig]
                    D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                    if D_sig > 0:
                        if (nodo_actual, nodo_sig) in model._x:
                            constraints.append(
                                model.addConstr(
                                    model._x[i, j] <= model._x[nodo_actual, nodo_sig],
                                    name=(f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual}_{nodo_sig}"),
                                )
                            )
                        nodo_actual = nodo_sig
                    else:
                        break
            else:
                # Pocket: escape via hull OR via trapped interior nodes
                expr_escape = gp.LinExpr()
                if (j, j_next) in model._x:
                    expr_escape.addTerms(1.0, model._x[j, j_next])
                for k in S_left:
                    if (j, k) in model._x:
                        expr_escape.addTerms(1.0, model._x[j, k])
                constraints.append(
                    model.addConstr(
                        model._x[i, j] <= expr_escape,
                        name=f"bolsillo_{i}_{j}_soporta_{len(S_left)}_puntos",
                    )
                )

    return model, constraints


def restricciones_semiplano_chain(
    model: gp.Model,
    points: NDArray[np.int64],
    CH: NDArray[np.int64],
) -> tuple[gp.Model, list[gp.Constr]]:
    """Half-plane chain constraints (dedicated variant of V2 empty-pocket logic).

    For every interior→hull arc (i, j) with an empty left half-plane, chains
    the forcing constraint along consecutive CCW hull arcs.
    """
    A_pp = [i for i in range(len(points)) if i not in CH]
    constraints: list[gp.Constr] = []

    for i in A_pp:
        for j in CH:
            if (i, j) not in model._x:
                continue

            left_clear = True
            for k in A_pp:
                if k == i:
                    continue
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                if D > 0:
                    left_clear = False
                    break

            if left_clear:
                idx_j = int(np.where(CH == j)[0][0])
                nodo_actual = j
                for step in range(1, len(CH)):
                    idx_sig = (idx_j + step) % len(CH)
                    nodo_sig = int(CH[idx_sig])
                    x_i, y_i = points[i]
                    x_j, y_j = points[j]
                    x_sig, y_sig = points[nodo_sig]
                    D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                    if D_sig > 0:
                        if (nodo_actual, nodo_sig) in model._x:
                            constraints.append(
                                model.addConstr(
                                    model._x[i, j] <= model._x[nodo_actual, nodo_sig],
                                    name=(f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual}_{nodo_sig}"),
                                )
                            )
                        nodo_actual = nodo_sig
                    else:
                        break

    return model, constraints


# ---------------------------------------------------------------------------
# Cutting-plane injectors
# ---------------------------------------------------------------------------


def inyectar_cortes_knapsack_locales(
    model: gp.Model,
    points: NDArray[np.int64],
    pesos_obj: dict[tuple[int, int], float],
) -> gp.Model:
    """Inject per-node knapsack cuts bounding fractional objective contribution."""
    num_nodos = len(points)
    cuts_added = 0

    for i in range(num_nodos):
        max_benefit = 0.0
        for j1 in range(num_nodos):
            if j1 == i or (i, j1) not in model._x:
                continue
            for j2 in range(j1 + 1, num_nodos):
                if j2 == i or (i, j2) not in model._x:
                    continue
                legal = True
                for k in range(num_nodos):
                    if k in (i, j1, j2):
                        continue
                    if point_in_triangle(points[k], points[j1], points[i], points[j2]):
                        legal = False
                        break
                if legal:
                    benefit = pesos_obj[i, j1] + pesos_obj[i, j2]
                    if benefit > max_benefit:
                        max_benefit = benefit

        expr = gp.LinExpr()
        for j in range(num_nodos):
            if j != i and (i, j) in model._x:
                expr.addTerms(pesos_obj[i, j], model._x[i, j])
        if expr.size() > 0:
            model.addConstr(expr <= max_benefit, name=f"knapsack_local_nodo_{i}")
            cuts_added += 1

    print(f"Inyectados {cuts_added} Cortes Knapsack Locales.")
    return model


def inyectar_cliques_de_cruce(
    model: gp.Model,
    points: NDArray[np.int64],
) -> gp.Model:
    """Inject clique-of-crossing-edges cuts (at most one arc in a crossing clique)."""
    print("Construyendo grafo de intersecciones para Cliques...")

    aristas: list[tuple[int, int]] = [
        (i, j) for i in range(len(points)) for j in range(len(points)) if (i, j) in model._x and i < j
    ]

    G_cruces: nx.Graph = nx.Graph()
    G_cruces.add_nodes_from(aristas)
    for idx, e1 in enumerate(aristas):
        for e2 in aristas[idx + 1 :]:
            if e1[0] in e2 or e1[1] in e2:
                continue
            p1, p2 = points[e1[0]], points[e1[1]]
            p3, p4 = points[e2[0]], points[e2[1]]
            if segments_intersect(p1, p2, p3, p4):
                G_cruces.add_edge(e1, e2)

    cuts_added = 0
    for clique in nx.find_cliques(G_cruces):
        if len(clique) < 3:
            continue
        expr = gp.LinExpr()
        for e in clique:
            if e in model._x:
                expr.addTerms(1.0, model._x[e[0], e[1]])
            if (e[1], e[0]) in model._x:
                expr.addTerms(1.0, model._x[e[1], e[0]])
        model.addConstr(expr <= 1, name=f"clique_cruce_{cuts_added}")
        cuts_added += 1

    print(f"Inyectados {cuts_added} Cortes de Clique de Cruces.")
    return model


def aplicar_semiplanos_por_capas(
    model: gp.Model,
    points: NDArray[np.int64],
    capas_onion: list[list[int]],
) -> tuple[gp.Model, list[gp.Constr]]:
    """Apply half-plane chain constraints iteratively across onion-peeling layers.

    *capas_onion[0]* is the outermost convex layer (L1), *capas_onion[1]* is
    the next layer, and so on.  Each layer must be ordered CCW.
    """
    constraints: list[gp.Constr] = []

    for k in range(len(capas_onion) - 1):
        capa_exterior = capas_onion[k]
        capa_interior = capas_onion[k + 1]

        puntos_profundos: list[int] = []
        for c in range(k + 1, len(capas_onion)):
            puntos_profundos.extend(capas_onion[c])

        for i in capa_interior:
            for j in capa_exterior:
                if (i, j) not in model._x:
                    continue

                left_clear = True
                for pk in puntos_profundos:
                    if pk == i:
                        continue
                    x_i, y_i = points[i]
                    x_j, y_j = points[j]
                    x_k, y_k = points[pk]
                    D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                    if D > 0:
                        left_clear = False
                        break

                if left_clear:
                    idx_j = list(capa_exterior).index(j)
                    nodo_actual = j
                    for step in range(1, len(capa_exterior)):
                        idx_sig = (idx_j + step) % len(capa_exterior)
                        nodo_sig = capa_exterior[idx_sig]
                        x_i, y_i = points[i]
                        x_j, y_j = points[j]
                        x_sig, y_sig = points[nodo_sig]
                        D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                        if D_sig > 0:
                            if (nodo_actual, nodo_sig) in model._x:
                                constraints.append(
                                    model.addConstr(
                                        model._x[i, j] <= model._x[nodo_actual, nodo_sig],
                                        name=(f"semiplano_L{k}_L{k + 1}_{i}_{j}_fuerza_{nodo_actual}_{nodo_sig}"),
                                    )
                                )
                            nodo_actual = nodo_sig
                        else:
                            break

    return model, constraints


# ---------------------------------------------------------------------------
# T_k (lifted cycle) inequality separation
# ---------------------------------------------------------------------------

_MAX_TK_CUTS: int = 10
"""Maximum T_k cuts injected per callback invocation (hardcoded constant)."""


class TkCut(NamedTuple):
    """A violated T_k (lifted cycle) inequality.

    Encodes:  x(S, S)  +  x_{pw}  +  x_{wq}  +  x_{pq}  ≤  |S|
    with  w ∈ S,  p ∉ S,  q ∉ S.
    """

    S: frozenset[int]
    p: int
    w: int
    q: int
    violation: float


def _build_tk_network(
    x_sol: dict[Arc, float],
    d: dict[int, float],
    N: int,
    p: int,
    w: int,
    q: int,
    INF: float,
) -> nx.DiGraph:
    """Build the directed min-cut network for T_k separation (fixed p, w, q).

    N + 2 nodes: 0 … N-1 are problem nodes; N = source s; N+1 = sink t.

    Capacity assignment (max-weight-closure → min-cut transformation):

    - s → w  :  INF   (force w on s-side / in S)
    - p → t  :  INF   (force p on t-side / not in S)
    - q → t  :  INF   (force q on t-side / not in S)
    - For each optional node i ∉ {p, q, w}:
        d_i  > 1  →  s → i  cap  (d_i − 1)   [positive profit]
        d_i  ≤ 1  →  i → t  cap  (1 − d_i)   [penalty]
    - For every LP arc (i, j) with x_ij > 0:  i → j  cap  x_ij

    where d_i = Σ_j x_{ij} (LP out-degree, precomputed by the caller).
    """
    s, t = N, N + 1
    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(range(N + 2))

    # Force-inclusion / force-exclusion INF arcs
    G.add_edge(s, w, capacity=INF)
    G.add_edge(p, t, capacity=INF)
    G.add_edge(q, t, capacity=INF)

    # Profit / penalty arcs for optional nodes
    for i in range(N):
        if i in (p, q, w):
            continue
        di = d[i]
        if di > 1.0:
            G.add_edge(s, i, capacity=di - 1.0)
        else:
            G.add_edge(i, t, capacity=1.0 - di)

    # LP arc capacities
    for (i, j), val in x_sol.items():
        if val > 1e-12:
            G.add_edge(i, j, capacity=val)

    return G


def separate_tk_cuts(
    x_sol: dict[Arc, float],
    N: int,
    threshold: float = 1e-6,
) -> list[TkCut]:
    """Find most-violated T_k (lifted cycle) inequalities via min-cut separation.

    For every ordered triple (p, w, q) of distinct nodes the T_k inequality is:

        x(S, S)  +  x_{pw}  +  x_{wq}  +  x_{pq}  ≤  |S|

    with w ∈ S,  p ∉ S,  q ∉ S,  and  |S| ≥ 2.

    Validity note: the inequality is valid for ATSP only when |S| ≥ 2.
    For |S| = 1 (i.e. S = {w}), the constraint reduces to
    x_pw + x_wq + x_pq ≤ 1, which is violated by any Hamiltonian cycle that
    uses the path p→w→q.  Cuts with |S| = 1 are therefore rejected.

    Necessary screening condition (Salazar, 2026-05-22): for LP-feasible
    solutions (d_i = 1 ∀i), the SEC guarantees x(S, V\S) ≥ 1 for all S with
    |S| ≥ 2, so violation = x_pw + x_wq + x_pq − mc can be positive only when
    x_pw + x_wq + x_pq > 1.  Triples with a smaller sum are skipped.

    The optimal S for a fixed triple (satisfying the screening) is found by
    solving a max-weight closure problem reformulated as a min s-t cut.
    The violation formula is:

        violation  =  D  −  min_cut  +  (d_w − 1)  +  x_{pw} + x_{wq} + x_{pq}

    where  D = Σ_{i ∉ {p,q,w},  d_i > 1}  (d_i − 1)
    and    d_i = Σ_j x_{ij}  (LP out-degree).

    Returns at most *_MAX_TK_CUTS* violated cuts sorted by violation descending.
    """
    if N < 3:
        return []

    INF: float = sum(x_sol.values()) + float(N) + 1.0

    # LP out-degrees (precomputed once for all triples)
    d: dict[int, float] = {i: 0.0 for i in range(N)}
    for (i, _j), val in x_sol.items():
        d[i] += val

    # Sum of all positive node profits (will be adjusted per triple)
    D_all: float = sum(d[i] - 1.0 for i in range(N) if d[i] > 1.0)

    s_node, t_node = N, N + 1
    cuts: list[TkCut] = []
    seen: set[tuple[frozenset[int], int, int, int]] = set()

    for p in range(N):
        for w in range(N):
            if w == p:
                continue
            for q in range(N):
                if q == p or q == w:
                    continue

                x_pw = x_sol.get((p, w), 0.0)
                x_wq = x_sol.get((w, q), 0.0)
                x_pq = x_sol.get((p, q), 0.0)

                # Screening: Salazar's necessary condition.
                # For LP-feasible solutions (d_i = 1 ∀i), the min-cut over any set
                # S ⊇ {w} with |S| ≥ 2 and {p,q} ∩ S = ∅ satisfies mc ≥ 1 (by the
                # SEC: x(S,S) ≤ |S|−1 ⟹ x(S,V\S) = |S| − x(S,S) ≥ 1).
                # Therefore violation = x_pw + x_wq + x_pq − mc can be positive only
                # when x_pw + x_wq + x_pq > 1.  Skip all triples that cannot yield a
                # violated valid T_k cut.
                if x_pw + x_wq + x_pq <= 1.0:
                    continue

                # D for this triple: exclude p, q, w from the profit sum
                D = D_all
                for node in (p, q, w):
                    if d[node] > 1.0:
                        D -= d[node] - 1.0

                G = _build_tk_network(x_sol, d, N, p, w, q, INF)
                mc, (s_side, _t_side) = nx.minimum_cut(G, s_node, t_node, capacity="capacity")

                violation = D - mc + (d[w] - 1.0) + x_pw + x_wq + x_pq

                if violation > threshold:
                    S = frozenset(v for v in s_side if v < N)
                    if w not in S:
                        continue  # degenerate cut — skip
                    if len(S) < 2:
                        continue  # T_k is invalid for |S| = 1: x_pw + x_wq + x_pq ≤ 1
                                   # does not hold for all Hamiltonian cycles (e.g. p→w→q).

                    key = (S, p, w, q)
                    if key in seen:
                        continue
                    seen.add(key)
                    cuts.append(TkCut(S=S, p=p, w=w, q=q, violation=violation))

    cuts.sort(key=lambda c: c.violation, reverse=True)
    return cuts[:_MAX_TK_CUTS]
