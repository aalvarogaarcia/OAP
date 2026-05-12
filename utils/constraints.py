"""Gurobi constraint injection helpers for OAP.

All functions in this module operate on a ``gp.Model`` instance and inject
geometric or combinatorial constraints directly into it.  They depend on
``utils.geometry`` for predicates but have no dependency on the logging or
visualisation submodules.
"""

from __future__ import annotations

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
