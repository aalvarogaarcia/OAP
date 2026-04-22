"""Visualisation utilities for OAP solutions and Benders cut diagnostics.

All functions in this module are side-effect-only (they produce plots or
save image files).  None of them are required by the solver pipeline and
they should never be imported unconditionally in hot paths.
"""
from __future__ import annotations

import os
from typing import Any, cast

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from numpy.typing import NDArray

import plotly.graph_objects as go

from utils.benders_log import SerializedCoeffMap, SerializedExpr, SerializedRayData, parse_edge
from utils.geometry import Arc, PointLookup, compute_convex_hull


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Solution plot
# ---------------------------------------------------------------------------

def plot_solution(model: gp.Model, title: str = "Solution") -> None:
    """Draw the tour stored in *model._x_results* over *model._points_*."""
    x = model._x_results  # type: ignore[attr-defined]
    points: NDArray[np.int64] = model._points_  # type: ignore[attr-defined]

    G: nx.DiGraph = nx.DiGraph()
    G.add_edges_from(x)

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], color="blue")

    for edge in G.edges():
        pt1 = points[edge[0]]
        pt2 = points[edge[1]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "r-", alpha=0.7)

    hull = compute_convex_hull(points)
    hull_set = set(hull)
    for i in range(len(hull)):
        pt1 = points[hull[i]]
        pt2 = points[hull[(i + 1) % len(hull)]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "g-.", alpha=0.5)

    for i, pt in enumerate(points):
        if i in hull_set:
            plt.annotate(
                str(i), (pt[0], pt[1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=10, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )
        else:
            plt.annotate(
                str(i), (pt[0], pt[1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Benders cut diagnostics
# ---------------------------------------------------------------------------

def plot_farkas_ray_network(
    log_entry: dict[str, Any],
    points: PointLookup | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """Draw the candidate solution and Farkas-ray components as a directed graph.

    Args:
        log_entry: A single record loaded from the JSONL cut log.
        points: Node coordinates ``{id: (x, y)}`` or an ``(n, 2)`` array.
            If ``None``, spring layout is used.
        save_path: File path to save the figure.  Mutually exclusive with
            *show_plot*.
        show_plot: Display the figure interactively when *save_path* is ``None``.
    """
    G: nx.DiGraph = nx.DiGraph()

    iteration = log_entry.get("iteration", "N/A")
    subproblem = log_entry.get("subproblem", "Unknown")
    violation = log_entry.get("violation", 0.0)
    active_x_strs = cast(SerializedCoeffMap, log_entry.get("active_x", {}))
    ray_components = cast(SerializedRayData, log_entry.get("ray_components", {}))

    active_edges: list[Arc] = [parse_edge(e) for e in active_x_strs]
    G.add_edges_from(active_edges)

    pos: dict[int, tuple[float, float]] | dict[int, NDArray[np.float64]]
    if points is not None:
        if isinstance(points, np.ndarray):
            pos = {int(i): (float(points[i][0]), float(points[i][1])) for i in G.nodes()}
        else:
            pos = {int(i): points[int(i)] for i in G.nodes()}
    else:
        pos = cast(
            dict[int, NDArray[np.float64]],
            nx.spring_layout(G, seed=42),
        )

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=300, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=active_edges,
        edge_color="gray", style="dashed", alpha=0.6, width=2, arrows=True,
    )

    color_map: dict[str, str] = {
        "alpha": "red", "alpha_p": "darkred",
        "beta": "blue", "beta_p": "darkblue",
        "gamma": "green", "gamma_p": "darkgreen",
        "delta": "orange", "delta_p": "darkorange",
    }

    legend_handles: list[Line2D] = []
    for comp_name, edges_dict in ray_components.items():
        if not edges_dict or not isinstance(edges_dict, dict):
            continue
        color = color_map.get(comp_name, "purple")
        farkas_edges: list[Arc] = [parse_edge(e) for e in edges_dict]
        G.add_edges_from(farkas_edges)
        if points is None:
            pos = cast(
                dict[int, NDArray[np.float64]],
                nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), seed=42),
            )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=farkas_edges,
            edge_color=color, width=2.5, arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f"Rayo {comp_name}"))

    legend_handles.append(
        Line2D([0], [0], color="gray", linestyle="dashed", lw=2,
               label=r"Solución candidata $\bar{x}$")
    )
    plt.title(
        f"Iteración: {iteration} | Subproblema: {subproblem}\nViolación: {violation}",
        fontsize=14,
    )
    plt.legend(handles=legend_handles, loc="best")
    plt.axis("off")

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    elif show_plot:
        plt.show()


def plot_cut_heatmap(
    log_entry: dict[str, Any],
    num_nodes: int,
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Render the cut-expression coefficients as an adjacency-matrix heatmap."""
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data:
        return

    adj_matrix: NDArray[np.float64] = np.zeros((num_nodes, num_nodes))
    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    for var_name, coeff in coeffs.items():
        try:
            parts = var_name.split("_")
            i, j = int(parts[1]), int(parts[2])
            adj_matrix[i, j] = coeff
        except Exception:
            continue

    plt.figure(figsize=(6, 5))
    sns.heatmap(adj_matrix, annot=True, cmap="RdBu", center=0)
    plt.title(f"Estructura del Corte - Iteración {log_entry['iteration']}")
    plt.xlabel("Nodo Destino (j)")
    plt.ylabel("Nodo Origen (i)")

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    elif show_plot:
        plt.show()


def plot_cut_weights(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Draw a bar chart of the cut-expression coefficients sorted by magnitude."""
    cut_data = cast(SerializedExpr | None, log_entry.get("cut_expr"))
    if not cut_data or not cut_data["coeffs"]:
        print("plot_cut_weights: no cut expression data in this log entry.")
        return

    coeffs = cast(SerializedCoeffMap, cut_data["coeffs"])
    sorted_coeffs: SerializedCoeffMap = dict(
        sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True)
    )
    names = list(sorted_coeffs.keys())
    values = list(sorted_coeffs.values())

    plt.figure(figsize=(10, 5))
    colors = ["red" if v < 0 else "blue" for v in values]
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.title(
        f"Pesos del Corte - Iteración {log_entry['iteration']} ({log_entry['subproblem']})"
    )
    plt.ylabel("Coeficiente en el Maestro")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    elif show_plot:
        plt.show()


def plot_sankey_traceability(
    log_entry: dict[str, Any],
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Render a Sankey diagram showing dual-component → variable traceability."""
    trace = cast(dict[str, dict[str, Any]] | None, log_entry.get("traceability"))
    if not trace:
        return

    sources: list[str] = []
    targets: list[str] = []
    values: list[float] = []
    for x_var, info in trace.items():
        sources.append(cast(str, info["source_component"]))
        targets.append(x_var)
        values.append(abs(float(cast(float, info["dual_value"]))))

    nodes = list(set(sources + targets))
    source_idx = [nodes.index(s) for s in sources]
    target_idx = [nodes.index(t) for t in targets]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes, pad=15, thickness=20),
                link=dict(source=source_idx, target=target_idx, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Trazabilidad: Del Subproblema al Maestro", font_size=12)

    if save_path:
        _ensure_dir(save_path)
        fig.write_image(save_path)
    elif show_plot:
        fig.show()
