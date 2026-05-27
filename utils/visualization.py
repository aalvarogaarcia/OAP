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
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from numpy.typing import NDArray

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
    x = model._x_results
    points: NDArray[np.int64] = model._points_

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
                str(i),
                (pt[0], pt[1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=10,
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )
        else:
            plt.annotate(
                str(i),
                (pt[0], pt[1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color="black",
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
        G,
        pos,
        edgelist=active_edges,
        edge_color="gray",
        style="dashed",
        alpha=0.6,
        width=2,
        arrows=True,
    )

    color_map: dict[str, str] = {
        "alpha": "red",
        "alpha_p": "darkred",
        "beta": "blue",
        "beta_p": "darkblue",
        "gamma": "green",
        "gamma_p": "darkgreen",
        "delta": "orange",
        "delta_p": "darkorange",
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
            G,
            pos,
            edgelist=farkas_edges,
            edge_color=color,
            width=2.5,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f"Rayo {comp_name}"))

    legend_handles.append(
        Line2D([0], [0], color="gray", linestyle="dashed", lw=2, label=r"Solución candidata $\bar{x}$")
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
    sorted_coeffs: SerializedCoeffMap = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
    names = list(sorted_coeffs.keys())
    values = list(sorted_coeffs.values())

    plt.figure(figsize=(10, 5))
    colors = ["red" if v < 0 else "blue" for v in values]
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Pesos del Corte - Iteración {log_entry['iteration']} ({log_entry['subproblem']})")
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


# ---------------------------------------------------------------------------
# Strengthening-constraint diagnostics
# ---------------------------------------------------------------------------


def plot_strengthening_constraints(
    points: NDArray[np.int64],
    ch: NDArray[np.int64],
    x_keys: list[Arc],
    crossing_pairs: set[tuple[Arc, Arc]],
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """Draw the point set annotated with R2 arcs and R3 crossing pairs.

    Args:
        points: (N, 2) array of point coordinates.
        ch: Convex-hull node indices (ordered).
        x_keys: All arcs present in the master model after CH cleanup.
        crossing_pairs: Set of (arc_a, arc_b) ordered crossing pairs used for R3/R4.
        save_path: If given, save to this path instead of displaying.
        show_plot: Display interactively when *save_path* is None.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # Points
    ax.scatter(points[:, 0], points[:, 1], color="steelblue", s=60, zorder=5)
    ch_set = set(ch.tolist())
    for i, pt in enumerate(points):
        color = "crimson" if i in ch_set else "black"
        ax.annotate(
            str(i),
            (pt[0], pt[1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )

    # Convex hull boundary
    for idx in range(len(ch)):
        u, v = int(ch[idx]), int(ch[(idx + 1) % len(ch)])
        ax.plot(
            [points[u, 0], points[v, 0]],
            [points[u, 1], points[v, 1]],
            color="green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )

    # All master arcs (light grey, undirected to reduce clutter)
    drawn_undirected: set[tuple[int, int]] = set()
    for i, j in x_keys:
        key = (min(i, j), max(i, j))
        if key not in drawn_undirected:
            drawn_undirected.add(key)
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                color="lightgrey",
                linewidth=0.6,
                alpha=0.5,
                zorder=1,
            )

    # R3 crossing pairs — draw each crossing pair once, colour-coded
    already_drawn: set[frozenset[tuple[int, int]]] = set()
    for arc_a, arc_b in crossing_pairs:
        pair_key: frozenset[tuple[int, int]] = frozenset({arc_a, arc_b})
        if pair_key in already_drawn:
            continue
        already_drawn.add(pair_key)
        ia, ja = arc_a
        ib, jb = arc_b
        # Midpoint of first arc to midpoint of second arc
        mx_a = (points[ia, 0] + points[ja, 0]) / 2
        my_a = (points[ia, 1] + points[ja, 1]) / 2
        mx_b = (points[ib, 0] + points[jb, 0]) / 2
        my_b = (points[ib, 1] + points[jb, 1]) / 2
        ax.plot(
            [points[ia, 0], points[ja, 0]],
            [points[ia, 1], points[ja, 1]],
            color="darkorange",
            linewidth=1.5,
            alpha=0.7,
            zorder=3,
        )
        ax.plot(
            [points[ib, 0], points[jb, 0]],
            [points[ib, 1], points[jb, 1]],
            color="darkorange",
            linewidth=1.5,
            alpha=0.7,
            zorder=3,
        )
        ax.plot([mx_a, mx_b], [my_a, my_b], color="darkorange", linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)

    legend_handles = [
        Line2D([0], [0], color="green", linestyle="--", lw=1.2, label="CH boundary"),
        Line2D([0], [0], color="lightgrey", lw=1.0, label="Master arcs"),
        Line2D([0], [0], color="darkorange", lw=1.5, label=f"R3 crossing pairs ({len(already_drawn)})"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    ax.set_title(
        f"Strengthening constraints — {len(x_keys)} arcs, {len(already_drawn)} crossing pairs",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    elif show_plot:
        plt.show()


# ---------------------------------------------------------------------------
# Polyhedral facet → LaTeX export
# ---------------------------------------------------------------------------

def facets_to_latex(jsonl_path: str, output_path: str) -> None:
    """Convert a polyhedral-description JSONL log to a LaTeX ``.tex`` file.

    Each line of *jsonl_path* must be a JSON object with the keys:

    * ``"iteration"`` – int
    * ``"num_facets"`` – int
    * ``"facets"`` – list of facet dicts, each with:

      * ``"sense"`` – ``"=="`` or ``"<="``
      * ``"rhs"``   – float
      * ``"components"`` – ``dict[str, float]`` mapping variable name →
        coefficient.  Variable naming conventions recognised:

        - ``x_i_j``  → :math:`x_{i,j}`
        - ``f_i_j``  → :math:`f_{i,j}`
        - ``y_k``    → :math:`y_k`
        - ``yp_k``   → :math:`yp_k`

    The output is a single ``.tex`` file with one ``\\maketitle`` block per
    iteration.  Each iteration is divided into sections by variable type
    (``x``, ``f``, ``y``, ``y+``) and subsections inferred from fixed
    structural rules.  Long ``align`` environments are split every 40
    equations to avoid LaTeX memory issues.

    Parameters
    ----------
    jsonl_path:
        Path to the input ``.json`` / ``.jsonl`` file.
    output_path:
        Destination ``.tex`` file path (parent directories are created
        automatically).
    """
    import json
    import re
    from datetime import date

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _var_to_tex(name: str) -> str:
        """Convert an internal variable name to a LaTeX subscript token."""
        # x_i_j  /  f_i_j  → var_{i,j}
        m2 = re.fullmatch(r"([a-zA-Z]+)_(\d+)_(\d+)", name)
        if m2:
            prefix, i, j = m2.group(1), m2.group(2), m2.group(3)
            return f"{prefix}_{{{i},{j}}}"
        # yp_k  /  y_k  → yp_k  (single subscript, no braces if single digit)
        m1 = re.fullmatch(r"([a-zA-Z]+)_(\d+)", name)
        if m1:
            prefix, k = m1.group(1), m1.group(2)
            sub = k if len(k) == 1 else f"{{{k}}}"
            return f"{prefix}_{sub}"
        return name  # fallback

    def _fmt_coeff(c: float, first: bool) -> str:
        """Render a coefficient as a LaTeX string prefix for its variable."""
        # Represent as integer when possible
        ci = int(c) if c == int(c) else None
        if ci is not None:
            if ci == 1:
                return "" if first else "+ "
            if ci == -1:
                return "-"
            # large integer coefficient: use \, thin space
            if first:
                return f"{ci}\\,"
            return f"+ {ci}\\," if ci > 0 else f"- {abs(ci)}\\,"
        # float fallback
        if first:
            return f"{c:.2f}"
        return f"+ {c:.2f}" if c > 0 else f"- {abs(c):.2f}"

    def _fmt_rhs(r: float) -> str:
        ri = int(r) if r == int(r) else None
        return str(ri) if ri is not None else f"{r:.2f}"

    def _build_lhs(components: dict[str, float]) -> str:
        """Render the left-hand side of a constraint in LaTeX."""
        terms: list[str] = []
        for idx, (var, coeff) in enumerate(components.items()):
            tex_var = _var_to_tex(var)
            prefix = _fmt_coeff(coeff, idx == 0)
            if prefix.endswith("\\,"):
                terms.append(f"{prefix}{tex_var}")
            elif prefix in ("", "+ "):
                terms.append(f"{prefix}{tex_var}")
            elif prefix == "-":
                terms.append(f"-{tex_var}")
            else:
                terms.append(f"{prefix}{tex_var}")
        return " ".join(terms)

    def _sense_tex(sense: str) -> str:
        return r"&=" if sense == "==" else r"&\leq"

    def _align_blocks(
        equations: list[str], split: int = 40
    ) -> list[list[str]]:
        """Split a flat list of equation strings into chunks of *split*."""
        return [equations[i : i + split] for i in range(0, len(equations), split)]

    def _render_align(equations: list[str], extra_vspace: list[int] | None = None) -> list[str]:
        """Return lines for one or more align environments."""
        if not equations:
            return []
        blocks = _align_blocks(equations)
        lines: list[str] = []
        for block in blocks:
            lines.append(r"\begin{align}")
            for eq in block:
                lines.append(f"  {eq}")
            lines.append(r"\end{align}")
            lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Variable-type classification
    # ------------------------------------------------------------------

    def _var_types(components: dict[str, float]) -> set[str]:
        """Return the set of variable prefixes present in *components*."""
        prefixes: set[str] = set()
        for name in components:
            m = re.match(r"([a-zA-Z]+)_", name)
            if m:
                prefixes.add(m.group(1))
        return prefixes

    def _primary_section(components: dict[str, float]) -> str:
        """Return the primary section key for a facet."""
        prefixes = _var_types(components)
        if "yp" in prefixes:
            return "yp"
        if "y" in prefixes:
            return "y"
        if "f" in prefixes:
            return "f"
        return "x"

    # ------------------------------------------------------------------
    # Subsection classification (fixed rule-based)
    # ------------------------------------------------------------------

    def _subsection_x(facet: dict[str, object]) -> str:
        sense = facet["sense"]
        rhs = float(facet["rhs"])  # type: ignore[arg-type]
        components: dict[str, float] = facet["components"]  # type: ignore[assignment]
        if sense == "==" and rhs == 1.0 and all(v == 1.0 for v in components.values()):
            return "Assignment constraints"
        if sense == "==":
            return "Fixing constraints"
        return "Inequality constraints"

    def _subsection_f(facet: dict[str, object]) -> str:
        sense = facet["sense"]
        components: dict[str, float] = facet["components"]  # type: ignore[assignment]
        prefixes = _var_types(components)
        if sense == "<=":
            return "Inequality constraints"
        # pure f_ terms → flow conservation
        if prefixes == {"f"}:
            return "Flow conservation constraints"
        # one x_ + one f_ → linking
        x_vars = [k for k in components if k.startswith("x_")]
        f_vars = [k for k in components if k.startswith("f_")]
        if len(x_vars) == 1 and len(f_vars) == 1:
            return "Linking constraints"
        return "Flow / balance constraints"

    def _subsection_yp(facet: dict[str, object]) -> str:
        sense = facet["sense"]
        components: dict[str, float] = facet["components"]  # type: ignore[assignment]
        if sense == "<=":
            return "Inequality constraints"
        # structural: only yp_ vars, all same sign → global/structural
        coeffs = list(components.values())
        if all(c > 0 for c in coeffs) or all(c < 0 for c in coeffs):
            return "Global and structural constraints"
        return "Flow / balance constraints"

    def _subsection_y(facet: dict[str, object]) -> str:
        sense = facet["sense"]
        components: dict[str, float] = facet["components"]  # type: ignore[assignment]
        if sense == "<=":
            return "Inequality constraints"
        coeffs = list(components.values())
        if all(c > 0 for c in coeffs) or all(c < 0 for c in coeffs):
            return "Global and structural constraints"
        return "Flow / balance constraints"

    _SUBSECTION_DISPATCH: dict[str, Any] = {
        "x": _subsection_x,
        "f": _subsection_f,
        "yp": _subsection_yp,
        "y": _subsection_y,
    }

    # Desired subsection ordering within each section
    _SUBSECTION_ORDER: dict[str, list[str]] = {
        "x": ["Assignment constraints", "Fixing constraints", "Inequality constraints"],
        "f": [
            "Linking constraints",
            "Flow conservation constraints",
            "Flow / balance constraints",
            "Inequality constraints",
        ],
        "yp": [
            "Global and structural constraints",
            "Flow / balance constraints",
            "Inequality constraints",
        ],
        "y": [
            "Global and structural constraints",
            "Flow / balance constraints",
            "Inequality constraints",
        ],
    }

    _SECTION_TITLES: dict[str, str] = {
        "x": r"Variables \boldmath$x$",
        "f": r"Variables \boldmath$f$",
        "yp": r"Variables \boldmath$y^+$",
        "y": r"Variables \boldmath$y^-$",
    }

    # Desired section ordering
    _SECTION_ORDER = ["x", "f", "y", "yp"]

    # ------------------------------------------------------------------
    # Read all iterations
    # ------------------------------------------------------------------

    iterations: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                iterations.append(json.loads(line))

    # ------------------------------------------------------------------
    # Build .tex content
    # ------------------------------------------------------------------

    today = date.today()
    today_str = f"{today.day} {today.strftime('%B %Y')}"

    output_lines: list[str] = []

    for iter_obj in iterations:
        iteration = int(iter_obj["iteration"])
        facets: list[dict[str, Any]] = iter_obj["facets"]

        # Title block
        output_lines += [
            rf"\title{{Polyhedral Description\\[0.5em]\large Iteration {iteration}}}",
            rf"\date{{{today_str}}}",
            "",
            r"\maketitle",
            "",
        ]

        # Group facets by section, then subsection
        # sections: dict[section_key, dict[subsection_label, list[str(equation)]]]
        from collections import defaultdict

        sections: dict[str, dict[str, list[str]]] = {
            k: defaultdict(list) for k in _SECTION_ORDER
        }

        for facet in facets:
            components: dict[str, float] = facet["components"]
            sense = str(facet["sense"])
            rhs = float(facet["rhs"])

            sec = _primary_section(components)
            sub_fn = _SUBSECTION_DISPATCH[sec]
            sub = sub_fn(facet)

            lhs = _build_lhs(components)
            sense_tok = _sense_tex(sense)
            rhs_tok = _fmt_rhs(rhs)
            eq_line = f"{lhs} {sense_tok} {rhs_tok} \\\\"
            sections[sec][sub].append(eq_line)

        # Render sections
        for sec_key in _SECTION_ORDER:
            sub_dict = sections[sec_key]
            if not sub_dict:
                continue

            total_in_section = sum(len(v) for v in sub_dict.values())
            sec_title = _SECTION_TITLES[sec_key]

            output_lines += [
                "% " + "=" * 60,
                rf"\section{{{sec_title}}}",
                f"% {total_in_section} facets",
                "% " + "=" * 60,
                "",
            ]

            sub_order = _SUBSECTION_ORDER[sec_key]
            # include any subsections not in the predefined order at the end
            extra = [s for s in sub_dict if s not in sub_order]
            for sub_label in sub_order + extra:
                eqs = sub_dict.get(sub_label)
                if not eqs:
                    continue
                output_lines.append(rf"\subsection*{{{sub_label}}}")
                output_lines.append("")
                output_lines += _render_align(eqs)

        output_lines.append("")  # blank line between iterations

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------

    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines))
        fh.write("\n")


# ---------------------------------------------------------------------------
# Polyhedral facet → integer solution enumeration (Gurobi solution pool)
# ---------------------------------------------------------------------------


def enumerate_facet_solutions(
    jsonl_path: str,
    iteration: int = 0,
    var_prefix: str = "x",
    max_solutions: int = 10_000,
    time_limit: float = 300.0,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Enumerate all integer feasible solutions described by a polyhedral JSONL log.

    Reads the facets for the given *iteration* from *jsonl_path*, builds a
    Gurobi MIP model where variables whose name starts with *var_prefix* are
    declared **binary** and all other variables are **continuous** (lb = 0),
    then uses Gurobi's solution-pool exhaustive search
    (``PoolSearchMode = 2``) to find up to *max_solutions* feasible points.

    Parameters
    ----------
    jsonl_path:
        Path to the polyhedral JSONL log produced by
        :py:meth:`~models.OAPBaseModel.log_facets`.
    iteration:
        Which iteration entry to use (matched by the ``"iteration"`` key).
        Defaults to ``0``.
    var_prefix:
        Variable-name prefix that should be treated as **binary**.  All
        variables whose name starts with this prefix get ``vtype=GRB.BINARY``;
        the rest are continuous (lb = 0, ub = GRB.INFINITY).
        Defaults to ``"x"`` (the arc-selection variables).
    max_solutions:
        Maximum number of feasible solutions to collect
        (``Params.PoolSolutions``).  Defaults to ``10_000``.
    time_limit:
        Gurobi time limit in seconds.  Defaults to ``300.0``.
    verbose:
        If ``True``, Gurobi output is printed to stdout.  Defaults to
        ``False``.

    Returns
    -------
    list[dict[str, float]]
        Each element is a dict ``{var_name: value}`` containing **only** the
        variables whose name starts with *var_prefix*, with values rounded to
        the nearest integer (0 or 1 for binary variables).

    Notes
    -----
    * The function uses a **dummy objective** (minimise 0) — it is a pure
      feasibility enumeration.
    * ``PoolSearchMode = 2`` performs exhaustive search; for large instances
      the number of solutions can be exponential.  Use *max_solutions* and
      *time_limit* to cap the run.
    * Non-``var_prefix`` variables (``f``, ``y``, ``yp``) are continuous so
      that the full 62-facet system remains feasible without requiring an
      integer assignment for auxiliary variables.
    """
    import json
    import re
    import time

    import gurobipy as gp
    from gurobipy import GRB

    # ------------------------------------------------------------------
    # 1. Read the requested iteration
    # ------------------------------------------------------------------
    target: dict[str, Any] | None = None
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj: dict[str, Any] = json.loads(line)
            if int(obj["iteration"]) == iteration:
                target = obj
                break

    if target is None:
        raise ValueError(
            f"Iteration {iteration} not found in {jsonl_path!r}"
        )

    facets: list[dict[str, Any]] = target["facets"]

    # ------------------------------------------------------------------
    # 2. Collect all variable names and infer types
    # ------------------------------------------------------------------
    all_vars: set[str] = set()
    for facet in facets:
        all_vars.update(facet["components"].keys())

    # ------------------------------------------------------------------
    # 3. Build Gurobi model
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()

    model = gp.Model(env=env)
    model.Params.OutputFlag = 1 if verbose else 0

    # Create variables
    gvars: dict[str, gp.Var] = {}
    for vname in sorted(all_vars):
        prefix_match = re.match(r"([a-zA-Z]+)_", vname)
        prefix = prefix_match.group(1) if prefix_match else ""
        if prefix == var_prefix:
            gvars[vname] = model.addVar(vtype=GRB.BINARY, name=vname)
        else:
            gvars[vname] = model.addVar(lb=0.0, ub=GRB.INFINITY,
                                         vtype=GRB.CONTINUOUS, name=vname)

    model.update()

    # Add constraints
    for facet in facets:
        components: dict[str, float] = facet["components"]
        sense: str = facet["sense"]
        rhs: float = float(facet["rhs"])

        lhs_expr: gp.LinExpr = gp.LinExpr()
        for vname, coeff in components.items():
            lhs_expr.add(gvars[vname], coeff)

        if sense == "==":
            model.addConstr(lhs_expr == rhs)
        elif sense == "<=":
            model.addConstr(lhs_expr <= rhs)
        else:
            raise ValueError(f"Unsupported constraint sense: {sense!r}")

    # Dummy objective — pure feasibility
    model.setObjective(gp.LinExpr(), GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # 4. Configure solution pool for exhaustive enumeration
    # ------------------------------------------------------------------
    model.Params.PoolSearchMode = 2       # exhaustive
    model.Params.PoolSolutions = max_solutions
    model.Params.TimeLimit = time_limit

    # ------------------------------------------------------------------
    # 5. Solve
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    n_found = model.SolCount
    if not verbose:
        print(f"[enumerate_facet_solutions] {n_found} solution(s) found "
              f"in {elapsed:.2f}s  (iteration={iteration}, "
              f"var_prefix={var_prefix!r})")

    # ------------------------------------------------------------------
    # 6. Extract solutions — only var_prefix variables
    # ------------------------------------------------------------------
    solutions: list[dict[str, float]] = []
    prefix_vars = sorted(
        [vname for vname in gvars if vname.startswith(f"{var_prefix}_")]
    )

    for s in range(n_found):
        model.Params.SolutionNumber = s
        sol: dict[str, float] = {
            vname: round(gvars[vname].Xn)
            for vname in prefix_vars
        }
        solutions.append(sol)

    model.dispose()
    env.dispose()

    return solutions
