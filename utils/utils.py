"""Backward-compatibility re-export shim for ``utils.utils``.

All implementation has been moved to focused submodules:

* :mod:`utils.geometry`      – pure geometry, I/O helpers, and type aliases
* :mod:`utils.constraints`   – Gurobi constraint injection helpers
* :mod:`utils.benders_log`   – Benders cut logging and JSONL serialisation
* :mod:`utils.visualization` – plotting and diagnostic visualisations

This file re-exports every public name so that the 22 existing import sites
(``from utils.utils import ...``) continue to work without any changes.
Do **not** add logic here.
"""

from utils.geometry import (
    Arc,
    PointLookup,
    are_triangles_incompatible,
    compute_convex_hull,
    compute_convex_hull_area,
    compute_crossing_edges,
    compute_triangles,
    contains_proper,
    cost_function_area,
    extract_metric_to_csv,
    incompatible_triangles,
    is_colineal,
    minimal_triangle_adjency_list,
    orientation_2d,
    point_in_triangle,
    read_data,
    read_indexed_instance,
    segments_intersect,
    signed_area,
    triangles_adjacency_list,
    triangles_area,
    write_prefile,
)
from utils.constraints import (
    ArcConstraintMap,
    aplicar_semiplanos_por_capas,
    inyectar_cliques_de_cruce,
    inyectar_cortes_knapsack_locales,
    restricciones_semiplano,
    restricciones_semiplano_chain,
    restricciones_semiplanoV2,
)
from utils.benders_log import (
    SerializedCoeffMap,
    SerializedExpr,
    SerializedRayData,
    format_cut_string,
    load_farkas_logs,
    log_benders_cut,
    log_inv_benders_cut,
    parse_edge,
    serialize_expr,
)
from utils.visualization import (
    plot_cut_heatmap,
    plot_cut_weights,
    plot_farkas_ray_network,
    plot_sankey_traceability,
    plot_solution,
)

__all__ = [
    # --- geometry ---
    "Arc",
    "PointLookup",
    "are_triangles_incompatible",
    "compute_convex_hull",
    "compute_convex_hull_area",
    "compute_crossing_edges",
    "compute_triangles",
    "contains_proper",
    "cost_function_area",
    "extract_metric_to_csv",
    "incompatible_triangles",
    "is_colineal",
    "minimal_triangle_adjency_list",
    "orientation_2d",
    "point_in_triangle",
    "read_data",
    "read_indexed_instance",
    "segments_intersect",
    "signed_area",
    "triangles_adjacency_list",
    "triangles_area",
    "write_prefile",
    # --- constraints ---
    "ArcConstraintMap",
    "aplicar_semiplanos_por_capas",
    "inyectar_cliques_de_cruce",
    "inyectar_cortes_knapsack_locales",
    "restricciones_semiplano",
    "restricciones_semiplano_chain",
    "restricciones_semiplanoV2",
    # --- benders_log ---
    "SerializedCoeffMap",
    "SerializedExpr",
    "SerializedRayData",
    "format_cut_string",
    "load_farkas_logs",
    "log_benders_cut",
    "log_inv_benders_cut",
    "parse_edge",
    "serialize_expr",
    # --- visualization ---
    "plot_cut_heatmap",
    "plot_cut_weights",
    "plot_farkas_ray_network",
    "plot_sankey_traceability",
    "plot_solution",
]
