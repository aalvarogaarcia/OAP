from __future__ import annotations

import csv
import glob
import os
from numbers import Real
from typing import Any

import inquirer

from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ResultRow = dict[str, Any]

# ---------------------------------------------------------------------------
# Constants — integer CLI codes mapped to the string keys the model expects
# ---------------------------------------------------------------------------
_OBJECTIVE_MAP: dict[int, str] = {
    0: "Fekete",
    1: "Internal",
    2: "External",
    3: "Diagonals",
}

_CSV_FIELDNAMES: list[str] = [
    "Instance",
    "N",
    "Convex Hull Area",
    "Cols",
    "Rows",
    "LP Value",
    "Gap (%)",
    "IP Value",
    "Time (s)",
    "Nodes",
]
_CSV_KEY_MAP: dict[str, str] = {
    "Instance": "Instance",
    "N": "N",
    "Convex_Hull_Area": "Convex Hull Area",
    "Cols": "Cols",
    "Rows": "Rows",
    "LP": "LP Value",
    "Gap": "Gap (%)",
    "IP": "IP Value",
    "Time": "Time (s)",
    "Nodes": "Nodes",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".2f") -> str:
    """Format *value* with *spec* when it is numeric; fall back to str() for
    sentinel values such as '-' returned by get_model_stats()."""
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def build_and_solve(
    instance_path: str,
    maximize: bool,
    subtour: str,
    time_limit: int,
    obj: int,
    mode: int,
    sum_constrain: bool,
    strengthen: bool,
    semiplane: int,
    use_knapsack: bool,
    use_cliques: bool,
    crossing_constrain: bool,
) -> OAPCompactModel:
    """Construct, build, and solve an OAPCompactModel for *instance_path*.

    All parameters map directly to OAPCompactModel.build() and OAPCompactModel.solve().
    """
    instance_name = os.path.basename(instance_path).replace(".instance", "").replace(".txt", "")
    points = read_indexed_instance(instance_path)
    triangles = compute_triangles(points)
    model = OAPCompactModel(points, triangles, name=instance_name)
    model.build(
        objective=_OBJECTIVE_MAP[obj],
        mode=mode,
        maximize=maximize,
        subtour=subtour,
        sum_constrain=sum_constrain,
        strengthen=strengthen,
        semiplane=semiplane,
        use_knapsack=use_knapsack,
        use_cliques=use_cliques,
        crossing_constrain=crossing_constrain,
    )
    model.solve(time_limit=time_limit)
    return model


# ---------------------------------------------------------------------------
# Interactive batch configurator
# ---------------------------------------------------------------------------


def get_batch_config() -> dict[str, object] | None:
    _LABEL_TO_OBJ: dict[str, int] = {
        "Fekete": 0,
        "Internal": 1,
        "External": 2,
        "Diagonals": 3,
    }
    _SEMIPLANE_LABEL_TO_INT: dict[str, int] = {
        "0 (off)": 0,
        "1 (V1)": 1,
        "2 (V2)": 2,
    }

    print("\n" + "=" * 50)
    print(" BATCH RUNNER - OAP (Full Options)")
    print("=" * 50 + "\n")

    questions = [
        inquirer.Text(
            "dir_path",
            message="Instance directory",
            default="instance",
            validate=lambda answers, current: len(current.strip()) > 0,
        ),
        inquirer.Text(
            "ext",
            message="File glob pattern",
            default="*.instance",
            validate=lambda answers, current: len(current.strip()) > 0,
        ),
        inquirer.List(
            "subtour",
            message="Subtour-elimination method",
            choices=["SCF", "MTZ", "MCF"],
            default="SCF",
        ),
        inquirer.List(
            "objective_label",
            message="Objective function",
            choices=["Fekete", "Internal", "External", "Diagonals"],
            default="Fekete",
        ),
        inquirer.Confirm(
            "maximize",
            message="Maximize this objective?",
            default=True,
        ),
        inquirer.List(
            "mode",
            message="Objective mode (0-3)",
            choices=["0", "1", "2", "3"],
            default="0",
        ),
        inquirer.Confirm(
            "sum_constrain",
            message="Enable triangle-sum constraints?",
            default=True,
        ),
        inquirer.Confirm(
            "strengthen",
            message="Enable strengthening constraints?",
            default=True,
        ),
        inquirer.List(
            "semiplane_label",
            message="Semiplane constraints",
            choices=["0 (off)", "1 (V1)", "2 (V2)"],
            default="0 (off)",
        ),
        inquirer.Confirm(
            "use_knapsack",
            message="Use local knapsack constraints?",
            default=False,
        ),
        inquirer.Confirm(
            "use_cliques",
            message="Use clique constraints?",
            default=False,
        ),
        inquirer.Confirm(
            "crossing_constrain",
            message="Enable crossing arc constraints?",
            default=False,
        ),
        inquirer.Text(
            "output_name",
            message="Output file name (no extension)",
            default="resultados",
            validate=lambda answers, current: len(current.strip()) > 0,
        ),
        inquirer.Text(
            "time_limit_str",
            message="Time limit (seconds)",
            default="7200",
            validate=lambda answers, current: current.strip().isdigit() and int(current.strip()) > 0,
        ),
    ]

    config = inquirer.prompt(questions)
    if config is None:
        return None

    # Post-process
    config["objective"] = config.pop("objective_label")
    config["obj"] = _LABEL_TO_OBJ[config["objective"]]
    config["mode"] = int(config["mode"])
    config["semiplane"] = _SEMIPLANE_LABEL_TO_INT[config.pop("semiplane_label")]
    config["output_name"] = config["output_name"].strip()
    config["time_limit"] = int(config["time_limit_str"].strip())
    del config["time_limit_str"]

    return config


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    dir_path: str = "instance",
    ext: str = "*.instance",
    subtour: str = "SCF",
    obj: int = 0,
    mode: int = 0,
    maximize: bool = True,
    sum_constrain: bool = True,
    strengthen: bool = True,
    semiplane: int = 0,
    use_knapsack: bool = False,
    use_cliques: bool = False,
    crossing_constrain: bool = False,
    output_name: str = "resultados",
    time_limit: int = 7200,
) -> None:
    files = glob.glob(os.path.join(dir_path, ext))
    files.sort()

    data_list: list[ResultRow] = []

    output_filename = f"outputs/LaTex/{output_name}.tex"
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        # Write LaTeX header
        f.write(r"""\documentclass{beamer}
\usepackage[english]{babel}
\usepackage[utf8x]{inputenc}
\usepackage{multicol}
\usepackage{multirow}
\mode<presentation> {
  \usetheme{default}
  \usecolortheme{default}
  \usefonttheme{default}
  \setbeamertemplate{caption}[numbered]
}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\title{OAP Optimization Results}
\author{OAP Solver}
\institute{OAP_NextGen}
\date{\today}

\begin{document}
\begin{frame}
\begin{table}[htbp]
\centering
\setbeamerfont{caption}{size=\tiny}
\caption{OAP Optimization Results}
\label{tab:oap_results}
\setlength{\tabcolsep}{1.5pt}
\resizebox{\textwidth}{!}{%
\tiny
\begin{tabular}{lrrrrrrrrr}
\toprule
\textbf{Instance} & \textbf{|N|} & \textbf{Conv.Hull} & \textbf{Cols} & \textbf{Rows} & \textbf{LP Val} & \textbf{Gap (\%)} & \textbf{IP Val} & \textbf{Time (s)} & \textbf{Nodes} \\
\midrule
""")

        for file in files:
            filename = os.path.basename(file)
            print(f"Processing: {filename}...")

            model = build_and_solve(
                instance_path=file,
                maximize=maximize,
                subtour=subtour,
                time_limit=time_limit,
                obj=obj,
                mode=mode,
                sum_constrain=sum_constrain,
                strengthen=strengthen,
                semiplane=semiplane,
                use_knapsack=use_knapsack,
                use_cliques=use_cliques,
                crossing_constrain=crossing_constrain,
            )

            lp, gap, ip, time_s, nodes = model.get_model_stats()

            instance_name = filename.replace(".txt", "").replace("_", "-").replace(".instance", "")
            size_n: int = len(model.points)
            conv_hull_area_str: str = f"{model.convex_hull_area:.2f}"
            cols: int = model.model.NumVars
            rows: int = model.model.NumConstrs

            ip_str = f"{ip:.0f}" if isinstance(ip, Real) and model.model.SolCount > 0 else str(ip)
            time_str = f"{time_s:.2f}" if isinstance(time_s, Real) and time_s < time_limit else "Timeout"

            res_row: ResultRow = {
                "Instance": instance_name,
                "N": size_n,
                "Convex_Hull_Area": model.convex_hull_area,
                "Cols": cols,
                "Rows": rows,
                "LP": lp,
                "Gap": gap,
                "IP": ip,
                "Time": time_s,
                "Nodes": nodes,
            }
            data_list.append(res_row)

            row_str = (
                f"{instance_name} & {size_n} & {conv_hull_area_str} & {cols} & {rows} & "
                f"{_fmt(lp)} & {_fmt(gap)} & {ip_str} & {time_str} & {nodes} \\\\"
            )
            f.write(row_str + "\n")

        f.write(r"""\bottomrule
\end{tabular}
}
\end{table}
\end{frame}
\end{document}
""")

    print(f"\nResults written to: {output_filename}")

    # --- CSV export ---
    csv_filename = f"outputs/CSV/{output_name}.csv"
    csv_dir = os.path.dirname(csv_filename)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    csv_rows = [{_CSV_KEY_MAP[k]: v for k, v in row.items()} for row in data_list]

    with open(csv_filename, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=_CSV_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"CSV results written to: {csv_filename}")


if __name__ == "__main__":
    _cfg = get_batch_config()
    if _cfg is None:
        print("\nCancelled.")
    else:
        main(
            dir_path=_cfg["dir_path"],
            ext=_cfg["ext"],
            subtour=_cfg["subtour"],
            obj=_cfg["obj"],
            mode=_cfg["mode"],
            maximize=_cfg["maximize"],
            sum_constrain=_cfg["sum_constrain"],
            strengthen=_cfg["strengthen"],
            semiplane=_cfg["semiplane"],
            use_knapsack=_cfg["use_knapsack"],
            use_cliques=_cfg["use_cliques"],
            crossing_constrain=_cfg["crossing_constrain"],
            output_name=_cfg["output_name"],
            time_limit=_cfg["time_limit"],
        )
