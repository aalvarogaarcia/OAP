# OAPCompactModel.py
import csv
from pathlib import Path
from typing import Any, Literal

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gurobipy import GRB
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from models.mixin.oap_builder_mixin import OAPBuilderMixin
from models.OAPBaseModel import OAPBaseModel
from utils.utils import (
    compute_convex_hull,
    compute_convex_hull_area,
    compute_hij_data,
    cost_function_area,
    incompatible_triangles,
    point_in_triangle,
    segments_intersect,
)

Arc = tuple[int, int]
ArcVarMap = dict[Arc, gp.Var]
McfArc = tuple[int, int, int]
McfVarMap = dict[McfArc, gp.Var]


class OAPCompactModel(OAPBaseModel, OAPBuilderMixin):
    def __init__(self, points: NDArray[np.int64], triangles: NDArray[np.int64], name: str = "OAPCompactModel"):
        super().__init__(points, triangles, name)
        self.N_list = range(len(points))
        self.N = len(points)
        self.CH = compute_convex_hull(points)
        self.V_list = range(len(triangles))
        self.convex_hull_area = compute_convex_hull_area(points)

        self.model = gp.Model(name)

        # Diccionarios de variables
        self.x: ArcVarMap = {}
        self.y: dict[int, gp.Var] = {}
        self.yp: dict[int, gp.Var] = {}
        self.c: dict[Arc, float] = {}

        # Variables opcionales (dependientes del modo/subtour)
        self.f: ArcVarMap = {}
        self.u: dict[int, gp.Var] = {}
        self.f_mcf: McfVarMap = {}
        self.z: ArcVarMap = {}
        self.zp: ArcVarMap = {}

    def build(
        self,
        objective: Literal["Fekete", "Internal", "External", "Diagonals"] = "Fekete",
        mode: int = 0,
        maximize: bool = True,
        subtour: Literal["SCF", "MTZ", "MCF"] = "SCF",
        sum_constrain: bool = True,
        strengthen: bool = False,
        semiplane: Literal[0, 1, 2] = 0,
        use_knapsack: bool = False,
        use_cliques: bool = False,
        crossing_constrain: bool = False,
        use_triangle_cliques: bool = False,
        arc_triangle_link: bool = False,
    ) -> None:
        """
        Orquestador principal que construye el modelo paso a paso.

        strengthen: si True añade R1 (balance de área vs CH), R2 (cota
            superior de triángulos por arco) y R3 (exclusión de arcos
            cruzados) — Sec. 5.4 del paper. Son redundantes para el IP
            pero cierran sustancialmente el gap LP.
        """
        self._create_variables(subtour, objective, mode)
        self._set_objective(objective, mode, maximize)
        self._add_degree_constraints()
        self._add_subtour_constraints(subtour)
        if sum_constrain:
            self._add_sum_constraints()
        self._add_triangle_ch_constraints()
        self._add_variable_relation_constraints(objective, mode)

        if strengthen:
            self._add_strengthening_constraints()

        if semiplane > 0:
            self.add_restricciones_semiplano(version=semiplane)

        if use_knapsack:
            self.inyectar_cortes_knapsack_locales()

        if use_cliques:
            self.inyectar_cliques_de_cruce()

        if crossing_constrain:
            self._add_crossing_constraints()

        if use_triangle_cliques:
            self._add_triangle_clique_constraints()

        if arc_triangle_link:
            self._add_arc_triangle_link_constraints()

        self.model.update()

    def solve(
        self,
        time_limit: int = 7200,
        verbose: bool = False,
        relaxed: bool = False,
        plot: bool = False,
        threads: int = 0,
    ) -> None:
        """Ejecuta la optimización del modelo, aplica relajación si es necesario
        y procesa los resultados.
        """
        if verbose:
            print("Constraints added. \nOptimizing model...")

        # --- Configuración de Parámetros ---
        self.model.setParam("OutputFlag", 1 if verbose else 0)
        self.model.setParam("TimeLimit", time_limit)
        self.model.Params.MIPGapAbs = 1.99
        self.model.Params.NodeLimit = GRB.INFINITY
        self.model.Params.SolutionLimit = GRB.MAXINT
        if threads > 0:
            self.model.setParam("Threads", threads)
            
        self.model.update()

        # --- Relajación Lineal (LP Relaxation) ---
        if relaxed:
            for v in self.model.getVars():
                if v.VType != GRB.CONTINUOUS:
                    v.VType = GRB.CONTINUOUS
            self.model.update()

        # --- Optimización ---
        self.model.optimize()

        # --- Extracción de Resultados ---
        self.x_results: list[Arc] = []
        self.x_relaxed: dict[Arc, float] = {}

        if self.model.SolCount > 0:
            self.x_results = [arc for arc, var in self.x.items() if var.X > 0.5]
            if relaxed:
                self.x_relaxed = {arc: var.X for arc, var in self.x.items() if var.X > 1e-6}

        # --- Visualización ---
        if plot:
            if self.model.SolCount > 0:
                title = "Optimal Tour" if self.model.Status == GRB.OPTIMAL else "Best Found Tour"
                self.plot(title=title, relaxed=relaxed)
            elif verbose:
                print("No feasible solution found to plot.")

    def plot(self, title: str = "Solution", relaxed: bool = False) -> None:
        """Dibuja la solución del modelo utilizando los resultados almacenados en la clase."""
        if relaxed:
            self._plot_relaxed(title)
            return

        if not hasattr(self, "x_results") or not self.x_results:
            print("No results to plot. Please solve the model first and ensure a solution was found.")
            return

        G = nx.DiGraph()
        G.add_edges_from(self.x_results)

        plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.scatter(self.points[:, 0], self.points[:, 1], color="blue")

        # Dibuja las aristas del tour
        for edge in G.edges():
            pt1 = self.points[edge[0]]
            pt2 = self.points[edge[1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "r-", alpha=0.7)

        # Dibuja la envolvente convexa (ya la calculamos en el __init__)
        hull_set = set(self.CH)
        for i in range(len(self.CH)):
            pt1 = self.points[self.CH[i]]
            pt2 = self.points[self.CH[(i + 1) % len(self.CH)]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "g-.", alpha=0.5)

        # Etiqueta los puntos con sus índices
        for i, pt in enumerate(self.points):
            if i in hull_set:
                # Puntos de la envolvente convexa en rojo y negrita
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
                # Puntos regulares en negro
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

    # ------------------------------------------------------------------
    # LP-relaxation plotting (grouped by arc value)
    # ------------------------------------------------------------------

    def _plot_relaxed(self, title: str) -> None:
        """Dibuja la solución LP relajada agrupando arcos por su valor fraccionario.

        Arcos con el mismo valor (redondeado a 3 decimales) comparten color,
        permitiendo identificar visualmente dónde se divide el flujo
        (e.g. x=0.5 indica bifurcación igual en dos caminos).
        El grosor de línea es proporcional al valor del arco.
        """
        if not hasattr(self, "x_relaxed") or not self.x_relaxed:
            print("No LP relaxation results to plot.")
            return

        # Agrupar arcos por valor redondeado
        groups: dict[float, list[Arc]] = {}
        for arc, val in self.x_relaxed.items():
            key = round(val, 3)
            groups.setdefault(key, []).append(arc)

        # Ordenar niveles de mayor a menor para que los más fuertes queden al frente
        sorted_levels = sorted(groups.keys(), reverse=True)
        cmap = plt.get_cmap("tab10")
        hull_set = set(self.CH)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{title} (LP relaxation — arc values)")
        ax.scatter(self.points[:, 0], self.points[:, 1], color="blue", zorder=3)

        # Convex hull
        for i in range(len(self.CH)):
            pt1 = self.points[self.CH[i]]
            pt2 = self.points[self.CH[(i + 1) % len(self.CH)]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "g-.", alpha=0.4, zorder=1)

        # Arcos coloreados por nivel de valor
        legend_handles = []
        for idx, level in enumerate(sorted_levels):
            color = cmap(idx % 10)
            lw = 1.0 + 4.0 * level
            for arc in groups[level]:
                pt1 = self.points[arc[0]]
                pt2 = self.points[arc[1]]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=lw, alpha=0.85, zorder=2)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=lw, label=f"x = {level:.3f}  ({len(groups[level])} arcos)")
            )

        # Etiquetas de nodos
        for i, pt in enumerate(self.points):
            if i in hull_set:
                ax.annotate(
                    str(i), (pt[0], pt[1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=10, color="red", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
            else:
                ax.annotate(
                    str(i), (pt[0], pt[1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
                )

        if legend_handles:
            ax.legend(handles=legend_handles, loc="best", fontsize=8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.show()

    def dump_vars_csv(self, filepath: str | Path) -> None:
        """Write solution variable values to a CSV file.

        Columns: var_type, idx_0, idx_1, value
          - x, f, z, zp  : idx_0 = arc source, idx_1 = arc destination
          - y, yp         : idx_0 = triangle index, idx_1 = ""
          - u             : idx_0 = node index,     idx_1 = ""
          - f_mcf         : idx_0 = commodity,      idx_1 = "src_dst"

        Only variables with |X| > 1e-9 are written.
        Rows are sorted by (var_type, idx_0, idx_1).
        Must be called after solve() when model.SolCount > 0.
        """
        if self.model.SolCount == 0:
            raise RuntimeError("dump_vars_csv: no feasible solution available (SolCount == 0)")

        path = Path(filepath)
        parent = path.parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []

        def _collect(var_type: str, idx_0: Any, idx_1: Any, var: gp.Var) -> None:
            if abs(var.X) < 1e-9:
                return
            rows.append({"var_type": var_type, "idx_0": idx_0, "idx_1": idx_1, "value": round(var.X, 8)})

        for (i, j), v in self.x.items():
            _collect("x", i, j, v)
        for t, v in self.y.items():
            _collect("y", t, "", v)
        for t, v in self.yp.items():
            _collect("yp", t, "", v)
        for (i, j), v in self.f.items():
            _collect("f", i, j, v)
        for n, v in self.u.items():
            _collect("u", n, "", v)
        for (i, j), v in self.z.items():
            _collect("z", i, j, v)
        for (i, j), v in self.zp.items():
            _collect("zp", i, j, v)
        for (k, i, j), v in self.f_mcf.items():
            _collect("f_mcf", k, f"{i}_{j}", v)

        rows.sort(key=lambda r: (r["var_type"], r["idx_0"], str(r["idx_1"])))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["var_type", "idx_0", "idx_1", "value"])
            writer.writeheader()
            writer.writerows(rows)

    def __str__(self) -> str:
        """Define lo que se muestra al hacer print() de la instancia de la clase."""

        # 1. Si el modelo aún no se ha construido o resuelto, evitamos errores
        if self.model.NumVars == 0:
            return "OAPCompactModel (Estado: No construido)"
        if not hasattr(self, "x_results") or not self.x_results:
            return "Modelo sin solución factible para mostrar."
        lp, gap, ip, elapsed_time, nodes = self.get_model_stats()
        tour = self.get_tour()

        # 3. Construimos el texto línea por línea
        lineas = [
            "-" * 30,
            "--- Valores del modelo matemático compacto ---",
            "-" * 30,
            f"Columnas modelo original: {self.model.NumVars}",
            f"Filas modelo original: {self.model.NumConstrs}",
            f"Area de la envolvente convexa: {self.convex_hull_area:.4f}",
            "",
            "-" * 30,
            "--- Valores del modelo IP y Relajado ---",
            "-" * 30,
            f"Instance: {self.model.ModelName}",
            f"IP Objective Value: {ip:.2f}",
            f"LP Objective Value: {lp:.2f}",
            f"Optimality Gap: {gap:.2f}%",
            f"Elapsed Time: {elapsed_time:.2f} seconds",
            f"Number of Nodes Explored: {nodes}",
            "",
            "-" * 30,
            "--- Tour obtenido ---",
            str(tour),
            "-" * 30,
            "",
            f"Resultados: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}",
        ]

        return "\n".join(lineas)

    def add_restricciones_semiplano(self, version: int = 1) -> None:
        """Agrega restricciones de semiplano (V1 o V2) al modelo."""
        # Filtrar índices que no están en la envolvente convexa
        A_pp = [i for i in range(self.N) if i not in self.CH]
        constrains = []

        if version == 1:
            for i in A_pp:
                for j in self.CH:
                    if (i, j) not in self.x:
                        continue

                    semiplano_izquierdo_limpio = True
                    for k in A_pp:
                        if k == i:
                            continue

                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]

                        D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                        if D > 0:
                            semiplano_izquierdo_limpio = False
                            break

                    if semiplano_izquierdo_limpio:
                        index = np.where(self.CH == j)[0][0]
                        j_siguiente = self.CH[(index + 1) % len(self.CH)]

                        if (j, j_siguiente) in self.x:
                            constrains.append(
                                self.model.addConstr(self.x[i, j] <= self.x[j, j_siguiente], name=f"semiplano_{i}_{j}")
                            )
            print(f"Añadidas {len(constrains)} restricciones de semiplano V1.")

        elif version == 2:
            for i in A_pp:
                for j in self.CH:
                    if (i, j) not in self.x:
                        continue

                    S_left = []
                    for k in A_pp:
                        if k == i:
                            continue
                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]
                        D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                        if D_k > 0:
                            S_left.append(k)

                    idx_j = np.where(self.CH == j)[0][0]
                    j_siguiente = self.CH[(idx_j + 1) % len(self.CH)]

                    if len(S_left) == 0:
                        nodo_actual_ch = j
                        for step in range(1, len(self.CH)):
                            idx_siguiente = (idx_j + step) % len(self.CH)
                            nodo_siguiente_ch = self.CH[idx_siguiente]

                            x_sig, y_sig = self.points[nodo_siguiente_ch]
                            D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)

                            if D_sig > 0:
                                if (nodo_actual_ch, nodo_siguiente_ch) in self.x:
                                    constrains.append(
                                        self.model.addConstr(
                                            self.x[i, j] <= self.x[nodo_actual_ch, nodo_siguiente_ch],
                                            name=f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual_ch}_{nodo_siguiente_ch}",
                                        )
                                    )
                                nodo_actual_ch = nodo_siguiente_ch
                            else:
                                break
                    else:
                        expr_escape = gp.LinExpr()
                        if (j, j_siguiente) in self.x:
                            expr_escape.addTerms(1.0, self.x[j, j_siguiente])

                        for k in S_left:
                            if (j, k) in self.x:
                                expr_escape.addTerms(1.0, self.x[j, k])

                        constrains.append(
                            self.model.addConstr(
                                self.x[i, j] <= expr_escape, name=f"bolsillo_{i}_{j}_soporta_{len(S_left)}_puntos"
                            )
                        )
            print(f"Añadidas {len(constrains)} restricciones de semiplano V2.")

        elif version == 3:
            for i in A_pp:
                for j in self.CH:
                    # Validamos si existe la arista en alguna dirección antes de calcular nada
                    if (i, j) not in self.x and (j, i) not in self.x:
                        continue

                    # 1. Clasificamos los puntos interiores respecto al vector i -> j
                    S_left = []
                    S_right = []

                    for k in A_pp:
                        if k == i:
                            continue
                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]

                        D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                        if D_k > 0:
                            S_left.append(k)
                        elif D_k < 0:
                            S_right.append(k)

                    idx_j = np.where(self.CH == j)[0][0]
                    j_siguiente = self.CH[(idx_j + 1) % len(self.CH)]
                    j_anterior = self.CH[(idx_j - 1) % len(self.CH)]

                    # --- LÓGICA FORWARD: Del interior (i) a la Envolvente (j) ---
                    if (i, j) in self.x:
                        if len(S_left) == 0:
                            # Cascada hacia adelante (Counter-Clockwise)
                            nodo_actual_ch = j
                            for step in range(1, len(self.CH)):
                                idx_siguiente = (idx_j + step) % len(self.CH)
                                nodo_siguiente_ch = self.CH[idx_siguiente]

                                x_sig, y_sig = self.points[nodo_siguiente_ch]
                                D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)

                                if D_sig > 0:
                                    if (nodo_actual_ch, nodo_siguiente_ch) in self.x:
                                        constrains.append(
                                            self.model.addConstr(
                                                self.x[i, j] <= self.x[nodo_actual_ch, nodo_siguiente_ch],
                                                name=f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual_ch}_{nodo_siguiente_ch}_forwards",
                                            )
                                        )
                                    nodo_actual_ch = nodo_siguiente_ch
                                else:
                                    break
                        else:
                            # Restricción de Bolsillo Forward (Escape)
                            expr_escape = gp.LinExpr()
                            if (j, j_siguiente) in self.x:
                                expr_escape.addTerms(1.0, self.x[j, j_siguiente])

                            for k in S_left:
                                if (j, k) in self.x:
                                    expr_escape.addTerms(1.0, self.x[j, k])

                            constrains.append(
                                self.model.addConstr(
                                    self.x[i, j] <= expr_escape,
                                    name=f"bolsillo_forward_{i}_{j}_soporta_{len(S_left)}_puntos",
                                )
                            )

                    # --- LÓGICA BACKWARD: De la Envolvente (j) al interior (i) ---
                    if (j, i) in self.x:
                        if len(S_right) == 0:
                            # Cascada hacia atrás (Clockwise)
                            nodo_actual_ch = j
                            for step in range(1, len(self.CH)):
                                idx_anterior_step = (idx_j - step) % len(self.CH)
                                nodo_anterior_ch = self.CH[idx_anterior_step]

                                x_ant, y_ant = self.points[nodo_anterior_ch]
                                D_ant = (x_j - x_i) * (y_ant - y_i) - (y_j - y_i) * (x_ant - x_i)

                                # Corregido: D_ant < 0 corresponde al lado S_right
                                if D_ant < 0:
                                    if (nodo_anterior_ch, nodo_actual_ch) in self.x:
                                        constrains.append(
                                            self.model.addConstr(
                                                self.x[j, i] <= self.x[nodo_anterior_ch, nodo_actual_ch],
                                                name=f"semiplano_cadena_{j}_{i}_fuerza_{nodo_anterior_ch}_{nodo_actual_ch}_backwards",
                                            )
                                        )
                                    nodo_actual_ch = nodo_anterior_ch
                                else:
                                    break
                        else:
                            # Restricción de Bolsillo Backward (Entrada)
                            expr_escape_back = gp.LinExpr()
                            if (j_anterior, j) in self.x:
                                expr_escape_back.addTerms(1.0, self.x[j_anterior, j])

                            for k in S_right:
                                if (k, j) in self.x:
                                    expr_escape_back.addTerms(1.0, self.x[k, j])

                            constrains.append(
                                self.model.addConstr(
                                    self.x[j, i] <= expr_escape_back,
                                    name=f"bolsillo_backward_{j}_{i}_soporta_{len(S_right)}_puntos",
                                )
                            )

            print(f"Añadidas {len(constrains)} restricciones de semiplano V3.")

    def inyectar_cortes_knapsack_locales(self) -> None:
        """Inyecta restricciones de mochila que limitan la contribución fraccionaria."""
        cortes_añadidos = 0
        for i in range(self.N):
            max_beneficio_real = 0.0

            for j1 in range(self.N):
                if j1 == i or (i, j1) not in self.x:
                    continue

                for j2 in range(j1 + 1, self.N):
                    if j2 == i or (i, j2) not in self.x:
                        continue

                    es_pareja_legal = True
                    for k in range(self.N):
                        if k in [i, j1, j2]:
                            continue
                        if point_in_triangle(self.points[k], self.points[j1], self.points[i], self.points[j2]):
                            es_pareja_legal = False
                            break

                    if es_pareja_legal:
                        beneficio_pareja = self.c.get((i, j1), 0) + self.c.get((i, j2), 0)
                        if beneficio_pareja > max_beneficio_real:
                            max_beneficio_real = beneficio_pareja

            expr_knapsack = gp.LinExpr()
            for j in range(self.N):
                if j != i and (i, j) in self.x:
                    expr_knapsack.addTerms(self.c.get((i, j), 0), self.x[i, j])

            if expr_knapsack.size() > 0:
                self.model.addConstr(expr_knapsack <= max_beneficio_real, name=f"knapsack_local_{i}")
                cortes_añadidos += 1

        print(f"Inyectados {cortes_añadidos} Cortes Knapsack Locales.")

    def inyectar_cliques_de_cruce(self) -> None:
        """Busca grupos de arcos que se cruzan TODOS entre sí (Cliques)."""
        print("Construyendo grafo de intersecciones para Cliques...")
        aristas = [(i, j) for (i, j) in self.x.keys() if i < j]

        G_cruces = nx.Graph()
        G_cruces.add_nodes_from(aristas)

        for idx, e1 in enumerate(aristas):
            for e2 in aristas[idx + 1 :]:
                if e1[0] in e2 or e1[1] in e2:
                    continue

                p1, p2 = self.points[e1[0]], self.points[e1[1]]
                p3, p4 = self.points[e2[0]], self.points[e2[1]]

                if segments_intersect(p1, p2, p3, p4):
                    G_cruces.add_edge(e1, e2)

        cliques = list(nx.find_cliques(G_cruces))
        cortes_añadidos = 0

        for clique in cliques:
            if len(clique) >= 3:
                expr_clique = gp.LinExpr()
                for e in clique:
                    if e in self.x:
                        expr_clique.addTerms(1.0, self.x[e[0], e[1]])
                    if (e[1], e[0]) in self.x:
                        expr_clique.addTerms(1.0, self.x[e[1], e[0]])

                self.model.addConstr(expr_clique <= 1, name=f"clique_cruce_{cortes_añadidos}")
                cortes_añadidos += 1

        print(f"Inyectados {cortes_añadidos} Cortes de Clique de Cruces.")

    # ------------------------------------------------------------------
    # Nuevas familias de desigualdades sobre triángulos
    # ------------------------------------------------------------------

    def _add_triangle_clique_constraints(self) -> None:
        """Cliques en I₃: conjuntos de triángulos internos mutuamente incompatibles.

        Para cada clique maximal K en el grafo de incompatibilidades de triángulos:
            Σ_{t ∈ K} y_t ≤ 1
        """
        pairs = incompatible_triangles(self.triangles, self.points)
        if len(pairs) == 0:
            return
        G_i3: nx.Graph[int] = nx.Graph()
        G_i3.add_nodes_from(self.V_list)
        for row in pairs:
            G_i3.add_edge(int(row[0]), int(row[1]))
        cliques = list(nx.find_cliques(G_i3))
        n_added = 0
        for clique in cliques:
            if len(clique) >= 2:
                self.model.addConstr(
                    gp.quicksum(self.y[t] for t in clique) <= 1,
                    name=f"i3_clique_{n_added}",
                )
                n_added += 1
        print(f"Añadidas {n_added} restricciones de clique I₃.")

    def _add_arc_triangle_link_constraints(self) -> None:
        """Restricciones H_ij: traducción directa de la constraint MSD (28).

        Para cada {i,j} ∈ E" y cada par {s,s'} ∈ H_ij (celdas izquierda/derecha
        de cada sub-segmento del arreglo sobre {i,j}):
            Σ_{t⊇s}  y_t − Σ_{t⊇s'} y_t  ≤  x_ij + x_ji
            Σ_{t⊇s'} y_t − Σ_{t⊇s}  y_t  ≤  x_ij + x_ji
        """
        e_double_prime = [
            (i, j)
            for (i, j) in self.x.keys()
            if i < j and (i not in self.CH or j not in self.CH)
        ]
        hij_data = compute_hij_data(self.points, self.triangles, e_double_prime)

        n_added = 0
        for (i, j), sub_segs in hij_data.items():
            x_rhs = gp.LinExpr()
            x_rhs.addTerms(1.0, self.x[i, j])
            if (j, i) in self.x:
                x_rhs.addTerms(1.0, self.x[j, i])

            for k, (left_tris, right_tris) in enumerate(sub_segs):
                left_expr = gp.LinExpr()
                for t in left_tris:
                    left_expr.addTerms(1.0, self.y[t])
                right_expr = gp.LinExpr()
                for t in right_tris:
                    right_expr.addTerms(1.0, self.y[t])
                self.model.addConstr(
                    left_expr - right_expr <= x_rhs,
                    name=f"hij_{i}_{j}_{k}_p",
                )
                self.model.addConstr(
                    right_expr - left_expr <= x_rhs,
                    name=f"hij_{i}_{j}_{k}_m",
                )
                n_added += 2

        print(f"Añadidas {n_added} restricciones H_ij arco-triángulo.")
