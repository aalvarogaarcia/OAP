import os
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class OAPStatsMixin:
    """
    Mixin para la extracción de estadísticas, cálculo geométrico de áreas
    y exportación de resultados del modelo OAP. Compatible con Compacto y Benders.
    """

    model: gp.Model
    points: np.ndarray
    N: int
    convex_hull_area: float
    x_results: list[tuple[int, int]]
    x: dict[tuple[int, int], gp.Var]

    def solve_lp_relaxation(self, time_limit: int = ..., verbose: bool = ...) -> None: ...

    def get_tour(self) -> list[int]:
        if not hasattr(self, "x_results") or not self.x_results:
            return []

        next_i = {i: j for i, j in self.x_results}
        start = self.x_results[0][0]
        tour = [start]
        actual = next_i.get(start)

        while actual is not None and actual != start:
            tour.append(actual)
            actual = next_i.get(actual)

        return tour

    def get_objval_int(self) -> float | None:
        """Calcula el área real del polígono usando la fórmula de Gauss (Shoelace)."""
        # Si el modelo no es MIP (ej: se resolvió la relajación LP), no hay valor entero
        if not self.model.IsMIP:
            return None

        if not hasattr(self, "x_results") or not self.x_results:
            return None

        obj_val = 0.0
        for i, j in self.x_results:
            pi = self.points[i]
            pj = self.points[j]
            obj_val += (pi[0] * pj[1] - pj[0] * pi[1]) / 2.0

        return abs(obj_val)

    def _shoelace_from_x_vars(self, x_dict: dict[str, Any]) -> float:
        """Calcula el área de Shoelace a partir de un dict {(i,j): valor_fraccionario}."""
        obj_val = 0.0
        for (i, j), val in x_dict.items():  # type: ignore[str-unpack]
            pi = self.points[i]
            pj = self.points[j]
            obj_val += (pi[0] * pj[1] - pj[0] * pi[1]) / 2.0 * val
        return abs(obj_val)

    def _compute_lp_tk_bound(self) -> float | str:
        """LP bound for a Compact MIP with T_k cuts, via a cutting-plane loop.

        Creates a fresh LP relaxation (``model.relax()``), then iterates:
        solve LP -> separate T_k inequalities -> add cuts -> repeat until no
        new violated cuts are found.  The result is cached in ``self.lp_objval``
        so subsequent calls to ``get_objval_lp()`` return immediately.

        Called automatically by ``get_objval_lp()`` when ``_use_tk_cuts`` is
        True and the model is still a MIP (i.e. the full B&C was run, not the
        pure LP relaxation path).
        """
        # Local import to avoid circular-import issues at module load time.
        from utils.constraints import separate_tk_cuts  # type: ignore[import]

        N: int = getattr(self, "N", len(self.points))
        if N < 3:
            return "-"

        lp: gp.Model = self.model.relax()
        lp.Params.OutputFlag = 0
        lp.update()

        # Build (i, j) -> lp_var mapping by variable name ("x_i_j").
        x_lp: dict[tuple[int, int], gp.Var] = {}
        for v in lp.getVars():
            name = v.VarName
            if name.startswith("x_"):
                parts = name.split("_")
                if len(parts) >= 3:
                    try:
                        x_lp[(int(parts[1]), int(parts[2]))] = v
                    except ValueError:
                        pass

        if not x_lp:
            return "-"

        iteration = 0
        while True:
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                self.lp_objval: float | str = "-"
                return "-"

            x_sol = {arc: var.X for arc, var in x_lp.items() if var.X > 1e-9}
            cuts = separate_tk_cuts(x_sol, N, threshold=1e-3)
            if not cuts:
                break  # Converged: no violated T_k cut found.

            for cut in cuts:
                S_list = list(cut.S)
                rhs = len(S_list)
                lhs = gp.LinExpr()
                for i in S_list:
                    for j in S_list:
                        if i != j and (i, j) in x_lp:
                            lhs.addTerms(1.0, x_lp[i, j])
                if (cut.p, cut.w) in x_lp:
                    lhs.addTerms(1.0, x_lp[cut.p, cut.w])
                if (cut.w, cut.q) in x_lp:
                    lhs.addTerms(1.0, x_lp[cut.w, cut.q])
                if (cut.p, cut.q) in x_lp:
                    lhs.addTerms(1.0, x_lp[cut.p, cut.q])
                lp.addConstr(
                    lhs <= rhs,
                    name=f"tk_lp_stat_{iteration}_{cut.p}_{cut.w}_{cut.q}",
                )
            lp.update()
            iteration += 1

        x_vals: dict[tuple[int, int], float] = {arc: var.X for arc, var in x_lp.items()}
        result: float | str = self._shoelace_from_x_vars(x_vals)  # type: ignore[arg-type]
        self.lp_objval = result  # Cache so future calls hit case 1 directly.
        return result

    def get_objval_lp(self) -> float | str:
        """
        Recupera o calcula la Relajación Lineal de manera segura dependiendo de la arquitectura.
        El valor LP se obtiene siempre a partir de la familia de variables x (fórmula de Shoelace).
        """
        # 1. Si ya calculamos el LP explícitamente y lo guardamos
        if hasattr(self, "lp_objval") and self.lp_objval is not None:
            return self.lp_objval  # type: ignore[no-any-return]

        # 2. Si el modelo actual ya es continuo (porque llamamos a solve_lp_relaxation)
        if not self.model.IsMIP:
            if self.model.SolCount == 0:
                return "-"
            if not getattr(self, "_lp_converged", True):
                return "-"  # LP bound unreliable: degenerate PI cuts or MAX_LP_ITER reached
            x_vals = {k: v.X for k, v in self.x.items()}
            return self._shoelace_from_x_vars(x_vals)  # type: ignore[arg-type]

        # 3. Si es Benders MIP y no hemos calculado el LP previamente
        if hasattr(self, "benders_method"):
            self.solve_lp_relaxation()
            if self.model.SolCount == 0:
                return "-"
            if not getattr(self, "_lp_converged", True):
                return "-"  # LP bound unreliable: degenerate PI cuts or MAX_LP_ITER reached
            x_vals = {k: v.X for k, v in self.x.items()}
            return self._shoelace_from_x_vars(x_vals)  # type: ignore[arg-type]

        # 4. Si es Compacto MIP — resolver relajación y leer x.
        #    When T_k cuts are active, run the cutting-plane loop on a fresh
        #    model.relax() copy so the bound reflects the tightened LP.
        if getattr(self, "_use_tk_cuts", False):
            return self._compute_lp_tk_bound()

        # Classic case: plain LP relaxation without T_k cuts.
        lp = self.model.relax()
        lp.Params.OutputFlag = 0
        lp.optimize()

        os.makedirs("outputs/Others", exist_ok=True)
        try:
            lp.write("outputs/Others/LP_Relaxation_Converged_Compact.sol")
            lp.write("outputs/Others/LP_Relaxation_Converged_Compact.lp")
        except gp.GurobiError:
            pass  # Falla silenciosamente si no hay solución

        if lp.SolCount == 0:
            return "-"

        # Calcular Shoelace desde las x de la relajación
        x_vals = {}
        for v in lp.getVars():
            if v.VarName.startswith("x_"):
                parts = v.VarName.split("_")
                i, j = int(parts[1]), int(parts[2])
                x_vals[(i, j)] = v.X
        return self._shoelace_from_x_vars(x_vals)  # type: ignore[arg-type]

    def get_model_stats(self) -> tuple[Any, ...]:
        """Extrae estadísticas clave: (LP_Val, Gap, IP_Val, Time, Nodes)."""
        if self.model.SolCount == 0:
            return "-", "-", "-", "-", "-"

        time_s = self.model.Runtime
        nodes = self.model.NodeCount if self.model.IsMIP else 0

        # Si resolvimos puramente el LP (ej. solve_lp_relaxation)
        if not self.model.IsMIP:
            lp_val = self.get_objval_lp()
            gap = 0.0 if isinstance(lp_val, (int, float)) else "-"
            return lp_val, gap, "-", time_s, nodes

        # Si es MIP (Compacto o Benders entero)
        ip_val = self.get_objval_int()
        lp_val = self.get_objval_lp()

        if ip_val is None:
            return lp_val, "-", "-", time_s, nodes

        # Cálculo de Gap personalizado para OAP (Protegido contra strings "-")
        gap = "-"
        if ip_val != 0 and isinstance(lp_val, (int, float)):
            if self.model.ModelSense == GRB.MINIMIZE:
                gap = (ip_val - lp_val) / ip_val * 100
            elif self.model.ModelSense == GRB.MAXIMIZE:
                if (self.convex_hull_area - ip_val) != 0:
                    gap = ((lp_val - ip_val) / (self.convex_hull_area - ip_val)) * 100

        return lp_val, gap, ip_val, time_s, nodes

    def __str__(self) -> str:
        if self.model.NumVars == 0:
            return "Modelo no construido."

        lp, gap, ip, elapsed_time, nodes = self.get_model_stats()

        # Formateo seguro para mezclar floats con guiones "-"
        lp_str = f"{lp:.2f}" if isinstance(lp, (int, float)) else str(lp)
        ip_str = f"{ip:.2f}" if isinstance(ip, (int, float)) else str(ip)
        gap_str = f"{gap:.2f}%" if isinstance(gap, (int, float)) else str(gap)

        tour = self.get_tour()

        str_format = f"""{"-" * 30}
--- Valores del modelo matemático ({"Benders" if hasattr(self, "benders_method") else "Compacto"}) ---
{"-" * 30}
Columnas modelo original: {self.model.NumVars}
Filas modelo original: {self.model.NumConstrs}
Area de la envolvente convexa: {self.convex_hull_area}

{"-" * 30}
--- Valores del modelo IP y Relajado ---
{"-" * 30}
Instance: {self.model.ModelName}
IP Objective Value: {ip_str}
LP Objective Value: {lp_str}
Optimality Gap: {gap_str}
Elapsed Time: {elapsed_time:.2f} seconds
Number of Nodes Explored: {nodes}

{"-" * 30}
--- Tour obtenido ---
{tour}
{"-" * 30}
Resultados: LP={lp_str}, Gap={gap_str}, IP={ip_str}, Time={elapsed_time:.2f}s, Nodes={nodes}\n\n\n"""
        return str_format
