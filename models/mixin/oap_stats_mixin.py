import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os

class OAPStatsMixin:
    """
    Mixin para la extracción de estadísticas, cálculo geométrico de áreas 
    y exportación de resultados del modelo OAP. Compatible con Compacto y Benders.
    """
    model: gp.Model
    points: np.ndarray
    convex_hull_area: float
    x_results: list[tuple[int, int]]

    def get_tour(self) -> list[int]:
        if not hasattr(self, 'x_results') or not self.x_results:
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

        if not hasattr(self, 'x_results') or not self.x_results:
            return None
            
        obj_val = 0.0
        for i, j in self.x_results:
            pi = self.points[i]
            pj = self.points[j]
            obj_val += (pi[0] * pj[1] - pj[0] * pi[1]) / 2.0
            
        return abs(obj_val)

    def get_objval_lp(self) -> float | str:
        """
        Recupera o calcula la Relajación Lineal de manera segura dependiendo de la arquitectura.
        """
        # 1. Si ya calculamos el LP explícitamente y lo guardamos
        if hasattr(self, 'lp_objval') and self.lp_objval is not None:
            return self.lp_objval
            
        # 2. Si el modelo actual ya es continuo (porque llamamos a solve_lp_relaxation)
        if not self.model.IsMIP:
            return self.model.ObjVal if self.model.SolCount > 0 else "-"
            
        # 3. Si es Benders MIP y no hemos calculado el LP previamente
        if hasattr(self, 'benders_method'):
            # No podemos usar .relax() porque destruiría los callbacks.
            # Avisamos de que el LP debe calcularse explícitamente primero.
            return "-"

        # 4. Si es Compacto MIP (Comportamiento Clásico)
        lp = self.model.relax()
        lp.Params.OutputFlag = 0 
        lp.optimize()

        os.makedirs("outputs/Others", exist_ok=True)
        try:
            lp.write("outputs/Others/LP_Relaxation_Converged_Compact.sol")
            lp.write("outputs/Others/LP_Relaxation_Converged_Compact.lp")
        except gp.GurobiError:
            pass # Falla silenciosamente si no hay solución
        
        return lp.ObjVal if lp.SolCount > 0 else "-"

    def get_model_stats(self) -> tuple:
        """Extrae estadísticas clave: (LP_Val, Gap, IP_Val, Time, Nodes)."""
        if self.model.SolCount == 0:
            return "-", "-", "-", "-", "-"

        time_s = self.model.Runtime
        nodes = self.model.NodeCount if self.model.IsMIP else 0

        # Si resolvimos puramente el LP (ej. solve_lp_relaxation)
        if not self.model.IsMIP:
            lp_val = self.get_objval_lp()
            return lp_val, 0.0, "-", time_s, nodes

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

        return (
            f"-" * 30 + "\n"
            f"--- Valores del modelo matemático ---\n"
            f"-" * 30 + "\n"
            f"Columnas modelo original: {self.model.NumVars}\n"
            f"Filas modelo original: {self.model.NumConstrs}\n"
            f"Area de la envolvente convexa: {self.convex_hull_area}\n\n"
            f"-" * 30 + "\n"
            f"--- Valores del modelo IP y Relajado ---\n"
            f"-" * 30 + "\n"
            f"Instance: {self.model.ModelName}\n"
            f"IP Objective Value: {ip_str}\n"
            f"LP Objective Value: {lp_str}\n"
            f"Optimality Gap: {gap_str}\n"
            f"Elapsed Time: {elapsed_time:.2f} seconds\n"
            f"Number of Nodes Explored: {nodes}\n\n"
            f"-" * 30 + "\n"
            f"--- Tour obtenido ---\n"
            f"{tour}\n"
            f"-" * 30 + "\n"
            f"Resultados: LP={lp_str}, Gap={gap_str}, IP={ip_str}, Time={elapsed_time:.2f}s, Nodes={nodes}"
        )