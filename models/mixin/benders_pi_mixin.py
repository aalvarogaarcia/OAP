# models/mixin/benders_pi_mixin.py
import logging
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from models.typing_oap import IndexArray, TrianglesAdjList
from utils.utils import plot_strengthening_constraints, signed_area

# Instanciamos el logger para este módulo
logger = logging.getLogger(__name__)

Arc = tuple[int, int]
RayComponents = dict[str, dict[Arc, float] | float]


class BendersPiMixin:
    """
    Mixin para construir los subproblemas de factibilidad de Benders (Fase 1)
    y extraer los cortes utilizando variables duales (.Pi) y variables artificiales.
    """

    # --- Pistas de Tipado para el linter ---
    model: gp.Model
    sub_y: gp.Model
    sub_yp: gp.Model
    x: dict[Arc, gp.Var]
    N_list: range
    N: int
    CH: IndexArray
    points: np.ndarray
    V_list: range
    triangles_adj_list: TrianglesAdjList
    constrs_y: dict[str, Any]
    constrs_yp: dict[str, Any]
    iteration: int
    convex_hull_area: float
    triangles: IndexArray
    _abs_areas: list[float]
    y: gp.tupledict[int, gp.Var]
    yp: gp.tupledict[int, gp.Var]
    art_y: dict[str, dict[Any, gp.Var]]
    art_yp: dict[str, dict[Any, gp.Var]]
    objective: str

    def _compute_crossing_arc_pairs(self) -> set[tuple[Arc, Arc]]: ...  # type: ignore[empty-body]

    def build_pi_subproblems(
        self, sum_constrain: bool = True, strengthen: bool = False, plot_strengthen: bool = False
    ) -> None:
        """Orquestador principal para construir los Subproblemas con método Pi."""
        logger.info("Construyendo subproblemas con método Pi (Fase 1 con variables artificiales)...")

        self._configurar_parametros_pi()

        # 1. Preparar conjuntos de arcos necesarios
        A_prime, A_double_prime, A_double_prime_beta = self._obtener_conjuntos_arcos()

        # 2. Inicializar diccionarios de restricciones
        self.constrs_y.update({"alpha": {}, "beta": {}, "gamma": {}, "delta": {}})
        self.constrs_yp.update({"alpha_p": {}, "beta_p": {}, "gamma_p": {}, "delta_p": {}})

        # 3. Crear variables reales y artificiales
        self._crear_variables_pi(A_prime, A_double_prime, A_double_prime_beta, sum_constrain)

        # 3b. Pre-create R3 artificial variables so the objective includes them
        if strengthen:
            self._crear_variables_strengthening_pi()

        # 4. Establecer funciones objetivo (Minimizar artificiales)
        self._establecer_objetivos_pi()

        # 5. Añadir las restricciones de Fase 1
        self._añadir_restricciones_pi(A_prime, A_double_prime, A_double_prime_beta, sum_constrain)

        # 6. Añadir restricciones de fortalecimiento
        if strengthen:
            self._add_strengthening_pi(plot=plot_strengthen)

        self.sub_y.update()
        self.sub_yp.update()

    def _crear_variables_strengthening_pi(self) -> None:
        """Pre-create artificial variables for R3/R4 crossing constraints (Pi mode).

        Must be called BEFORE _establecer_objetivos_pi() so the new artificials
        are included in the Phase-1 minimisation objective.
        """
        crossing_pairs = self._compute_crossing_arc_pairs()
        r3_keys = [
            (i, j, k, l)
            for (i, j) in self.x.keys()
            if self.triangles_adj_list[i][j]
            for (k, l) in self.x.keys()
            if (i, j) != (k, l) and ((i, j), (k, l)) in crossing_pairs and self.triangles_adj_list[k][l]
        ]
        self._r3_keys: list[tuple[int, int, int, int]] = r3_keys
        if r3_keys:
            self.art_y["r3_p"] = self.sub_y.addVars(r3_keys, lb=0, name="art_y_r3_p")
            self.art_y["r3_n"] = self.sub_y.addVars(r3_keys, lb=0, name="art_y_r3_n")
            self.art_yp["r3_p"] = self.sub_yp.addVars(r3_keys, lb=0, name="art_yp_r3_p")
            self.art_yp["r3_n"] = self.sub_yp.addVars(r3_keys, lb=0, name="art_yp_r3_n")

    def _add_strengthening_pi(self, plot: bool = False) -> None:
        """Add LP-strengthening constraints to Pi subproblems.

        R1 and R2 are plain ≤ (always feasible, no artificials needed).
        R3/R4 use the artificial variables created by _crear_variables_strengthening_pi.
        """
        if not hasattr(self, "_abs_areas"):
            self._abs_areas = [
                abs(signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]))
                for tri in self.triangles
            ]
        at = self._abs_areas

        # R1: area balance (plain ≤, fixed RHS)
        self.constrs_y["r1"] = self.sub_y.addConstr(
            gp.quicksum(at[t] * self.y[t] for t in self.V_list) <= self.convex_hull_area,
            name="r1_area_balance_y",
        )
        self.constrs_yp["r1_p"] = self.sub_yp.addConstr(
            gp.quicksum(at[t] * self.yp[t] for t in self.V_list) <= self.convex_hull_area,
            name="r1_area_balance_yp",
        )

        self.constrs_y["r2"] = {}
        self.constrs_yp["r2_p"] = {}
        self.constrs_y["r3"] = {}
        self.constrs_yp["r3_p"] = {}

        for i, j in self.x.keys():
            tris = self.triangles_adj_list[i][j]
            if not tris:
                continue

            # R2: arc coverage (plain ≤, fixed RHS = 1)
            self.constrs_y["r2"][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in tris) <= 1,
                name=f"r2_arc_coverage_y_{i}_{j}",
            )
            self.constrs_yp["r2_p"][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in tris) <= 1,
                name=f"r2_arc_coverage_yp_{i}_{j}",
            )

        # R3/R4: crossing exclusion (with artificials for Phase-1 feasibility)
        art_r3_p_y: dict[Any, gp.Var] = self.art_y.get("r3_p", {})
        art_r3_n_y: dict[Any, gp.Var] = self.art_y.get("r3_n", {})
        art_r3_p_yp: dict[Any, gp.Var] = self.art_yp.get("r3_p", {})
        art_r3_n_yp: dict[Any, gp.Var] = self.art_yp.get("r3_n", {})

        for key in getattr(self, "_r3_keys", []):
            i, j, k, l = key
            tris = self.triangles_adj_list[i][j]
            tris_kl = self.triangles_adj_list[k][l]

            self.constrs_y["r3"][key] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in tris)
                + gp.quicksum(self.y[t] for t in tris_kl)
                + art_r3_p_y[key]
                - art_r3_n_y[key]
                <= 1,
                name=f"r3_crossing_y_{i}_{j}_{k}_{l}",
            )
            self.constrs_yp["r3_p"][key] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in tris)
                + gp.quicksum(self.yp[t] for t in tris_kl)
                + art_r3_p_yp[key]
                - art_r3_n_yp[key]
                <= 1,
                name=f"r3_crossing_yp_{i}_{j}_{k}_{l}",
            )

        if plot:
            crossing_pairs = self._compute_crossing_arc_pairs()
            plot_strengthening_constraints(
                points=self.points,
                ch=self.CH,
                x_keys=list(self.x.keys()),
                crossing_pairs=crossing_pairs,
            )

    def get_pi_cut_y(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de Benders usando variables duales (.Pi) del subproblema Y."""
        cut_y_expr = gp.LinExpr()
        cut_y_val = 0.0
        v_comps: RayComponents = {"alpha": {}, "beta": {}, "gamma": {}, "delta": {}}

        for (i, j), constr in self.constrs_y["alpha"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps["alpha"][i, j] = pi
                cut_y_expr += pi * self.x[i, j]
                cut_y_val += pi * x_sol[i, j]

        for (i, j), constr in self.constrs_y["beta"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps["beta"][i, j] = pi
                cut_y_expr += pi * (self.x[i, j] - self.x[j, i])
                cut_y_val += pi * (x_sol[i, j] - x_sol[j, i])

        for (i, j), constr in self.constrs_y["gamma"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps["gamma"][i, j] = pi
                cut_y_expr += pi * self.x[i, j]
                cut_y_val += pi * x_sol[i, j]

        for (i, j), constr in self.constrs_y["delta"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps["delta"][i, j] = pi
                cut_y_expr += pi * (1 - self.x[j, i])
                cut_y_val += pi * (1 - x_sol[j, i])

        if "global" in self.constrs_y:
            pi_global = self.constrs_y["global"].Pi
            if abs(pi_global) > TOL:
                rhs_global = self.constrs_y["global"].RHS
                v_comps["global"] = pi_global
                cut_y_expr += pi_global * rhs_global
                cut_y_val += pi_global * rhs_global

        # R1: area balance (fixed RHS = CH_area)
        if "r1" in self.constrs_y:
            pi_r1 = self.constrs_y["r1"].Pi
            if abs(pi_r1) > TOL:
                cut_y_expr += pi_r1 * self.convex_hull_area
                cut_y_val += pi_r1 * self.convex_hull_area

        # R2: arc coverage (fixed RHS = 1)
        for (i, j), constr in self.constrs_y.get("r2", {}).items():
            pi_r2 = constr.Pi
            if abs(pi_r2) > TOL:
                cut_y_expr += pi_r2
                cut_y_val += pi_r2

        # R3: crossing exclusion — RHS = 1 - x[j,i] - x[k,l]
        for (i, j, k, l), constr in self.constrs_y.get("r3", {}).items():
            pi_r3 = constr.Pi
            if abs(pi_r3) > TOL:
                cut_y_expr += pi_r3 * (1 - self.x[j, i] - self.x[k, l])
                cut_y_val += pi_r3 * (1 - x_sol.get((j, i), 0.0) - x_sol.get((k, l), 0.0))

        if cut_y_val > TOL:
            sense = "<="
        elif cut_y_val < -TOL:
            sense = ">="
        else:
            sense = "=="
        self._log_and_print_pi(v_comps, cut_y_val, "Y", TOL, x_sol, cut_y_expr, sense)

        return cut_y_expr, cut_y_val

    def get_pi_cut_yp(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de Benders usando variables duales (.Pi) del subproblema YP."""
        cut_yp_expr = gp.LinExpr()
        cut_yp_val = 0.0
        v_comps_p: RayComponents = {"alpha_p": {}, "beta_p": {}, "gamma_p": {}, "delta_p": {}}

        for (i, j), constr in self.constrs_yp["alpha_p"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p["alpha_p"][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])

        for (i, j), constr in self.constrs_yp["beta_p"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p["beta_p"][i, j] = pi
                cut_yp_expr += pi * (self.x[j, i] - self.x[i, j])
                cut_yp_val += pi * (x_sol[j, i] - x_sol[i, j])

        for (i, j), constr in self.constrs_yp["gamma_p"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p["gamma_p"][i, j] = pi
                cut_yp_expr += pi * self.x[j, i]
                cut_yp_val += pi * x_sol[j, i]

        for (i, j), constr in self.constrs_yp["delta_p"].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p["delta_p"][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])

        if "global_p" in self.constrs_yp:
            pi_global_p = self.constrs_yp["global_p"].Pi
            if abs(pi_global_p) > TOL:
                rhs_global_p = self.constrs_yp["global_p"].RHS
                v_comps_p["global_p"] = pi_global_p
                cut_yp_expr += pi_global_p * rhs_global_p
                cut_yp_val += pi_global_p * rhs_global_p

        # R1_p: area balance (fixed RHS = CH_area)
        if "r1_p" in self.constrs_yp:
            pi_r1_p = self.constrs_yp["r1_p"].Pi
            if abs(pi_r1_p) > TOL:
                cut_yp_expr += pi_r1_p * self.convex_hull_area
                cut_yp_val += pi_r1_p * self.convex_hull_area

        # R2_p: arc coverage (fixed RHS = 1)
        for (i, j), constr in self.constrs_yp.get("r2_p", {}).items():
            pi_r2_p = constr.Pi
            if abs(pi_r2_p) > TOL:
                cut_yp_expr += pi_r2_p
                cut_yp_val += pi_r2_p

        # R4: crossing exclusion — RHS = 1 - x[i,j] - x[l,k]
        for (i, j, k, l), constr in self.constrs_yp.get("r3_p", {}).items():
            pi_r4 = constr.Pi
            if abs(pi_r4) > TOL:
                cut_yp_expr += pi_r4 * (1 - self.x[i, j] - self.x[l, k])
                cut_yp_val += pi_r4 * (1 - x_sol.get((i, j), 0.0) - x_sol.get((l, k), 0.0))

        if cut_yp_val > TOL:
            sense = "<="
        elif cut_yp_val < -TOL:
            sense = ">="
        else:
            sense = "=="

        self._log_and_print_pi(v_comps_p, cut_yp_val, "Y'", TOL, x_sol, cut_yp_expr, sense)
        return cut_yp_expr, cut_yp_val

    def _log_and_print_pi(self, v_components, cut_val, sub_name, TOL, x_sol, cut_expr, sense=None):
        """Método auxiliar interno para registrar el log de variables duales (.Pi).
        Diferencia automáticamente entre cortes de Optimalidad y Factibilidad.
        """
        verbose = getattr(self, "verbose", False)
        save_cuts = getattr(self, "save_cuts", False)

        # Determinar si el corte es de optimalidad basado en el nombre del subproblema
        is_optimality = sub_name == "Y_OPT"

        tipo_corte = "OPTIMALIDAD" if is_optimality else "FACTIBILIDAD"
        nombre_impreso = "Y" if is_optimality else sub_name

        # 1. Registro en la consola / archivo log estándar de Python
        if save_cuts and verbose:
            log_msg = [
                f"\n{'=' * 50}",
                f"CORTE DE {tipo_corte} (MÉTODO PI) DETECTADO EN SUBPROBLEMA {nombre_impreso}",
                f"Valor {'del coste real evaluado' if is_optimality else 'numérico de la violación'}: {cut_val:.6f}",
            ]

            for comp, values in v_components.items():
                if values:
                    log_msg.append(f"Componente {comp}:")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            log_msg.append(f"  {k}: {v:.4f}")
                    else:
                        log_msg.append(f"  Valor: {values:.4f}")

            log_msg.append(f"{'=' * 50}\n")
            logger.info("\n".join(log_msg))

        # 2. Registro JSON/Estructurado personalizado (para el Post-Mortem)
        if getattr(self, "save_cuts", False) and hasattr(self, "log_path"):
            try:
                # Nota: Uso log_benders_cut para mantener compatibilidad con tu código actual.
                # Si en el futuro la renombras a log_benders_cut en utils.py, actualízalo aquí.
                from utils.utils import log_benders_cut

                # Para el JSON, mantenemos la etiqueta Y_OPT para que el visualizador
                # (Post-Mortem) sepa que es de optimalidad.
                json_subproblem_type = "Y_OPT" if is_optimality else ("Y" if sub_name == "Y" else "Y_prime")

                log_benders_cut(
                    filepath=self.log_path,
                    iteration=getattr(self, "iteration", 0),
                    node_depth=0,
                    subproblem_type=json_subproblem_type,
                    x_sol=x_sol,
                    v_components=v_components,
                    cut_value=cut_val,
                    tolerance=TOL,
                    cut_expr=cut_expr,
                    sense=sense,
                )
            except ImportError:
                logger.warning("No se pudo guardar el log estructurado: log_benders_cut no está definido.")

    def _configurar_parametros_pi(self) -> None:
        """Configura los parámetros de Gurobi óptimos para el método de Fase 1."""
        for sub_model in [self.sub_y, self.sub_yp]:
            sub_model.Params.Presolve = 0
            sub_model.Params.InfUnbdInfo = 1

    def _obtener_conjuntos_arcos(self) -> tuple[list[Arc], list[Arc], list[Arc]]:
        """Identifica y clasifica los arcos según si pertenecen a la envolvente convexa o no."""
        A_prime = []
        for i in range(len(self.CH)):
            u, v = self.CH[i], self.CH[(i + 1) % len(self.CH)]
            if (u, v) in self.x:
                A_prime.append((u, v))

        A_double_prime = []
        for i in self.N_list:
            for j in self.N_list:
                if ((i not in self.CH) or (j not in self.CH)) and i != j:
                    A_double_prime.append((i, j))

        A_double_prime_beta = [(i, j) for (i, j) in A_double_prime if i < j]

        return A_prime, A_double_prime, A_double_prime_beta

    def _crear_variables_pi(
        self, A_prime: list[Arc], A_double_prime: list[Arc], A_double_prime_beta: list[Arc], sum_constrain: bool
    ) -> None:
        """Crea las variables 'y', 'yp' y todas las variables artificiales de holgura."""
        # Variables Reales
        self.y = self.sub_y.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.yp = self.sub_yp.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="yp")

        # Variables Artificiales encapsuladas en diccionarios para mantener el orden
        self.art_y = {
            "alpha_p": self.sub_y.addVars(A_prime, lb=0, name="art_y_alpha_p"),
            "alpha_n": self.sub_y.addVars(A_prime, lb=0, name="art_y_alpha_n"),
            "beta_p": self.sub_y.addVars(A_double_prime_beta, lb=0, name="art_y_beta_p"),
            "beta_n": self.sub_y.addVars(A_double_prime_beta, lb=0, name="art_y_beta_n"),
            "gamma_p": self.sub_y.addVars(A_double_prime, lb=0, name="art_y_gamma_p"),
            "gamma_n": self.sub_y.addVars(A_double_prime, lb=0, name="art_y_gamma_n"),
            "delta_p": self.sub_y.addVars(A_double_prime, lb=0, name="art_y_delta_p"),
            "delta_n": self.sub_y.addVars(A_double_prime, lb=0, name="art_y_delta_n"),
        }

        self.art_yp = {
            "alpha_p": self.sub_yp.addVars(A_prime, lb=0, name="art_yp_alpha_p"),
            "alpha_n": self.sub_yp.addVars(A_prime, lb=0, name="art_yp_alpha_n"),
            "beta_p": self.sub_yp.addVars(A_double_prime_beta, lb=0, name="art_yp_beta_p"),
            "beta_n": self.sub_yp.addVars(A_double_prime_beta, lb=0, name="art_yp_beta_n"),
            "gamma_p": self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_gamma_p"),
            "gamma_n": self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_gamma_n"),
            "delta_p": self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_delta_p"),
            "delta_n": self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_delta_n"),
        }

        if sum_constrain:
            self.art_y["global_p"] = self.sub_y.addVar(lb=0, name="art_y_global_p")
            self.art_y["global_n"] = self.sub_y.addVar(lb=0, name="art_y_global_n")
            self.art_yp["global_p"] = self.sub_yp.addVar(lb=0, name="art_yp_global_p")
            self.art_yp["global_n"] = self.sub_yp.addVar(lb=0, name="art_yp_global_n")

    def _establecer_objetivos_pi(self) -> None:
        """Define la función objetivo: Minimizar la suma de todas las variables artificiales."""
        obj_y = gp.LinExpr()
        for art_vars in self.art_y.values():
            if isinstance(art_vars, gp.tupledict):
                obj_y += art_vars.sum()
            else:
                obj_y += art_vars  # Para las variables individuales como 'global_p'

        obj_yp = gp.LinExpr()
        for art_vars in self.art_yp.values():
            if isinstance(art_vars, gp.tupledict):
                obj_yp += art_vars.sum()
            else:
                obj_yp += art_vars
        # Phase-1 ALWAYS minimises the sum of artificials.
        # The master's ModelSense (maximize for MaxArea) is irrelevant here —
        # inheriting it would make sub_y/sub_yp unbounded when maximize=True.
        self.sub_y.setObjective(obj_y, GRB.MINIMIZE)
        self.sub_yp.setObjective(obj_yp, GRB.MINIMIZE)

    def _añadir_restricciones_pi(
        self, A_prime: list[Arc], A_double_prime: list[Arc], A_double_prime_beta: list[Arc], sum_constrain: bool
    ) -> None:
        """Construye las ecuaciones matemáticas integrando las variables reales y las artificiales."""

        # --- Restricciones Globales ---
        if sum_constrain:
            self.constrs_y["global"] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.V_list) + self.art_y["global_p"] - self.art_y["global_n"]
                == self.N - 2,
                name="triangulos_internos_totales",
            )
            self.constrs_yp["global_p"] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.V_list) + self.art_yp["global_p"] - self.art_yp["global_n"]
                == self.N - len(self.CH),
                name="triangulos_externos_totales",
            )

        # --- A': Arcos dirigidos en la frontera de la CH ---
        for i, j in A_prime:
            self.constrs_y["alpha"][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j])
                + self.art_y["alpha_p"][i, j]
                - self.art_y["alpha_n"][i, j]
                == 0,
                name=f"alpha_{i}_{j}",
            )
            self.constrs_yp["alpha_p"][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j])
                + self.art_yp["alpha_p"][i, j]
                - self.art_yp["alpha_n"][i, j]
                == 0,
                name=f"alpha_p_{i}_{j}",
            )

        # --- E'' y A'': Arcos no pertenecientes a la CH ---
        for i, j in A_double_prime_beta:
            self.constrs_y["beta"][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j])
                - gp.quicksum(self.y[t] for t in self.triangles_adj_list[j][i])
                + self.art_y["beta_p"][i, j]
                - self.art_y["beta_n"][i, j]
                == 0,
                name=f"beta_{i}_{j}",
            )
            self.constrs_yp["beta_p"][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j])
                - gp.quicksum(self.yp[t] for t in self.triangles_adj_list[j][i])
                + self.art_yp["beta_p"][i, j]
                - self.art_yp["beta_n"][i, j]
                == 0,
                name=f"beta_p_{i}_{j}",
            )

        for i, j in A_double_prime:
            self.constrs_y["gamma"][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j])
                + self.art_y["gamma_p"][i, j]
                - self.art_y["gamma_n"][i, j]
                >= 0,
                name=f"gamma_{i}_{j}",
            )
            self.constrs_y["delta"][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j])
                + self.art_y["delta_p"][i, j]
                - self.art_y["delta_n"][i, j]
                <= 0,
                name=f"delta_{i}_{j}",
            )

            self.constrs_yp["gamma_p"][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j])
                + self.art_yp["gamma_p"][i, j]
                - self.art_yp["gamma_n"][i, j]
                >= 0,
                name=f"gamma_p_{i}_{j}",
            )
            self.constrs_yp["delta_p"][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j])
                + self.art_yp["delta_p"][i, j]
                - self.art_yp["delta_n"][i, j]
                <= 0,
                name=f"delta_p_{i}_{j}",
            )
