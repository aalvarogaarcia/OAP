# models/mixin/benders_optimize_mixin.py
import logging
import os
from typing import Any, Literal

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)

Arc = tuple[int, int]


class BendersOptimizeMixin:
    """
    Mixin encargado de la orquestación, actualización de RHS,
    el Callback de Gurobi y la resolución general del modelo.
    """

    # --- Attributes provided by sibling mixins / OAPBaseModel ---
    model: gp.Model
    N: int
    constrs_y: dict[str, dict[Arc, gp.Constr]]
    constrs_yp: dict[str, dict[Arc, gp.Constr]]
    iteration: int
    cortes_añadidos: int
    x: dict[Arc, gp.Var]
    sub_y: gp.Model
    sub_yp: gp.Model
    benders_method: str
    objective: Literal["Fekete", "Internal"]
    eta: gp.Var
    log_path: str
    poly_log_path: str
    _log_buffer: list[str]
    _cut_buffer: list[dict[str, Any]]
    x_results: list[Arc]
    polihedral: bool
    verbose: bool
    save_cuts: bool

    # --- Methods provided by sibling mixins (Farkas / Pi / CGSP / MW / DDMA / Analysis) ---
    def _detect_subtour_components(self, x_sol: dict[Arc, float]) -> list[set[int]]: ...  # type: ignore[empty-body]
    def get_farkas_cut_y(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr, float]: ...  # type: ignore[empty-body]
    def get_farkas_cut_yp(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr, float]: ...  # type: ignore[empty-body]
    def get_pi_cut_y(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr, float]: ...  # type: ignore[empty-body]
    def get_pi_cut_yp(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr, float]: ...  # type: ignore[empty-body]
    def get_optimality_cut_y(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr, float]: ...  # type: ignore[empty-body]
    def get_cgsp_cut_y(self, x_sol: dict[Arc, float], eta_sol: float = ..., TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def get_cgsp_cut_yp(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def get_mw_cut_y(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def get_mw_cut_yp(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def get_ddma_cut_y(self, x_sol: dict[Arc, float], eta_sol: float = ..., TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def get_ddma_cut_yp(self, x_sol: dict[Arc, float], TOL: float = ...) -> tuple[gp.LinExpr | None, float | None, dict[str, Any]]: ...  # type: ignore[empty-body]
    def log_facets(self, filepath: str, var_prefixes: list[str] | str = ..., verbose: bool = ...) -> None: ...

    def _update_subproblem_rhs(self, x_sol: dict[Arc, float]) -> None:
        """
        Actualiza los lados derechos (RHS) de los subproblemas Y y Y'
        usando la solución propuesta por el problema Maestro.
        """
        self.iteration += 1

        # ==========================================
        # ACTUALIZAR RHS PARA EL SUBPROBLEMA Y
        # ==========================================
        for (i, j), constr in self.constrs_y.get("alpha", {}).items():
            constr.RHS = x_sol[i, j]

        for (i, j), constr in self.constrs_y.get("beta", {}).items():
            constr.RHS = x_sol[i, j] - x_sol[j, i]

        for (i, j), constr in self.constrs_y.get("gamma", {}).items():
            constr.RHS = x_sol[i, j]

        for (i, j), constr in self.constrs_y.get("delta", {}).items():
            constr.RHS = 1 - x_sol[j, i]

        # ==========================================
        # ACTUALIZAR RHS PARA EL SUBPROBLEMA Y'
        # ==========================================
        for (i, j), constr in self.constrs_yp.get("alpha_p", {}).items():
            constr.RHS = 1 - x_sol[i, j]

        for (i, j), constr in self.constrs_yp.get("beta_p", {}).items():
            constr.RHS = x_sol[j, i] - x_sol[i, j]

        for (i, j), constr in self.constrs_yp.get("gamma_p", {}).items():
            constr.RHS = x_sol[j, i]

        for (i, j), constr in self.constrs_yp.get("delta_p", {}).items():
            constr.RHS = 1 - x_sol[i, j]

        # R3 / R4 strengthening constraints (parameterised by x)
        for (i, j, k, s), constr in self.constrs_y.get("r3", {}).items():
            constr.RHS = 1 - x_sol.get((j, i), 0.0) - x_sol.get((k, s), 0.0)

        for (i, j, k, s), constr in self.constrs_yp.get("r3_p", {}).items():
            constr.RHS = 1 - x_sol.get((i, j), 0.0) - x_sol.get((s, k), 0.0)

    def _benders_callback(self, model: gp.Model, where: int) -> None:
        """
        Callback de Gurobi. Intercepta soluciones enteras candidatas (MIPSOL)
        y comprueba su factibilidad en los subproblemas.

        Dispatch logic
        --------------
        When self.use_deepest_cuts is True (requires BendersCGSPMixin in MRO),
        the callback calls get_cgsp_cut_yp / get_cgsp_cut_y to obtain the
        deepest cut via a Cut-Generating Subproblem (CGSP).

        When self.use_deepest_cuts is False (default, backward compatible),
        the existing Farkas / Pi logic is used unchanged.
        """
        if where == GRB.Callback.MIPSOL:
            # Inicializar buffers de logging: acumulan mensajes durante el callback
            # y se vacían al salir, evitando I/O síncrono dentro del hot-path del solver.
            self._log_buffer: list[str] = []
            self._cut_buffer: list[dict[str, Any]] = []

            # 1. Extraer la solución actual del Maestro (x_bar)
            x_sol = model.cbGetSolution(self.x)

            # --- DFJ: detectar subtours e inyectar SECs lazy antes de evaluar
            # los subproblemas Benders.  Si hay subtours, se retorna inmediatamente
            # para no desperdiciar tiempo resolviendo Y/Y' sobre una solución
            # topológicamente inválida. ---
            if getattr(self, "_subtour_method", "SCF") == "DFJ":
                components = self._detect_subtour_components(x_sol)
                if len(components) > 1:
                    for S in components:
                        if 1 < len(S) < self.N:
                            model.cbLazy(
                                gp.quicksum(self.x[i, j] for i in S for j in S if i != j and (i, j) in self.x)
                                <= len(S) - 1
                            )
                    # Vaciar buffers (vacíos en este caso) y salir
                    self._log_buffer = []
                    self._cut_buffer = []
                    return

            eta_sol = (
                model.cbGetSolution(self.eta) if hasattr(self, "objective") and self.objective == "Internal" else 0.0
            )

            # 2. Inyectar x_bar en los RHS de los subproblemas
            self._update_subproblem_rhs(x_sol)

            # 3. Resolver ambos subproblemas
            self.sub_y.optimize()
            self.sub_yp.optimize()

            TOL = 1e-6

            use_deepest = getattr(self, "use_deepest_cuts", False)
            use_mw = getattr(self, "use_magnanti_wong", False)
            use_ddma = getattr(self, "use_ddma", False)

            # 4. Análisis del Subproblema Y
            y_status = self.sub_y.Status
            y_objval = self.sub_y.ObjVal if y_status == GRB.OPTIMAL else None

            if use_deepest:
                rel_tol = max(TOL, 1e-5 * abs(eta_sol))
                needs_cgsp_y = (
                    y_status == GRB.INFEASIBLE
                    or (
                        self.benders_method == "pi"
                        and y_status == GRB.OPTIMAL
                        and y_objval is not None
                        and y_objval > TOL
                    )
                    or (
                        y_status == GRB.OPTIMAL
                        and getattr(self, "objective", "Fekete") == "Internal"
                        and y_objval is not None
                        and y_objval > eta_sol + rel_tol  # gate: only fire on real violation
                    )
                )
                if needs_cgsp_y:
                    # --- CGSP / deepest-cut branch ---
                    cut_expr_y, cut_rhs_y, _witness_y = self.get_cgsp_cut_y(x_sol, eta_sol=eta_sol, TOL=TOL)
                    if cut_expr_y is not None:
                        model.cbLazy(cut_expr_y <= cut_rhs_y)
            elif use_ddma and (
                y_status == GRB.INFEASIBLE
                or (self.benders_method == "pi" and y_status == GRB.OPTIMAL and y_objval is not None and y_objval > TOL)
            ):
                # --- DDMA branch (F3) ---
                cut_expr_y, cut_rhs_y, _witness_y = self.get_ddma_cut_y(x_sol, eta_sol=eta_sol, TOL=TOL)
                if cut_expr_y is not None:
                    model.cbLazy(cut_expr_y <= cut_rhs_y)
            elif use_mw and (
                y_status == GRB.INFEASIBLE
                or (self.benders_method == "pi" and y_status == GRB.OPTIMAL and y_objval is not None and y_objval > TOL)
            ):
                # --- Magnanti-Wong branch ---
                cut_expr_y, cut_rhs_y, _witness_y = self.get_mw_cut_y(x_sol, TOL=TOL)
                if cut_expr_y is not None:
                    model.cbLazy(cut_expr_y <= cut_rhs_y)
                else:
                    # MW fallback to legacy
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_y(x_sol, TOL)
                    if cut_val > TOL:
                        model.cbLazy(cut_expr <= 0)
                    elif cut_val < -TOL:
                        model.cbLazy(cut_expr >= 0)
            elif not use_deepest:
                # --- Legacy Farkas / Pi branch ---
                if self.sub_y.Status == GRB.INFEASIBLE or (
                    self.benders_method == "pi" and self.sub_y.Status == GRB.OPTIMAL and self.sub_y.ObjVal > TOL
                ):
                    # Generar corte según el método elegido
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_y(x_sol, TOL)

                    # Inyectar el corte lazy en el Maestro
                    if cut_val > TOL:
                        model.cbLazy(cut_expr <= 0)
                    elif cut_val < -TOL:
                        model.cbLazy(cut_expr >= 0)

                elif self.sub_y.Status == GRB.OPTIMAL and getattr(self, "objective", "Fekete") == "Internal":
                    rel_tol = max(TOL, 1e-5 * abs(eta_sol))
                    if self.sub_y.ObjVal > eta_sol + rel_tol:
                        cut_expr, _ = self.get_optimality_cut_y(x_sol, TOL)
                        model.cbLazy(self.eta >= cut_expr)
                    elif self.sub_y.ObjVal < eta_sol - rel_tol:
                        cut_expr, _ = self.get_optimality_cut_y(x_sol, TOL)
                        model.cbLazy(self.eta <= cut_expr)

            # 5. Análisis del Subproblema Y'
            _yp_violated = self.sub_yp.Status == GRB.INFEASIBLE or (
                self.benders_method == "pi" and self.sub_yp.Status == GRB.OPTIMAL and self.sub_yp.ObjVal > TOL
            )
            if use_deepest and _yp_violated:
                # --- CGSP / deepest-cut branch ---
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_cgsp_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    model.cbLazy(cut_expr_yp <= cut_rhs_yp)
            elif use_ddma and _yp_violated:
                # --- DDMA branch (F3) ---
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_ddma_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    model.cbLazy(cut_expr_yp <= cut_rhs_yp)
            elif use_mw and _yp_violated:
                # --- Magnanti-Wong branch ---
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_mw_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    model.cbLazy(cut_expr_yp <= cut_rhs_yp)
                else:
                    # MW fallback to legacy
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_yp(x_sol, TOL)
                    if cut_val > TOL:
                        model.cbLazy(cut_expr <= 0)
                    elif cut_val < -TOL:
                        model.cbLazy(cut_expr >= 0)
            elif not use_deepest and not use_mw:
                # --- Legacy Farkas / Pi branch ---
                if _yp_violated:
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_yp(x_sol, TOL)

                    # Inyectar el corte lazy en el Maestro
                    if cut_val > TOL:
                        model.cbLazy(cut_expr <= 0)
                    elif cut_val < -TOL:
                        model.cbLazy(cut_expr >= 0)

            # --- Vaciar buffers de logging fuera del hot-path del solver ---
            _log_buf = getattr(self, "_log_buffer", [])
            _cut_buf = getattr(self, "_cut_buffer", [])
            if _log_buf:
                logger.info("\n".join(_log_buf))
            if _cut_buf and hasattr(self, "log_path"):
                try:
                    from utils.utils import log_benders_cut

                    for _entry in _cut_buf:
                        log_benders_cut(
                            filepath=self.log_path,
                            iteration=_entry["iteration"],
                            node_depth=_entry["node_depth"],
                            subproblem_type=_entry["subproblem_type"],
                            x_sol={},  # ya no se pasa x_sol al JSON (reducción de tamaño)
                            v_components=_entry["v_components"],
                            cut_value=_entry["cut_value"],
                            tolerance=_entry["tolerance"],
                            cut_expr=None,
                            sense=_entry["sense"],
                        )
                except NameError:
                    pass
            self._log_buffer = []
            self._cut_buffer = []

    def solve(
        self,
        save_cuts: bool = False,
        time_limit: int = 7200,
        verbose: bool = False,
        relaxed: bool = False,
        polihedral: bool = False,
    ) -> None:
        """
        Configura el modelo y ejecuta la optimización con el callback.
        """
        logger.info(f"Iniciando resolución del modelo (Límite: {time_limit}s)...")

        # Guardar configuración local
        self.verbose = verbose
        self.save_cuts = save_cuts
        # self.log_path = None  # Se asignará si save_cuts es True
        self.polihedral = polihedral

        # Parámetros obligatorios de Gurobi para Benders
        self.model.Params.LazyConstraints = 1
        self.model.Params.TimeLimit = time_limit

        # Desactivar salida de consola de Gurobi si no somos verbose
        if not verbose:
            self.model.Params.OutputFlag = 0

            # Limpiar el archivo si ya existía de una ejecución anterior
            _log_parent = os.path.dirname(self.log_path)
            if _log_parent:
                os.makedirs(_log_parent, exist_ok=True)
            with open(self.log_path, "w"):
                pass

        # Dentro del Solver / Callback de Benders:

        if relaxed:
            logger.info("Resolviendo la relajación LP del modelo usando un bucle manual de Benders...")
            self.solve_lp_relaxation(time_limit=time_limit, verbose=verbose)
        else:
            # Lanzar la optimización pasando nuestro método callback
            self.model.optimize(self._benders_callback)

        if self.model.SolCount > 0:
            self.x_results = [(i, j) for (i, j), var in self.x.items() if var.X > 0.5]
            logger.info(f"Solución factible extraída: {len(self.x_results)} arcos activos.")

        # Loggear el resultado final
        if self.model.Status == GRB.OPTIMAL:
            logger.info(f"Optimización completada. Solución óptima encontrada: {self.model.ObjVal:.4f}")
        else:
            logger.warning(f"La optimización terminó con estado: {self.model.Status}")

    def solve_lp_relaxation(
        self, time_limit: int = 7200, verbose: bool = False, save_cuts: bool = False, polihedral: bool = False
    ) -> None:
        """
        Resuelve la relajación LP del modelo usando un bucle manual de Benders.
        (Obligatorio para LPs, ya que Gurobi no dispara callbacks MIPSOL en continuas).
        """
        logger.info("Iniciando resolución de la relajación LP con Benders...")
        self.verbose = verbose
        self.polihedral = polihedral

        # 1. Transformar el modelo a LP (in-place para mantener referencias a self.x)
        for v in self.model.getVars():
            if v.VType != GRB.CONTINUOUS:
                v.VType = GRB.CONTINUOUS

        # PREVENCIÓN DE ERROR '.X': Acotar 'eta' inicialmente previene que el LP sea UNBOUNDED.
        if hasattr(self, "eta"):
            if self.model.ModelSense == GRB.MAXIMIZE:
                if self.eta.UB > 1e12:
                    self.eta.UB = 1e12
            else:
                if self.eta.LB < -1e12:
                    self.eta.LB = -1e12

        self.model.update()

        # 2. Configurar parámetros para el bucle manual LP
        self.model.Params.Presolve = 1
        self.model.Params.LazyConstraints = 0  # No usamos callbacks aquí
        self.model.Params.TimeLimit = time_limit
        # DualReductions can cause Gurobi to report INF_OR_UNBD (status 4) instead of
        # correctly solving the LP when eta is bounded. Disabling it forces Gurobi to
        # distinguish infeasible from unbounded and solve correctly.
        if hasattr(self, "eta"):
            self.model.Params.DualReductions = 0
        if not verbose:
            self.model.Params.OutputFlag = 0

        TOL = 1e-6
        converged = False
        self.iteration = 0
        MAX_LP_ITER = 500 * len(list(self.x))  # safety valve: O(n²) iterations max

        # 3. Bucle Manual de Benders
        while not converged:
            self.iteration += 1
            if self.iteration > MAX_LP_ITER:
                logger.error(
                    f"solve_lp_relaxation: no convergió tras {MAX_LP_ITER} iteraciones. "
                    "Posible degeneración dual en el método PI. Abortando bucle."
                )
                break
            if verbose:
                logger.info(f"\n=== Iteración LP: {self.iteration} ===")

            # Resolver el Maestro relajado
            self.model.optimize()

            # Evita fallo al acceder a `.X` si la relajación acaba infactible o no tiene solución
            if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                logger.warning(
                    f"El modelo Maestro LP terminó con estado {self.model.Status} en la iteración {self.iteration}."
                )
                break

            # Extraer solución fraccional (v.X funciona perfectamente para continuas)
            x_sol = {k: v.X for k, v in self.x.items()}

            # BIEN (extracción estándar):
            if hasattr(self, "eta"):
                eta_sol = self.eta.X
            else:
                eta_sol = 0.0

            # Actualizar RHS de los subproblemas
            # (Restamos 1 a iteration temporalmente porque _update_subproblem_rhs suma 1 por dentro)
            self.iteration -= 1
            self._update_subproblem_rhs(x_sol)

            # Resolver ambos subproblemas
            self.sub_y.optimize()
            self.sub_yp.optimize()

            converged_y = False
            converged_yp = False

            use_deepest = getattr(self, "use_deepest_cuts", False)
            use_mw = getattr(self, "use_magnanti_wong", False)
            use_ddma = getattr(self, "use_ddma", False)

            # --- Análisis del Subproblema Y ---
            lp_y_status = self.sub_y.Status
            lp_y_objval = self.sub_y.ObjVal if lp_y_status == GRB.OPTIMAL else None
            lp_rel_tol = max(TOL, 1e-5 * abs(eta_sol))
            needs_cut_y = (
                lp_y_status == GRB.INFEASIBLE
                or (
                    self.benders_method == "pi"
                    and lp_y_status == GRB.OPTIMAL
                    and lp_y_objval is not None
                    and lp_y_objval > TOL
                )
                or (
                    lp_y_status == GRB.OPTIMAL
                    and getattr(self, "objective", "Fekete") == "Internal"
                    and lp_y_objval is not None
                    and lp_y_objval > eta_sol + lp_rel_tol  # gate: only fire on real violation
                )
            )
            if use_deepest and needs_cut_y:
                cut_expr_y, cut_rhs_y, _witness_y = self.get_cgsp_cut_y(x_sol, eta_sol=eta_sol, TOL=TOL)
                if cut_expr_y is not None:
                    self.model.addConstr(cut_expr_y <= cut_rhs_y, name=f"lp_cgsp_y_{self.iteration}")
                else:
                    converged_y = True
            elif use_deepest and not needs_cut_y:
                converged_y = True
            elif use_ddma and needs_cut_y:
                # --- DDMA branch (F3) ---
                cut_expr_y, cut_rhs_y, _witness_y = self.get_ddma_cut_y(x_sol, eta_sol=eta_sol, TOL=TOL)
                if cut_expr_y is not None:
                    self.model.addConstr(cut_expr_y <= cut_rhs_y, name=f"lp_ddma_y_{self.iteration}")
                else:
                    converged_y = True
            elif use_ddma and not needs_cut_y:
                converged_y = True
            elif use_mw and needs_cut_y:
                # --- Magnanti-Wong branch ---
                cut_expr_y, cut_rhs_y, _witness_y = self.get_mw_cut_y(x_sol, TOL=TOL)
                if cut_expr_y is not None:
                    self.model.addConstr(cut_expr_y <= cut_rhs_y, name=f"lp_mw_y_{self.iteration}")
                else:
                    # MW fallback to legacy
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_y(x_sol, TOL)
                    if cut_val > TOL:
                        self.model.addConstr(cut_expr <= 0, name=f"lp_cut_y_{self.iteration}")
                    elif cut_val < -TOL:
                        self.model.addConstr(cut_expr >= 0, name=f"lp_cut_y_{self.iteration}")
                    else:
                        converged_y = True
            elif use_mw and not needs_cut_y:
                converged_y = True
            elif self.sub_y.Status == GRB.INFEASIBLE or (
                self.benders_method == "pi" and self.sub_y.Status == GRB.OPTIMAL and self.sub_y.ObjVal > TOL
            ):
                if self.benders_method == "farkas":
                    cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
                else:
                    cut_expr, cut_val = self.get_pi_cut_y(x_sol, TOL)

                # Inyectar el corte duro en el Maestro
                if cut_val > TOL:
                    self.model.addConstr(cut_expr <= 0, name=f"lp_cut_y_{self.iteration}")
                elif cut_val < -TOL:
                    self.model.addConstr(cut_expr >= 0, name=f"lp_cut_y_{self.iteration}")
                else:
                    # Dual degenerate: all Pi = 0, cut is trivial (π^T x_sol = 0).
                    # No constraint can be injected; mark as converged to exit loop.
                    logger.warning(
                        f"Iter {self.iteration}: corte PI para Y es trivial "
                        f"(cut_val={cut_val:.2e}, sub_y.ObjVal={self.sub_y.ObjVal:.2e}). "
                        "Degeneración dual — se marca Y como convergido."
                    )
                    converged_y = True

            elif self.sub_y.Status == GRB.OPTIMAL and getattr(self, "objective", "Fekete") == "Internal":
                if self.sub_y.ObjVal > eta_sol + TOL:
                    cut_expr, _ = self.get_optimality_cut_y(x_sol, TOL)
                    self.model.addConstr(self.eta >= cut_expr, name=f"lp_opt_cut_y_{self.iteration}")
                elif self.sub_y.ObjVal < eta_sol - TOL:
                    cut_expr, _ = self.get_optimality_cut_y(x_sol, TOL)
                    self.model.addConstr(self.eta <= cut_expr, name=f"lp_opt_cut_y_{self.iteration}")
                else:
                    converged_y = True

            else:
                converged_y = True

            # --- Análisis del Subproblema Y' ---
            needs_cut_yp = self.sub_yp.Status == GRB.INFEASIBLE or (
                self.benders_method == "pi" and self.sub_yp.Status == GRB.OPTIMAL and self.sub_yp.ObjVal > TOL
            )
            if use_deepest and needs_cut_yp:
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_cgsp_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    self.model.addConstr(cut_expr_yp <= cut_rhs_yp, name=f"lp_cgsp_yp_{self.iteration}")
                else:
                    converged_yp = True
            elif use_deepest and not needs_cut_yp:
                converged_yp = True
            elif use_ddma and needs_cut_yp:
                # --- DDMA branch (F3) ---
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_ddma_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    self.model.addConstr(cut_expr_yp <= cut_rhs_yp, name=f"lp_ddma_yp_{self.iteration}")
                else:
                    converged_yp = True
            elif use_ddma and not needs_cut_yp:
                converged_yp = True
            elif use_mw and needs_cut_yp:
                # --- Magnanti-Wong branch ---
                cut_expr_yp, cut_rhs_yp, _witness_yp = self.get_mw_cut_yp(x_sol, TOL=TOL)
                if cut_expr_yp is not None:
                    self.model.addConstr(cut_expr_yp <= cut_rhs_yp, name=f"lp_mw_yp_{self.iteration}")
                else:
                    # MW fallback to legacy
                    if self.benders_method == "farkas":
                        cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
                    else:
                        cut_expr, cut_val = self.get_pi_cut_yp(x_sol, TOL)
                    if cut_val > TOL:
                        self.model.addConstr(cut_expr <= 0, name=f"lp_cut_yp_{self.iteration}")
                    elif cut_val < -TOL:
                        self.model.addConstr(cut_expr >= 0, name=f"lp_cut_yp_{self.iteration}")
                    else:
                        converged_yp = True
            elif use_mw and not needs_cut_yp:
                converged_yp = True
            elif self.sub_yp.Status == GRB.INFEASIBLE or (
                self.benders_method == "pi" and self.sub_yp.Status == GRB.OPTIMAL and self.sub_yp.ObjVal > TOL
            ):
                if self.benders_method == "farkas":
                    cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
                else:
                    cut_expr, cut_val = self.get_pi_cut_yp(x_sol, TOL)

                # Inyectar el corte duro en el Maestro
                if cut_val > TOL:
                    self.model.addConstr(cut_expr <= 0, name=f"lp_cut_yp_{self.iteration}")
                elif cut_val < -TOL:
                    self.model.addConstr(cut_expr >= 0, name=f"lp_cut_yp_{self.iteration}")
                else:
                    # Dual degenerate: all Pi = 0, cut is trivial (π^T x_sol = 0).
                    logger.warning(
                        f"Iter {self.iteration}: corte PI para Y' es trivial "
                        f"(cut_val={cut_val:.2e}, sub_yp.ObjVal={self.sub_yp.ObjVal:.2e}). "
                        "Degeneración dual — se marca Y' como convergido."
                    )
                    converged_yp = True
            else:
                converged_yp = True

            if self.polihedral:
                self.poly_log_path = self.poly_log_path.replace("_iter_*.json", f"_iter_{self.iteration}.json")
                self.log_facets(filepath=self.poly_log_path, var_prefixes=["x"], verbose=False)

            # --- Condición de parada ---
            if converged_y and converged_yp:
                converged = True
                logger.info(f"Relajación LP convergida exitosamente tras {self.iteration} iteraciones.")

        # Signal: True only when the loop exited via genuine Benders convergence.
        # False when it broke out (MAX_LP_ITER or degenerate PI cuts forced exit).
        # Checked by get_objval_lp() to avoid reporting an unreliable LP bound.
        self._lp_converged: bool = converged
