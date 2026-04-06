# models/mixin/benders_optimize_mixin.py
import os

import gurobipy as gp
from gurobipy import GRB
import logging

from typing import Literal

logger = logging.getLogger(__name__)

Arc = tuple[int, int]

class BendersOptimizeMixin:
    """
    Mixin encargado de la orquestación, actualización de RHS,
    el Callback de Gurobi y la resolución general del modelo.
    """
    constrs_y: dict[str, dict[Arc, gp.Constr]]
    constrs_yp: dict[str, dict[Arc, gp.Constr]]
    iteration: int
    cortes_añadidos: int
    x: dict[Arc, gp.Var]
    sub_y: gp.Model
    sub_yp: gp.Model
    benders_method: str
    objective: Literal["Fekete", "Internal"]
    
    def _update_subproblem_rhs(self, x_sol: dict[Arc, float]) -> None:
        """
        Actualiza los lados derechos (RHS) de los subproblemas Y y Y' 
        usando la solución propuesta por el problema Maestro.
        """
        self.iteration += 1

        # ==========================================
        # ACTUALIZAR RHS PARA EL SUBPROBLEMA Y
        # ==========================================
        for (i, j), constr in self.constrs_y.get('alpha', {}).items():
            constr.RHS = x_sol[i, j]
            
        for (i, j), constr in self.constrs_y.get('beta', {}).items():
            constr.RHS = x_sol[i, j] - x_sol[j, i]
            
        for (i, j), constr in self.constrs_y.get('gamma', {}).items():
            constr.RHS = x_sol[i, j]
            
        for (i, j), constr in self.constrs_y.get('delta', {}).items():
            constr.RHS = 1 - x_sol[j, i]

        # ==========================================
        # ACTUALIZAR RHS PARA EL SUBPROBLEMA Y'
        # ==========================================
        for (i, j), constr in self.constrs_yp.get('alpha_p', {}).items():
            constr.RHS = 1 - x_sol[i, j]
            
        for (i, j), constr in self.constrs_yp.get('beta_p', {}).items():
            constr.RHS = x_sol[j, i] - x_sol[i, j]
            
        for (i, j), constr in self.constrs_yp.get('gamma_p', {}).items():
            constr.RHS = x_sol[j, i]
            
        for (i, j), constr in self.constrs_yp.get('delta_p', {}).items():
            constr.RHS = 1 - x_sol[i, j]

    def _benders_callback(self, model: gp.Model, where: int) -> None:
        """
        Callback de Gurobi. Intercepta soluciones enteras candidatas (MIPSOL)
        y comprueba su factibilidad en los subproblemas.
        """
        if where == GRB.Callback.MIPSOL:
            # 1. Extraer la solución actual del Maestro (x_bar)
            x_sol = model.cbGetSolution(self.x)
            
            eta_sol = model.cbGetSolution(self.eta) if hasattr(self, 'objective') and self.objective == "Internal" else 0.0

            # 2. Inyectar x_bar en los RHS de los subproblemas
            self._update_subproblem_rhs(x_sol)

            # 3. Resolver ambos subproblemas
            self.sub_y.optimize()
            self.sub_yp.optimize()

            TOL = 1e-6

            # 4. Análisis del Subproblema Y
            if self.sub_y.Status == GRB.INFEASIBLE or (self.benders_method == "pi" and self.sub_y.ObjVal > TOL):
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

            elif self.sub_y.Status == GRB.OPTIMAL and getattr(self, 'objective', 'Fekete') == "Internal":
                rel_tol = max(TOL, 1e-5 * abs(eta_sol))
                if self.sub_y.ObjVal > eta_sol + rel_tol:
                    cut_expr, cut_val = self.get_optimality_cut_y(x_sol, TOL)
                    model.cbLazy(self.eta >= cut_expr)
                elif self.sub_y.ObjVal < eta_sol - rel_tol:
                    cut_expr, cut_val = self.get_optimality_cut_y(x_sol, TOL)
                    model.cbLazy(self.eta <= cut_expr)


            # 5. Análisis del Subproblema Y'
            if self.sub_yp.Status == GRB.INFEASIBLE or (self.benders_method == "pi" and self.sub_yp.ObjVal > TOL):
                if self.benders_method == "farkas":
                    cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
                else:
                    cut_expr, cut_val = self.get_pi_cut_yp(x_sol, TOL)

                # Inyectar el corte lazy en el Maestro
                if cut_val > TOL:
                    model.cbLazy(cut_expr <= 0)
                elif cut_val < -TOL:
                    model.cbLazy(cut_expr >= 0)

    def solve(self, save_cuts: bool = False, time_limit: int = 7200, verbose: bool = False, relaxed: bool = False, polihedral: bool = False) -> None:
        """
        Configura el modelo y ejecuta la optimización con el callback.
        """
        logger.info(f"Iniciando resolución del modelo (Límite: {time_limit}s)...")
        
        # Guardar configuración local
        self.verbose = verbose
        self.save_cuts = save_cuts
        #self.log_path = None  # Se asignará si save_cuts es True
        self.polihedral = polihedral
        
        # Parámetros obligatorios de Gurobi para Benders
        self.model.Params.LazyConstraints = 1
        self.model.Params.TimeLimit = time_limit
        
        # Desactivar salida de consola de Gurobi si no somos verbose
        if not verbose:
            self.model.Params.OutputFlag = 0

            # Limpiar el archivo si ya existía de una ejecución anterior
            import os
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
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


    def solve_lp_relaxation(self, time_limit: int = 7200, verbose: bool = False, save_cuts: bool = False, polihedral: bool = False) -> None:
        """
        Resuelve la relajación LP del modelo usando un bucle manual de Benders.
        (Obligatorio para LPs, ya que Gurobi no dispara callbacks MIPSOL en continuas).
        """
        logger.info("Iniciando resolución de la relajación LP con Benders...")
        self.verbose = verbose
        self.save_cuts = save_cuts
        self.polihedral = polihedral

        # 1. Transformar el modelo a LP (in-place para mantener referencias a self.x)
        for v in self.model.getVars():
            if v.VType != GRB.CONTINUOUS:
                v.VType = GRB.CONTINUOUS
        
        # PREVENCIÓN DE ERROR '.X': Acotar 'eta' inicialmente previene que el LP sea UNBOUNDED. 
        if hasattr(self, 'eta'):
            if self.model.ModelSense == GRB.MAXIMIZE:
                if self.eta.UB > 1e12:
                    self.eta.UB = 1e12
            else:
                if self.eta.LB < -1e12:
                    self.eta.LB = -1e12
        
        self.model.update()
        self.model.update()

        # 2. Configurar parámetros para el bucle manual LP
        self.model.Params.Presolve = 1
        self.model.Params.LazyConstraints = 0  # No usamos callbacks aquí
        self.model.Params.TimeLimit = time_limit
        if not verbose:
            self.model.Params.OutputFlag = 0

        TOL = 1e-6
        converged = False
        self.iteration = 0
        



        # 3. Bucle Manual de Benders
        while not converged:
            self.iteration += 1
            if verbose:
                logger.info(f"\n=== Iteración LP: {self.iteration} ===")

            # Resolver el Maestro relajado
            self.model.optimize()
            
            # Evita fallo al acceder a `.X` si la relajación acaba infactible
            if self.model.Status == GRB.INFEASIBLE:
                logger.warning(f"El modelo Maestro LP se volvió INFACTIBLE en la iteración {self.iteration}.")
                break
            
            # Extraer solución fraccional (v.X funciona perfectamente para continuas)
            x_sol = {k: v.X for k, v in self.x.items()}
            
                        # BIEN (extracción estándar):
            if hasattr(self, 'eta'):
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

            # --- Análisis del Subproblema Y ---
            if self.sub_y.Status == GRB.INFEASIBLE or (self.benders_method == "pi" and self.sub_y.ObjVal > TOL):
                if self.benders_method == "farkas":
                    cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
                else:
                    cut_expr, cut_val = self.get_pi_cut_y(x_sol, TOL)

                # Inyectar el corte duro en el Maestro
                if cut_val > TOL:
                    self.model.addConstr(cut_expr <= 0, name=f"lp_cut_y_{self.iteration}")
                elif cut_val < -TOL:
                    self.model.addConstr(cut_expr >= 0, name=f"lp_cut_y_{self.iteration}")

            elif self.sub_y.Status == GRB.OPTIMAL and getattr(self, 'objective', 'Fekete') == "Internal":
                if self.sub_y.ObjVal > eta_sol + TOL:
                    cut_expr, cut_val = self.get_optimality_cut_y(x_sol, TOL)
                    self.model.addConstr(self.eta >= cut_expr, name=f"lp_opt_cut_y_{self.iteration}")
                elif self.sub_y.ObjVal < eta_sol - TOL:
                    cut_expr, cut_val = self.get_optimality_cut_y(x_sol, TOL)
                    self.model.addConstr(self.eta <= cut_expr, name=f"lp_opt_cut_y_{self.iteration}")
                else:
                    converged_y = True

            else:
                converged_y = True

           
            # --- Análisis del Subproblema Y' ---
            if self.sub_yp.Status == GRB.INFEASIBLE or (self.benders_method == "pi" and self.sub_yp.ObjVal > TOL):
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
                converged_yp = True

            if self.polihedral:
                self.poly_log_path = self.poly_log_path.replace("_iter_*.json", f"_iter_{self.iteration}.json")
                self.log_facets(filepath=self.poly_log_path, var_prefixes=['x'], verbose=False)

            # --- Condición de parada ---
            if converged_y and converged_yp:
                converged = True
                logger.info(f"Relajación LP convergida exitosamente tras {self.iteration} iteraciones.")
        