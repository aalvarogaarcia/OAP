# models/mixin/benders_pi_mixin.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import logging
from models.typing_oap import IndexArray, TrianglesAdjList

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
    constrs_y: dict
    constrs_yp: dict
    iteration: int

    def build_pi_subproblems(self, sum_constrain: bool = True) -> None:
        """Orquestador principal para construir los Subproblemas con método Pi."""
        logger.info("Construyendo subproblemas con método Pi (Fase 1 con variables artificiales)...")

        self._configurar_parametros_pi()
        
        # 1. Preparar conjuntos de arcos necesarios
        A_prime, A_double_prime, A_double_prime_beta = self._obtener_conjuntos_arcos()
        
        # 2. Inicializar diccionarios de restricciones
        self.constrs_y.update({'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}})
        self.constrs_yp.update({'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}})
        
        # 3. Crear variables reales y artificiales
        self._crear_variables_pi(A_prime, A_double_prime, A_double_prime_beta, sum_constrain)
        
        # 4. Establecer funciones objetivo (Minimizar artificiales)
        self._establecer_objetivos_pi()
        
        # 5. Añadir las restricciones de Fase 1
        self._añadir_restricciones_pi(A_prime, A_double_prime, A_double_prime_beta, sum_constrain)

        self.sub_y.update()
        self.sub_yp.update()

    def get_pi_cut_y(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de Benders usando variables duales (.Pi) del subproblema Y."""
        cut_y_expr = gp.LinExpr()
        cut_y_val = 0.0
        v_comps: RayComponents = {'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}}
        
        for (i, j), constr in self.constrs_y['alpha'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['alpha'][i, j] = pi
                cut_y_expr += pi * self.x[i, j]
                cut_y_val += pi * x_sol[i, j]
                
        for (i, j), constr in self.constrs_y['beta'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['beta'][i, j] = pi
                cut_y_expr += pi * (self.x[i, j] - self.x[j, i])
                cut_y_val += pi * (x_sol[i, j] - x_sol[j, i])
                
        for (i, j), constr in self.constrs_y['gamma'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['gamma'][i, j] = pi
                cut_y_expr += pi * self.x[i, j]
                cut_y_val += pi * x_sol[i, j]
                
        for (i, j), constr in self.constrs_y['delta'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['delta'][i, j] = pi
                cut_y_expr += pi * (1 - self.x[j, i])
                cut_y_val += pi * (1 - x_sol[j, i])
        
        if 'global' in self.constrs_y:
            pi_global = self.constrs_y['global'].Pi
            if abs(pi_global) > TOL:
                rhs_global = self.constrs_y['global'].RHS
                v_comps['global'] = pi_global
                cut_y_expr += pi_global * rhs_global
                cut_y_val += pi_global * rhs_global
        

   
        if cut_y_val > TOL:
            sense = "<="
        elif cut_y_val < -TOL:
            sense = ">="
        self._log_and_print_pi(v_comps, cut_y_val, "Y", TOL, x_sol, cut_y_expr, sense)
        
        return cut_y_expr, cut_y_val
    
    def get_pi_cut_yp(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de Benders usando variables duales (.Pi) del subproblema YP."""
        cut_yp_expr = gp.LinExpr()
        cut_yp_val = 0.0
        v_comps_p: RayComponents = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}}
        
        for (i, j), constr in self.constrs_yp['alpha_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p['alpha_p'][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['beta_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p['beta_p'][i, j] = pi
                cut_yp_expr += pi * (self.x[j, i] - self.x[i, j])
                cut_yp_val += pi * (x_sol[j, i] - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['gamma_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p['gamma_p'][i, j] = pi
                cut_yp_expr += pi * self.x[j, i]
                cut_yp_val += pi * x_sol[j, i]
                
        for (i, j), constr in self.constrs_yp['delta_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps_p['delta_p'][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])
        
        if 'global_p' in self.constrs_yp:
            pi_global_p = self.constrs_yp['global_p'].Pi
            if abs(pi_global_p) > TOL:
                rhs_global_p = self.constrs_yp['global_p'].RHS
                v_comps_p['global_p'] = pi_global_p
                cut_yp_expr += pi_global_p * rhs_global_p
                cut_yp_val += pi_global_p * rhs_global_p

        if cut_yp_val > TOL:
            sense = "<="
        elif cut_yp_val < -TOL:
            sense = ">="
        else:
            sense = "==" # Por si acaso

        self._log_and_print_pi(v_comps_p, cut_yp_val, "Y'", TOL, x_sol, cut_yp_expr, sense)
        return cut_yp_expr, cut_yp_val

    def _log_and_print_pi(self, v_components, cut_val, sub_name, TOL, x_sol, cut_expr, sense=None):
        """Método auxiliar interno para registrar el log de variables duales (.Pi).
        Diferencia automáticamente entre cortes de Optimalidad y Factibilidad.
        """
        verbose = getattr(self, 'verbose', False)
        save_cuts = getattr(self, 'save_cuts', False)
        
        # Determinar si el corte es de optimalidad basado en el nombre del subproblema
        is_optimality = (sub_name == 'Y_OPT')
        
        tipo_corte = "OPTIMALIDAD" if is_optimality else "FACTIBILIDAD"
        nombre_impreso = "Y" if is_optimality else sub_name
        
        # 1. Registro en la consola / archivo log estándar de Python
        if save_cuts and verbose:
            log_msg = [
                f"\n{'='*50}",
                f"CORTE DE {tipo_corte} (MÉTODO PI) DETECTADO EN SUBPROBLEMA {nombre_impreso}",
                f"Valor {'del coste real evaluado' if is_optimality else 'numérico de la violación'}: {cut_val:.6f}"
            ]
            
            for comp, values in v_components.items():
                if values:
                    log_msg.append(f"Componente {comp}:")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            log_msg.append(f"  {k}: {v:.4f}")
                    else:
                        log_msg.append(f"  Valor: {values:.4f}")
                        
            log_msg.append(f"{'='*50}\n")
            logger.info("\n".join(log_msg))

        # 2. Registro JSON/Estructurado personalizado (para el Post-Mortem)
        if self.save_cuts and hasattr(self, 'log_path'):
            try:
                # Nota: Uso log_benders_cut para mantener compatibilidad con tu código actual. 
                # Si en el futuro la renombras a log_benders_cut en utils.py, actualízalo aquí.
                from utils.utils import log_benders_cut
                
                # Para el JSON, mantenemos la etiqueta Y_OPT para que el visualizador 
                # (Post-Mortem) sepa que es de optimalidad.
                json_subproblem_type = 'Y_OPT' if is_optimality else ('Y' if sub_name == 'Y' else 'Y_prime')
                
                log_benders_cut(
                    filepath=self.log_path,
                    iteration=getattr(self, 'iteration', 0),
                    node_depth=0,
                    subproblem_type=json_subproblem_type,
                    x_sol=x_sol,
                    v_components=v_components,
                    cut_value=cut_val,
                    tolerance=TOL,
                    cut_expr=cut_expr,
                    sense=sense
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

    def _crear_variables_pi(self, A_prime: list[Arc], A_double_prime: list[Arc], A_double_prime_beta: list[Arc], sum_constrain: bool) -> None:
        """Crea las variables 'y', 'yp' y todas las variables artificiales de holgura."""
        # Variables Reales
        self.y = self.sub_y.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.yp = self.sub_yp.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="yp")
        
        # Variables Artificiales encapsuladas en diccionarios para mantener el orden
        self.art_y = {
            'alpha_p': self.sub_y.addVars(A_prime, lb=0, name="art_y_alpha_p"),
            'alpha_n': self.sub_y.addVars(A_prime, lb=0, name="art_y_alpha_n"),
            'beta_p':  self.sub_y.addVars(A_double_prime_beta, lb=0, name="art_y_beta_p"),
            'beta_n':  self.sub_y.addVars(A_double_prime_beta, lb=0, name="art_y_beta_n"),
            'gamma_p': self.sub_y.addVars(A_double_prime, lb=0, name="art_y_gamma_p"),
            'gamma_n': self.sub_y.addVars(A_double_prime, lb=0, name="art_y_gamma_n"),
            'delta_p': self.sub_y.addVars(A_double_prime, lb=0, name="art_y_delta_p"),
            'delta_n': self.sub_y.addVars(A_double_prime, lb=0, name="art_y_delta_n")
        }

        self.art_yp = {
            'alpha_p': self.sub_yp.addVars(A_prime, lb=0, name="art_yp_alpha_p"),
            'alpha_n': self.sub_yp.addVars(A_prime, lb=0, name="art_yp_alpha_n"),
            'beta_p':  self.sub_yp.addVars(A_double_prime_beta, lb=0, name="art_yp_beta_p"),
            'beta_n':  self.sub_yp.addVars(A_double_prime_beta, lb=0, name="art_yp_beta_n"),
            'gamma_p': self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_gamma_p"),
            'gamma_n': self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_gamma_n"),
            'delta_p': self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_delta_p"),
            'delta_n': self.sub_yp.addVars(A_double_prime, lb=0, name="art_yp_delta_n")
        }

        if sum_constrain:
            self.art_y['global_p'] = self.sub_y.addVar(lb=0, name="art_y_global_p")
            self.art_y['global_n'] = self.sub_y.addVar(lb=0, name="art_y_global_n")
            self.art_yp['global_p'] = self.sub_yp.addVar(lb=0, name="art_yp_global_p")
            self.art_yp['global_n'] = self.sub_yp.addVar(lb=0, name="art_yp_global_n")

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
        sense = self.model.ModelSense if hasattr(self, 'model') else GRB.MINIMIZE
        self.sub_y.setObjective(obj_y, sense)
        self.sub_yp.setObjective(obj_yp, sense)

    def _añadir_restricciones_pi(self, A_prime: list[Arc], A_double_prime: list[Arc], A_double_prime_beta: list[Arc], sum_constrain: bool) -> None:
        """Construye las ecuaciones matemáticas integrando las variables reales y las artificiales."""
        
        # --- Restricciones Globales ---
        if sum_constrain:
            self.constrs_y['global'] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.V_list) + self.art_y['global_p'] - self.art_y['global_n'] == self.N - 2, 
                name="triangulos_internos_totales"
            )
            self.constrs_yp['global_p'] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.V_list) + self.art_yp['global_p'] - self.art_yp['global_n'] == self.N - len(self.CH),
                name="triangulos_externos_totales"
            )

        # --- A': Arcos dirigidos en la frontera de la CH ---
        for (i, j) in A_prime:
            self.constrs_y['alpha'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) + self.art_y['alpha_p'][i, j] - self.art_y['alpha_n'][i, j] == 0, name=f"alpha_{i}_{j}"
            )
            self.constrs_yp['alpha_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) + self.art_yp['alpha_p'][i, j] - self.art_yp['alpha_n'][i, j] == 0, name=f"alpha_p_{i}_{j}"
            )

        # --- E'' y A'': Arcos no pertenecientes a la CH ---
        for (i, j) in A_double_prime_beta:
            self.constrs_y['beta'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.y[t] for t in self.triangles_adj_list[j][i]) + self.art_y['beta_p'][i, j] - self.art_y['beta_n'][i, j] == 0,
                name=f"beta_{i}_{j}"
            )
            self.constrs_yp['beta_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.yp[t] for t in self.triangles_adj_list[j][i]) + self.art_yp['beta_p'][i, j] - self.art_yp['beta_n'][i, j] == 0,
                name=f"beta_p_{i}_{j}"
            )

        for (i, j) in A_double_prime:
            self.constrs_y['gamma'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) + self.art_y['gamma_p'][i, j] - self.art_y['gamma_n'][i, j] >= 0, name=f"gamma_{i}_{j}"
            )
            self.constrs_y['delta'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) + self.art_y['delta_p'][i, j] - self.art_y['delta_n'][i, j] <= 0, name=f"delta_{i}_{j}"
            )
            
            self.constrs_yp['gamma_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) + self.art_yp['gamma_p'][i, j] - self.art_yp['gamma_n'][i, j] >= 0, name=f"gamma_p_{i}_{j}"
            )
            self.constrs_yp['delta_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) + self.art_yp['delta_p'][i, j] - self.art_yp['delta_n'][i, j] <= 0, name=f"delta_p_{i}_{j}"
            )