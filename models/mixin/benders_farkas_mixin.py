# models/mixin/benders_farkas_mixin.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from models.typing_oap import IndexArray, TrianglesAdjList
import logging


# Instanciamos el logger para este módulo
logger = logging.getLogger(__name__)

# Si mantienes la función de logging en un archivo de utilidades, impórtala aquí:


Arc = tuple[int, int]
RayComponents = dict[str, dict[Arc, float] | float]

class BendersFarkasMixin:
    """
    Mixin para construir los subproblemas de factibilidad de Benders 
    y extraer los cortes mediante el Lema de Farkas.
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
    
    def build_farkas_subproblems(self, sum_constrain: bool = True) -> None:
        """Construye los Subproblemas (SP_Y y SP_YP) de factibilidad."""
        print("Construyendo subproblemas con método Farkas...")

        # 1. Configuración de parámetros vitales para extraer Farkas Duals
        for sub_model in [self.sub_y, self.sub_yp]:
            sub_model.Params.InfUnbdInfo = 1
            sub_model.Params.DualReductions = 0
            sub_model.Params.Presolve = 1
        
        # 2. Creación de variables Y y YP en sus respectivos modelos
        self.y = self.sub_y.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.yp = self.sub_yp.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="yp")
        
        # 3. Inicializar diccionarios de restricciones
        self.constrs_y.update({'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}})
        self.constrs_yp.update({'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}})
        
        # --- Restricciones Globales ---
        if sum_constrain:
            self._add_sum_constrain_farkas()

        # --- A': Arcos dirigidos en la frontera de la CH ---
        self._add_A_prime_constrs_farkas()

        # --- E'' y A'': Arcos no pertenecientes a la CH ---
        self._add_non_ch_constr_farkas()

        self.sub_y.update()
        self.sub_yp.update()


    def get_farkas_cut_y(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el rayo de Farkas del subproblema Y y genera el corte lineal."""
        cut_y_expr = gp.LinExpr()
        cut_y_val = 0.0
        v_comps: RayComponents = {'alpha': {}, 'beta': {}, 'gamma': {}, 'delta': {}}
        
        for (i, j), constr in self.constrs_y['alpha'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps['alpha'][i, j] = farkas
                cut_y_expr += farkas * self.x[i, j]
                cut_y_val += farkas * x_sol[i, j]
                
        for (i, j), constr in self.constrs_y['beta'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps['beta'][i, j] = farkas
                cut_y_expr += farkas * (self.x[i, j] - self.x[j, i])
                cut_y_val += farkas * (x_sol[i, j] - x_sol[j, i])
                
        for (i, j), constr in self.constrs_y['gamma'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps['gamma'][i, j] = farkas
                cut_y_expr += farkas * self.x[i, j]
                cut_y_val += farkas * x_sol[i, j]
                
        for (i, j), constr in self.constrs_y['delta'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps['delta'][i, j] = farkas
                cut_y_expr += farkas * (1 - self.x[j, i])
                cut_y_val += farkas * (1 - x_sol[j, i])
        
        if 'global' in self.constrs_y:
            farkas_global = self.constrs_y['global'].FarkasDual
            if abs(farkas_global) > TOL:
                rhs_global = self.constrs_y['global'].RHS
                v_comps['global'] = farkas_global
                cut_y_expr += farkas_global * rhs_global
                cut_y_val += farkas_global * rhs_global

        self._log_and_print_farkas(v_comps, cut_y_val, "Y", TOL, x_sol, cut_y_expr)
        return cut_y_expr, cut_y_val


    def get_farkas_cut_yp(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el rayo de Farkas del subproblema YP y genera el corte lineal."""
        cut_yp_expr = gp.LinExpr()
        cut_yp_val = 0.0
        v_comps_p: RayComponents = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}}
        
        for (i, j), constr in self.constrs_yp['alpha_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps_p['alpha_p'][i, j] = farkas
                cut_yp_expr += farkas * (1 - self.x[i, j])
                cut_yp_val += farkas * (1 - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['beta_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps_p['beta_p'][i, j] = farkas
                cut_yp_expr += farkas * (self.x[j, i] - self.x[i, j])
                cut_yp_val += farkas * (x_sol[j, i] - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['gamma_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps_p['gamma_p'][i, j] = farkas
                cut_yp_expr += farkas * self.x[j, i]
                cut_yp_val += farkas * x_sol[j, i]
                
        for (i, j), constr in self.constrs_yp['delta_p'].items():
            farkas = constr.FarkasDual
            if abs(farkas) > TOL:
                v_comps_p['delta_p'][i, j] = farkas
                cut_yp_expr += farkas * (1 - self.x[i, j])
                cut_yp_val += farkas * (1 - x_sol[i, j])
        
        if 'global_p' in self.constrs_yp:
            farkas_global_p = self.constrs_yp['global_p'].FarkasDual
            if abs(farkas_global_p) > TOL:
                rhs_global_p = self.constrs_yp['global_p'].RHS
                v_comps_p['global_p'] = farkas_global_p
                cut_yp_expr += farkas_global_p * rhs_global_p
                cut_yp_val += farkas_global_p * rhs_global_p

        self._log_and_print_farkas(v_comps_p, cut_yp_val, "Y'", TOL, x_sol, cut_yp_expr)
        return cut_yp_expr, cut_yp_val



    def _log_and_print_farkas(self, v_components, cut_val, sub_name, TOL, x_sol, cut_expr):
        """Método auxiliar interno para registrar el log del rayo de Farkas."""
        verbose = getattr(self, 'verbose', False)
        save_cuts = getattr(self, 'save_cuts', False)
        
        # 1. Registro en la consola / archivo log estándar de Python
        if save_cuts and verbose:
            # Construimos el mensaje de forma eficiente usando una lista
            log_msg = [
                f"\n{'='*50}",
                f"RAYO DE FARKAS DETECTADO EN SUBPROBLEMA {sub_name}",
                f"Valor numérico de la violación (v^T * b(x_bar)): {cut_val:.6f}"
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
            
            # Lanzamos todo el bloque de texto como un evento INFO (o DEBUG si prefieres)
            logger.info("\n".join(log_msg))

        # 2. Tu registro JSON/Estructurado personalizado (para el Post-Mortem)
        if self.save_cuts and hasattr(self, 'farkas_log_path'):
            try:
                from utils.utils import log_farkas_ray
                log_farkas_ray(
                    filepath=self.farkas_log_path,
                    iteration=self.iteration,
                    node_depth=0,
                    subproblem_type='Y' if sub_name == 'Y' else 'Y_prime',
                    x_sol=x_sol,
                    v_components=v_components,
                    violation_value=cut_val,
                    tolerance=TOL,
                    cut_expr=cut_expr,
                )
            except NameError:
                logger.warning("No se pudo guardar el log estructurado: log_farkas_ray no está definido.")





    def _add_sum_constrain_farkas(self):
        """Método auxiliar para añadir la restricción de suma en los subproblemas."""
        self.constrs_y['global'] = self.sub_y.addConstr(
            gp.quicksum(self.y[t] for t in self.V_list) == self.N - 2, 
            name="triangulos_internos_totales"
        )
        self.constrs_yp['global_p'] = self.sub_yp.addConstr(
            gp.quicksum(self.yp[t] for t in self.V_list) == self.N - len(self.CH),
            name="triangulos_externos_totales"
        )

    def _add_A_prime_constrs_farkas(self):
        """Método auxiliar para añadir las restricciones de A' en los subproblemas."""
        A_prime = []
        for i in range(len(self.CH)):
            u, v = self.CH[i], self.CH[(i + 1) % len(self.CH)]
            if (u, v) in self.x:
                A_prime.append((u, v))

        for (i, j) in A_prime:
            self.constrs_y['alpha'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) == 0, name=f"alpha_{i}_{j}"
            )
            self.constrs_yp['alpha_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) == 0, name=f"alpha_p_{i}_{j}"
            )

    def _add_non_ch_constr_farkas(self):
        """Método auxiliar para añadir las restricciones de no-ch in los subproblemas."""
        for i in self.N_list:
            for j in self.N_list:
                is_non_ch = ((i not in self.CH) or (j not in self.CH))
                
                # Balance de aristas
                if is_non_ch and i < j:
                    self.constrs_y['beta'][i, j] = self.sub_y.addConstr(
                        gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.y[t] for t in self.triangles_adj_list[j][i]) == 0,
                        name=f"beta_{i}_{j}"
                    )
                    self.constrs_yp['beta_p'][i, j] = self.sub_yp.addConstr(
                        gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) - gp.quicksum(self.yp[t] for t in self.triangles_adj_list[j][i]) == 0,
                        name=f"beta_p_{i}_{j}"
                    )

                # Cotas superior e inferior
                if is_non_ch and i != j:
                    self.constrs_y['gamma'][i, j] = self.sub_y.addConstr(
                        gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) >= 0, name=f"gamma_{i}_{j}"
                    )
                    self.constrs_y['delta'][i, j] = self.sub_y.addConstr(
                        gp.quicksum(self.y[t] for t in self.triangles_adj_list[i][j]) <= 0, name=f"delta_{i}_{j}"
                    )
                    
                    self.constrs_yp['gamma_p'][i, j] = self.sub_yp.addConstr(
                        gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) >= 0, name=f"gamma_p_{i}_{j}"
                    )
                    self.constrs_yp['delta_p'][i, j] = self.sub_yp.addConstr(
                        gp.quicksum(self.yp[t] for t in self.triangles_adj_list[i][j]) <= 0, name=f"delta_p_{i}_{j}"
                    )