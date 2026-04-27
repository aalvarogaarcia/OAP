# models/mixin/benders_farkas_mixin.py
import logging

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from models.typing_oap import IndexArray, TrianglesAdjList
from utils.utils import plot_strengthening_constraints, signed_area


# Instanciamos el logger para este módulo
logger = logging.getLogger(__name__)

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
    
    def build_farkas_subproblems(self, sum_constrain: bool = True, strengthen: bool = False, plot_strengthen: bool = False) -> None:
        """Construye los Subproblemas (SP_Y y SP_YP) de factibilidad."""
        print("Construyendo subproblemas con método Farkas...")

        # 1. Configuración de parámetros vitales para extraer Farkas Duals
        for sub_model in [self.sub_y, self.sub_yp]:
            sub_model.Params.InfUnbdInfo = 1
            sub_model.Params.DualReductions = 0
            sub_model.Params.Presolve = 1
        
        # 2. Creación de variables Y y YP en sus respectivos modelos
        self.y = self.sub_y.addVars(self.V_list, vtype=GRB.CONTINUOUS, lb=0, name="y")

        if hasattr(self, 'objective') and self.objective == "Internal":
            at = [np.abs(signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]])) for tri in self.triangles]
            sense = self.model.ModelSense if hasattr(self, 'model') else GRB.MINIMIZE
            self.sub_y.setObjective(gp.quicksum(at[t] * self.y[t] for t in self.V_list), sense)


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

        if strengthen:
            self._add_strengthening_farkas(plot=plot_strengthen)

        self.sub_y.update()
        self.sub_yp.update()


    def _add_strengthening_farkas(self, plot: bool = False) -> None:
        """Add LP-strengthening constraints to both Farkas subproblems.

        Benders split of the compact-model constraints:
          R1  Σ at*y_t  ≤ CH_area        (sub_y / sub_yp, fixed RHS)
          R2  Σ_{t∈V_ij} y_t ≤ 1        (per arc, fixed RHS)
          R3  Σ y_t[i,j] + Σ y_t[k,l] ≤ 1 - x[j,i] - x[k,l]  (sub_y, param. RHS)
          R4  Σ yp_t[i,j] + Σ yp_t[k,l] ≤ 1 - x[i,j] - x[l,k] (sub_yp, param. RHS)
        """
        if not hasattr(self, '_abs_areas'):
            self._abs_areas = [
                abs(signed_area(self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]))
                for tri in self.triangles
            ]
        at = self._abs_areas

        # R1: area balance
        self.constrs_y['r1'] = self.sub_y.addConstr(
            gp.quicksum(at[t] * self.y[t] for t in self.V_list) <= self.convex_hull_area,
            name="r1_area_balance_y",
        )
        self.constrs_yp['r1_p'] = self.sub_yp.addConstr(
            gp.quicksum(at[t] * self.yp[t] for t in self.V_list) <= self.convex_hull_area,
            name="r1_area_balance_yp",
        )

        crossing_pairs = self._compute_crossing_arc_pairs()
        self.constrs_y['r2'] = {}
        self.constrs_yp['r2_p'] = {}
        self.constrs_y['r3'] = {}
        self.constrs_yp['r3_p'] = {}

        for (i, j) in self.x.keys():
            tris = self.triangles_adj_list[i][j]
            if not tris:
                continue

            # R2: arc coverage
            self.constrs_y['r2'][i, j] = self.sub_y.addConstr(
                gp.quicksum(self.y[t] for t in tris) <= 1,
                name=f"r2_arc_coverage_y_{i}_{j}",
            )
            self.constrs_yp['r2_p'][i, j] = self.sub_yp.addConstr(
                gp.quicksum(self.yp[t] for t in tris) <= 1,
                name=f"r2_arc_coverage_yp_{i}_{j}",
            )

            # R3 / R4: crossing exclusion (one constraint per ordered crossing pair)
            for (k, l) in self.x.keys():
                if (i, j) == (k, l) or ((i, j), (k, l)) not in crossing_pairs:
                    continue
                tris_kl = self.triangles_adj_list[k][l]
                if not tris_kl:
                    continue
                key = (i, j, k, l)

                # R3 in sub_y — RHS updated each iteration: 1 - x[j,i] - x[k,l]
                self.constrs_y['r3'][key] = self.sub_y.addConstr(
                    gp.quicksum(self.y[t] for t in tris)
                    + gp.quicksum(self.y[t] for t in tris_kl) <= 1,
                    name=f"r3_crossing_y_{i}_{j}_{k}_{l}",
                )

                # R4 in sub_yp — RHS updated each iteration: 1 - x[i,j] - x[l,k]
                self.constrs_yp['r3_p'][key] = self.sub_yp.addConstr(
                    gp.quicksum(self.yp[t] for t in tris)
                    + gp.quicksum(self.yp[t] for t in tris_kl) <= 1,
                    name=f"r3_crossing_yp_{i}_{j}_{k}_{l}",
                )

        if plot:
            plot_strengthening_constraints(
                points=self.points,
                ch=self.CH,
                x_keys=list(self.x.keys()),
                crossing_pairs=crossing_pairs,
            )

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

        # R1: area balance (fixed RHS = CH_area)
        if 'r1' in self.constrs_y:
            f_r1 = self.constrs_y['r1'].FarkasDual
            if abs(f_r1) > TOL:
                cut_y_expr += f_r1 * self.convex_hull_area
                cut_y_val += f_r1 * self.convex_hull_area

        # R2: arc coverage (fixed RHS = 1)
        for (i, j), constr in self.constrs_y.get('r2', {}).items():
            f_r2 = constr.FarkasDual
            if abs(f_r2) > TOL:
                cut_y_expr += f_r2
                cut_y_val += f_r2

        # R3: crossing exclusion — RHS = 1 - x[j,i] - x[k,l]
        for (i, j, k, l), constr in self.constrs_y.get('r3', {}).items():
            f_r3 = constr.FarkasDual
            if abs(f_r3) > TOL:
                cut_y_expr += f_r3 * (1 - self.x[j, i] - self.x[k, l])
                cut_y_val += f_r3 * (1 - x_sol.get((j, i), 0.0) - x_sol.get((k, l), 0.0))

        sense = None
        if cut_y_val > TOL:
            sense = "<="  # cut_expr <= 0
        elif cut_y_val < -TOL:
            sense = ">="  # cut_expr >= 0

        self._log_and_print_farkas(v_comps, cut_y_val, "Y", TOL, x_sol, cut_y_expr, sense)
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

        # R1_p: area balance (fixed RHS = CH_area)
        if 'r1_p' in self.constrs_yp:
            f_r1_p = self.constrs_yp['r1_p'].FarkasDual
            if abs(f_r1_p) > TOL:
                cut_yp_expr += f_r1_p * self.convex_hull_area
                cut_yp_val += f_r1_p * self.convex_hull_area

        # R2_p: arc coverage (fixed RHS = 1)
        for (i, j), constr in self.constrs_yp.get('r2_p', {}).items():
            f_r2_p = constr.FarkasDual
            if abs(f_r2_p) > TOL:
                cut_yp_expr += f_r2_p
                cut_yp_val += f_r2_p

        # R4: crossing exclusion — RHS = 1 - x[i,j] - x[l,k]
        for (i, j, k, l), constr in self.constrs_yp.get('r3_p', {}).items():
            f_r4 = constr.FarkasDual
            if abs(f_r4) > TOL:
                cut_yp_expr += f_r4 * (1 - self.x[i, j] - self.x[l, k])
                cut_yp_val += f_r4 * (1 - x_sol.get((i, j), 0.0) - x_sol.get((l, k), 0.0))

        sense = None
        if cut_yp_val > TOL:
            sense = ">="  # 0 >= cut_expr
        elif cut_yp_val < -TOL:
            sense = "<="  # 0 <= cut_expr

        self._log_and_print_farkas(v_comps_p, cut_yp_val, "Y'", TOL, x_sol, cut_yp_expr, sense)

        return cut_yp_expr, cut_yp_val

    
    def get_optimality_cut_y(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de optimalidad usando variables duales (.Pi) del subproblema Y."""
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

        if 'r1' in self.constrs_y:
            pi_r1 = self.constrs_y['r1'].Pi
            if abs(pi_r1) > TOL:
                cut_y_expr += pi_r1 * self.convex_hull_area
                cut_y_val += pi_r1 * self.convex_hull_area

        for (i, j), constr in self.constrs_y.get('r2', {}).items():
            pi_r2 = constr.Pi
            if abs(pi_r2) > TOL:
                cut_y_expr += pi_r2
                cut_y_val += pi_r2

        for (i, j, k, l), constr in self.constrs_y.get('r3', {}).items():
            pi_r3 = constr.Pi
            if abs(pi_r3) > TOL:
                cut_y_expr += pi_r3 * (1 - self.x[j, i] - self.x[k, l])
                cut_y_val += pi_r3 * (1 - x_sol.get((j, i), 0.0) - x_sol.get((k, l), 0.0))

        if hasattr(self, '_log_and_print_pi'):
            self._log_and_print_pi(v_comps, cut_y_val, "Y_OPT", TOL, x_sol, cut_y_expr, ">=")

        return cut_y_expr, cut_y_val

    def get_optimality_cut_yp(self, x_sol: dict[Arc, float], TOL: float = 1e-10) -> tuple[gp.LinExpr, float]:
        """Extrae el corte de optimalidad usando variables duales (.Pi) del subproblema Y'."""
        cut_yp_expr = gp.LinExpr()
        cut_yp_val = 0.0
        v_comps: RayComponents = {'alpha_p': {}, 'beta_p': {}, 'gamma_p': {}, 'delta_p': {}}
        
        # FIX: Ahora extrae de constrs_yp correctamente
        for (i, j), constr in self.constrs_yp['alpha_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['alpha_p'][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['beta_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['beta_p'][i, j] = pi
                cut_yp_expr += pi * (self.x[j, i] - self.x[i, j])
                cut_yp_val += pi * (x_sol[j, i] - x_sol[i, j])
                
        for (i, j), constr in self.constrs_yp['gamma_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['gamma_p'][i, j] = pi
                cut_yp_expr += pi * self.x[j, i]
                cut_yp_val += pi * x_sol[j, i]
                
        for (i, j), constr in self.constrs_yp['delta_p'].items():
            pi = constr.Pi
            if abs(pi) > TOL:
                v_comps['delta_p'][i, j] = pi
                cut_yp_expr += pi * (1 - self.x[i, j])
                cut_yp_val += pi * (1 - x_sol[i, j])
        
        if 'global_p' in self.constrs_yp:
            pi_global = self.constrs_yp['global_p'].Pi
            if abs(pi_global) > TOL:
                rhs_global = self.constrs_yp['global_p'].RHS
                v_comps['global_p'] = pi_global
                cut_yp_expr += pi_global * rhs_global
                cut_yp_val += pi_global * rhs_global

        if 'r1_p' in self.constrs_yp:
            pi_r1_p = self.constrs_yp['r1_p'].Pi
            if abs(pi_r1_p) > TOL:
                cut_yp_expr += pi_r1_p * self.convex_hull_area
                cut_yp_val += pi_r1_p * self.convex_hull_area

        for (i, j), constr in self.constrs_yp.get('r2_p', {}).items():
            pi_r2_p = constr.Pi
            if abs(pi_r2_p) > TOL:
                cut_yp_expr += pi_r2_p
                cut_yp_val += pi_r2_p

        for (i, j, k, l), constr in self.constrs_yp.get('r3_p', {}).items():
            pi_r4 = constr.Pi
            if abs(pi_r4) > TOL:
                cut_yp_expr += pi_r4 * (1 - self.x[i, j] - self.x[l, k])
                cut_yp_val += pi_r4 * (1 - x_sol.get((i, j), 0.0) - x_sol.get((l, k), 0.0))

        if hasattr(self, '_log_and_print_pi'):
            self._log_and_print_pi(v_comps, cut_yp_val, "Y'_OPT", TOL, x_sol, cut_yp_expr, ">=")

        return cut_yp_expr, cut_yp_val
    

    def _log_and_print_farkas(self, v_components, cut_val, sub_name, TOL, x_sol, cut_expr, sense=None):
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
        if getattr(self, 'save_cuts', False) and hasattr(self, 'log_path'):
            try:
                from utils.utils import log_benders_cut
                
                log_benders_cut(
                    filepath=self.log_path,
                    iteration=self.iteration,
                    node_depth=0,
                    subproblem_type='Y' if sub_name == 'Y' else 'Y_prime',
                    x_sol=x_sol,
                    v_components=v_components,
                    cut_value=cut_val,
                    tolerance=TOL,
                    cut_expr=cut_expr,
                    sense=sense
                )
            except NameError:
                logger.warning("No se pudo guardar el log estructurado: log_benders_cut no está definido.")
            

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