

from gurobipy import GRB
import gurobipy as gp
import os
from utils.utils import plot_solution

from models.benders.farkas import generate_farkas_cut, generate_farkas_cut_y, generate_farkas_cut_yp
from models.benders.pi import generate_pi_cut, generate_pi_cut_y, generate_pi_cut_yp
from models.benders.master import build_master_problem
from models.benders.utils import update_subproblem_rhs


def benders_callback(model, where):
    """
    Callback de Benders. Se ejecuta cuando el Maestro encuentra una solución entera (MIPSOL).
    Verifica la factibilidad en los DOS subproblemas independientes y añade cortes Lazy si es necesario.
    """
    if where == GRB.Callback.MIPSOL:
        # 1. Obtener la solución actual del Maestro (x barra)
        x_sol = model.cbGetSolution(model._x)

        update_subproblem_rhs(model, x_sol)

        model._sub_y.optimize()
        model._sub_yp.optimize()
        
        TOL = 1e-10
        if model._benders_method == "pi":
            if model._sub_y.ObjVal > TOL:
                cut_y_expr, cut_y_val, _ = generate_pi_cut_y(model._constrs_y, x_sol, model, TOL)
                if cut_y_val > TOL:
                    model.cbLazy(cut_y_expr <= 0)

            if model._sub_yp.ObjVal > TOL:
                cut_yp_expr, cut_yp_val, _ = generate_pi_cut_yp(model._constrs_yp, x_sol, model, TOL)
                if cut_yp_val > TOL:
                    model.cbLazy(cut_yp_expr <= 0)

        elif model._benders_method == "farkas":
            generate_farkas_cut(model._sub_y, model._sub_yp, model._constrs_y, model._constrs_yp, x_sol, model, TOL)

        else:
            print("Método de Benders desconocido en callback. Por favor, elige 'farkas' o 'pi'.")


    if where == GRB.Callback.MIPNODE:
        
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x_sol = model.cbGetNodeRel(model._x)

            update_subproblem_rhs(model, x_sol)

            model._sub_y.optimize()
            model._sub_yp.optimize()

            TOL = 1e-10

            if model._benders_method == "pi":
                generate_pi_cut(model._constrs_y, model._constrs_yp, x_sol, model, TOL)  

            elif model._benders_method == "farkas":
                generate_farkas_cut(model._sub_y, model._sub_yp, model._constrs_y, model._constrs_yp, x_sol, model, TOL)

            else:
                print("Método de Benders desconocido en callback. Por favor, elige 'farkas' o 'pi'.")



def optimize_master_MILP(instance_path: str, verbose: bool = False, plot: bool = False, 
                         time_limit: int = 7200, maximize: bool = True, save_cuts: bool = False,
                         crosses_constrain: bool = False, benders_method: str = "farkas") -> gp.Model:
    """
    Construye y resuelve el Problema Maestro (PM) usando Descomposición de Benders.
    """

    model = build_master_problem(
        instance_path,
        verbose=verbose,
        plot=plot,
        time_limit=time_limit,
        maximize=maximize,
        save_cuts=save_cuts,
        crosses_constrain=crosses_constrain,
        benders_method=benders_method
    )

    # --- Optimización ---
    if verbose:
        print("Starting optimization with Benders decomposition...")
    
    if model._save_cuts:
        model._instance_name = instance_path.split('/')[-1].replace('.instance', '')
        model._iteration = 0
        
        # NUEVO: Ruta para el log de rayos de Farkas
        if crosses_constrain:
            model._farkas_log_path = f"outputs/Others/Benders/{model._instance_name}-Crosses/farkas_log.jsonl"
        else:
            model._farkas_log_path = f"outputs/Others/Benders/{model._instance_name}/farkas_log.jsonl"
        # Limpiar el archivo si ya existe de una corrida anterior
        if os.path.exists(model._farkas_log_path):
            os.remove(model._farkas_log_path)

    model.optimize(benders_callback)
    x = model._x


    # --- Resultados ---
    model._x_results = []
    if model.SolCount > 0:
        for k, v in x.items():
            if v.X > 0.5:
                model._x_results.append(k)
        
        if plot:
            # Asumiendo que existe una función plot_solution en utils
            plot_solution(model, title="Optimal Tour" if model.Status == GRB.OPTIMAL else "Best Found")
    
    model._instance_name = instance_path.split('/')[-1].replace('.instance', '')
    model.write(f"outputs/Others/Benders/{model._instance_name}.lp")

    return model


def optimize_master_LP(instance_path: str, verbose: bool = False, plot: bool = False, 
                         time_limit: int = 7200, maximize: bool = True, save_cuts: bool = False,
                         crosses_constrain: bool = False, benders_method: str = "farkas") -> gp.Model:
    """
    Construye y resuelve la relajación LP del Problema Maestro (PM) usando Descomposición de Benders.
    """
    model = build_master_problem(
        instance_path,
        verbose=verbose,
        plot=plot,
        time_limit=time_limit,
        maximize=maximize,
        save_cuts=save_cuts,
        crosses_constrain=crosses_constrain,
        benders_method=benders_method
    )

    model.params.OutputFlag = 0
    model.params.Presolve = 1
    model.params.LazyConstraints = 0
    model.params.Presolve = 1


    # Cambiamos manualmente el tipo de todas las variables a continuas
    for v in model.getVars():
        if v.VType != GRB.CONTINUOUS:
            v.VType = GRB.CONTINUOUS
    model.update()
    
    sub_y = model._sub_y
    sub_yp = model._sub_yp

    constrs_y = model._constrs_y
    constrs_yp = model._constrs_yp

    model._iteration = 0
    TOL = 1e-10

    converged = False
    converged_sub_y = False
    converged_sub_yp = False

    if verbose:
        print("Starting optimization of LP relaxation with Benders decomposition...")
    

    while not converged:

        if verbose:
                print("\n=== Iteración: {} ===".format(model._iteration))


        model.optimize()
        x_sol = {k: v.X for k, v in model._x.items()}

        update_subproblem_rhs(model, x_sol)

        sub_y.optimize()
        sub_yp.optimize()

        if benders_method == "pi":
            # --- Evaluación Subproblema Y ---
            if sub_y.ObjVal <= TOL:
                converged_sub_y = True
            else:
                converged_sub_y = False
                cut_y_expr, cut_y_val, _ = generate_pi_cut_y(constrs_y, x_sol, model, TOL)

                # Como el Subproblema Fase 1 siempre busca minimizar una función objetivo positiva
                # (las variables artificiales >= 0), la violación (cut_y_val) siempre será positiva.
                # Por tanto, forzamos a que esa combinación valga <= 0.
                if cut_y_val > TOL:
                    model.addConstr(cut_y_expr <= 0, name=f"Phase1_Cut_Y_iter_{model._iteration}")

            # --- Evaluación Subproblema Y' ---
            if sub_yp.ObjVal <= TOL:
                converged_sub_yp = True
            else:
                converged_sub_yp = False
                cut_yp_expr, cut_yp_val, _ = generate_pi_cut_yp(constrs_yp, x_sol, model, TOL)

                if cut_yp_val > TOL:
                    model.addConstr(cut_yp_expr <= 0, name=f"Phase1_Cut_YP_iter_{model._iteration}")


        elif benders_method == "farkas":
                # --- Evaluación Subproblema Y ---

            if sub_y.Status == GRB.OPTIMAL:
                converged_sub_y = True # El maestro aproximó bien, terminamos.

            elif sub_y.Status == GRB.INFEASIBLE:
                converged_sub_y = False
                cut_y_expr, cut_y_val, _ = generate_farkas_cut_y(constrs_y, x_sol, model, TOL)

                if cut_y_val > TOL:
                    model.addConstr(cut_y_expr <= 0, name=f"benders_cut_y_{model._iteration}")
                elif cut_y_val < -TOL:
                    model.addConstr(cut_y_expr >= 0, name=f"benders_cut_y_{model._iteration}")
            else:
                converged_sub_y = False



            if sub_yp.Status == GRB.OPTIMAL:
                # Evaluar si el costo real del SP es mayor que la aproximación theta
                converged_sub_yp = True # El maestro aproximó bien, terminamos.

            elif sub_yp.Status == GRB.INFEASIBLE:
                converged_sub_yp = False
                cut_yp_expr, cut_yp_val, _ = generate_farkas_cut_yp(constrs_yp, x_sol, model, TOL)

                if cut_yp_val > TOL:
                    model.addConstr(cut_yp_expr <= 0, name=f"benders_cut_yp_{model._iteration}")
                elif cut_yp_val < -TOL:
                    model.addConstr(cut_yp_expr >= 0, name=f"benders_cut_yp_{model._iteration}")

            else:
                converged_sub_yp = False

            model._iteration += 1   

            if converged_sub_y and converged_sub_yp:
                converged = True
        
        else:
            print("Método de Benders desconocido. Por favor, elige 'farkas' o 'pi'.")
            break

    print("LP Relaxation converged after {} iterations.".format(model._iteration))

    return model