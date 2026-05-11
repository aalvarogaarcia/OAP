import os
from typing import Literal

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


def get_ObjVal_int(model: gp.Model) -> float | None:  # type: ignore[return]
    """
    Retorna el valor objetivo entero del modelo si existe solución.
    Si no hay solución, retorna None.
    """
    if model and model.SolCount > 0:
        x: list[tuple[int, int]] = model._x_results
        obj_val = 0.0
        for i, j in x:
            i = model._points_[i]
            j = model._points_[j]

            obj_val += (i[0] * j[1] - j[0] * i[1]) / 2

        return obj_val


def get_tour(model: gp.Model) -> list[int]:
    """
    Retorna la lista de aristas seleccionadas en la solución del modelo.
    Cada arista se representa como una tupla (i, j) de índices de puntos.
    """
    if model and model.SolCount > 0:
        x: list[tuple[int, int]] = model._x_results
        next_i = {i: j for i, j in x}

        start = x[0][0]  # Tomamos el primer punto como inicio
        tour = [start]
        actual = next_i[start]

        while actual != start:
            tour.append(actual)
            actual = next_i[actual]

        return tour

    else:
        return []


def get_Objval_lp(model: gp.Model) -> float:
    if model._benders_:
        # The Benders LP relaxation path was removed when the codebase migrated to
        # OAPBendersModel. Use OAPBendersModel.get_objval_lp() instead.
        raise NotImplementedError(
            "get_Objval_lp() does not support Benders models. "
            "Use OAPBendersModel.get_objval_lp() (via OAPStatsMixin) instead."
        )

    else:
        # 1. Crear la relajación lineal
        lp = model.relax()
        lp.Params.OutputFlag = 0  # Desactivar salida de Gurobi para la relajación

        # 2. Optimizar (SIN asignar el resultado a la variable lp)
        lp.optimize()

        os.makedirs("outputs/Others", exist_ok=True)

        lp.write("outputs/Others/LP_Relaxation_Converged_Compact.sol")
        lp.write("outputs/Others/LP_Relaxation_Converged_Compact.lp")

    obj_val = lp.ObjVal if lp.SolCount > 0 else 0

    return obj_val


def get_x_values(model: gp.Model) -> dict[str, float]:
    """
    Extrae los valores de las variables de decisión que comienzan por 'x'.
    Retorna un diccionario con {nombre_variable: valor}
    """
    # Verificar si el modelo tiene solución (Optimal o Suboptimal)
    if model and model.SolCount > 0:
        x_values: dict[str, float] = {}

        # model.getVars() devuelve TODAS las variables
        for var in model.getVars():
            if var.VarName.startswith("x_"):
                x_values[var.VarName] = var.X

        return x_values
    else:
        return {}


def get_model_stats(
    model: gp.Model,
) -> (
    tuple[float, float, float | None, float, float]
    | tuple[Literal["-"], Literal["-"], Literal["-"], Literal["-"], Literal["-"]]
):
    """
    Extrae estadísticas clave del modelo y su relajación.
    Retorna: (LP_Val, Gap, IP_Val, Time, Nodes)
    """

    if model and model.SolCount > 0:
        ip_val = get_ObjVal_int(model)
        if ip_val is None:
            return "-", "-", "-", "-", "-"
        time_s = model.Runtime
        nodes = model.NodeCount
        lp_val = get_Objval_lp(model)
    else:
        return "-", "-", "-", "-", "-"

    # Calcular Gap: (IP - LP )/ IP * 100 if MinArea (evitando división por cero)
    # Si MaxArea, el gap es (LP - IP) / (Area(CH)-IP) * 100
    gap = 0.0
    if ip_val != 0 and model.ModelSense == GRB.MINIMIZE:
        gap = (ip_val - lp_val) / ip_val * 100

    elif ip_val != 0 and model.ModelSense == GRB.MAXIMIZE:
        # Asumimos que el área del casco convexo es accesible como atributo
        area_ch = model._convex_hull_area if hasattr(model, "_convex_hull_area") else None
        # print(f"Convex Hull Area for gap calculation: {area_ch}")
        if area_ch is not None and (area_ch - ip_val) != 0:
            gap = ((lp_val - ip_val) / (area_ch - ip_val)) * 100

    return lp_val, gap, ip_val, time_s, nodes


#
# def get_model_stats_cplex(model: Model, relaxed_model: Model):
#    """
#    Extrae estadísticas clave del modelo CPLEX y su relajación.
#    Retorna: (LP_Val, Gap, IP_Val, Time, Nodes)
#    """
#
#    # 1. Validar si el modelo principal (IP) tiene solución
#    # En docplex, 'model.solution' es None si no se encontró solución.
#    if model and model.solution:
#        ip_val = model.objective_value
#
#        # El tiempo y los nodos se extraen de los detalles de la resolución
#        solve_details = model.get_solve_details()
#        time_s = solve_details.time  # Tiempo en segundos
#        nodes = solve_details.nb_nodes_processed
#    else:
#        return "-", "-", "-", "-", "-"
#
#    # 2. Validar si el modelo relajado (LP) tiene solución
#    if relaxed_model and relaxed_model.solution:
#        lp_val = relaxed_model.objective_value
#    else:
#        lp_val = 0 # O manejar error de relajación infactible
#
#    # 3. Calcular Gap
#    # (IP - LP )/ IP * 100 if MinArea (evitando división por cero)
#    # Si MaxArea, el gap es (LP - IP) / (Area(CH)-IP) * 100
#    gap = 0.0
#
#    # Detectar sentido de optimización (Minimizar vs Maximizar)
#    is_minimization = model.objective_sense.is_minimize()
#
#    if ip_val != 0 and is_minimization:
#        # Gap estándar para minimización
#        gap = (ip_val - lp_val) / ip_val * 100
#
#    elif ip_val != 0 and not is_minimization: # Maximización
#        # Asumimos que el área del casco convexo fue guardada en el atributo _convex_hull_area
#        area_ch = model._convex_hull_area if hasattr(model, '_convex_hull_area') else None
#
#        # print(f"Convex Hull Area for gap calculation: {area_ch}")
#
#        if area_ch is not None and (area_ch - ip_val) != 0:
#            # Fórmula específica para OAP (Area Poligonización)
#            gap = (lp_val - ip_val) / (area_ch - ip_val) * 100
#        else:
#            # Fallback a cálculo estándar si no hay Convex Hull info
#            gap = (lp_val - ip_val) / abs(ip_val) * 100
#
#    return lp_val, gap, ip_val, time_s, nodes
#
#


def save_results_excel(model: gp.Model, outputfile: str) -> None:
    """
    Guarda estadísticas y configuración del modelo en Excel.
    - Si el archivo existe, intenta leerlo.
    - Usa una hoja con el nombre de la instancia (o del archivo) y la crea si no existe.
    - Agrega la definición del modelo (sentido, objetivo, sum_constrain) y las métricas (LP, Gap, IP, Time, Nodes).
    """

    # 1) Obtener métricas básicas
    lp_val, gap, ip_val, time_s, nodes = get_model_stats(model)

    # 2) Obtener metadatos de configuración del modelo, con valores por defecto si no existen
    sense = (
        "Minimize"
        if getattr(model, "ModelSense", None) == GRB.MINIMIZE
        else "Maximize"
        if getattr(model, "ModelSense", None) == GRB.MAXIMIZE
        else "Unknown"
    )
    objective_desc = getattr(model, "_objective_desc", "Unknown")
    sum_constrain = getattr(model, "_sum_constrain", None)
    instance_name = getattr(model, "_instance_name", None)

    # Determinar el nombre de la hoja: preferimos el nombre de la instancia; si no, usamos el nombre del archivo
    sheet_name = instance_name if instance_name else os.path.splitext(os.path.basename(outputfile))[0]

    # 3) Construir fila de datos a insertar
    new_row = {
        "Instance": instance_name if instance_name else "-",
        "Sense": sense,
        "Objective": objective_desc,
        "SumConstrain": sum_constrain if sum_constrain is not None else "-",
        "LP Value": lp_val,
        "Gap (%)": gap,
        "IP Value": ip_val,
        "Time (s)": time_s,
        "Nodes": nodes,
    }
    new_df = pd.DataFrame([new_row])

    # 4) Leer archivo/hoja si existe y combinar
    file_exists = os.path.exists(outputfile)
    sheet_exists = False
    existing_df = None

    if file_exists:
        try:
            xls = pd.ExcelFile(outputfile)
            sheet_exists = sheet_name in xls.sheet_names
            if sheet_exists:
                existing_df = pd.read_excel(outputfile, sheet_name=sheet_name)
        except Exception:
            # Si no se puede leer por alguna razón, continuamos creando/escribiendo
            existing_df = None
            sheet_exists = False

    combined_df = new_df if existing_df is None else pd.concat([existing_df, new_df], ignore_index=True)

    # 5) Escribir al Excel (reemplazar hoja si existe, o crear nueva; preservar otras hojas)
    if file_exists:
        if sheet_exists:
            with pd.ExcelWriter(outputfile, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(outputfile, engine="openpyxl", mode="a") as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Crear nuevo libro
        with pd.ExcelWriter(outputfile, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
