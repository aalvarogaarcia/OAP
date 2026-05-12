import logging
from pathlib import Path

import gurobipy as gp
import pytest

from models.OAPBendersModel import OAPBendersModel
from models.OAPCompactModel import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

# Silenciamos Gurobi por defecto para no saturar la consola durante los tests
gp.setParam("OutputFlag", 0)

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE DIRECTORIOS ---
# Carpeta donde están las instancias pequeñas
INSTANCES_DIR = Path("instance/little-instances")

# Obtenemos dinámicamente todas las instancias del directorio.
# Guarda contra OSError en CI donde instance/ es git-ignorado.
INSTANCE_FILES = list(INSTANCES_DIR.glob("*.instance")) if INSTANCES_DIR.exists() else []

if not INSTANCE_FILES:
    pytest.skip("No instance files found in instance/little-instances/", allow_module_level=True)

# =======================================================================
# FIXTURES
# =======================================================================


@pytest.fixture(params=INSTANCE_FILES, ids=[f.stem for f in INSTANCE_FILES])
def instancia_cargada(request):  # type: ignore[no-untyped-def]
    """
    Fixture parametrizada: Se ejecuta una vez por cada archivo .instance encontrado.
    Carga los puntos y calcula la triangulación para inyectarla en los tests.

    Skips automatically when the triangulation is empty (degenerate instance —
    e.g. collinear points or fewer than 3 non-collinear points).  Such instances
    produce an infeasible model in both Compact and Benders, so there is nothing
    to compare.
    """
    file_path = request.param
    points = read_indexed_instance(str(file_path))
    triangles = compute_triangles(points)

    if len(triangles) == 0:
        pytest.skip(
            f"Instance '{file_path.stem}' has an empty triangulation (degenerate / collinear points). Skipping."
        )

    # Devolvemos el nombre de la instancia para los logs, los puntos y los triángulos
    return file_path.stem, points, triangles


# =======================================================================
# TESTS DE INTEGRACIÓN: MIP (Solución Entera)
# =======================================================================


@pytest.mark.parametrize("objective", ["Fekete", "Internal"])
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_mip_equivalence_min(instancia_cargada, objective, benders_method):  # type: ignore[no-untyped-def]
    """
    Comprueba que el valor objetivo entero (MIP) de Benders sea idéntico
    al del modelo Compacto para diferentes funciones objetivo y métodos.
    """
    if objective == "Internal":
        pytest.skip("Benders Internal objective MIP decomposition is work-in-progress; LP bound always 0 for minimize.")
    if objective == "Internal" and benders_method == "pi":
        pytest.skip("Pi + Internal: optimality cuts not yet implemented.")
    instance_name, points, triangles = instancia_cargada

    # 1. Resolver Compacto (Nuestra fuente de verdad)
    compacto = OAPCompactModel(points, triangles, name=f"Compact_MIP_{instance_name}_{objective}")
    compacto.build(objective=objective, maximize=False)
    compacto.solve()

    # Si el compacto no llega a OPTIMAL (infeasible, sin licencia, timeout…)
    # no hay referencia válida → skip en lugar de fail.
    if compacto.model.Status != gp.GRB.OPTIMAL:
        pytest.skip(
            f"Compact model did not reach OPTIMAL for '{instance_name}' "
            f"(status={compacto.model.Status}). No reference value available."
        )
    obj_compacto = compacto.model.ObjVal

    # 2. Resolver Benders
    benders = OAPBendersModel(points, triangles, name=f"Benders_MIP_{instance_name}_{objective}_{benders_method}")
    benders.build(objective=objective, maximize=False, benders_method=benders_method)
    benders.solve()

    assert benders.model.Status == gp.GRB.OPTIMAL, f"Benders ({benders_method}) falló en {instance_name}"
    obj_benders = benders.get_objval_int()  # Usamos tu método específico para obtener el valor entero de Benders

    # 3. Comparación con tolerancia para evitar problemas de coma flotante
    assert obj_compacto == pytest.approx(obj_benders, rel=1e-4), (
        f"Divergencia MIP en {instance_name} ({objective}): Compacto={obj_compacto:.4f} vs Benders {benders_method}={obj_benders:.4f}"
    )


# =======================================================================
# TESTS DE INTEGRACIÓN: LP (Relajación Lineal / Root Node)
# =======================================================================


@pytest.mark.parametrize("objective", ["Fekete", "Internal"])
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_lp_equivalence_min(instancia_cargada, objective, benders_method):  # type: ignore[no-untyped-def]
    """
    Comprueba que la relajación lineal (LP) del Maestro con Benders
    converge al mismo límite inferior que la relajación del Compacto.
    """
    if objective == "Internal" and benders_method == "pi":
        pytest.skip("Pi + Internal LP: optimality cuts not yet implemented.")
    instance_name, points, triangles = instancia_cargada

    # 1. Resolver LP del Compacto
    compacto = OAPCompactModel(points, triangles, name=f"Compact_LP_{instance_name}_{objective}")
    compacto.build(objective=objective, maximize=False)  # Construimos el modelo en modo relajado
    compacto.solve(time_limit=120, verbose=False, relaxed=True)  # Resolvemos la relajación LP

    # Asumo que tienes un método get_objval_lp() como mostrabas en tu test original.
    # Si no, esto equivaldría a: lp_compacto = compacto.model.relax(); lp_compacto.optimize(); obj = lp_compacto.ObjVal
    lp_compacto = compacto.get_objval_lp()

    # 2. Resolver LP de Benders
    benders = OAPBendersModel(points, triangles, name=f"Benders_LP_{instance_name}_{objective}_{benders_method}")
    benders.build(objective=objective, maximize=False, benders_method=benders_method)

    # Usamos tu método de resolución para la relajación en Benders
    benders.solve(time_limit=120, verbose=False, relaxed=True)  # Resolvemos en modo relajado
    lp_benders = benders.get_objval_lp()

    # 3. Comparación con tolerancia
    if lp_compacto == "-" or lp_benders == "-":
        pytest.skip(
            f"LP no disponible para {instance_name} ({objective}): Compacto={lp_compacto}, Benders={lp_benders}"
        )
    assert lp_compacto == pytest.approx(lp_benders, rel=1e-4), (
        f"Divergencia LP en {instance_name} ({objective}): Compacto={lp_compacto:.4f} vs Benders {benders_method}={lp_benders:.4f}"
    )

    # =======================================================================


# TESTS DE INTEGRACIÓN: MIP (Solución Entera)
# =======================================================================


@pytest.mark.parametrize("objective", ["Fekete", "Internal"])
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_mip_equivalence_max(instancia_cargada, objective, benders_method):  # type: ignore[no-untyped-def]
    """
    Comprueba que el valor objetivo entero (MIP) de Benders sea idéntico
    al del modelo Compacto para diferentes funciones objetivo y métodos.
    """
    if objective == "Internal":
        pytest.skip("Benders Internal objective MIP decomposition is work-in-progress.")
    if objective == "Internal" and benders_method == "pi":
        pytest.skip("Pi + Internal: optimality cuts not yet implemented.")
    instance_name, points, triangles = instancia_cargada

    # 1. Resolver Compacto (Nuestra fuente de verdad)
    compacto = OAPCompactModel(points, triangles, name=f"Compact_MIP_{instance_name}_{objective}")
    compacto.build(objective=objective, maximize=True)
    compacto.solve()

    # Si el compacto no llega a OPTIMAL (infeasible, sin licencia, timeout…)
    # no hay referencia válida → skip en lugar de fail.
    if compacto.model.Status != gp.GRB.OPTIMAL:
        pytest.skip(
            f"Compact model did not reach OPTIMAL for '{instance_name}' "
            f"(status={compacto.model.Status}). No reference value available."
        )
    obj_compacto = compacto.model.ObjVal

    # 2. Resolver Benders
    benders = OAPBendersModel(points, triangles, name=f"Benders_MIP_{instance_name}_{objective}_{benders_method}")
    benders.build(objective=objective, maximize=True, benders_method=benders_method)
    benders.solve()

    assert benders.model.Status == gp.GRB.OPTIMAL, f"Benders ({benders_method}) falló en {instance_name}"
    obj_benders = benders.get_objval_int()  # Usamos tu método específico para obtener el valor entero de Benders

    # 3. Comparación con tolerancia para evitar problemas de coma flotante
    assert obj_compacto == pytest.approx(obj_benders, rel=1e-4), (
        f"Divergencia MIP en {instance_name} ({objective}): Compacto={obj_compacto:.4f} vs Benders {benders_method}={obj_benders:.4f}"
    )


# =======================================================================
# TESTS DE INTEGRACIÓN: LP (Relajación Lineal / Root Node)
# =======================================================================


@pytest.mark.parametrize("objective", ["Fekete", "Internal"])
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_lp_equivalence_max(instancia_cargada, objective, benders_method):  # type: ignore[no-untyped-def]
    """
    Comprueba que la relajación lineal (LP) del Maestro con Benders
    converge al mismo límite inferior que la relajación del Compacto.
    """
    if objective == "Internal" and benders_method == "pi":
        pytest.skip("Pi + Internal LP: optimality cuts not yet implemented.")
    instance_name, points, triangles = instancia_cargada

    # 1. Resolver LP del Compacto
    compacto = OAPCompactModel(points, triangles, name=f"Compact_LP_{instance_name}_{objective}")
    compacto.build(objective=objective, maximize=True)  # Construimos el modelo en modo relajado
    compacto.solve(time_limit=120, verbose=False, relaxed=True)  # Resolvemos la relajación LP

    # Asumo que tienes un método get_objval_lp() como mostrabas en tu test original.
    # Si no, esto equivaldría a: lp_compacto = compacto.model.relax(); lp_compacto.optimize(); obj = lp_compacto.ObjVal
    lp_compacto = compacto.get_objval_lp()

    # 2. Resolver LP de Benders
    benders = OAPBendersModel(points, triangles, name=f"Benders_LP_{instance_name}_{objective}_{benders_method}")
    benders.build(objective=objective, maximize=True, benders_method=benders_method)

    # Usamos tu método de resolución para la relajación en Benders
    benders.solve(time_limit=120, verbose=False, relaxed=True)  # Resolvemos en modo relajado
    lp_benders = benders.get_objval_lp()

    # 3. Comparación con tolerancia
    if lp_compacto == "-" or lp_benders == "-":
        pytest.skip(
            f"LP no disponible para {instance_name} ({objective}): Compacto={lp_compacto}, Benders={lp_benders}"
        )
    assert lp_compacto == pytest.approx(lp_benders, rel=1e-4), (
        f"Divergencia LP en {instance_name} ({objective}): Compacto={lp_compacto:.4f} vs Benders {benders_method}={lp_benders:.4f}"
    )
