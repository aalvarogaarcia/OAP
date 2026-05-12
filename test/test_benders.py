import logging
from pathlib import Path

import pytest

from models.OAPBendersModel import OAPBendersModel
from models.OAPCompactModel import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE LA INSTANCIA DE PRUEBA ---
# Usamos una instancia pequeña (10 o 15 nodos) para que los tests pasen rápido
INSTANCE_PATH = "instance/us-night-0000010.instance"


@pytest.fixture(scope="module")
def instancia_prueba():  # type: ignore[no-untyped-def]
    """Carga los datos una sola vez para todos los tests de este módulo."""
    path = Path(INSTANCE_PATH)
    if not path.exists():
        pytest.skip(f"No se encontró la instancia en {path}")

    points = read_indexed_instance(str(path))
    triangles = compute_triangles(points)
    return points, triangles


# =======================================================================
# TEST 1: Equivalencia del Modelo Entero (MIP)
# =======================================================================
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_mip_equivalence(instancia_prueba, benders_method):  # type: ignore[no-untyped-def]
    """Verifica que el valor objetivo MIP de Benders sea idéntico al del Compacto."""
    points, triangles = instancia_prueba

    # 1. Resolver Compacto (Nuestra fuente de verdad)
    compacto = OAPCompactModel(points, triangles, name="Test_Compact_MIP")
    compacto.build(objective="Fekete", maximize=False)
    compacto.solve(time_limit=120, verbose=False)
    obj_compacto = compacto.get_objval_int()

    # 2. Resolver Benders
    benders = OAPBendersModel(points, triangles, name=f"Test_Benders_MIP_{benders_method}")
    benders.build(objective="Fekete", maximize=False, benders_method=benders_method)
    benders.solve(time_limit=120, verbose=False)
    obj_benders = benders.get_objval_int()

    # 3. Comprobaciones de seguridad
    assert obj_compacto is not None, "El modelo compacto no encontró solución factible."
    assert obj_benders is not None, f"Benders ({benders_method}) no encontró solución factible."

    # 4. Aserción matemática (usamos approx por los decimales de los floats en Gurobi)
    assert obj_compacto == pytest.approx(obj_benders, rel=1e-4), (
        f"Divergencia MIP: Compacto={obj_compacto:.4f} vs Benders {benders_method}={obj_benders:.4f}"
    )


# =======================================================================
# TEST 2: Equivalencia de la Relajación Lineal (LP)
# =======================================================================
@pytest.mark.parametrize("benders_method", ["farkas", "pi"])
def test_benders_lp_equivalence(instancia_prueba, benders_method):  # type: ignore[no-untyped-def]
    """Verifica que el límite inferior (Root Node Bound LP) sea idéntico."""
    points, triangles = instancia_prueba

    # 1. Resolver LP del Compacto
    compacto = OAPCompactModel(points, triangles, name="Test_Compact_LP")
    compacto.build(objective="Fekete", maximize=False)
    lp_compacto = compacto.get_objval_lp()

    # 2. Resolver LP de Benders (con el bucle manual que creamos)
    benders = OAPBendersModel(points, triangles, name=f"Test_Benders_LP_{benders_method}")
    benders.build(objective="Fekete", maximize=False, benders_method=benders_method)
    benders.solve_lp_relaxation(time_limit=120, verbose=False)
    lp_benders = benders.model.ObjVal

    # 3. Comprobaciones de seguridad
    assert lp_compacto != "-", "El modelo compacto falló al calcular el LP."

    # 4. Aserción matemática
    assert lp_compacto == pytest.approx(lp_benders, rel=1e-4), (
        f"Divergencia LP: Compacto={lp_compacto:.4f} vs Benders {benders_method}={lp_benders:.4f}"
    )
