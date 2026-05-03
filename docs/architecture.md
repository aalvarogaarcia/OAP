# Arquitectura del framework OAP·NextGen

## Visión general

OAP·NextGen implementa un **branch-and-cut auténtico** basado en Benders decomposition para resolver los problemas MIN-OAP y MAX-OAP. Sigue fielmente el esquema de Hernández-Pérez et al. (2025) con callback de Gurobi (`LazyConstraints=1`), dos subproblemas separados con extracción de `FarkasDual`, y parámetros explícitos `InfUnbdInfo=1`, `DualReductions=0`.

---

## Jerarquía de clases

```
OAPStatsMixin
└── OAPBaseModel
    ├── OAPCompactModel
    │   └── [formulación monolítica: DFJ / MTZ / SCF]
    └── OAPBendersModel (MRO)
        ├── BendersMasterMixin      — maestro + variables x, f, eta
        ├── BendersFarkasMixin      — cortes de factibilidad (rayos Farkas)
        ├── BendersPiMixin          — cortes de optimalidad (variables Pi)
        ├── BendersOptimizeMixin    — callback, bucle LP, actualización RHS
        └── BendersAnalysisMixin    — diagnóstico post-resolución
```

El orden MRO es crítico: Python resuelve las llamadas de método de izquierda a derecha.

---

## Módulos y responsabilidades

### `models/OAPBaseModel.py`

Clase base abstracta. Proporciona:

- Cálculo de envolvente convexa (`CH`) y área.
- Triangulación de Delaunay y lista de adyacencia `triangles_adj_list`.
- Extracción de facetas del poliedro LP (`extract_facets`, `extract_subspace_facets`) usando `pycddlib`.
- Serialización de facetas a JSONL (`log_facets`).

### `models/OAPCompactModel.py`

Formulación MILP monolítica. Parámetros clave del método `build()`:

| Parámetro | Opciones | Descripción |
|---|---|---|
| `objective` | `"Fekete"`, `"Internal"`, `"External"`, `"Diagonals"` | Función objetivo |
| `subtour` | `"DFJ"`, `"MTZ"`, `"SCF"` | Eliminación de subtours |
| `semiplane` | `0`, `1`, `2` | Restricciones de semiplano (strengthening) |
| `use_knapsack` | `bool` | Cortes de mochila sobre aristas cruzadas |
| `use_cliques` | `bool` | Cortes de clique sobre cruces |

**DFJ** (Dantzig-Fleischer-Johnson) es la formulación más fuerte para eliminación de subtours y debe ser el valor por defecto para benchmarks serios.

### `models/OAPBendersModel.py`

Punto de entrada de la descomposición. El método `build()` orquesta:

1. `build_master()` → variables `x` (binarias), `f` (flujo SCF), opcionalmente `eta`.
2. `build_subproblems_farkas()` o `build_subproblems_pi()` según `benders_method`.

### `models/mixin/benders_master_mixin.py`

Construye el problema maestro. Contiene:

- Variables `x[i,j]` binarias sobre el grafo completo dirigido.
- Variables `f[i,j]` de flujo para eliminación de subtours (actualmente SCF).
- Variable `eta` (solo si `objective == "Internal"`).
- Limpieza de aristas de la convex hull: las aristas de CH en sentido contrario se eliminan para reducir simetría.

**Deuda técnica**: la eliminación de subtours usa SCF (Single Commodity Flow), que genera la relajación LP más débil. Migrar a DFJ con SECs lazy es la mejora de mayor impacto (ver `docs/technical_debt.md`).

### `models/mixin/benders_farkas_mixin.py`

Subproblemas de factibilidad. Extrae rayos de Farkas cuando `sub_y` o `sub_yp` son infactibles:

- `get_farkas_cut_y(x_sol)`: corte sobre triángulos interiores a la CH.
- `get_farkas_cut_yp(x_sol)`: corte sobre triángulos exteriores a la CH.
- `_log_and_print_farkas()`: logging estructurado (activo solo si `save_cuts=True`).

Los componentes del rayo se extraen directamente de los atributos Gurobi `FarkasDual` y `FarkasProof`.

### `models/mixin/benders_pi_mixin.py`

Subproblemas de optimalidad. Usa variables duales `.Pi` en lugar de rayos de Farkas:

- `get_pi_cut_y(x_sol)`: corte dual para subproblema Y.
- `get_pi_cut_yp(x_sol)`: corte dual para subproblema Y'.

Componentes de corte: `alpha`, `beta`, `gamma`, `delta`, `global`, `r1`, `r2`, `r3`.

**Bug conocido**: existe un `UnboundLocalError: sense` en versiones anteriores al commit que añade `else: sense = "=="` (líneas 233-238 de `benders_pi_mixin.py`). Verificar con `git log --oneline` que ese fix está en `main`.

### `models/mixin/benders_optimize_mixin.py`

Orquestación del solve. Dos modos:

**1. Modo MIP** (`solve()`):
- `model.optimize(self._benders_callback)` con `LazyConstraints=1`.
- El callback intercepta `MIPSOL`, actualiza RHS, resuelve subproblemas, inyecta cortes lazy.

**2. Modo LP** (`solve_lp_relaxation()`):
- Bucle manual: resolver maestro LP → actualizar RHS → resolver subs → añadir corte duro → repetir.
- Necesario porque Gurobi no dispara `MIPSOL` en modelos LP puros.

`_update_subproblem_rhs(x_sol)` actualiza los RHS de todas las restricciones de ambos subproblemas en función de la solución propuesta por el maestro.

### `models/mixin/benders_analysis_mixin.py`

Diagnóstico post-resolución: estadísticas de cortes añadidos, distribución por tipo, visualizaciones.

---

## Utils

| Módulo | Contenido |
|---|---|
| `utils/geometry.py` | Geometría pura, I/O de instancias, alias de tipos (`Arc`, `PointLookup`) |
| `utils/constraints.py` | Inyectores de restricciones Gurobi, `ArcConstraintMap` |
| `utils/benders_log.py` | Logging de cortes en JSONL, tipos serializados |
| `utils/visualization.py` | Visualizaciones (Matplotlib / Plotly) |
| `utils/utils.py` | **Solo re-exportación** — no añadir lógica aquí |

---

## Flujo de ejecución estándar

```python
from utils.geometry import read_indexed_instance, compute_triangles
from models import OAPBendersModel

points    = read_indexed_instance("instance/example.instance")
triangles = compute_triangles(points)

model = OAPBendersModel(points, triangles, name="example")
model.build(
    objective="Fekete",
    maximize=False,
    benders_method="farkas",
    sum_constrain=True,
)
model.solve(time_limit=600, verbose=True, save_cuts=False)

lp, gap, ip, t, nodes = model.get_model_stats()
print(f"ObjVal={ip:.4f}  Gap={gap:.2%}  Nodes={nodes}  Time={t:.1f}s")
```

---

## Parámetros Gurobi relevantes

| Parámetro | Valor | Motivo |
|---|---|---|
| `LazyConstraints` | `1` | Obligatorio para `cbLazy()` en el callback |
| `InfUnbdInfo` | `1` | Permite extraer rayos de Farkas cuando el subproblema es infactible |
| `DualReductions` | `0` | Evita que Gurobi reporte `INF_OR_UNBD` (status 4) en lugar de `INFEASIBLE` |
| `OutputFlag` | `0` | Suprime salida Gurobi cuando `verbose=False` |

---

## Alias de tipos

| Alias | Definición | Módulo |
|---|---|---|
| `Arc` | `tuple[int, int]` | `utils/geometry.py` |
| `PointLookup` | `dict[int, tuple[float,float]] \| NDArray[np.int64]` | `utils/geometry.py` |
| `ArcConstraintMap` | `dict[Arc, gp.Constr]` | `utils/constraints.py` |
| `NumericArray` | `NDArray[np.number]` | `models/typing_oap.py` |
| `IndexArray` | `NDArray[np.integer]` | `models/typing_oap.py` |
| `RayComponents` | `dict[str, dict[Arc, float]]` | `models/mixin/benders_farkas_mixin.py` |

---

## Referencias

- Hernández-Pérez, H., Riera-Ledesma, J., Rodríguez-Martín, I., & Salazar-González, J. J. *Optimal area polygonisation problems: Mixed integer linear programming models*. EJOR, 329(3), 767-777. DOI: 10.1016/j.ejor.2025.08.023
- Benders, J. F. (1962). *Partitioning procedures for solving mixed-variables programming problems*. Numerische Mathematik, 4(1), 238-252.
- Magnanti, T. L., & Wong, R. T. (1981). *Accelerating Benders decomposition*. Operations Research, 29(3), 464-484.
