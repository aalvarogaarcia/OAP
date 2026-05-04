# Plan: arreglar CGSP / MW + implementar DDMA (paper Hosseini-Turner 2025)

**Fecha:** 2026-05-04
**Origen:** auditoría del agente `mathematical-formalist` contra el paper *Deepest Cuts for Benders Decomposition* (Hosseini & Turner 2025), publicada en `.claude/context/reviews/2026-05-04-cgsp-paper-dissonance.md`.
**Diagnóstico empírico:** `outputs/reports/benchmark_general_20260504_173713.md` (Farkas ≈ MW, CGSP no converge o falla por `round("-")`).

---

## 0. Resumen ejecutivo

El benchmark general (5 métodos × 6 instancias n=20) muestra:

- **`mw_lp` y `mw_uniform` son numéricamente idénticos a `farkas`** — Magnanti-Wong nunca se aplica.
- **`cgsp_farkas` y `cgsp_pi` agotan los 600 s** o **se caen con `type str doesn't define __round__`** en 4/6 instancias.

El paper de referencia es **Hosseini & Turner (2025)** — el archivo `Deepest-BendersCuts.md` lo corresponde a *este* artículo (no García-Muñoz como sugería el docstring). El paper propone:
- una familia de **pseudonormas** (MISD, Relaxed-ℓ₁, MWP, CW, ℓ_p) que cambian *qué* corte deepest se elige;
- una **alternativa al LP de separación**: el algoritmo **DDMA** (Algorithm 3, §4.1, p. 2602), que reusa el subproblema clásico de Benders con un RHS perturbado y consigue ≈80% reducción de tiempo vs el LP normalizado.

**La implementación actual realiza solo la mitad del paper** (CGSP-LP con normalización ℓ₁ unitaria = MISD), y de manera incorrecta en varios puntos. La causa del bug `round("-")` es periférica (el script de benchmark llama `round` sobre el string sentinela `"-"` que devuelve `get_model_stats` cuando la relajación LP no converge); el síntoma de fondo es que **la relajación LP no converge en CGSP** porque cada llamada al callback reconstruye un modelo Gurobi desde cero.

Plan en tres fases: **(F1) parches mínimos para hacer correr el benchmark y desbloquear MW**, **(F2) corrección matemática del CGSP existente** y **(F3) implementación de DDMA como mixin nuevo**.

---

## 1. Bugs identificados — tabla maestra

| # | Severidad | Archivo:línea | Diagnóstico | Fase |
|---|-----------|---------------|-------------|------|
| B1 | BLOQUEANTE | `models/mixin/benders_mw_mixin.py:69,79` | `getattr(self, "_x", {})` → debería ser `"x"`. El core point siempre es `{}`, MW aborta con `no_core_point` y cae a Farkas legacy. | F1 |
| B2 | BLOQUEANTE | `experiments/benchmark_benders_general.py:208-211` | `round(lp, 4)` cuando `get_model_stats()` devuelve el sentinela `"-"`. Provoca el `TypeError` que vemos en el reporte. | F1 |
| B3 | DIAGNÓSTICO | `experiments/benchmark_benders_general.py:217-219` | El `except Exception` come el traceback. Imposible ver el origen real de fallos futuros. | F1 |
| B4 | CORRECCIÓN | `models/mixin/benders_cgsp_mixin.py:325-331` | El coeficiente del dual de `r3` cuenta `1` cuando un triángulo $t$ está en `adj(i,j) ∩ adj(k,s)`; lo correcto es `2` (cuenta ambas indicatrices). Cortes no válidos cuando ese caso aparece. | F2 |
| B5 | CORRECCIÓN | `models/mixin/benders_mw_mixin.py:212,257` | MW resuelve el LP secundario, **descarta `π★`** y vuelve a llamar a `get_cgsp_cut_yp`/`_y` que reconstruye y resuelve el CGSP primario. El corte que se inyecta no es el corte de Pareto de MW. | F2 |
| B6 | CORRECCIÓN | `models/mixin/benders_mw_mixin.py:188-191,236-239` | Cuando `sub_yp.Status == INFEASIBLE`, `q_opt = 0.0` y se añade la igualdad `π^T b(x̄) = 0`. Lo correcto (Papadakos / paper §3.3.3 eq. 29) es **no añadir la restricción de Pareto** en el caso infeasible. | F2 |
| B7 | RENDIMIENTO | `models/mixin/benders_cgsp_mixin.py:379,556` | Se reconstruye un `gp.Model` nuevo en cada callback. Con $\binom{n}{3}$ triángulos × $|A|$ arcos esto domina el tiempo. Es **la causa principal del 100×** observado. | F2 |
| B8 | CORRECCIÓN | `models/mixin/benders_cgsp_mixin.py:514,705` | La normalización `Σ(u+v)=1` con pesos unitarios = **MISD** (Fischetti et al. 2010), no la *Relaxed-ℓ₁* recomendada (§3.3.2, p. 2600). Pesos correctos: $w_i = \sum_j |B_{ij}|$. | F2 |
| B9 | COSMÉTICO | `models/mixin/benders_cgsp_mixin.py:790-848,935-990` | Las constantes de los términos x-dependientes (e.g. `1.0` en `pi_val * (1.0 - x[i,j])`) se acumulan dentro de `cut_expr` y dependen de que Gurobi las mueva a la RHS. Funciona hoy; frágil ante cualquier `LinExpr.copy()`. Refactor: mover constantes explícitamente a `cut_rhs`. | F2 |
| B10 | OK | `models/mixin/benders_cgsp_mixin.py:996-1004` | Rechazo de `pi0_net < 0`: matemáticamente correcto (cono de Hosseini-Turner pide $π_0 ≥ 0$). No es bug. | — |

---

## 2. Plan por fases

### Fase 1 — desbloqueo (≤1 día, sin cambiar matemáticas)

Objetivo: poder *ver* qué pasa y que el benchmark deje de mentir (MW ≠ Farkas).

| Tarea | Archivo | Cambio | Acceptance |
|-------|---------|--------|------------|
| F1.1 | `models/mixin/benders_mw_mixin.py:69,79` | Reemplazar `getattr(self, "_x", {})` → `getattr(self, "x", {})` | El core point uniforme tiene tantos elementos como `len(self.x)`; en el callback de un test rápido de `n=10`, `_core_point` es no vacío. |
| F1.2 | `experiments/benchmark_benders_general.py:208-211` | Helper `_safe_round(v, n) = round(v, n) if isinstance(v, (int,float)) else None` y aplicar a `root_lp`, `final_ip`, `gap_pct`, `time_s`. | El benchmark sobre `london-0000020 cgsp_farkas` ya no levanta `TypeError`; en el reporte aparece `—` en las celdas en lugar de `FAILED`. |
| F1.3 | `experiments/benchmark_benders_general.py:217-219` | Logging del traceback completo (`logger.error(..., traceback.format_exc())`) antes de que el `except` resuma el mensaje. | Al provocar un fallo manual (e.g. `raise RuntimeError("X")` en CGSP), el log muestra el stack frame real. |
| F1.4 | `experiments/benchmark_benders_general.py` (ejecución) | Re-correr el mismo conjunto de 6 instancias con F1.1-1.3 aplicados. | (a) MW produce números *distintos* a Farkas en al menos 1 instancia; (b) si CGSP sigue fallando, sabemos exactamente dónde. |

**Salida F1:** un reporte de benchmark donde MW tiene números propios, CGSP tiene un traceback útil, y los `FAILED` desaparecen del cuello de botella `round`.

---

### Fase 2 — correctitud matemática del CGSP existente

Objetivo: que `cgsp_farkas` produzca cortes correctos *y* converja antes de los 600 s en n=20.

| Tarea | Archivo | Cambio | Acceptance |
|-------|---------|--------|------------|
| F2.1 (B4) | `models/mixin/benders_cgsp_mixin.py:325-331` | Contar ambos hits: `coef = (1 if t in _adj(i,j) else 0) + (1 if t in _adj(k,s) else 0); if coef and v: expr -= coef * v`. | Test unit `test_cgsp_validity.py`: añadir un caso sintético con un triángulo `t ∈ adj(i,j) ∩ adj(k,s)` y verificar que el dual feasibility constraint tiene coeficiente 2 en ese `t`. |
| F2.2 (B5) | `models/mixin/benders_cgsp_mixin.py` y `benders_mw_mixin.py` | Refactor: extraer `_reconstruct_cut_from_pi(self, pi_vars, x_sol, which) -> (LinExpr, float, dict)` desde `get_cgsp_cut_y/yp` (líneas 759-848 / 935-990). En MW, sustituir las llamadas `self.get_cgsp_cut_yp(...)` por `self._reconstruct_cut_from_pi(pi_vars, x_sol, "yp")` después de resolver el LP secundario. | El corte inyectado por MW depende explícitamente del `π★` del LP secundario (verificable comparando contra el `π` que devolvería un CGSP plano para el mismo `x_sol`). |
| F2.3 (B6) | `models/mixin/benders_mw_mixin.py:188-191,236-239` | Cuando `sub_yp.Status == INFEASIBLE`: **no** añadir `mw.addConstr(x_bar_expr == q_opt, ...)`. Mantener solo la normalización L₁ y el objetivo en `x^0`. | Comparar nodos de B&B en una instancia donde Y' es infeasible al inicio: MW Papadakos debe producir cortes diferentes (más profundos) que la versión actual. |
| F2.4 (B7) | `models/mixin/benders_cgsp_mixin.py` | Cachear el modelo CGSP en `self._cgsp_yp_model`, `self._cgsp_y_model` (con sus `pi_vars`). Solo actualizar coeficientes del objetivo (los únicos que dependen de `x_bar`). Construir-una-vez en `build()`. | Tiempo medio por callback CGSP ≤ 5× el de Farkas (vs ≥100× actual) en `uniform-0000020-1`. |
| F2.5 (B8) | `models/mixin/benders_cgsp_mixin.py` | Calcular pesos por defecto como column-sums: $w_{α,(i,j)}=1$, $w_{β,(i,j)}=2$, $w_{γ,(i,j)}=1$, $w_{δ,(i,j)}=1$, $w_{r3,(i,j,k,s)}=2$, $w_g=w_{r1}=w_{r2}=0$. Exponer flag `cgsp_norm: Literal["misd","relaxed_l1"]` en `build()` (default `relaxed_l1`). | A iguales condiciones, `cgsp_farkas` con `relaxed_l1` produce gap raíz ≤ el gap raíz con `misd` en una mayoría de las instancias del benchmark. |
| F2.6 (B9) | `models/mixin/benders_cgsp_mixin.py` (cosmético) | En la reconstrucción del corte, descomponer cada término `pi * (const + lin(x))` en `cut_expr += pi * lin(x)` y `cut_rhs += -pi * const`, eliminando la dependencia de Gurobi para mover constantes. | Test: emitir el corte como string y verificar que `cut_expr` es puramente lineal en x sin constante; `cut_rhs` recoge todas las constantes. |

**Salida F2:** `cgsp_farkas` y `cgsp_pi` corren en tiempos comparables a `farkas` (factor ≤5), MW produce cortes Pareto-óptimos verdaderos, todos los tests siguen pasando.

---

### Fase 3 — implementación de DDMA (la "otra implementación del paper")

DDMA = *Distance-based Depth-Maximisation Algorithm*, paper §4.1, Algoritmo 3 (p. 2602). Es **la otra contribución algorítmica del paper** distinta del CGSP-LP, y es la que el paper recomienda en §5.2 (≈80% reducción de tiempo de generación de cortes).

#### 3.1 Idea (en notación OAP)

Dado:
- Subproblema primal $Y$ (o $Y'$) con datos $A, B, b$ (ya construidos en `sub_y` / `sub_yp`).
- Pseudonorma lineal $h(π) = w^T π$ con vector de pesos $w \ge 0$ (Relaxed-ℓ₁).
- Solución actual del maestro $\bar x$.

Iteración:

```
z ← 0
loop:
    # Step 1 — generación de certificado clásica con RHS perturbado
    actualizar RHS de sub_y(p):  b - B x_bar - z * w
    sub_y(p).optimize()
    if INFEASIBLE:  π ← FarkasDual    (Farkas mode)
    else:           π ← .Pi            (Pi mode)

    # Step 2 — maximización de profundidad
    ẑ = (π^T (b - B x_bar)) / (w^T π)
    if ẑ - z < ε:  break
    z ← ẑ

return cut π^T B x ≤ π^T b
```

#### 3.2 Por qué es la pieza que falta

- No requiere LP auxiliar — reusa `sub_y` / `sub_yp` que ya están construidos en `BendersFarkasMixin` / `BendersPiMixin` con bases LP cacheadas (warm start gratis).
- **Termination temprana sigue produciendo un corte válido** (paper Remark 6, p. 2602). Esto es robusto a límites de tiempo.
- En el paper, $\ell_1$-DDMA resolvió **todas las 169 instancias** vs $\ell_2$/$\ell_\infty$ que dejaron 2 sin resolver (§5.2 p. 2605).

#### 3.3 Tareas

| Tarea | Archivo | Cambio | Acceptance |
|-------|---------|--------|------------|
| F3.1 | `models/mixin/benders_ddma_mixin.py` (nuevo) | Crear `BendersDDMAMixin` con métodos `_compute_relaxed_l1_weights_y/yp`, `_perturb_subproblem_rhs(z)` y `get_ddma_cut_y/yp(x_sol, eta_sol, max_iter=20, eps=1e-6)`. | Test unit: en un caso pequeño donde el corte óptimo se conoce analíticamente, DDMA converge en ≤5 iteraciones a la profundidad esperada con tolerancia 1e-6. |
| F3.2 | `models/OAPBendersModel.py` | Insertar `BendersDDMAMixin` en el MRO, justo antes de `BendersCGSPMixin`. Añadir flag `use_ddma: bool = False` en `build()`. | Construcción del modelo no rompe ningún test existente (default off). |
| F3.3 | `models/mixin/benders_optimize_mixin.py` | Nuevo branch en `_benders_callback`: si `use_ddma` y hay violación, llamar a `get_ddma_cut_y/yp`. Mismo patrón en `solve_lp_relaxation`. | Compatibilidad: `use_deepest_cuts`, `use_magnanti_wong`, `use_ddma` mutuamente excluyentes; el constructor lo valida. |
| F3.4 | `experiments/benchmark_benders_general.py` | Añadir métodos `ddma_farkas` y `ddma_pi` al `METHOD_CONFIG`. | El benchmark ejecuta los 7 métodos y produce el reporte. |
| F3.5 | `test/test_ddma.py` (nuevo) | Tests de validez (corte separa `x_bar`), de profundidad (no-dominado por el corte de Farkas plano), de robustez (early termination produce corte válido). | Todos los tests verdes en `pytest test/test_ddma.py`. |
| F3.6 | Re-ejecución del benchmark | Correr el campaña general comparando: `farkas`, `cgsp_farkas` (post-F2), `mw_lp` (post-F2), `ddma_farkas`. | El reporte muestra: (a) DDMA termina antes que CGSP-LP; (b) gap raíz de DDMA ≤ gap raíz de Farkas; (c) DDMA resuelve más instancias dentro del budget. |

#### 3.4 Diferencia con CGSP

| Dimensión | CGSP-LP (actual) | DDMA (a implementar) |
|-----------|------------------|----------------------|
| Estructura | Resuelve un LP de separación que selecciona $π$ y profundidad simultáneamente | Iterativo: cada paso resuelve el subproblema de Benders clásico |
| Modelo Gurobi | Construye y resuelve un nuevo `gp.Model` por callback (sin warm start) | Reusa `sub_y` / `sub_yp` con base LP caliente |
| Costo por corte | `~ |constraints|×|vars|` LP simplex | `≤ max_iter` Pi/Farkas solves (típicamente 3-5) |
| Terminación temprana | Corte = `None` (no se inyecta) | Corte válido pero menos profundo (graceful) |
| Resultados del paper | $\ell_1$-CGSP-LP resuelve todas las 169, pero el LP es el cuello de botella | $\ell_1$-DDMA resuelve todas las 169 con ≈80% menos tiempo |

---

## 3. Orden de ejecución recomendado

```
F1 (1 día)
  F1.1  fix MW _x → x
  F1.2  fix benchmark round("-")
  F1.3  benchmark traceback logging
  F1.4  re-run benchmark → snapshot baseline real

F2 (3-4 días)
  F2.1  fix r3 dual feasibility coefficient
  F2.2  refactor _reconstruct_cut_from_pi + fix MW cut source
  F2.3  fix MW Papadakos infeasible case
  F2.4  cache CGSP model across callbacks
  F2.5  switch normalisation weights to Relaxed-ℓ₁
  F2.6  cosmetic: explicit constants in cut_rhs
  ----  benchmark intermedio (CGSP debería ahora competir con Farkas)

F3 (4-6 días)
  F3.1  BendersDDMAMixin
  F3.2  hook into OAPBendersModel
  F3.3  callback dispatch
  F3.4  benchmark METHOD_CONFIG
  F3.5  test/test_ddma.py
  F3.6  re-run benchmark final
```

**Hito final**: un reporte general 7-method (`farkas`, `cgsp_farkas` arreglado, `cgsp_pi` arreglado, `mw_lp` arreglado, `mw_uniform` arreglado, `ddma_farkas`, `ddma_pi`) sobre las 6 instancias del último run + 6 instancias adicionales (n=30) que demuestre:

1. **Validez**: todos los métodos llegan al mismo óptimo IP cuando convergen.
2. **Velocidad**: `cgsp_*` ≤ 2× `farkas` (vs 100× actual); `ddma_*` ≤ `farkas`.
3. **Calidad**: `ddma_farkas` y `cgsp_farkas` tienen gap raíz ≤ `farkas` en ≥75% de las instancias.

---

## 4. Riesgos y notas

- **CGSP `cgsp_farkas` y `cgsp_pi` produjeron números idénticos** en el reporte (mismo Final IP 39171230, mismas 179 K nodos en `euro-night-0000020`). Esto sugiere que en `use_deepest_cuts=True` la elección Farkas vs Pi del subproblema base es irrelevante (ambos generan el mismo CGSP). Es esperable — el CGSP es independiente del modo. Después de F2.4, podríamos colapsar `cgsp_pi` y `cgsp_farkas` en un único método `cgsp` (no es necesario para la corrección).
- **`mw_lp` y `mw_uniform` también producirán los mismos números entre sí hasta F1.1**. Después de F2.2 deberían divergir según el `core_point_strategy`.
- **Convergencia de DDMA**: el paper garantiza terminación finita (Teorema 3) para *cualquier* pseudonorma lineal con $w \ge 0$. El test crítico es F3.1 — si DDMA no converge en ≤20 iteraciones en n=20, hay un bug en el cómputo de `ẑ` (división por `w^T π` cercano a 0).
- **El docstring de `benders_cgsp_mixin.py` apunta a "Garcia-Munoz (2026)"** (líneas 10-11). Después de la corrección debería referenciar Hosseini & Turner (2025) — este es el paper real que estamos siguiendo.

---

## 5. Entregables

- Este documento de plan (`.claude/context/plans/2026-05-04-cgsp-mw-fix-and-ddma-plan.md`)
- El reporte de auditoría (`.claude/context/reviews/2026-05-04-cgsp-paper-dissonance.md`) con todas las referencias línea-a-línea.
- Tras F1: nuevo benchmark report con números reales de MW.
- Tras F2: tests `test/test_cgsp_validity.py` (existe) ampliados + benchmark intermedio.
- Tras F3: nuevo mixin DDMA + tests + benchmark final 7-method.
