# Diagnóstico técnico y deuda de rendimiento

> Última actualización: mayo 2026  
> Estado del código auditado: `main` branch, v0.2.0

---

## Resumen ejecutivo

La arquitectura es correcta y sigue el esquema de Hernández-Pérez et al. (2025). El solver implementa un branch-and-cut auténtico con callback de Gurobi. El problema **no está en la matemática**, sino en cinco fugas de rendimiento concretas que explican por qué el solver no escala más allá de ~30 puntos.

---

## Fuga #1 — Maestro usa SCF en lugar de DFJ

**Severidad: Crítica**  
**Fichero**: `models/mixin/benders_master_mixin.py`, método `_add_subtour_constraints_master()`

### Problema

El maestro usa **Single Commodity Flow (SCF)** para eliminación de subtours. SCF es la formulación TSP de relajación LP más débil disponible:

- Introduce `O(n²)` variables de flujo continuas `f[i,j]`.
- La relajación LP de SCF es extremadamente laxa.
- El LP-gap alto no viene de los cortes Benders — viene del maestro.

### Solución

Migrar a **DFJ (Dantzig-Fleischer-Johnson)** con SECs lazy via callback:

```python
# En _benders_callback, ANTES de comprobar factibilidad de los subproblemas:
if where == GRB.Callback.MIPSOL:
    x_sol = model.cbGetSolution(self.x)
    
    # 1. Comprobar subtours en la solución entera
    components = connected_components(x_sol)  # usando networkx
    if len(components) > 1:
        # Inyectar SEC lazy para cada componente no trivial
        for S in components:
            model.cbLazy(
                gp.quicksum(self.x[i,j] for i in S for j in S if i!=j and (i,j) in self.x) 
                <= len(S) - 1
            )
        return  # No comprobar subproblemas hasta que no haya subtours
    
    # 2. Solo si no hay subtours: resolver subproblemas Benders
    ...
```

La clase `OAPCompactModel` ya soporta DFJ (`subtour="DFJ"`). Reutilizar esa lógica.

### Impacto esperado

Mejora de 1–2 órdenes de magnitud en número de nodos del árbol B&B. Es el cambio de mayor retorno por línea de código después de Magnanti–Wong.

---

## Fuga #2 — Cortes Farkas no son Pareto-óptimos

**Severidad: Alta**  
**Fichero**: `models/mixin/benders_farkas_mixin.py`, métodos `get_farkas_cut_y` y `get_farkas_cut_yp`

### Problema

Se toma el **primer rayo extremo** que devuelve Gurobi. La degeneración dual en problemas con estructura geométrica es masiva: en la práctica se añaden cortes débiles que apenas acortan la brecha LP.

### Solución: Magnanti–Wong

Tras detectar infactibilidad, resolver un **segundo LP auxiliar** con función objetivo que maximice la profundidad del corte sobre un *core point* (solución promedio de las soluciones MIPSOL aceptadas hasta ahora):

```python
def get_farkas_cut_y_MW(self, x_sol: dict, TOL: float = 1e-6):
    """Corte de Farkas Pareto-óptimo via Magnanti–Wong."""
    # 1. Obtener corte base (rayo Farkas estándar)
    cut_base, _ = self.get_farkas_cut_y(x_sol, TOL)
    
    # 2. Core point: media de soluciones maestro aceptadas
    x_core = self._get_core_point()  # dict[Arc, float]
    
    # 3. LP auxiliar: max profundidad del corte sobre x_core
    # sujeto a: el corte sea válido (pertenezca al cono de rayos)
    aux = gp.Model("MW_aux")
    aux.Params.OutputFlag = 0
    # ... construir LP auxiliar con los multiplicadores como variables ...
    aux.optimize()
    
    # 4. Devolver el corte profundo
    return self._extract_cut_from_aux(aux, x_sol)

def _get_core_point(self) -> dict:
    """Media de las soluciones MIPSOL aceptadas (sin cortes infactibles)."""
    if not self._accepted_solutions:
        return {k: 0.5 for k in self.x}
    return {k: sum(s[k] for s in self._accepted_solutions) / len(self._accepted_solutions)
            for k in self.x}
```

### Impacto esperado

Los cortes Pareto-óptimos reducen significativamente el número de iteraciones Benders. Es el cambio de mayor retorno matemático.

---

## Fuga #3 — Sin cortes en MIPNODE

**Severidad: Alta**  
**Fichero**: `models/mixin/benders_optimize_mixin.py`, método `_benders_callback()`

### Problema

El callback solo corta en `MIPSOL` (soluciones enteras). Todo el árbol de búsqueda fraccional queda sin explotar. La rama `ruflo_dev` apunta a "deepest cuts" (Hosseini–Turner) — esa es la dirección correcta.

### Solución

Añadir un bloque `MIPNODE` con `cbCut()`:

```python
def _benders_callback(self, model, where):
    if where == GRB.Callback.MIPSOL:
        # ... código actual ...
        pass
    
    elif where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
            return
        
        x_sol = model.cbGetNodeRel(self.x)  # solución fraccional del nodo
        
        # Actualizar RHS y resolver subproblemas con x fraccional
        self._update_subproblem_rhs(x_sol)
        self.sub_y.optimize()
        self.sub_yp.optimize()
        
        TOL = 1e-6
        
        if self.sub_y.Status == GRB.INFEASIBLE:
            cut_expr, cut_val = self.get_farkas_cut_y(x_sol, TOL)
            if abs(cut_val) > TOL:
                model.cbCut(cut_expr <= 0 if cut_val > 0 else cut_expr >= 0)
        
        if self.sub_yp.Status == GRB.INFEASIBLE:
            cut_expr, cut_val = self.get_farkas_cut_yp(x_sol, TOL)
            if abs(cut_val) > TOL:
                model.cbCut(cut_expr <= 0 if cut_val > 0 else cut_expr >= 0)
```

**Importante**: `cbCut()` requiere que el corte sea válido para cualquier solución entera, no solo para la actual. Verificar que los cortes Farkas satisfacen esta condición antes de activar MIPNODE.

### Impacto esperado

Reducción del árbol B&B al podar nodos antes de llegar a soluciones enteras. Especialmente útil en instancias grandes (>50 puntos).

---

## Fuga #4 — Logging masivo dentro del callback

**Severidad: Media**  
**Ficheros**: `models/mixin/benders_farkas_mixin.py` (`_log_and_print_farkas`), `models/mixin/benders_pi_mixin.py` (`_log_and_print_pi`)

### Problema

`json.dump`, formateo de strings y escritura a disco en **cada llamada a MIPSOL**. En producción con instancias de 100 puntos esto genera cientos de operaciones I/O dentro del callback, destruyendo el rendimiento.

### Solución

```python
def _log_and_print_farkas(self, v_components, cut_val, sub_name, TOL, x_sol, cut_expr, sense=None):
    # GUARDA: solo loggear si save_cuts=True (ya implementado)
    if not getattr(self, 'save_cuts', False):
        return
    
    # MEJORA: buffer en memoria en lugar de disco en cada iteración
    if not hasattr(self, '_cut_log_buffer'):
        self._cut_log_buffer = []
    
    self._cut_log_buffer.append({
        "iteration": self.iteration,
        "sub": sub_name,
        "val": cut_val,
        "components": {k: dict(v) for k, v in v_components.items() if v},
    })
    
    # Volcar a disco solo cada N iteraciones o al finalizar
    if len(self._cut_log_buffer) >= 100:
        self._flush_cut_log()

def _flush_cut_log(self):
    """Vuelca el buffer de cortes a disco."""
    if not hasattr(self, '_cut_log_buffer') or not self._cut_log_buffer:
        return
    with open(self.log_path, 'a') as f:
        for entry in self._cut_log_buffer:
            f.write(json.dumps(entry) + '\n')
    self._cut_log_buffer.clear()
```

Llamar a `_flush_cut_log()` al final de `solve()` para asegurar que no se pierden cortes.

---

## Fuga #5 — Bug `UnboundLocalError: sense` en método PI

**Severidad: Bloqueante para benchmarks PI**  
**Fichero**: `models/mixin/benders_pi_mixin.py`, función `get_pi_cut_y`

### Problema

En versiones anteriores, la variable `sense` podía no estar inicializada si `cut_y_val == 0.0` exactamente, causando un `UnboundLocalError` que tiraba el callback en la primera iteración.

### Estado actual

El fix (`else: sense = "=="`) está implementado en las líneas 233-238 de `benders_pi_mixin.py`. El error en `bench_output.txt` corresponde a un commit anterior.

### Verificación

```bash
git log --oneline models/mixin/benders_pi_mixin.py
# Buscar el commit que añade "else: sense = '=='"

# Rerun del benchmark con método PI:
.venv/bin/python main.py instance/little-instances "*.instance" \
    --time-limit 60 --obj 0 --model benders
```

Si el benchmark PI sigue fallando con `UnboundLocalError`, significa que el fix **no está en `main`** y hay que cherry-pick desde la rama donde se corrigió.

---

## Orden de implementación recomendado

| Semana | Tarea | Impacto |
|---|---|---|
| 19 | Fix logging (flag `save_cuts` → buffer) | Producción usable |
| 19 | Verificar bug `sense` y rerun benchmark PI | Datos válidos |
| 19–20 | Migrar maestro SCF → DFJ + SECs lazy | Crítico |
| 20 | Implementar Magnanti–Wong | Alto |
| 21 | Cortes en MIPNODE (`cbCut`) | Alto |
| 22 | Pipeline HPC reproducible | Hito-puerta SEIO |

---

## Lo que NO tocar

- **Cortes R1–R3** de strengthening: están bien planteados.
- **Separación Y/Y'**: es la descomposición natural del problema (triángulos interiores vs exteriores a la CH).
- **`OAPBaseModel`** y la jerarquía de mixins base.
- **`utils/geometry.py`**: geometría pura, sin deuda.

---

## Referencias

- Magnanti, T. L., & Wong, R. T. (1981). *Accelerating Benders decomposition: Algorithmic enhancement and model selection criteria*. Operations Research, 29(3), 464-484.
- Hosseini, M., & Turner, J. *Deepest Cuts for Benders Decomposition*.
- Dantzig, G., Fulkerson, R., & Johnson, S. (1954). *Solution of a large-scale traveling-salesman problem*. Operations Research, 2(4), 393-410.
