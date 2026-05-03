# Plan de investigación OAP — Mayo a Septiembre 2026

> Objetivo: demostrar que Benders + cortes propios resuelve instancias mayores que la formulación compacta.  
> **Hito-puerta**: cuadro comparativo en HPC, finales de julio 2026, listo para SEIO septiembre 2026.

---

## Contexto estratégico

El cruce en la curva tiempo-vs-tamaño se espera a partir de ~50 puntos. Para conseguirlo es necesario resolver los cinco problemas de rendimiento descritos en `docs/technical_debt.md` en el orden correcto. El plan siguiente respeta ese orden y encaja con la distribución de tiempo disponible (24 h/semana de tesis, lunes-miércoles).

---

## Horizonte activo: semanas 19–22 (mayo 2026)

### Semana 19 — 5 al 11 de mayo: Consolidar maestro y arreglar fugas básicas

**Lunes–Martes (tesis)**
- Migrar el maestro de SCF a DFJ con SECs perezosas.
  - Reutilizar la lógica de `OAPCompactModel` con `subtour="DFJ"`.
  - Implementar detección de componentes conexas en `MIPSOL` antes del chequeo de subproblemas.
  - Eliminar las variables `f[i,j]` de flujo SCF del maestro Benders.
- Fix logging: añadir buffer en memoria, activar solo con `save_cuts=True`.

**Miércoles (tesis)**
- Verificar bug `sense` en `benders_pi_mixin.py`:
  ```bash
  git log --oneline models/mixin/benders_pi_mixin.py
  ```
- Unificar en una sola rama `develop` limpia: descartar commits de exploración que no aporten.
- Rerun benchmark con instancias de 10–20 puntos para establecer baseline post-fix.

**Jueves (XGBoost/nutrición)**
- Definir variables de la escala pronóstica en nutrición.
- Esquema de validación: k-fold estratificado, métricas AUC, calibración, Brier score.

**Viernes (lectura/planificación)**
- Leer Magnanti & Wong (1981) completo.
- Leer una revisión reciente sobre Benders acceleration en problemas TSP-like.

**Entregable**: rama `develop` con maestro DFJ + benchmark baseline documentado.

---

### Semana 20 — 12 al 18 de mayo: Cortes Pareto-óptimos (Magnanti–Wong)

**Lunes–Martes (tesis)**
- Derivar matemáticamente los cortes Magnanti–Wong para `sub_y` y `sub_yp`.
- Implementar `get_farkas_cut_y_MW()` y `get_farkas_cut_yp_MW()`.
  - Core point inicial: media de soluciones MIPSOL aceptadas (`self._accepted_solutions`).
  - LP auxiliar: maximizar `cut_expr.getValue(x_core)` sujeto a multiplicadores válidos.
- Integrar en `_benders_callback()` con flag `use_magnanti_wong=True`.

**Miércoles (tesis)**
- Primer benchmark interno comparativo:
  - `compact-DFJ` vs `benders-vanilla` vs `benders-MW`
  - Instancias: 20, 25, 30 puntos, 5 réplicas cada una, time-limit 5 min.
  - Métricas: nodos, tiempo, gap final, número de cortes añadidos.
- Generar tabla comparativa y gráfico tiempo-vs-tamaño.

**Jueves (XGBoost)**
- Pipeline mínimo en sklearn/xgboost con datos sintéticos del esquema clínico.

**Viernes (revisión)**
- Revisión con Salazar-González de resultados de semanas 19–20.
- Ajuste de plan según resultados.

**Entregable**: gráfico con primera curva de cruce (esperado a 30 puntos con MW activo).

---

### Semana 21 — 19 al 25 de mayo: Cortes en MIPNODE

**Lunes–Martes (tesis)**
- Implementar bloque `MIPNODE` en `_benders_callback()` con `cbCut()`.
  - Usar `cbGetNodeRel(self.x)` para extraer solución fraccional.
  - Verificar que los cortes Farkas son válidos para cualquier solución entera.
- Estrategia "deepest cuts" (Hosseini–Turner): maximizar violación del corte en el nodo LP.

**Miércoles (tesis)**
- Benchmark intermedio: añadir variante `benders-MW-deepest` a la comparación.
- Instancias hasta 50 puntos.

**Jueves (XGBoost)**
- Feature engineering específico de nutrición: composición corporal, bioquímica, escalas previas.

**Viernes (redacción)**
- Primer borrador de la sección de Métodos del paper SEIO.
  - Subsección: formulación del maestro (DFJ).
  - Subsección: descomposición Y/Y'.
  - Subsección: cortes Farkas y Magnanti–Wong.

**Entregable**: variante MW+Deepest funcionando; borrador de sección Métodos.

---

### Semana 22 — 26 de mayo al 1 de junio: Primer benchmark serio en HPC

**Lunes–Martes (tesis)**
- Preparar pipeline reproducible para el cluster HPC (ULL):
  - Script `analysis/run_batch.py` actualizado con las tres variantes.
  - Job script SLURM con parámetros: 5 réplicas/instancia, time-limit 1 h, 1 core/job.
  - Familias de instancias: 30, 50, 70 puntos (distribuciones: uniforme, clustered, convex).

**Miércoles (tesis)**
- Lanzar jobs en HPC.
- Análisis de resultados: tablas con media ± desv. std. por variante y tamaño.
- Gráficos: tiempo vs tamaño, nodos vs tamaño, gap vs iteraciones Benders.

**Jueves (XGBoost)**
- Primera iteración del modelo con datos reales (si están curados).

**Viernes (redacción + decisión-puerta)**
- Actualizar manuscrito con resultados preliminares.
- **Decisión-puerta**: si a 70 puntos Benders-MW-Deepest gana al compact en ≥3/5 réplicas → Plan A (continuar en esta dirección). Si no → reunión con Salazar para replantear.

**Entregable**: tabla HPC semana 22 + decisión-puerta documentada.

---

## Horizonte SEIO: junio–septiembre 2026

### Junio: Cierre del algoritmo

- Cierre de todas las mejoras algorítmicas (sin nuevas features a partir del 15/06).
- Benchmark exhaustivo: instancias 30, 50, 70, 100 puntos, 5 réplicas, tres familias.
- Análisis de sensibilidad: impacto individual de cada mejora (ablation study).

### Julio: HPC masivo + redacción

- Experimentos finales en HPC con configuración definitiva.
- Redacción completa del paper: Introducción, Problema, Métodos, Resultados, Conclusiones.
- **Hito-puerta SEIO (finales de julio)**: cuadro comparativo con tres familias (`compact-DFJ` / `Benders-Farkas-MW` / `Benders-Farkas-MW+Deepest`) sobre instancias de 30, 50, 70 y 100 puntos, mínimo 5 réplicas por tamaño.

### Agosto: Presentación SEIO + draft

- Preparación de diapositivas para SEIO.
- Cierre del draft submitted.
- Revisión interna con Salazar-González.

### Septiembre: SEIO 2026

- Presentación en SEIO.
- Subida del preprint.

---

## Distribución semanal del tiempo

| Día | Frente | Horas/día | Foco |
|---|---|---|---|
| Lunes | Tesis (OAP/Benders) | 8–9 h | Matemática y código profundo |
| Martes | Tesis (OAP/Benders) | 8–9 h | Código + debugging |
| Miércoles | Tesis (OAP/Benders) | 8–9 h | Experimentos + análisis |
| Jueves | XGBoost / nutrición | ~8 h | Pipeline, feature engineering, validación |
| Viernes | FUNCIS + redacción | ~6 h | Coautorías, borrador tesis, lectura |
| Sábado | Slack / descanso | — | Sin trabajo planificado |
| Domingo | Descanso | — | Off completo |

**Total estimado**: ~24 h/semana tesis OAP, ~8 h/semana XGBoost, ~6 h/semana FUNCIS.

---

## Criterios de éxito para SEIO

1. **Benchmark**: Benders-MW-Deepest supera a compact-DFJ en tiempo para instancias ≥50 puntos en ≥3/5 réplicas.
2. **Paper**: sección de Métodos completa con demostración de cortes Pareto-óptimos.
3. **Reproducibilidad**: pipeline HPC reproducible con semilla fija, scripts públicos.
4. **Presentación**: diapositivas con la curva tiempo-vs-tamaño mostrando el cruce.

---

## Gestión de riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| MW no mejora suficientemente el gap | Media | Alto | Pasar a variante con MIPNODE antes; revisar formulación del core point |
| Bug `sense` no está fixeado en `main` | Baja | Medio | Verificar en semana 19 antes de cualquier benchmark PI |
| HPC ULL no disponible en julio | Baja | Alto | Correr subset de benchmarks en local con time-limit reducido |
| Retraso en datos XGBoost | Media | Bajo (para OAP) | XGBoost es independiente; retrasar sin impacto en hito SEIO |

---

## Decisiones de diseño cerradas

Las siguientes decisiones están tomadas y **no deben reabrirse** sin datos que lo justifiquen:

- **Descomposición Y/Y'**: es la descomposición natural del problema OAP (triángulos interiores vs exteriores a la CH). No cambiar.
- **Branch-and-cut con callback**: vs Benders iterativo manual. El callback es más eficiente y permite cortes en MIPNODE.
- **Parámetros Gurobi**: `InfUnbdInfo=1`, `DualReductions=0` son obligatorios para la extracción correcta de rayos de Farkas.
- **Cortes R1–R3**: están bien planteados; no tocar hasta tener datos que muestren que son el cuello de botella.
