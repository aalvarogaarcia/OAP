# Optimal Area Polygonization (OAP): A MILP and Benders-Decomposition Framework

## Abstract

This repository provides a research implementation for the **Optimal Area Polygonization (OAP)** problem, where a simple polygon must be constructed over a fixed planar point set while optimizing enclosed area.  
Two variants are addressed: **MIN-OAP** (minimum area) and **MAX-OAP** (maximum area).  
The codebase implements exact optimization methods based on **Mixed-Integer Linear Programming (MILP)**, including both compact formulations and a decomposed strategy via **Benders decomposition**.

## 1. Problem Statement

Let \(S=\{p_1,\dots,p_n\}\subset\mathbb{R}^2\) be a set of points.  
The objective is to find a **simple Hamiltonian polygon** whose vertex set is exactly \(S\), optimizing:

- **MIN-OAP**: \(\min \text{Area}(P)\)
- **MAX-OAP**: \(\max \text{Area}(P)\)

The optimization is constrained by polygon simplicity (non-self-intersection) and full vertex inclusion.  
Both variants are NP-hard and require exact combinatorial optimization techniques for reliable optimality guarantees.

## 2. Methodological Framework

### 2.1 Compact MILP Formulation (`models/OAPCompactModel.py`)

The compact approach models polygon construction directly in a monolithic MIP via `OAPCompactModel`, including:

- edge-selection decision variables over the complete directed graph,
- configurable subtour-elimination mechanisms (`DFJ`, `MTZ`, `GCS`),
- geometric consistency constraints to enforce polygon feasibility and simplicity.

### 2.2 Benders Decomposition (`models/OAPBendersModel.py`)

To improve scalability and modularity, the formulation is decomposed via `OAPBendersModel` into:

- **Master problem** over global combinatorial edge-selection decisions (`mixin/benders_master_mixin.py`),
- **Feasibility subproblem** generating Farkas-ray cuts (`mixin/benders_farkas_mixin.py`),
- **Optimality subproblem** generating π-based cuts (`mixin/benders_pi_mixin.py`),
- **Solve-loop orchestration** and LP-relaxation warmstart (`mixin/benders_optimize_mixin.py`),
- **Post-solve diagnostics** (`mixin/benders_analysis_mixin.py`).

Both model classes inherit from `OAPBaseModel`, which provides convex-hull computation, triangulation, and result-export logic via `OAPStatsMixin`.

## 3. Repository Organization

```text
OAP_NextGen/
├── main.py                    – batch solver entry point
├── run.py                     – single-instance driver (interactive)
├── run_single_instance.py     – headless single-instance runner
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
├── README.md
├── models/
│   ├── __init__.py
│   ├── OAPBaseModel.py        – abstract base (stats, hull, triangulation)
│   ├── OAPCompactModel.py     – compact MILP formulation
│   ├── OAPBendersModel.py     – Benders decomposition entry point
│   ├── typing_oap.py          – shared type aliases
│   └── mixin/
│       ├── oap_builder_mixin.py       – variable/constraint construction
│       ├── oap_stats_mixin.py         – result statistics & export
│       ├── benders_master_mixin.py    – master problem setup
│       ├── benders_farkas_mixin.py    – Farkas feasibility cuts
│       ├── benders_pi_mixin.py        – π-based optimality cuts
│       ├── benders_optimize_mixin.py  – solve-loop orchestration
│       └── benders_analysis_mixin.py  – post-solve diagnostics
├── utils/
│   ├── __init__.py
│   ├── utils.py               – backward-compat re-export shim
│   ├── geometry.py            – pure geometry, I/O helpers, type aliases
│   ├── constraints.py         – Gurobi constraint injection helpers
│   ├── benders_log.py         – cut logging & JSONL serialisation
│   ├── visualization.py       – plotting & diagnostic visualisations
│   ├── geometry_classifier.py – convex-layer onion peeling
│   ├── model_stats.py         – tabular result aggregation
│   └── analyze_benders.py     – Benders cut analysis CLI
├── analysis/
│   ├── __init__.py
│   ├── benders_analysis.py    – cut-pattern analysis
│   ├── run_batch_analysis.py  – batch experiment runner
│   └── umap_benders_analysis.py – UMAP dimensionality reduction
├── test/
│   ├── __init__.py
│   ├── test_compact_minimize.py
│   ├── test_compact_maximize.py
│   ├── test_benders.py
│   ├── test_bendersv2.py
│   └── TablaResultadosA4.tsv  – reference results for parametric tests
├── instance/                  – problem instances (git-ignored)
└── outputs/                   – solver output files (git-ignored)
```

## 4. Reproducibility and Execution

### Environment setup

```bash
pip install -r requirements.txt
```

or (if using `uv`):

```bash
uv sync
```

> **Solver requirement:** A valid **Gurobi** installation and license are required.

### Batch run

```bash
python main.py instance/little-instances "*.instance" --time-limit 60 --obj 0
```

CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `instance_dir` | — | Directory containing `.instance` files |
| `pattern` | — | Glob pattern to filter instance files |
| `--time-limit` | `600` | Gurobi time limit in seconds |
| `--obj` | `0` | Objective: `0` = minimize area, `1` = maximize area |
| `--model` | `compact` | Model type: `compact` or `benders` |
| `--subtour` | `DFJ` | Subtour elimination: `DFJ`, `MTZ`, or `GCS` |

### Single instance

```bash
python run_single_instance.py instance/little-instances/uniform-0000010-1-HIPOLITO.instance
```

### Tests

```bash
pytest -q
```

## 5. Current Research Status

This repository is under active development. Recent work (v0.2.0) includes:

- full migration from the legacy monolithic `models/gurobi.py` to the mixin-based `OAPCompactModel` / `OAPBendersModel` architecture,
- split of the monolithic `utils/utils.py` into focused submodules (`geometry`, `constraints`, `benders_log`, `visualization`),
- Farkas-ray and π-cut logging infrastructure with JSONL serialisation,
- geometry-aware constraint helpers (half-plane, knapsack, clique-of-crossing-edges),
- improvements in typing, PEP 8 compliance, and parametric test coverage.

The project is intended as a research codebase; interfaces and experimentation workflows may evolve.

## 6. Citation

If this repository contributes to your research, please cite via `CITATION.cff` (or GitHub "Cite this repository").

```text
García, Á. Optimal Area Polygonization (OAP) NextGen: MILP Framework. v0.2.0
Repository: https://github.com/aalvarogaarcia/OAP
```

## 7. References

- Hernández-Pérez, H., Riera-Ledesma, J., Rodríguez-Martín, I., & Salazar-González, J. J.  
  *Optimal area polygonisation problems: Mixed integer linear programming models*.  
  European Journal of Operational Research, 329(3), 767–777. DOI: 10.1016/j.ejor.2025.08.023
- Hosseini, M., & Turner, J.  
  *Deepest Cuts for Benders Decomposition*.
