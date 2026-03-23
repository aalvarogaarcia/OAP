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

### 2.1 Compact MILP Formulation (`models/gurobi.py`)

The compact approach models polygon construction directly in a monolithic MIP, including:

- edge-selection decision variables over graph structures,
- subtour-elimination mechanisms (configurable variants),
- geometric consistency constraints to enforce polygon feasibility and simplicity.

### 2.2 Benders Decomposition (`models/benders/`)

To improve scalability and modularity, the formulation is decomposed into:

- **Master problem** over global combinatorial decisions,
- **Subproblem checks** for geometric/structural feasibility,
- **Feasibility/optimality cuts** generated iteratively (including Farkas- and \(\pi\)-based modules in the current implementation).

This architecture is implemented in:
- `master.py`
- `optimize.py`
- `farkas.py`
- `pi.py`
- `utils.py`

## 3. Repository Organization

```text
OAP/
├── main.py
├── models/
│   ├── __init__.py
│   ├── gurobi.py
│   └── benders/
│       ├── __init__.py
│       ├── master.py
│       ├── optimize.py
│       ├── farkas.py
│       ├── pi.py
│       └── utils.py
├── polihedral/
│   └── function.py
├── utils/
│   ├── __init__.py
│   ├── utils.py
│   ├── geometry_classifier.py
│   ├── model_stats.py
│   └── analyze_benders.py
├── test/
│   ├── test.py
│   ├── test_benders.py
│   ├── test_huge.py
│   ├── test_instance.py
│   └── benders_postmortem_analysis.py
├── instance/
├── outputs/
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── README.md
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

### Run

```bash
python main.py
```

### Tests

```bash
pytest -q
```

> **Solver requirement:** A valid **Gurobi** installation and license are required.

## 5. Current Research Status

This repository is under active development. Recent work includes:

- refactoring of the Benders pipeline into a dedicated package (`models/benders/`),
- extension of geometry-aware utilities (`utils/geometry_classifier.py`),
- improvements in typing and constraint-level testing.

The project is intended as a research codebase; interfaces and experimentation workflows may evolve.

## 6. Citation

If this repository contributes to your research, please cite via `CITATION.cff` (or GitHub “Cite this repository”).

```text
García, Á. Optimal Area Polygonization (OAP): MILP Framework.
Repository: https://github.com/aalvarogaarcia/OAP
```

## 7. References

- Hernández-Pérez, H., Riera-Ledesma, J., Rodríguez-Martín, I., & Salazar-González, J. J.  
  *Optimal area polygonisation problems: Mixed integer linear programming models*.
- Hosseini, M., & Turner, J.  
  *Deepest Cuts for Benders Decomposition*.