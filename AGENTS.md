# AGENTS.md — OAP_NextGen

## Environment

- **Python interpreter**: `.venv/bin/python` (native Linux venv). Use this path or activate the venv; never use bare `python` or `python3`.
- **All commands must be run from the repo root.**
- Python 3.12.12 (`requires-python = ">=3.11, <3.13"` in `pyproject.toml`).
- Gurobi installation + valid licence required for any model execution or solver tests.

---

## Key Commands

```bash
# Batch run (all instances in a directory) — interactive CLI
.venv/bin/python main.py

# Single instance — interactive CLI
.venv/bin/python run_single_instance.py

# Single instance — CLI mode with flags
.venv/bin/python run_single_instance.py instance-name --model Compacto --objective Fekete --maximize

# Headless single instance (edit instance_name + model config directly in file)
.venv/bin/python run.py

# Tests (many auto-skip when instance files or TSV reference are absent)
.venv/bin/python -m pytest -q

# Lint
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .

# Type check
.venv/bin/python -m mypy . --ignore-missing-imports
```

CI runs lint + type-check only. The test job in `.github/workflows/ci.yml` is commented out — it requires a `GRB_LICENSE_FILE` secret.

---

## Batch Runner (`main.py`)

The batch runner processes multiple instances and produces both LaTeX and CSV outputs. It is fully interactive — no CLI flags needed. When executed, it prompts for:

1. Instance directory path (default: `instance`)
2. File glob pattern (default: `*.instance`)
3. Subtour-elimination method: `SCF`, `MTZ`, `MCF`
4. Objective function: `Fekete`, `Internal`, `External`, `Diagonals`
5. Maximize objective (boolean)
6. Objective mode (0–3, shoelace variants)
7. Triangle-sum constraints (boolean)
8. Strengthening constraints (boolean)
9. Semiplane constraints: `0 (off)`, `1 (V1)`, `2 (V2)`
10. Local knapsack constraints (boolean)
11. Clique constraints (boolean)
12. Crossing arc constraints (boolean)
13. Output file name base (default: `resultados`)
14. Time limit in seconds (default: `7200`)

**Output format:**
- LaTeX table: `outputs/LaTex/{output_name}.tex` (beamer presentation format)
- CSV spreadsheet: `outputs/CSV/{output_name}.csv` (with columns: Instance, |N|, Convex Hull Area, Cols, Rows, LP Value, Gap (%), IP Value, Time (s), Nodes)

Both outputs contain the same data; LaTeX is for reporting, CSV is for further analysis.

---

## Architecture

### Models

```
from models import OAPCompactModel, OAPBendersModel   # only public import path
```

Both inherit from `OAPBaseModel(OAPStatsMixin)`.  
`OAPBendersModel` also inherits eight mixins (MRO order matters):
```
OAPBendersModel(
    BendersMasterMixin,
    BendersDDMAMixin,         # DDMA Algorithm 3 (Hosseini & Turner 2025 §4.1)
    BendersCGSPMixin,         # Deepest cuts via CGSP normalised LP
    BendersMagnantiWongMixin, # Magnanti-Wong Pareto-optimal cuts
    BendersFarkasMixin,
    BendersPiMixin,
    BendersOptimizeMixin,
    BendersAnalysisMixin,
    OAPBaseModel
)
```

Mixin responsibilities:
- `BendersDDMAMixin` — implements Algorithm 3 from Hosseini & Turner (2025). No auxiliary LP; perturbs RHS iteratively to generate deepest cuts.
- `BendersCGSPMixin` — deepest cuts via a normalised separation LP (CGSP). Relaxed-ℓ₁ weights by default; caches the Gurobi model across callbacks.
- `BendersMagnantiWongMixin` — secondary LP to select the cut with maximum core-point depth (Magnanti-Wong Pareto-optimal cuts).

### Standard workflow (always this order)

```python
points    = read_indexed_instance(filepath)   # NDArray[np.int64] shape (n, 2)
triangles = compute_triangles(points)         # NDArray[np.int64] shape (T, 3)
model     = OAPCompactModel(points, triangles, name="my_instance")
model.build(objective="Fekete", maximize=True, subtour="SCF")
model.solve(time_limit=300, verbose=True)
lp, gap, ip, time_s, nodes = model.get_model_stats()
```

Never reuse `points`/`triangles` computed for one model in a different model instance.

### Utils layout — `utils/utils.py` is a shim, never add logic there

| Module | Contents |
|---|---|
| `utils/geometry.py` | Pure geometry, I/O helpers, `Arc`, `PointLookup` |
| `utils/constraints.py` | Gurobi constraint injectors, `ArcConstraintMap` |
| `utils/benders_log.py` | Cut JSONL logging, `SerializedCoeffMap/Expr/RayData` |
| `utils/visualization.py` | All plotting functions |
| `utils/utils.py` | Re-export shim only — imports from the four above |

New geometry/I-O code → `utils/geometry.py`.  
New constraint logic → `utils/constraints.py`.  
New Benders logging → `utils/benders_log.py`.  
New plots → `utils/visualization.py`.

---

## Model API Reference

### `OAPCompactModel`

```python
model.build(
    objective: Literal["Fekete", "Internal", "External", "Diagonals"] = "Fekete",
    mode: int = 0,                         # shoelace variant 0–3
    maximize: bool = True,
    subtour: Literal["SCF", "MTZ", "MCF"] = "SCF",
    sum_constrain: bool = True,
    semiplane: Literal[0, 1, 2] = 0,       # 0=off, 1=V1, 2=V2
    use_knapsack: bool = False,
    use_cliques: bool = False,
)
model.solve(time_limit=7200, verbose=False, relaxed=False, plot=False)
model.log_facets(filepath, var_prefixes="x", verbose=False)  # call after solve()
```

### `OAPBendersModel`

```python
model.build(
    objective: Literal["Fekete", "Internal", "External", "Diagonals"] = "Fekete",
    mode: int = 0,
    maximize: bool = True,
    benders_method: Literal["farkas", "pi"] = "farkas",
    subtour: Literal["SCF", "DFJ"] = "SCF",
    sum_constrain: bool = True,
    crosses_constrain: bool = False,
    strengthen: bool = False,
    plot_strengthen: bool = False,
    use_deepest_cuts: bool = False,       # CGSP deepest cuts
    cut_weights_y: dict | None = None,    # custom L₁ weights for Y
    cut_weights_yp: dict | None = None,   # custom L₁ weights for Y'
    semiplane: Literal[0, 1] = 0,
    use_knapsack: bool = False,
    use_cliques: bool = False,
    use_magnanti_wong: bool = False,      # Magnanti-Wong cuts
    core_point_strategy: Literal["lp_relaxation", "uniform"] = "lp_relaxation",
    cgsp_norm: Literal["misd", "relaxed_l1"] = "relaxed_l1",
    use_ddma: bool = False,               # DDMA Algorithm 3
) -> None
```

**Mutual exclusivity:** `use_deepest_cuts`, `use_magnanti_wong`, and `use_ddma` are mutually exclusive — at most one may be `True` at a time.

```python
model.solve(time_limit=7200, verbose=False, relaxed=False,
            save_cuts=False, polihedral=False)
model.set_log_path(path)   # override default outputs/Others/Benders/{name}/log.json
model.solve_lp_relaxation(time_limit, verbose)   # standalone LP solve
```

---

## Benchmark

`experiments/benchmark_benders_general.py` compares all seven cut strategies across instances.

**Available methods:** `farkas`, `cgsp_farkas`, `cgsp_pi`, `mw_lp`, `mw_uniform`, `ddma_farkas`, `ddma_pi`

```bash
# Interactive (inquirer prompts with descriptive labels for each method)
.venv/bin/python experiments/benchmark_benders_general.py

# Headless from JSON config
.venv/bin/python experiments/benchmark_benders_general.py --config experiments/configs/general_all7_n20.json

# Quick smoke-test (little-instances/, 60s, all methods)
.venv/bin/python experiments/benchmark_benders_general.py --smoke-test
```

Pre-built configs in `experiments/configs/`:

| File | Description |
|---|---|
| `general_all7_n10_n15.json` | 12 instances n=10,15, all 7 methods |
| `general_all7_n20.json` | 6 instances n=20, all 7 methods |
| `general_ddma_only_n20.json` | farkas + ddma_farkas + ddma_pi only |

---

### `main.py` CLI integer codes (different from model string keys)

| `--obj` | model `objective` |
|---|---|
| `0` | `"Fekete"` |
| `1` | `"Internal"` |
| `2` | `"External"` |
| `3` | `"Diagonals"` |

Subtour in `main.py` is controlled by the `subtour_methods` list (line ~106), currently hardcoded to `[2]` (MCF). Edit that list to run comparisons. The CLI has no `--subtour` flag.

---

## Type Aliases

| Alias | Definition | Location |
|---|---|---|
| `Arc` | `tuple[int, int]` | `utils/geometry.py` |
| `PointLookup` | `dict[int, tuple[float,float]] \| NDArray[np.int64]` | `utils/geometry.py` |
| `ArcConstraintMap` | `dict[Arc, gp.Constr]` | `utils/constraints.py` |
| `SerializedCoeffMap` | `dict[str, float]` | `utils/benders_log.py` |
| `SerializedExpr` | `dict[str, SerializedCoeffMap \| float]` | `utils/benders_log.py` |
| `NumericArray` / `IndexArray` | `NDArray[np.number]` / `NDArray[np.integer]` | `models/typing_oap.py` |

---

## Testing Quirks

- **`test_benders.py`** requires `instance/us-night-0000010.instance`. Missing → auto-skip (not failure).
- **`test_compact_minimize.py` and `test_compact_maximize.py`** are parametrised from a reference TSV. Both files look for it at:
  ```
  {repo_root}/test/data/TablaResultadosA4.tsv
  ```
  `main.py` looks for the same file at `test/TablaResultadosA4.tsv` (no `data/` subdirectory). **These paths are inconsistent.** If the TSV is missing, the parametrised test list is empty and all tests are silently skipped.
- `instance/` and `outputs/` are git-ignored. Solver tests that need specific instance files will skip gracefully via `pytest.skip`.
- `outputs/Others/Testing/` is created at import time in the test files — this is a side effect of importing the test module.

---

## Known Gotchas

- **`os.makedirs(os.path.dirname(filepath), exist_ok=True)` crashes when `filepath` has no directory component** — `os.path.dirname("file.jsonl")` returns `""`. Guard with:
  ```python
  parent = os.path.dirname(filepath)
  if parent:
      os.makedirs(parent, exist_ok=True)
  ```
  `utils/visualization.py` provides `_ensure_dir(path)` for this. `OAPBaseModel.log_facets` (line 183) still has the bare call — it only works because the default path always contains a directory.

- **`model._x` is a plain dict, not a Gurobi attribute.** The codebase attaches `model._x`, `model._x_results`, and `model._points_` as Python attributes on the `gp.Model` object. This is the established pattern — continue it.

- **`run_single_instance.py` supports both interactive and CLI modes.** Omit `instance_name` argument to use interactive `inquirer` prompts; provide it with optional flags for CLI mode (e.g., `run_single_instance.py my-instance --model Benders`). Use `run.py` for fully headless execution without any prompts.

- **Mutable default arguments in `OAPBaseModel.extract_subspace_facets` and `extract_facets`** use `var_prefixes: str | list[str] = ['x']`. Do not mutate the default. Pass an explicit list when calling.

- **`cdd` (pycddlib)** is used in `OAPBaseModel.extract_facets`. Requires `pycddlib<3.0.0` (pinned in `pyproject.toml`). The `cdd.Matrix` / `cdd.RepType` API changed in v3.

- **Benders cut log default path**: `outputs/Others/Benders/{name}/log.json` — the `outputs/` directory is git-ignored. CI will never have this file.

- **`main.py` output directories** (`outputs/LaTex/`, `outputs/CSV/`) are created automatically by the script. They are produced by solver batch runs and are git-ignored.
