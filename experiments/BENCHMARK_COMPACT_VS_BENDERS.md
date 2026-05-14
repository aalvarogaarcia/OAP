# Benchmark: OAPCompactModel vs 3 OAPBendersModel Variants

## Overview

This benchmark compares the performance of:

1. **OAPCompactModel** (Threads=1) — Direct compact formulation
2. **Benders Farkas (Vanilla)** — Basic Benders with farkas cuts
3. **Benders Magnanti-Wong (LP core)** — Benders with MW-optimal cuts, core point from LP relaxation
4. **Benders DDMA** — Benders with Algorithm 3 from Hosseini & Turner 2025

All models use:
- Objective: Fekete (area maximization)
- Subtour elimination: SCF (Subtour Elimination Flow)
- **sum_constrain=True** (mandatory for all)
- Seed=0, Threads=1 (deterministic)
- TimeLimit=7200s (default, configurable)

Each instance is solved twice: once with **maximize=True** and once with **maximize=False**, enabling comparison of min-area vs max-area formulations.

---

## Usage

### Interactive Mode (Default)

```bash
.venv/bin/python experiments/benchmark_compact_vs_benders.py
```

Prompts you to:
1. Select instance directory and glob pattern
2. Choose instances to benchmark (checkboxes)
3. Choose models to compare (checkboxes)
4. Set time limit (s)
5. Enable profiling (yes/no)

### Smoke Test (Quick Validation)

```bash
.venv/bin/python experiments/benchmark_compact_vs_benders.py --smoke-test
```

Runs the first 2 instances in `instance/little-instances/` with all 4 methods and 60s time limit. Completes in < 5 minutes for validation.

### Headless Mode (JSON Config)

```bash
.venv/bin/python experiments/benchmark_compact_vs_benders.py --config config.json
```

Example `config.json`:
```json
{
  "dir_path": "instance",
  "ext": "*.instance",
  "instances": [
    "instance/euro-night-0000010.instance",
    "instance/euro-night-0000015.instance"
  ],
  "methods": [
    "compact_threads1",
    "benders_farkas_vanilla",
    "benders_mw_lp",
    "benders_ddma"
  ],
  "time_limit": 7200,
  "enable_profiling": false
}
```

---

## Output Files

### CSV Results

**Location:** `outputs/CSV/benchmark_compact_vs_benders_{YYYYMMDD_HHMMSS}.csv`

**Columns:**
- `instance` — instance name (stem of filename)
- `n_nodes` — number of points in the instance
- `method` — method key (compact_threads1, benders_farkas_vanilla, benders_mw_lp, benders_ddma)
- `maximize` — True if objective was maximization, False for minimization
- `root_lp` — LP relaxation value (4 decimals)
- `final_ip` — best integer solution found (4 decimals), or "-" if no solution
- `gap_pct` — optimality gap as percentage (4 decimals), or "-"
- `time_s` — wall-clock time in seconds (2 decimals)
- `nodes` — number of B&B nodes explored, or "-"
- `total_cuts` — number of Benders cuts added as lazy constraints during B&B; always 0 for compact_threads1
- `status` — "OK" for success, or "FAILED: {ExceptionType}: {message}"

**Example row:**
```
feat-02-01,25,benders_mw_lp,True,1234.5600,1240.9800,0.5100,4.1200,89,142,OK
```

### Configuration Snapshot

**Location:** `outputs/CSV/benchmark_compact_vs_benders_{YYYYMMDD_HHMMSS}_config.json`

Captures:
- Full benchmark configuration (instances, methods, time limit, profiling)
- System info (platform, Python version, processor)
- Run timestamp (ISO 8601)

Use this for reproducibility and audit trails.

### Markdown Report

**Location:** `outputs/reports/benchmark_compact_vs_benders_{YYYYMMDD_HHMMSS}.md`

Includes:
- **Metadata** — Run ID, date, hardware, solver info
- **Method Descriptions** — Table with model type and label for each method
- **Raw Results** — Per-instance tables showing all solves (instance × method × maximize)
- **Summary Statistics**:
  - Time performance (total, average, min, max, OK count per method)
  - Optimality gap (median, max, average per method)
- **Profiling Info** (if enabled) — pointer to profiling CSVs

---

## Profiling (Optional)

If `enable_profiling=yes`, the benchmark generates:

### Per-Instance Profiling CSV

**Location:** `outputs/profiling/profile_{method}_{instance_stem}.csv`

**Columns:** function, ncalls, tottime, percall_tot, cumtime, percall_cum

Shows which functions consumed the most time during the solve for that instance+method combination.

### Profiling Design

- **Zero overhead on timing:** cProfile wraps only `model.build() + model.solve()`. Wall-clock time is measured outside the profiler.
- **Safe for comparison:** Profiling overhead (~5-10%) is not included in reported `time_s`.
- **Optional:** Disabled by default; enable only for diagnostic/bottleneck analysis.

---

## Method Configuration Reference

| Method | Benders Method | MW? | DDMA? | Core Point Strategy |
|--------|-----------|-----|-------|---------------------|
| `compact_threads1` | — | No | No | — |
| `benders_farkas_vanilla` | farkas | No | No | — |
| `benders_mw_lp` | farkas | Yes | No | lp_relaxation |
| `benders_ddma` | farkas | No | Yes | — |

All share: Fekete objective, SCF subtour, sum_constrain=True, Seed=0, Threads=1.

---

## Interpreting Results

### For Time Comparison

Lower is better. Focus on:
- **Total (s):** Sum across all instances. Indicates scalability.
- **Avg (s):** Mean time per instance. Better for comparing single-instance performance.
- **Max (s):** Slowest instance. Indicates worst-case behavior.

Example interpretation:
```
Benders DDMA: Total=128.5s, Avg=12.8s
Compact:      Total=123.4s, Avg=12.3s
Conclusion: Compact is slightly faster on average, but DDMA dominates on harder instances.
```

### For Optimality Gap

Lower is better. Metrics:
- **Median gap (%):** Robust to outliers; use for typical performance.
- **Max gap (%):** Worst case; indicates robustness.
- **Avg gap (%):** Arithmetic mean; useful for aggregation.

Example:
```
Method            | Median | Max  | Avg
compact_threads1  | 0.00%  | 1.52 | 0.41%
benders_mw_lp     | 0.04%  | 0.67 | 0.15%
Conclusion: MW achieves tighter gaps and less variance.
```

### Statistical Significance

With n instances and 2 objectives (max/min), you have 2n data points per method.
- Small differences (<5% in time, <0.1% in gap) may be noise.
- Use the CSV to compute confidence intervals or run paired t-tests.

---

## Tips & Troubleshooting

### Instance Selection

- Start with **small instances** (n=10–15) for quick validation.
- Scale to **medium** (n=20–30) for performance trends.
- **Large instances** (n=45–50) may timeout; increase `time_limit` or use fewer methods.

### Profiling Overhead

- Profiling adds ~5–10% wall-clock overhead.
- Disable profiling for accurate timing comparisons.
- Enable only for bottleneck identification (which functions use the most time).

### Reproducing Results

- Always record the config JSON snapshot (automatically saved).
- Use the same **instance set** and **time limit** for comparisons.
- Run with **Seed=0, Threads=1** (enforced by default).

### Missing Instance Files

The benchmark will skip missing instances and warn you:
```
WARNING: Missing instances: ['instance/foo.instance']
```

Ensure instances are in the correct directory and match the glob pattern.

---

## Example Workflow

### 1. Quick Validation (Smoke Test)

```bash
.venv/bin/python experiments/benchmark_compact_vs_benders.py --smoke-test
```

Output: `benchmark_compact_vs_benders_20260514_142000.*` in `outputs/CSV/` and `outputs/reports/`

### 2. Interactive Full Run

```bash
.venv/bin/python experiments/benchmark_compact_vs_benders.py
```

Prompts you through:
- Instance dir: `instance`
- Glob: `*.instance`
- Select: 10–15 point instances (e.g., euro-night-000001*0, london-000001*0)
- Methods: all 4 (default)
- Time limit: 7200 (default)
- Profiling: no (default)

### 3. Analyze CSV Output

Open `outputs/CSV/benchmark_compact_vs_benders_*.csv` in Excel/Pandas:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/CSV/benchmark_compact_vs_benders_20260514_142000.csv")

# Filter to maximize=True only
df_max = df[df['maximize'] == True]

# Convert sentinel "-" to NaN for numeric columns
for col in ['root_lp', 'final_ip', 'gap_pct', 'time_s', 'nodes', 'total_cuts']:
    df_max[col] = pd.to_numeric(df_max[col], errors='coerce')

# Time per method
print(df_max.groupby('method')['time_s'].agg(['sum', 'mean', 'min', 'max']))

# Gap per method
print(df_max.groupby('method')['gap_pct'].agg(['median', 'max', 'mean']))

# Cuts per method (Benders only)
benders = df_max[df_max['method'] != 'compact_threads1']
print(benders.groupby('method')['total_cuts'].agg(['sum', 'mean', 'max']))

# Performance profile: cumulative instances solved within time t
import matplotlib.pyplot as plt
for method, grp in df_max.groupby('method'):
    times_sorted = np.sort(grp['time_s'].dropna())
    plt.plot(times_sorted, np.arange(1, len(times_sorted)+1), label=method)
plt.xlabel('Time (s)'); plt.ylabel('# Instances solved'); plt.legend(); plt.show()
```

### 4. Compare vs Baseline

Using Compact as baseline (fastest on small instances):
```python
compact_avg = df_max[df_max['method'] == 'compact_threads1']['time_s'].mean()
for method in ['benders_farkas_vanilla', 'benders_mw_lp', 'benders_ddma']:
    benders_avg = df_max[df_max['method'] == method]['time_s'].mean()
    slowdown = (benders_avg / compact_avg - 1) * 100
    print(f"{method}: {slowdown:+.1f}% vs Compact")
```

---

## Known Limitations

1. **No automatic plotting** — CSV is provided; use pandas + matplotlib for performance profiles (see example above).
2. **Single-threaded only** — Threads=1 enforced for reproducibility; parallelism not benchmarked.
3. **Deterministic seed** — Seed=0 for reproducibility; stochastic behavior not explored.
4. **Fekete objective only** — Other objectives (Internal, External, Diagonals) not compared.
5. **`total_cuts` counts lazy constraints delta** — this equals the number of Benders cuts injected via the lazy callback. For Compact, it is always 0. Note that Gurobi may also add its own cuts (Gomory, clique, etc.); those are reflected in `model.model.NumGenConstrs` separately and are not counted here.

---

## Citation & Reproducibility

If you use this benchmark in a paper:

```
@misc{OAP_BenchmarkCompactVsBenders2026,
  author = {García-Muñoz, Álvaro},
  title = {Benchmark: OAPCompactModel vs Benders Variants (DDMA, MW, Farkas)},
  year = {2026},
  howpublished = {OAP GitHub repository},
  note = {Run ID: {run_id}, Config snapshot: {config_json_file}}
}
```

Always reference the **config JSON snapshot** (captured run metadata) in appendices.

---

## Further Reading

- **Benders decomposition theory** — `context/own-benders-derivation.md`
- **DDMA Algorithm 3** — Hosseini & Turner 2025, §4.1
- **Magnanti-Wong cuts** — Magnanti & Wong 1981; applied in Benders context
- **MT3D baseline** — Hernández-Pérez et al. 2025
