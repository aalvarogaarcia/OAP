# OAP_NextGen — Claude Code entry point

This is the project context Claude Code loads automatically. It points at three deeper sources of truth and adds only what isn't covered there.

## What this project is

**Optimal Area Polygonization (OAP)** — research codebase for exact MILP and Benders-decomposition methods for MIN-OAP and MAX-OAP on the CG:SHOP 2019 instance set. Goal: improve the state-of-the-art **MT3D** model of Hernández-Pérez et al. (EJOR 2025) along two fronts: Benders decomposition and new families of valid inequalities. Author: Álvaro García-Muñoz (`agarmun@gmail.com`).

## Read these first (in order)

@.claude/steering.md
@.claude/requirements.md
@AGENTS.md

- `.claude/steering.md` — project identity, principles, scope, research-stage pipeline.
- `.claude/requirements.md` — FRs, NFRs, milestones, open questions.
- `AGENTS.md` — environment, commands, architecture, model API, known gotchas.

For deeper context (math derivations, formulations, prior art) see `.claude/context/`. The current state of the codebase lives in `.claude/context/code-status.md` — refresh after every feature lands.

## Subagents

Ten specialised subagents live in `.claude/agents/`. Each carries its own startup checklist. Pick the one matching the stage:

| Stage | Agent |
|---|---|
| Formal math, Benders dual derivations, validity proofs | `mathematical-formalist` |
| Plan a build (milestones, dependencies, exit criteria) | `implementation-planner` |
| Design code (modules, signatures, data contracts) | `implementation-designer` |
| Break a design into ≤1-day developer tasks | `task-definer` |
| Write code (Gurobi/Python, fixtures, baseline regression) | `developer` |
| Review a commit / PR (math, regression, design adherence) | `code-reviewer` |
| Design experiment protocols (instances, metrics, baselines) | `experiment-designer` |
| Analyse experiment CSVs, write paper-grade results | `results-analyst` |
| Maintain `references.md`, refresh state-of-the-art | `literature-reviewer` |
| Draft / revise paper sections | `paper-writer` |

Invoke via `/agents` or via the `Task` tool with `subagent_type` set to the agent name.

## Quick dev commands

All commands run from repo root with the uv-managed venv. **Never use bare `python` / `python3`.**

```bash
# Headless single-instance solve (edit instance_name + config in run.py)
.venv/bin/python run.py

# Batch over a directory (--obj 0=Fekete, 1=Internal, 2=External, 3=Diagonals)
.venv/bin/python main.py instance/little-instances "*.instance" --time-limit 60 --obj 0

# Tests (auto-skip when instance/ files or reference TSV are absent)
.venv/bin/python -m pytest -q

# Lint + type-check (mirrors CI)
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy . --ignore-missing-imports
```

`AGENTS.md` carries the full command reference and the model-API cheat sheet — read it before writing any solver code.

## Standard model workflow

```python
from models import OAPCompactModel  # or OAPBendersModel, OAPInverseBendersModel
from utils.utils import compute_triangles, read_indexed_instance

points    = read_indexed_instance("instance/us-night-0000008.instance")
triangles = compute_triangles(points)
model     = OAPCompactModel(points, triangles, name="us-night-0000008")
model.build(objective="Fekete", maximize=True, subtour="SCF")
model.solve(time_limit=300, verbose=True)
lp, gap, ip, time_s, nodes = model.get_model_stats()
```

Never reuse `points` / `triangles` across different model instances.

## Project principles (binding — see steering.md for the full list)

1. **Notation discipline.** Use Hernández-Pérez et al. (2025) notation. `.claude/context/problem-statement.md` is canonical.
2. **MT3D direct solve is the unmovable baseline.** Every new method is benchmarked against vanilla `OAPCompactModel`.
3. **Math first, code second.** If code disagrees with the derivation in `.claude/context/own-benders-derivation.md`, the derivation wins.
4. **Symmetry-break and Theorem 10 filtering up-front.** Never rely on the solver to discover them.
5. **Negative results are first-class.** Document and park; do not silently drop.
6. **The paper is the ultimate artefact.** Every derivation, design, and result must be writable into a paper section without retroactive cleanup.

## Maintenance

- Update `.claude/context/code-status.md` whenever a feature lands or a direction shifts.
- Steering and requirements change deliberately; surface scope changes through the user, not silently.
- `.claude/context/notes/` is append-only; never rewrite a dated note.
- After completing any non-trivial task, surface the task to `task-definer` for tracking, or close the existing one.
