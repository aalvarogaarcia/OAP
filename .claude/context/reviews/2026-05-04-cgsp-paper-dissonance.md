# CGSP Implementation vs Hosseini-Turner (2025): Dissonance Audit

**Date:** 2026-05-04
**Paper:** Hosseini, M. and Turner, J. *Deepest Cuts for Benders Decomposition*, Operations Research 73(5):2591-2609, 2025. ([`papers/Deepest-BendersCuts.md`](../../papers/Deepest-BendersCuts.md))
**Code under review:** main repo (uncommitted),
- `models/mixin/benders_cgsp_mixin.py`
- `models/mixin/benders_mw_mixin.py`
- `models/mixin/benders_optimize_mixin.py`
- `experiments/benchmark_benders_general.py`
**Benchmark report reviewed:** `outputs/reports/benchmark_general_20260504_173713.md`

---

## 1. Paper summary (≤300 words)

The paper unifies a wide family of Benders cut-selection strategies under one umbrella by introducing **distance-based Benders decomposition**. The starting object is the *certificate-generating subproblem* (CGSP, eq. 12, p. 2593):

$$\max_{(\pi,\pi_0)\in\Pi}\ \pi^T(b-By) + \pi_0(f^T y-\bar\eta),\quad \Pi=\{(\pi,\pi_0):\pi^T A\le \pi_0 c^T,\ \pi\ge 0,\ \pi_0\ge 0\}.$$

The cone $\Pi$ is unbounded, so $(\pi,\pi_0)$ must be normalised. The paper shows (Prop. 6, p. 2598) that any positive-homogeneous normalisation $h(\pi,\pi_0)$ is admissible; truncating $\Pi$ by $h\le 1$ converts the CGSP into a **normalised separation problem** (NSP, eq. 22). Concretely:

- **$\ell_p$-norm** normalisation (§2.2, eq. 17): $h(\pi,\pi_0)=\|(\pi^TB-\pi_0 f^T,\pi_0)\|_p$, $p\in\{1,2,\infty\}$, giving Euclidean / sparse / dense deepest cuts. $\ell_1$ and $\ell_\infty$ yield LPs.
- **Linear pseudonorms** $h(\pi,\pi_0)=w^T\pi+w_0\pi_0$ (§3.3, p. 2599-2601): MIS-default (Fischetti et al. 2010), Relaxed-$\ell_1$ (§3.3.2), Magnanti-Wong-Papadakos (§3.3.3, eq. 28-29), Conforti-Wolsey (§3.3.4, eq. 30).
- **DDMA** (Algorithm 3, p. 2602): an iterative method that *avoids building the full normalised LP* — it solves a sequence of classic Benders subproblems with a perturbed RHS $b-By-zw$, increasing $z$ until the depth constraint becomes feasible.

Theorem 3 (p. 2602) gives finite convergence for any **convex piecewise-linear** $h$. Section 5 (computational study) reports that on CFLP / MCNDP / SNIP, $\ell_1$-deepest cuts solve **all** instances with smaller root gaps than CB / MISD / RW / MWP / CW, and that DDMA delivers ≈80% reduction in cut-generation time vs solving the LP via a generic LP solver.

The paper thus separates **what cut to select** ($h$) from **how to compute it** (LP / DDMA / GPA). The current implementation only realises the first half (LP-based with $\ell_1$ normalisation), and incompletely.

---

## 2. Mathematical dissonances — concrete bugs

### 2.1 BLOCKING — MW core point is empty by construction (`self._x` vs `self.x`)

- **Files / lines:** `models/mixin/benders_mw_mixin.py:69` and `:79`.
- **Implementation:** `_compute_core_point` enumerates arcs as
  ```
  return {arc: 1.0 / n for arc in getattr(self, "_x", {})}
  ```
  i.e. it iterates over `self._x` (with leading underscore).
- **Reality:** the master variable dictionary is `self.x` (no underscore), defined in `models/mixin/benders_master_mixin.py:95`. There is no `self._x` anywhere in the codebase (verified by `rg "self\._x"`). `getattr(self, "_x", {})` therefore always returns `{}`, and the comprehension produces an empty dict.
- **Consequence cascade:**
  1. `OAPBendersModel.__init__` → `_compute_core_point` returns `{}`.
  2. `self._core_point = {}`.
  3. In `get_mw_cut_y` / `get_mw_cut_yp`, the guard `if not getattr(self, "_core_point", None):` is truthy (empty dict is falsy), so the function returns `(None, None, {"aborted": "no_core_point"})`.
  4. In `benders_optimize_mixin.py:165-173` and `:218-226` the dispatch falls back to legacy Farkas. **Hence `mw_lp` and `mw_uniform` produce identical numerics to `farkas`** — exactly the symptom in the benchmark report (rows 31-32, 41-42, 51-52, 61-62, 71-72, 81-82 of `outputs/reports/benchmark_general_20260504_173713.md`).
- **Severity:** BLOCKING. MW cuts are never applied.
- **Fix:** replace `getattr(self, "_x", {})` with `getattr(self, "x", {})` at `benders_mw_mixin.py:69` and `:79`.

### 2.2 BLOCKING — `r3` dual feasibility coefficient wrong when a triangle hits both arcs of a crossing pair

- **File / lines:** `benders_cgsp_mixin.py:325-331`.
- **Implementation:** for each $r_3$ key $(i,j,k,s)$ and each triangle $t$,
  ```
  if t in _adj(i, j) or t in _adj(k, s):
      ...
      expr -= v
  ```
  The coefficient of $y_t$ in the column $A^T\pi$ is taken as $1$ when $t$ is in either adjacency, and the dual term contributes $-v_{r3}$.
- **Paper / math:** the primal $r_3$ row is
  $$\sum_{t\in\mathrm{adj}(i,j)} y_t + \sum_{t\in\mathrm{adj}(k,s)} y_t \le 1 - x_{ji} - x_{ks},$$
  so the coefficient of $y_t$ in that row equals $\mathbb{1}_{t\in\mathrm{adj}(i,j)}+\mathbb{1}_{t\in\mathrm{adj}(k,s)}\in\{0,1,2\}$. The dual feasibility row (col $t$) is
  $$\sum_{(i,j,k,s)}\bigl(\mathbb{1}_{t\in\mathrm{adj}(i,j)}+\mathbb{1}_{t\in\mathrm{adj}(k,s)}\bigr)\cdot\pi_{r3,(i,j,k,s)} \le 0.$$
  When $t$ lies on both arcs the implementation subtracts $v_{r3}$ once instead of twice.
- **Agreement?** No — disagreement when $t\in\mathrm{adj}(i,j)\cap\mathrm{adj}(k,s)$. Geometrically this happens for crossing pairs that share a triangle; on Delaunay-style triangulations of convex point sets it is rare but not impossible.
- **Severity:** CORRECTNESS (validity preserved only in the absence of double-incidence; otherwise the cut may be invalid/infeasible).
- **Fix:** count both indicators and accumulate accordingly:
  ```
  coef = (1 if t in _adj(i,j) else 0) + (1 if t in _adj(k,s) else 0)
  if coef and v is not None:
      expr -= coef * v
  ```

### 2.3 CORRECTNESS — only $\ell_1$ normalisation is implemented, but the paper finds $\ell_1$/$\ell_\infty$ comparable and recommends DDMA when there is structure

- **Files / lines:** `benders_cgsp_mixin.py:514`, `:705` (`norm_expr == 1.0`).
- **Implementation:** the splitting $\pi = u-v$, $u,v\ge 0$, with `Σ w_i (u_i+v_i) = 1` is exactly the LP encoding of $\|\pi\|_{1,w}=1$. With unit weights this is **MISD** in the taxonomy of §3.3 (eq. 27, p. 2599) — *not* the geometric $\ell_1$-deepest cut, which uses $h=\|(\pi^TB-\pi_0 f^T,\pi_0)\|_1$ (paper eq. 17, p. 2596) i.e. $w_i=\sum_j |B_{ij}|$, $w_0=1+\sum_j |f_j|$ (Relaxed-$\ell_1$ in §3.3.2, p. 2600). The deepest cuts in the paper (§5.2, p. 2604) significantly outperform MISD on both root gap and time.
- **Severity:** CORRECTNESS — cuts are valid but suboptimal.
- **Fix:** compute the column sums $w_i^{(yp)} = \sum_j |B_{ij}^{(yp)}|$ at build time (one-shot, since $B$ is data-only) and use them as the default weight map; expose $\ell_\infty$ as an alternative single-bound encoding `-1 ≤ π_i ≤ 1` to compare.

### 2.4 BLOCKING — runtime reports `type str doesn't define __round__ method` are NOT a CGSP bug, they are a benchmark post-processing crash

- **Files / lines:** `experiments/benchmark_benders_general.py:208-211`:
  ```
  "root_lp": round(lp, 4) if lp is not None else None,
  "final_ip": round(ip, 4) if ip is not None else None,
  "gap_pct": round(gap, 4) if gap is not None else None,
  ```
  Trace: `get_model_stats()` (`models/mixin/oap_stats_mixin.py:111-141`) returns the literal string `"-"` (not `None`) when (a) `model.SolCount == 0`, (b) `_lp_converged is False`, or (c) `lp.SolCount == 0` after the relax. The benchmark only checks `lp is not None`, so `round("-", 4)` raises the observed exception.
- **Why london, stars, uniform-2, us-night fail and euro-night, uniform-1 pass:** for the failing instances the *manual LP-relaxation Benders loop* in `benders_optimize_mixin.py:357-554` either hits `MAX_LP_ITER = 500·|x|` (line 354) or its termination triggers degenerate-PI handling (lines 462-469, 530-538), in either case setting `_lp_converged = False` and producing `"-"`. This happens whenever CGSP fails to drive the LP relaxation to convergence — which is consistent with section 2.5 below (slow CGSP cuts).
- **Severity:** BLOCKING for the benchmark script (instances are silently dropped from the geometric-mean tables); **diagnostic value** for CGSP itself: the failure marker is correlated with non-convergence of CGSP-LP.
- **Fix (benchmark):** at `benchmark_benders_general.py:208-211`,
  ```
  def _safe_round(v, n):
      return round(v, n) if isinstance(v, (int, float)) else None
  ```
  and apply to all three fields. This is post-processing only; it does not affect Benders correctness.
- **Suggested 5-line debug instrumentation patch** (insert at `benchmark_benders_general.py:218`, *before* the `except` line that swallows the trace):
  ```python
  import traceback
  except Exception as exc:
      logger.error("FAILED [%s] %s\n%s", method, stem, traceback.format_exc())
      row["status"] = f"FAILED: {type(exc).__name__}: {str(exc)[:80]}"
  ```
  This will capture the actual stack trace on stderr while still recording a one-line CSV status. Run a single failing instance (e.g. `london-0000020`) with `cgsp_farkas` to confirm the trace lands inside `_safe_round` (or elsewhere) before applying the cleaner fix.

### 2.5 CORRECTNESS — CGSP is >100× slower than Farkas because every callback rebuilds a fresh Gurobi model

- **Files / lines:** `benders_cgsp_mixin.py:379-380`, `:556-557` — every call to `_build_cgsp_yp` / `_build_cgsp_y` calls `gp.Model("cgsp_yp")` and re-creates all dual variables, the dual-feasibility constraints, and the normalisation. With $|V| \approx \binom{n}{3}$ triangles and constraints per triangle, on $n=20$ instances the CGSP LP can have several hundred constraints and is rebuilt at each integer-feasible MIPSOL.
- **Paper:** Algorithm 2 + §4.1 (DDMA, p. 2602) explicitly recommend warm-starting / re-optimisation. The paper §3 also notes (p. 2598, footer): "a certificate produced at iteration $t$ of the BD algorithm can be used for warm starting the separation subproblem at iteration $t+1$ if $\Pi_h$ does not depend on $(y^{(t)},\eta^{(t)})$" — and indeed for our linear pseudonorms $\Pi_h$ does not depend on $\bar x$.
- **Severity:** CORRECTNESS (cuts are valid but the implementation pays an unnecessary 100× overhead, which is what the benchmark measures).
- **Fix:** persist the CGSP model across callback invocations; in `_build_cgsp_yp`/`_build_cgsp_y`, only update objective coefficients (the only $\bar x$-dependent quantity). Or, more aggressively: implement DDMA (§4.1) which avoids the LP altogether.

### 2.6 CORRECTNESS — sign convention is consistent, but the LinExpr-constant trick is implicit and brittle

- **File / lines:** `benders_cgsp_mixin.py:790-847`, `:935-990` (cut reconstruction).
- **Mathematics:** for the $b_i(x)=b_i^{const}+B_i x$ decomposition, the cut $\sum_i \pi_i b_i(x) \le 0$ becomes
  $$\sum_i \pi_i\,B_i\, x \le -\sum_i \pi_i\,b_i^{const}.$$
  For groups with constant RHS (`global`, `r1`, `r2`) the implementation subtracts from `cut_rhs` (lines 824, 831, 839 — the recently-flipped sign IS correct). For groups with $x$-dependent RHS (`alpha`, `beta`, `gamma`, `delta`, `r3`) the implementation accumulates the entire RHS, including the constant `1.0`, *inside the LinExpr* (e.g. line 795: `cut_expr += pi_val * (1.0 - self.x[i, j])`). When Gurobi receives `cbLazy(cut_expr <= cut_rhs)`, it auto-moves the LinExpr's accumulated constant to the RHS, matching the algebra.
- **Status:** mathematically correct **today**, but coupled to implementation detail of `gp.LinExpr`. If the LinExpr ever loses its constant (e.g. via simplification or a re-emit through `LinExpr.copy()`-like operations) the cut becomes a different inequality.
- **Severity:** CORRECTNESS-COSMETIC (no math impact today; refactor risk).
- **Fix:** make constants explicit by accumulating them into `cut_rhs` directly:
  ```
  cut_expr += -pi_val * self.x[i, j]
  cut_rhs  += -pi_val * 1.0    # alpha_p constant
  ```
  This makes the cut transparent and survives any LinExpr massaging.

### 2.7 CORRECTNESS — optimality-cut sign for `pi0_net < 0` is rejected; the paper leaves room for $\pi_0\le 0$

- **File / lines:** `benders_cgsp_mixin.py:996-1004`.
- **Implementation:** when $\pi_0 < -\text{TOL}$ the cut is dropped (`return None, None, {"aborted": "pi0_negative"}`).
- **Paper:** Definition of $\Pi$ (p. 2593) constrains $\pi_0 \ge 0$. So $\pi_0<0$ is *outside the cone*; rejecting is correct. However, $|π_0|<\text{TOL}$ falls through to a feasibility cut, which is fine. **No bug.** Listed for completeness.
- **Severity:** COSMETIC.

---

## 3. Forensic trace: `type str doesn't define __round__` (continued)

Origin (already triangulated in §2.4): `experiments/benchmark_benders_general.py:208-211`. The exception is raised on the *Python* `round("-", 4)` call inside the benchmark's `run_single_solve`, not inside Gurobi or CGSP.

Differential structure of the failing vs. passing instances:

| Instance | passes cgsp_farkas | passes cgsp_pi |
|---|---|---|
| `euro-night-0000020` | yes (600s, OK) | yes |
| `uniform-0000020-1` | yes (600s, OK) | yes |
| `london-0000020` | no | no |
| `stars-0000020` | no | no |
| `uniform-0000020-2` | no | no |
| `us-night-0000020` | no | no |

Same set fails for both Farkas-base and Pi-base, which (since CGSP itself is independent of Farkas/Pi when in `use_deepest_cuts=True` mode) confirms the failure is in the LP-relaxation pre-pass. The instances that *do* pass time out at the Gurobi MIP level (Status `OK` with 600 s and a valid LP bound), implying their LP relaxation **did** converge inside `MAX_LP_ITER`. The four failing instances have LP relaxations that don't converge — possibly due to combinatorial degeneracy, possibly due to numerical issues with the very large coefficient magnitudes (cf. `stars-0000020` final IP of $2.16\times 10^{11}$ vs. `uniform-0000020-1` of $7.62\times 10^5$).

**Recommended 5-line patch** (already given in §2.4) — confirm by running:
```
python experiments/benchmark_benders_general.py --instances london-0000020 --methods cgsp_farkas
```
once the trace patch is in. The expected stack will pinpoint either `round("-", 4)` (post-processing — fixable in 2 lines) or, if the trace is elsewhere, give us the actual CGSP defect.

---

## 4. Magnanti-Wong dissonances

### 4.1 (See §2.1 above) `self._x` vs `self.x` — BLOCKING.

### 4.2 CORRECTNESS — MW does not actually use the optimal core-point dual; it re-runs CGSP

- **File / lines:** `benders_mw_mixin.py:209-217`, `:255-262`.
- **Implementation:** after the secondary LP is solved with optimal $\pi^\star$, the function discards $\pi^\star$ and calls `self.get_cgsp_cut_yp(x_sol, TOL=TOL)`, which **rebuilds** the CGSP from scratch and re-solves it. The re-solved $\pi$ is in general *different from the MW-Pareto-optimal $\pi^\star$* — it is whatever Gurobi happens to pick this time, which may be Pareto-dominated.
- **Paper:** Magnanti-Wong (1981) and the MWP pseudonorm (Hosseini-Turner §3.3.3, eq. 28, p. 2600) require using the *secondary LP's* solution. Calling the primary solver afterwards is incorrect.
- **Severity:** CORRECTNESS — even after the `self._x` fix, MW will not behave as MW.
- **Fix:** reconstruct the cut directly from `pi_vars` (the variables of `mw`, the *secondary* LP that was just solved), not by re-calling `get_cgsp_cut_*`. The reconstruction logic in `get_cgsp_cut_yp` lines 759-848 should be factored out into `_reconstruct_cut_from_pi(pi_vars, x_sol, which)` and called by both code paths.

### 4.3 CORRECTNESS — the Pareto constraint uses `q_opt` that may be 0 from a Farkas (infeasible) subproblem

- **File / lines:** `benders_mw_mixin.py:188-191`, `:236-239`.
- **Implementation:** when `sub_yp.Status == GRB.INFEASIBLE`, `q_opt = 0.0`. Then the Pareto constraint $\pi^Tb(\bar x) = 0$ is added.
- **Paper / math:** for an *infeasible* subproblem the dual is unbounded, and the "optimal value" is $+\infty$ (Farkas). The MW Pareto constraint $u^T(b-By)=Q(\bar y)$ (eq. 28, p. 2600) is only well-defined when $\bar y\in\mathrm{dom}(Q)$. The Papadakos relaxation (eq. 29) drops the constraint, which is the correct behaviour for infeasibility cuts. Setting $q_{opt}=0$ instead is **mathematically wrong** — it forces $\pi^Tb(\bar x)=0$, which is not the same as "any Pareto-optimal Farkas certificate".
- **Severity:** CORRECTNESS.
- **Fix:** when `sub_yp.Status == GRB.INFEASIBLE`, *do not add the Pareto constraint at all* (Papadakos), and use the existing CGSP-style normalisation. This recovers the MWP pseudonorm of §3.3.3.

---

## 5. Alternative method from the paper not yet implemented

Two strong candidates exist; ranked by the paper's own results:

1. **DDMA (Algorithm 3, §4.1, p. 2602)** — *not implemented*.
2. **Relaxed-$\ell_1$ pseudonorm with column-sum weights (§3.3.2, p. 2600)** — partially implemented (the constraint shape is right but the weights are unit, i.e. MISD).

I recommend **DDMA** as the highest-leverage missing method, motivated by these passages:

> §5.2 (p. 2604): "we observe similar trends in terms of relative effectiveness of different choices of cuts but with a much lower computation cost (around 80% reduction for distance-based cuts and 50% reduction for CB)."
> §5.2 (p. 2605): "BD with $\ell_1$-deepest cuts solved all 169 instances, followed by $\ell_\infty$ and $\ell_2$ (167 instances), CW (166), R$\ell_1$ and MWP (165), and CB (164)."

DDMA is exactly the missing piece between "we have a CGSP LP" and "CGSP is fast enough to be useful in branch-and-cut". It also sidesteps the slow-rebuild cost that §2.5 above pins down as the dominant overhead.

### DDMA in our notation

For a linear pseudonorm $h(\pi)=w^T\pi$ with $w\ge 0$ (no $\pi_0$ in the feasibility-only case):

```
Step 0:  z ← 0.
Step 1 (cut generation): solve the standard primal subproblem
            min { c^T x : A x ≥ b - B x_bar - z·w, x ≥ 0 }.
        If feasible → set π = optimal dual u; if infeasible → set π = Farkas ray v.
Step 2 (depth maximisation): set
            ẑ = (π^T (b - B x_bar)) / (w^T π).
        If ẑ - z < ε  →  stop, return π with depth ẑ.
        Else z ← ẑ, go to Step 1.
```

**Mapping to OAP.** Our $A,B,b$ are split across `alpha`, `beta`, `gamma`, `delta`, `global`, `r1`, `r2`, `r3`. Setting $w$ on each row equal to the column-sum of $|B|$ on that row (the Relaxed-$\ell_1$ recipe of §3.3.2) gives a projective normalisation; for OAP this means

- $w_{\alpha,(i,j)} = 1$ (since $b_{\alpha,(i,j)}(x)=1-x_{ij}$, $|B|$-row-sum is 1),
- $w_{\beta,(i,j)} = 2$ (RHS $x_{ji}-x_{ij}$),
- $w_{\gamma,(i,j)} = 1$,
- $w_{\delta,(i,j)} = 1$,
- $w_{r3,(i,j,k,s)} = 2$,
- $w_g = w_{r1} = w_{r2} = 0$ (constant RHS — MIS recipe says $w=0$ for null rows of $B$).

This avoids the auxiliary normalisation LP altogether: each DDMA iteration is just a perturbed Farkas / Pi solve, which **reuses the existing `sub_y` / `sub_yp` Gurobi models with hot-started LP bases**. Expected payoff: 80% speedup vs. CGSP-LP per the paper, *and* the failing-instance MAX_LP_ITER issue from §2.4 plausibly disappears because DDMA produces deeper cuts with fewer iterations.

**Difference from CGSP.** CGSP solves one LP that simultaneously selects $(\pi,\pi_0)$ and depth $z$. DDMA decouples: at each iteration it produces *some* certificate (classic Benders if PSP feasible, Farkas if not) and lifts the depth iteratively. Termination criterion is $|z-\hat z|<\varepsilon$ (paper line 8 of Alg. 3). Crucially, **early termination still produces a valid cut** (Remark 6, p. 2602), so DDMA degrades gracefully under budget — unlike the current CGSP where a fail returns `(None, None, ...)`.

---

## Summary table

| # | Severity | Where | One-line fix |
|---|----------|-------|--------------|
| 2.1 | BLOCKING | `mw_mixin.py:69,79` | `_x` → `x` |
| 2.2 | CORRECTNESS | `cgsp_mixin.py:325-331` | count both adjacency hits |
| 2.3 | CORRECTNESS | `cgsp_mixin.py:514,705` | use Relaxed-$\ell_1$ weights, not unit |
| 2.4 | BLOCKING | `benchmark_benders_general.py:208-211` | guard `round` against `"-"` |
| 2.5 | CORRECTNESS | `cgsp_mixin.py:379,556` | persist CGSP model across calls |
| 2.6 | COSMETIC | `cgsp_mixin.py:790+` | move LinExpr constants to `cut_rhs` explicitly |
| 4.2 | CORRECTNESS | `mw_mixin.py:212,257` | reconstruct cut from MW's `pi_vars`, not via `get_cgsp_cut_*` |
| 4.3 | CORRECTNESS | `mw_mixin.py:188-191,236-239` | drop Pareto constraint on infeasible subproblem |
| §5 | NEW METHOD | new mixin | implement DDMA (Alg. 3, §4.1) |

Word count: ≈ 2,300.
