# Jane Street Neural Net Puzzle — Report 06

> Previous reports: `report_01.md`–`report_05.md`. This report covers experiments 24–27: trajectory coherence, scale-normalized trajectory analysis, the `true` signal diagnostic, and wide beam search with diversity forcing.

---

## Experiment 24: Trajectory Coherence Test

### The proposal

If the correct block ordering produces a "smooth" residual stream trajectory, then trajectory metrics computed over the 48 step-updates should distinguish good orderings from bad. Five metrics were computed for each ordering:

1. **Step alignment**: mean cosine similarity between consecutive deltas `cos(delta_k, delta_{k+1})`
2. **Path curvature**: total `sum ||delta_{k+1} - delta_k||^2`
3. **Backtracking**: mean projection of each step onto the cumulative displacement (positive = forward, negative = undoing work)
4. **Path length**: total `sum ||delta_k||`
5. **Displacement efficiency**: `||final - start|| / path_length` (1.0 = straight line)

### Results

Computed for SA-best (MSE=0.274), beam-10 (MSE=0.510), and 200 random shuffles of SA-best's pairings:

| Metric | SA-best | Rand mean | Rand std | SA rank | corr(MSE) |
|---|---|---|---|---|---|
| mse | 0.2723 | 1.5965 | 0.5805 | 0/200 | +1.00 |
| alignment | -0.0052 | -0.0020 | 0.0034 | 118/200 | +0.14 |
| curvature | 108.3 | 169.3 | 31.9 | 6/200 | **+0.69** |
| backtrack | +0.5449 | +0.5253 | 0.0080 | 2/200 | **+0.37** |
| path_length | 38.1 | 45.3 | 3.2 | 7/200 | **+0.69** |
| efficiency | 0.6312 | 0.5638 | 0.0140 | 0/200 | -0.22 |

Several metrics appeared to show strong signal — curvature and path length both correlated at r=0.69 with MSE, and SA-best ranked in the top 5% on curvature, path_length, and backtracking.

### Why we were suspicious

All the strong-signal metrics (curvature, path_length, backtrack) share a common property: they grow with the norm of the hidden state. If bad orderings produce norm explosion (which inflates MSE), these metrics would correlate with MSE trivially, not because they capture "trajectory quality" but because they're proxies for norm growth.

---

## Experiment 25: Scale-Normalized Trajectory Analysis

### Design

To test the norm-proxy hypothesis, we computed:

1. **Scale-free metrics**: directional curvature (`||hat_delta_{k+1} - hat_delta_k||^2` where hat = unit-normalized), relative step size (`||delta_k|| / ||h_k||`), and scale-normalized backtracking.
2. **Partial correlations**: for each metric, computed `corr(metric, MSE | final_norm)` — the correlation with MSE after controlling for final hidden state norm.

The partial correlation formula:
```
r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
```
where z = final_norm.

### Results

| Metric | Raw corr(MSE) | corr(norm) | Partial corr(MSE \| norm) |
|---|---|---|---|
| alignment | +0.14 | +0.15 | -0.12 |
| dir_curvature | -0.14 | -0.15 | +0.12 |
| mean_rel_step | +0.58 | +0.76 | +0.05 |
| total_rel_step | +0.58 | +0.76 | +0.05 |
| backtrack | +0.37 | +0.52 | -0.10 |
| rel_step_std | +0.39 | +0.56 | -0.04 |
| path_length | +0.69 | +0.81 | +0.20 |
| curvature | +0.69 | +0.70 | +0.30 |

After controlling for final norm, **every metric drops below |0.30|**, and most drop below |0.12|. The raw correlations were entirely driven by norm growth.

### Verdict

**Trajectory coherence is dead.** All apparent signal was a norm proxy. There is no independent trajectory-quality signal that can guide search. This is consistent with every prior negative result: local/pairwise metrics carry no information; only full-sequence MSE evaluation works.

---

## Experiment 26: `true` Signal Diagnostic

### The proposal

Column 49 of the historical data (`true`) is the ground truth label, with `MSE(pred, true) = 0.1065`. Several ideas were proposed for leveraging `true`:

1. **Noise-model whitening**: decompose `pred = signal + noise` using `true`, score only the signal component
2. **Pareto search**: optimize both `MSE(output, pred)` and `MSE(output, true)` simultaneously
3. **Auxiliary loss**: add `MSE(output, true)` as a regularizer during search

All of these assume that `pred - true` has structure predictable from the inputs `x`. If the residual is unpredictable noise, `true` adds no useful information.

### Test design

1. **Ridge regression**: `x → (pred - true)` with train/test split and multiple regularization strengths
2. **Quadratic features**: `x_i * x_j` features (1,224 total) to capture interaction effects
3. **Sign pattern analysis**: variance decomposition by input sign patterns (2^48 possible patterns, ~10k observed)

### Results

**Ridge regression: x → (pred - true)**

| alpha | R^2_train | R^2_test | MSE_test |
|---|---|---|---|
| 0.00 | 0.0489 | 0.0230 | 0.1043 |
| 0.01 | 0.0488 | 0.0233 | 0.1042 |
| 0.10 | 0.0476 | 0.0242 | 0.1041 |
| 1.00 | 0.0368 | 0.0251 | 0.1040 |

R^2 = 0.023 — the residual is essentially unpredictable from inputs.

**For comparison:**

| Target | R^2 (alpha=0.1) |
|---|---|
| x → pred | 0.7938 |
| x → true | 0.6869 |
| x → (pred - true) | 0.0242 |

The inputs strongly predict both `pred` and `true` individually, but the difference between them is noise.

**Quadratic features: x → (pred - true)**

| alpha | R^2 |
|---|---|
| 1.0 | 0.0130 |
| 10.0 | -0.0051 |
| 100.0 | -0.0353 |

Quadratic features don't help — overfitting at low regularization, underfitting at high.

**Sign pattern analysis:**

| Variance component | Value |
|---|---|
| Total variance of residual | 0.1068 |
| Between-pattern variance | 0.0957 |
| Within-pattern variance | 0.0067 |
| SNR (between/within) | 14.3x |
| Fraction explained by sign patterns | 0.8961 |

The residual IS structured at the sign-pattern level (SNR=14x), but sign patterns are a lookup table over ~10k unique keys. This information can't be captured by any smooth model or used to guide search, since it requires knowing the exact sign pattern, which is the input itself.

### Verdict

**`true` is useless for reconstruction.** The residual `pred - true` is unpredictable from inputs in any smooth or low-complexity way. Any method that tries to use `true` as a denoiser, whitener, or auxiliary target will inject noise, not signal. The only valid use of `true` is as a weak global sanity check: `MSE(output, true) ≈ 0.1065`.

---

## Experiment 27: Wide Beam Search with Diversity Forcing

### Motivation

Previous beam searches found different basins at different widths (beam-10: MSE=0.510, beam-200: MSE=0.410). Wider beam might find a fundamentally better basin. But naive top-K pruning risks beam collapse — all entries converge to the same prefix, wasting the wider beam.

### Design

**`solve_beam_wide.py`** — key features:

1. **Beam width 5000** (vs previous max of 200)
2. **Output-signature diversity forcing**: at each step, compute a 32-bit signature for each candidate (`sign(output - pred)` on 32 fixed samples), bucket by signature, and cap each bucket at 100 entries. This prevents beam collapse by forcing coverage of qualitatively different output patterns.
3. **Fused MSE projection**: instead of materializing the full `[B, C, N, 48]` hidden state tensor, pre-project through the last layer: `out = h_base + z @ contract_proj + contract_bias_proj`. Reduces memory from ~1.8 GB to ~40 MB per chunk. Enables beam-5000 in 10 minutes.
4. **Subsampled scoring**: score on 2000 randomly-selected samples (vs all 10000) for 5x speedup. Re-score top results on full data at the end.

### Results

Runtime: **10.2 minutes** (48 steps, ~611 seconds total).

**Step-by-step MSE trajectory (subsample scoring):**

| Step | Beam size | Best MSE | Buckets |
|---|---|---|---|
| 1 | 234 | 0.680 | 9 |
| 5 | 5,000 | 0.512 | 292 |
| 10 | 5,000 | 0.411 | 540 |
| 20 | 5,000 | 0.331 | 515 |
| 30 | 5,000 | 0.308 | 389 |
| 40 | 5,000 | 0.301 | 305 |
| 43 | 5,000 | 0.299 | 411 |
| 48 | 4,639 | 0.329 | 837 |

**Final re-scored on full 10,000 samples:**

| Rank | MSE (full) |
|---|---|
| #1 | 0.356 |
| #2 | 0.374 |
| #3 | 0.383 |

**Comparison with previous methods:**

| Method | MSE |
|---|---|
| Stochastic beam + SA | **0.274** (current best) |
| **Beam-5000 + diversity** | **0.356** |
| Beam-200 | 0.410 |
| Beam-10 | 0.510 |

### Key findings

1. **No new basin discovered.** Beam-5000 is better than beam-200 (0.356 vs 0.410) but still in the same general MSE range. The wider beam didn't break through to a fundamentally different solution space.

2. **MSE reversal in final steps.** Best subsample MSE peaked at 0.299 (step 43) then degraded to 0.329 (step 48). The beam greedily optimizes early blocks, then the remaining pieces can't complete the sequence well. This is the myopia problem that rollout scoring is designed to fix.

3. **Subsample noise.** The 2000-sample scoring gave MSE=0.329 at step 48, but full-data re-scoring gave 0.356 — a gap of 0.027. The subsample was not representative enough for the final fine-grained ranking.

4. **Early prefix convergence.** All top 10 results share the same first ~27 blocks. Diversity forcing maintained variation in later blocks but didn't prevent convergence of the critical early prefix. The 32-bit output signature was too coarse to distinguish early-stage differences.

5. **SA refinement still needed.** The raw beam output (0.356) should drop to ~0.27–0.30 with SA refinement, matching our current best. The beam is a seeding mechanism, not a solver.

---

## Updated "What Doesn't Work" List

| Method | Why it fails |
|---|---|
| Trajectory coherence (this report) | All metrics are norm-growth proxies; zero signal after controlling for final norm |
| `true` signal (this report) | R^2(x → residual) = 0.023; unpredictable noise |
| Wider beam alone (this report) | No new basin; same 0.3–0.4 range before SA |
| Adjacency gate entropy (report 05) | Residual updates too small; gate patterns dominated by accumulated state |
| Frobenius norm matching (report 04) | Expand norms too tightly clustered; zero signal |
| Frobenius/trace/spectral similarity (report 01) | 0/48 overlap with beam |
| Residual magnitude (report 03) | Uniform across all 2304 pairings |
| Minimum-residual greedy pairing (report 03) | 0/48 overlap with beam-10 |
| Greedy MSE scoring (report 02) | 3–13x worse than beam |
| Adding `true` to greedy scoring (report 02) | Adds noise, makes things worse |
| Crossover between solutions (report 02) | Basins are incompatible |

---

## Current State

| Method | MSE |
|---|---|
| **Stochastic beam + SA** | **0.274** (current best) |
| Beam-5000 + diversity (no SA) | 0.356 |
| SA from beam-10 | ~0.327 |
| Parallel tempering | 0.336 |
| Beam-200 + swaps | 0.374 |
| Correct answer | **0.000** |

---

## Where This Leaves Us

The wide beam experiment confirms that **scaling beam width alone does not solve the problem**. The search landscape has many local minima in the 0.3–0.5 range, and wider beams find slightly better ones, but none near the global optimum at 0.0.

Every "smart" scoring proxy has been eliminated:
- Pairwise metrics: dead
- Adjacency metrics: dead
- Trajectory metrics: dead (norm proxy)
- `true` signal: dead
- Wider beam: incremental, not transformative

**What remains viable:**
1. **SA refinement of beam-5000 output** — should reach ~0.27, matching or slightly improving current best
2. **Rollout scoring** in the beam's final ~10 steps — cheap to add, addresses the MSE-reversal problem
3. **Full-data scoring** (N=10000 instead of 2000) — eliminates subsample noise, ~50 min runtime
4. **Multiple diverse seeds** — run beam-5000 with different random seeds, SA-refine each, take best
5. **Fundamentally different approach** — the puzzle may require exact combinatorial methods (e.g., constraint propagation, ILP) rather than heuristic search
