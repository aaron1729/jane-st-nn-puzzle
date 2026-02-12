# Jane Street Neural Net Puzzle — Report 02

> Previous report: `report_01.md`. This report covers new experiments and findings since then.

## Quick Recap

- **Puzzle**: Reassemble 97 neural net pieces (48 expand, 48 contract, 1 last-layer) into the correct permutation, validated by SHA256 hash.
- **Architecture**: 48 residual blocks (`h + contract(relu(expand(h)))`) → `Linear(48→1)` last layer.
- **Data**: 10k rows × 50 cols: 48 binary-ish inputs, `pred` (correct network output), `true` (ground truth).
- **Best MSE so far**: **0.327** (SA from beam-10 + local search). Correct answer is MSE = **0.000**.
- **Key prior findings**: Beam-10 and beam-200 find completely different solutions. Weight matrix analysis for pairing is useless (0/48 overlap). Ordering matters enormously (independent approx MSE = 4.94 vs sequential 0.51).

---

## New Data Analysis

### Pred vs True statistics

| Metric | Value |
|--------|-------|
| MSE(pred, true) | **0.1065** |
| Correlation(pred, true) | 0.94 |
| R² | 0.884 |
| Pred std | 0.919 |
| True std | 0.958 |

### Binary input structure

| Metric | Value |
|--------|-------|
| Unique sign patterns in 10k data | **9,602** (96% unique) |
| Pred within-pattern variance | 0.0015 |
| Pred between-pattern variance | 0.827 |
| **Pred SNR** | **557×** |
| True within-pattern variance | 0.0064 |
| True between-pattern variance | 0.903 |
| **True SNR** | **142×** |

Almost every data point has a unique binary sign pattern — no grouping possible. But the network output is 99.8% determined by the sign pattern (SNR = 557×). The continuous noise barely matters for `pred`.

---

## New Experiments (This Session)

### Experiment 14: Crossover between beam-10 and beam-200

**The most important experiment.** Tested what happens when you mix prefixes from the two beam solutions.

**Crossover at position k** (take first k blocks from one, fill rest from other):

| k | beam10[:k]+beam200 MSE | beam200[:k]+beam10 MSE |
|---|---|---|
| 0 | 0.410 | 0.510 |
| 4 | 2.06 | 8.54 |
| 8 | 0.93 | 76.56 |
| 12 | 17.45 | 94.77 |
| 16 | 7.22 | 45.59 |
| 20 | 1.72 | 104.27 |
| 24 | 21.14 | 28.29 |
| 32 | 72.77 | 40.35 |
| 40 | 3.03 | 5.26 |
| 48 | 0.510 | 0.410 |

**Result: Catastrophic.** Mixing ANY prefix gives MSE 1–100+. The two solutions are in **completely incompatible basins**. This is not like TSP where two good tours share most edges — these are fundamentally different internal representations that cannot be spliced.

**Pairing agreement**: Only **5/48 pairs** are shared between beam-10 and beam-200:
- `(49, 93)`, `(58, 78)`, `(60, 29)`, `(73, 72)`, `(87, 71)`

These 5 appear in similar positions (within ±2 of each other). The other **43 pairs are completely different**.

### Experiment 15: Using `true` as auxiliary scoring

Tested several uses of `true`:

**MSE(output, true) for current solutions:**

| Solution | MSE(pred) | MSE(true) | |MSE(true) - target| | corr(residuals) |
|---|---|---|---|---|
| Beam-10 | 0.510 | 0.585 | 0.478 | 0.365 |
| Beam-200 | 0.410 | 0.482 | 0.376 | 0.394 |

For the correct solution: MSE(true) should be exactly 0.1065 and corr(residuals) should be 1.0. We're nowhere close — residual correlation is only 0.36–0.39.

**Dual-scored greedy (MSE(pred) + α * |MSE(true) - target|):**

| α | Final MSE |
|---|---|
| 0.0 | 1.986 |
| 0.05 | 6.881 |
| 0.1 | similar |
| 0.2 | similar |

**Result: Adding `true` penalty to greedy scoring makes things WORSE.** The true signal adds noise at each step and the greedy approach amplifies errors.

### Experiment 16: Norm-aware greedy

Tested penalizing hidden-state norm growth during greedy search:

| β (norm penalty) | Final MSE | Final h_norm |
|---|---|---|
| 0.0 | 1.986 | 55.0 |
| 0.001 | 6.176 | 36.7 |
| 0.01 | 0.864 | 31.3 |
| 0.1 | 2.066 | 7.5 |

**Result: Norm penalty keeps norms low but doesn't improve MSE.** The β=0.01 result (0.864) is better than unpenalized (1.986) but still bad. The problem isn't norm explosion per se — it's that greedy picks wrong blocks.

For reference, hidden state norms of beam-found solutions:
- Beam-10 + local search: h_norm = **367**
- Beam-200 + swaps: h_norm = **81.5**

The beam-200 solution (lower MSE) also has much more stable norms.

### Experiment 17: Parallel tempering SA

8 chains at different temperatures (0.5 down to 1e-7), with periodic state swaps between adjacent chains. Chains 0–3 used auxiliary `true`-based scoring; chains 4–7 used pure MSE(pred).

Starting from beam-200 + swaps (MSE 0.374) and beam-10 + local (MSE 0.449), plus 2 random-start chains.

**Results over 120k iterations:**

| Iteration | Global best MSE | Cold chain MSE | Hot chain MSE |
|---|---|---|---|
| 5k | 0.365 | 0.365 | 1.054 |
| 10k | 0.362 | 0.362 | 1.639 |
| 15k | 0.362 | 0.362 | 0.995 |
| 100k | 0.336 | 0.336 | 0.428 |
| 120k | **0.336** | 0.336 | 0.398 |

**Result: 0.374 → 0.336.** Modest improvement. The temperature schedule cooled too aggressively (300k iters, 0.5→1e-7), so chains froze early. Swap acceptance was low (47/1680 = 2.8%). The random-start chains never found anything competitive.

### Experiment 18: Pairing SA (SA over pairings, greedy ordering)

Tried separating pairing search from ordering search: SA explores pairings by swapping contract pieces, and for each trial pairing, greedy ordering finds the best sequence.

**Critical finding**: Greedy ordering is a TERRIBLE proxy for beam ordering:

| Pairing set | Greedy order MSE | Beam order MSE |
|---|---|---|
| Beam-10 pairings | **5.36** | 0.510 |
| Beam-200 pairings | **1.47** | 0.410 |

Greedy is 3-13× worse than beam for the same pairings. The beam search ordering is doing a LOT of work that greedy can't replicate.

SA over pairings (with greedy evaluator) got stuck at 0.556 — better than beam-200's greedy (1.47) but worse than beam-200's beam ordering (0.41). **The evaluator is too noisy to guide pairing search.**

### Experiment 19: Stochastic beam search (in progress)

Running 100 beam searches with Gaussian noise added to scores. Varies beam width (10/20/50) and noise level (0.001–0.2).

**Results so far (25/100 runs complete):**

| Runs | Best MSE | Config |
|---|---|---|
| 5 | 0.489 | w=10 |
| 10 | 0.447 | w=20 |
| 20 | 0.422 | w=50, noise=0.01 |
| 25 | 0.422 | (same) |

SA refinement phase will follow. Based on prior experience, SA should bring the best result down to ~0.33–0.35.

---

## Updated Results Summary

| Method | MSE | Notes |
|--------|-----|-------|
| Greedy (LastLayer) | 1.986 | Diverges after step 35 |
| Beam-10 (CPU) | 0.510 | 40-block consensus |
| Beam-200 (GPU) | 0.410 | 32-block consensus, different pairings |
| Beam-10 + local search | 0.449 | |
| Beam-200 + pairwise swaps | 0.374 | |
| Parallel tempering (8 chains) | **0.336** | From 0.374, modest improvement |
| **SA from beam-10 (prior session)** | **~0.327** | **Overall best (ordering lost)** |
| Stochastic beam (in progress) | 0.422 | Before SA refinement |
| Pairing SA + greedy | 0.556 | Greedy is a bad evaluator |
| Dual-scored greedy (+true) | 6.88 | Worse than baseline |
| Norm-aware greedy | 0.864 | Marginal |
| Crossover beam-10/200 | 2–100 | Catastrophic |

**Correct answer: MSE = 0.000. Our best: ~0.327. Gap: enormous.**

---

## Key Insights from This Round

### 1. We are in the wrong basin, not close to the right answer

The crossover catastrophe proves our beam solutions are **self-consistent local optima**, not approximations to the correct answer. The two best solutions share only 5/48 pairings and cannot be mixed at all. We're not "95% right and need to fix the tail" — we're in a fundamentally wrong part of the search space.

### 2. Greedy scoring is the root failure

Every greedy approach fails the same way: MSE drops until ~step 30–36, then rises. This happens regardless of scoring variant (MSE, ridge, dual, norm-penalized). The scoring function is **deceptive** — blocks that look good at step k are not the blocks that lead to correct assembly at step 48.

This explains why beam-10 and beam-200 find different solutions: the greedy landscape has many attractors, and wider beams fall into different ones.

### 3. `true` doesn't help with greedy/local scoring

MSE(output, true) = 0.48–0.58 for our solutions vs target 0.1065. Residual correlation only 0.36–0.39 vs target 1.0. But adding `true` as a penalty in greedy makes things WORSE — it adds noise to an already noisy scoring signal. The `true` signal might help in a global search (like filtering SA results) but not in stepwise scoring.

### 4. Pairings are the key unknown

The 5 shared pairs (stable across beam widths) are probably correct. The other 43 are up for grabs. Weight matrix analysis gives 0/48 overlap. Greedy ordering is too noisy to evaluate pairings. We need a way to score pairings that doesn't depend on greedy ordering.

### 5. SA can refine but not escape basins

SA consistently takes beam results from ~0.4–0.5 down to ~0.33–0.35. But it never finds anything fundamentally better. Random restarts from shuffled orderings don't beat beam-seeded SA. The basins are deep and the correct solution is far away in permutation space.

---

## What Hasn't Worked

| Approach | Why it failed |
|---|---|
| Weight matrix pairing (Frobenius, trace, spectral) | 0/48 overlap with beam — weights don't reveal pairings |
| Ridge regression scoring | Optimizes wrong objective |
| Adding `true` to greedy scoring | Adds noise, makes greedy worse |
| Norm penalty in greedy | Diverts from useful blocks |
| Crossover between solutions | Basins are incompatible |
| Pairing SA + greedy ordering | Greedy is 3-13× worse than beam as pairing evaluator |
| Parallel tempering | Temps froze too fast; modest improvement only |
| Order-independent approximation | MSE 4.94 vs sequential 0.51 |

---

## What Might Still Work (Proposals)

### Tier 1: Most promising

**A. Pairing SA with beam-search evaluator.** The pairing SA idea is sound — the evaluator was the problem. If we use width-10 beam search (~20s) instead of greedy (~0.7s) to evaluate each pairing, the signal is much better. 3000 iterations × 20s = 17 hours. Could be made faster with width-5 beam.

**B. Much wider beam (2000–5000).** Beam-200 found different pairings than beam-10. Beam-5000 might find yet another basin, possibly closer to the correct one. GPU memory: 5000 × 10000 × 48 × 4 bytes ≈ 9.6 GB (fits in 24GB RTX 3090). Runtime ~30 min.

**C. Diverse beam search.** Instead of keeping top-K by MSE, keep K entries that are maximally diverse (different pairings/orderings). This explicitly prevents beam collapse into a single basin.

### Tier 2: Worth trying

**D. Backward beam search.** Start from the last layer and build the network from right to left. The last block's contribution should be small (it's the final refinement), so scoring from the end might be less deceptive. Could find a different set of "confident" blocks.

**E. Block-compatibility graph.** For each pair of blocks (A, B), compute how well B performs after A (for all data). Build a compatibility graph and find high-weight Hamiltonian paths. This is TSP-like but the scores depend on the full prefix, not just the immediate predecessor.

**F. Iterative refinement with learned value function.** Use beam search data (hidden states at each step + final MSE) to train a value predictor. Then re-run beam search using the value predictor instead of raw MSE. This addresses the deceptive scoring directly.

### Tier 3: Long shots

**G. Constraint propagation from `true`.** For the correct permutation, MSE(output, true) = 0.1065 and corr(output - true, pred - true) = 1.0. These are strong global constraints. Could use them to filter/rank SA results or as an acceptance criterion in a population-based search.

**H. Meet in the middle.** Forward-assemble first 24 blocks, backward-assemble last 24. Match at the boundary. Requires figuring out backward assembly (hard due to ReLU non-invertibility).

**I. Exhaustive search over pairing subsets.** Fix the 5 "trusted" pairs. For the remaining 43, try random pairings + beam ordering. 1000 trials × 20s = 5.5 hours.

---

## Hardware

- **GPU**: RTX 3090 (24GB VRAM), vast.ai, `ssh -p 50350 root@100.34.4.6`
- **Throughput**: ~125 full-pipeline evaluations/sec (SA), beam-200 in ~6 min, beam-10 in ~20s
- **Currently running**: Stochastic beam search (100 runs, ~1h remaining)

## Files (new this session)

| File | Description |
|------|------------|
| `crossover.py` | Crossover experiment + dual/norm greedy |
| `solve_pt.py` | Parallel tempering SA (8 chains) |
| `solve_pairing_sa.py` | SA over pairings with greedy evaluator |
| `solve_stochastic_beam.py` | Stochastic beam search (many noise levels) |
| `report_01.md` | Previous full report |
| `report_02.md` | This report |
