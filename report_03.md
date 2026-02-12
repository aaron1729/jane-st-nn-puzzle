# Jane Street Neural Net Puzzle — Report 03

> Previous reports: `report_01.md`, `report_02.md`. This report covers experiments 20–21.

---

## New Best Result

**MSE 0.274** (was 0.327). From stochastic beam search (width-50, noise=0.001) + SA refinement (200k iterations). Still far from the target of 0.000.

---

## Experiment 20: Stochastic Beam Search (completed)

100 beam search runs with Gaussian noise added to scores. Configurations: width ∈ {10, 20, 50}, noise ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. Total runtime: ~1.7h (phase 1) + ~2.8h (phase 2 SA).

### Phase 1: 100 beam runs

| Config | Best MSE | Notes |
|---|---|---|
| w=50, noise=0.001 | 0.422 | Two runs tied for best |
| w=50, noise=0.005 | 0.437 | |
| w=10, noise=0.010 | 0.433 | |
| w=20, noise=0.000 | 0.447 | |

Top 10 all between 0.422–0.458. Width-50 clearly dominates. Very small noise (0.001–0.01) works best; large noise (0.1–0.2) is destructive. Zero noise isn't quite as good as tiny noise — confirming the stochastic approach adds some diversity.

**Most stable pairings across top 20 runs:**

| Pair | Frequency | Also in beam-10? | Also in beam-200? |
|---|---|---|---|
| (87, 71) | **20/20** | yes | yes |
| (49, 93) | 13/20 | yes | yes |
| (58, 78) | 13/20 | yes | yes |
| (73, 72) | 12/20 | yes | yes |
| (31, 36) | 9/20 | yes | no |
| (42, 9) | 8/20 | no | no |
| (56, 30) | 8/20 | no | no |
| (45, 6) | 7/20 | no | no |
| (43, 53) | 6/20 | no | yes |
| (27, 17) | 6/20 | yes | yes |

The 5 pairs shared between beam-10 and beam-200 also dominate here. Plus a few new consensus pairs emerge: (42,9), (56,30), (45,6), (43,53).

### Phase 2: SA refinement of top 5

| Rank | Beam MSE | SA MSE | Δ |
|---|---|---|---|
| 0 | 0.422 | **0.274** | -0.149 |
| 1 | 0.422 | 0.359 | -0.063 |
| 2 | 0.432 | 0.387 | -0.045 |
| 3 | 0.433 | **0.303** | -0.130 |
| 4 | 0.437 | 0.381 | -0.056 |

SA improvement is highly variable (0.045–0.149). The two best runs started from the two best beams but ended far apart (0.274 vs 0.359), suggesting SA is sensitive to the initial basin.

### Best ordering found (MSE = 0.274)

```
(27,76), (94,96), (50,66), (3,53), (1,21), (56,30), (18,78), (15,67),
(43,55), (84,63), (13,7), (77,25), (2,70), (44,75), (61,20), (59,29),
(58,40), (95,90), (42,36), (74,92), (91,72), (88,89), (41,22), (39,33),
(86,47), (87,8), (31,26), (14,11), (73,46), (65,17), (45,54), (35,32),
(4,52), (48,19), (23,12), (49,51), (28,24), (62,82), (60,57), (68,80),
(64,79), (16,71), (0,34), (69,83), (5,9), (10,38), (37,6), (81,93)
```

Note: SA has **scrambled the pairings** from the beam. E.g. beam had (87,71) but SA's best has (87,8). Only 2 of the original beam pairings survive SA: (3,53) and (56,30). This means SA's pairing-swap moves are doing significant work.

---

## Experiment 21: Residual Magnitude Analysis

**Question**: How big is `||B(ReLU(A(x)))|| / ||x||` — the relative magnitude of the residual update — for each possible (A, B) pairing? Can this distinguish correct from incorrect pairings?

### Position-independent analysis (on raw input X)

Computed the ratio for all 2304 possible (expand, contract) pairings on the raw input.

| Statistic | Value |
|---|---|
| Input mean L2 norm | 5.52 |
| Mean ratio across all 2304 pairings | 0.282 |
| Std | 0.083 |
| Min | 0.108 |
| Max | 0.793 |

**Beam-found pairings are NOT distinguished by this metric:**

| Source | Mean ratio |
|---|---|
| Beam-10 pairings | 0.283 |
| Beam-200 pairings | 0.285 |
| Random pairings | 0.282 |
| Shared (5 pairs) | **0.444** |

The beam-found pairings have the *same* mean ratio as random pairings. The 5 shared/"confident" pairs actually have *higher* ratios — they are the **least** identity-like blocks, with (49,93) at 0.78 being nearly at the matrix maximum.

**Rank of beam-chosen contract within each expand piece's ratio distribution:**

| | Beam-10 | Beam-200 |
|---|---|---|
| Mean rank (of 48) | 23.3 | 24.0 |
| Median rank | 25 | 22 |
| In top 5 | 6/48 | 4/48 |
| In top 10 | 11/48 | 8/48 |

Essentially random (expected mean = 23.5). The beam's contract choices are not the ones minimizing the residual ratio.

**Greedy minimum-ratio pairing (assign each expand its smallest-ratio contract):**
- Overlap with beam-10: **0/48**
- Overlap with beam-200: **1/48**
- MSE when evaluated: **32.7** (terrible)

### Sequential analysis (through the network)

The per-position ratio `||delta||/||h||` as blocks are applied sequentially tells a different story:

**Beam-10:**
| Position | Ratio | ||h|| | ||delta|| |
|---|---|---|---|
| 0 | 0.28 | 5.5 | 1.5 |
| 5 | 0.42 | 6.6 | 2.8 |
| 10 | 0.28 | 9.1 | 2.6 |
| 20 | 0.13 | 25.8 | 3.3 |
| 30 | 0.26 | 42.2 | 10.9 |
| 40 | 0.13 | 78.7 | 10.5 |
| 47 | 0.20 | 117.3 | 23.6 |

**Beam-200:**
| Position | Ratio | ||h|| | ||delta|| |
|---|---|---|---|
| 0 | 0.23 | 5.5 | 1.3 |
| 5 | 0.36 | 6.5 | 2.3 |
| 10 | 0.38 | 10.4 | 4.0 |
| 20 | 0.18 | 17.2 | 3.1 |
| 30 | 0.18 | 24.3 | 4.5 |
| 40 | 0.27 | 36.4 | 9.9 |
| 47 | 0.24 | 53.9 | 13.3 |

Key observations:
- **||h|| grows monotonically** through the network (5.5 → 117 for beam-10, 5.5 → 54 for beam-200)
- **Beam-200 has 2× lower final norm** (54 vs 117), consistent with its better MSE
- The ratio varies between 0.08–0.58 depending on position, with no clear trend
- **||delta|| also grows**, tracking ||h|| growth — each block's update scales with the current state magnitude
- A few blocks have notably large ratios (pos 8 beam-10: 0.56, pos 11: 0.58) — these are "high-impact" blocks that disproportionately alter the hidden state

### What this tells us

1. **Pairing is NOT about matching expand/contract by residual magnitude.** The correct pairs don't minimize (or maximize) the update ratio. Any expand works with any contract to produce a similar-magnitude output.

2. **The ratio varies mostly by expand piece, not by contract.** The active fraction of the ReLU is determined entirely by the expand piece (since it only depends on `A(x)`). The contract piece just linearly transforms the ReLU output.

3. **Some blocks are structurally "large" and some are "small."** Piece 43 has ratio 0.11–0.28 (smallest row), while piece 31 has 0.26–0.79 (largest). These seem to be intrinsic properties of the expand piece's weight magnitudes.

4. **The hidden state norm diverges.** 48 residual additions with ratios ~0.2–0.4 cause the norm to grow ~10–20×. This is expected (each step adds ~0.3||h|| so after 48 steps: ||h|| ≈ 1.3^48 ≈ 300× original — we see 10–20× because many updates partially cancel). The correct solution likely has even more cancellation.

---

## Updated Results Summary

| Method | MSE | Notes |
|---|---|---|
| Greedy (LastLayer) | 1.986 | Baseline |
| Beam-10 | 0.510 | |
| Beam-200 | 0.410 | |
| Beam-200 + swaps | 0.374 | |
| Parallel tempering (8 chains) | 0.336 | |
| SA from beam-10 (prior session) | ~0.327 | Ordering lost |
| **Stochastic beam + SA (new)** | **0.274** | **Current best** |
| Stochastic beam (SA-3) | 0.303 | Second best |

**Correct answer: MSE = 0.000. Our best: 0.274. Gap: still enormous.**

---

## Pairing Consensus (across all methods)

Combining evidence from beam-10, beam-200, stochastic beam top-20, and SA refinement:

| Tier | Pairs | Evidence |
|---|---|---|
| Very high confidence | (87,71) | 20/20 stochastic, both beams, survived SA |
| High confidence | (49,93), (58,78), (73,72) | 12-13/20 stochastic, both beams |
| Medium confidence | (31,36), (43,53), (27,17) | 6-9/20 stochastic, at least one beam |
| New candidates | (42,9), (56,30), (45,6) | 7-8/20 stochastic, neither original beam |

But SA scrambles pairings aggressively — only 2/48 of SA-0's pairings match the initial beam. This means either (a) beam pairings are wrong and SA finds better ones, or (b) SA is overfit to a local minimum with wrong pairings but a lucky ordering.

---

## What We've Learned That Narrows the Search

1. **Residual magnitude is not a useful pairing signal.** The ratio `||B(ReLU(A(x)))||/||x||` is essentially uniform across all 2304 pairings. Weight-based pairing (Frobenius, trace, spectral — report 01) also failed. Greedy-MSE pairing fails. We have **no working method to independently determine correct pairings**.

2. **SA pairing swaps are powerful but undirected.** SA-0 improved from 0.422 to 0.274 by swapping pairings, but the resulting pairings share little with the input. We have no way to know if these SA pairings are closer to the truth or just another local minimum.

3. **Wider beam helps but has diminishing returns.** Width-50 is clearly better than 10 or 20, but all width-50 runs converge to similar MSE (0.42–0.46). Going to width-500 or 5000 might find a different basin.

4. **The 0.274 barrier.** Three independent SA runs (prior session ~0.327, PT 0.336, stochastic-0 0.274) all plateau around 0.27–0.34. This likely represents the depth of the local minima basin(s) reachable by our SA moves.

---

## Open Questions for Discussion

1. **Are we exploring enough of pairing space?** SA's pairing-swap moves change one pair at a time. Maybe we need moves that change many pairings simultaneously (e.g., random 10-pair shuffles with high temperature).

2. **Should we try width-5000 beam?** It fits in GPU memory (~10GB). If beam-200 found different pairings than beam-10, maybe beam-5000 finds the correct ones. Runtime ~30min.

3. **Is there a closed-form or spectral approach to pairing?** We've tried weight norms and residual magnitudes — neither works. What about the *directions* of the weight matrices? E.g., do correct (A,B) pairs have aligned singular vectors?

4. **Can we use `true` more effectively?** We know MSE(output, true) should be 0.1065 for the correct solution. Our best gives ~0.35–0.50. Could we use this as a **hard constraint** in SA (only accept moves that bring MSE(true) closer to 0.1065)?

5. **Backward search?** Building from the last layer backward. The last block should make the smallest refinement (closest to identity). Could we identify which block goes last by finding the pair with the smallest contribution to the final output?

6. **Are there structural constraints we're missing?** E.g., does the network have batch normalization-like properties? Is there a specific input (like all-zeros or all-ones) that reveals the correct ordering?
