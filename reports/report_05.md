# Jane Street Neural Net Puzzle — Report 05

> Previous reports: `report_01.md`–`report_04.md`. This report covers experiment 23 (adjacency gate entropy test).

---

## Experiment 23: Adjacency Gate Entropy Test

### The proposal

If block p feeds into block q in the correct ordering, then q's expand preactivations on the post-p hidden state should be "healthy" — gates neither mostly dead nor mostly saturated. Wrong adjacency might cause gate collapse. Concretely, for each ordered pair (p → q):

1. Apply block p to input: `H' = H + B_p · ReLU(A_p · H + b_A^p) + b_B^p`
2. Compute q's preactivations: `U_q = H' · A_q^T + b_A^q`
3. For each of 96 units, measure activation fraction `p_j = P(U_q_j > 0)`
4. Compute mean binary entropy: `-(p log p + (1-p) log(1-p))`

High entropy = healthy gates (near 50/50 active). Low entropy = degenerate (mostly dead or always on).

This was proposed as a way to score block adjacency without relying on the final output MSE.

### Why we were skeptical

Each residual block adds `||delta|| / ||h|| ≈ 0.2–0.4` to the hidden state. So `H' ≈ H + small_correction`, and the next block's preactivations `U_q ≈ H·A_q^T + b_A^q + small_correction·A_q^T` are ~80–95% determined by the accumulated state H, not by which block p preceded it. Swapping the predecessor barely moves the preactivations.

### Test design

Used SA-best ordering (MSE=0.274) and beam-10 ordering (MSE=0.510). For each, computed the full 48×48 adjacency matrix (gate entropy of block j after block i) and checked:

1. **Do correct consecutive pairs have higher entropy than random?** Compare mean entropy of SA-best's 47 consecutive pairs against 1000 random permutations.
2. **Does greedy max-entropy Hamiltonian path match SA-best?** Find the ordering that maximizes total consecutive gate entropy and check overlap.
3. **Does signal improve at deeper contexts?** Recompute adjacency using intermediate hidden states from SA-best at depths 0, 8, 16, 24, 32, 40.

### Results

**Test 1 — SA-best vs random on raw X:**

| Ordering | Mean consecutive entropy | Rank among 1000 random |
|---|---|---|
| SA-best | 0.6481 | 332/1000 |
| Random mean | 0.6475 | — |

Difference: +0.0006. Completely indistinguishable from random.

**Beam-10 vs random:**

| Ordering | Mean consecutive entropy | Rank among 1000 random |
|---|---|---|
| Beam-10 | 0.6445 | 956/1000 |
| Random mean | 0.6469 | — |

Beam-10 is actually *worse* than random (but the difference is still tiny: 0.002).

**Test 2 — Greedy Hamiltonian path:**

| Metric | Value |
|---|---|
| Greedy path total entropy | 30.79 |
| SA-best total entropy | 30.46 |
| Positional overlap | **0/48** |
| Adjacent-pair overlap | **1/47** |

Zero signal. The max-entropy path is unrelated to SA-best.

**Test 3 — Multi-depth contexts:**

| Depth | SA-best entropy | Random entropy | Diff | SA-best rank |
|---|---|---|---|---|
| 0 | 0.6481 | 0.6475 | +0.0006 | 160/500 |
| 8 | 0.5757 | 0.5774 | −0.0017 | 430/500 |
| 16 | 0.4895 | 0.4923 | −0.0028 | 475/500 |
| 24 | 0.4090 | 0.4125 | −0.0035 | 480/500 |
| 32 | 0.3261 | 0.3308 | −0.0047 | 496/500 |
| 40 | 0.2437 | 0.2454 | −0.0016 | 398/500 |

At deeper contexts, SA-best ordering has *slightly worse* gate entropy than random, ranking in the bottom 1–5%. But the absolute differences are tiny (0.002–0.005 on a scale of ~0.3–0.6). If anything the signal goes the wrong direction.

Note that overall entropy decreases with depth (0.65 → 0.24), meaning gates progressively specialize as the hidden state evolves — expected behavior for a deep residual network.

### Why it fails

The adjacency entropy values are tightly clustered: std = 0.02 on a mean of 0.65 (coefficient of variation ≈ 3%). The residual updates are too small relative to the carrier signal. The next block's gate pattern is determined by the *accumulated* hidden state across all prior blocks, not by the immediately preceding block. Changing one predecessor changes the preactivations by ~1–3%, which is noise-level for entropy estimation.

This is the same fundamental issue that killed weight norm matching, residual magnitude analysis, and every other position-independent metric: in a residual network with small per-block updates, pairwise interactions between adjacent blocks carry negligible information compared to the global sequence context.

---

## Updated "What Doesn't Work" List

| Method | Why it fails |
|---|---|
| Adjacency gate entropy (this report) | Residual updates too small; gate patterns determined by accumulated state, not predecessor |
| Frobenius norm matching (report 04) | Expand norms too tightly clustered; zero signal |
| Frobenius/trace/spectral similarity (report 01) | 0/48 overlap with beam |
| Residual magnitude ‖B(ReLU(A(x)))‖/‖x‖ (report 03) | Uniform across all 2304 pairings |
| Minimum-residual greedy pairing (report 03) | 0/48 overlap with beam-10, MSE=32.7 |
| Greedy MSE scoring (report 02) | Greedy is 3–13× worse than beam as evaluator |
| Adding `true` to greedy scoring (report 02) | Adds noise, makes greedy worse |
| Crossover between solutions (report 02) | Basins are incompatible; MSE 2–100+ |

---

## Current State

| Method | MSE |
|---|---|
| **Stochastic beam + SA** | **0.274** (current best) |
| SA from beam-10 | ~0.327 |
| Parallel tempering | 0.336 |
| Beam-200 + swaps | 0.374 |
| Correct answer | **0.000** |

---

## Where This Leaves Us

Every attempt to find a local/pairwise signal for correct block composition has failed:
- **Pairing signals** (norms, residual magnitudes, weight similarity): zero signal
- **Adjacency signals** (gate entropy, preactivation health): zero signal
- **Scoring improvements** (dual objectives, norm penalties, `true` signal): noise or counterproductive

The consistent failure mode is the same: residual blocks make small additive updates (~20–40% of ‖h‖), so local interactions between 1–2 blocks are dominated by the accumulated global state. **The correct permutation is only identifiable through the full sequential computation**, not through any pairwise or local proxy.

This means our search must operate over complete sequences, which is what beam search + SA already does. The question remains whether we can search this space more effectively — wider beams, better SA moves, or a fundamentally different approach that doesn't decompose the problem into local decisions.
