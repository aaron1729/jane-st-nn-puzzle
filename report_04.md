# Jane Street Neural Net Puzzle — Report 04

> Previous reports: `report_01.md`–`report_03.md`. This report covers experiment 22 (weight norm analysis) and theoretical analysis of rescaling symmetry.

---

## Experiment 22: Weight Norm Analysis & Rescaling Symmetry

### The rescaling symmetry argument

Each residual block computes:
```
h_new = h + B_w · ReLU(A_w · h + b_A) + b_B
```

There is a continuous symmetry: for any α > 0, rescaling `A_w → αA_w, b_A → αb_A, B_w → (1/α)B_w` leaves the output **exactly invariant** (since `ReLU(αx) = α·ReLU(x)` for α > 0). The contract bias `b_B` is not part of this symmetry — it stays fixed.

**Prediction under L2 weight penalty**: If only weights (not biases) are L2-penalized, the regularization contribution from a block is:
```
λ(α²||A_w||² + (1/α²)||B_w||²)
```
Since the task loss is invariant to α, training minimizes this over α. Setting the derivative to zero gives `α⁴ = ||B_w||²/||A_w||²`, and at this optimum both terms equal `||A_w||·||B_w||`. This means **the stored norms should satisfy `||A_w||_F = ||B_w||_F` for correctly paired (A, B)**.

This would be a powerful pairing signal: just match expand and contract pieces by Frobenius norm.

### Empirical results

**Expand weight norms** (96×48 matrices):
- Range: 6.11 – 7.16
- Mean: 6.50, Std: 0.30

**Contract weight norms** (48×96 matrices):
- Range: 3.35 – 6.29
- Mean: 4.20, Std: 0.60

**Expand bias norms** (96-dim): 0.85 – 1.59
**Contract bias norms** (48-dim): 0.15 – 0.36

The prediction `||A_w|| = ||B_w||` **fails completely**. Expand norms are systematically ~50% larger than contract norms, with almost no overlap between the two distributions. The closest possible match is expand piece 43 (norm 6.11) with contract piece 51 (norm 6.29).

### Norm matching has zero signal

| Pairing method | Mean |log(||A||/||B||)| |
|---|---|
| Beam-10 | 0.4452 |
| Beam-200 | 0.4452 |
| SA-best (MSE=0.274) | 0.4452 |
| Random | 0.4453 |
| Hungarian optimal | 0.4452 |

All methods produce **identical** mean log-ratios (to 4 decimal places). This is because the expand norms are so tightly clustered (std=0.30) that permuting the contracts barely changes anything. The Hungarian optimal norm-matching has:
- Overlap with beam-10: 2/48
- Overlap with beam-200: 0/48
- Overlap with SA-best: 0/48

### Why the prediction fails

Three possible explanations:
1. **No weight decay was used** during training — the rescaling is unconstrained
2. **Training didn't converge** along the symmetry direction — SGD's implicit norm balancing is slow
3. **AdamW-style decay** was used, which applies `w -= lr·wd·w` to stored parameters directly, rather than adding `wd·w` to the gradient. AdamW doesn't respect the rescaling symmetry the same way as true L2 regularization

Regardless of the reason, the empirical conclusion is clear: **weight norms carry no pairing information**.

### Notable outlier pieces

A few contract pieces have anomalously large norms:

| Contract piece | ||W||_F | Notes |
|---|---|---|
| 51 | 6.29 | Closest to expand range |
| 72 | 5.87 | |
| 55 | 5.73 | |
| 78 | 5.38 | |
| 93 | 4.90 | |

These 5 pieces appear in many of our "confident" pairings:
- (73, **72**) — shared across beam-10/200/stochastic
- (49, **93**) — shared across beam-10/200/stochastic
- (58, **78**) — shared across beam-10/200/stochastic
- (81, **51**) — beam-10 and beam-200
- (94, **55**) — beam-10 only

This is interesting but likely coincidental — these are simply the contract pieces with the most "capacity" (largest weight norms), so beam search gravitates toward them early because they produce the largest output changes.

---

## Updated "What Doesn't Work for Pairing" List

| Method | Why it fails |
|---|---|
| Frobenius norm matching (||A|| vs ||B||) | Expand norms too tightly clustered; zero signal |
| Frobenius/trace/spectral similarity (report 01) | 0/48 overlap with beam |
| Residual magnitude ||B(ReLU(A(x)))||/||x|| (report 03) | Uniform across all 2304 pairings; beam pairings indistinguishable from random |
| Minimum-residual greedy pairing (report 03) | 0/48 overlap with beam-10, MSE=32.7 |
| Greedy MSE scoring of pairings (report 02) | Greedy is 3-13× worse than beam as pairing evaluator |

**We have exhausted simple weight-based and data-based pairing signals.** Every approach that tries to determine correct (A,B) pairs independently — without considering ordering and sequential interaction — has produced zero signal.

---

## Current State

| Method | MSE |
|---|---|
| **Stochastic beam + SA** | **0.274** (current best) |
| SA from beam-10 | ~0.327 (ordering lost) |
| Parallel tempering | 0.336 |
| Beam-200 + swaps | 0.374 |
| Correct answer | **0.000** |

---

## What This Suggests

The failure of ALL position-independent pairing methods (norms, residual magnitudes, weight similarity) points to a key insight: **correct pairings may only be identifiable in context** — i.e., which (A,B) pair works at position k depends on what blocks came before it. The pairing and ordering problems are **coupled**, not separable.

This means approaches that separate pairing from ordering (like pairing SA with greedy ordering evaluator, or any independent pairing criterion) are fundamentally limited. The search must operate over the joint space of pairings × orderings, which is what beam search and SA already do.

The question is whether we can search this joint space more effectively. Our best MSE (0.274) comes from beam search + SA, suggesting we need either:
- **Much wider beam** (5000+) to explore more basins
- **Better SA moves** that make coordinated changes to pairing + ordering
- **A fundamentally different approach** that doesn't rely on greedy/sequential scoring
