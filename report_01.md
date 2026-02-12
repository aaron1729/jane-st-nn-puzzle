# Jane Street "I Dropped a Neural Net" Puzzle — Full Report

## 1. Puzzle Description

From https://huggingface.co/spaces/jane-street/droppedaneuralnet

A neural network has been disassembled into 97 numbered pieces (0–96). The goal is to find the correct permutation of all 97 pieces, validated by SHA256 hash:
```
093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4
```

### Architecture

The network is 48 residual Blocks followed by a LastLayer:

```
Input X (48-dim) → Block_1 → Block_2 → ... → Block_48 → LastLayer → scalar output
```

Each Block is a residual unit:
```
Block(h) = h + Linear_contract(ReLU(Linear_expand(h)))
```
- `Linear_expand`: 48 → 96 (doubles dimension)
- `Linear_contract`: 96 → 48 (halves dimension back)
- Skip connection: output = input + residual

LastLayer: `Linear(48 → 1)` — projects final hidden state to scalar prediction.

### Pieces

| Type | Count | Weight shape | Piece IDs |
|------|-------|-------------|-----------|
| Expand (48→96) | 48 | `[96, 48]` | 0,1,2,3,4,5,10,13,14,15,16,18,23,27,28,31,35,37,39,41,42,43,44,45,48,49,50,56,58,59,60,61,62,64,65,68,69,73,74,77,81,84,86,87,88,91,94,95 |
| Contract (96→48) | 48 | `[48, 96]` | 6,7,8,9,11,12,17,19,20,21,22,24,25,26,29,30,32,33,34,36,38,40,46,47,51,52,53,54,55,57,63,66,67,70,71,72,75,76,78,79,80,82,83,89,90,92,93,96 |
| LastLayer (48→1) | 1 | `[1, 48]` | **85** |

Each Block uses one expand piece + one contract piece. The 97-element permutation interleaves them:
```
expand_1, contract_1, expand_2, contract_2, ..., expand_48, contract_48, 85
```

### Search Space

- 48! ways to pair expand pieces with contract pieces
- 48! ways to order the 48 blocks
- Total: (48!)² ≈ 10^122 possibilities

### Data

`historical_data.csv`: 10,000 rows × 50 columns
- `measurement_0` through `measurement_47`: 48 input features
- `pred`: the correctly-assembled network's output
- `true`: ground-truth labels the network was trained to predict

For the correct permutation, `assembled_network(X) = pred` exactly (MSE = 0).

---

## 2. Data Analysis

### Input structure: bimodal / binary

Each of the 48 input features is **bimodal**, with values clustered around two modes:
- Low mode: ~-0.7
- High mode: ~+0.8

The inputs are essentially binary (sign determines the mode), with small continuous noise around each mode.

**Unique sign patterns**: 9,602 out of 10,000 data points have unique binary sign patterns (96%). Almost no grouping possible.

### Pred vs True

| Metric | Pred | True |
|--------|------|------|
| Mean | 0.021 | 0.011 |
| Std | 0.919 | 0.958 |
| Range | [-1.53, 1.58] | [-1.73, 1.81] |

| Relationship | Value |
|-------------|-------|
| Correlation(pred, true) | 0.94 |
| MSE(pred, true) | 0.1065 |
| R² | 0.884 |

### Signal-to-noise (binary pattern analysis)

| | Within-pattern variance | Between-pattern variance | SNR |
|---|---|---|---|
| **pred** | 0.0015 | 0.827 | **557×** |
| **true** | 0.0064 | 0.903 | **142×** |

**Key insight**: The network output (`pred`) is almost entirely determined by the binary sign pattern of the inputs (99.8% of variance). The ReLU activation patterns are essentially fixed per data point.

---

## 3. Experiments

### Experiment 1: Greedy search with LastLayer projection scoring

**File**: `solve.py`

**Method**: At each step, for each candidate (expand, contract) pair, compute:
```
score = MSE(W_last @ h_new + b_last, pred)
```
Pick the candidate with lowest MSE.

**Results**:
- Steps 1–35: MSE drops monotonically 0.775 → 0.401
- Steps 36–48: MSE rises to **1.986** (worse than no network at all)
- The LastLayer projection scoring is only 1D and loses information in later stages

### Experiment 2: Greedy search with Ridge regression scoring

**File**: `solve_ridge.py`

**Method**: At each step, fit ridge regression from h_new (48D) to pred. Score = training MSE.

**Results**:
- Steps 1–48: MSE stable 0.420 → 0.290, never explodes
- BUT final LastLayer MSE = **56.08** — ridge optimizes the wrong objective
- Gaps between candidates too tight for confident greedy selection

### Experiment 3: Ridge alpha sweep

**File**: `test_alphas.py`

**Result**: Alpha barely matters (0.01 to 100 → identical blocks chosen). The system is overdetermined (10k samples / 48 features), so regularization has no effect.

### Experiment 4: Beam search, width=10, CPU

**File**: `solve_beam.py`
**Runtime**: 901s (~15 min)

**Key finding**: ALL 10 beam survivors share the **same first 40 blocks**. They only diverge in the last 8 positions.

**Width-10 consensus prefix (40 blocks)**:
```python
(87,71), (31,36), (58,78), (73,72), (18,6), (49,93), (43,11), (95,33),
(81,51), (68,26), (13,75), (94,55), (5,20), (60,29), (37,40), (10,21),
(15,9), (16,54), (4,19), (28,47), (74,96), (35,12), (48,38), (2,66),
(3,53), (61,30), (0,76), (59,79), (44,70), (69,52), (64,83), (45,32),
(84,63), (41,46), (39,90), (91,34), (62,25), (56,80), (88,22), (65,8)
```

**Best MSE**: 0.510 (max absolute error 3.12)

**MSE trajectory**: Minimum around step 35 (MSE ≈ 0.393), then rises for the tail.

### Experiment 5: Brute-force the last 8 blocks

**File**: `solve_tail.py`

**Method**: Fix first 40 blocks from beam. Try all 8! = 40,320 orderings of the last 8 blocks, plus all 8! pairings.

**Results**:
- Best MSE with last 8 blocks: **0.468** (worse than the 0.410 MSE at step 40)
- Adding ANY of the last 8 blocks hurts MSE — they always make predictions worse
- The tail blocks are being paired/ordered incorrectly by the beam search

### Experiment 6: Weight matrix analysis for pairing

**File**: `find_pairings.py`

**Method**: Try to infer correct (expand, contract) pairings from weight matrix properties:
- Frobenius norm of W_contract @ W_expand
- Trace of W_contract @ W_expand
- Spectral norm
- Residual norm

**Results**: All methods give near-zero overlap with beam search pairings:
- Frobenius: 3/40 overlap
- Trace: 2/40 overlap
- ResidualNorm: 1/40 overlap

**Conclusion**: Weight matrix properties are useless for determining correct pairings.

### Experiment 7: Spectral/trace-based pairings + beam

**File**: `solve_spectral.py`

**Result**: MSE exploded to **1057**. Completely wrong approach.

### Experiment 8: Local search from beam result

**File**: `check_pairings.py`

**Method**: Starting from beam-10 result (MSE 0.510), apply:
1. Pairwise block position swaps
2. Pairwise contract piece swaps

**Results**: MSE improved from **0.510 → 0.449** over 4 iterations.

### Experiment 9: GPU simulated annealing from beam-10

**File**: `solve_gpu.py`
**Hardware**: RTX 3090 (vast.ai), ~125 iterations/sec

**Method**: SA with 4 move types (40% position swap, 30% pairing swap, 15% segment reversal, 15% block insertion). Starting from local search result (MSE 0.449).

**SA schedule**: T_start=0.1, T_end=1e-7, 200k iterations

**Results**:
| Iteration | Temperature | Best MSE |
|-----------|------------|----------|
| 40k | 0.006 | 0.445 |
| 60k | 0.002 | 0.375 |
| 80k | 0.0004 | 0.337 |
| 100k | 0.0001 | 0.328 |
| 120k | 0.00003 | **0.327** |

**Best MSE: 0.327** — our overall best result from any method.

SA converged/stuck around 0.327 after 120k iterations. Further iterations and SA restarts from random permutations did not improve.

### Experiment 10: GPU beam search, width=200

**File**: `solve_beam_gpu.py`
**Runtime**: ~349s on RTX 3090

**Key finding**: Width-200 beam finds **completely different pairings** than width-10! The first block is (48,9) vs (87,71).

**Width-200 consensus prefix (32 blocks)**:
```python
(48,9), (87,71), (58,78), (49,93), (73,72), (31,26), (81,75), (0,54),
(41,51), (39,32), (4,52), (45,33), (3,40), (2,70), (68,47), (59,92),
(61,83), (15,66), (35,22), (16,90), (91,30), (56,21), (42,46), (10,20),
(13,34), (1,12), (18,63), (28,25), (74,80), (44,7), (86,76), (69,89)
```

- All 200 beams agree on first **32** blocks (vs 40 for width-10)
- MSE minimum around step 36: **0.346**
- Final MSE: **0.410**

**This means the beam search is trapped in local optima — different beam widths find fundamentally different paths.**

### Experiment 11: Diagnostics on beam-200 result

**File**: `diagnose.py`

**Leave-one-out**: Block 45 `(62,57)` hurts MSE (delta = -0.014 when removed). Block 17 `(15,66)` slightly hurts.

**Cyclic coordinate descent**: No improvements found — at each position, only 1 available (expand, contract) pair exists since all others are in use.

**Pairwise swaps**: Improved from **0.410 → 0.374** via position and contract swaps.

### Experiment 12: Order-independence test

**File**: `test_order_independence.py`

**Sequential vs order-independent**:
- Sequential (correct) MSE: 0.510
- Independent (sum all residuals applied to raw X): MSE **4.939**

**Ordering matters enormously.** The hidden state norm grows from 0.64 to 322 (500× increase) through the 48 blocks.

All analytical pairing methods tested (Hungarian assignment, greedy sequential, feature-based matching) give **0/48 overlap** with beam search pairings.

### Experiment 13: SA + window optimization from beam-200

**Files**: `solve_from_beam200.py`, `solve_window.py`

SA from beam-200 result got stuck at 0.410 after 80k iterations. Window optimization scripts were written but not fully run.

---

## 4. Summary of Results

| Method | Starting MSE | Best MSE | Notes |
|--------|-------------|----------|-------|
| Greedy (LastLayer) | 0.775 | 1.986 | Diverges after step 35 |
| Greedy (Ridge) | 0.420 | 56.08 | Wrong objective |
| Beam-10 | - | 0.510 | Consensus on 40/48 blocks |
| Beam-10 + local search | 0.510 | 0.449 | Position + pairing swaps |
| **Beam-10 + local + SA (GPU)** | 0.449 | **0.327** | **Best overall** |
| Beam-200 (GPU) | - | 0.410 | Different pairings than beam-10! |
| Beam-200 + pairwise swaps | 0.410 | 0.374 | |
| Beam-200 + SA | 0.410 | stuck at 0.410 | |
| Weight matrix pairings | - | catastrophic | 0-3/48 overlap with beam |
| Random restarts + SA | random | > 0.327 | Never beats beam-seeded SA |

**Correct answer has MSE = 0.0000. Our best is 0.327. We are still far off.**

---

## 5. Key Observations & Lessons

### What works
1. **Beam search** gives strong initial orderings — the consensus prefixes are valuable
2. **Simulated annealing** can refine beam results (0.449 → 0.327)
3. **LastLayer MSE scoring** is the right objective (it's what we're minimizing)

### What doesn't work
1. **Weight matrix analysis** for pairing: Frobenius, trace, spectral, residual-norm all give 0-3/48 overlap with beam pairings
2. **Ridge regression scoring**: optimizes wrong objective
3. **Wider beam search**: finds different local optima, not necessarily better
4. **Order-independent approximation**: completely wrong (MSE 4.94 vs 0.51)
5. **Adding tail blocks**: always hurts MSE regardless of ordering — suggests early blocks are also wrong

### Core problems
1. **Local optima everywhere**: Width-10 and width-200 beams find completely different solutions with similar MSE (~0.4–0.5). SA gets stuck at 0.327.
2. **Greedy scoring degrades**: MSE scoring works well for first ~30–35 blocks but degrades afterward. The last 12–16 blocks always hurt MSE, suggesting errors accumulate and compound.
3. **No analytical shortcut**: Weight/feature analysis cannot determine pairings. The correct pairing depends on the data distribution, not weight matrix properties.
4. **Vast search space**: Even after beam search narrows to a single path, the remaining tail has 8!² ≈ 10^9 possibilities.

---

## 6. Unused Information & Potential New Approaches

### The `true` column (ground truth labels)

We have barely used `true` at all. Key facts:
- MSE(pred, true) = 0.1065 (the network's known accuracy)
- For the correct permutation: MSE(output, true) = 0.1065 exactly
- For wrong permutations: MSE(output, true) ≠ 0.1065

**Potential uses**:
1. **Dual scoring**: Score candidates by both MSE(output, pred) AND |MSE(output, true) − 0.1065|. These have different loss landscapes and may have different local optima.
2. **Error pattern matching**: The correct network produces specific residuals `pred − true`. An incorrectly assembled network would produce different error patterns. We could score by correlation between (output − true) and (pred − true).
3. **Regularization signal**: During SA, accept moves that keep MSE(output, true) close to 0.1065, even if MSE(output, pred) temporarily increases.

### Binary input structure

The 48 inputs are essentially binary (±1 with small noise). This means:
- The network is computing a real-valued function of 48 Boolean inputs
- ReLU activation patterns are fixed per data point (since they depend on signs)
- Each block is effectively a **fixed linear map** conditioned on the input's binary pattern
- SNR = 557× means pred is 99.8% determined by the sign pattern alone

**Potential uses**:
1. **Block fingerprinting**: For each block, compute its effective Jacobian on representative binary patterns. Blocks with similar Jacobians are interchangeable in ordering; blocks with complementary Jacobians should be adjacent.
2. **Activation pattern analysis**: For each (expand, contract) pair, compute which of the 96 ReLU neurons are active for each data point. Pairs that share activation patterns might belong together.
3. **Piecewise-linear decomposition**: The network is piecewise-linear (different linear function per binary pattern). We could decompose the target function `pred` into a sum of block contributions, each conditioned on the binary pattern.

### Backward search / meet-in-the-middle

We always search forward (left to right). But we know the last layer:
- `pred = h_48 @ w_last + b_last`
- This constrains h_48: for each data point, h_48 must lie on a specific hyperplane

**Potential approach**:
1. **Backward from output**: Given pred and the last few blocks, compute what h_45 must be, then h_44, etc.
2. **Meet in the middle**: Assemble first 24 blocks forward, last 24 backward, match hidden states at the boundary.
3. **Constraint propagation**: Each block constrains the relationship between adjacent hidden states. Use this to prune the search.

### Hybrid scoring

Current beam search uses only MSE(W_last @ h + b_last, pred). Ideas:
1. **Multi-objective beam**: Maintain diverse beam entries by scoring on multiple criteria
2. **Look-ahead**: Instead of greedy MSE at each step, score by MSE after adding 2–3 more random blocks
3. **Confidence-weighted scoring**: Weight data points by how well they're currently predicted (focus on hard examples)

### Massive parallel search

With GPU, we can evaluate ~125 full orderings/sec. Ideas:
1. **Population-based SA**: Run many SA chains in parallel with occasional migration
2. **Genetic algorithm**: Crossover between beam-10 and beam-200 consensus orderings
3. **Very wide beam** (1000+): May find yet another basin, possibly the correct one

---

## 7. Infrastructure

### Local machine
- macOS, CPU-only
- Python venv with torch, numpy, scipy

### GPU server (vast.ai)
- RTX 3090, 24GB VRAM
- CUDA 12.6, Python 3.12
- `ssh -p 50350 root@100.34.4.6`
- Files at `/root/puzzle/`
- Throughput: ~125 SA iterations/sec (bottleneck: Python loop + CUDA kernel launches)

### Files

| File | Description |
|------|------------|
| `solve.py` | Greedy search with LastLayer scoring |
| `solve_ridge.py` | Greedy search with ridge regression scoring |
| `test_alphas.py` | Ridge alpha sweep |
| `solve_beam.py` | CPU beam search, width=10 |
| `solve_tail.py` | Brute-force last 8 blocks |
| `find_pairings.py` | Weight matrix pairing analysis |
| `solve_spectral.py` | Spectral pairing + beam search |
| `check_pairings.py` | Pairing comparison + local search |
| `solve_gpu.py` | GPU simulated annealing |
| `solve_beam_gpu.py` | GPU beam search, width=200 |
| `solve_sa_fast.py` | SA with subsampling (didn't help) |
| `test_order_independence.py` | Order independence analysis |
| `solve_from_beam200.py` | SA from beam-200 result |
| `diagnose.py` | Leave-one-out + CCD + pairwise swaps |
| `solve_window.py` | Sliding window optimizer |
| `results.md` | Early experiment log |
| `beam_results.json` | Width-10 beam search full results |
