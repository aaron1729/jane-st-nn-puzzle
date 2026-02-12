# Neural Net Puzzle — Experiment Log

## Puzzle Structure
- 97 pieces: 48 Linear(48→96) "expand", 48 Linear(96→48) "contract", 1 Linear(48→1) "output" (piece 85)
- Each Block = residual: `h + contract(relu(expand(h)))`
- Network = 48 Blocks → LastLayer → scalar prediction
- Data: 10k rows, 48 input features (measurement_0..47), `pred` (model output), `true` (ground truth)
- Goal: find correct permutation of all 97 pieces (validated by SHA256 hash)

## Experiment 1: Greedy with LastLayer projection scoring

**Scoring**: At each step, for candidate block, compute `MSE(W_last @ h_new + b_last, pred)`

**Results** (solve.py):
- Steps 1–35: MSE drops monotonically 0.775 → 0.401
- Steps 36–48: MSE rises, ending at 1.986 (worse than null!)
- Gaps mostly clear early (0.001–0.01), some tight (step 6: 0.000014)
- Final LastLayer MSE: 1.986 (bad)

**Ordering found**:
```
(87,71), (49,72), (45,6), (64,11), (4,8), (61,93), (18,26), (37,47), (65,51),
(2,32), (77,33), (94,38), (3,12), (27,57), (69,20), (10,19), (88,83), (1,25),
(59,75), (86,46), (91,76), (28,52), (0,22), (68,36), (15,55), (16,80), (73,54),
(13,79), (31,40), (84,89), (43,34), (23,30), (95,9), (5,67), (73,82), (86,76),
(43,40), (50,34), (39,83), (5,92), (56,30), (44,7), (95,52), (81,90), (14,78),
(42,24), (58,70), (48,66)
```

**Diagnosis**: LastLayer projection is only 1D, good early signal but diverges in tail.

## Experiment 2: Greedy with Ridge regression scoring (alpha=1.0)

**Scoring**: At each step, fit ridge regression (alpha=1.0) from h_new (48D) to pred, score = training MSE.

**Results** (solve_ridge.py):
- Steps 1–30: steady decline 0.420 → 0.295
- Steps 31–48: plateaus ~0.290–0.298, never explodes
- Gaps much tighter throughout (many < 0.001, some < 0.0001)
- Final LastLayer MSE: 56.08 (terrible — ridge optimizes wrong objective)

**Ordering found**:
```
(60,93), (2,63), (50,17), (42,90), (81,72), (88,24), (1,33), (59,75), (74,66),
(28,20), (41,25), (39,30), (45,6), (13,89), (15,55), (61,52), (3,92), (68,80),
(87,79), (69,46), (43,54), (0,22), (48,26), (58,32), (91,67), (14,76), (27,40),
(56,36), (16,83), (77,9), (86,38), (64,21), (95,19), (5,34), (65,7), (84,70),
(10,82), (23,71), (4,12), (44,96), (62,57), (94,78), (31,29), (37,47), (18,51),
(49,11), (73,53), (35,8)
```

**Diagnosis**: Ridge uses all 48D (more info) but gaps are too tight for confident greedy.
Optimizes for general linear predictability, not the specific LastLayer direction.

## Permutation format
Each pair `(expand_id, contract_id)` becomes two entries in the permutation, followed by piece 85 (last layer).
Full permutation = 97 comma-separated piece indices.

## Experiment 3: Ridge alpha sweep (test_alphas.py)

Tested alpha = 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 on first 5 steps.

**Result**: Alpha barely matters. From 0.01 to 100, identical blocks chosen with nearly
identical gaps. At 1000 different blocks are picked but gaps don't improve.

10k samples / 48 features = so overdetermined that regularization has no effect.
The issue is the scoring objective itself, not the regularization strength.

## Experiment 4: Beam search, width=10, LastLayer scoring (solve_beam.py)

**Runtime**: 901s (~15 min). Full results in beam_results.json.

**Key finding**: ALL 10 beam survivors share the **same first 40 blocks**! They only diverge
in the last 8 blocks. The beam collapsed to a single path through step 40.

Shared prefix (40 blocks):
```
(87,71), (31,36), (58,78), (73,72), (18,6), (49,93), (43,11), (95,33),
(81,51), (68,26), (13,75), (94,55), (5,20), (60,29), (37,40), (10,21),
(15,9), (16,54), (4,19), (28,47), (74,96), (35,12), (48,38), (2,66),
(3,53), (61,30), (0,76), (59,79), (44,70), (69,52), (64,83), (45,32),
(84,63), (41,46), (39,90), (91,34), (62,25), (56,80), (88,22), (65,8)
```

Remaining 8 pieces (expand): {14, 42, 1, 50, 27, 77, 86, 23}
Remaining 8 pieces (contract): {92, 17, 7, 24, 67, 89, 82, 57}

**MSE trajectory**:
- Steps 1–35: steady decline 0.775 → 0.393 (healthy)
- Steps 36–48: MSE rises, best beam entry reaches 0.510 (still bad, but better than greedy's 1.986)

**Best beam entry**: MSE = 0.510 (max absolute error 3.12)
- Still not close to zero — the tail ordering is wrong

**Diagnosis**: The first 40 blocks are likely correct (beam fully agrees). The puzzle
reduces to finding the right ordering of the last 8 blocks — that's 8! × 8! pairings
(if pairing is unknown) or 8! = 40,320 orderings (if pairings are known).
Since there are 8 expand and 8 contract pieces left, brute-forcing all 8! = 40,320
orderings of paired blocks should be feasible.

## Ideas to try
- Brute-force the last 8 blocks: fix first 40 from beam, try all 8! orderings of last 8
  - Need to determine pairings first (8! = 40,320 pairings × 8! orderings = too much)
  - But if we fix pairings from the best beam entry, just try 8! = 40,320 orderings
  - Or try all 8!×8! with pruning
- Wider beam (20? 50?) might help the tail converge
- Combine scores: use LastLayer for direction + ridge for stability
