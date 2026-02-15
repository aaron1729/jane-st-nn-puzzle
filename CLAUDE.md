# Jane Street "I Dropped a Neural Net" Puzzle

## Problem

Reassemble 97 neural network pieces into the correct permutation, validated by SHA256 hash `093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4`.

### Architecture
- **48 residual blocks**: each computes `h_new = h + B(ReLU(A(h)))` where A is Linear(48→96) ("expand") and B is Linear(96→48) ("contract")
- **1 last layer**: piece 85, Linear(48→1)
- **Permutation format**: `expand_1,contract_1,expand_2,contract_2,...,expand_48,contract_48,85`

### Pieces
- 97 total in `data/pieces/piece_{0..96}.pth` (each has `weight` and `bias`)
- 48 expand pieces: weight shape (96, 48) — IDs: 0–5, 10, 13–16, 18, 23, 27–28, 31, 35, 37, 39, 41–45, 48–50, 56, 58–62, 64–65, 68–69, 73–74, 77, 81, 84, 86–88, 91, 94–95
- 48 contract pieces: weight shape (48, 96) — IDs: 6–9, 11–12, 17, 19–22, 24–26, 29–30, 32–34, 36, 38, 40, 46–47, 51–55, 57, 63, 66–67, 70–72, 75–76, 78–80, 82–83, 89–90, 92–93, 96
- 1 last layer: piece 85, weight shape (1, 48)

### Data
- `data/historical_data.csv`: 10k rows × 50 cols
- Columns 0–47: binary-ish input features (mostly ±1 with small continuous noise)
- Column 48 (`pred`): correct network output (what the assembled network should produce)
- Column 49 (`true`): ground truth labels
- MSE(pred, true) = 0.1065, so `pred` is NOT the same as `true`

## GPU Server

- **RTX 3090 (24GB VRAM)** on vast.ai
- `ssh -p 50350 root@100.34.4.6`
- Files at `/root/puzzle/`
- Use `python3` (not `python`)
- Throughput: ~125 full-pipeline evals/sec (SA), beam-200 in ~6 min, beam-10 in ~20s
- Scripts must be uploaded via `scp -P 50350 file root@100.34.4.6:/root/puzzle/`

## Current Best Result

**MSE = 0.274** (correct answer is 0.000)

Best ordering (from stochastic beam + SA):
```
(27,76), (94,96), (50,66), (3,53), (1,21), (56,30), (18,78), (15,67),
(43,55), (84,63), (13,7), (77,25), (2,70), (44,75), (61,20), (59,29),
(58,40), (95,90), (42,36), (74,92), (91,72), (88,89), (41,22), (39,33),
(86,47), (87,8), (31,26), (14,11), (73,46), (65,17), (45,54), (35,32),
(4,52), (48,19), (23,12), (49,51), (28,24), (62,82), (60,57), (68,80),
(64,79), (16,71), (0,34), (69,83), (5,9), (10,38), (37,6), (81,93)
```

Saved in `results/stochastic_beam_result.json` (local) and `/root/puzzle/stochastic_beam_result.json` (GPU).

## Pairing Consensus

Most stable pairs across beam-10, beam-200, and 20 stochastic beam runs:
- **(87, 71)**: 20/20 stochastic runs, both beams — very high confidence
- **(49, 93), (58, 78), (73, 72)**: 12–13/20 stochastic, both beams — high confidence
- **(60, 29)**: both beams (shared pair)
- **(31, 36), (43, 53), (27, 17)**: 6–9/20 stochastic, at least one beam

Note: SA scrambles pairings aggressively — only 2/48 of SA-best pairings match the initial beam.

## What Doesn't Work (Exhaustively Tested)

### Pairing signals (all dead)
- Frobenius/trace/spectral norm matching: 0/48 overlap with beam
- Weight norm matching via rescaling symmetry (||A||=||B|| prediction): expand norms too clustered (6.5±0.3 vs 4.2±0.6)
- Residual magnitude ||B(ReLU(A(x)))||/||x||: uniform across all 2304 pairings
- Minimum-residual greedy pairing: 0/48 overlap, MSE=32.7

### Adjacency signals (dead)
- Gate entropy of next block's preactivations: SA-best ranks 332/1000 among random (no signal)
- Multi-depth context doesn't help (signal goes wrong direction at depth)

### Scoring improvements (counterproductive or marginal)
- Adding `true` to greedy scoring: makes things worse
- Norm-penalized greedy: marginal improvement (0.864 vs 1.986)
- Greedy ordering is 3–13× worse than beam for same pairings
- Crossover between beam-10 and beam-200: catastrophic (MSE 2–100+)

### Key insight
Pairing and ordering are **coupled, not separable**. Correct pairings can only be identified in context of the full sequence. Local/pairwise signals carry zero information.

## What Has Worked

| Method | MSE |
|---|---|
| Stochastic beam (w=50, noise=0.001) + SA | **0.274** |
| SA from beam-10 | ~0.327 |
| Parallel tempering (8 chains) | 0.336 |
| Beam-200 + pairwise swaps | 0.374 |
| Beam-200 | 0.410 |
| Beam-10 | 0.510 |

Beam search with MSE scoring is the only approach that produces reasonable solutions. SA refinement consistently shaves 0.05–0.15 off beam results.

## Also Doesn't Work (Experiments 24–27)

- **Trajectory coherence** (exp 24–25): step alignment, curvature, backtracking, path length — all metrics are norm-growth proxies. Zero signal after controlling for final norm via partial correlation.
- **`true` signal** (exp 26): R²(x → pred-true) = 0.023. Residual is unpredictable noise. Useless for denoising/whitening.
- **Wide beam (beam-5000)** (exp 27): MSE=0.356 raw, no new basin. Same 0.3–0.4 range. MSE reversal in final steps (myopia). All top-10 share first ~27 blocks despite diversity forcing.
- **Sinkhorn continuous relaxation** (Codex overnight, 1260 runs): soft MSE ~0.3–0.4 but discrete rounding gives 0.616+ at best. Massive relaxation gap — blended weights don't approximate any real discrete network.

## Reports

Detailed experiment logs in `reports/`:
- `report_01.md`: Initial experiments (1–13), beam search, weight analysis
- `report_02.md`: Experiments 14–19, crossover, parallel tempering, stochastic beam
- `report_03.md`: Experiments 20–21, stochastic beam results, residual magnitudes
- `report_04.md`: Experiment 22, weight norm analysis, rescaling symmetry
- `report_05.md`: Experiment 23, adjacency gate entropy test
- `report_06.md`: Experiments 24–27, trajectory, true signal, wide beam, diversity

## Project Layout

```
├── CLAUDE.md                  # This file
├── data/                      # Input data and pieces
│   ├── historical_data.csv
│   └── pieces/piece_{0..96}.pth
├── scripts/                   # All solver and analysis scripts
│   ├── solve_beam_gpu.py      # GPU beam search (configurable width)
│   ├── solve_beam_wide.py     # Beam-5000 with diversity forcing
│   ├── solve_stochastic_beam.py # Stochastic beam + SA refinement
│   ├── solve_sinkhorn.py      # Sinkhorn continuous relaxation
│   ├── solve_pt.py            # Parallel tempering SA
│   ├── test_trajectory*.py    # Trajectory coherence tests
│   ├── test_true_signal.py    # True signal diagnostic
│   └── ...                    # Other solvers and analysis
├── results/                   # Result JSON files
│   ├── stochastic_beam_result.json  # Best result (MSE=0.274)
│   ├── beam_wide_results.json       # Beam-5000 (MSE=0.356)
│   ├── sinkhorn_overnight/    # 1260-run Sinkhorn sweep
│   └── sinkhorn_sweeps/       # Earlier Sinkhorn experiments
├── logs/                      # Output logs from runs
└── reports/                   # Experiment reports (01–06)
```
