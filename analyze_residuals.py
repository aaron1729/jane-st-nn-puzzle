"""
Analyze residual magnitudes: ||B(ReLU(A(x)))|| / ||x|| for all possible (A, B) pairings.
How big is the residual update relative to the input for each block?
"""
import torch
import numpy as np
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
X = torch.tensor(data[:, :48], device=device)
pred = torch.tensor(data[:, 48], device=device)

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85
w_last = pieces[last_id]["weight"].squeeze(0)
b_last = pieces[last_id]["bias"]

# Per-sample L2 norms of input
x_norms = torch.norm(X, dim=1)  # [N]
x_mean_norm = x_norms.mean().item()
print(f"Input X: mean L2 norm = {x_mean_norm:.4f}", flush=True)
print(f"         std  L2 norm = {x_norms.std().item():.4f}", flush=True)

# Our beam-found pairings
beam10_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

beam200_pairs = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (39, 32), (4, 52), (45, 33), (3, 40), (2, 70), (68, 47), (59, 92),
    (61, 83), (15, 66), (35, 22), (16, 90), (91, 30), (56, 21), (42, 46), (10, 20),
    (13, 34), (1, 12), (18, 63), (28, 25), (74, 80), (44, 7), (86, 76), (69, 89),
    (14, 8), (43, 53), (84, 96), (95, 79), (88, 38), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (62, 57), (65, 24), (77, 82),
]

beam10_set = set(beam10_pairs)
beam200_set = set(beam200_pairs)
shared_pairs = beam10_set & beam200_set

# ============================================================
# Full 48x48 matrix: ||B(ReLU(A(x)))|| / ||x|| for all pairings
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Full pairing matrix: mean( ||B(ReLU(A(x)))||₂ / ||x||₂ ) for all 48×48 combos", flush=True)
print(f"{'='*70}", flush=True)

ratio_matrix = np.zeros((48, 48))
delta_norm_matrix = np.zeros((48, 48))
active_frac_matrix = np.zeros((48, 48))

with torch.no_grad():
    for i, ei in enumerate(expand_ids):
        z = X @ pieces[ei]["weight"].T + pieces[ei]["bias"]  # [N, 96]
        z = torch.relu(z)
        active_frac_row = (z > 0).float().mean().item()

        for j, ci in enumerate(contract_ids):
            delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]  # [N, 48]
            delta_norms = torch.norm(delta, dim=1)  # [N]
            ratio_matrix[i, j] = (delta_norms / x_norms).mean().item()
            delta_norm_matrix[i, j] = delta_norms.mean().item()
            active_frac_matrix[i, j] = active_frac_row

print(f"\nRatio ||delta||/||x|| stats across all 2304 pairings:", flush=True)
print(f"  mean={ratio_matrix.mean():.5f}  std={ratio_matrix.std():.5f}  "
      f"min={ratio_matrix.min():.5f}  max={ratio_matrix.max():.5f}", flush=True)

print(f"\n||delta|| stats:", flush=True)
print(f"  mean={delta_norm_matrix.mean():.4f}  std={delta_norm_matrix.std():.4f}  "
      f"min={delta_norm_matrix.min():.4f}  max={delta_norm_matrix.max():.4f}", flush=True)

# ============================================================
# For each expand, show its ratio distribution + where beam picks land
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Per expand piece: ratio distribution and beam-found contract rank", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'exp':>4s}  {'min_ratio':>10s}  {'mean_ratio':>10s}  {'max_ratio':>10s}  "
      f"{'b10_con':>7s}  {'b10_ratio':>10s}  {'b10_rank':>8s}  "
      f"{'b200_con':>8s}  {'b200_ratio':>10s}  {'b200_rank':>9s}", flush=True)
print("-" * 105, flush=True)

b10_dict = dict(beam10_pairs)
b200_dict = dict(beam200_pairs)

b10_ranks = []
b200_ranks = []

for i, ei in enumerate(expand_ids):
    row = ratio_matrix[i]
    sorted_row = np.sort(row)
    rank_order = np.argsort(row)

    b10_ci = b10_dict.get(ei)
    b200_ci = b200_dict.get(ei)

    b10_j = contract_ids.index(b10_ci) if b10_ci is not None else None
    b200_j = contract_ids.index(b200_ci) if b200_ci is not None else None

    b10_ratio = row[b10_j] if b10_j is not None else None
    b200_ratio = row[b200_j] if b200_j is not None else None

    # Rank (0 = smallest ratio)
    b10_rank = list(rank_order).index(b10_j) if b10_j is not None else -1
    b200_rank = list(rank_order).index(b200_j) if b200_j is not None else -1

    if b10_rank >= 0:
        b10_ranks.append(b10_rank)
    if b200_rank >= 0:
        b200_ranks.append(b200_rank)

    b10_str = f"{b10_ci:>7d}  {b10_ratio:10.5f}  {b10_rank:>8d}" if b10_ci is not None else "     --          --        --"
    b200_str = f"{b200_ci:>8d}  {b200_ratio:10.5f}  {b200_rank:>9d}" if b200_ci is not None else "      --          --         --"
    print(f"  {ei:2d}  {row.min():10.5f}  {row.mean():10.5f}  {row.max():10.5f}  "
          f"{b10_str}  {b200_str}", flush=True)

print(f"\nBeam-10 contract rank stats (0=smallest ratio, 47=largest):", flush=True)
print(f"  mean={np.mean(b10_ranks):.1f}  median={np.median(b10_ranks):.0f}  "
      f"min={np.min(b10_ranks)}  max={np.max(b10_ranks)}", flush=True)
print(f"  How many in top-5: {sum(1 for r in b10_ranks if r < 5)}/48", flush=True)
print(f"  How many in top-10: {sum(1 for r in b10_ranks if r < 10)}/48", flush=True)

print(f"\nBeam-200 contract rank stats:", flush=True)
print(f"  mean={np.mean(b200_ranks):.1f}  median={np.median(b200_ranks):.0f}  "
      f"min={np.min(b200_ranks)}  max={np.max(b200_ranks)}", flush=True)
print(f"  How many in top-5: {sum(1 for r in b200_ranks if r < 5)}/48", flush=True)
print(f"  How many in top-10: {sum(1 for r in b200_ranks if r < 10)}/48", flush=True)

# ============================================================
# Do beam pairings tend to have smaller or larger ratios?
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Beam pairings vs random pairings: ratio comparison", flush=True)
print(f"{'='*70}", flush=True)

beam10_ratios = [ratio_matrix[expand_ids.index(ei), contract_ids.index(ci)]
                 for ei, ci in beam10_pairs]
beam200_ratios = [ratio_matrix[expand_ids.index(ei), contract_ids.index(ci)]
                  for ei, ci in beam200_pairs]
shared_ratios = [ratio_matrix[expand_ids.index(ei), contract_ids.index(ci)]
                 for ei, ci in shared_pairs]

# Random: diagonal of random permutations
import random
random.seed(42)
random_ratios = []
for _ in range(1000):
    perm = list(range(48))
    random.shuffle(perm)
    for k in range(48):
        random_ratios.append(ratio_matrix[k, perm[k]])

print(f"  Beam-10 pairings:  mean={np.mean(beam10_ratios):.5f}  "
      f"std={np.std(beam10_ratios):.5f}  "
      f"min={np.min(beam10_ratios):.5f}  max={np.max(beam10_ratios):.5f}", flush=True)
print(f"  Beam-200 pairings: mean={np.mean(beam200_ratios):.5f}  "
      f"std={np.std(beam200_ratios):.5f}  "
      f"min={np.min(beam200_ratios):.5f}  max={np.max(beam200_ratios):.5f}", flush=True)
print(f"  Shared (5 pairs):  mean={np.mean(shared_ratios):.5f}  "
      f"vals={sorted([f'{v:.4f}' for v in shared_ratios])}", flush=True)
print(f"  Random pairings:   mean={np.mean(random_ratios):.5f}  "
      f"std={np.std(random_ratios):.5f}  "
      f"min={np.min(random_ratios):.5f}  max={np.max(random_ratios):.5f}", flush=True)

# ============================================================
# Sequential ratios through the network for beam orderings
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Sequential: ||delta|| / ||h|| through the network", flush=True)
print(f"{'='*70}", flush=True)

with torch.no_grad():
    for name, pairs in [("Beam-10", beam10_pairs), ("Beam-200", beam200_pairs)]:
        print(f"\n  {name}:", flush=True)
        h = X.clone()
        for pos, (ei, ci) in enumerate(pairs):
            h_norms = torch.norm(h, dim=1)  # [N]
            z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
            z = torch.relu(z)
            delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
            delta_norms = torch.norm(delta, dim=1)  # [N]

            ratio_mean = (delta_norms / h_norms).mean().item()
            ratio_max = (delta_norms / h_norms).max().item()
            h_mean_norm = h_norms.mean().item()
            delta_mean_norm = delta_norms.mean().item()

            h = h + delta

            print(f"    Pos {pos:2d} ({ei:2d},{ci:2d}): "
                  f"||delta||/||h||={ratio_mean:.4f} (max={ratio_max:.3f})  "
                  f"||h||={h_mean_norm:.3f}  ||delta||={delta_mean_norm:.4f}", flush=True)

# ============================================================
# Which pairings give the SMALLEST residual? (closest to identity)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Minimum-ratio pairing (Hungarian-style: smallest total ratio)", flush=True)
print(f"{'='*70}", flush=True)

# Greedy minimum-ratio pairing
used_cols = set()
greedy_min_pairs = []
for i in np.argsort(ratio_matrix.min(axis=1)):
    j = np.argsort(ratio_matrix[i])
    for jj in j:
        if jj not in used_cols:
            used_cols.add(jj)
            greedy_min_pairs.append((expand_ids[i], contract_ids[jj], ratio_matrix[i, jj]))
            break

greedy_min_pairs.sort(key=lambda x: x[0])
print("Greedy min-ratio pairing:", flush=True)
for ei, ci, ratio in greedy_min_pairs:
    tag = ""
    if (ei, ci) in beam10_set:
        tag += " [b10]"
    if (ei, ci) in beam200_set:
        tag += " [b200]"
    if (ei, ci) in shared_pairs:
        tag = " [BOTH]"
    print(f"  ({ei:2d},{ci:2d}): ratio={ratio:.5f}{tag}", flush=True)

overlap_b10 = sum(1 for ei, ci, _ in greedy_min_pairs if (ei, ci) in beam10_set)
overlap_b200 = sum(1 for ei, ci, _ in greedy_min_pairs if (ei, ci) in beam200_set)
print(f"\nOverlap with beam-10: {overlap_b10}/48", flush=True)
print(f"Overlap with beam-200: {overlap_b200}/48", flush=True)

# Evaluate this pairing with beam ordering
print(f"\nEvaluating greedy-min-ratio pairing with beam-10 ordering logic...", flush=True)
# Simple: just use the pairing, order them same as beam-10 expand order
min_ratio_dict = {ei: ci for ei, ci, _ in greedy_min_pairs}
min_ratio_order = [(ei, min_ratio_dict[ei]) for ei, _ in beam10_pairs]
with torch.no_grad():
    h = X.clone()
    for ei, ci in min_ratio_order:
        z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
        z = torch.relu(z)
        delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
        h = h + delta
    out = h @ w_last + b_last[0]
    mse = torch.mean((out - pred) ** 2).item()
print(f"  MSE (beam-10 expand order): {mse:.6f}", flush=True)
