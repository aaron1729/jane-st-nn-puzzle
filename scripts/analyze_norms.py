"""
Analyze weight norms: if the network was trained with L2 penalty,
correct (A, B) pairs should have ||A|| ≈ ||B|| due to the
rescaling symmetry A→αA, B→(1/α)B.
"""
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])

# Compute Frobenius norms of all weight matrices
expand_norms = {}
contract_norms = {}

print("Expand piece norms (W: 96×48):", flush=True)
for ei in expand_ids:
    w = pieces[ei]["weight"]
    b = pieces[ei]["bias"]
    fnorm = torch.norm(w, 'fro').item()
    bnorm = torch.norm(b).item()
    expand_norms[ei] = fnorm
    print(f"  Exp {ei:2d}: ||W||_F = {fnorm:.4f}  ||b|| = {bnorm:.4f}", flush=True)

print(f"\nContract piece norms (W: 48×96):", flush=True)
for ci in contract_ids:
    w = pieces[ci]["weight"]
    b = pieces[ci]["bias"]
    fnorm = torch.norm(w, 'fro').item()
    bnorm = torch.norm(b).item()
    contract_norms[ci] = fnorm
    print(f"  Con {ci:2d}: ||W||_F = {fnorm:.4f}  ||b|| = {bnorm:.4f}", flush=True)

exp_norm_vals = [expand_norms[ei] for ei in expand_ids]
con_norm_vals = [contract_norms[ci] for ci in contract_ids]
print(f"\nExpand norms:   mean={np.mean(exp_norm_vals):.4f}  std={np.std(exp_norm_vals):.4f}  "
      f"min={np.min(exp_norm_vals):.4f}  max={np.max(exp_norm_vals):.4f}", flush=True)
print(f"Contract norms: mean={np.mean(con_norm_vals):.4f}  std={np.std(con_norm_vals):.4f}  "
      f"min={np.min(con_norm_vals):.4f}  max={np.max(con_norm_vals):.4f}", flush=True)

# ============================================================
# Full 48×48 norm-ratio matrix: ||A|| / ||B|| for all pairings
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Norm ratio ||W_expand|| / ||W_contract|| for all 48×48 pairings", flush=True)
print(f"{'='*70}", flush=True)

ratio_matrix = np.zeros((48, 48))
diff_matrix = np.zeros((48, 48))
for i, ei in enumerate(expand_ids):
    for j, ci in enumerate(contract_ids):
        ratio_matrix[i, j] = expand_norms[ei] / contract_norms[ci]
        diff_matrix[i, j] = abs(expand_norms[ei] - contract_norms[ci])

print(f"Ratio stats: mean={ratio_matrix.mean():.4f}  std={ratio_matrix.std():.4f}  "
      f"min={ratio_matrix.min():.4f}  max={ratio_matrix.max():.4f}", flush=True)
print(f"|Diff| stats: mean={diff_matrix.mean():.4f}  std={diff_matrix.std():.4f}  "
      f"min={diff_matrix.min():.4f}  max={diff_matrix.max():.4f}", flush=True)

# How close to 1.0 is the ratio? Use |log(ratio)| as distance from balanced
log_ratio_matrix = np.abs(np.log(ratio_matrix))
print(f"|log(ratio)| stats: mean={log_ratio_matrix.mean():.4f}  std={log_ratio_matrix.std():.4f}  "
      f"min={log_ratio_matrix.min():.4f}  max={log_ratio_matrix.max():.4f}", flush=True)

# ============================================================
# Check beam pairings
# ============================================================
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

# SA best (from stochastic beam)
sa_best_pairs = [
    (27, 76), (94, 96), (50, 66), (3, 53), (1, 21), (56, 30), (18, 78), (15, 67),
    (43, 55), (84, 63), (13, 7), (77, 25), (2, 70), (44, 75), (61, 20), (59, 29),
    (58, 40), (95, 90), (42, 36), (74, 92), (91, 72), (88, 89), (41, 22), (39, 33),
    (86, 47), (87, 8), (31, 26), (14, 11), (73, 46), (65, 17), (45, 54), (35, 32),
    (4, 52), (48, 19), (23, 12), (49, 51), (28, 24), (62, 82), (60, 57), (68, 80),
    (64, 79), (16, 71), (0, 34), (69, 83), (5, 9), (10, 38), (37, 6), (81, 93),
]

beam10_set = set(beam10_pairs)
beam200_set = set(beam200_pairs)
shared_pairs = beam10_set & beam200_set

print(f"\n{'='*70}", flush=True)
print("Norm ratios for beam-found pairings", flush=True)
print(f"{'='*70}", flush=True)

for name, pairs in [("Beam-10", beam10_pairs), ("Beam-200", beam200_pairs),
                     ("SA-best (MSE=0.274)", sa_best_pairs)]:
    ratios = []
    log_ratios = []
    diffs = []
    print(f"\n  {name}:", flush=True)
    for ei, ci in pairs:
        en = expand_norms[ei]
        cn = contract_norms[ci]
        r = en / cn
        lr = abs(np.log(r))
        d = abs(en - cn)
        ratios.append(r)
        log_ratios.append(lr)
        diffs.append(d)
        tag = ""
        if (ei, ci) in shared_pairs:
            tag = " [SHARED]"
        print(f"    ({ei:2d},{ci:2d}): ||A||={en:.4f}  ||B||={cn:.4f}  "
              f"ratio={r:.4f}  |log(ratio)|={lr:.4f}  |diff|={d:.4f}{tag}", flush=True)

    print(f"  Summary: mean_ratio={np.mean(ratios):.4f}  std={np.std(ratios):.4f}  "
          f"mean_|log|={np.mean(log_ratios):.4f}  mean_|diff|={np.mean(diffs):.4f}", flush=True)

# ============================================================
# Compare: beam pairings vs random vs min-|log(ratio)| pairing
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Distribution comparison: beam vs random vs matched-norm pairing", flush=True)
print(f"{'='*70}", flush=True)

import random
random.seed(42)
random_log_ratios = []
for _ in range(1000):
    perm = list(range(48))
    random.shuffle(perm)
    for k in range(48):
        ei = expand_ids[k]
        ci = contract_ids[perm[k]]
        random_log_ratios.append(abs(np.log(expand_norms[ei] / contract_norms[ci])))

b10_lr = [abs(np.log(expand_norms[ei] / contract_norms[ci])) for ei, ci in beam10_pairs]
b200_lr = [abs(np.log(expand_norms[ei] / contract_norms[ci])) for ei, ci in beam200_pairs]
sa_lr = [abs(np.log(expand_norms[ei] / contract_norms[ci])) for ei, ci in sa_best_pairs]
shared_lr = [abs(np.log(expand_norms[ei] / contract_norms[ci])) for ei, ci in shared_pairs]

print(f"  Beam-10:    mean |log(ratio)| = {np.mean(b10_lr):.4f}  std={np.std(b10_lr):.4f}", flush=True)
print(f"  Beam-200:   mean |log(ratio)| = {np.mean(b200_lr):.4f}  std={np.std(b200_lr):.4f}", flush=True)
print(f"  SA-best:    mean |log(ratio)| = {np.mean(sa_lr):.4f}  std={np.std(sa_lr):.4f}", flush=True)
print(f"  Shared (5): mean |log(ratio)| = {np.mean(shared_lr):.4f}  vals={sorted([f'{v:.4f}' for v in shared_lr])}", flush=True)
print(f"  Random:     mean |log(ratio)| = {np.mean(random_log_ratios):.4f}  std={np.std(random_log_ratios):.4f}", flush=True)

# ============================================================
# Find the norm-matched pairing (minimize total |log(ratio)|)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Optimal norm-matched pairing (minimize sum of |log(||A||/||B||)|)", flush=True)
print(f"{'='*70}", flush=True)

# Use scipy Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(log_ratio_matrix)
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        ei = expand_ids[i]
        ci = contract_ids[j]
        lr = log_ratio_matrix[i, j]
        matched_pairs.append((ei, ci, lr))

    matched_pairs.sort(key=lambda x: x[2])
    print("Hungarian optimal norm-matched pairing:", flush=True)
    for ei, ci, lr in matched_pairs:
        en = expand_norms[ei]
        cn = contract_norms[ci]
        tag = ""
        if (ei, ci) in beam10_set:
            tag += " [b10]"
        if (ei, ci) in beam200_set:
            tag += " [b200]"
        if (ei, ci) in shared_pairs:
            tag = " [BOTH]"
        print(f"  ({ei:2d},{ci:2d}): ||A||={en:.4f}  ||B||={cn:.4f}  "
              f"|log(ratio)|={lr:.4f}{tag}", flush=True)

    matched_lr = [lr for _, _, lr in matched_pairs]
    print(f"\n  Total |log(ratio)|: {sum(matched_lr):.4f}", flush=True)
    print(f"  Mean  |log(ratio)|: {np.mean(matched_lr):.4f}", flush=True)

    overlap_b10 = sum(1 for ei, ci, _ in matched_pairs if (ei, ci) in beam10_set)
    overlap_b200 = sum(1 for ei, ci, _ in matched_pairs if (ei, ci) in beam200_set)
    overlap_sa = sum(1 for ei, ci, _ in matched_pairs if (ei, ci) in set(sa_best_pairs))
    print(f"\n  Overlap with beam-10: {overlap_b10}/48", flush=True)
    print(f"  Overlap with beam-200: {overlap_b200}/48", flush=True)
    print(f"  Overlap with SA-best: {overlap_sa}/48", flush=True)
except ImportError:
    print("scipy not available, skipping Hungarian matching", flush=True)

# ============================================================
# Also check: do expand and contract norms have matching distributions?
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Sorted norms comparison", flush=True)
print(f"{'='*70}", flush=True)

exp_sorted = sorted(expand_norms.items(), key=lambda x: x[1])
con_sorted = sorted(contract_norms.items(), key=lambda x: x[1])

print(f"{'Rank':>4s}  {'Exp ID':>6s}  {'Exp ||W||':>10s}  {'Con ID':>6s}  {'Con ||W||':>10s}  {'ratio':>8s}", flush=True)
print("-" * 55, flush=True)
for rank, ((ei, en), (ci, cn)) in enumerate(zip(exp_sorted, con_sorted)):
    print(f"  {rank:2d}     {ei:2d}     {en:10.4f}     {ci:2d}     {cn:10.4f}  {en/cn:8.4f}", flush=True)
