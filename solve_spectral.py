"""
Use weight matrix properties to determine pairings, then beam search for ordering.
"""
import torch
import numpy as np
import csv
from scipy.optimize import linear_sum_assignment

# Load data
data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
X = data[:, :48]
pred = data[:, 48]

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location="cpu")

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85

W_last = pieces[last_id]["weight"].numpy()
b_last = pieces[last_id]["bias"].numpy()

expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}

n = len(expand_ids)


def apply_block(h, exp_id, con_id):
    z = h @ expand_W[exp_id].T + expand_b[exp_id]
    z = np.maximum(z, 0)
    z = z @ contract_W[con_id].T + contract_b[con_id]
    return h + z


def score_mse(h):
    out = (h @ W_last.T).squeeze() + b_last[0]
    return np.mean((out - pred) ** 2)


# Compute trace score matrix
print("Computing trace score matrix...", flush=True)
trace_scores = np.zeros((n, n))
for i, ei in enumerate(expand_ids):
    for j, ci in enumerate(contract_ids):
        M = contract_W[ci] @ expand_W[ei]  # [48, 48]
        trace_scores[i, j] = np.abs(np.trace(M))

# Find optimal pairing by min trace
row_ind, col_ind = linear_sum_assignment(trace_scores)
trace_pairs = [(expand_ids[i], contract_ids[j]) for i, j in zip(row_ind, col_ind)]

print(f"\nTrace-based pairings ({len(trace_pairs)} blocks):", flush=True)
for ei, ci in trace_pairs:
    M = contract_W[ci] @ expand_W[ei]
    print(f"  ({ei:2d}, {ci:2d})  trace={np.abs(np.trace(M)):.6f}", flush=True)

# Now find ordering using greedy with LastLayer scoring
print(f"\n{'='*60}", flush=True)
print("Greedy ordering with LastLayer scoring", flush=True)
print(f"{'='*60}", flush=True)

h = X.copy()
remaining = list(range(len(trace_pairs)))
order = []

for step in range(len(trace_pairs)):
    best_mse = float("inf")
    best_idx = None
    all_scores = []

    for idx in remaining:
        ei, ci = trace_pairs[idx]
        h_cand = apply_block(h, ei, ci)
        s = score_mse(h_cand)
        all_scores.append((s, idx))
        if s < best_mse:
            best_mse = s
            best_idx = idx

    all_scores.sort()
    order.append(best_idx)
    remaining.remove(best_idx)
    ei, ci = trace_pairs[best_idx]
    h = apply_block(h, ei, ci)

    gap = all_scores[1][0] - all_scores[0][0] if len(all_scores) > 1 else 0
    print(f"  Step {step+1:2d}: block ({ei:2d},{ci:2d}) MSE={best_mse:.6f} gap={gap:.6f}", flush=True)

# Final result
final_out = (h @ W_last.T).squeeze() + b_last[0]
final_mse = np.mean((final_out - pred) ** 2)
max_err = np.max(np.abs(final_out - pred))

ordered_pairs = [trace_pairs[i] for i in order]
print(f"\nFinal MSE: {final_mse:.10f}", flush=True)
print(f"Max absolute error: {max_err:.10f}", flush=True)
print(f"Ordered pairs: {ordered_pairs}", flush=True)

# Build permutation
perm = []
for ei, ci in ordered_pairs:
    perm.append(ei)
    perm.append(ci)
perm.append(last_id)
print(f"\nPermutation ({len(perm)} pieces):", flush=True)
print(",".join(str(p) for p in perm), flush=True)

# Also try beam search for ordering
print(f"\n{'='*60}", flush=True)
print("Beam search (width 20) for ordering", flush=True)
print(f"{'='*60}", flush=True)

BEAM_WIDTH = 20
beam = [(score_mse(X), X.copy(), [], list(range(len(trace_pairs))))]

for step in range(len(trace_pairs)):
    candidates = []
    for b_score, b_h, b_order, b_remaining in beam:
        for idx in b_remaining:
            ei, ci = trace_pairs[idx]
            h_new = apply_block(b_h, ei, ci)
            s = score_mse(h_new)
            candidates.append((s, h_new, b_order + [idx], idx, id(b_remaining), b_remaining))

    candidates.sort(key=lambda x: x[0])

    new_beam = []
    for s, h_new, new_order, chosen_idx, _, b_remaining in candidates[:BEAM_WIDTH]:
        new_remaining = [x for x in b_remaining if x != chosen_idx]
        new_beam.append((s, h_new, new_order, new_remaining))

    beam = new_beam

    best = beam[0]
    worst = beam[-1]
    gap = beam[1][0] - beam[0][0] if len(beam) > 1 else 0
    ei, ci = trace_pairs[best[2][-1]]
    print(f"  Step {step+1:2d}: block ({ei:2d},{ci:2d}) best={best[0]:.6f} "
          f"spread={worst[0]-best[0]:.6f} gap={gap:.6f}", flush=True)

# Best beam result
best_score, best_h, best_order, _ = beam[0]
best_ordered_pairs = [trace_pairs[i] for i in best_order]

final_out = (best_h @ W_last.T).squeeze() + b_last[0]
final_mse = np.mean((final_out - pred) ** 2)
max_err = np.max(np.abs(final_out - pred))

print(f"\nBeam best MSE: {final_mse:.10f}", flush=True)
print(f"Max absolute error: {max_err:.10f}", flush=True)
print(f"Ordered pairs: {best_ordered_pairs}", flush=True)

perm = []
for ei, ci in best_ordered_pairs:
    perm.append(ei)
    perm.append(ci)
perm.append(last_id)
print(f"\nPermutation ({len(perm)} pieces):", flush=True)
print(",".join(str(p) for p in perm), flush=True)
