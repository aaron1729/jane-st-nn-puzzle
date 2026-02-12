"""
Compare pairing methods against the beam search's first 40 pairings.
Also try a local search approach: start from beam solution, swap blocks.
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

ei2idx = {eid: i for i, eid in enumerate(expand_ids)}
ci2idx = {cid: j for j, cid in enumerate(contract_ids)}


def apply_block(h, exp_id, con_id):
    z = h @ expand_W[exp_id].T + expand_b[exp_id]
    z = np.maximum(z, 0)
    z = z @ contract_W[con_id].T + contract_b[con_id]
    return h + z


def full_mse(block_order):
    """Compute MSE for a full ordering of blocks."""
    h = X.copy()
    for ei, ci in block_order:
        h = apply_block(h, ei, ci)
    out = (h @ W_last.T).squeeze() + b_last[0]
    return np.mean((out - pred) ** 2)


# Beam search first 40 pairs
beam_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
]
beam_pair_set = set(beam_pairs)

n = len(expand_ids)

# Compute score matrices
print("Computing score matrices...", flush=True)

fro_scores = np.zeros((n, n))
trace_scores = np.zeros((n, n))
residual_norms = np.zeros((n, n))

for i, ei in enumerate(expand_ids):
    z = X @ expand_W[ei].T + expand_b[ei]
    z_relu = np.maximum(z, 0)
    for j, ci in enumerate(contract_ids):
        M = contract_W[ci] @ expand_W[ei]
        fro_scores[i, j] = np.linalg.norm(M, 'fro')
        trace_scores[i, j] = np.abs(np.trace(M))
        r = z_relu @ contract_W[ci].T + contract_b[ci]
        residual_norms[i, j] = np.linalg.norm(r) / len(X)

# For each scoring method, find optimal assignment and check overlap with beam
for name, scores in [("Frobenius", fro_scores), ("Trace", trace_scores),
                     ("ResidualNorm", residual_norms)]:
    row_ind, col_ind = linear_sum_assignment(scores)
    method_pairs = set()
    for i, j in zip(row_ind, col_ind):
        method_pairs.add((expand_ids[i], contract_ids[j]))

    overlap = method_pairs & beam_pair_set
    print(f"\n{name}: {len(overlap)}/40 pairs match beam ({len(overlap)/40*100:.0f}%)", flush=True)

    # Show mismatches
    beam_only = beam_pair_set - method_pairs
    method_only = method_pairs - beam_pair_set
    if beam_only:
        beam_only_involved = set()
        for ei, ci in beam_only:
            beam_only_involved.add(ei)
            beam_only_involved.add(ci)
        print(f"  Beam pairs not in {name}: {sorted(beam_only)}", flush=True)
        print(f"  {name} pairs not in beam: {sorted([(e,c) for e,c in method_only if e in beam_only_involved or c in beam_only_involved])}", flush=True)

# Local search: start from beam solution, try swapping pairs and positions
print(f"\n{'='*60}", flush=True)
print("Local search from beam solution", flush=True)
print(f"{'='*60}", flush=True)

# Start with beam ordering + greedy tail from experiment 1
beam_tail = [(42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57)]
current_order = beam_pairs + beam_tail
current_mse = full_mse(current_order)
print(f"Starting MSE: {current_mse:.6f}", flush=True)

# Local search: swap positions of two blocks
import itertools
improved = True
iteration = 0
while improved:
    improved = False
    iteration += 1
    swaps_tried = 0
    swaps_accepted = 0

    for i in range(48):
        for j in range(i + 1, 48):
            new_order = list(current_order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_mse = full_mse(new_order)
            swaps_tried += 1
            if new_mse < current_mse:
                current_order = new_order
                current_mse = new_mse
                improved = True
                swaps_accepted += 1

    print(f"  Iteration {iteration}: MSE={current_mse:.6f} "
          f"({swaps_accepted} swaps accepted / {swaps_tried} tried)", flush=True)

    # Also try swapping pairings: swap contract pieces between two blocks
    for i in range(48):
        for j in range(i + 1, 48):
            ei_i, ci_i = current_order[i]
            ei_j, ci_j = current_order[j]
            new_order = list(current_order)
            new_order[i] = (ei_i, ci_j)
            new_order[j] = (ei_j, ci_i)
            new_mse = full_mse(new_order)
            swaps_tried += 1
            if new_mse < current_mse:
                current_order = new_order
                current_mse = new_mse
                improved = True
                swaps_accepted += 1
                print(f"    Pair swap ({ei_i},{ci_i})<->({ei_j},{ci_j}): MSE={current_mse:.6f}", flush=True)

    if improved:
        print(f"  After pair swaps: MSE={current_mse:.6f} "
              f"({swaps_accepted} total accepted)", flush=True)

print(f"\nFinal MSE after local search: {current_mse:.10f}", flush=True)
print(f"Final ordering: {current_order}", flush=True)

# Build permutation
perm = []
for ei, ci in current_order:
    perm.append(ei)
    perm.append(ci)
perm.append(last_id)
print(f"\nPermutation ({len(perm)} pieces):", flush=True)
print(",".join(str(p) for p in perm), flush=True)

# Validate
h = X.copy()
for ei, ci in current_order:
    h = apply_block(h, ei, ci)
out = (h @ W_last.T).squeeze() + b_last[0]
print(f"Max absolute error: {np.max(np.abs(out - pred)):.10f}", flush=True)
