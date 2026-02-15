"""
Test whether block ordering matters significantly.
Key idea: if residuals are small, h_k ≈ X throughout, and ordering doesn't matter.
Only pairings matter.
"""
import torch
import numpy as np
import csv
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Load data
data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
X_np = data[:, :48]
pred_np = data[:, 48]

X = torch.tensor(X_np, device=device)
pred = torch.tensor(pred_np, device=device)

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85

W_last = pieces[last_id]["weight"]
b_last = pieces[last_id]["bias"]
w_last = W_last.squeeze(0)

expand_W = {i: pieces[i]["weight"] for i in expand_ids}
expand_b = {i: pieces[i]["bias"] for i in expand_ids}
contract_W = {i: pieces[i]["weight"] for i in contract_ids}
contract_b = {i: pieces[i]["bias"] for i in contract_ids}


@torch.no_grad()
def apply_block(h, ei, ci):
    z = h @ expand_W[ei].T + expand_b[ei]
    z = torch.relu(z)
    z = z @ contract_W[ci].T + contract_b[ci]
    return h + z


@torch.no_grad()
def block_residual(h, ei, ci):
    """Return the residual delta = contract(relu(expand(h)))"""
    z = h @ expand_W[ei].T + expand_b[ei]
    z = torch.relu(z)
    z = z @ contract_W[ci].T + contract_b[ci]
    return z


# Use beam pairings
beam_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

# Test 1: Compare sequential vs. independent application
print("=" * 60, flush=True)
print("Test 1: Sequential vs. order-independent residuals", flush=True)
print("=" * 60, flush=True)

# Sequential (correct)
h_seq = X.clone()
for ei, ci in beam_pairs:
    h_seq = apply_block(h_seq, ei, ci)

# Order-independent: apply all blocks to RAW X, sum residuals
h_indep = X.clone()
total_delta = torch.zeros_like(X)
for ei, ci in beam_pairs:
    delta = block_residual(X, ei, ci)
    total_delta += delta
h_indep = X + total_delta

# Compare
seq_out = (h_seq @ w_last + b_last[0])
indep_out = (h_indep @ w_last + b_last[0])

seq_mse = torch.mean((seq_out - pred) ** 2).item()
indep_mse = torch.mean((indep_out - pred) ** 2).item()
diff = torch.mean((h_seq - h_indep) ** 2).item()

print(f"Sequential MSE(out, pred): {seq_mse:.6f}", flush=True)
print(f"Independent MSE(out, pred): {indep_mse:.6f}", flush=True)
print(f"||h_seq - h_indep||² (per dim): {diff:.6f}", flush=True)
print(f"h_seq norm: {torch.mean(h_seq ** 2).item():.6f}", flush=True)
print(f"total_delta norm: {torch.mean(total_delta ** 2).item():.6f}", flush=True)

# Test 2: How much does each block change the hidden state?
print(f"\n{'='*60}", flush=True)
print("Test 2: Per-block residual magnitudes", flush=True)
print("=" * 60, flush=True)

h = X.clone()
for i, (ei, ci) in enumerate(beam_pairs):
    delta = block_residual(h, ei, ci)
    delta_norm = torch.mean(delta ** 2).item()
    h_norm = torch.mean(h ** 2).item()
    ratio = delta_norm / h_norm
    if i < 10 or i >= 40:
        print(f"  Block {i:2d} ({ei:2d},{ci:2d}): ||delta||²={delta_norm:.6f} "
              f"||h||²={h_norm:.6f} ratio={ratio:.4f}", flush=True)
    elif i == 10:
        print(f"  ...", flush=True)
    h = h + delta

# Test 3: Does the linear (order-independent) approximation work for SCORING?
# i.e., can we find correct pairings using the linear approx?
print(f"\n{'='*60}", flush=True)
print("Test 3: Order-independent pairing search", flush=True)
print("=" * 60, flush=True)

# For each (expand, contract) pair, compute s(x) = W_last @ delta(X, e, c)
# We want sum_k s_k(x) ≈ pred - b_last - W_last @ X for all x
target = pred - b_last[0] - X @ w_last  # [N] — what residuals should sum to

print(f"Target mean: {target.mean().item():.6f}, std: {target.std().item():.6f}", flush=True)

# Compute contribution of each (expand, contract) pair
n_exp = len(expand_ids)
n_con = len(contract_ids)
contributions = torch.zeros(n_exp, n_con, len(X), device=device)

for i, ei in enumerate(expand_ids):
    z = X @ expand_W[ei].T + expand_b[ei]  # [N, 96]
    z = torch.relu(z)
    for j, ci in enumerate(contract_ids):
        delta = z @ contract_W[ci].T + contract_b[ci]  # [N, 48]
        contributions[i, j] = delta @ w_last  # [N] — projection onto last layer

print(f"Contributions shape: {contributions.shape}", flush=True)

# Hungarian assignment: minimize MSE(sum_k contributions[i_k, j_k, :], target)
# This is hard because it's a sum. Let's try: assign each pair to contribute target/48.
from scipy.optimize import linear_sum_assignment

# Simple approach: cost[i,j] = ||contribution[i,j] - target/48||²
target_per_block = target / 48.0  # [N]
cost = torch.zeros(n_exp, n_con)
for i in range(n_exp):
    for j in range(n_con):
        diff = contributions[i, j] - target_per_block
        cost[i, j] = torch.mean(diff ** 2).item()

row_ind, col_ind = linear_sum_assignment(cost.numpy())
equal_share_pairs = [(expand_ids[r], contract_ids[c]) for r, c in zip(row_ind, col_ind)]
print(f"\nEqual-share assignment pairs: {equal_share_pairs[:10]}...", flush=True)

# Check overlap with beam pairings
beam_pair_set = set(beam_pairs)
eq_pair_set = set(equal_share_pairs)
overlap = beam_pair_set & eq_pair_set
print(f"Overlap with beam: {len(overlap)}/48", flush=True)

# Better approach: greedy assignment maximizing correlation with target
# After assigning pair k, subtract its contribution from target
remaining_target = target.clone()
used_exp = set()
used_con = set()
greedy_pairs = []

for step in range(48):
    best_mse = float('inf')
    best_pair = None
    step_target = remaining_target / (48 - step)  # remaining share

    for i, ei in enumerate(expand_ids):
        if ei in used_exp:
            continue
        for j, ci in enumerate(contract_ids):
            if ci in used_con:
                continue
            # How well does this pair explain remaining target?
            diff = contributions[i, j] - step_target
            mse = torch.mean(diff ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_pair = (ei, ci, i, j)

    ei, ci, i, j = best_pair
    greedy_pairs.append((ei, ci))
    remaining_target = remaining_target - contributions[i, j]
    used_exp.add(ei)
    used_con.add(ci)

greedy_pair_set = set(greedy_pairs)
overlap2 = beam_pair_set & greedy_pair_set
print(f"\nGreedy sequential assignment: overlap with beam = {len(overlap2)}/48", flush=True)
print(f"Greedy pairs: {greedy_pairs[:10]}...", flush=True)

# Test the independent approximation with greedy pairs
h_greedy = X.clone()
for ei, ci in greedy_pairs:
    h_greedy += block_residual(X, ei, ci)
out_greedy = h_greedy @ w_last + b_last[0]
greedy_mse = torch.mean((out_greedy - pred) ** 2).item()
print(f"Greedy pairs independent MSE: {greedy_mse:.6f}", flush=True)

# Also evaluate beam pairs with independent approx for comparison
h_beam_indep = X.clone()
for ei, ci in beam_pairs:
    h_beam_indep += block_residual(X, ei, ci)
out_beam_indep = h_beam_indep @ w_last + b_last[0]
beam_indep_mse = torch.mean((out_beam_indep - pred) ** 2).item()
print(f"Beam pairs independent MSE: {beam_indep_mse:.6f}", flush=True)

# Test 4: Feature structure analysis
print(f"\n{'='*60}", flush=True)
print("Test 4: Which features does each expand/contract piece focus on?", flush=True)
print("=" * 60, flush=True)

# For each expand piece, which input features have highest weight magnitude?
print("\nTop features per expand piece (by L2 norm of weight column):", flush=True)
for idx, ei in enumerate(expand_ids[:5]):
    w = expand_W[ei]  # [96, 48]
    col_norms = torch.norm(w, dim=0)  # [48] — L2 norm per input feature
    top_feats = torch.argsort(col_norms, descending=True)[:5]
    print(f"  Expand {ei}: top features = {top_feats.tolist()} "
          f"(norms: {col_norms[top_feats].tolist()[:5]})", flush=True)

# Check for feature-based clustering
print("\nExpand piece feature focus (top-3 features per piece):", flush=True)
exp_features = {}
for ei in expand_ids:
    w = expand_W[ei]
    col_norms = torch.norm(w, dim=0)
    top3 = torch.argsort(col_norms, descending=True)[:3].tolist()
    exp_features[ei] = top3
    print(f"  Expand {ei:2d}: {top3}", flush=True)

print("\nContract piece feature focus (top-3 features per piece):", flush=True)
con_features = {}
for ci in contract_ids:
    w = contract_W[ci]  # [48, 96]
    row_norms = torch.norm(w, dim=1)  # [48] — L2 norm per output feature
    top3 = torch.argsort(row_norms, descending=True)[:3].tolist()
    con_features[ci] = top3
    print(f"  Contract {ci:2d}: {top3}", flush=True)

# Try matching by feature overlap
print("\nFeature-based pairing (match by top-3 feature overlap):", flush=True)
feat_cost = np.zeros((n_exp, n_con))
for i, ei in enumerate(expand_ids):
    ef = set(exp_features[ei])
    for j, ci in enumerate(contract_ids):
        cf = set(con_features[ci])
        # Cost = negative overlap (more overlap = lower cost = better match)
        feat_cost[i, j] = -len(ef & cf)

row_ind, col_ind = linear_sum_assignment(feat_cost)
feat_pairs = set((expand_ids[r], contract_ids[c]) for r, c in zip(row_ind, col_ind))
feat_overlap = feat_pairs & beam_pair_set
print(f"Feature-based pairs overlap with beam: {len(feat_overlap)}/48", flush=True)
