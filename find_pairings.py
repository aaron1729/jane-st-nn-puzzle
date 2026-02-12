"""
Try to determine correct (expand, contract) pairings from weight matrix properties alone.

Key idea: in a trained residual block h + out(relu(inp(h))), the matrices inp and out
were trained together. The effective linear map (ignoring ReLU) is out_W @ inp_W, a 48x48
matrix. For correctly paired layers from a stable trained network, this should have
specific spectral properties (e.g., bounded spectral norm, specific singular value patterns).

We compute various similarity scores for all 48x48 possible pairings and look for structure.
"""
import torch
import numpy as np
import csv
from scipy.optimize import linear_sum_assignment

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location="cpu")

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])

expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}

# Also load data for data-dependent tests
data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
X = data[:, :48]

print("Computing pairing scores for all 48x48 (expand, contract) combinations...\n")

n = len(expand_ids)
assert n == 48

# Score 1: Frobenius norm of out_W @ inp_W
# This is the effective linear map of the block (ignoring ReLU)
fro_scores = np.zeros((n, n))
for i, ei in enumerate(expand_ids):
    for j, ci in enumerate(contract_ids):
        M = contract_W[ci] @ expand_W[ei]  # [48, 48]
        fro_scores[i, j] = np.linalg.norm(M, 'fro')

# Score 2: Spectral norm (largest singular value) of out_W @ inp_W
spec_scores = np.zeros((n, n))
for i, ei in enumerate(expand_ids):
    for j, ci in enumerate(contract_ids):
        M = contract_W[ci] @ expand_W[ei]
        spec_scores[i, j] = np.linalg.svd(M, compute_uv=False)[0]

# Score 3: Trace of out_W @ inp_W (related to gradient flow)
trace_scores = np.zeros((n, n))
for i, ei in enumerate(expand_ids):
    for j, ci in enumerate(contract_ids):
        M = contract_W[ci] @ expand_W[ei]
        trace_scores[i, j] = np.abs(np.trace(M))

# Score 4: Data-dependent: norm of residual when applied to X
# residual = out(relu(inp(X)))
residual_norms = np.zeros((n, n))
for i, ei in enumerate(expand_ids):
    z = X @ expand_W[ei].T + expand_b[ei]  # [N, 96]
    z_relu = np.maximum(z, 0)
    for j, ci in enumerate(contract_ids):
        r = z_relu @ contract_W[ci].T + contract_b[ci]  # [N, 48]
        residual_norms[i, j] = np.linalg.norm(r) / len(X)

# Score 5: Cosine similarity of bias vectors
# inp has bias [96], out has bias [48]. Not directly comparable.
# But we can look at the column norms/patterns.

# Analysis: for each score matrix, find the optimal assignment and see how
# "peaked" each row is (how much the best match stands out)
for name, scores in [("Frobenius", fro_scores), ("Spectral", spec_scores),
                     ("Trace", trace_scores), ("ResidualNorm", residual_norms)]:
    print(f"\n{'='*60}")
    print(f"Score: {name}")
    print(f"{'='*60}")

    # For each expand piece, how peaked is the distribution across contract pieces?
    peakedness = []
    for i in range(n):
        row = scores[i]
        sorted_row = np.sort(row)
        # Ratio of best to second-best
        if name in ("Frobenius", "Spectral", "Trace"):
            # Lower might mean more "matched" (compressed residual)
            ratio = sorted_row[1] / sorted_row[0] if sorted_row[0] > 0 else 0
        else:
            ratio = sorted_row[1] / sorted_row[0] if sorted_row[0] > 0 else 0
        peakedness.append(ratio)

    print(f"  Row peakedness (2nd/1st ratio): "
          f"mean={np.mean(peakedness):.4f} min={np.min(peakedness):.4f} max={np.max(peakedness):.4f}")

    # Find optimal assignment minimizing total score
    row_ind, col_ind = linear_sum_assignment(scores)
    min_total = scores[row_ind, col_ind].sum()

    # Also try maximizing
    row_ind_max, col_ind_max = linear_sum_assignment(-scores)
    max_total = scores[row_ind_max, col_ind_max].sum()

    print(f"  Min-cost assignment total: {min_total:.4f}")
    print(f"  Max-cost assignment total: {max_total:.4f}")

    # Show the min-cost assignment
    print(f"  Min-cost pairings:")
    for i, j in zip(row_ind[:10], col_ind[:10]):
        print(f"    expand={expand_ids[i]:2d} <-> contract={contract_ids[j]:2d}  score={scores[i,j]:.6f}")
    if n > 10:
        print(f"    ... ({n} total)")

    # Look at the diagonal-like structure: sort and see if there's a pattern
    # Check: is the min-cost assignment "much better" than random?
    rng = np.random.default_rng(42)
    random_costs = []
    for _ in range(10000):
        perm = rng.permutation(n)
        cost = sum(scores[i, perm[i]] for i in range(n))
        random_costs.append(cost)
    random_costs = np.array(random_costs)
    print(f"  Random assignment total: mean={random_costs.mean():.4f} std={random_costs.std():.4f}")
    print(f"  Min-cost is {(random_costs.mean() - min_total) / random_costs.std():.1f} stdevs below random mean")
    print(f"  Max-cost is {(max_total - random_costs.mean()) / random_costs.std():.1f} stdevs above random mean")
