import torch
import numpy as np
import csv

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

# Load all pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location="cpu")

# Categorize
expand_ids = [i for i in range(97) if pieces[i]["weight"].shape == (96, 48)]
contract_ids = [i for i in range(97) if pieces[i]["weight"].shape == (48, 96)]
last_id = 85  # Linear(48, 1)

W_last = pieces[last_id]["weight"].numpy()  # [1, 48]
b_last = pieces[last_id]["bias"].numpy()    # [1]

# Convert to numpy for speed
expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}


def apply_block(h, exp_id, con_id):
    """Apply a residual block: h + contract(relu(expand(h)))"""
    z = h @ expand_W[exp_id].T + expand_b[exp_id]  # [N, 96]
    z = np.maximum(z, 0)  # ReLU
    z = z @ contract_W[con_id].T + contract_b[con_id]  # [N, 48]
    return h + z


def score(h):
    """Score current representation by MSE of LastLayer projection vs pred."""
    out = (h @ W_last.T).squeeze() + b_last[0]  # [N]
    return np.mean((out - pred) ** 2)


# Greedy search
h = X.copy()
remaining_expand = set(expand_ids)
remaining_contract = set(contract_ids)
chosen_order = []

null_score = score(h)
print(f"Initial (null model) MSE: {null_score:.6f}")
print()

N_STEPS = 48

for step in range(N_STEPS):
    best_mse = float("inf")
    best_pair = None
    all_scores = []

    for ei in remaining_expand:
        for ci in remaining_contract:
            h_candidate = apply_block(h, ei, ci)
            s = score(h_candidate)
            all_scores.append((s, ei, ci))
            if s < best_mse:
                best_mse = s
                best_pair = (ei, ci)

    # Sort to see the landscape
    all_scores.sort()
    ei, ci = best_pair
    h = apply_block(h, ei, ci)
    remaining_expand.remove(ei)
    remaining_contract.remove(ci)
    chosen_order.append(best_pair)

    print(f"Step {step + 1}: best pair = (expand={ei}, contract={ci}), MSE = {best_mse:.6f}")
    print(f"  Top 5 scores:  {[(f'{s:.6f}', e, c) for s, e, c in all_scores[:5]]}")
    print(f"  Worst 5 scores: {[(f'{s:.6f}', e, c) for s, e, c in all_scores[-5:]]}")
    if len(all_scores) > 1:
        gap = all_scores[1][0] - all_scores[0][0]
        print(f"  Gap (2nd - 1st): {gap:.6f}  (ratio: {all_scores[1][0] / all_scores[0][0]:.6f})")
    print()

print(f"Chosen so far: {chosen_order}")

# Build full permutation: each block contributes (expand_id, contract_id), then last layer
permutation = []
for ei, ci in chosen_order:
    permutation.append(ei)
    permutation.append(ci)
permutation.append(last_id)

print(f"\nFull permutation ({len(permutation)} pieces):")
print(",".join(str(p) for p in permutation))

# Final validation
final_out = (h @ W_last.T).squeeze() + b_last[0]
final_mse = np.mean((final_out - pred) ** 2)
print(f"\nFinal MSE vs pred: {final_mse:.8f}")
print(f"Max absolute error: {np.max(np.abs(final_out - pred)):.8f}")
