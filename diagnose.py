"""
Diagnostics on the beam-200 result:
1. Leave-one-out: which blocks are hurting MSE?
2. Cyclic coordinate descent: optimize each position independently
3. Try different pairings at each position
"""
import torch
import numpy as np
import csv
import time
import hashlib

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
def full_mse(block_order):
    h = X.clone()
    for ei, ci in block_order:
        h = apply_block(h, ei, ci)
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def mse_skipping(block_order, skip_idx):
    """MSE with block at skip_idx replaced by identity (skip connection only)."""
    h = X.clone()
    for i, (ei, ci) in enumerate(block_order):
        if i == skip_idx:
            continue  # skip = identity
        h = apply_block(h, ei, ci)
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


def check_hash(block_order):
    perm = []
    for ei, ci in block_order:
        perm.append(ei)
        perm.append(ci)
    perm.append(last_id)
    perm_str = ",".join(str(p) for p in perm)
    h = hashlib.sha256(perm_str.encode()).hexdigest()
    target = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"
    return h == target, perm_str


# Best known orderings
beam200 = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (39, 32), (4, 52), (45, 33), (3, 40), (2, 70), (68, 47), (59, 92),
    (61, 83), (15, 66), (35, 22), (16, 90), (91, 30), (56, 21), (42, 46), (10, 20),
    (13, 34), (1, 12), (18, 63), (28, 25), (74, 80), (44, 7), (86, 76), (69, 89),
    (14, 8), (43, 53), (84, 96), (95, 79), (88, 38), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (62, 57), (65, 24), (77, 82),
]

base_mse = full_mse(beam200)
print(f"Base MSE: {base_mse:.6f}", flush=True)

# ============================================================
# Test 1: Leave-one-out — which blocks hurt MSE?
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Leave-one-out analysis", flush=True)
print(f"{'='*60}", flush=True)

for i in range(48):
    skip_mse = mse_skipping(beam200, i)
    delta = skip_mse - base_mse
    flag = "  HURTS" if delta < -0.001 else "  helps" if delta > 0.001 else ""
    if abs(delta) > 0.001:
        print(f"  Block {i:2d} ({beam200[i][0]:2d},{beam200[i][1]:2d}): "
              f"skip_MSE={skip_mse:.6f} delta={delta:+.6f}{flag}", flush=True)

# ============================================================
# Test 2: For each position, which block fits best?
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Best block at each position (cyclic coordinate descent)", flush=True)
print(f"{'='*60}", flush=True)

current_order = list(beam200)

for cycle in range(3):
    print(f"\n--- Cycle {cycle+1} ---", flush=True)
    improved_any = False

    for pos in range(48):
        # Cache hidden state before this position
        h_pre = X.clone()
        for i in range(pos):
            h_pre = apply_block(h_pre, current_order[i][0], current_order[i][1])

        # Current block at this position
        cur_ei, cur_ci = current_order[pos]
        suffix = current_order[pos+1:]

        # Try all available expand/contract pairs at this position
        used_exp = set(b[0] for i, b in enumerate(current_order) if i != pos)
        used_con = set(b[1] for i, b in enumerate(current_order) if i != pos)

        avail_exp = [e for e in expand_ids if e not in used_exp]
        avail_con = [c for c in contract_ids if c not in used_con]

        best_mse = full_mse(current_order)
        best_pair = (cur_ei, cur_ci)

        for ei in avail_exp:
            for ci in avail_con:
                trial_order = current_order[:pos] + [(ei, ci)] + suffix
                mse = full_mse(trial_order)
                if mse < best_mse:
                    best_mse = mse
                    best_pair = (ei, ci)

        if best_pair != (cur_ei, cur_ci):
            old_mse = full_mse(current_order)
            current_order[pos] = best_pair
            print(f"  Pos {pos:2d}: ({cur_ei},{cur_ci}) -> ({best_pair[0]},{best_pair[1]}) "
                  f"MSE: {old_mse:.6f} -> {best_mse:.6f}", flush=True)
            improved_any = True

    cycle_mse = full_mse(current_order)
    print(f"Cycle {cycle+1} MSE: {cycle_mse:.6f}", flush=True)

    match, perm_str = check_hash(current_order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
        break

    if not improved_any:
        print("No improvements, converged.", flush=True)
        break

# ============================================================
# Test 3: Also try swapping contract pieces between positions
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Pairwise contract swaps", flush=True)
print(f"{'='*60}", flush=True)

improved = True
while improved:
    improved = False
    base_mse = full_mse(current_order)

    for i in range(48):
        for j in range(i+1, 48):
            trial = list(current_order)
            # Swap contract pieces between positions i and j
            trial[i] = (trial[i][0], current_order[j][1])
            trial[j] = (trial[j][0], current_order[i][1])
            mse = full_mse(trial)
            if mse < base_mse:
                current_order = trial
                base_mse = mse
                improved = True
                print(f"  Swap contracts ({i},{j}): MSE={mse:.6f}", flush=True)

    # Also try position swaps
    for i in range(48):
        for j in range(i+1, 48):
            trial = list(current_order)
            trial[i], trial[j] = trial[j], trial[i]
            mse = full_mse(trial)
            if mse < base_mse:
                current_order = trial
                base_mse = mse
                improved = True
                print(f"  Swap positions ({i},{j}): MSE={mse:.6f}", flush=True)

print(f"\nFinal MSE: {full_mse(current_order):.10f}", flush=True)
print(f"Final ordering: {current_order}", flush=True)
match, perm_str = check_hash(current_order)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)
