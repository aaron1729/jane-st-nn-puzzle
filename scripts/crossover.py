"""
Crossover experiment: mix prefixes from beam-10 and beam-200 solutions.
If either basin contains many correct blocks, mixing should help.
Also tries using 'true' as an auxiliary scoring signal.
"""
import torch
import numpy as np
import csv
import hashlib
import random
import itertools

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
true_vals = torch.tensor(data[:, 49], device=device)
TRUE_MSE = torch.mean((pred - true_vals) ** 2).item()

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
def apply_blocks(h, blocks):
    for ei, ci in blocks:
        z = h @ expand_W[ei].T + expand_b[ei]
        z = torch.relu(z)
        z = z @ contract_W[ci].T + contract_b[ci]
        h = h + z
    return h


@torch.no_grad()
def full_mse(block_order):
    h = apply_blocks(X.clone(), block_order)
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def full_mse_true(block_order):
    h = apply_blocks(X.clone(), block_order)
    out = h @ w_last + b_last[0]
    return torch.mean((out - true_vals) ** 2).item()


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


# Two beam solutions with different basins
beam10 = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

beam200 = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (39, 32), (4, 52), (45, 33), (3, 40), (2, 70), (68, 47), (59, 92),
    (61, 83), (15, 66), (35, 22), (16, 90), (91, 30), (56, 21), (42, 46), (10, 20),
    (13, 34), (1, 12), (18, 63), (28, 25), (74, 80), (44, 7), (86, 76), (69, 89),
    (14, 8), (43, 53), (84, 96), (95, 79), (88, 38), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (62, 57), (65, 24), (77, 82),
]

print(f"Beam-10 MSE:  {full_mse(beam10):.6f}", flush=True)
print(f"Beam-200 MSE: {full_mse(beam200):.6f}", flush=True)

# ============================================================
# Test 1: Take first k blocks from one, fill remainder from other
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Test 1: Crossover at position k", flush=True)
print(f"{'='*60}", flush=True)


def fill_remainder(prefix, pool):
    """Given a prefix of blocks, fill remaining positions using blocks from pool.
    Only use expand/contract pieces not already in prefix."""
    used_exp = set(b[0] for b in prefix)
    used_con = set(b[1] for b in prefix)
    remaining = [(e, c) for e, c in pool if e not in used_exp and c not in used_con]

    # If some pieces are used with different pairings, we need to be smarter
    if len(prefix) + len(remaining) < 48:
        # Can't directly fill — pairings conflict. Use remaining pieces greedily.
        avail_exp = [e for e in expand_ids if e not in used_exp]
        avail_con = [c for c in contract_ids if c not in used_con]
        # Try to use pairings from pool where possible
        pool_map = {e: c for e, c in pool}
        remaining = []
        used_e = set()
        used_c = set()
        for e in avail_exp:
            if e in pool_map and pool_map[e] in avail_con and pool_map[e] not in used_c:
                remaining.append((e, pool_map[e]))
                used_e.add(e)
                used_c.add(pool_map[e])
        # Fill any remaining with arbitrary pairing
        leftover_exp = [e for e in avail_exp if e not in used_e]
        leftover_con = [c for c in avail_con if c not in used_c]
        for e, c in zip(leftover_exp, leftover_con):
            remaining.append((e, c))

    return prefix + remaining


for k in range(0, 49, 4):
    # Beam-10 prefix + beam-200 remainder
    order_10_200 = fill_remainder(beam10[:k], beam200)
    # Beam-200 prefix + beam-10 remainder
    order_200_10 = fill_remainder(beam200[:k], beam10)

    if len(order_10_200) == 48 and len(order_200_10) == 48:
        mse_10_200 = full_mse(order_10_200)
        mse_200_10 = full_mse(order_200_10)
        print(f"  k={k:2d}: beam10[:{k}]+beam200 MSE={mse_10_200:.6f}  |  "
              f"beam200[:{k}]+beam10 MSE={mse_200_10:.6f}", flush=True)

# ============================================================
# Test 2: Which individual blocks agree between the two beams?
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Test 2: Block agreement between beam-10 and beam-200", flush=True)
print(f"{'='*60}", flush=True)

beam10_set = set(beam10)
beam200_set = set(beam200)
shared_blocks = beam10_set & beam200_set
print(f"Shared (expand,contract) pairs: {len(shared_blocks)}/48", flush=True)
print(f"Shared pairs: {sorted(shared_blocks)}", flush=True)

# Which expand pieces are paired differently?
beam10_pairing = {e: c for e, c in beam10}
beam200_pairing = {e: c for e, c in beam200}
same_pairing = 0
diff_pairing = []
for e in expand_ids:
    if beam10_pairing.get(e) == beam200_pairing.get(e):
        same_pairing += 1
    else:
        diff_pairing.append((e, beam10_pairing.get(e), beam200_pairing.get(e)))
print(f"Same pairing: {same_pairing}/48", flush=True)
print(f"Different pairings:", flush=True)
for e, c10, c200 in diff_pairing:
    print(f"  Expand {e:2d}: beam10→{c10}, beam200→{c200}", flush=True)

# Which expand pieces appear in same relative position?
beam10_pos = {b: i for i, b in enumerate(beam10)}
beam200_pos = {b: i for i, b in enumerate(beam200)}
for b in sorted(shared_blocks, key=lambda x: beam10_pos[x]):
    p10 = beam10_pos[b]
    p200 = beam200_pos[b]
    if abs(p10 - p200) <= 2:
        print(f"  Block {b}: pos {p10} (beam10) vs {p200} (beam200) — CLOSE", flush=True)

# ============================================================
# Test 3: Score using MSE(output, true) proximity to target
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Test 3: MSE(output, true) for both solutions", flush=True)
print(f"{'='*60}", flush=True)

for name, order in [("beam10", beam10), ("beam200", beam200)]:
    mse_p = full_mse(order)
    mse_t = full_mse_true(order)
    h = apply_blocks(X.clone(), order)
    out = h @ w_last + b_last[0]
    corr_pred_true = torch.corrcoef(torch.stack([out - true_vals, pred - true_vals]))[0, 1].item()
    print(f"  {name:8s}: MSE(pred)={mse_p:.6f} MSE(true)={mse_t:.6f} "
          f"|MSE(true)-target|={abs(mse_t - TRUE_MSE):.6f} "
          f"corr(residuals)={corr_pred_true:.4f}", flush=True)

# ============================================================
# Test 4: Greedy beam search scored by BOTH pred and true
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Test 4: Dual-scored greedy (pred MSE + true shape penalty)", flush=True)
print(f"{'='*60}", flush=True)


@torch.no_grad()
def dual_greedy(alpha=0.1):
    """Greedy block selection using MSE(pred) + alpha * |MSE(true) - target|."""
    used_exp = set()
    used_con = set()
    order = []
    h = X.clone()

    for step in range(48):
        best_score = float('inf')
        best_pair = None
        best_h = None

        for ei in expand_ids:
            if ei in used_exp:
                continue
            for ci in contract_ids:
                if ci in used_con:
                    continue
                z = h @ expand_W[ei].T + expand_b[ei]
                z = torch.relu(z)
                z = z @ contract_W[ci].T + contract_b[ci]
                h_new = h + z
                out = h_new @ w_last + b_last[0]
                mse_p = torch.mean((out - pred) ** 2).item()
                mse_t = torch.mean((out - true_vals) ** 2).item()
                combined = mse_p + alpha * abs(mse_t - TRUE_MSE)
                if combined < best_score:
                    best_score = combined
                    best_pair = (ei, ci)
                    best_h = h_new.clone()

        order.append(best_pair)
        used_exp.add(best_pair[0])
        used_con.add(best_pair[1])
        h = best_h

        if step < 5 or step >= 44 or step % 10 == 0:
            out = h @ w_last + b_last[0]
            mse_p = torch.mean((out - pred) ** 2).item()
            print(f"  Step {step:2d}: {best_pair} MSE(pred)={mse_p:.6f}", flush=True)

    return order


for alpha in [0.0, 0.05, 0.1, 0.2, 0.5]:
    print(f"\n  --- alpha={alpha} ---", flush=True)
    order = dual_greedy(alpha=alpha)
    mse = full_mse(order)
    mse_t = full_mse_true(order)
    print(f"  Final: MSE(pred)={mse:.6f} MSE(true)={mse_t:.6f} "
          f"|MSE(true)-target|={abs(mse_t - TRUE_MSE):.6f}", flush=True)

    match, perm_str = check_hash(order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)

# ============================================================
# Test 5: Build ordering that STABILIZES hidden state norm
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Test 5: Norm-aware greedy (minimize MSE + penalize norm growth)", flush=True)
print(f"{'='*60}", flush=True)


@torch.no_grad()
def norm_greedy(beta=0.01):
    """Greedy selection penalizing hidden state norm growth."""
    used_exp = set()
    used_con = set()
    order = []
    h = X.clone()

    for step in range(48):
        best_score = float('inf')
        best_pair = None
        best_h = None

        h_norm_before = torch.mean(h ** 2).item()

        for ei in expand_ids:
            if ei in used_exp:
                continue
            for ci in contract_ids:
                if ci in used_con:
                    continue
                z = h @ expand_W[ei].T + expand_b[ei]
                z = torch.relu(z)
                z = z @ contract_W[ci].T + contract_b[ci]
                h_new = h + z
                out = h_new @ w_last + b_last[0]
                mse_p = torch.mean((out - pred) ** 2).item()
                h_norm_after = torch.mean(h_new ** 2).item()
                norm_growth = max(0, h_norm_after - h_norm_before)
                combined = mse_p + beta * norm_growth
                if combined < best_score:
                    best_score = combined
                    best_pair = (ei, ci)
                    best_h = h_new.clone()

        order.append(best_pair)
        used_exp.add(best_pair[0])
        used_con.add(best_pair[1])
        h = best_h

        if step < 5 or step >= 44 or step % 10 == 0:
            out = h @ w_last + b_last[0]
            mse_p = torch.mean((out - pred) ** 2).item()
            h_norm = torch.mean(h ** 2).item()
            print(f"  Step {step:2d}: {best_pair} MSE={mse_p:.6f} h_norm={h_norm:.1f}", flush=True)

    return order


for beta in [0.0, 0.001, 0.01, 0.1]:
    print(f"\n  --- beta={beta} ---", flush=True)
    order = norm_greedy(beta=beta)
    mse = full_mse(order)
    h_final = apply_blocks(X.clone(), order)
    h_norm = torch.mean(h_final ** 2).item()
    print(f"  Final: MSE={mse:.6f} h_norm={h_norm:.1f}", flush=True)

    match, perm_str = check_hash(order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
