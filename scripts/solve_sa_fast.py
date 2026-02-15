"""
Fast simulated annealing with subsampled scoring + wide beam search.
Uses 1000 data points for SA (10x faster), validates with full 10k.
Then runs wide beam search.
"""
import torch
import numpy as np
import csv
import time
import hashlib
import random
import json

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

X_full = torch.tensor(X_np, device=device)
pred_full = torch.tensor(pred_np, device=device)

# Subsample for fast scoring
SUB_N = 1000
rng = np.random.RandomState(0)
sub_idx = rng.choice(len(X_np), SUB_N, replace=False)
X_sub = torch.tensor(X_np[sub_idx], device=device)
pred_sub = torch.tensor(pred_np[sub_idx], device=device)

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

expand_W_stack = torch.stack([pieces[i]["weight"] for i in expand_ids])
expand_b_stack = torch.stack([pieces[i]["bias"] for i in expand_ids])
contract_W_stack = torch.stack([pieces[i]["weight"] for i in contract_ids])
contract_b_stack = torch.stack([pieces[i]["bias"] for i in contract_ids])

ei2idx = {eid: i for i, eid in enumerate(expand_ids)}
ci2idx = {cid: j for j, cid in enumerate(contract_ids)}


@torch.no_grad()
def full_mse(block_order, X=X_full, pred=pred_full):
    h = X.clone()
    for ei, ci in block_order:
        z = h @ expand_W_stack[ei2idx[ei]].T + expand_b_stack[ei2idx[ei]]
        z = torch.relu(z)
        z = z @ contract_W_stack[ci2idx[ci]].T + contract_b_stack[ci2idx[ci]]
        h = h + z
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


def fast_mse(block_order):
    return full_mse(block_order, X_sub, pred_sub)


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


# Starting solutions
beam_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
]
beam_tail = [(42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57)]
beam_full = beam_pairs + beam_tail

local_search_best = [
    (2, 66), (87, 71), (3, 53), (73, 72), (49, 93), (43, 11), (68, 26), (58, 78),
    (81, 51), (95, 33), (13, 75), (94, 55), (5, 20), (60, 29), (10, 21), (37, 40),
    (15, 9), (4, 19), (16, 54), (28, 47), (48, 38), (35, 12), (50, 89), (74, 96),
    (18, 6), (61, 30), (0, 76), (59, 79), (69, 52), (44, 70), (64, 83), (45, 32),
    (41, 46), (39, 90), (84, 63), (91, 34), (42, 92), (56, 80), (88, 22), (65, 8),
    (62, 25), (27, 67), (1, 24), (77, 7), (31, 36), (14, 17), (86, 82), (23, 57),
]


def sa_run(initial_order, max_iters, T_start, T_end, seed, label, use_fast=True):
    random.seed(seed)
    np.random.seed(seed)

    mse_fn = fast_mse if use_fast else full_mse
    current = list(initial_order)
    current_mse = mse_fn(current)
    best = list(current)
    best_mse = current_mse

    n = len(current)
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iters)

    accepted = 0
    improved = 0
    start = time.time()

    for it in range(max_iters):
        r = random.random()
        if r < 0.4:
            i, j = random.sample(range(n), 2)
            new_order = list(current)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif r < 0.7:
            i, j = random.sample(range(n), 2)
            ei_i, ci_i = current[i]
            ei_j, ci_j = current[j]
            new_order = list(current)
            new_order[i] = (ei_i, ci_j)
            new_order[j] = (ei_j, ci_i)
        elif r < 0.85:
            i, j = sorted(random.sample(range(n), 2))
            new_order = list(current)
            new_order[i:j+1] = reversed(new_order[i:j+1])
        else:
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            new_order = list(current)
            block = new_order.pop(i)
            new_order.insert(j, block)

        new_mse = mse_fn(new_order)
        delta = new_mse - current_mse

        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-30)):
            current = new_order
            current_mse = new_mse
            accepted += 1
            if current_mse < best_mse:
                best = list(current)
                best_mse = current_mse
                improved += 1
                match, perm_str = check_hash(best)
                if match:
                    print(f"\n*** SOLUTION FOUND! ***", flush=True)
                    print(f"Permutation: {perm_str}", flush=True)
                    return best, best_mse

        T *= cooling

        if (it + 1) % 50000 == 0:
            elapsed = time.time() - start
            rate = (it + 1) / elapsed
            full_best = full_mse(best) if use_fast else best_mse
            print(f"  [{label}] iter {it+1}/{max_iters}: T={T:.2e} curr={current_mse:.6f} "
                  f"best_sub={best_mse:.6f} best_full={full_best:.6f} "
                  f"accepted={accepted} improved={improved} ({rate:.0f} it/s)", flush=True)

    elapsed = time.time() - start
    full_best = full_mse(best) if use_fast else best_mse
    print(f"  [{label}] Done {elapsed:.1f}s. best_sub={best_mse:.6f} "
          f"best_full={full_best:.6f}", flush=True)
    return best, full_best


# ========================================
# Phase 1: Fast SA from multiple starts
# ========================================
print("=" * 60, flush=True)
print("Phase 1: Fast SA (subsampled, 1000 pts) from beam + local", flush=True)
print("=" * 60, flush=True)

beam_mse = full_mse(beam_full)
local_mse = full_mse(local_search_best)
print(f"Beam MSE (full): {beam_mse:.6f}", flush=True)
print(f"Local search MSE (full): {local_mse:.6f}", flush=True)

global_best_mse = min(beam_mse, local_mse)
global_best = list(local_search_best if local_mse < beam_mse else beam_full)

# Run fast SA from local search best (1M iters with subsampling)
print("\nSA from local search result (subsampled)...", flush=True)
best1, mse1 = sa_run(local_search_best, 1000000, 0.1, 1e-8, 42, "SA-fast-1")
if mse1 < global_best_mse:
    global_best_mse = mse1
    global_best = list(best1)
    print(f"  New global best: {global_best_mse:.6f}", flush=True)

# Run fast SA from beam solution
print("\nSA from beam solution (subsampled)...", flush=True)
best2, mse2 = sa_run(beam_full, 1000000, 0.5, 1e-8, 123, "SA-fast-2")
if mse2 < global_best_mse:
    global_best_mse = mse2
    global_best = list(best2)
    print(f"  New global best: {global_best_mse:.6f}", flush=True)

# Run from current global best with low temperature
print("\nSA refinement from best so far (subsampled)...", flush=True)
best3, mse3 = sa_run(global_best, 1000000, 0.01, 1e-9, 456, "SA-refine")
if mse3 < global_best_mse:
    global_best_mse = mse3
    global_best = list(best3)
    print(f"  New global best: {global_best_mse:.6f}", flush=True)

# ========================================
# Phase 2: Random restarts with fast SA
# ========================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: Random restarts (20 runs)", flush=True)
print(f"{'='*60}", flush=True)

for restart in range(20):
    # Create random starting point: shuffle beam's blocks + randomize some pairings
    order = list(beam_full)
    random.seed(2000 + restart)
    random.shuffle(order)
    # Randomly re-pair a subset
    n_repair = random.randint(4, 24)
    indices = random.sample(range(48), n_repair)
    contracts = [order[i][1] for i in indices]
    random.shuffle(contracts)
    for k, idx in enumerate(indices):
        order[idx] = (order[idx][0], contracts[k])

    best_r, mse_r = sa_run(order, 500000, 0.3, 1e-8, 3000 + restart, f"R{restart}")
    if mse_r < global_best_mse:
        global_best_mse = mse_r
        global_best = list(best_r)
        print(f"  *** New global best from R{restart}: {global_best_mse:.6f}", flush=True)
        match, perm_str = check_hash(global_best)
        if match:
            print(f"\n*** SOLUTION FOUND! ***", flush=True)
            print(f"Permutation: {perm_str}", flush=True)
            break

# ========================================
# Phase 3: Final refinement with full data
# ========================================
print(f"\n{'='*60}", flush=True)
print("Phase 3: Final refinement (full 10k data)", flush=True)
print(f"{'='*60}", flush=True)

best_final, mse_final = sa_run(global_best, 200000, 0.005, 1e-9, 999, "Final",
                                use_fast=False)
print(f"\nFinal MSE: {mse_final:.10f}", flush=True)
print(f"Final ordering: {best_final}", flush=True)
match, perm_str = check_hash(best_final)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)
