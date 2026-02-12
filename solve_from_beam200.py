"""
Take the width-200 beam search result and optimize it with SA + window optimization.
"""
import torch
import numpy as np
import csv
import time
import hashlib
import random
import itertools

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
def apply_blocks(h, block_list):
    for ei, ci in block_list:
        h = apply_block(h, ei, ci)
    return h


@torch.no_grad()
def full_mse(block_order):
    h = apply_blocks(X.clone(), block_order)
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


# Width-200 beam best result
beam200_best = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (39, 32), (4, 52), (45, 33), (3, 40), (2, 70), (68, 47), (59, 92),
    (61, 83), (15, 66), (35, 22), (16, 90), (91, 30), (56, 21), (42, 46), (10, 20),
    (13, 34), (1, 12), (18, 63), (28, 25), (74, 80), (44, 7), (86, 76), (69, 89),
    (14, 8), (43, 53), (84, 96), (95, 79), (88, 38), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (62, 57), (65, 24), (77, 82),
]

# Width-10 beam best result
beam10_best = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

print(f"Beam-200 MSE: {full_mse(beam200_best):.6f}", flush=True)
print(f"Beam-10 MSE: {full_mse(beam10_best):.6f}", flush=True)


# ============================================================
# Phase 1: SA from beam-200 best
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Phase 1: Simulated Annealing from beam-200 result", flush=True)
print(f"{'='*60}", flush=True)

def sa_run(initial_order, max_iters, T_start, T_end, seed, label):
    random.seed(seed)
    np.random.seed(seed)

    current = list(initial_order)
    current_mse = full_mse(current)
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
            new_order = list(current)
            new_order[i] = (current[i][0], current[j][1])
            new_order[j] = (current[j][0], current[i][1])
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

        new_mse = full_mse(new_order)
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

        if (it + 1) % 10000 == 0:
            elapsed = time.time() - start
            rate = (it + 1) / elapsed
            print(f"  [{label}] it={it+1}/{max_iters} T={T:.2e} "
                  f"curr={current_mse:.6f} best={best_mse:.6f} "
                  f"acc={accepted} imp={improved} ({rate:.0f}/s)", flush=True)

    return best, best_mse


best_sa, mse_sa = sa_run(beam200_best, 200000, 0.1, 1e-7, 42, "SA-200")
print(f"SA result: MSE={mse_sa:.6f}", flush=True)

# Second SA run with more exploration
best_sa2, mse_sa2 = sa_run(best_sa, 200000, 0.01, 1e-8, 123, "SA-refine")
print(f"SA refined: MSE={mse_sa2:.6f}", flush=True)

# Also try from beam-10
best_sa3, mse_sa3 = sa_run(beam10_best, 200000, 0.1, 1e-7, 456, "SA-10")
print(f"SA from beam-10: MSE={mse_sa3:.6f}", flush=True)

# Pick the best
results = [(mse_sa2, best_sa2, "SA-200-refined"), (mse_sa3, best_sa3, "SA-10")]
results.sort(key=lambda x: x[0])
best_mse_overall, best_order, label = results[0]
print(f"\nBest overall: {label} MSE={best_mse_overall:.6f}", flush=True)

# ============================================================
# Phase 2: Window optimization on best SA result
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: Window optimization (W=8)", flush=True)
print(f"{'='*60}", flush=True)

current_order = list(best_order)
WINDOW = 8
improved_overall = True
sweep = 0

while improved_overall and sweep < 5:
    improved_overall = False
    sweep += 1
    print(f"\n--- Sweep {sweep} ---", flush=True)

    # Ordering sweep
    for start in range(0, 48 - WINDOW + 1):
        h_pre = apply_blocks(X.clone(), current_order[:start])
        window_blocks = current_order[start:start + WINDOW]
        suffix_blocks = current_order[start + WINDOW:]
        base_mse = full_mse(current_order)
        best_window = list(window_blocks)
        best_mse = base_mse

        for perm in itertools.permutations(range(WINDOW)):
            reordered = [window_blocks[i] for i in perm]
            h = apply_blocks(h_pre.clone(), reordered + suffix_blocks)
            out = h @ w_last + b_last[0]
            mse = torch.mean((out - pred) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_window = reordered

        if best_window != list(window_blocks):
            current_order = current_order[:start] + best_window + suffix_blocks
            improved_overall = True
            print(f"  Ordering [{start}:{start+WINDOW}]: {base_mse:.6f} -> {best_mse:.6f}", flush=True)

    current_mse = full_mse(current_order)
    print(f"  After ordering: MSE={current_mse:.6f}", flush=True)

    # Pairing sweep (W=6)
    PAIR_W = 6
    for start in range(0, 48 - PAIR_W + 1):
        h_pre = apply_blocks(X.clone(), current_order[:start])
        window_blocks = current_order[start:start + PAIR_W]
        suffix_blocks = current_order[start + PAIR_W:]
        exps = [b[0] for b in window_blocks]
        cons = [b[1] for b in window_blocks]
        base_mse = full_mse(current_order)
        best_mse = base_mse
        best_cons = list(cons)

        for perm in itertools.permutations(range(PAIR_W)):
            new_pairs = [(exps[i], cons[perm[i]]) for i in range(PAIR_W)]
            h = apply_blocks(h_pre.clone(), new_pairs + suffix_blocks)
            out = h @ w_last + b_last[0]
            mse = torch.mean((out - pred) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_cons = [cons[perm[i]] for i in range(PAIR_W)]

        new_window = [(exps[i], best_cons[i]) for i in range(PAIR_W)]
        if new_window != window_blocks:
            current_order = current_order[:start] + new_window + suffix_blocks
            improved_overall = True
            print(f"  Pairing [{start}:{start+PAIR_W}]: {base_mse:.6f} -> {best_mse:.6f}", flush=True)

    current_mse = full_mse(current_order)
    print(f"  After pairing: MSE={current_mse:.6f}", flush=True)

    match, perm_str = check_hash(current_order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
        break

# Final SA polish
print(f"\n{'='*60}", flush=True)
print("Phase 3: Final SA polish", flush=True)
print(f"{'='*60}", flush=True)

best_final, mse_final = sa_run(current_order, 200000, 0.005, 1e-9, 789, "Final")
print(f"\nFinal MSE: {mse_final:.10f}", flush=True)
print(f"Final ordering: {best_final}", flush=True)
match, perm_str = check_hash(best_final)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)
