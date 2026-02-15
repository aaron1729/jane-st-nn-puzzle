"""
GPU-accelerated solver using simulated annealing.
Runs on CUDA for fast MSE evaluation of full 48-block pipeline.
"""
import torch
import numpy as np
import csv
import time
import hashlib
import random

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

X = torch.tensor(X_np, device=device)          # [10000, 48]
pred = torch.tensor(pred_np, device=device)     # [10000]

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85

W_last = pieces[last_id]["weight"]  # [1, 48]
b_last = pieces[last_id]["bias"]    # [1]

# Pre-load all weights to GPU
expand_W = {i: pieces[i]["weight"] for i in expand_ids}   # [96, 48]
expand_b = {i: pieces[i]["bias"] for i in expand_ids}     # [96]
contract_W = {i: pieces[i]["weight"] for i in contract_ids}  # [48, 96]
contract_b = {i: pieces[i]["bias"] for i in contract_ids}    # [48]


@torch.no_grad()
def full_mse(block_order):
    """Compute MSE for a full ordering of blocks. GPU-accelerated."""
    h = X.clone()
    for ei, ci in block_order:
        z = h @ expand_W[ei].T + expand_b[ei]    # [N, 96]
        z = torch.relu(z)
        z = z @ contract_W[ci].T + contract_b[ci]  # [N, 48]
        h = h + z
    out = (h @ W_last.T).squeeze() + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def full_mse_from_pos(block_order, start_pos, h_cached):
    """Compute MSE reusing cached hidden state up to start_pos."""
    h = h_cached.clone()
    for idx in range(start_pos, len(block_order)):
        ei, ci = block_order[idx]
        z = h @ expand_W[ei].T + expand_b[ei]
        z = torch.relu(z)
        z = z @ contract_W[ci].T + contract_b[ci]
        h = h + z
    out = (h @ W_last.T).squeeze() + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def compute_hidden_states(block_order):
    """Compute hidden states at every position for cache."""
    states = [X.clone()]
    h = X.clone()
    for ei, ci in block_order:
        z = h @ expand_W[ei].T + expand_b[ei]
        z = torch.relu(z)
        z = z @ contract_W[ci].T + contract_b[ci]
        h = h + z
        states.append(h.clone())
    return states


def check_hash(block_order):
    """Check if a solution matches the target hash."""
    perm = []
    for ei, ci in block_order:
        perm.append(ei)
        perm.append(ci)
    perm.append(last_id)
    perm_str = ",".join(str(p) for p in perm)
    h = hashlib.sha256(perm_str.encode()).hexdigest()
    target = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"
    return h == target, perm_str


# Beam search first 40 pairs (from previous experiments)
beam_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
]
beam_tail = [(42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57)]

# Local search result (best so far from check_pairings.py)
local_search_best = [
    (2, 66), (87, 71), (3, 53), (73, 72), (49, 93), (43, 11), (68, 26), (58, 78),
    (81, 51), (95, 33), (13, 75), (94, 55), (5, 20), (60, 29), (10, 21), (37, 40),
    (15, 9), (4, 19), (16, 54), (28, 47), (48, 38), (35, 12), (50, 89), (74, 96),
    (18, 6), (61, 30), (0, 76), (59, 79), (69, 52), (44, 70), (64, 83), (45, 32),
    (41, 46), (39, 90), (84, 63), (91, 34), (42, 92), (56, 80), (88, 22), (65, 8),
    (62, 25), (27, 67), (1, 24), (77, 7), (31, 36), (14, 17), (86, 82), (23, 57),
]


def simulated_annealing(initial_order, max_iters=500000, T_start=0.05, T_end=1e-6,
                        seed=None, label="SA"):
    """Simulated annealing with position swaps, pairing swaps, segment reversals."""
    if seed is not None:
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
    start_time = time.time()

    for it in range(max_iters):
        # Choose move type
        r = random.random()
        if r < 0.4:
            # Position swap
            i, j = random.sample(range(n), 2)
            new_order = list(current)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif r < 0.7:
            # Pairing swap (swap contract pieces between two blocks)
            i, j = random.sample(range(n), 2)
            ei_i, ci_i = current[i]
            ei_j, ci_j = current[j]
            new_order = list(current)
            new_order[i] = (ei_i, ci_j)
            new_order[j] = (ei_j, ci_i)
        elif r < 0.85:
            # Segment reversal
            i, j = sorted(random.sample(range(n), 2))
            new_order = list(current)
            new_order[i:j+1] = reversed(new_order[i:j+1])
        else:
            # Random block insertion (remove and reinsert)
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            new_order = list(current)
            block = new_order.pop(i)
            new_order.insert(j, block)

        new_mse = full_mse(new_order)
        delta = new_mse - current_mse

        if delta < 0 or random.random() < np.exp(-delta / T):
            current = new_order
            current_mse = new_mse
            accepted += 1

            if current_mse < best_mse:
                best = list(current)
                best_mse = current_mse
                improved += 1

                # Check hash
                match, perm_str = check_hash(best)
                if match:
                    print(f"\n*** SOLUTION FOUND! ***", flush=True)
                    print(f"Permutation: {perm_str}", flush=True)
                    return best, best_mse

        T *= cooling

        if (it + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (it + 1) / elapsed
            print(f"  [{label}] iter {it+1}/{max_iters}: T={T:.6f} current={current_mse:.6f} "
                  f"best={best_mse:.6f} accepted={accepted} improved={improved} "
                  f"({rate:.0f} it/s)", flush=True)

    elapsed = time.time() - start_time
    print(f"  [{label}] Done in {elapsed:.1f}s. Best MSE={best_mse:.6f} "
          f"(accepted={accepted}, improved={improved})", flush=True)
    return best, best_mse


print("=" * 60, flush=True)
print("Phase 1: Simulated annealing from beam + local search result", flush=True)
print("=" * 60, flush=True)

# Verify starting points
beam_full = beam_pairs + beam_tail
beam_mse = full_mse(beam_full)
local_mse = full_mse(local_search_best)
print(f"Beam solution MSE: {beam_mse:.6f}", flush=True)
print(f"Local search solution MSE: {local_mse:.6f}", flush=True)

# Run SA from the better starting point
start_order = local_search_best if local_mse < beam_mse else beam_full
print(f"\nStarting SA from {'local search' if local_mse < beam_mse else 'beam'} solution "
      f"(MSE={min(local_mse, beam_mse):.6f})", flush=True)

# First run: moderate temperature, lots of iterations
best_order, best_mse = simulated_annealing(
    start_order, max_iters=200000, T_start=0.1, T_end=1e-7, seed=42, label="SA1"
)
print(f"\nSA1 result: MSE={best_mse:.6f}", flush=True)

# Second run: from SA1 result, lower temperature for fine-tuning
best_order2, best_mse2 = simulated_annealing(
    best_order, max_iters=200000, T_start=0.01, T_end=1e-8, seed=123, label="SA2"
)
print(f"\nSA2 result: MSE={best_mse2:.6f}", flush=True)

# Third run: from beam solution with higher temperature to explore more
best_order3, best_mse3 = simulated_annealing(
    beam_full, max_iters=200000, T_start=0.5, T_end=1e-6, seed=456, label="SA3"
)
print(f"\nSA3 result: MSE={best_mse3:.6f}", flush=True)

# Pick overall best
results = [(best_mse, best_order, "SA1->SA2"), (best_mse2, best_order2, "SA2"),
           (best_mse3, best_order3, "SA3")]
results.sort(key=lambda x: x[0])
overall_best_mse, overall_best, label = results[0]

print(f"\n{'='*60}", flush=True)
print(f"Overall best: {label} MSE={overall_best_mse:.10f}", flush=True)
print(f"Ordering: {overall_best}", flush=True)

match, perm_str = check_hash(overall_best)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)

# If not solved, do more aggressive search with random restarts
if not match and overall_best_mse > 0.01:
    print(f"\n{'='*60}", flush=True)
    print("Phase 2: Random restarts with SA", flush=True)
    print(f"{'='*60}", flush=True)

    global_best_mse = overall_best_mse
    global_best = list(overall_best)

    for restart in range(20):
        # Random permutation of blocks
        order = list(beam_full)
        random.shuffle(order)
        # Also randomly re-pair some blocks
        if random.random() < 0.5:
            indices = random.sample(range(48), k=random.randint(2, 16))
            contract_pieces = [order[i][1] for i in indices]
            random.shuffle(contract_pieces)
            for k, idx in enumerate(indices):
                order[idx] = (order[idx][0], contract_pieces[k])

        restart_best, restart_mse = simulated_annealing(
            order, max_iters=100000, T_start=0.3, T_end=1e-7,
            seed=1000 + restart, label=f"R{restart}"
        )

        if restart_mse < global_best_mse:
            global_best_mse = restart_mse
            global_best = list(restart_best)
            print(f"  *** New global best: MSE={global_best_mse:.6f}", flush=True)

            match, perm_str = check_hash(global_best)
            if match:
                print(f"\n*** SOLUTION FOUND! ***", flush=True)
                print(f"Permutation: {perm_str}", flush=True)
                break

    print(f"\nFinal global best MSE: {global_best_mse:.10f}", flush=True)
    print(f"Final ordering: {global_best}", flush=True)
    match, perm_str = check_hash(global_best)
    print(f"Hash match: {match}", flush=True)
    print(f"Permutation: {perm_str}", flush=True)
