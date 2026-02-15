"""
Sliding window optimization: brute-force reorder blocks within small windows.
For each window of W=8 blocks, try all W! orderings.
Uses cached hidden states for efficiency.
"""
import torch
import numpy as np
import csv
import time
import hashlib
import itertools
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
def apply_blocks(h, block_list):
    for ei, ci in block_list:
        h = apply_block(h, ei, ci)
    return h


@torch.no_grad()
def score(h):
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def full_mse(block_order):
    h = apply_blocks(X.clone(), block_order)
    return score(h)


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


# Best known solution from SA (MSE ~0.327)
# Using the beam+local search starting point; SA will improve it further
beam_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
]
beam_tail = [(42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57)]
current_order = beam_pairs + beam_tail

print(f"Starting MSE: {full_mse(current_order):.6f}", flush=True)

# ============================================================
# Phase 1: Sliding window ordering optimization (W=8)
# For each window, try all W! orderings
# ============================================================
WINDOW = 8
improved_overall = True
sweep = 0

while improved_overall:
    improved_overall = False
    sweep += 1
    sweep_start = time.time()
    print(f"\n--- Ordering sweep {sweep} (window={WINDOW}) ---", flush=True)

    for start in range(0, 48 - WINDOW + 1):
        # Cache hidden state before window
        h_pre = apply_blocks(X.clone(), current_order[:start])

        # Blocks in the window
        window_blocks = current_order[start:start + WINDOW]
        # Suffix blocks (after window)
        suffix_blocks = current_order[start + WINDOW:]

        best_window_mse = full_mse(current_order)
        best_window_order = list(window_blocks)

        # Try all permutations of window blocks
        count = 0
        for perm in itertools.permutations(range(WINDOW)):
            reordered = [window_blocks[i] for i in perm]
            h = apply_blocks(h_pre.clone(), reordered)
            h = apply_blocks(h, suffix_blocks)
            mse = score(h)
            count += 1

            if mse < best_window_mse:
                best_window_mse = mse
                best_window_order = reordered

        if best_window_order != list(window_blocks):
            current_order = current_order[:start] + best_window_order + suffix_blocks
            improved_overall = True
            print(f"  Window [{start}:{start+WINDOW}]: MSE={best_window_mse:.6f} "
                  f"(improved, tried {count})", flush=True)

    sweep_time = time.time() - sweep_start
    current_mse = full_mse(current_order)
    print(f"Sweep {sweep} done ({sweep_time:.1f}s): MSE={current_mse:.6f}", flush=True)

    # Check hash
    match, perm_str = check_hash(current_order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
        break

print(f"\nAfter ordering optimization: MSE={full_mse(current_order):.6f}", flush=True)

# ============================================================
# Phase 2: Sliding window PAIRING optimization (W=6)
# For each window, try all W! pairings of contract pieces
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: Sliding window pairing optimization (W=6)", flush=True)
print(f"{'='*60}", flush=True)

PAIR_WINDOW = 6
improved_overall = True
sweep = 0

while improved_overall:
    improved_overall = False
    sweep += 1
    sweep_start = time.time()
    print(f"\n--- Pairing sweep {sweep} (window={PAIR_WINDOW}) ---", flush=True)

    for start in range(0, 48 - PAIR_WINDOW + 1):
        h_pre = apply_blocks(X.clone(), current_order[:start])
        window_blocks = current_order[start:start + PAIR_WINDOW]
        suffix_blocks = current_order[start + PAIR_WINDOW:]

        expand_pieces = [b[0] for b in window_blocks]
        contract_pieces = [b[1] for b in window_blocks]

        best_mse = full_mse(current_order)
        best_pairing = list(contract_pieces)

        for perm in itertools.permutations(range(PAIR_WINDOW)):
            repairing = [(expand_pieces[i], contract_pieces[perm[i]]) for i in range(PAIR_WINDOW)]
            h = apply_blocks(h_pre.clone(), repairing)
            h = apply_blocks(h, suffix_blocks)
            mse = score(h)

            if mse < best_mse:
                best_mse = mse
                best_pairing = [contract_pieces[perm[i]] for i in range(PAIR_WINDOW)]

        new_window = [(expand_pieces[i], best_pairing[i]) for i in range(PAIR_WINDOW)]
        if new_window != window_blocks:
            current_order = current_order[:start] + new_window + suffix_blocks
            improved_overall = True
            print(f"  Window [{start}:{start+PAIR_WINDOW}]: MSE={best_mse:.6f} (improved)", flush=True)

    sweep_time = time.time() - sweep_start
    current_mse = full_mse(current_order)
    print(f"Pairing sweep {sweep} done ({sweep_time:.1f}s): MSE={current_mse:.6f}", flush=True)

    match, perm_str = check_hash(current_order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
        break

# ============================================================
# Phase 3: Alternating ordering and pairing sweeps
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Phase 3: Alternating optimization", flush=True)
print(f"{'='*60}", flush=True)

for mega_iter in range(5):
    prev_mse = full_mse(current_order)
    print(f"\nMega iteration {mega_iter+1}, starting MSE={prev_mse:.6f}", flush=True)

    # Ordering sweep
    improved_inner = True
    while improved_inner:
        improved_inner = False
        for start in range(0, 48 - WINDOW + 1):
            h_pre = apply_blocks(X.clone(), current_order[:start])
            window_blocks = current_order[start:start + WINDOW]
            suffix_blocks = current_order[start + WINDOW:]
            best_mse = full_mse(current_order)
            best_window = list(window_blocks)
            for perm in itertools.permutations(range(WINDOW)):
                reordered = [window_blocks[i] for i in perm]
                h = apply_blocks(h_pre.clone(), reordered + suffix_blocks)
                mse = score(h)
                if mse < best_mse:
                    best_mse = mse
                    best_window = reordered
            if best_window != list(window_blocks):
                current_order = current_order[:start] + best_window + suffix_blocks
                improved_inner = True

    # Pairing sweep
    improved_inner = True
    while improved_inner:
        improved_inner = False
        for start in range(0, 48 - PAIR_WINDOW + 1):
            h_pre = apply_blocks(X.clone(), current_order[:start])
            window_blocks = current_order[start:start + PAIR_WINDOW]
            suffix_blocks = current_order[start + PAIR_WINDOW:]
            exps = [b[0] for b in window_blocks]
            cons = [b[1] for b in window_blocks]
            best_mse = full_mse(current_order)
            best_cons = list(cons)
            for perm in itertools.permutations(range(PAIR_WINDOW)):
                new_pairs = [(exps[i], cons[perm[i]]) for i in range(PAIR_WINDOW)]
                h = apply_blocks(h_pre.clone(), new_pairs + suffix_blocks)
                mse = score(h)
                if mse < best_mse:
                    best_mse = mse
                    best_cons = [cons[perm[i]] for i in range(PAIR_WINDOW)]
            new_window = [(exps[i], best_cons[i]) for i in range(PAIR_WINDOW)]
            if new_window != window_blocks:
                current_order = current_order[:start] + new_window + suffix_blocks
                improved_inner = True

    new_mse = full_mse(current_order)
    print(f"After mega iteration {mega_iter+1}: MSE={new_mse:.6f}", flush=True)

    match, perm_str = check_hash(current_order)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)
        break

    if abs(new_mse - prev_mse) < 1e-8:
        print("Converged.", flush=True)
        break

# Final output
final_mse = full_mse(current_order)
print(f"\nFinal MSE: {final_mse:.10f}", flush=True)
print(f"Final ordering: {current_order}", flush=True)
match, perm_str = check_hash(current_order)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)
