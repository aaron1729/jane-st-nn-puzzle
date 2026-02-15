"""
Two-level optimization: SA over pairings, fast greedy for ordering.
Key insight: separate the pairing search from the ordering search.
For each pairing, greedy ordering is fast (batched GPU eval).
SA explores the pairing space by swapping contract pieces.
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

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85
w_last = pieces[last_id]["weight"].squeeze(0)
b_last = pieces[last_id]["bias"]

# Stack weights for batched evaluation
exp_W_stack = torch.stack([pieces[i]["weight"] for i in expand_ids])  # [48, 96, 48]
exp_b_stack = torch.stack([pieces[i]["bias"] for i in expand_ids])    # [48, 96]
con_W_stack = torch.stack([pieces[i]["weight"] for i in contract_ids])  # [48, 48, 96]
con_b_stack = torch.stack([pieces[i]["bias"] for i in contract_ids])    # [48, 48]

# Map piece IDs to stack indices
exp_id_to_idx = {eid: idx for idx, eid in enumerate(expand_ids)}
con_id_to_idx = {cid: idx for idx, cid in enumerate(contract_ids)}


@torch.no_grad()
def apply_block(h, ei, ci):
    z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
    z = torch.relu(z)
    z = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
    return h + z


@torch.no_grad()
def full_mse(block_order):
    h = X.clone()
    for ei, ci in block_order:
        h = apply_block(h, ei, ci)
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def greedy_order(pairs):
    """Given a list of (expand_id, contract_id) pairs, find greedy ordering.
    At each step, pick the block that reduces MSE the most.
    Returns (ordering, final_mse)."""
    remaining = list(range(len(pairs)))
    order = []
    h = X.clone()

    for step in range(len(pairs)):
        best_mse = float('inf')
        best_idx = -1
        best_h = None

        # Evaluate all remaining blocks
        for idx in remaining:
            ei, ci = pairs[idx]
            z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
            z = torch.relu(z)
            z = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
            h_new = h + z
            out = h_new @ w_last + b_last[0]
            mse = torch.mean((out - pred) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_idx = idx
                best_h = h_new

        order.append(pairs[best_idx])
        remaining.remove(best_idx)
        h = best_h

    # Final MSE
    out = h @ w_last + b_last[0]
    final_mse = torch.mean((out - pred) ** 2).item()
    return order, final_mse


@torch.no_grad()
def greedy_order_fast(pairs):
    """Faster greedy using pre-indexed weights."""
    remaining = list(range(len(pairs)))
    order = []
    h = X.clone()

    for step in range(len(pairs)):
        best_mse = float('inf')
        best_idx = -1
        best_h = None

        for idx in remaining:
            ei, ci = pairs[idx]
            z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
            z = torch.relu(z)
            z = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
            h_new = h + z
            out = h_new @ w_last + b_last[0]
            mse = torch.mean((out - pred) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_idx = idx
                best_h = h_new.clone()

        order.append(pairs[best_idx])
        remaining.remove(best_idx)
        h = best_h

    out = h @ w_last + b_last[0]
    return order, torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def beam_order(pairs, width=10):
    """Beam search ordering for given pairs. Returns (ordering, mse)."""
    # beam entries: (h, mse, used_set, order_list)
    beam = [(X.clone(), None, set(), [])]
    n = len(pairs)

    for step in range(n):
        candidates = []
        for h, _, used, order in beam:
            for idx in range(n):
                if idx in used:
                    continue
                ei, ci = pairs[idx]
                z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
                z = torch.relu(z)
                z = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
                h_new = h + z
                out = h_new @ w_last + b_last[0]
                mse = torch.mean((out - pred) ** 2).item()
                new_used = used | {idx}
                candidates.append((h_new, mse, new_used, order + [pairs[idx]]))

        # Keep top-width by MSE
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:width]

    return beam[0][3], beam[0][1]


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


# ============================================================
# Starting pairings from our two beam results
# ============================================================
beam10_pairs = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

beam200_pairs = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (39, 32), (4, 52), (45, 33), (3, 40), (2, 70), (68, 47), (59, 92),
    (61, 83), (15, 66), (35, 22), (16, 90), (91, 30), (56, 21), (42, 46), (10, 20),
    (13, 34), (1, 12), (18, 63), (28, 25), (74, 80), (44, 7), (86, 76), (69, 89),
    (14, 8), (43, 53), (84, 96), (95, 79), (88, 38), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (62, 57), (65, 24), (77, 82),
]

# Test: greedy ordering for each pairing
print("Testing greedy ordering for starting pairings...", flush=True)
t0 = time.time()
order10, mse10 = greedy_order_fast(beam10_pairs)
t1 = time.time()
print(f"Beam-10 pairings + greedy order: MSE={mse10:.6f} ({t1-t0:.1f}s)", flush=True)

order200, mse200 = greedy_order_fast(beam200_pairs)
t2 = time.time()
print(f"Beam-200 pairings + greedy order: MSE={mse200:.6f} ({t2-t1:.1f}s)", flush=True)

# Also test: full_mse of the original orderings
print(f"Beam-10 original ordering: MSE={full_mse(beam10_pairs):.6f}", flush=True)
print(f"Beam-200 original ordering: MSE={full_mse(beam200_pairs):.6f}", flush=True)

# ============================================================
# SA over pairings
# ============================================================
print(f"\n{'='*60}", flush=True)
print("SA over pairings (greedy ordering for each trial)", flush=True)
print(f"{'='*60}", flush=True)


def pairing_sa(initial_pairs, max_iters=5000, T_start=0.2, T_end=1e-5,
               seed=42, label="PSA", use_beam=False, beam_width=5):
    random.seed(seed)
    np.random.seed(seed)

    # Current pairing as dict: expand_id -> contract_id
    pairing = {e: c for e, c in initial_pairs}
    exp_list = [e for e, c in initial_pairs]

    # Evaluate initial
    pairs = [(e, pairing[e]) for e in exp_list]
    if use_beam:
        current_order, current_mse = beam_order(pairs, width=beam_width)
    else:
        current_order, current_mse = greedy_order_fast(pairs)
    best_mse = current_mse
    best_order = list(current_order)
    best_pairing = dict(pairing)

    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    accepted = 0
    improved = 0
    start_time = time.time()

    for it in range(max_iters):
        # Mutate pairing: swap contract pieces between 2 or 3 expand pieces
        r = random.random()
        new_pairing = dict(pairing)
        if r < 0.6:
            # 2-way swap
            e1, e2 = random.sample(exp_list, 2)
            new_pairing[e1], new_pairing[e2] = new_pairing[e2], new_pairing[e1]
        elif r < 0.85:
            # 3-way rotation
            e1, e2, e3 = random.sample(exp_list, 3)
            c1, c2, c3 = new_pairing[e1], new_pairing[e2], new_pairing[e3]
            new_pairing[e1], new_pairing[e2], new_pairing[e3] = c2, c3, c1
        else:
            # 4-way rotation
            indices = random.sample(exp_list, 4)
            contracts = [new_pairing[e] for e in indices]
            # Rotate
            contracts = contracts[1:] + contracts[:1]
            for e, c in zip(indices, contracts):
                new_pairing[e] = c

        # Evaluate new pairing
        new_pairs = [(e, new_pairing[e]) for e in exp_list]
        if use_beam:
            new_order, new_mse = beam_order(new_pairs, width=beam_width)
        else:
            new_order, new_mse = greedy_order_fast(new_pairs)

        delta = new_mse - current_mse

        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-30)):
            pairing = new_pairing
            current_mse = new_mse
            current_order = new_order
            accepted += 1

            if current_mse < best_mse:
                best_mse = current_mse
                best_order = list(current_order)
                best_pairing = dict(pairing)
                improved += 1

                match, perm_str = check_hash(best_order)
                if match:
                    print(f"\n*** SOLUTION FOUND at iter {it}! ***", flush=True)
                    print(f"Permutation: {perm_str}", flush=True)
                    with open("solution.json", "w") as f:
                        json.dump({"perm": perm_str, "order": best_order}, f)
                    return best_order, best_mse, best_pairing

        T *= cooling

        if (it + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (it + 1) / elapsed
            print(f"  [{label}] it={it+1}/{max_iters} T={T:.2e} "
                  f"curr={current_mse:.6f} best={best_mse:.6f} "
                  f"acc={accepted} imp={improved} ({rate:.1f}/s)", flush=True)

    return best_order, best_mse, best_pairing


# Run SA over pairings from beam-200 (our better starting point)
print("\n--- SA from beam-200 pairings (greedy ordering) ---", flush=True)
best_order, best_mse, best_pairing = pairing_sa(
    beam200_pairs, max_iters=3000, T_start=0.3, T_end=1e-5, seed=42, label="PSA-200"
)
print(f"\nBest from beam-200 pairings: MSE={best_mse:.6f}", flush=True)

# Run SA from beam-10 pairings
print("\n--- SA from beam-10 pairings (greedy ordering) ---", flush=True)
best_order2, best_mse2, best_pairing2 = pairing_sa(
    beam10_pairs, max_iters=3000, T_start=0.3, T_end=1e-5, seed=123, label="PSA-10"
)
print(f"\nBest from beam-10 pairings: MSE={best_mse2:.6f}", flush=True)

# Run SA from random pairings (explore completely new territory)
print("\n--- SA from random pairings ---", flush=True)
random.seed(999)
rand_pairs = list(zip(expand_ids, contract_ids))
random.shuffle(rand_pairs)
# Randomly pair
cons = list(contract_ids)
random.shuffle(cons)
rand_pairs = list(zip(expand_ids, cons))

best_order3, best_mse3, best_pairing3 = pairing_sa(
    rand_pairs, max_iters=3000, T_start=0.5, T_end=1e-4, seed=777, label="PSA-rnd"
)
print(f"\nBest from random pairings: MSE={best_mse3:.6f}", flush=True)

# Pick overall best
results = [
    (best_mse, best_order, best_pairing, "beam-200"),
    (best_mse2, best_order2, best_pairing2, "beam-10"),
    (best_mse3, best_order3, best_pairing3, "random"),
]
results.sort(key=lambda x: x[0])
best_mse_overall, best_order_overall, best_pairing_overall, label = results[0]

print(f"\n{'='*60}", flush=True)
print(f"Overall best: {label}, MSE={best_mse_overall:.6f}", flush=True)
print(f"Ordering: {best_order_overall}", flush=True)

# Now refine the best result with beam ordering
print(f"\n--- Refining best with beam search ordering (width=10) ---", flush=True)
best_pairs_list = [(e, best_pairing_overall[e]) for e in expand_ids
                   if e in best_pairing_overall]
t0 = time.time()
refined_order, refined_mse = beam_order(best_pairs_list, width=10)
t1 = time.time()
print(f"Beam-refined MSE: {refined_mse:.6f} ({t1-t0:.1f}s)", flush=True)

# Save results
match, perm_str = check_hash(best_order_overall)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)

with open("pairing_sa_result.json", "w") as f:
    json.dump({
        "mse": best_mse_overall,
        "order": best_order_overall,
        "pairing": {str(k): v for k, v in best_pairing_overall.items()},
        "perm": perm_str,
        "hash_match": match,
        "refined_mse": refined_mse,
        "refined_order": refined_order,
    }, f, indent=2)
print("Results saved to pairing_sa_result.json", flush=True)
