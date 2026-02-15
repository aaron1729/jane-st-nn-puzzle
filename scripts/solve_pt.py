"""
Parallel tempering SA with auxiliary 'true'-based scoring.
Multiple SA chains at different temperatures; periodically swap states between adjacent chains.
Also includes norm stability penalty and true-MSE shape constraint.
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
X = torch.tensor(data[:, :48], device=device)
pred = torch.tensor(data[:, 48], device=device)
true_vals = torch.tensor(data[:, 49], device=device)

# Known: MSE(pred, true) for the correct network
TRUE_MSE_TARGET = torch.mean((pred - true_vals) ** 2).item()
print(f"MSE(pred, true) = {TRUE_MSE_TARGET:.6f}", flush=True)

# Load pieces
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
def forward(block_order):
    """Run full forward pass, return output and final hidden state norm."""
    h = X.clone()
    for ei, ci in block_order:
        z = h @ expand_W[ei].T + expand_b[ei]
        z = torch.relu(z)
        z = z @ contract_W[ci].T + contract_b[ci]
        h = h + z
    out = h @ w_last + b_last[0]
    return out, torch.mean(h ** 2).item()


@torch.no_grad()
def score(block_order, alpha=0.0, beta=0.0):
    """
    Composite score:
    - MSE(output, pred)                         [primary — this is what we minimize to 0]
    - + alpha * |MSE(output, true) - target|    [shape constraint — should be ~0.1065]
    - + beta * max(0, h_norm - norm_budget)     [stability — don't let norms explode]
    """
    out, h_norm = forward(block_order)
    mse_pred = torch.mean((out - pred) ** 2).item()

    if alpha > 0:
        mse_true = torch.mean((out - true_vals) ** 2).item()
        shape_penalty = abs(mse_true - TRUE_MSE_TARGET)
    else:
        shape_penalty = 0.0

    if beta > 0:
        # Norm budget: input norm is ~0.64, reasonable output might be ~50-100
        norm_penalty = max(0, h_norm - 100.0)
    else:
        norm_penalty = 0.0

    return mse_pred + alpha * shape_penalty + beta * norm_penalty, mse_pred


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


def make_neighbor(current, n):
    """Generate a neighbor solution."""
    r = random.random()
    new_order = list(current)
    if r < 0.35:
        # Position swap
        i, j = random.sample(range(n), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    elif r < 0.60:
        # Contract piece swap (re-pair)
        i, j = random.sample(range(n), 2)
        new_order[i] = (current[i][0], current[j][1])
        new_order[j] = (current[j][0], current[i][1])
    elif r < 0.75:
        # Segment reversal
        i, j = sorted(random.sample(range(n), 2))
        new_order[i:j+1] = reversed(new_order[i:j+1])
    elif r < 0.88:
        # Block insertion (remove and reinsert)
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            block = new_order.pop(i)
            new_order.insert(j, block)
    else:
        # 3-way contract rotation
        indices = random.sample(range(n), 3)
        ci_vals = [current[idx][1] for idx in indices]
        # Rotate: a->b, b->c, c->a
        random.shuffle(indices)
        new_order[indices[0]] = (current[indices[0]][0], ci_vals[1])
        new_order[indices[1]] = (current[indices[1]][0], ci_vals[2])
        new_order[indices[2]] = (current[indices[2]][0], ci_vals[0])
    return new_order


# ============================================================
# Starting points
# ============================================================

# Beam-10 + local search (MSE ~0.449)
beam10_local = [
    (2, 66), (87, 71), (3, 53), (73, 72), (49, 93), (43, 11), (68, 26), (58, 78),
    (81, 51), (95, 33), (13, 75), (94, 55), (5, 20), (60, 29), (10, 21), (37, 40),
    (15, 9), (4, 19), (16, 54), (28, 47), (48, 38), (35, 12), (50, 89), (74, 96),
    (18, 6), (61, 30), (0, 76), (59, 79), (69, 52), (44, 70), (64, 83), (45, 32),
    (41, 46), (39, 90), (84, 63), (91, 34), (42, 92), (56, 80), (88, 22), (65, 8),
    (62, 25), (27, 67), (1, 24), (77, 7), (31, 36), (14, 17), (86, 82), (23, 57),
]

# Beam-200 + pairwise swaps (MSE ~0.374)
beam200_swaps = [
    (48, 9), (87, 71), (58, 78), (49, 93), (73, 72), (31, 26), (81, 75), (0, 54),
    (41, 51), (45, 33), (39, 32), (4, 52), (3, 40), (68, 47), (59, 92), (42, 46),
    (61, 83), (15, 66), (35, 22), (16, 90), (10, 20), (56, 30), (2, 70), (91, 21),
    (62, 57), (13, 34), (28, 25), (18, 63), (86, 76), (74, 80), (44, 7), (69, 89),
    (14, 8), (43, 53), (84, 96), (88, 38), (95, 79), (27, 17), (50, 36), (37, 67),
    (5, 11), (23, 19), (94, 6), (64, 55), (60, 29), (1, 12), (65, 24), (77, 82),
]

# Print starting MSEs
for name, order in [("Beam-10+local", beam10_local), ("Beam-200+swaps", beam200_swaps)]:
    _, mse = score(order)
    out, h_norm = forward(order)
    mse_true = torch.mean((out - true_vals) ** 2).item()
    print(f"{name}: MSE(pred)={mse:.6f}, MSE(true)={mse_true:.6f}, h_norm={h_norm:.1f}", flush=True)

# ============================================================
# Parallel Tempering
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Parallel Tempering SA", flush=True)
print(f"{'='*60}", flush=True)

NUM_CHAINS = 8
ITERS = 300000
SWAP_EVERY = 500  # Try chain swaps every N iterations

# Temperature ladder (geometric from hot to cold)
T_max = 0.5
T_min = 1e-7
temps = [T_max * (T_min / T_max) ** (i / (NUM_CHAINS - 1)) for i in range(NUM_CHAINS)]
print(f"Temperature ladder: {[f'{t:.2e}' for t in temps]}", flush=True)

# Cooling: all chains cool by same factor
cooling = (T_min / T_max) ** (1.0 / ITERS)

# Alpha/beta for auxiliary scoring (only on colder chains)
ALPHA = 0.05  # shape constraint weight
BETA = 0.001  # norm penalty weight

# Initialize chains: mix of starting points + random perturbations
chains = []
for c in range(NUM_CHAINS):
    if c < 3:
        init = list(beam200_swaps)
    elif c < 6:
        init = list(beam10_local)
    else:
        # Random shuffle of one of the starting points
        init = list(beam200_swaps if random.random() < 0.5 else beam10_local)
        random.shuffle(init)
        # Also randomly re-pair some
        indices = random.sample(range(48), k=random.randint(4, 16))
        cons = [init[idx][1] for idx in indices]
        random.shuffle(cons)
        for k, idx in enumerate(indices):
            init[idx] = (init[idx][0], cons[k])

    # Use auxiliary scoring only for colder chains
    use_alpha = ALPHA if c < 4 else 0.0
    use_beta = BETA if c < 4 else 0.0

    composite, mse = score(init, alpha=use_alpha, beta=use_beta)
    chains.append({
        "order": init,
        "composite": composite,
        "mse": mse,
        "alpha": use_alpha,
        "beta": use_beta,
        "accepted": 0,
        "improved": 0,
    })

# Global best tracking
global_best_mse = min(c["mse"] for c in chains)
global_best_order = list(min(chains, key=lambda c: c["mse"])["order"])
print(f"Initial global best MSE: {global_best_mse:.6f}", flush=True)

n = 48
start_time = time.time()
swaps_accepted = 0
swaps_tried = 0

for it in range(ITERS):
    # Each chain does one SA step
    for c_idx, chain in enumerate(chains):
        T = temps[c_idx]
        new_order = make_neighbor(chain["order"], n)
        new_composite, new_mse = score(new_order, alpha=chain["alpha"], beta=chain["beta"])
        delta = new_composite - chain["composite"]

        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-30)):
            chain["order"] = new_order
            chain["composite"] = new_composite
            chain["mse"] = new_mse
            chain["accepted"] += 1

            if new_mse < global_best_mse:
                global_best_mse = new_mse
                global_best_order = list(new_order)
                chain["improved"] += 1

                match, perm_str = check_hash(global_best_order)
                if match:
                    print(f"\n*** SOLUTION FOUND at iter {it}! ***", flush=True)
                    print(f"Permutation: {perm_str}", flush=True)
                    with open("solution.json", "w") as f:
                        json.dump({"perm": perm_str, "order": global_best_order}, f)
                    exit(0)

    # Cool all chains
    temps = [t * cooling for t in temps]

    # Swap adjacent chains (parallel tempering exchange)
    if (it + 1) % SWAP_EVERY == 0:
        for c_idx in range(NUM_CHAINS - 1):
            swaps_tried += 1
            # Metropolis criterion for swap
            # Use raw MSE for swap criterion (not composite) so all chains comparable
            mse_i = chains[c_idx]["mse"]
            mse_j = chains[c_idx + 1]["mse"]
            T_i = temps[c_idx]
            T_j = temps[c_idx + 1]

            # Swap acceptance: exp((1/T_i - 1/T_j) * (E_i - E_j))
            log_accept = (1.0/max(T_i, 1e-30) - 1.0/max(T_j, 1e-30)) * (mse_i - mse_j)
            if log_accept > 0 or random.random() < np.exp(log_accept):
                # Swap states
                chains[c_idx]["order"], chains[c_idx + 1]["order"] = \
                    chains[c_idx + 1]["order"], chains[c_idx]["order"]
                chains[c_idx]["mse"], chains[c_idx + 1]["mse"] = \
                    chains[c_idx + 1]["mse"], chains[c_idx]["mse"]
                # Recompute composite scores with each chain's own alpha/beta
                chains[c_idx]["composite"], _ = score(
                    chains[c_idx]["order"], chains[c_idx]["alpha"], chains[c_idx]["beta"])
                chains[c_idx + 1]["composite"], _ = score(
                    chains[c_idx + 1]["order"], chains[c_idx + 1]["alpha"], chains[c_idx + 1]["beta"])
                swaps_accepted += 1

    # Logging
    if (it + 1) % 5000 == 0:
        elapsed = time.time() - start_time
        rate = (it + 1) * NUM_CHAINS / elapsed
        chain_mses = [f"{c['mse']:.4f}" for c in chains]
        print(f"  iter {it+1}/{ITERS} T=[{temps[0]:.2e}..{temps[-1]:.2e}] "
              f"global_best={global_best_mse:.6f} "
              f"chain_mses=[{','.join(chain_mses)}] "
              f"swaps={swaps_accepted}/{swaps_tried} "
              f"({rate:.0f} eval/s)", flush=True)

elapsed = time.time() - start_time
print(f"\nParallel tempering done in {elapsed:.1f}s", flush=True)
print(f"Global best MSE: {global_best_mse:.10f}", flush=True)
print(f"Global best ordering: {global_best_order}", flush=True)

# Save results
match, perm_str = check_hash(global_best_order)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)

out, h_norm = forward(global_best_order)
mse_true = torch.mean((out - true_vals) ** 2).item()
print(f"MSE(output, true) = {mse_true:.6f} (target: {TRUE_MSE_TARGET:.6f})", flush=True)
print(f"Final h_norm = {h_norm:.1f}", flush=True)

with open("pt_result.json", "w") as f:
    json.dump({
        "mse": global_best_mse,
        "mse_true": mse_true,
        "h_norm": h_norm,
        "order": global_best_order,
        "perm": perm_str,
        "hash_match": match,
    }, f, indent=2)
print("Results saved to pt_result.json", flush=True)
