import torch
import numpy as np
import csv
import itertools
import time
import sys

def log(msg):
    print(msg, flush=True)

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

expand_ids = [i for i in range(97) if pieces[i]["weight"].shape == (96, 48)]
contract_ids = [i for i in range(97) if pieces[i]["weight"].shape == (48, 96)]
last_id = 85

W_last = pieces[last_id]["weight"].numpy()
b_last = pieces[last_id]["bias"].numpy()

expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}


def apply_block(h, exp_id, con_id):
    z = h @ expand_W[exp_id].T + expand_b[exp_id]
    z = np.maximum(z, 0)
    z = z @ contract_W[con_id].T + contract_b[con_id]
    return h + z


def final_mse(h):
    out = (h @ W_last.T).squeeze() + b_last[0]
    return np.mean((out - pred) ** 2)


# Build h_40 from beam consensus (first 40 blocks)
prefix = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
]

h_40 = X.copy()
for ei, ci in prefix:
    h_40 = apply_block(h_40, ei, ci)

log(f"h_40 MSE (LastLayer projection): {final_mse(h_40):.6f}")

remaining_expand = [14, 42, 1, 50, 27, 77, 86, 23]
remaining_contract = [92, 17, 7, 24, 67, 89, 82, 57]

# ============================================================
# Phase 1: Fix pairings from beam #0, try all 8! orderings
# ============================================================
log("\n" + "=" * 60)
log("Phase 1: Fixed pairings from beam #0, brute-force 8! orderings")
log("=" * 60)

beam0_pairs = [(42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57)]

t0 = time.time()
best_mse = float("inf")
top_results = []

for count, perm in enumerate(itertools.permutations(range(8))):
    ordered_pairs = [beam0_pairs[i] for i in perm]
    h = h_40.copy()
    for ei, ci in ordered_pairs:
        h = apply_block(h, ei, ci)
    mse = final_mse(h)

    if len(top_results) < 20 or mse < top_results[-1][0]:
        top_results.append((mse, list(perm), ordered_pairs))
        top_results.sort()
        top_results = top_results[:20]

    if mse < best_mse:
        best_mse = mse

    if (count + 1) % 10000 == 0:
        log(f"  ...{count+1}/40320, best so far: {best_mse:.6f}")

log(f"\nPhase 1 done in {time.time()-t0:.1f}s")
log(f"Best MSE: {best_mse:.8f}")

log(f"\nTop 10 orderings:")
for rank, (mse, perm, pairs) in enumerate(top_results[:10]):
    log(f"  #{rank+1}: MSE={mse:.8f} pairs={pairs}")

if best_mse < 1e-6:
    log("\nPhase 1 found near-perfect MSE! Done.")
    best_tail = top_results[0][2]
else:
    # ============================================================
    # Phase 2: Try all 8! pairings, each with a FIXED ordering
    #          (identity order) to cheaply find best pairings
    # ============================================================
    log(f"\nPhase 1 best MSE = {best_mse:.6f}, not close to 0. Trying all pairings.")
    log("\n" + "=" * 60)
    log("Phase 2: All 8! pairings, fixed identity ordering (cheap screen)")
    log("=" * 60)

    t0 = time.time()
    pairing_scores = []

    for count, perm in enumerate(itertools.permutations(range(8))):
        pairs = [(remaining_expand[i], remaining_contract[perm[i]]) for i in range(8)]
        h = h_40.copy()
        for ei, ci in pairs:
            h = apply_block(h, ei, ci)
        mse = final_mse(h)
        pairing_scores.append((mse, list(perm), pairs))

        if (count + 1) % 10000 == 0:
            log(f"  ...{count+1}/40320")

    pairing_scores.sort()
    log(f"\nPhase 2 done in {time.time()-t0:.1f}s")
    log(f"\nTop 20 pairings (by identity-order MSE):")
    for rank, (mse, perm, pairs) in enumerate(pairing_scores[:20]):
        log(f"  #{rank+1}: MSE={mse:.8f} pairing={perm} pairs={pairs}")

    # ============================================================
    # Phase 3: For top 20 pairings, brute-force all 8! orderings
    # ============================================================
    log("\n" + "=" * 60)
    log("Phase 3: Top 20 pairings, brute-force 8! orderings each")
    log("=" * 60)

    overall_best_mse = float("inf")
    overall_best_tail = None

    for p_rank, (p_mse, p_perm, p_pairs) in enumerate(pairing_scores[:20]):
        t0 = time.time()
        pairs = [(remaining_expand[i], remaining_contract[p_perm[i]]) for i in range(8)]

        local_best_mse = float("inf")
        local_best_order = None

        for order_perm in itertools.permutations(range(8)):
            ordered = [pairs[i] for i in order_perm]
            h = h_40.copy()
            for ei, ci in ordered:
                h = apply_block(h, ei, ci)
            mse = final_mse(h)
            if mse < local_best_mse:
                local_best_mse = mse
                local_best_order = ordered

        log(f"  Pairing #{p_rank+1} (screen MSE={p_mse:.4f}): "
            f"best MSE={local_best_mse:.8f} ({time.time()-t0:.1f}s) "
            f"order={local_best_order}")

        if local_best_mse < overall_best_mse:
            overall_best_mse = local_best_mse
            overall_best_tail = local_best_order

    log(f"\nOverall best MSE: {overall_best_mse:.10f}")
    log(f"Overall best tail: {overall_best_tail}")
    best_tail = overall_best_tail

# Build final permutation
full_order = prefix + best_tail
perm_list = []
for ei, ci in full_order:
    perm_list.append(ei)
    perm_list.append(ci)
perm_list.append(last_id)

log(f"\nFull permutation ({len(perm_list)} pieces):")
log(",".join(str(p) for p in perm_list))

# Validate
h = X.copy()
for ei, ci in full_order:
    h = apply_block(h, ei, ci)
out = (h @ W_last.T).squeeze() + b_last[0]
log(f"\nFinal validation MSE: {np.mean((out - pred)**2):.10f}")
log(f"Max absolute error: {np.max(np.abs(out - pred)):.10f}")
