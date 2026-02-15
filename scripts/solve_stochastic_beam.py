"""
Stochastic beam search: run many beam searches with random score perturbations.
Uses batched GPU evaluation for speed.
Different noise levels break ties, exploring different basins.
Then refine top results with SA.
"""
import torch
import numpy as np
import csv
import time
import hashlib
import json
import random

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

expand_W_stack = torch.stack([pieces[i]["weight"] for i in expand_ids])
expand_b_stack = torch.stack([pieces[i]["bias"] for i in expand_ids])
contract_W_stack = torch.stack([pieces[i]["weight"] for i in contract_ids])
contract_b_stack = torch.stack([pieces[i]["bias"] for i in contract_ids])

ei2idx = {eid: i for i, eid in enumerate(expand_ids)}
ci2idx = {cid: j for j, cid in enumerate(contract_ids)}


@torch.no_grad()
def score_h(h):
    out = h @ w_last + b_last[0]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def apply_block(h, ei, ci):
    z = h @ expand_W_stack[ei2idx[ei]].T + expand_b_stack[ei2idx[ei]]
    z = torch.relu(z)
    z = z @ contract_W_stack[ci2idx[ci]].T + contract_b_stack[ci2idx[ci]]
    return h + z


@torch.no_grad()
def full_mse(block_order):
    h = X.clone()
    for ei, ci in block_order:
        h = apply_block(h, ei, ci)
    return score_h(h)


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


@torch.no_grad()
def noisy_beam_search(width, noise_std, rng):
    """Single beam search run with Gaussian noise on scores. Batched GPU eval."""
    beam = [(X.clone(), set(), set(), [])]

    for step in range(48):
        all_candidates = []

        for b_idx, (h, used_exp, used_con, _) in enumerate(beam):
            avail_exp = [ei2idx[e] for e in expand_ids if e not in used_exp]
            avail_con = [ci2idx[c] for c in contract_ids if c not in used_con]
            if not avail_exp or not avail_con:
                continue

            con_W = contract_W_stack[avail_con]
            con_b = contract_b_stack[avail_con]
            C = len(avail_con)

            for e_pos, e_idx in enumerate(avail_exp):
                z = h @ expand_W_stack[e_idx].T + expand_b_stack[e_idx]
                z = torch.relu(z)
                h_new = torch.einsum('nr,cdr->cnd', z, con_W) + con_b[:, None, :]
                h_new = h_new + h[None, :, :]
                out = torch.einsum('cnd,d->cn', h_new, w_last) + b_last[0]
                mse = torch.mean((out - pred[None, :]) ** 2, dim=1)

                for c_pos in range(C):
                    noisy = mse[c_pos].item() + rng.normal(0, noise_std)
                    ei = expand_ids[avail_exp[e_pos]]
                    ci = contract_ids[avail_con[c_pos]]
                    all_candidates.append((noisy, b_idx, ei, ci))

        all_candidates.sort(key=lambda x: x[0])

        new_beam = []
        for score, b_idx, ei, ci in all_candidates:
            if len(new_beam) >= width:
                break
            old_h, old_ue, old_uc, old_blocks = beam[b_idx]
            h_new = apply_block(old_h, ei, ci)
            new_beam.append((h_new, old_ue | {ei}, old_uc | {ci}, old_blocks + [(ei, ci)]))

        beam = new_beam
        if not beam:
            break

    if not beam:
        return [], float('inf')

    best = min(beam, key=lambda x: score_h(x[0]))
    return best[3], score_h(best[0])


# ============================================================
# Phase 1: Many stochastic beam searches
# ============================================================
print(f"{'='*60}", flush=True)
print("Phase 1: Stochastic beam search (many runs)", flush=True)
print(f"{'='*60}", flush=True)

NUM_RUNS = 100
results = []
global_best_mse = float('inf')
global_best_order = []
start_time = time.time()

configs = []
for width in [10, 20, 50]:
    for noise in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        configs.append((width, noise))
while len(configs) < NUM_RUNS:
    configs.append((random.choice([10, 20, 50]),
                     random.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])))

for run in range(NUM_RUNS):
    width, noise_std = configs[run]
    rng = np.random.RandomState(run)

    order, mse = noisy_beam_search(width, noise_std, rng)

    if mse < global_best_mse:
        global_best_mse = mse
        global_best_order = order

    results.append((mse, order, width, noise_std, run))

    if (run + 1) % 5 == 0:
        elapsed = time.time() - start_time
        rate = (run + 1) / elapsed
        print(f"  Run {run+1}/{NUM_RUNS}: best_mse={global_best_mse:.6f} "
              f"this_run={mse:.6f} w={width} noise={noise_std:.3f} "
              f"({rate:.2f} runs/s)", flush=True)

results.sort(key=lambda x: x[0])
elapsed = time.time() - start_time
print(f"\nPhase 1 done in {elapsed:.1f}s", flush=True)
print(f"Best MSE: {results[0][0]:.6f} (w={results[0][2]}, noise={results[0][3]:.3f})", flush=True)
print(f"Top 10: {[(f'{r[0]:.4f}', f'w={r[2]}', f'n={r[3]:.3f}') for r in results[:10]]}", flush=True)

from collections import Counter
pair_counts = Counter()
for mse, order, w, n, s in results[:20]:
    for pair in order:
        pair_counts[pair] += 1
print(f"\nMost common pairs in top 20:", flush=True)
for pair, count in pair_counts.most_common(15):
    print(f"  {pair}: {count}/20", flush=True)

# ============================================================
# Phase 2: SA refinement of top 5 results
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Phase 2: SA refinement of top 5 beam results", flush=True)
print(f"{'='*60}", flush=True)

overall_best_mse = float('inf')
overall_best_order = []

for rank in range(min(5, len(results))):
    beam_mse, beam_order, w, n, s = results[rank]
    random.seed(rank + 1000)
    np.random.seed(rank + 1000)

    current = list(beam_order)
    current_mse = full_mse(current)
    best = list(current)
    best_mse = current_mse
    nn = 48
    T = 0.1
    T_end = 1e-8
    max_iters = 200000
    cooling = (T_end / T) ** (1.0 / max_iters)
    accepted = 0
    improved = 0
    sa_start = time.time()

    for it in range(max_iters):
        r = random.random()
        new_order = list(current)
        if r < 0.35:
            i, j = random.sample(range(nn), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif r < 0.60:
            i, j = random.sample(range(nn), 2)
            new_order[i] = (current[i][0], current[j][1])
            new_order[j] = (current[j][0], current[i][1])
        elif r < 0.75:
            i, j = sorted(random.sample(range(nn), 2))
            new_order[i:j+1] = reversed(new_order[i:j+1])
        elif r < 0.88:
            i = random.randrange(nn)
            j = random.randrange(nn)
            if i != j:
                block = new_order.pop(i)
                new_order.insert(j, block)
        else:
            indices = random.sample(range(nn), 3)
            ci_vals = [current[idx][1] for idx in indices]
            new_order[indices[0]] = (current[indices[0]][0], ci_vals[1])
            new_order[indices[1]] = (current[indices[1]][0], ci_vals[2])
            new_order[indices[2]] = (current[indices[2]][0], ci_vals[0])

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
                    with open("solution.json", "w") as f:
                        json.dump({"perm": perm_str, "order": best}, f)
                    exit(0)
        T *= cooling

        if (it + 1) % 20000 == 0:
            elapsed_sa = time.time() - sa_start
            rate = (it + 1) / elapsed_sa
            print(f"  [SA-{rank}] it={it+1}/{max_iters} T={T:.2e} "
                  f"curr={current_mse:.6f} best={best_mse:.6f} "
                  f"acc={accepted} imp={improved} ({rate:.0f}/s)", flush=True)

    print(f"  Rank {rank}: beam={beam_mse:.6f} -> SA={best_mse:.6f}", flush=True)
    if best_mse < overall_best_mse:
        overall_best_mse = best_mse
        overall_best_order = list(best)

print(f"\nOverall best MSE: {overall_best_mse:.10f}", flush=True)
print(f"Ordering: {overall_best_order}", flush=True)
match, perm_str = check_hash(overall_best_order)
print(f"Hash match: {match}", flush=True)
print(f"Permutation: {perm_str}", flush=True)

with open("stochastic_beam_result.json", "w") as f:
    json.dump({
        "mse": overall_best_mse,
        "order": overall_best_order,
        "perm": perm_str,
        "hash_match": match,
    }, f, indent=2)
print("Results saved to stochastic_beam_result.json", flush=True)
