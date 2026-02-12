"""
Quick falsifiability test: does gate entropy of the next block's expand
preactivations distinguish correct from incorrect block adjacency?

If SA-best consecutive pairs have meaningfully different gate entropy
than random consecutive pairs, the adjacency idea has signal.
If not, toss it.
"""
import torch
import numpy as np
import csv
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
X = torch.tensor(data[:, :48], device=device)

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

# SA best ordering (MSE=0.274)
sa_best = [
    (27,76), (94,96), (50,66), (3,53), (1,21), (56,30), (18,78), (15,67),
    (43,55), (84,63), (13,7), (77,25), (2,70), (44,75), (61,20), (59,29),
    (58,40), (95,90), (42,36), (74,92), (91,72), (88,89), (41,22), (39,33),
    (86,47), (87,8), (31,26), (14,11), (73,46), (65,17), (45,54), (35,32),
    (4,52), (48,19), (23,12), (49,51), (28,24), (62,82), (60,57), (68,80),
    (64,79), (16,71), (0,34), (69,83), (5,9), (10,38), (37,6), (81,93),
]

beam10 = [
    (87, 71), (31, 36), (58, 78), (73, 72), (18, 6), (49, 93), (43, 11), (95, 33),
    (81, 51), (68, 26), (13, 75), (94, 55), (5, 20), (60, 29), (37, 40), (10, 21),
    (15, 9), (16, 54), (4, 19), (28, 47), (74, 96), (35, 12), (48, 38), (2, 66),
    (3, 53), (61, 30), (0, 76), (59, 79), (44, 70), (69, 52), (64, 83), (45, 32),
    (84, 63), (41, 46), (39, 90), (91, 34), (62, 25), (56, 80), (88, 22), (65, 8),
    (42, 92), (27, 67), (1, 24), (77, 89), (50, 7), (14, 17), (86, 82), (23, 57),
]

N_BLOCKS = 48

def apply_block(h, ei, ci):
    z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
    z = torch.relu(z)
    delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
    return h + delta

def gate_entropy(h, exp_id):
    """Mean binary entropy across 96 gates."""
    u = h @ pieces[exp_id]["weight"].T + pieces[exp_id]["bias"]
    p = (u > 0).float().mean(dim=0).clamp(1e-7, 1 - 1e-7)
    ent = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    return ent.mean().item()

def gate_dead_frac(h, exp_id):
    """Fraction of gates that are <5% or >95% active."""
    u = h @ pieces[exp_id]["weight"].T + pieces[exp_id]["bias"]
    p = (u > 0).float().mean(dim=0)
    return ((p < 0.05) | (p > 0.95)).float().mean().item()

# ============================================================
# Test 1: On raw input X, compute adjacency for SA-best blocks
# ============================================================
print("=" * 70, flush=True)
print("Test 1: Gate entropy after one block applied to raw X", flush=True)
print("=" * 70, flush=True)

with torch.no_grad():
    # Baseline: entropy on raw X
    baseline = [gate_entropy(X, ei) for ei, ci in sa_best]
    print(f"Baseline (raw X): mean entropy={np.mean(baseline):.4f}  std={np.std(baseline):.4f}", flush=True)

    # 48x48 adjacency matrix (using SA-best pairings)
    adj = np.zeros((N_BLOCKS, N_BLOCKS))
    for i, (ei, ci) in enumerate(sa_best):
        h_after = apply_block(X, ei, ci)
        for j, (ej, cj) in enumerate(sa_best):
            if i == j:
                adj[i, j] = -1  # sentinel
                continue
            adj[i, j] = gate_entropy(h_after, ej)

    mask = np.ones((N_BLOCKS, N_BLOCKS), dtype=bool)
    np.fill_diagonal(mask, False)
    print(f"Adjacency entropy: mean={adj[mask].mean():.4f}  std={adj[mask].std():.4f}  "
          f"min={adj[mask].min():.4f}  max={adj[mask].max():.4f}", flush=True)

    # SA-best consecutive pairs
    sa_consec = [adj[k, k+1] for k in range(N_BLOCKS - 1)]

    # Random consecutive pairs (1000 random orderings)
    random.seed(42)
    rand_consec = []
    for _ in range(1000):
        perm = list(range(N_BLOCKS))
        random.shuffle(perm)
        trial = [adj[perm[k], perm[k+1]] for k in range(N_BLOCKS - 1)]
        rand_consec.append(np.mean(trial))

    print(f"\nSA-best consecutive entropy: mean={np.mean(sa_consec):.4f}  std={np.std(sa_consec):.4f}", flush=True)
    print(f"Random consecutive entropy:  mean={np.mean(rand_consec):.4f}  std={np.std(rand_consec):.4f}", flush=True)
    print(f"SA-best vs random: {np.mean(sa_consec) - np.mean(rand_consec):+.4f}", flush=True)
    # Where does SA-best rank among random?
    sa_mean = np.mean(sa_consec)
    rank = sum(1 for r in rand_consec if r > sa_mean)
    print(f"SA-best rank among 1000 random: {rank}/1000 (higher entropy = healthier)", flush=True)

# ============================================================
# Test 2: Greedy Hamiltonian path (maximize gate entropy)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Test 2: Greedy Hamiltonian path (max total gate entropy)", flush=True)
print(f"{'='*70}", flush=True)

best_path = None
best_total = -float('inf')
for start in range(N_BLOCKS):
    path = [start]
    used = {start}
    total = 0
    for _ in range(N_BLOCKS - 1):
        cur = path[-1]
        best_next = max((j for j in range(N_BLOCKS) if j not in used),
                        key=lambda j: adj[cur, j])
        path.append(best_next)
        used.add(best_next)
        total += adj[cur, best_next]
    if total > best_total:
        best_total = total
        best_path = path

sa_total = sum(sa_consec)
print(f"Greedy path total entropy: {best_total:.4f}", flush=True)
print(f"SA-best total entropy:     {sa_total:.4f}", flush=True)

# Positional overlap
greedy_pos = {idx: pos for pos, idx in enumerate(best_path)}
pos_overlap = sum(1 for i in range(N_BLOCKS) if greedy_pos[i] == i)
print(f"Positional overlap: {pos_overlap}/48", flush=True)

# Adjacent-pair overlap
sa_adj_set = set((k, k+1) for k in range(N_BLOCKS-1))
greedy_adj_set = set((best_path[k], best_path[k+1]) for k in range(N_BLOCKS-1))
adj_overlap = len(sa_adj_set & greedy_adj_set)
print(f"Adjacent-pair overlap: {adj_overlap}/47", flush=True)

# ============================================================
# Test 3: Multi-depth context (states from SA-best at various depths)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Test 3: Adjacency at multiple depths (SA-best intermediate states)", flush=True)
print(f"{'='*70}", flush=True)

depths = [0, 8, 16, 24, 32, 40]
with torch.no_grad():
    h = X.clone()
    for pos, (ei, ci) in enumerate(sa_best):
        if pos in depths:
            # Compute adjacency at this depth
            ctx_adj = np.zeros((N_BLOCKS, N_BLOCKS))
            for i, (ei2, ci2) in enumerate(sa_best):
                h_after = apply_block(h, ei2, ci2)
                for j, (ej2, cj2) in enumerate(sa_best):
                    if i == j:
                        ctx_adj[i, j] = -1
                        continue
                    ctx_adj[i, j] = gate_entropy(h_after, ej2)

            sa_ent = [ctx_adj[k, k+1] for k in range(N_BLOCKS-1)]
            random.seed(42)
            rand_ents = []
            for _ in range(500):
                perm = list(range(N_BLOCKS))
                random.shuffle(perm)
                rand_ents.append(np.mean([ctx_adj[perm[k], perm[k+1]] for k in range(N_BLOCKS-1)]))

            sa_rank = sum(1 for r in rand_ents if r > np.mean(sa_ent))
            print(f"  Depth {pos:2d}: SA-best={np.mean(sa_ent):.4f}  "
                  f"random={np.mean(rand_ents):.4f}  "
                  f"diff={np.mean(sa_ent)-np.mean(rand_ents):+.5f}  "
                  f"rank={sa_rank}/500", flush=True)

        h = apply_block(h, ei, ci)

# ============================================================
# Test 4: Also check beam-10 ordering with its own pairings
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Test 4: Same test for beam-10 ordering", flush=True)
print(f"{'='*70}", flush=True)

with torch.no_grad():
    b10_adj = np.zeros((N_BLOCKS, N_BLOCKS))
    for i, (ei, ci) in enumerate(beam10):
        h_after = apply_block(X, ei, ci)
        for j, (ej, cj) in enumerate(beam10):
            if i == j:
                b10_adj[i, j] = -1
                continue
            b10_adj[i, j] = gate_entropy(h_after, ej)

    b10_consec = [b10_adj[k, k+1] for k in range(N_BLOCKS-1)]
    random.seed(42)
    b10_rand = []
    for _ in range(1000):
        perm = list(range(N_BLOCKS))
        random.shuffle(perm)
        b10_rand.append(np.mean([b10_adj[perm[k], perm[k+1]] for k in range(N_BLOCKS-1)]))

    b10_rank = sum(1 for r in b10_rand if r > np.mean(b10_consec))
    print(f"Beam-10 consecutive entropy: mean={np.mean(b10_consec):.4f}", flush=True)
    print(f"Random consecutive entropy:  mean={np.mean(b10_rand):.4f}", flush=True)
    print(f"Diff: {np.mean(b10_consec)-np.mean(b10_rand):+.4f}  rank={b10_rank}/1000", flush=True)

print(f"\n{'='*70}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*70}", flush=True)
print("If SA-best/beam-10 rank near 500/1000 (middle), adjacency has NO signal.", flush=True)
print("If they rank near 0 (top) or 1000 (bottom), there's something there.", flush=True)
