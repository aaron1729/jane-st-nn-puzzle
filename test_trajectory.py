"""
Quick trajectory coherence test: do correct orderings produce smoother
residual stream trajectories (step alignment, curvature, backtracking)?
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
pred = torch.tensor(data[:, 48], device=device)

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

w_last = pieces[85]["weight"].squeeze(0)
b_last = pieces[85]["bias"]

sa_best = [
    (27,76), (94,96), (50,66), (3,53), (1,21), (56,30), (18,78), (15,67),
    (43,55), (84,63), (13,7), (77,25), (2,70), (44,75), (61,20), (59,29),
    (58,40), (95,90), (42,36), (74,92), (91,72), (88,89), (41,22), (39,33),
    (86,47), (87,8), (31,26), (14,11), (73,46), (65,17), (45,54), (35,32),
    (4,52), (48,19), (23,12), (49,51), (28,24), (62,82), (60,57), (68,80),
    (64,79), (16,71), (0,34), (69,83), (5,9), (10,38), (37,6), (81,93),
]

beam10 = [
    (87,71), (31,36), (58,78), (73,72), (18,6), (49,93), (43,11), (95,33),
    (81,51), (68,26), (13,75), (94,55), (5,20), (60,29), (37,40), (10,21),
    (15,9), (16,54), (4,19), (28,47), (74,96), (35,12), (48,38), (2,66),
    (3,53), (61,30), (0,76), (59,79), (44,70), (69,52), (64,83), (45,32),
    (84,63), (41,46), (39,90), (91,34), (62,25), (56,80), (88,22), (65,8),
    (42,92), (27,67), (1,24), (77,89), (50,7), (14,17), (86,82), (23,57),
]

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])


def trajectory_metrics(ordering):
    """Compute trajectory metrics for a given block ordering.
    Returns dict with step_alignment, curvature, backtracking, mse."""
    with torch.no_grad():
        h = X.clone()
        deltas = []
        for ei, ci in ordering:
            z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
            z = torch.relu(z)
            delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
            deltas.append(delta)
            h = h + delta

        out = h @ w_last + b_last[0]
        mse = torch.mean((out - pred) ** 2).item()

        # (A) Step alignment: mean cosine between consecutive deltas
        cosines = []
        for k in range(len(deltas) - 1):
            d1 = deltas[k]      # [N, 48]
            d2 = deltas[k + 1]  # [N, 48]
            # Per-sample cosine, then average over samples
            dot = (d1 * d2).sum(dim=1)
            n1 = torch.norm(d1, dim=1).clamp(min=1e-8)
            n2 = torch.norm(d2, dim=1).clamp(min=1e-8)
            cos = (dot / (n1 * n2)).mean().item()
            cosines.append(cos)
        mean_alignment = np.mean(cosines)

        # (B) Path curvature: sum of ||delta_{k+1} - delta_k||^2
        curvatures = []
        for k in range(len(deltas) - 1):
            diff = deltas[k + 1] - deltas[k]
            curv = torch.mean(torch.sum(diff ** 2, dim=1)).item()
            curvatures.append(curv)
        total_curvature = sum(curvatures)

        # (C) Backtracking: projection of step onto cumulative direction
        backtrack_vals = []
        cumulative = torch.zeros_like(X)
        for k in range(len(deltas)):
            if k > 0:
                cum_norm = torch.norm(cumulative, dim=1).clamp(min=1e-8)
                d_norm = torch.norm(deltas[k], dim=1).clamp(min=1e-8)
                proj = (deltas[k] * cumulative).sum(dim=1) / (d_norm * cum_norm)
                backtrack_vals.append(proj.mean().item())
            cumulative = cumulative + deltas[k]
        mean_backtrack = np.mean(backtrack_vals)  # positive = forward, negative = undoing

        # (D) Path length: total distance traveled
        path_length = sum(torch.mean(torch.norm(d, dim=1)).item() for d in deltas)

        # (E) Displacement efficiency: ||final - start|| / path_length
        total_disp = torch.mean(torch.norm(cumulative, dim=1)).item()
        efficiency = total_disp / path_length if path_length > 0 else 0

    return {
        "mse": mse,
        "alignment": mean_alignment,
        "curvature": total_curvature,
        "backtrack": mean_backtrack,
        "path_length": path_length,
        "efficiency": efficiency,
    }


# ============================================================
# Compute for known solutions
# ============================================================
print("=" * 70, flush=True)
print("Trajectory metrics for known solutions", flush=True)
print("=" * 70, flush=True)

for name, order in [("SA-best (0.274)", sa_best), ("Beam-10 (0.510)", beam10)]:
    m = trajectory_metrics(order)
    print(f"\n  {name}:", flush=True)
    print(f"    MSE:          {m['mse']:.4f}", flush=True)
    print(f"    Alignment:    {m['alignment']:+.4f}  (cosine between consecutive steps)", flush=True)
    print(f"    Curvature:    {m['curvature']:.2f}  (total ||delta_{{k+1}} - delta_k||^2)", flush=True)
    print(f"    Backtrack:    {m['backtrack']:+.4f}  (mean projection onto cumulative)", flush=True)
    print(f"    Path length:  {m['path_length']:.2f}", flush=True)
    print(f"    Efficiency:   {m['efficiency']:.4f}  (displacement / path_length)", flush=True)

# ============================================================
# Compare against random permutations (same pairings as SA-best)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Random orderings (SA-best pairings, shuffled order)", flush=True)
print(f"{'='*70}", flush=True)

random.seed(42)
rand_metrics = {"mse": [], "alignment": [], "curvature": [],
                "backtrack": [], "path_length": [], "efficiency": []}

for trial in range(200):
    shuffled = list(sa_best)
    random.shuffle(shuffled)
    m = trajectory_metrics(shuffled)
    for k in rand_metrics:
        rand_metrics[k].append(m[k])

sa_m = trajectory_metrics(sa_best)
print(f"\n{'Metric':<14s}  {'SA-best':>10s}  {'Rand mean':>10s}  {'Rand std':>10s}  {'SA rank':>8s}", flush=True)
print("-" * 60, flush=True)
for key in ["mse", "alignment", "curvature", "backtrack", "path_length", "efficiency"]:
    val = sa_m[key]
    rm = np.mean(rand_metrics[key])
    rs = np.std(rand_metrics[key])
    # Rank: how many random are better (lower MSE/curvature, higher alignment/backtrack/efficiency)
    if key in ("mse", "curvature", "path_length"):
        rank = sum(1 for r in rand_metrics[key] if r < val)
    else:
        rank = sum(1 for r in rand_metrics[key] if r > val)
    print(f"  {key:<12s}  {val:>10.4f}  {rm:>10.4f}  {rs:>10.4f}  {rank:>4d}/200", flush=True)

# ============================================================
# Also: random pairings + random orderings (fully random)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Fully random (random pairings AND random orderings)", flush=True)
print(f"{'='*70}", flush=True)

full_rand_metrics = {"mse": [], "alignment": [], "curvature": [],
                     "backtrack": [], "path_length": [], "efficiency": []}

for trial in range(200):
    exp_perm = list(expand_ids)
    con_perm = list(contract_ids)
    random.shuffle(exp_perm)
    random.shuffle(con_perm)
    order = list(zip(exp_perm, con_perm))
    m = trajectory_metrics(order)
    for k in full_rand_metrics:
        full_rand_metrics[k].append(m[k])

print(f"\n{'Metric':<14s}  {'SA-best':>10s}  {'Rand-order':>10s}  {'Full-rand':>10s}", flush=True)
print("-" * 50, flush=True)
for key in ["mse", "alignment", "curvature", "backtrack", "path_length", "efficiency"]:
    print(f"  {key:<12s}  {sa_m[key]:>10.4f}  {np.mean(rand_metrics[key]):>10.4f}  "
          f"{np.mean(full_rand_metrics[key]):>10.4f}", flush=True)

# ============================================================
# Correlation: do trajectory metrics predict MSE across random orderings?
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Correlation between trajectory metrics and MSE (200 random orderings)", flush=True)
print(f"{'='*70}", flush=True)

mse_arr = np.array(rand_metrics["mse"])
for key in ["alignment", "curvature", "backtrack", "path_length", "efficiency"]:
    vals = np.array(rand_metrics[key])
    corr = np.corrcoef(mse_arr, vals)[0, 1]
    print(f"  corr(MSE, {key:<12s}) = {corr:+.4f}", flush=True)

print(f"\n{'='*70}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*70}", flush=True)
print("If SA-best ranks near 100/200 on all metrics: no signal.", flush=True)
print("If any metric has |corr with MSE| > 0.3: potentially useful.", flush=True)
