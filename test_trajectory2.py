"""
Trajectory test v2: scale-normalized metrics to separate genuine
directional signal from norm-growth proxy.
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

eps = 1e-8

def trajectory_metrics(ordering):
    with torch.no_grad():
        h = X.clone()
        deltas = []
        h_norms = []
        for ei, ci in ordering:
            h_norms.append(torch.norm(h, dim=1))  # [N]
            z = h @ pieces[ei]["weight"].T + pieces[ei]["bias"]
            z = torch.relu(z)
            delta = z @ pieces[ci]["weight"].T + pieces[ci]["bias"]
            deltas.append(delta)
            h = h + delta

        out = h @ w_last + b_last[0]
        mse = torch.mean((out - pred) ** 2).item()
        final_norm = torch.norm(h, dim=1).mean().item()

        # === SCALE-FREE METRICS ===

        # (A) Step alignment (already cosine-normalized)
        cosines = []
        for k in range(len(deltas) - 1):
            d1, d2 = deltas[k], deltas[k+1]
            dot = (d1 * d2).sum(dim=1)
            n1 = torch.norm(d1, dim=1).clamp(min=eps)
            n2 = torch.norm(d2, dim=1).clamp(min=eps)
            cosines.append((dot / (n1 * n2)).mean().item())
        alignment = np.mean(cosines)

        # (B) Directional curvature: sum of ||hat_delta_{k+1} - hat_delta_k||^2
        dir_curvatures = []
        for k in range(len(deltas) - 1):
            d1_hat = deltas[k] / torch.norm(deltas[k], dim=1, keepdim=True).clamp(min=eps)
            d2_hat = deltas[k+1] / torch.norm(deltas[k+1], dim=1, keepdim=True).clamp(min=eps)
            diff = d2_hat - d1_hat
            dir_curvatures.append(torch.mean(torch.sum(diff**2, dim=1)).item())
        dir_curvature = sum(dir_curvatures)
        # Note: ||hat_d2 - hat_d1||^2 = 2(1 - cos(d1,d2)), so this is redundant with alignment
        # but let's compute both to confirm

        # (C) Relative step size: ||delta_k|| / ||h_k|| (dimensionless)
        rel_steps = []
        for k in range(len(deltas)):
            d_norm = torch.norm(deltas[k], dim=1)
            h_norm = h_norms[k].clamp(min=eps)
            rel_steps.append((d_norm / h_norm).mean().item())
        mean_rel_step = np.mean(rel_steps)
        total_rel_step = sum(rel_steps)

        # (D) Backtracking (already cosine-normalized)
        backtrack_vals = []
        cumulative = torch.zeros_like(X)
        for k in range(len(deltas)):
            if k > 0:
                cum_norm = torch.norm(cumulative, dim=1).clamp(min=eps)
                d_norm = torch.norm(deltas[k], dim=1).clamp(min=eps)
                proj = (deltas[k] * cumulative).sum(dim=1) / (d_norm * cum_norm)
                backtrack_vals.append(proj.mean().item())
            cumulative = cumulative + deltas[k]
        backtrack = np.mean(backtrack_vals)

        # (E) Relative step size variance (are some steps disproportionately large?)
        rel_step_std = np.std(rel_steps)

        # === NORM-DEPENDENT METRICS (for comparison) ===
        path_length = sum(torch.mean(torch.norm(d, dim=1)).item() for d in deltas)
        total_curvature = 0
        for k in range(len(deltas) - 1):
            diff = deltas[k+1] - deltas[k]
            total_curvature += torch.mean(torch.sum(diff**2, dim=1)).item()

    return {
        "mse": mse,
        "final_norm": final_norm,
        # Scale-free
        "alignment": alignment,
        "dir_curvature": dir_curvature,
        "mean_rel_step": mean_rel_step,
        "total_rel_step": total_rel_step,
        "backtrack": backtrack,
        "rel_step_std": rel_step_std,
        # Norm-dependent
        "path_length": path_length,
        "curvature": total_curvature,
    }

# ============================================================
# Known solutions
# ============================================================
print("=" * 70, flush=True)
print("Known solutions", flush=True)
print("=" * 70, flush=True)

for name, order in [("SA-best", sa_best), ("Beam-10", beam10)]:
    m = trajectory_metrics(order)
    print(f"\n  {name} (MSE={m['mse']:.4f}, final_norm={m['final_norm']:.1f}):", flush=True)
    print(f"    [scale-free]", flush=True)
    print(f"      alignment:      {m['alignment']:+.4f}", flush=True)
    print(f"      dir_curvature:  {m['dir_curvature']:.4f}  (= sum of 2(1-cos))", flush=True)
    print(f"      mean_rel_step:  {m['mean_rel_step']:.4f}", flush=True)
    print(f"      total_rel_step: {m['total_rel_step']:.4f}", flush=True)
    print(f"      backtrack:      {m['backtrack']:+.4f}", flush=True)
    print(f"      rel_step_std:   {m['rel_step_std']:.4f}", flush=True)
    print(f"    [norm-dependent]", flush=True)
    print(f"      path_length:    {m['path_length']:.2f}", flush=True)
    print(f"      curvature:      {m['curvature']:.2f}", flush=True)

# ============================================================
# 200 random orderings (SA-best pairings, shuffled order)
# ============================================================
print(f"\n{'='*70}", flush=True)
print("200 random orderings (SA-best pairings, shuffled)", flush=True)
print(f"{'='*70}", flush=True)

random.seed(42)
all_keys = ["mse", "final_norm",
            "alignment", "dir_curvature", "mean_rel_step", "total_rel_step",
            "backtrack", "rel_step_std",
            "path_length", "curvature"]
rand_m = {k: [] for k in all_keys}

for trial in range(200):
    shuffled = list(sa_best)
    random.shuffle(shuffled)
    m = trajectory_metrics(shuffled)
    for k in all_keys:
        rand_m[k].append(m[k])

sa_m = trajectory_metrics(sa_best)

# For ranking: lower is better for mse, final_norm, dir_curvature, curvature, path_length, rel_step_std
# Higher is better for alignment, backtrack, mean_rel_step(?), total_rel_step(?)
lower_better = {"mse", "final_norm", "dir_curvature", "curvature", "path_length",
                "rel_step_std", "mean_rel_step", "total_rel_step"}

print(f"\n{'Metric':<16s}  {'SA-best':>10s}  {'Rand mean':>10s}  {'Rand std':>10s}  {'SA rank':>8s}  {'corr(MSE)':>10s}", flush=True)
print("-" * 75, flush=True)

mse_arr = np.array(rand_m["mse"])
for key in all_keys:
    val = sa_m[key]
    rm = np.mean(rand_m[key])
    rs = np.std(rand_m[key])
    vals = np.array(rand_m[key])
    corr = np.corrcoef(mse_arr, vals)[0, 1] if key != "mse" else 1.0

    if key in lower_better:
        rank = sum(1 for r in rand_m[key] if r < val)
    else:
        rank = sum(1 for r in rand_m[key] if r > val)

    marker = ""
    if key != "mse" and abs(corr) > 0.3:
        marker = " ***"
    print(f"  {key:<14s}  {val:>10.4f}  {rm:>10.4f}  {rs:>10.4f}  {rank:>4d}/200  {corr:>+10.4f}{marker}", flush=True)

# ============================================================
# Partial correlations: control for final_norm
# ============================================================
print(f"\n{'='*70}", flush=True)
print("Partial correlations (controlling for final_norm)", flush=True)
print(f"{'='*70}", flush=True)

norm_arr = np.array(rand_m["final_norm"])
for key in ["alignment", "dir_curvature", "mean_rel_step", "total_rel_step",
            "backtrack", "rel_step_std", "path_length", "curvature"]:
    vals = np.array(rand_m[key])
    # Partial correlation of key with MSE, controlling for final_norm
    # r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    r_xy = np.corrcoef(mse_arr, vals)[0, 1]
    r_xz = np.corrcoef(mse_arr, norm_arr)[0, 1]
    r_yz = np.corrcoef(vals, norm_arr)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    partial = (r_xy - r_xz * r_yz) / denom if denom > eps else 0
    print(f"  partial_corr(MSE, {key:<14s} | final_norm) = {partial:+.4f}  "
          f"(raw={r_xy:+.4f}, corr_w_norm={r_yz:+.4f})", flush=True)
