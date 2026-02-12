import torch
import numpy as np
import csv

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

expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}


def apply_block(h, exp_id, con_id):
    z = h @ expand_W[exp_id].T + expand_b[exp_id]
    z = np.maximum(z, 0)
    z = z @ contract_W[con_id].T + contract_b[con_id]
    return h + z


def ridge_mse(h, y, alpha):
    H = np.hstack([h, np.ones((h.shape[0], 1))])
    HtH = H.T @ H
    HtH[:-1, :-1] += alpha * np.eye(h.shape[1])
    Hty = H.T @ y
    w = np.linalg.solve(HtH, Hty)
    residuals = H @ w - y
    return np.mean(residuals ** 2)


alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for alpha in alphas:
    print(f"\n{'='*60}")
    print(f"ALPHA = {alpha}")
    print(f"{'='*60}")

    h = X.copy()
    remaining_expand = set(expand_ids)
    remaining_contract = set(contract_ids)

    for step in range(5):
        all_scores = []
        for ei in remaining_expand:
            for ci in remaining_contract:
                h_candidate = apply_block(h, ei, ci)
                s = ridge_mse(h_candidate, pred, alpha)
                all_scores.append((s, ei, ci))

        all_scores.sort()
        best = all_scores[0]
        second = all_scores[1]
        ei, ci = best[1], best[2]
        h = apply_block(h, ei, ci)
        remaining_expand.remove(ei)
        remaining_contract.remove(ci)

        gap = second[0] - best[0]
        print(f"  Step {step+1}: ({ei:2d},{ci:2d}) MSE={best[0]:.6f}  gap={gap:.6f}  ratio={second[0]/best[0]:.6f}")
