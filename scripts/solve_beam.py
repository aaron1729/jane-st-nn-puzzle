import torch
import numpy as np
import csv
import json
import time

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
N = len(pred)

# Load all pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location="cpu")

expand_ids = [i for i in range(97) if pieces[i]["weight"].shape == (96, 48)]
contract_ids = [i for i in range(97) if pieces[i]["weight"].shape == (48, 96)]
last_id = 85

W_last = pieces[last_id]["weight"].numpy()  # [1, 48]
b_last = pieces[last_id]["bias"].numpy()    # [1]

expand_W = {i: pieces[i]["weight"].numpy() for i in expand_ids}
expand_b = {i: pieces[i]["bias"].numpy() for i in expand_ids}
contract_W = {i: pieces[i]["weight"].numpy() for i in contract_ids}
contract_b = {i: pieces[i]["bias"].numpy() for i in contract_ids}


def apply_block(h, exp_id, con_id):
    z = h @ expand_W[exp_id].T + expand_b[exp_id]
    z = np.maximum(z, 0)
    z = z @ contract_W[con_id].T + contract_b[con_id]
    return h + z


def score(h):
    out = (h @ W_last.T).squeeze() + b_last[0]
    return np.mean((out - pred) ** 2)


BEAM_WIDTH = 10

# Each beam entry: (score, h, chosen_order, remaining_expand, remaining_contract)
initial_score = score(X)
beam = [(initial_score, X.copy(), [], set(expand_ids), set(contract_ids))]

# Log: per-step details for recording
step_log = []

print(f"Beam search with width {BEAM_WIDTH}, LastLayer scoring")
print(f"Initial MSE: {initial_score:.6f}")
print()

t0 = time.time()

for step in range(48):
    ts = time.time()

    # Expand all beam entries
    candidates = []
    for b_idx, (b_score, b_h, b_order, b_re, b_rc) in enumerate(beam):
        for ei in b_re:
            for ci in b_rc:
                h_new = apply_block(b_h, ei, ci)
                s = score(h_new)
                candidates.append((s, b_idx, ei, ci))

    candidates.sort()

    # Record top candidates for this step (across all beam parents)
    top_20 = candidates[:20]

    # Keep top BEAM_WIDTH
    top = candidates[:BEAM_WIDTH]
    new_beam = []
    for s, b_idx, ei, ci in top:
        _, b_h, b_order, b_re, b_rc = beam[b_idx]
        h_new = apply_block(b_h, ei, ci)
        new_order = b_order + [(ei, ci)]
        new_re = b_re - {ei}
        new_rc = b_rc - {ci}
        new_beam.append((s, h_new, new_order, new_re, new_rc))

    beam = new_beam
    te = time.time()

    # How many distinct beam parents survived?
    parent_ids = [b_idx for _, b_idx, _, _ in top]
    unique_parents = len(set(parent_ids))

    # Log this step
    step_info = {
        "step": step + 1,
        "n_candidates": len(candidates),
        "beam_best_mse": top[0][0],
        "beam_worst_mse": top[-1][0],
        "unique_parents": unique_parents,
        "top_10": [(f"{s:.6f}", b_idx, ei, ci) for s, b_idx, ei, ci in candidates[:10]],
        "elapsed": te - ts,
    }
    step_log.append(step_info)

    # Gap between 1st and 2nd
    gap = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else 0
    gap_1_10 = top[-1][0] - top[0][0] if len(top) > 1 else 0

    print(f"Step {step+1:2d} ({te-ts:.1f}s): best={top[0][0]:.6f}  "
          f"beam_spread={gap_1_10:.6f}  gap_1v2={gap:.6f}  "
          f"parents={unique_parents}  "
          f"pair=({top[0][2]},{top[0][3]}) from beam#{top[0][1]}")

    # Print top 5 with details
    for rank, (s, b_idx, ei, ci) in enumerate(candidates[:5]):
        print(f"    #{rank+1}: MSE={s:.6f} beam#{b_idx} ({ei},{ci})")


print(f"\nTotal time: {time.time()-t0:.1f}s")
print()

# Report on all beam survivors
print(f"{'='*60}")
print(f"Final beam ({len(beam)} entries):")
print(f"{'='*60}")

results = []
for rank, (s, h, order, re, rc) in enumerate(beam):
    final_out = (h @ W_last.T).squeeze() + b_last[0]
    final_mse = np.mean((final_out - pred) ** 2)
    max_err = np.max(np.abs(final_out - pred))

    perm = []
    for ei, ci in order:
        perm.append(ei)
        perm.append(ci)
    perm.append(last_id)

    result = {
        "rank": rank,
        "beam_mse": s,
        "final_lastlayer_mse": float(final_mse),
        "max_abs_error": float(max_err),
        "order": order,
        "permutation": perm,
    }
    results.append(result)

    print(f"\n  Beam #{rank}: beam_MSE={s:.6f}  final_LastLayer_MSE={final_mse:.8f}  max_err={max_err:.6f}")
    print(f"    Order: {order}")
    print(f"    Permutation: {','.join(str(p) for p in perm)}")

# Save full results
output = {
    "config": {"beam_width": BEAM_WIDTH, "scoring": "LastLayer_MSE", "n_steps": 48},
    "step_log": step_log,
    "final_beam": results,
}
with open("beam_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nDetailed results saved to beam_results.json")
