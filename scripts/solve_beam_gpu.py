"""
GPU-accelerated beam search with batched evaluation.
Key: batch all contract pieces for each (beam, expand) pair.
"""
import torch
import numpy as np
import csv
import time
import hashlib
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

X = torch.tensor(X_np, device=device)          # [N, 48]
pred = torch.tensor(pred_np, device=device)     # [N]

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth", weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97) if pieces[i]["weight"].shape == (48, 96)])
last_id = 85

W_last = pieces[last_id]["weight"]  # [1, 48]
b_last = pieces[last_id]["bias"]    # [1]

# Pre-stack weights for batched computation
# expand: W [48, 96, 48], b [48, 96] (indexed by position in expand_ids)
expand_W_stack = torch.stack([pieces[i]["weight"] for i in expand_ids])  # [48, 96, 48]
expand_b_stack = torch.stack([pieces[i]["bias"] for i in expand_ids])    # [48, 96]
contract_W_stack = torch.stack([pieces[i]["weight"] for i in contract_ids])  # [48, 48, 96]
contract_b_stack = torch.stack([pieces[i]["bias"] for i in contract_ids])    # [48, 48]

ei2idx = {eid: i for i, eid in enumerate(expand_ids)}
ci2idx = {cid: j for j, cid in enumerate(contract_ids)}

N = len(X)
w_last = W_last.squeeze(0)  # [48]


@torch.no_grad()
def score_h(h):
    """Score hidden state by LastLayer MSE. h is [N, 48]."""
    out = h @ w_last + b_last[0]  # [N]
    return torch.mean((out - pred) ** 2).item()


@torch.no_grad()
def eval_candidates_batched(h, avail_exp_idx, avail_con_idx):
    """
    For a single beam hidden state h [N, 48], evaluate all (expand, contract) candidates.
    Returns list of (mse, exp_pos, con_pos) where pos is index into avail lists.
    """
    results = []
    E = len(avail_exp_idx)
    C = len(avail_con_idx)

    # Stack available contract weights/biases
    con_W = contract_W_stack[avail_con_idx]  # [C, 48, 96]
    con_b = contract_b_stack[avail_con_idx]  # [C, 48]

    for e_pos, e_idx in enumerate(avail_exp_idx):
        # Apply expand: h @ W_e^T + b_e → [N, 96]
        z = h @ expand_W_stack[e_idx].T + expand_b_stack[e_idx]  # [N, 96]
        z = torch.relu(z)  # [N, 96]

        # Apply ALL available contracts at once:
        # z @ con_W.transpose(1,2) → [N, 96] @ [C, 96, 48] → need einsum
        # h_new = h + z @ W_c^T + b_c for each c
        h_new = torch.einsum('nr,cdr->cnd', z, con_W) + con_b[:, None, :]  # [C, N, 48]
        h_new = h_new + h[None, :, :]  # [C, N, 48] — add residual

        # Score all C candidates at once
        out = torch.einsum('cnd,d->cn', h_new, w_last) + b_last[0]  # [C, N]
        mse = torch.mean((out - pred[None, :]) ** 2, dim=1)  # [C]

        for c_pos in range(C):
            results.append((mse[c_pos].item(), e_pos, c_pos))

    return results


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
def apply_block(h, ei, ci):
    z = h @ expand_W_stack[ei2idx[ei]].T + expand_b_stack[ei2idx[ei]]
    z = torch.relu(z)
    z = z @ contract_W_stack[ci2idx[ci]].T + contract_b_stack[ci2idx[ci]]
    return h + z


BEAM_WIDTH = 200
N_BLOCKS = 48

print(f"Beam width: {BEAM_WIDTH}", flush=True)
print(f"Expand pieces: {len(expand_ids)}, Contract pieces: {len(contract_ids)}", flush=True)
print(flush=True)

start_time = time.time()

# Beam entries: list of (h [N,48], used_exp_set, used_con_set, block_list)
beam = [(X.clone(), set(), set(), [])]

for step in range(N_BLOCKS):
    step_start = time.time()

    # Collect all candidates across all beam entries
    all_candidates = []  # (mse, beam_idx, expand_id, contract_id)

    for beam_idx, (h, used_exp, used_con, blocks) in enumerate(beam):
        avail_exp = [ei2idx[e] for e in expand_ids if e not in used_exp]
        avail_con = [ci2idx[c] for c in contract_ids if c not in used_con]

        if not avail_exp or not avail_con:
            continue

        avail_exp_t = torch.tensor(avail_exp, device=device, dtype=torch.long)
        avail_con_t = torch.tensor(avail_con, device=device, dtype=torch.long)

        results = eval_candidates_batched(h, avail_exp_t, avail_con_t)

        for mse, e_pos, c_pos in results:
            ei = expand_ids[avail_exp[e_pos]]
            ci = contract_ids[avail_con[c_pos]]
            all_candidates.append((mse, beam_idx, ei, ci))

    # Sort by MSE
    all_candidates.sort(key=lambda x: x[0])

    # Build new beam: top BEAM_WIDTH distinct entries
    new_beam = []
    for mse, beam_idx, ei, ci in all_candidates:
        if len(new_beam) >= BEAM_WIDTH:
            break

        old_h, old_used_exp, old_used_con, old_blocks = beam[beam_idx]
        h_new = apply_block(old_h, ei, ci)
        new_used_exp = old_used_exp | {ei}
        new_used_con = old_used_con | {ci}
        new_blocks = old_blocks + [(ei, ci)]

        new_beam.append((h_new, new_used_exp, new_used_con, new_blocks))

    beam = new_beam

    step_time = time.time() - step_start
    best_mse = score_h(beam[0][0])
    worst_mse = score_h(beam[-1][0]) if len(beam) > 1 else best_mse

    # Count unique latest blocks
    unique_latest = len(set(b[-1] for _, _, _, b in beam))

    # How many unique prefixes?
    if step >= 1:
        unique_prefixes = len(set(tuple(b[:step]) for _, _, _, b in beam))
    else:
        unique_prefixes = len(beam)

    gap = ""
    if len(beam) > 1:
        second_mse = score_h(beam[1][0])
        gap = f"gap={second_mse - best_mse:.6f}"

    print(f"  Step {step+1:2d}/{N_BLOCKS}: beam={len(beam):3d} "
          f"best={best_mse:.6f} worst={worst_mse:.6f} "
          f"unique_last={unique_latest} {gap} "
          f"({step_time:.1f}s)",
          flush=True)

elapsed = time.time() - start_time
print(f"\nBeam search done in {elapsed:.1f}s", flush=True)

# Report top results
print(f"\nTop 10 results:", flush=True)
for i, (h, ue, uc, blocks) in enumerate(beam[:10]):
    mse = score_h(h)
    match, perm_str = check_hash(blocks)
    print(f"  #{i+1}: MSE={mse:.10f} hash={match}", flush=True)
    if i < 3:
        print(f"    Blocks: {blocks}", flush=True)
    if match:
        print(f"\n*** SOLUTION FOUND! ***", flush=True)
        print(f"Permutation: {perm_str}", flush=True)

# Check consensus prefix length
if len(beam) > 1:
    ref = beam[0][3]
    for plen in range(N_BLOCKS, 0, -1):
        if all(entry[3][:plen] == ref[:plen] for entry in beam):
            print(f"\nAll {len(beam)} beams agree on first {plen} blocks", flush=True)
            print(f"Consensus prefix: {ref[:plen]}", flush=True)
            break

# Save
best_blocks = beam[0][3]
with open("beam_gpu_results.json", "w") as f:
    json.dump({"blocks": best_blocks, "mse": score_h(beam[0][0])}, f)
print(f"Results saved to beam_gpu_results.json", flush=True)
