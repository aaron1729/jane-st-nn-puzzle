"""
solve_beam_wide.py — Wide beam search with diversity forcing.

Features:
- Large beam width (default 5000)
- Fused MSE projection: avoids materializing [B, C, N, 48] tensors
- Output-signature diversity to prevent beam collapse
- Subsampled scoring (default 2000) for speed
- Final full-data re-scoring of top results

Usage:
  python3 solve_beam_wide.py
  python3 solve_beam_wide.py --width 2000 --subsample 5000
  python3 solve_beam_wide.py --max-per-bucket 0   # disable diversity
"""
import torch
import numpy as np
import csv
import time
import hashlib
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=5000)
parser.add_argument("--subsample", type=int, default=2000)
parser.add_argument("--max-per-bucket", type=int, default=100,
                    help="Max beam entries per signature bucket (0=disabled)")
parser.add_argument("--sig-bits", type=int, default=32)
parser.add_argument("--chunk", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--top-k-factor", type=int, default=10)
args = parser.parse_args()

BEAM_WIDTH = args.width
N_SUB = args.subsample
MAX_PER_BUCKET = args.max_per_bucket
SIG_BITS = args.sig_bits
CHUNK = args.chunk
SEED = args.seed
N_BLOCKS = 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Config: width={BEAM_WIDTH} sub={N_SUB} bucket={MAX_PER_BUCKET} "
      f"sig={SIG_BITS} chunk={CHUNK} seed={SEED}", flush=True)

# ── Load data ────────────────────────────────────────────────────
t0 = time.time()
data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)
N_FULL = len(data)
X_full = torch.tensor(data[:, :48], device=device)
pred_full = torch.tensor(data[:, 48], device=device)

rng = np.random.RandomState(SEED)
sub_idx = rng.choice(N_FULL, N_SUB, replace=False)
sub_idx.sort()
X_sub = X_full[sub_idx]
pred_sub = pred_full[sub_idx]

sig_idx = torch.tensor(rng.choice(N_SUB, SIG_BITS, replace=False),
                        device=device, dtype=torch.long)
pred_sig = pred_sub[sig_idx]

pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"data/pieces/piece_{i}.pth",
                            weights_only=False, map_location=device)

expand_ids = sorted([i for i in range(97)
                     if pieces[i]["weight"].shape == (96, 48)])
contract_ids = sorted([i for i in range(97)
                       if pieces[i]["weight"].shape == (48, 96)])
last_id = 85
w_last = pieces[last_id]["weight"].squeeze(0)   # [48]
b_last = pieces[last_id]["bias"][0].item()       # scalar

expand_W = torch.stack([pieces[i]["weight"] for i in expand_ids])    # [48,96,48]
expand_b = torch.stack([pieces[i]["bias"] for i in expand_ids])      # [48,96]
contract_W = torch.stack([pieces[i]["weight"] for i in contract_ids])# [48,48,96]
contract_b = torch.stack([pieces[i]["bias"] for i in contract_ids])  # [48,48]

ei2pos = {eid: i for i, eid in enumerate(expand_ids)}
ci2pos = {cid: j for j, cid in enumerate(contract_ids)}

# ── Fused projections ───────────────────────────────────────────
# out = (h + delta) @ w_last + b_last
#     = h @ w_last + b_last  +  z @ contract_proj + contract_bias_proj
# Avoids materializing [B, C, N, 48] tensor.
contract_proj = torch.einsum('cdr,d->cr', contract_W, w_last)  # [48,96]
contract_bias_proj = contract_b @ w_last                         # [48]

sig_powers = 2 ** torch.arange(SIG_BITS, device=device, dtype=torch.int64)

print(f"Loaded {N_FULL} rows, scoring on {N_SUB}. "
      f"{len(expand_ids)} expand, {len(contract_ids)} contract. "
      f"({time.time()-t0:.1f}s)", flush=True)

# ── Helpers ──────────────────────────────────────────────────────
@torch.no_grad()
def apply_block(h, ei, ci):
    e, c = ei2pos[ei], ci2pos[ci]
    z = h @ expand_W[e].T + expand_b[e]
    z = torch.relu(z)
    return h + z @ contract_W[c].T + contract_b[c]

def check_hash(blocks):
    perm = []
    for ei, ci in blocks:
        perm += [ei, ci]
    perm.append(last_id)
    s = ",".join(str(p) for p in perm)
    h = hashlib.sha256(s.encode()).hexdigest()
    return h == "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4", s

# ── Beam search ──────────────────────────────────────────────────
beam_h = [X_sub.clone()]
beam_used_exp = [set()]
beam_used_con = [set()]
beam_blocks = [[]]

start_time = time.time()

for step in range(N_BLOCKS):
    step_t = time.time()
    B = len(beam_h)
    n_avail = N_BLOCKS - step

    # Availability masks
    avail_exp = np.ones((B, 48), dtype=bool)
    avail_con = np.ones((B, 48), dtype=bool)
    for bi in range(B):
        for ep, eid in enumerate(expand_ids):
            if eid in beam_used_exp[bi]:
                avail_exp[bi, ep] = False
        for cp, cid in enumerate(contract_ids):
            if cid in beam_used_con[bi]:
                avail_con[bi, cp] = False
    avail_con_t = torch.tensor(avail_con, device=device)

    # Stack all hidden states and pre-compute h_base
    all_h = torch.stack(beam_h)                        # [B, N_sub, 48]
    h_base_all = all_h @ w_last + b_last               # [B, N_sub]

    # Candidate storage
    max_cands = B * n_avail * n_avail
    c_mse = np.full(max_cands, np.inf, dtype=np.float32)
    c_bi  = np.zeros(max_cands, dtype=np.int32)
    c_ei  = np.zeros(max_cands, dtype=np.int32)
    c_ci  = np.zeros(max_cands, dtype=np.int32)
    c_sig = np.zeros(max_cands, dtype=np.int64)
    ptr = 0

    for e_pos in range(48):
        batch_bis = np.where(avail_exp[:, e_pos])[0]
        if len(batch_bis) == 0:
            continue
        e_id = expand_ids[e_pos]

        for cs in range(0, len(batch_bis), CHUNK):
            cb = batch_bis[cs:cs + CHUNK]
            nb = len(cb)

            h_ch = all_h[cb]                            # [nb, N, 48]
            hb_ch = h_base_all[cb]                      # [nb, N]
            cm_ch = avail_con_t[cb]                     # [nb, 48]

            # Expand
            z = torch.einsum('bnd,rd->bnr', h_ch, expand_W[e_pos])
            z = z + expand_b[e_pos]
            z = torch.relu(z)                           # [nb, N, 96]

            # Fused contract → scalar output
            d_out = torch.einsum('bnr,cr->bcn', z, contract_proj)
            d_out = d_out + contract_bias_proj[None, :, None]  # [nb, 48, N]
            out = hb_ch[:, None, :] + d_out             # [nb, 48, N]

            # MSE
            mse = torch.mean((out - pred_sub[None, None, :]) ** 2, dim=2)  # [nb,48]
            mse[~cm_ch] = float('inf')

            # Signatures
            out_s = out[:, :, sig_idx]                  # [nb, 48, SIG_BITS]
            sb = (out_s > pred_sig[None, None, :]).long()
            si = (sb * sig_powers[None, None, :]).sum(dim=2)  # [nb, 48]

            mse_np = mse.cpu().numpy()
            sig_np = si.cpu().numpy()

            for k, bi in enumerate(cb):
                for cp in range(48):
                    if not avail_con[bi, cp]:
                        continue
                    c_mse[ptr] = mse_np[k, cp]
                    c_bi[ptr]  = bi
                    c_ei[ptr]  = e_id
                    c_ci[ptr]  = contract_ids[cp]
                    c_sig[ptr] = sig_np[k, cp]
                    ptr += 1

            del h_ch, z, d_out, out, mse, out_s, sb, si

    del all_h, h_base_all, avail_con_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Pre-filter top candidates
    c_mse = c_mse[:ptr]; c_bi = c_bi[:ptr]; c_ei = c_ei[:ptr]
    c_ci = c_ci[:ptr]; c_sig = c_sig[:ptr]

    pre_k = min(args.top_k_factor * BEAM_WIDTH, ptr)
    if pre_k < ptr:
        top_idx = np.argpartition(c_mse, pre_k)[:pre_k]
        top_idx = top_idx[np.argsort(c_mse[top_idx])]
    else:
        top_idx = np.argsort(c_mse)

    # Diversity-forced selection
    new_h, new_ue, new_uc, new_bl = [], [], [], []
    bucket = defaultdict(int)
    skips = 0

    for i in top_idx:
        if len(new_h) >= BEAM_WIDTH:
            break
        bi, ei, ci, sig = int(c_bi[i]), int(c_ei[i]), int(c_ci[i]), int(c_sig[i])
        if MAX_PER_BUCKET > 0 and bucket[sig] >= MAX_PER_BUCKET:
            skips += 1
            continue
        bucket[sig] += 1
        h_new = apply_block(beam_h[bi], ei, ci)
        new_h.append(h_new)
        new_ue.append(beam_used_exp[bi] | {ei})
        new_uc.append(beam_used_con[bi] | {ci})
        new_bl.append(beam_blocks[bi] + [(ei, ci)])

    beam_h, beam_used_exp, beam_used_con, beam_blocks = new_h, new_ue, new_uc, new_bl

    # Logging
    best_out = beam_h[0] @ w_last + b_last
    best_mse = torch.mean((best_out - pred_sub) ** 2).item()
    worst_out = beam_h[-1] @ w_last + b_last
    worst_mse = torch.mean((worst_out - pred_sub) ** 2).item()
    n_bkt = len(bucket)
    max_bkt = max(bucket.values()) if bucket else 0
    ulast = len(set(b[-1] for b in beam_blocks))
    dt = time.time() - step_t

    print(f"  Step {step+1:2d}/{N_BLOCKS}: beam={len(beam_h):5d} "
          f"best={best_mse:.6f} worst={worst_mse:.6f} "
          f"uniq_last={ulast:4d} buckets={n_bkt:4d} "
          f"max_bkt={max_bkt:3d} skips={skips} "
          f"cands={ptr} ({dt:.1f}s)", flush=True)

elapsed = time.time() - start_time
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

# ── Re-score top results on full data ────────────────────────────
print(f"\nTop 10 (re-scored on {N_FULL} samples):", flush=True)

# Sort beam by subsample MSE first
beam_sub_mse = []
for h in beam_h:
    o = h @ w_last + b_last
    beam_sub_mse.append(torch.mean((o - pred_sub) ** 2).item())
order = np.argsort(beam_sub_mse)

for rank, idx in enumerate(order[:10]):
    blocks = beam_blocks[idx]
    h = X_full.clone()
    for ei, ci in blocks:
        h = apply_block(h, ei, ci)
    o = h @ w_last + b_last
    mse_full = torch.mean((o - pred_full) ** 2).item()
    match, perm_str = check_hash(blocks)
    print(f"  #{rank+1}: MSE={mse_full:.10f} hash={match}", flush=True)
    if rank < 3:
        print(f"    {blocks}", flush=True)
    if match:
        print(f"\n*** SOLUTION FOUND ***\n{perm_str}", flush=True)

# Save best
best_idx = order[0]
best_blocks = beam_blocks[best_idx]
h = X_full.clone()
for ei, ci in best_blocks:
    h = apply_block(h, ei, ci)
o = h @ w_last + b_last
best_mse = torch.mean((o - pred_full) ** 2).item()

with open("beam_wide_results.json", "w") as f:
    json.dump({"blocks": best_blocks, "mse": best_mse,
               "config": vars(args)}, f)
print(f"\nSaved to beam_wide_results.json (MSE={best_mse:.6f})", flush=True)
