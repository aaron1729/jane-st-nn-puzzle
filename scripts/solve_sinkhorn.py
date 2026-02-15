"""
Sinkhorn permutation relaxation for the Jane Street neural net puzzle.

Optimizes two global assignment matrices:
  - P_in  (position -> expand layer)
  - P_out (position -> contract layer)

Each is obtained by applying Sinkhorn normalization to learnable logits.
After optimization, assignments are projected to exact permutations with
Hungarian assignment and evaluated as a discrete 48-block network.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - runtime dependency check
    linear_sum_assignment = None


TARGET_HASH = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"
N_BLOCKS = 48
LAST_ID = 85


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row])
    data = np.array(rows, dtype=np.float32)
    x = torch.tensor(data[:, :48], device=device)
    pred = torch.tensor(data[:, 48], device=device)
    return x, pred


def load_pieces(pieces_dir: str, device: torch.device):
    pieces = {}
    for i in range(97):
        pieces[i] = torch.load(
            f"{pieces_dir}/piece_{i}.pth",
            weights_only=False,
            map_location=device,
        )

    expand_ids = sorted(
        [i for i in range(97) if pieces[i]["weight"].shape == (96, 48)]
    )
    contract_ids = sorted(
        [i for i in range(97) if pieces[i]["weight"].shape == (48, 96)]
    )

    expand_w = torch.stack([pieces[i]["weight"] for i in expand_ids])  # [48,96,48]
    expand_b = torch.stack([pieces[i]["bias"] for i in expand_ids])  # [48,96]
    contract_w = torch.stack([pieces[i]["weight"] for i in contract_ids])  # [48,48,96]
    contract_b = torch.stack([pieces[i]["bias"] for i in contract_ids])  # [48,48]

    w_last = pieces[LAST_ID]["weight"].squeeze(0)  # [48]
    b_last = pieces[LAST_ID]["bias"][0]  # scalar tensor

    return expand_ids, contract_ids, expand_w, expand_b, contract_w, contract_b, w_last, b_last


def sinkhorn_from_logits(logits: torch.Tensor, tau: float, n_iters: int) -> torch.Tensor:
    # log-space Sinkhorn for numerical stability
    log_p = logits / tau
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    return torch.exp(log_p)


def row_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=1)


def forward_soft(
    x: torch.Tensor,
    p_in: torch.Tensor,
    p_out: torch.Tensor,
    expand_w: torch.Tensor,
    expand_b: torch.Tensor,
    contract_w: torch.Tensor,
    contract_b: torch.Tensor,
    w_last: torch.Tensor,
    b_last: torch.Tensor,
) -> torch.Tensor:
    # Blend layer weights/biases per block position using assignment rows.
    a_w = torch.einsum("ki,iar->kar", p_in, expand_w)  # [48,96,48]
    a_b = torch.einsum("ki,ia->ka", p_in, expand_b)  # [48,96]
    b_w = torch.einsum("kj,jdr->kdr", p_out, contract_w)  # [48,48,96]
    b_b = torch.einsum("kj,jd->kd", p_out, contract_b)  # [48,48]

    h = x
    for k in range(N_BLOCKS):
        z = h @ a_w[k].T + a_b[k]
        z = torch.relu(z)
        delta = z @ b_w[k].T + b_b[k]
        h = h + delta
    return h @ w_last + b_last


@torch.no_grad()
def discrete_mse(
    x: torch.Tensor,
    pred: torch.Tensor,
    order: Sequence[Tuple[int, int]],
    expand_ids: Sequence[int],
    contract_ids: Sequence[int],
    expand_w: torch.Tensor,
    expand_b: torch.Tensor,
    contract_w: torch.Tensor,
    contract_b: torch.Tensor,
    w_last: torch.Tensor,
    b_last: torch.Tensor,
) -> float:
    ei2pos = {eid: i for i, eid in enumerate(expand_ids)}
    ci2pos = {cid: j for j, cid in enumerate(contract_ids)}

    h = x.clone()
    for ei, ci in order:
        ep = ei2pos[ei]
        cp = ci2pos[ci]
        z = h @ expand_w[ep].T + expand_b[ep]
        z = torch.relu(z)
        delta = z @ contract_w[cp].T + contract_b[cp]
        h = h + delta
    out = h @ w_last + b_last
    return torch.mean((out - pred) ** 2).item()


def project_to_permutation(p: torch.Tensor) -> List[int]:
    if linear_sum_assignment is None:
        raise RuntimeError(
            "scipy is required for Hungarian projection. Install scipy to project."
        )
    scores = p.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-scores)
    cols = np.empty(scores.shape[0], dtype=np.int64)
    cols[row_ind] = col_ind
    return cols.tolist()


def build_order(
    in_cols: Sequence[int],
    out_cols: Sequence[int],
    expand_ids: Sequence[int],
    contract_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    return [(expand_ids[in_cols[k]], contract_ids[out_cols[k]]) for k in range(N_BLOCKS)]


def check_hash(order: Sequence[Tuple[int, int]]) -> Tuple[bool, str]:
    perm = []
    for ei, ci in order:
        perm.append(ei)
        perm.append(ci)
    perm.append(LAST_ID)
    perm_str = ",".join(str(p) for p in perm)
    h = hashlib.sha256(perm_str.encode()).hexdigest()
    return h == TARGET_HASH, perm_str


def temp_at_step(step: int, total_steps: int, tau_start: float, tau_end: float) -> float:
    if total_steps <= 1:
        return tau_end
    frac = (step - 1) / (total_steps - 1)
    return tau_start * ((tau_end / tau_start) ** frac)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinkhorn permutation solver")
    parser.add_argument("--data-csv", default="data/historical_data.csv")
    parser.add_argument("--pieces-dir", default="data/pieces")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--lambda-ent", type=float, default=1e-3)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--train-subsample", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-json", default="sinkhorn_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    x_full, pred_full = load_data(args.data_csv, device)
    (
        expand_ids,
        contract_ids,
        expand_w,
        expand_b,
        contract_w,
        contract_b,
        w_last,
        b_last,
    ) = load_pieces(args.pieces_dir, device)

    if args.train_subsample > 0 and args.train_subsample < x_full.shape[0]:
        idx = np.random.choice(x_full.shape[0], size=args.train_subsample, replace=False)
        idx.sort()
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        x_train = x_full.index_select(0, idx_t)
        pred_train = pred_full.index_select(0, idx_t)
    else:
        x_train = x_full
        pred_train = pred_full

    print(
        f"Train rows: {x_train.shape[0]} / Full rows: {x_full.shape[0]}, "
        f"steps={args.steps}, lr={args.lr}",
        flush=True,
    )

    z_in = torch.nn.Parameter(args.init_scale * torch.randn(N_BLOCKS, N_BLOCKS, device=device))
    z_out = torch.nn.Parameter(args.init_scale * torch.randn(N_BLOCKS, N_BLOCKS, device=device))
    optimizer = torch.optim.Adam([z_in, z_out], lr=args.lr)

    best_discrete_mse = float("inf")
    best_order: List[Tuple[int, int]] = []
    best_in_cols: List[int] = []
    best_out_cols: List[int] = []

    start = time.time()

    for step in range(1, args.steps + 1):
        tau = temp_at_step(step, args.steps, args.tau_start, args.tau_end)
        p_in = sinkhorn_from_logits(z_in, tau=tau, n_iters=args.sinkhorn_iters)
        p_out = sinkhorn_from_logits(z_out, tau=tau, n_iters=args.sinkhorn_iters)

        pred_soft = forward_soft(
            x_train,
            p_in,
            p_out,
            expand_w,
            expand_b,
            contract_w,
            contract_b,
            w_last,
            b_last,
        )
        loss_pred = torch.mean((pred_soft - pred_train) ** 2)
        ent_in = row_entropy(p_in).mean()
        ent_out = row_entropy(p_out).mean()
        loss = loss_pred + args.lambda_ent * (ent_in + ent_out)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([z_in, z_out], max_norm=args.clip_grad)
        optimizer.step()

        should_log = step == 1 or step == args.steps or (step % args.eval_every == 0)
        if not should_log:
            continue

        with torch.no_grad():
            rowmax_in = p_in.max(dim=1).values
            rowmax_out = p_out.max(dim=1).values
            discrete_eval = None
            hash_ok = None

            if linear_sum_assignment is not None:
                in_cols = project_to_permutation(p_in)
                out_cols = project_to_permutation(p_out)
                order = build_order(in_cols, out_cols, expand_ids, contract_ids)
                discrete_eval = discrete_mse(
                    x_full,
                    pred_full,
                    order,
                    expand_ids,
                    contract_ids,
                    expand_w,
                    expand_b,
                    contract_w,
                    contract_b,
                    w_last,
                    b_last,
                )
                hash_ok, _ = check_hash(order)
                if discrete_eval < best_discrete_mse:
                    best_discrete_mse = discrete_eval
                    best_order = list(order)
                    best_in_cols = list(in_cols)
                    best_out_cols = list(out_cols)

            elapsed = time.time() - start
            msg = (
                f"step {step:5d}/{args.steps} tau={tau:.4f} "
                f"loss={loss.item():.6f} mse_soft={loss_pred.item():.6f} "
                f"ent_in={ent_in.item():.4f} ent_out={ent_out.item():.4f} "
                f"rowmax_in={rowmax_in.mean().item():.4f} "
                f"rowmax_out={rowmax_out.mean().item():.4f}"
            )
            if discrete_eval is not None:
                msg += (
                    f" discrete_mse={discrete_eval:.6f} "
                    f"best_discrete={best_discrete_mse:.6f} "
                    f"hash={hash_ok}"
                )
            msg += f" elapsed={elapsed:.1f}s"
            print(msg, flush=True)

    # Final evaluation at tau_end
    with torch.no_grad():
        tau = args.tau_end
        p_in = sinkhorn_from_logits(z_in, tau=tau, n_iters=args.sinkhorn_iters)
        p_out = sinkhorn_from_logits(z_out, tau=tau, n_iters=args.sinkhorn_iters)
        pred_soft_full = forward_soft(
            x_full,
            p_in,
            p_out,
            expand_w,
            expand_b,
            contract_w,
            contract_b,
            w_last,
            b_last,
        )
        final_soft_mse = torch.mean((pred_soft_full - pred_full) ** 2).item()

        print("\nFinal metrics:", flush=True)
        print(f"  soft_mse_full={final_soft_mse:.6f}", flush=True)
        print(f"  rowmax_in_mean={p_in.max(dim=1).values.mean().item():.4f}", flush=True)
        print(f"  rowmax_out_mean={p_out.max(dim=1).values.mean().item():.4f}", flush=True)
        print(f"  ent_in_mean={row_entropy(p_in).mean().item():.4f}", flush=True)
        print(f"  ent_out_mean={row_entropy(p_out).mean().item():.4f}", flush=True)

        final_discrete_mse = None
        final_hash_ok = None
        final_perm_str = None
        final_in_cols = None
        final_out_cols = None

        if linear_sum_assignment is not None:
            final_in_cols = project_to_permutation(p_in)
            final_out_cols = project_to_permutation(p_out)
            final_order = build_order(final_in_cols, final_out_cols, expand_ids, contract_ids)
            final_discrete_mse = discrete_mse(
                x_full,
                pred_full,
                final_order,
                expand_ids,
                contract_ids,
                expand_w,
                expand_b,
                contract_w,
                contract_b,
                w_last,
                b_last,
            )
            final_hash_ok, final_perm_str = check_hash(final_order)

            print(f"  final_discrete_mse={final_discrete_mse:.6f}", flush=True)
            print(f"  hash_match={final_hash_ok}", flush=True)
            print(
                f"  best_discrete_mse_seen={best_discrete_mse:.6f}"
                if best_order
                else "  best_discrete_mse_seen=nan",
                flush=True,
            )
        else:
            print("  Hungarian projection unavailable (scipy not installed).", flush=True)

    result = {
        "config": {
            "steps": args.steps,
            "lr": args.lr,
            "tau_start": args.tau_start,
            "tau_end": args.tau_end,
            "sinkhorn_iters": args.sinkhorn_iters,
            "lambda_ent": args.lambda_ent,
            "train_subsample": args.train_subsample,
            "seed": args.seed,
            "device": str(device),
        },
        "final": {
            "soft_mse_full": final_soft_mse,
            "discrete_mse": final_discrete_mse,
            "hash_match": final_hash_ok,
            "perm_str": final_perm_str,
            "in_cols": final_in_cols,
            "out_cols": final_out_cols,
        },
        "best_seen": {
            "discrete_mse": best_discrete_mse if best_order else None,
            "order": best_order if best_order else None,
            "in_cols": best_in_cols if best_in_cols else None,
            "out_cols": best_out_cols if best_out_cols else None,
        },
    }

    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.save_json}", flush=True)


if __name__ == "__main__":
    main()
