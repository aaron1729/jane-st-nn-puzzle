"""
Quick diagnostic: is (pred - true) predictable from x?
If R^2 ~ 0: true adds no information beyond loose global constraint.
If R^2 >> 0: true provides a useful decomposition of pred.
"""
import numpy as np
import csv

data = []
with open("data/historical_data.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data, dtype=np.float32)

X = data[:, :48]
pred = data[:, 48]
true = data[:, 49]
residual = pred - true

N = len(X)
print(f"N = {N}", flush=True)
print(f"MSE(pred, true) = {np.mean(residual**2):.6f}", flush=True)
print(f"Var(pred) = {np.var(pred):.6f}", flush=True)
print(f"Var(true) = {np.var(true):.6f}", flush=True)
print(f"Var(residual) = {np.var(residual):.6f}", flush=True)
print(f"Corr(pred, true) = {np.corrcoef(pred, true)[0,1]:.6f}", flush=True)

# Split train/test
np.random.seed(42)
idx = np.random.permutation(N)
n_train = N * 4 // 5
train_idx = idx[:n_train]
test_idx = idx[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
r_train, r_test = residual[train_idx], residual[test_idx]
pred_train, pred_test = pred[train_idx], pred[test_idx]
true_train, true_test = true[train_idx], true[test_idx]

# ============================================================
# Ridge regression: x -> residual
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Ridge regression: x -> (pred - true)", flush=True)
print(f"{'='*60}", flush=True)

from numpy.linalg import lstsq

# Add bias column
X_train_b = np.hstack([X_train, np.ones((len(X_train), 1))])
X_test_b = np.hstack([X_test, np.ones((len(X_test), 1))])

for alpha in [0, 0.01, 0.1, 1.0, 10.0]:
    if alpha == 0:
        w, _, _, _ = lstsq(X_train_b, r_train, rcond=None)
    else:
        A = X_train_b.T @ X_train_b + alpha * np.eye(X_train_b.shape[1])
        w = np.linalg.solve(A, X_train_b.T @ r_train)

    r_pred_train = X_train_b @ w
    r_pred_test = X_test_b @ w

    mse_train = np.mean((r_pred_train - r_train)**2)
    mse_test = np.mean((r_pred_test - r_test)**2)
    r2_train = 1 - mse_train / np.var(r_train)
    r2_test = 1 - mse_test / np.var(r_test)

    print(f"  alpha={alpha:>5.2f}: R²_train={r2_train:.4f}  R²_test={r2_test:.4f}  "
          f"MSE_test={mse_test:.6f}", flush=True)

# ============================================================
# Also: x -> pred and x -> true (for comparison)
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Ridge regression: x -> pred, x -> true (for comparison)", flush=True)
print(f"{'='*60}", flush=True)

for target_name, y_train, y_test in [("pred", pred_train, pred_test),
                                      ("true", true_train, true_test)]:
    A = X_train_b.T @ X_train_b + 0.1 * np.eye(X_train_b.shape[1])
    w = np.linalg.solve(A, X_train_b.T @ y_train)
    y_hat = X_test_b @ w
    mse = np.mean((y_hat - y_test)**2)
    r2 = 1 - mse / np.var(y_test)
    print(f"  x -> {target_name}: R²={r2:.4f}  MSE={mse:.6f}", flush=True)

# ============================================================
# Sign pattern analysis: is residual constant within sign patterns?
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Sign pattern analysis of residual", flush=True)
print(f"{'='*60}", flush=True)

signs = np.sign(X)
sign_strs = [''.join(['1' if s > 0 else '0' for s in row]) for row in signs]

from collections import defaultdict
pattern_residuals = defaultdict(list)
for i, s in enumerate(sign_strs):
    pattern_residuals[s].append(residual[i])

# Within-pattern vs between-pattern variance
within_vars = []
pattern_means = []
for s, vals in pattern_residuals.items():
    if len(vals) > 1:
        within_vars.append(np.var(vals))
    pattern_means.append(np.mean(vals))

within_var = np.mean(within_vars) if within_vars else 0
between_var = np.var(pattern_means)
total_var = np.var(residual)

print(f"  Total variance of residual: {total_var:.6f}", flush=True)
print(f"  Between-pattern variance:   {between_var:.6f}", flush=True)
print(f"  Within-pattern variance:    {within_var:.6f}", flush=True)
print(f"  SNR (between/within):       {between_var/within_var:.1f}x" if within_var > 0 else "  SNR: inf", flush=True)
print(f"  Fraction explained by sign: {between_var/total_var:.4f}", flush=True)

# ============================================================
# Interaction features (sign patterns have 2^48 possible values,
# so linear in x can't capture them — try quadratic)
# ============================================================
print(f"\n{'='*60}", flush=True)
print("Quadratic features: x -> (pred - true)", flush=True)
print(f"{'='*60}", flush=True)

# Create x_i * x_j features (upper triangle)
n_feat = 48
quad_feats_train = []
quad_feats_test = []
for i in range(n_feat):
    for j in range(i, n_feat):
        quad_feats_train.append(X_train[:, i] * X_train[:, j])
        quad_feats_test.append(X_test[:, i] * X_test[:, j])

Q_train = np.column_stack([X_train] + quad_feats_train + [np.ones(len(X_train))])
Q_test = np.column_stack([X_test] + quad_feats_test + [np.ones(len(X_test))])

print(f"  Quadratic feature dim: {Q_train.shape[1]}", flush=True)

for alpha in [1.0, 10.0, 100.0]:
    A = Q_train.T @ Q_train + alpha * np.eye(Q_train.shape[1])
    w = np.linalg.solve(A, Q_train.T @ r_train)
    r_hat = Q_test @ w
    mse = np.mean((r_hat - r_test)**2)
    r2 = 1 - mse / np.var(r_test)
    print(f"  alpha={alpha:>5.1f}: R²={r2:.4f}  MSE={mse:.6f}", flush=True)

# Same for pred and true
print(f"\n  Quadratic: x -> pred, x -> true (comparison)", flush=True)
for target_name, y_train, y_test in [("pred", pred_train, pred_test),
                                      ("true", true_train, true_test)]:
    A = Q_train.T @ Q_train + 10.0 * np.eye(Q_train.shape[1])
    w = np.linalg.solve(A, Q_train.T @ y_train)
    y_hat = Q_test @ w
    mse = np.mean((y_hat - y_test)**2)
    r2 = 1 - mse / np.var(y_test)
    print(f"  x -> {target_name}: R²={r2:.4f}  MSE={mse:.6f}", flush=True)

print(f"\n{'='*60}", flush=True)
print("VERDICT", flush=True)
print(f"{'='*60}", flush=True)
print("If R²(x -> residual) ~ 0: true is just noise, won't help.", flush=True)
print("If R² >> 0: residual has structure, whitening may sharpen scoring.", flush=True)
