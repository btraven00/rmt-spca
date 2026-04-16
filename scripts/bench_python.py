#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///
"""
Benchmark the Python reference implementation on the same sizes as the
Rust criterion benchmarks, for apples-to-apples comparison.

Usage:
    uv run scripts/bench_python.py
"""
import time
import numpy as np
import scipy.integrate
import scipy.optimize


# ── Biwhitening (from ../spcarmt/scrna/rmt/_covariance.py) ──────────────────

def biwhitening(X, max_iter=1000, tol=1e-5):
    P = X**2
    p = P.shape[1]
    n = P.shape[0]
    r = p * np.ones((n, 1))
    c = n * np.ones((p, 1))
    _D1 = np.diag(np.squeeze(r))
    _D2 = np.diag(np.squeeze(c))
    F1 = 0
    F2 = 0
    for i in range(max_iter):
        Fold1 = F1
        Fold2 = F2
        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        target_col = 1 + _X.mean(axis=0)[:, None]**2
        c = n * target_col / P.T.dot(r)
        _D2 = np.diag(np.squeeze(c))
        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        target_row = 1 + _X.mean(axis=1)[:, None]**2
        r = p * target_row / P.dot(c)
        _D1 = np.diag(np.squeeze(r))
        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        F1 = np.abs(_X.var(axis=0) - 1)
        F2 = np.abs(_X.var(axis=1) - 1)
        lim1 = np.max(np.abs((F1 - Fold1)))
        lim2 = np.max(np.abs((F2 - Fold2)))
        if max(lim1, lim2) < tol:
            break
    D1 = np.diag(np.sqrt(np.squeeze(r)))
    D2 = np.diag(np.sqrt(np.squeeze(c)))
    return D1, D2


# ── FISTA SPCA (from ../spcarmt/scrna/prc/_spca.py) ─────────────────────────

def _gs_decorrelation(w, W):
    w -= np.linalg.multi_dot([w, W.T, W])
    return w

def _gs_orthogonalize(W):
    for k in range(W.shape[1]):
        w = W[:, k]
        w = _gs_decorrelation(w, W[:, :k].T)
        w /= np.sqrt((w**2).sum())
        W[:, k] = w
    return W

def _fista_spca(X, n_comps, max_iter=10000, penalty=0.01, tol=1e-3):
    tk = 1
    p_c = 1/20
    q_c = 1
    r_c = 4
    S = X.T @ X / (X.shape[0] - 1)
    s, V = np.linalg.eigh(S)
    lmax = np.max(s)
    W = V[:, -n_comps:][:, ::-1]
    W_temp = W.copy()

    def _prox(S, W, t):
        W_ = W + 2*t*S@W
        W_ = np.maximum((np.abs(W_) - penalty*t), 0) * np.sign(W_)
        return W_

    t = 0.5 / (2*lmax)
    for i in range(max_iter):
        W_old = np.copy(W)
        W = _prox(S, W_temp, t)
        W = _gs_orthogonalize(W)
        lim = np.sqrt(np.trace((W - W_old) @ (W - W_old).T))
        tk_temp = (p_c + np.sqrt(q_c + r_c*tk**2)) / 2
        W_temp = W + (tk - 1)*(W - W_old) / tk_temp
        tk = tk_temp
        if lim < tol:
            break
    return W


# ── Synthetic data (matching Rust bench) ─────────────────────────────────────

def synthetic_scrna(n, p, k_sig=3, snr=2.0, seed=42, sparsity=0.8):
    rng = np.random.RandomState(seed)
    q = p / n
    lp = (1 + np.sqrt(q))**2
    signal_scale = np.sqrt(snr * lp)
    dirs = []
    for _ in range(k_sig):
        v = rng.randn(p)
        v /= np.linalg.norm(v)
        dirs.append(v)
    X = np.abs(rng.randn(n, p))
    for i in range(n):
        for d in dirs:
            X[i] += signal_scale * d[i % p]
    mask = rng.rand(n, p) < sparsity
    X[mask] = 0
    return np.abs(X)


# ── Benchmarks ───────────────────────────────────────────────────────────────

def bench(label, fn, repeats=3):
    # warmup
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    print(f"  {label:40s}  {med*1000:>10.1f} ms  (median of {repeats})")
    return med


print("=" * 70)
print("Python reference benchmark")
print("=" * 70)

for n, p in [(500, 2000), (1000, 5000)]:
    print(f"\n--- {n} x {p} ---")
    X = synthetic_scrna(n, p)

    t_bw = bench(f"biwhitening {n}x{p}", lambda: biwhitening(X))

    D1, D2 = biwhitening(X)
    Xw = D1 @ X @ D2
    Xwc = Xw - Xw.mean(axis=0)

    bench(f"covariance X^T X / (n-1) {n}x{p}", lambda: Xwc.T @ Xwc / (n-1))

    bench(f"full EVD {p}x{p}", lambda: np.linalg.eigh(Xwc.T @ Xwc / (n-1)))

    # FISTA on the biwhitened, centred data
    bench(f"FISTA SPCA (k=3) {n}x{p}",
          lambda: _fista_spca(Xwc, n_comps=3, penalty=0.01, max_iter=1000))

print()
