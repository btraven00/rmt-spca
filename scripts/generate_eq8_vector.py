#!/usr/bin/env python3
"""Reference Eq. 8 penalty selection on the spiked test vector, for cross-validation.

Ports `_fista_spca`, `_gs_orthogonalize` and the root-find of `_adjust_penalty`
from ../spcarmt/scrna/prc/_spca.py (Victor Chardès, MIT) with the class plumbing
removed: given S = XᵀX/(n−1) from test_vectors/spiked_model.json and a fixed
target distance, find γ* such that Σ(1 − σ²) of WᵀV equals the target, with V
the top-k eigenvectors. Writes test_vectors/eq8_penalty.json.

Usage: python scripts/generate_eq8_vector.py
"""
import json, os
import numpy as np, scipy.optimize

HERE = os.path.dirname(__file__)
VEC = os.path.join(HERE, "..", "test_vectors")


def _gs_orthogonalize(W):
    for k in range(W.shape[1]):
        w = W[:, k]
        w -= np.linalg.multi_dot([w, W[:, :k], W[:, :k].T]) if k else 0.0
        w /= np.sqrt((w**2).sum())
        W[:, k] = w
    return W


def _fista_spca(S, V, n_comps, max_iter=10000, penalty=0.01, tol=1e-3):
    tk, p, q, r = 1, 1 / 20, 1, 4
    lmax = np.linalg.eigvalsh(S).max()
    W = V[:, :n_comps].copy()
    W_temp = W.copy()
    t = 0.5 / (2 * lmax)
    for _ in range(max_iter):
        W_old = W.copy()
        W_ = W_temp + 2 * t * S @ W_temp
        W_ = np.maximum(np.abs(W_) - penalty * t, 0) * np.sign(W_)
        W = _gs_orthogonalize(W_)
        lim = np.sqrt(np.trace((W - W_old) @ (W - W_old).T))
        tk_temp = (p + np.sqrt(q + r * tk**2)) / 2
        W_temp = W + (tk - 1) * (W - W_old) / tk_temp
        tk = tk_temp
        if lim < tol:
            break
    return W


d = json.load(open(os.path.join(VEC, "spiked_model.json")))
X = np.array(d["X"]); n, p = X.shape; k = d["k"]
S = X.T @ X / (n - 1)
s, V = np.linalg.eigh(S); V = V[:, ::-1][:, :k]


def distance(penalty):
    W = _fista_spca(S, V, k, penalty=penalty)
    sv = np.linalg.svd(W.T @ V, compute_uv=False)
    return float((1 - sv**2).sum())


d_hi = distance(4.0)
target = 0.5 * d_hi if d_hi < 0.4 else 0.2         # make sure a root exists in the reference bracket
res = scipy.optimize.root_scalar(lambda g: distance(g) - target, bracket=(1e-4, 4), xtol=5e-3)
out = dict(n=n, p=p, k=k, target=target, gamma_star=res.root, distance_at_star=distance(res.root),
           distance_at_06=distance(0.6 * res.root), distance_at_4=d_hi)
json.dump(out, open(os.path.join(VEC, "eq8_penalty.json"), "w"), indent=2)
print(out)
