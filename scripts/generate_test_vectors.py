#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///
"""
Generate test vectors from the Python reference implementation for cross-validation
with the Rust rmt-spca crate.

The reference functions are copied verbatim from ../spcarmt/scrna/rmt/ to avoid
importing the full package (which pulls in scanpy, rpy2, sklearn, etc.).

Usage:
    cd /home/b/phd/rmt-spca
    uv run scripts/generate_test_vectors.py
"""
import json
import os
import numpy as np
import scipy.integrate
import scipy.optimize

OUT = os.path.join(os.path.dirname(__file__), '..', 'test_vectors')
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Reference functions — copied from ../spcarmt/scrna/rmt/_covariance.py
# ═══════════════════════════════════════════════════════════════════════════════

def marchenko_pastur_density(x, q):
    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    return ((1/(2*np.pi*x*q)) if (x > 0) else 0) * (
        0 if (x > lambda_max or x < lambda_min)
        else np.sqrt((lambda_max - x) * (x - lambda_min))
    )

def marchenko_pastur_cumulative(x, q):
    lambda_min = (1 - np.sqrt(q))**2
    F = 0
    if q > 1:
        F = 1 - 1/q
    if x >= lambda_min:
        F += scipy.integrate.quad(marchenko_pastur_density, lambda_min, x, args=(q))[0]
    return F

def marchenko_pastur_median(q):
    if q > 1:
        q = 1/q
    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    def _func(x):
        _int = scipy.integrate.quad(marchenko_pastur_density, lambda_min, x, args=(q))[0]
        return _int - 0.5
    return scipy.optimize.brentq(_func, lambda_min, lambda_max, xtol=0.003)


# ═══════════════════════════════════════════════════════════════════════════════
# Reference functions — copied from ../spcarmt/scrna/rmt/_toy.py
# ═══════════════════════════════════════════════════════════════════════════════

def predicted_overlap(pop_spike, gamma, s2=1):
    _pop_spike = pop_spike / s2
    return ((_pop_spike - 1)**2 - gamma) / ((_pop_spike - 1) * (_pop_spike - 1 + gamma))

def predicted_eigval(pop_spike, gamma, s2=1):
    _pop_spike = pop_spike / s2
    return s2 * (_pop_spike + gamma * _pop_spike / (_pop_spike - 1))

def invert_predicted_eigval(sample_spike, gamma, s2=1):
    _sample_spike = sample_spike / s2
    mbar = ((gamma - 1 - _sample_spike) +
            np.sqrt((_sample_spike - 1 - gamma)**2 - 4*gamma)) / 2 / _sample_spike
    return (-1/mbar) * s2


# ═══════════════════════════════════════════════════════════════════════════════
# Reference biwhitening — copied from ../spcarmt/scrna/rmt/_covariance.py
# ═══════════════════════════════════════════════════════════════════════════════

def biwhitening(X, max_iter=1000, vperiod=5, tol=1e-5):
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
    # Skip alpha scaling — we want the raw biwhitened matrix for testing
    return D1, D2


# ═══════════════════════════════════════════════════════════════════════════════
# Test vector generation
# ═══════════════════════════════════════════════════════════════════════════════

def save(name, data):
    path = os.path.join(OUT, name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


# ─── 1. Marchenko-Pastur PDF ─────────────────────────────────────────────────
print("1. MP PDF")
pdf_cases = []
for q in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    lm = (1 - np.sqrt(q))**2
    lp = (1 + np.sqrt(q))**2
    xs = np.linspace(lm + 1e-6, lp - 1e-6, 10).tolist()
    for x in xs:
        val = marchenko_pastur_density(x, q)
        pdf_cases.append({"q": q, "x": x, "expected": val})
save("mp_pdf.json", pdf_cases)


# ─── 2. Marchenko-Pastur CDF ─────────────────────────────────────────────────
print("2. MP CDF")
cdf_cases = []
for q in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    lm = (1 - np.sqrt(q))**2
    lp = (1 + np.sqrt(q))**2
    xs = np.linspace(lm + 1e-6, lp - 1e-6, 10).tolist()
    for x in xs:
        val = marchenko_pastur_cumulative(x, q)
        cdf_cases.append({"q": q, "x": x, "expected": val})
    cdf_cases.append({"q": q, "x": lm - 0.01, "expected": marchenko_pastur_cumulative(lm - 0.01, q)})
    cdf_cases.append({"q": q, "x": lp + 0.01, "expected": marchenko_pastur_cumulative(lp + 0.01, q)})
save("mp_cdf.json", cdf_cases)


# ─── 3. Marchenko-Pastur Median ──────────────────────────────────────────────
print("3. MP Median")
median_cases = []
for q in [0.05, 0.1, 0.25, 0.5, 0.75]:
    val = marchenko_pastur_median(q)
    median_cases.append({"q": q, "expected": val})
# For q > 1, Python inverts to 1/q internally; test the raw median of MP(1/q)
# and let Rust verify q * median(1/q).
for q in [1.5, 2.0, 5.0, 10.0]:
    val = marchenko_pastur_median(q)
    # Python returns median of MP(1/q), not q*median(1/q).
    # Rust's mp_median(q) returns q * mp_median(1/q) for q > 1.
    # So expected_rust = q * val.
    median_cases.append({"q": q, "expected_python": val, "expected_rust": q * val})
save("mp_median.json", median_cases)


# ─── 4. BBP inverse: invert_predicted_eigval (calculate_alpha) ───────────────
print("4. BBP calculate_alpha")
alpha_cases = []
for q in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    lp = (1 + np.sqrt(q))**2
    for fac in [1.1, 1.5, 2.0, 3.0, 5.0, 10.0]:
        lam = lp * fac
        alpha = invert_predicted_eigval(lam, q)
        alpha_cases.append({"q": q, "lambda": lam, "expected_alpha": float(alpha)})
save("bbp_alpha.json", alpha_cases)


# ─── 5. BBP predicted overlap ────────────────────────────────────────────────
print("5. BBP predicted overlap")
overlap_cases = []
for q in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    lp = (1 + np.sqrt(q))**2
    for fac in [1.1, 1.5, 2.0, 3.0, 5.0, 10.0]:
        lam = lp * fac
        alpha = invert_predicted_eigval(lam, q)
        ov = predicted_overlap(alpha, q)
        overlap_cases.append({
            "q": q,
            "lambda": lam,
            "expected_overlap": float(ov),
        })
save("bbp_overlap.json", overlap_cases)


# ─── 6. Biwhitening on a small seeded matrix ─────────────────────────────────
print("6. Biwhitening")
np.random.seed(42)
n, p = 60, 30
X = np.abs(np.random.randn(n, p)) + 0.1

D1, D2 = biwhitening(X, max_iter=5000, tol=1e-8)
Xw = D1 @ X @ D2

col_var = Xw.var(axis=0)
row_var = Xw.var(axis=1)
col_second_moments = (Xw**2).mean(axis=0)
row_second_moments = (Xw**2).mean(axis=1)
print(f"  col variance: mean={col_var.mean():.6f}, max_dev={np.max(np.abs(col_var - 1)):.2e}")
print(f"  row variance: mean={row_var.mean():.6f}, max_dev={np.max(np.abs(row_var - 1)):.2e}")
print(f"  col 2nd moments (= 1 + mean²): mean={col_second_moments.mean():.4f}")
print(f"  row 2nd moments (= 1 + mean²): mean={row_second_moments.mean():.4f}")

# Covariance eigenvalues (1/(n-1) convention, matching both Rust and Python)
Xwc = Xw - Xw.mean(axis=0)
S = Xwc.T @ Xwc / (n - 1)
eigs = np.sort(np.linalg.eigvalsh(S))[::-1]

bw_data = {
    "n": n,
    "p": p,
    "seed": 42,
    "X": X.tolist(),
    "Xw": Xw.tolist(),
    "col_variances": col_var.tolist(),
    "row_variances": row_var.tolist(),
    "eigenvalues": eigs.tolist(),
}
save("biwhitening.json", bw_data)


# ─── 7. Spiked covariance end-to-end ─────────────────────────────────────────
print("7. Spiked model")
np.random.seed(123)
n, p, k = 200, 50, 2
q = p / n
lp = (1 + np.sqrt(q))**2
signal_strengths = np.array([8.0, 4.0])
h_vecs = np.linalg.qr(np.random.randn(p, k))[0][:, :k]
Sigma = np.eye(p)
for i in range(k):
    Sigma += signal_strengths[i] * np.outer(h_vecs[:, i], h_vecs[:, i])
sqrt_Sigma = np.real(np.linalg.cholesky(Sigma))
X_spiked = np.random.randn(n, p) @ sqrt_Sigma.T

S = X_spiked.T @ X_spiked / (n - 1)
eigs = np.sort(np.linalg.eigvalsh(S))[::-1]
predicted_sample_eigs = [float(predicted_eigval(a, q)) for a in signal_strengths]
predicted_overlaps_vals = [float(predicted_overlap(a, q)) for a in signal_strengths]
recovered_alphas = [float(invert_predicted_eigval(e, q)) for e in eigs[:k]]

spiked_data = {
    "n": n, "p": p, "k": k, "q": q,
    "lambda_plus": lp,
    "signal_strengths": signal_strengths.tolist(),
    "X": X_spiked.tolist(),
    "eigenvalues": eigs.tolist(),
    "top_eigenvalues": eigs[:k].tolist(),
    "predicted_sample_eigenvalues": predicted_sample_eigs,
    "predicted_overlaps": predicted_overlaps_vals,
    "recovered_alphas": recovered_alphas,
    "h_vecs": h_vecs.tolist(),
}
save("spiked_model.json", spiked_data)

print("\nDone. Test vectors written to test_vectors/")
