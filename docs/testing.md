# Testing

## Test suites

### Unit tests (`src/`)

In-source tests that verify mathematical properties without external data.
Run with `cargo test`.

| Module | Tests | What is verified |
|--------|-------|-----------------|
| `rmt` | 6 | λ+ formula; CDF integrates to 1 for q<1 and q>1; point mass at 0 for q>1; `mp_cdf_bulk` normalises to 1; median in support; median is 50th percentile of bulk CDF; overlap in (0,1) |
| `biwhitening` | 1 | Convergence to positive scalings; fixed-point stability (second pass ≈ identity) |
| `spca` | 8 | Gram-Schmidt orthonormality; FISTA recovers top eigenvector of a known matrix; subspace distance; Eq. 8 root on a 3-spike covariance and through `fit_covariance`; subspace iteration on disjoint supports; **FISTA W vs Python at four fixed penalties** (`eq8_penalty.json`, round-off agreement); Eq. 8 root vs Python |
| `verification` | 2 | KS ≈ 0 on quantile-spaced bulk eigenvalues for q<1 and q>1 |
| `markers` | 1 | top-N selection: sorted by |weight|, zeros dropped, truncated |
| `sparse` | 3 | CSR validation, transpose/matvec vs dense, empty row/col filtering |

### Cross-validation tests (`tests/cross_validation.rs`)

Integration tests that compare Rust output against the Python reference
implementation (`../spcarmt/scrna/rmt`).  They load pre-generated test
vectors from `test_vectors/` and assert numerical agreement within documented
tolerances.

Run with `cargo test --test cross_validation`.

| Test | Vector file | Tolerance | Notes |
|------|------------|-----------|-------|
| `mp_pdf_matches_python` | `mp_pdf.json` | 1e-12 relative | Exact closed-form formula |
| `mp_cdf_matches_python` | `mp_cdf.json` | 5e-4 (3e-2 at q=1) | Simpson 2000 panels vs `scipy.integrate.quad`; looser at q=1 due to √x singularity at λ-=0 |
| `mp_median_matches_python` | `mp_median.json` | 1% relative | Python uses `brentq` with `xtol=0.003`; Rust uses binary search (60 iterations) |
| `bbp_alpha_matches_python` | `bbp_alpha.json` | round-trip | `calculate_alpha` is private; verified indirectly via `predicted_overlap` |
| `bbp_overlap_matches_python` | `bbp_overlap.json` | 1e-10 | Exact closed-form formula |
| `biwhitening_produces_unit_variance` | `biwhitening.json` | var ∈ [0.95, 1.05] | Rust and Python converge to the same fixed point; different stopping criteria may yield slightly different c,d |
| `biwhitening_matrix_matches_python` | `biwhitening.json` | 1e-5 relative, entrywise | X_w itself (`Xw` field). Different Sinkhorn start (Python (p, n), Rust (1, 1)), same stopping rule, same X_w: the fixed point is unique up to the (c·t, d/t) gauge. Observed 5e-7 at tol 1e-8 |
| `sigma_sq_matches_python_alpha` | `biwhitening.json` | 1% relative | `fit().sigma_sq` vs the reference's `alpha` (`_get_scaling_marchenko_pastur`). Conventions differ (see Known limitations); observed 0.1% |
| `biwhitening_eigenvalues_near_python` | `biwhitening.json` | 10% top-5; 5% mean | Bulk spectrum shape should match; residual differences from stopping criteria |
| `spiked_model_eigenvalues_match` | `spiked_model.json` | 1e-8 | Same data, same LAPACK; should be bit-identical up to floating-point order |
| `spiked_model_bbp_predictions` | `spiked_model.json` | 1e-10 | BBP overlap from population spikes matches Python formula |
| `spiked_model_full_pipeline` | `spiked_model.json` | KS < 0.20; sparsity > 0.10 | End-to-end pipeline smoke test |
| `fista_matches_python_reference_at_fixed_penalties` (in `src/spca.rs`) | `eq8_penalty.json` | 1e-12 entrywise; identical support | `fista_sparse_pca` from the reference's exact V and γ at λ ∈ {0.05, 0.3, 0.6γ*, 2.0}, ‖ΔW‖ < 1e-10. Observed 4e-16, with and without `fista_reuse_products` |
| `eq8_root_matches_python_reference` (in `src/spca.rs`) | `eq8_penalty.json` | 15% on λ*; 0.05 on distance | Root of the Eq. 8 bisection vs `scipy.optimize.root_scalar`; the distance is a staircase in λ so the root is only located to the step width |

### Precision tests (`tests/sparse_precision.rs`)

Dense vs CSR vs streamed vs subsample paths, stage by stage, on a seeded
Poisson-count fixture (`counts()`), plus the eigendecomposition's residuals.
Also that `Full` mode's k, component eigenvalues and FISTA start are the
exact spectrum's (and `k_max` caps without hiding the count), that the
subspace-iteration path returns ordered Ritz values, and the two pipeline
branches that have no reference: the per-gene standardisation fallback when
Sinkhorn is cut off (`bw_max_iter: 1`; dense and CSR must agree) and
`EigensolverMode::Fast` (σ² equals the mean of the
`Full` spectrum exactly; no spectrum, no KS). On that fixture the trace
estimate is 17% above the bulk median: the strong-signal, small-p bias the
mode's documentation warns about.

Run with `cargo test --test sparse_precision` (~45 s in debug).

---

## Coverage

```bash
cargo llvm-cov --summary-only      # per-file table
cargo llvm-cov report --text       # per-line, reusing the last run's profiles
```

Takes ~8 min (instrumented debug build). As of 2026-10-13: 87% lines overall;
`main.rs` (binary demo) is 0%, and the remaining misses in `spca.rs` are
`verbose` logging, the Eq. 8 bracket-widening branches, and the `k_max`
saturation warning.

---

## Test vectors

Test vectors are JSON files in `test_vectors/`, generated by
`scripts/generate_test_vectors.py`.  They are committed to the repository so
that `cargo test` works without a Python environment.  Regenerate them when
the Python reference implementation changes:

```bash
uv run scripts/generate_test_vectors.py
```

The script is self-contained (no import of the full `scrna` package) — it
copies the handful of reference functions verbatim from
`../spcarmt/scrna/rmt/_covariance.py` and `_toy.py`, used under the MIT
License:

> MIT License — Copyright (c) 2025 Victor Chardès

Functions copied: `marchenko_pastur_density`, `marchenko_pastur_cumulative`,
`marchenko_pastur_median`, `biwhitening`, `predicted_overlap`,
`predicted_eigval`, `invert_predicted_eigval`.  No mathematical changes were
made; only imports were removed to make them self-contained.

### `mp_pdf.json`

70 (x, q) pairs sampling 10 points inside the support [λ-, λ+] for each of
q ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0}.

```json
{ "q": 0.5, "x": 0.085..., "expected": 0.412... }
```

### `mp_cdf.json`

72 entries: 10 interior points per q plus one point below λ- and one above λ+,
for q ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0}.  Boundary cases test the point-mass
behaviour for q > 1 (CDF below λ- = 1 − 1/q, CDF above λ+ = 1.0).

```json
{ "q": 2.0, "x": 0.17..., "expected": 0.063... }
```

### `mp_median.json`

9 entries spanning q ∈ {0.05, 0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 5.0, 10.0}.
For q ≤ 1: `expected` is the Python value directly.
For q > 1: two fields are stored to document the Python/Rust difference:
- `expected_python`: Python's `marchenko_pastur_median(q)` returns `median(MP(1/q))` (always q ≤ 1).
- `expected_rust`: Rust's `mp_median(q)` returns `q × median(MP(1/q))` (the median of the non-zero eigenvalues of X^T X / n).

```json
{ "q": 2.0, "expected_python": 0.830..., "expected_rust": 1.660... }
```

### `bbp_alpha.json`

36 entries: 6 q values × 6 λ multipliers (1.1×, 1.5×, 2×, 3×, 5×, 10× λ+).
`expected_alpha` is the population spike α recovered by Python's
`invert_predicted_eigval(λ, q)`.  Verified in Rust via round-trip through
`predicted_overlap`.

```json
{ "q": 0.5, "lambda": 10.65..., "expected_alpha": 4.35... }
```

### `bbp_overlap.json`

Same 36 (q, λ) pairs as `bbp_alpha.json`.  `expected_overlap` is Python's
`predicted_overlap(invert_predicted_eigval(λ, q), q)`.

```json
{ "q": 0.5, "lambda": 10.65..., "expected_overlap": 0.847... }
```

### `biwhitening.json`

A single 60×30 non-negative matrix generated with `np.random.seed(42)`.
Python's `biwhitening()` is run to convergence (`tol=1e-8`) to produce the
reference biwhitened matrix and covariance eigenvalues. The reference's
`alpha` (its σ²) is stored separately rather than folded into `Xw`.

| Field | Shape | Description |
|-------|-------|-------------|
| `n`, `p`, `seed` | scalar | Matrix dimensions and RNG seed |
| `X` | 60×30 | Input matrix (non-negative, float) |
| `Xw` | 60×30 | Python-biwhitened matrix |
| `col_variances` | 30 | Per-gene variances of `Xw` (should be ≈1) |
| `row_variances` | 60 | Per-cell variances of `Xw` (should be ≈1) |
| `eigenvalues` | 30 | Covariance eigenvalues of `Xw` (1/(n-1) convention, descending) |
| `alpha` | scalar | `_get_scaling_marchenko_pastur(Xw)`: the reference's σ² |

### `eq8_penalty.json`

Generated by `scripts/generate_eq8_vector.py` (run with
`uv run --with numpy --with scipy python scripts/generate_eq8_vector.py`
from `scripts/`), which ports `_fista_spca`, `_gs_orthogonalize` and the
root-find of `_adjust_penalty` from `../spcarmt/scrna/prc/_spca.py` and runs
them on `spiked_model.json`'s S = XᵀX/(n−1).

| Field | Shape | Description |
|-------|-------|-------------|
| `target` | scalar | Distance target handed to the root-find (0.2) |
| `gamma_star` | scalar | Reference root γ*; `distance_at_star`, `distance_at_06`, `distance_at_4` the achieved distances |
| `V` | 50×2 | Exact top-k eigenvectors of S (`np.linalg.eigh`), FISTA's start |
| `lmax` | scalar | λ_max(S); γ = 0.5/(2·lmax) |
| `fista[]` | list | `{penalty, tol, iters, W (50×2), nnz}`: FISTA from `V` at a fixed penalty, ‖ΔW‖_F < 1e-10 |

### `spiked_model.json`

A 200×50 spiked Gaussian model with k=2 planted signal components at
population spikes α ∈ {8.0, 4.0} (both well above the BBP threshold for
q=0.25, λ+=2.25).  Used for end-to-end pipeline validation.

| Field | Shape | Description |
|-------|-------|-------------|
| `n`, `p`, `k`, `q` | scalar | Dimensions and aspect ratio |
| `lambda_plus` | scalar | MP bulk edge (1+√q)² = 2.25 |
| `signal_strengths` | 2 | Population spikes [8.0, 4.0] |
| `X` | 200×50 | Data matrix (Gaussian spiked model) |
| `eigenvalues` | 50 | All covariance eigenvalues (1/(n-1), descending) |
| `top_eigenvalues` | 2 | Top-k eigenvalues (signal outliers) |
| `predicted_sample_eigenvalues` | 2 | BBP forward map: α + qα/(α-1) |
| `predicted_overlaps` | 2 | BBP overlap: ((α-1)²-q)/((α-1)(α-1+q)) |
| `recovered_alphas` | 2 | Inverse BBP from observed outlier eigenvalues |
| `h_vecs` | 50×2 | True signal directions (orthonormal) |

---

## Known limitations

- **MP CDF at q=1**: the density ρ_MP(x) ~ 1/√x is singular at λ-=0.
  Simpson's rule with 2000 panels underestimates the integral by ~1–3% near
  the singularity.  The cross-validation test uses a looser tolerance (3e-2)
  for q=1 cases.  Could be improved with Gauss-Jacobi quadrature.

- **Biwhitening convergence**: the Rust implementation uses a 95th-percentile
  relative change in c as its stopping criterion; Python uses
  max(|Δ(var−1)|).  Both converge to the same fixed point but may stop at
  different iterates.  The cross-validation test therefore checks fixed-point
  conditions (unit variance) rather than exact agreement of c and d.

- **σ² convention**: the reference's `alpha` is the median of *all*
  eigenvalues of the *uncentred* X_wᵀX_w/n over the MP median; Rust uses the
  bulk median (0.001 < λ ≤ λ+) of the centred X_wcᵀX_wc/(n−1). Both put the
  bulk median of the rescaled S on the MP median, so they are self-consistent;
  they differ by the n/(n−1) factor and by whether the mean direction and the
  signal outliers sit in the median. 0.1% on the test vector.

- **FISTA**: the iteration is the reference's step for step (gradient,
  soft-threshold at γλ, Gram-Schmidt, momentum (1/20 + √(1+4t²))/2) and
  agrees to round-off from the same start. What differs is around it:
  - Start: in `Full` mode Rust takes V, λ_max and k from Stage 4's exact
    EVD, as the reference takes them from `eigh`. `Fast` and `Subsample`
    have no full EVD and use subspace iteration (100 steps + Rayleigh–Ritz)
    on `k_max` vectors; components near λ+ converge slowly there, so the
    tail count and values are approximate (~1% on a 600×300 probe).
  - Stopping: Rust ORs an objective-change test (`tol_obj`, default 1e-6)
    with the iterate test; the reference has only ‖ΔW‖_F < tol. Defaults
    differ too (`tolerance` 1e-6 / `max_iterations` 1000 vs 1e-4 / 10000).
  - Component count: Rust counts eigenvalues > λ+ with a floor of 1 and
    fits at most `k_max` of them (the exact count is in
    `n_above_threshold`); the reference counts eigenvalues > λ+ + 0.01,
    uncapped. The +0.01 is an absolute margin with no scaling argument, so
    it was not copied.
  - A column that soft-thresholds to zero is left as zero; the reference
    divides by zero (NaN).
