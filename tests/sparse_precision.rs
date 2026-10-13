//! Numerical-precision tests for the sparse (CSR) pipeline and the decomposition.
//!
//! 1. CSR path == dense path, stage by stage: biwhitening factors, covariance,
//!    full eigenspectrum, σ², k, Rayleigh quotients, sparse components.
//! 2. The eigendecomposition itself is accurate: residuals ‖Su − λu‖,
//!    orthonormality, trace identity, positive semi-definiteness.
//!
//! Tolerances are relative to the problem scale (max |S| or λ_max) and sit a
//! few orders above f64 round-off: the two paths sum in different orders, so
//! bit-equality is not expected, but anything beyond ~1e-10 is a real bug.

use faer::{Col, Mat};
use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::sparse::CsrMatrix;
use rmt_spca::spca::{EigensolverMode, FistaConfig, SparsePCA, SparsePCAResult};
use serde::Deserialize;

// ─── fixtures ────────────────────────────────────────────────────────────────

/// xorshift64*: deterministic, dependency-free uniform in (0, 1).
struct Rng(u64);
impl Rng {
    fn unif(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn poisson(&mut self, lam: f64) -> f64 {
        let (l, mut k, mut prod) = ((-lam).exp(), 0.0, 1.0);
        loop {
            prod *= self.unif();
            if prod <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

/// Sparse Poisson counts with 3 cell groups, 2 up-regulated gene programs,
/// variable library size, plus one empty cell and one empty gene (Stage 0).
fn counts(n: usize, p: usize, seed: u64) -> Mat<f64> {
    let mut rng = Rng(seed);
    let base: Vec<f64> = (0..p).map(|_| 0.05 + 1.2 * rng.unif().powi(2)).collect();
    let size: Vec<f64> = (0..n).map(|_| 0.5 + 1.5 * rng.unif()).collect();
    let mut x = Mat::zeros(n, p);
    // last cell and last gene stay empty
    for (i, &sz) in size.iter().enumerate().take(n - 1) {
        let g = i % 3;
        for (j, &b) in base.iter().enumerate().take(p - 1) {
            let up = (g == 0 && j < p / 5) || (g == 1 && (p / 5..2 * p / 5).contains(&j));
            let lam = b * sz * if up { 5.0 } else { 1.0 };
            x.write(i, j, rng.poisson(lam));
        }
    }
    x
}

fn cfg() -> FistaConfig {
    FistaConfig {
        lambda_frac: Some(0.3),
        top_eigvec_count: 10,
        ..FistaConfig::default()
    }
}

fn max_abs(m: &Mat<f64>) -> f64 {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| m.read(i, j).abs())
        .fold(0.0, f64::max)
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

fn col_vec(c: &Col<f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c.read(i)).collect()
}

/// Reference covariance, the straightforward dense way: centre X_w, then XᵀX/(n−1).
fn dense_cov(x: &Mat<f64>, c: &Col<f64>, d: &Col<f64>) -> Mat<f64> {
    let (n, p) = (x.nrows(), x.ncols());
    let xw = Biwhitener::apply(x, c, d);
    let mu: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| xw.read(i, j)).sum::<f64>() / n as f64)
        .collect();
    let xc = Mat::from_fn(n, p, |i, j| xw.read(i, j) - mu[j]);
    let s: Mat<f64> = xc.as_ref().transpose() * xc.as_ref();
    Mat::from_fn(p, p, |i, j| s.read(i, j) / (n - 1) as f64)
}

/// Dense input with empty rows/cols removed (what Stage 0 does in both paths).
fn filtered(x: &Mat<f64>) -> Mat<f64> {
    let (_, rows, cols) = CsrMatrix::from_dense(x).drop_empty();
    Mat::from_fn(rows.len(), cols.len(), |i, j| x.read(rows[i], cols[j]))
}

fn assert_results_match(a: &SparsePCAResult, b: &SparsePCAResult, tol: f64) {
    assert_eq!(a.q, b.q, "q");
    assert_eq!(a.lambda_plus, b.lambda_plus, "λ+");
    assert_eq!(a.sk_iters, b.sk_iters, "Sinkhorn iterations");
    assert!(
        rel_diff(&a.bw_c, &b.bw_c) < tol,
        "bw_c rel diff {:.2e}",
        rel_diff(&a.bw_c, &b.bw_c)
    );
    assert!(
        rel_diff(&a.bw_d, &b.bw_d) < tol,
        "bw_d rel diff {:.2e}",
        rel_diff(&a.bw_d, &b.bw_d)
    );
    assert!(
        (a.sigma_sq - b.sigma_sq).abs() < tol * a.sigma_sq,
        "σ² {} vs {}",
        a.sigma_sq,
        b.sigma_sq
    );
    let ev = rel_diff(&a.s_eigenvalues, &b.s_eigenvalues);
    assert!(ev < tol, "eigenspectrum rel diff {ev:.2e}");
    assert_eq!(a.eigenvalues.len(), b.eigenvalues.len(), "signal count k");
    let rq = rel_diff(&a.eigenvalues, &b.eigenvalues);
    assert!(rq < tol, "Rayleigh quotients rel diff {rq:.2e}");
    let ks = (a.ks_distance.unwrap() - b.ks_distance.unwrap()).abs();
    assert!(ks < 1e-8, "KS diff {ks:.2e}");

    // Components: same support, same values up to column sign.
    let (wa, wb) = (&a.components, &b.components);
    assert_eq!((wa.nrows(), wa.ncols()), (wb.nrows(), wb.ncols()));
    for j in 0..wa.ncols() {
        let dot: f64 = (0..wa.nrows()).map(|i| wa.read(i, j) * wb.read(i, j)).sum();
        let sgn = dot.signum();
        for i in 0..wa.nrows() {
            let (x, y) = (wa.read(i, j), sgn * wb.read(i, j));
            assert_eq!(
                x == 0.0,
                y == 0.0,
                "support differs at ({i},{j}): {x} vs {y}"
            );
            assert!((x - y).abs() < 1e-8, "component ({i},{j}): {x} vs {y}");
        }
    }
}

// ─── 1. CSR path == dense path ───────────────────────────────────────────────

#[test]
fn csr_biwhitening_matches_dense() {
    let x = filtered(&counts(500, 120, 7));
    let bw = Biwhitener::default();
    let (c1, d1, it1, ok1, _) = bw.compute(&x);
    let (c2, d2, it2, ok2, _) = bw.compute_csr(&CsrMatrix::from_dense(&x));
    assert!(ok1 && ok2, "both must converge");
    assert_eq!(it1, it2, "iteration count");
    let (dc, dd) = (
        rel_diff(&col_vec(&c1), &col_vec(&c2)),
        rel_diff(&col_vec(&d1), &col_vec(&d2)),
    );
    assert!(
        dc < 1e-12 && dd < 1e-12,
        "c rel diff {dc:.2e}, d rel diff {dd:.2e}"
    );
}

#[test]
fn csr_covariance_matches_dense_and_is_symmetric() {
    let x = filtered(&counts(500, 120, 11));
    let (c, d, ..) = Biwhitener::default().compute(&x);
    let a = CsrMatrix::from_dense(&x);
    let s = a.biwhitened_covariance(&a.transpose(), &c, &d);
    let r = dense_cov(&x, &c, &d);
    let diff = Mat::from_fn(r.nrows(), r.ncols(), |i, j| s.read(i, j) - r.read(i, j));
    let err = max_abs(&diff) / max_abs(&r);
    assert!(err < 1e-12, "covariance rel err {err:.2e}");
    for i in 0..s.nrows() {
        for j in 0..i {
            assert_eq!(
                s.read(i, j),
                s.read(j, i),
                "S not exactly symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn csr_fit_matches_dense_fit_on_counts() {
    let x = counts(600, 150, 3);
    let dense = SparsePCA::new(cfg()).fit(&x);
    let sparse = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    assert!(
        dense.eigenvalues.len() >= 2,
        "fixture should carry ≥2 signal components"
    );
    assert_results_match(&dense, &sparse, 1e-10);
}

#[derive(Deserialize)]
struct BiwhiteningData {
    n: usize,
    p: usize,
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
}

#[test]
fn csr_fit_matches_dense_fit_on_python_vector() {
    let txt = std::fs::read_to_string("test_vectors/biwhitening.json").unwrap();
    let data: BiwhiteningData = serde_json::from_str(&txt).unwrap();
    let x = Mat::from_fn(data.n, data.p, |i, j| data.x[i][j]);
    let dense = SparsePCA::new(cfg()).fit(&x);
    let sparse = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    assert_results_match(&dense, &sparse, 1e-10);
}

#[test]
fn csr_fit_with_precomputed_factors_matches_dense() {
    let x = filtered(&counts(400, 100, 5));
    let (c, d, ..) = Biwhitener::default().compute(&x);
    let with = |c: &Col<f64>, d: &Col<f64>| FistaConfig {
        precomputed_factors: Some((c.clone(), d.clone())),
        ..cfg()
    };
    let dense = SparsePCA::new(with(&c, &d)).fit(&x);
    let sparse = SparsePCA::new(with(&c, &d)).fit_csr(&CsrMatrix::from_dense(&x));
    assert_results_match(&dense, &sparse, 1e-10);
}

// ─── 2. accuracy of the eigendecomposition ───────────────────────────────────

/// When Sinkhorn is cut off before convergence, both paths fall back to
/// per-gene standardisation and must agree with each other. The dense path
/// reports the unconverged Sinkhorn factors and the CSR path the applied ones,
/// so `bw_c`/`bw_d` are not compared; everything downstream of X_w is.
#[test]
fn fallback_to_standardisation_matches_between_dense_and_csr() {
    let x = filtered(&counts(300, 80, 11));
    let cfg = || FistaConfig {
        bw_max_iter: 1,
        ..cfg()
    };
    let dense = SparsePCA::new(cfg()).fit(&x);
    let csr = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    assert!(!dense.sk_converged && !csr.sk_converged);
    assert!(
        dense.sk_residual > 1e-2,
        "residual {} did not trigger the fallback",
        dense.sk_residual
    );
    assert!(rel_diff(&dense.s_eigenvalues, &csr.s_eigenvalues) < 1e-10);
    assert!((dense.sigma_sq - csr.sigma_sq).abs() < 1e-10);
    assert_eq!(dense.eigenvalues.len(), csr.eigenvalues.len());
    let diff = Mat::from_fn(
        dense.components.nrows(),
        dense.components.ncols(),
        |i, j| dense.components.read(i, j) - csr.components.read(i, j),
    );
    assert!(
        max_abs(&diff) < 1e-8,
        "components differ by {:.2e}",
        max_abs(&diff)
    );
    // the CSR path reports the factors it applied: c = 1, d = 1/sd
    assert!(csr.bw_c.iter().all(|&c| c == 1.0));
}

/// `Full` mode takes k, the component eigenvalues and FISTA's start from the
/// exact EVD (as the reference's `eigh` does), so `eigenvalues` must be the
/// top of `s_eigenvalues` to round-off and `n_above_threshold` the exact
/// count above λ+; `k_max` only caps the fit and leaves the count visible.
#[test]
fn full_mode_k_and_eigenvalues_come_from_the_exact_spectrum() {
    let x = filtered(&counts(300, 80, 11));
    let r = SparsePCA::new(FistaConfig { k_max: 50, ..cfg() }).fit(&x);
    let mut desc = r.s_eigenvalues.clone();
    desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let above = desc.iter().filter(|&&e| e > r.lambda_plus).count();
    assert!(
        above >= 2,
        "fixture has {above} eigenvalues above λ+; too weak"
    );
    assert_eq!(r.n_above_threshold, above);
    assert_eq!(r.eigenvalues.len(), above);
    for (j, &e) in r.eigenvalues.iter().enumerate() {
        assert!(
            (e - desc[j]).abs() < 1e-12,
            "eigenvalue {j}: {e} vs exact {}",
            desc[j]
        );
    }
    // FISTA's start spans the exact eigenvectors: with λ = 0 FISTA is a fixed
    // point there, so the fitted components keep that span.
    let v = Mat::from_fn(x.ncols(), above, |i, j| r.top_eigenvectors[j][i]);
    let r0 = SparsePCA::new(FistaConfig {
        lambda: 0.0,
        lambda_frac: None,
        k_max: 50,
        ..cfg()
    })
    .fit(&x);
    assert!(rmt_spca::spca::subspace_distance(&r0.components, &v) < 1e-8);
    // the cap is honest: fewer components fitted, same count reported
    let capped = SparsePCA::new(FistaConfig { k_max: 1, ..cfg() }).fit(&x);
    assert_eq!(capped.eigenvalues.len(), 1);
    assert_eq!(capped.n_above_threshold, above);
}

/// The subspace-iteration path (no full EVD) returns Ritz values in
/// descending order after Rayleigh–Ritz, and its leading values agree with
/// the exact spectrum; only the tail near λ+ is approximate.
#[test]
fn subspace_path_ritz_values_are_ordered_and_lead_exactly() {
    let x = filtered(&counts(300, 80, 11));
    let full = SparsePCA::new(FistaConfig { k_max: 50, ..cfg() }).fit(&x);
    let sub = SparsePCA::new(FistaConfig {
        eigensolver: EigensolverMode::Subsample { n_cells: 300 },
        k_max: 50,
        ..cfg()
    })
    .fit(&x);
    assert!(
        sub.eigenvalues.windows(2).all(|w| w[0] >= w[1]),
        "Ritz values unordered"
    );
    assert!(sub.n_above_threshold <= 50);
    for j in 0..2 {
        let rel = (sub.eigenvalues[j] - full.eigenvalues[j]).abs() / full.eigenvalues[j];
        assert!(rel < 1e-8, "leading Ritz value {j} off by {rel:.1e}");
    }
}

/// `EigensolverMode::Fast`: σ² = Tr(S)/p is exactly the mean of the spectrum
/// `Full` computes, so the two are tied by an identity; the spectrum-dependent
/// diagnostics are absent. On this fixture (two strong programs in 80 genes)
/// the mean sits 17% above the bulk median: the bias the mode's docs warn of.
#[test]
fn fast_mode_sigma_is_spectrum_mean_and_skips_spectrum() {
    let x = filtered(&counts(300, 80, 11));
    let full = SparsePCA::new(cfg()).fit(&x);
    let fast = SparsePCA::new(FistaConfig {
        eigensolver: EigensolverMode::Fast,
        ..cfg()
    })
    .fit(&x);
    let mean = full.s_eigenvalues_unrescaled.iter().sum::<f64>()
        / full.s_eigenvalues_unrescaled.len() as f64;
    eprintln!(
        "σ² full={:.4} fast={:.4} (spectrum mean {mean:.4})",
        full.sigma_sq, fast.sigma_sq
    );
    assert!(
        (fast.sigma_sq - mean).abs() < 1e-10,
        "Fast σ² {} != Tr(S)/p {mean}",
        fast.sigma_sq
    );
    assert!(fast.ks_distance.is_none());
    assert!(fast.s_eigenvalues.is_empty() && fast.top_eigenvectors.is_empty());
}

#[test]
fn eigendecomposition_is_accurate() {
    let x = filtered(&counts(600, 150, 13));
    let r = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    let c = Col::from_fn(r.bw_c.len(), |i| r.bw_c[i]);
    let d = Col::from_fn(r.bw_d.len(), |j| r.bw_d[j]);
    // Independent dense S (unrescaled), the matrix the EVD ran on.
    let s = dense_cov(&x, &c, &d);
    let p = s.nrows();
    let lam = &r.s_eigenvalues_unrescaled;
    let lmax = lam[p - 1];

    // ascending, PSD up to round-off
    assert!(
        lam.windows(2).all(|w| w[0] <= w[1]),
        "eigenvalues not ascending"
    );
    assert!(lam[0] > -1e-12 * lmax, "S not PSD: λ_min = {:.3e}", lam[0]);

    // trace identity Σλ = tr S
    let tr: f64 = (0..p).map(|i| s.read(i, i)).sum();
    let sum: f64 = lam.iter().sum();
    assert!((sum - tr).abs() < 1e-12 * tr, "Σλ = {sum}, tr S = {tr}");

    // top eigenpairs: residual and orthonormality
    let u = Mat::from_fn(p, r.top_eigenvectors.len(), |i, k| r.top_eigenvectors[k][i]);
    let su: Mat<f64> = s.as_ref() * u.as_ref();
    for k in 0..u.ncols() {
        let l = lam[p - 1 - k];
        let res: f64 = (0..p)
            .map(|i| (su.read(i, k) - l * u.read(i, k)).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(res < 1e-11 * lmax, "‖Su−λu‖ = {res:.2e} for pair {k}");
    }
    let utu: Mat<f64> = u.as_ref().transpose() * u.as_ref();
    for a in 0..u.ncols() {
        for b in 0..u.ncols() {
            let want = if a == b { 1.0 } else { 0.0 };
            assert!(
                (utu.read(a, b) - want).abs() < 1e-12,
                "UᵀU[{a},{b}] = {}",
                utu.read(a, b)
            );
        }
    }

    // σ²-rescaled spectrum is exactly the unrescaled one / σ²
    for (a, b) in r.s_eigenvalues.iter().zip(lam) {
        assert!((a * r.sigma_sq - b).abs() < 1e-14 * lmax);
    }
}

// ─── 3. EigensolverMode::Subsample ───────────────────────────────────────────

fn sub_cfg(n_cells: usize) -> FistaConfig {
    FistaConfig {
        eigensolver: EigensolverMode::Subsample { n_cells },
        ..cfg()
    }
}

/// Same σ², KS, k, Rayleigh quotients and components; spectra differ in length
/// (p vs m) so they are compared on their non-zero part only.
fn assert_same_decomposition(full: &SparsePCAResult, sub: &SparsePCAResult, tol: f64) {
    assert!(
        (full.sigma_sq - sub.sigma_sq).abs() < tol * full.sigma_sq,
        "σ² full {} vs subsample {}",
        full.sigma_sq,
        sub.sigma_sq
    );
    let (kf, ks) = (full.ks_distance.unwrap(), sub.ks_distance.unwrap());
    assert!((kf - ks).abs() < 1e-8, "KS full {kf} vs subsample {ks}");
    let nz = |v: &[f64]| -> Vec<f64> {
        let top = v.iter().cloned().fold(0.0, f64::max);
        v.iter().cloned().filter(|&e| e > 1e-9 * top).collect()
    };
    let ev = rel_diff(&nz(&full.s_eigenvalues), &nz(&sub.s_eigenvalues));
    assert!(ev < tol, "non-zero spectrum rel diff {ev:.2e}");
    assert_eq!(full.eigenvalues.len(), sub.eigenvalues.len(), "k");
    assert!(
        rel_diff(&full.eigenvalues, &sub.eigenvalues) < tol,
        "Rayleigh quotients"
    );
    // Full starts FISTA from the exact eigenvectors, Subsample from subspace
    // iteration: the same vectors up to a sign per column, which FISTA keeps.
    let (p, k) = (full.components.nrows(), full.components.ncols());
    let dw = (0..k)
        .map(|j| {
            let dot: f64 = (0..p)
                .map(|i| full.components.read(i, j) * sub.components.read(i, j))
                .sum();
            let sgn = if dot < 0.0 { -1.0 } else { 1.0 };
            (0..p)
                .map(|i| (full.components.read(i, j) - sgn * sub.components.read(i, j)).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    assert!(dw < 1e-8, "components max diff (sign-aligned) {dw:.2e}");
}

#[test]
fn subsample_with_all_cells_reproduces_full_q_lt_1() {
    let x = counts(600, 150, 3);
    let full = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    let sub = SparsePCA::new(sub_cfg(usize::MAX)).fit_csr(&CsrMatrix::from_dense(&x));
    assert!(sub.top_eigenvectors.is_empty());
    assert_same_decomposition(&full, &sub, 1e-10);
}

#[test]
fn subsample_with_all_cells_reproduces_full_q_gt_1() {
    let x = counts(200, 500, 21);
    let full = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    let sub = SparsePCA::new(sub_cfg(usize::MAX)).fit_csr(&CsrMatrix::from_dense(&x));
    assert!(full.q > 1.0);
    assert_same_decomposition(&full, &sub, 1e-10);
}

#[test]
fn subsample_dense_and_csr_paths_agree() {
    let x = counts(600, 150, 9);
    let dense = SparsePCA::new(sub_cfg(200)).fit(&x);
    let sparse = SparsePCA::new(sub_cfg(200)).fit_csr(&CsrMatrix::from_dense(&x));
    assert_results_match(&dense, &sparse, 1e-10);
}

#[test]
fn partial_subsample_estimates_sigma_and_ks() {
    // Statistical part: a third of the cells must give σ² within 3% and KS
    // within 0.05 of the full spectrum's.
    //
    // Exact part: S is the same matrix in both runs, so the unscaled Rayleigh
    // quotients agree and k_sub must be exactly the full quotients thresholded
    // at λ+·σ²_sub. k itself may differ: a ~1% σ² error rescales every
    // eigenvalue by ~1% and moves components lying that close to λ+ (this
    // fixture has several) — that is the estimator's real sensitivity.
    let x = counts(1500, 200, 17);
    let full =
        SparsePCA::new(FistaConfig { k_max: 40, ..cfg() }).fit_csr(&CsrMatrix::from_dense(&x));
    let sub = SparsePCA::new(FistaConfig {
        k_max: 40,
        ..sub_cfg(500)
    })
    .fit_csr(&CsrMatrix::from_dense(&x));
    let rel = (sub.sigma_sq - full.sigma_sq).abs() / full.sigma_sq;
    let (kf, ks) = (full.ks_distance.unwrap(), sub.ks_distance.unwrap());
    eprintln!(
        "σ² full {:.4} sub {:.4} (rel {rel:.2e}); KS full {kf:.4} sub {ks:.4}; k {} vs {}",
        full.sigma_sq,
        sub.sigma_sq,
        full.eigenvalues.len(),
        sub.eigenvalues.len()
    );
    assert!(rel < 0.03, "σ² rel err {rel:.3}");
    assert!(
        (kf - ks).abs() < 0.05,
        "KS full {kf:.4} vs subsample {ks:.4}"
    );

    let unscaled = |r: &SparsePCAResult| -> Vec<f64> {
        r.eigenvalues.iter().map(|e| e * r.sigma_sq).collect()
    };
    let common = sub.eigenvalues.len().min(full.eigenvalues.len());
    assert!(
        rel_diff(&unscaled(&full)[..common], &unscaled(&sub)[..common]) < 1e-10,
        "unscaled quotients"
    );
    // The full run lists only its k quotients, so the identity below is checkable
    // when σ²_sub ≥ σ²_full (k_sub ≤ k_full) — true for this fixed fixture.
    assert!(sub.sigma_sq >= full.sigma_sq, "fixture assumption");
    let k_expected = unscaled(&full)
        .iter()
        .filter(|&&e| e > full.lambda_plus * sub.sigma_sq)
        .count()
        .max(1);
    assert_eq!(
        sub.eigenvalues.len(),
        k_expected,
        "k_sub = full spectrum thresholded at λ+·σ²_sub"
    );
}

// ─── 4. FISTA product reuse ──────────────────────────────────────────────────

#[test]
fn fista_reuse_matches_default() {
    let x = counts(600, 150, 3);
    let a = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x));
    let b = SparsePCA::new(FistaConfig {
        fista_reuse_products: true,
        ..cfg()
    })
    .fit_csr(&CsrMatrix::from_dense(&x));
    assert_results_match(&a, &b, 1e-10);
}

// ─── 5. streaming: closed-form cell factors, block accumulation, fit_covariance ─

use rmt_spca::streaming::{
    cell_factors, gene_nnz, reset_rare_genes, subsample_spectrum, CovAccumulator, GeneMoments,
};

/// Row blocks [a, b) of a CSR matrix, rebuilt via the dense copy (test-only).
fn row_block(x: &Mat<f64>, a: usize, b: usize) -> CsrMatrix {
    CsrMatrix::from_dense(&Mat::from_fn(b - a, x.ncols(), |i, j| x.read(a + i, j)))
}

/// R refinement passes of d on all cells (closed-form c per block, then GeneMoments).
fn refine(x: &Mat<f64>, mut d: Col<f64>, bs: usize, passes: usize) -> Col<f64> {
    for _ in 0..passes {
        let mut gm = GeneMoments::new(x.ncols());
        for a in (0..x.nrows()).step_by(bs) {
            let blk = row_block(x, a, (a + bs).min(x.nrows()));
            gm.add(&blk, &cell_factors(&blk, &d));
        }
        d = gm.finish();
    }
    d
}

/// Stream X in blocks of `bs` rows: c from `cell_factors`, S from the accumulator.
fn stream(x: &Mat<f64>, d: &Col<f64>, bs: usize) -> (Mat<f64>, Vec<f64>, Col<f64>) {
    let mut acc = CovAccumulator::new(x.ncols());
    let mut c_all = Vec::new();
    for a in (0..x.nrows()).step_by(bs) {
        let blk = row_block(x, a, (a + bs).min(x.nrows()));
        let c = cell_factors(&blk, d);
        c_all.extend(col_vec(&c));
        acc.add(&blk, &c, d);
    }
    let (s, mu) = acc.finish();
    (s, mu, Col::from_fn(c_all.len(), |i| c_all[i]))
}

/// Row and column variances of X_w = diag(c) X diag(d) (population, as Sinkhorn targets).
fn bw_variances(x: &Mat<f64>, c: &Col<f64>, d: &Col<f64>) -> (Vec<f64>, Vec<f64>) {
    let xw = Biwhitener::apply(x, c, d);
    let (n, p) = (x.nrows(), x.ncols());
    let var = |v: &mut dyn Iterator<Item = f64>, m: usize| {
        let v: Vec<f64> = v.collect();
        let mean = v.iter().sum::<f64>() / m as f64;
        v.iter().map(|a| a * a).sum::<f64>() / m as f64 - mean * mean
    };
    let rows = (0..n)
        .map(|i| var(&mut (0..p).map(|j| xw.read(i, j)), p))
        .collect();
    let cols = (0..p)
        .map(|j| var(&mut (0..n).map(|i| xw.read(i, j)), n))
        .collect();
    (rows, cols)
}

#[test]
fn cell_factors_solve_the_row_condition_exactly() {
    // Sinkhorn's row and column targets are not jointly satisfiable (see
    // Biwhitener docs), so its c is a compromise. Given d, the closed form is
    // the c that makes every row variance exactly 1; gene variances must stay
    // as close to 1 as Sinkhorn's own.
    let x = filtered(&counts(500, 120, 7));
    let (c_sk, d, _, ok, _) = Biwhitener::default().compute(&x);
    assert!(ok);
    let c = cell_factors(&CsrMatrix::from_dense(&x), &d);
    let (rows, cols) = bw_variances(&x, &c, &d);
    let row_err = rows.iter().map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
    assert!(row_err < 1e-12, "row variance max |Var−1| = {row_err:.2e}");
    let (_, cols_sk) = bw_variances(&x, &c_sk, &d);
    let dev = |v: &[f64]| v.iter().map(|a| (a - 1.0).abs()).sum::<f64>() / v.len() as f64;
    eprintln!(
        "gene mean |Var−1|: closed-form c {:.4}, Sinkhorn c {:.4}",
        dev(&cols),
        dev(&cols_sk)
    );
    // Exact rows cost a little on genes (fixture: 0.062 vs Sinkhorn's 0.048):
    // the targets conflict, and Sinkhorn spreads the error over both sides.
    assert!(
        dev(&cols) < 0.08,
        "gene variances drifted: mean |Var−1| = {:.4}",
        dev(&cols)
    );
}

#[test]
fn block_accumulation_matches_one_shot() {
    let x = filtered(&counts(500, 120, 11));
    let (c, d, ..) = Biwhitener::default().compute(&x);
    let a = CsrMatrix::from_dense(&x);
    let one = a.biwhitened_covariance(&a.transpose(), &c, &d);
    let mut acc = CovAccumulator::new(x.ncols());
    for s in (0..x.nrows()).step_by(37) {
        let e = (s + 37).min(x.nrows());
        acc.add(
            &row_block(&x, s, e),
            &Col::from_fn(e - s, |i| c.read(s + i)),
            &d,
        );
    }
    assert_eq!(acc.n(), x.nrows());
    let (blk, _) = acc.finish();
    let diff = Mat::from_fn(one.nrows(), one.ncols(), |i, j| {
        blk.read(i, j) - one.read(i, j)
    });
    let err = max_abs(&diff) / max_abs(&one);
    assert!(err < 1e-13, "blocked vs one-shot rel err {err:.2e}");
    for i in 0..blk.nrows() {
        for j in 0..i {
            assert_eq!(
                blk.read(i, j),
                blk.read(j, i),
                "blocked S not exactly symmetric"
            );
        }
    }
}

#[test]
fn streamed_fit_matches_fit_csr_with_same_factors() {
    // d from Sinkhorn on all cells, c in closed form: the streamed pipeline must
    // equal fit_csr run with those factors precomputed, to round-off.
    let x = filtered(&counts(600, 150, 3));
    let (_, d, ..) = Biwhitener::default().compute(&x);
    let (s, _, c) = stream(&x, &d, 97);
    let streamed = SparsePCA::new(cfg()).fit_covariance(s, x.nrows(), c.clone(), d.clone(), None);
    let direct = SparsePCA::new(FistaConfig {
        precomputed_factors: Some((c, d)),
        ..cfg()
    })
    .fit_csr(&CsrMatrix::from_dense(&x));
    assert_results_match(&direct, &streamed, 1e-10);
}

#[test]
fn streamed_factors_match_sinkhorn() {
    // The full streaming recipe for the factors: d from Sinkhorn on half the
    // cells, 2 refinement passes on all cells, c in closed form, S accumulated
    // in blocks. Under the same σ² estimator (full EVD) it must match the
    // all-cells Sinkhorn fit: σ² within 2%, same KS to 0.01, same k.
    //
    // (σ² from EigensolverMode::Subsample is tested separately; on this fixture,
    // whose bulk is only roughly MP (KS ≈ 0.10), it varies ±3% with the cells
    // picked — an estimator property, not a streaming one.)
    let x = filtered(&counts(1500, 200, 17));
    let (n, p) = (x.nrows(), x.ncols());
    let rows: Vec<usize> = (0..n)
        .filter(|&i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63 == 0)
        .collect();
    let sub = Mat::from_fn(rows.len(), p, |t, j| x.read(rows[t], j));
    let (_, d0, _, ok, _) = Biwhitener::default().compute_csr(&CsrMatrix::from_dense(&sub));
    assert!(ok);
    let d = refine(&x, d0, 128, 2);
    let (s, _, c) = stream(&x, &d, 128);
    let gdev_st = (0..p).map(|j| (s.read(j, j) - 1.0).abs()).sum::<f64>() / p as f64;
    let streamed =
        SparsePCA::new(FistaConfig { k_max: 40, ..cfg() }).fit_covariance(s, n, c, d, None);

    let a = CsrMatrix::from_dense(&x);
    let (c_sk, d_sk, ..) = Biwhitener::default().compute(&x);
    let s_sk = a.biwhitened_covariance(&a.transpose(), &c_sk, &d_sk);
    let gdev_sk = (0..p).map(|j| (s_sk.read(j, j) - 1.0).abs()).sum::<f64>() / p as f64;
    let full = SparsePCA::new(FistaConfig { k_max: 40, ..cfg() }).fit_csr(&a);

    let rel = (streamed.sigma_sq - full.sigma_sq).abs() / full.sigma_sq;
    let (kf, ks) = (full.ks_distance.unwrap(), streamed.ks_distance.unwrap());
    eprintln!(
        "gene |Var−1| streamed {gdev_st:.4} Sinkhorn {gdev_sk:.4}; σ² {:.4} vs {:.4} (rel {rel:.2e}); KS {ks:.4} vs {kf:.4}; k {} vs {}",
        streamed.sigma_sq,
        full.sigma_sq,
        streamed.eigenvalues.len(),
        full.eigenvalues.len()
    );
    // The final c is the closed form (rows exact), so genes carry the residual
    // that Sinkhorn splits over both sides (fixture: 0.050 vs 0.037).
    assert!(
        gdev_st < gdev_sk + 0.02,
        "gene variances: streamed {gdev_st:.4} vs Sinkhorn {gdev_sk:.4}"
    );
    assert!(rel < 0.02, "σ² rel err {rel:.3}");
    assert!((kf - ks).abs() < 0.01, "KS {ks:.4} vs {kf:.4}");
    assert_eq!(streamed.eigenvalues.len(), full.eigenvalues.len(), "k");
}

#[test]
fn fit_covariance_with_subsample_spectrum_matches_fit_csr() {
    // The streaming handoff: S and the subsample spectrum built outside fit_csr
    // must give fit_csr's Subsample result to round-off.
    let x = filtered(&counts(600, 150, 9));
    let n = x.nrows();
    let a = CsrMatrix::from_dense(&x);
    let direct = SparsePCA::new(sub_cfg(200)).fit_csr(&a);
    let c = Col::from_fn(n, |i| direct.bw_c[i]);
    let d = Col::from_fn(x.ncols(), |j| direct.bw_d[j]);
    let at = a.transpose();
    let s = a.biwhitened_covariance(&at, &c, &d);
    let mu = a.biwhitened_mean(&at, &c, &d);
    let rows: Vec<usize> = (0..200).map(|t| t * n / 200).collect();
    let spec = subsample_spectrum(&a.centred_rows(&rows, &c, &d, &mu));
    let via = SparsePCA::new(sub_cfg(200)).fit_covariance(s, n, c, d, Some(spec));
    assert_eq!(via.sk_iters, 0);
    assert!(
        (via.sigma_sq - direct.sigma_sq).abs() < 1e-12 * direct.sigma_sq,
        "σ²"
    );
    assert!(
        (via.ks_distance.unwrap() - direct.ks_distance.unwrap()).abs() < 1e-12,
        "KS"
    );
    assert!(
        rel_diff(&via.eigenvalues, &direct.eigenvalues) < 1e-12,
        "Rayleigh quotients"
    );
    assert_eq!(via.components.ncols(), direct.components.ncols(), "k");
}

#[test]
fn rare_gene_reset_and_refine_matches_sinkhorn() {
    // A gene expressed in 4 cells, only 1 of them in the Sinkhorn subsample.
    // Feasible regime: n/(k·p) = 1200/(4·600) < 1, so full-data Sinkhorn has a
    // finite answer. (For n/(k·p) > 1 — or several such genes sharing a cell —
    // there is none: Sinkhorn drifts c_i → 0, d_j → ∞ and erases those cells.
    // That is a property of the biwhitening problem, not of streaming.)
    // After reset_rare_genes + one full-data refinement pass the rare gene's
    // factor must sit where full Sinkhorn puts it, up to the common gauge.
    let mut x = counts(1200, 600, 29);
    let n = x.nrows();
    let g = 598; // the fixture leaves gene 599 empty
    for i in 0..n {
        x.write(i, g, 0.0);
    }
    for i in [0usize, 1, 2, 3] {
        x.write(i, g, 1.0);
    }
    let x = filtered(&x);
    let sub_rows: Vec<usize> = (0..n).step_by(4).collect(); // has cell 0, not 1–3
    let sub = Mat::from_fn(sub_rows.len(), x.ncols(), |t, j| x.read(sub_rows[t], j));
    let sub_csr = CsrMatrix::from_dense(&sub);
    let (_, mut d, ..) = Biwhitener::default().compute_csr(&sub_csr);
    let (_, d_full, ..) = Biwhitener::default().compute(&x);
    let nnz = gene_nnz(&sub_csr);
    assert_eq!(
        nnz[g], 1,
        "fixture: gene seen in exactly one subsample cell"
    );
    assert!(reset_rare_genes(&mut d, &nnz, 5) >= 1);
    let off_after = |passes: usize| {
        let d = refine(&x, d.clone(), 128, passes);
        let mut r: Vec<f64> = (0..d.nrows())
            .map(|k| (d.read(k) / d_full.read(k)).ln())
            .collect();
        let rg = r[g];
        r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (rg - r[r.len() / 2]).abs()
    };
    // closed-form passes contract the error by ≈ √(n/(k·p)) each (here ≈ 0.7):
    // measured 0.44, 0.21, 0.07, 0.027 after 1, 2, 4, 8 passes, floor ≈ 0.02
    let off = off_after(4);
    assert!(
        off < 0.1,
        "4 passes: rare gene factor off by {off:.3} in log"
    );
    let off16 = off_after(16);
    assert!(
        off16 < 0.05,
        "16 passes: rare gene factor off by {off16:.3} in log"
    );
}

// ─── 6. biwhitening collapse diagnostic ──────────────────────────────────────

/// counts(1200, 120) plus one gene detected in exactly `k` cells (count 1).
fn with_rare_gene(k: usize) -> Mat<f64> {
    let mut x = counts(1200, 120, 29);
    let g = 118;
    for i in 0..x.nrows() {
        x.write(i, g, if i < k { 1.0 } else { 0.0 });
    }
    filtered(&x)
}

#[test]
fn collapse_diagnostic_flags_infeasible_rare_gene() {
    // n/(k·p) = 1200/(4·120) = 2.5 > 1: no finite biwhitening, cells 0–3 collapse.
    let x = with_rare_gene(4);
    for r in [
        SparsePCA::new(cfg()).fit(&x),
        SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&x)),
    ] {
        assert!(r.collapsed_cells >= 4, "collapsed {}", r.collapsed_cells);
        let med = {
            let mut v = r.bw_c.clone();
            v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        assert!(
            (0..4).all(|i| r.bw_c[i] < 1e-3 * med),
            "the 4 expressing cells are the collapsed ones"
        );
        assert!(r.max_gene_factor_ratio > 1e3);
    }
}

#[test]
fn collapse_diagnostic_clean_when_feasible_and_after_detection_filter() {
    // healthy fixture: nothing collapses
    let r =
        SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&filtered(&counts(1200, 120, 29))));
    assert_eq!(r.collapsed_cells, 0);
    // the remedy: drop genes detected in < 2·n/p cells, then nothing collapses
    let x = with_rare_gene(4);
    let (n, p) = (x.nrows(), x.ncols());
    let min_cells = (2.0 * n as f64 / p as f64).ceil() as usize;
    let keep: Vec<usize> = (0..p)
        .filter(|&j| (0..n).filter(|&i| x.read(i, j) != 0.0).count() >= min_cells)
        .collect();
    assert!(keep.len() < p, "filter must remove the rare gene");
    let xf = Mat::from_fn(n, keep.len(), |i, j| x.read(i, keep[j]));
    let r = SparsePCA::new(cfg()).fit_csr(&CsrMatrix::from_dense(&xf));
    assert_eq!(
        r.collapsed_cells, 0,
        "after detection filter (≥ {min_cells} cells)"
    );
}
