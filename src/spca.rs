use std::time::Instant;

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Mat, Side};

use crate::biwhitening::Biwhitener;
use crate::rmt::RmtTheory;

/// Configuration for the full Sparse PCA pipeline (Algorithm 2).
///
/// The pipeline implements the method from Chardès et al.:
///   biwhiten → centre → covariance → RMT thresholding → FISTA
pub struct FistaConfig {
    /// Maximum FISTA iterations (default 1000).
    pub max_iterations: usize,
    /// Iterate-change stopping threshold: ‖W_{t+1} − W_t‖_F < tolerance.
    pub tolerance: f64,
    /// Objective-change stopping threshold:
    /// |Tr(W_{t+1}^T S W_{t+1}) − Tr(W_t^T S W_t)| / |Tr(W_t^T S W_t)| < tol_obj.
    pub tol_obj: f64,
    /// L1 sparsity penalty λ (absolute value, in units of the covariance eigenvalues).
    /// Ignored when `lambda_frac` is set.
    pub lambda: f64,
    /// L1 sparsity penalty as a fraction of the MP bulk edge λ+.
    /// When set, overrides `lambda` with `lambda_frac × λ+` inside `fit()`.
    /// Typical range: 0.1–1.0 × λ+.  The paper's "200–500 genes/component"
    /// target corresponds to roughly 0.3–0.7 × λ+ on Zheng2017.
    pub lambda_frac: Option<f64>,
    /// Print per-stage timing and FISTA progress to stderr.
    pub verbose: bool,
    /// Maximum Sinkhorn-Knopp iterations for biwhitening (default 1000).
    pub bw_max_iter: usize,
    /// Sinkhorn under-relaxation factor α ∈ (0, 1].  Default 1.0 (no damping).
    /// Set to 0.5–0.8 if biwhitening oscillates on your data.
    pub bw_damp: f64,
}

impl Default for FistaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            tol_obj: 1e-6,
            lambda: 0.1,
            lambda_frac: None,
            verbose: false,
            bw_max_iter: 1000,
            bw_damp: 1.0,
        }
    }
}

pub struct SparsePCA {
    config: FistaConfig,
}

impl SparsePCA {
    pub fn new(config: FistaConfig) -> Self {
        Self { config }
    }

    /// Run the full Sparse PCA pipeline from Chardès et al. (Algorithm 2).
    ///
    /// # Pipeline stages
    ///
    /// **Stage 0 — Zero filtering**
    /// Drop all-zero rows (empty cells) and columns (never-expressed genes).
    /// These contribute zero-variance dimensions that inflate the bulk
    /// eigenspectrum and cause divide-by-zero in biwhitening.
    ///
    /// **Stage 1 — Biwhitening (Algorithm 1)**
    /// Find scalings c (cells) and d (genes) via Sinkhorn-Knopp so that
    /// X_w = diag(c) X diag(d) has unit per-cell and per-gene second moments.
    /// The algorithm is run on the *original non-negative* matrix; centring
    /// afterward prevents oscillation from negative values in the update.
    /// If biwhitening stagnates with residual > 1e-2, falls back to per-gene
    /// standardisation (divide each column by its standard deviation).
    ///
    /// **Stage 2 — Mean centring**
    /// Subtract column means from X_w.  Required so that
    /// S = X_wc^T X_wc / n equals the true covariance matrix (Section 2.3).
    ///
    /// **Stage 3 — Sample covariance**
    /// Compute the p×p covariance S = X_wc^T X_wc / n using a BLAS-backed
    /// matrix product.  Uses the 1/n convention consistent with MP theory.
    ///
    /// **Stage 4 — Full eigenspectrum (diagnostic)**
    /// Compute all p eigenvalues of S via symmetric eigendecomposition.
    /// Used for the Kolmogorov-Smirnov goodness-of-fit test against the
    /// Marchenko-Pastur distribution (Section 3.1).
    ///
    /// **Stage 5 — RMT thresholding + subspace initialisation**
    /// Compute the MP bulk edge λ+ = (1 + √q)² (Eq. 3).  Run subspace
    /// iteration on S to find the top-k_max eigenvectors and their Rayleigh
    /// quotients.  Count components with Rayleigh quotient > λ+ as signal
    /// (BBP phase transition, Section 2.2).  The top-k eigenvectors
    /// initialise FISTA, warm-starting it close to the solution.
    ///
    /// **Stage 6 — FISTA Sparse PCA (Algorithm 2)**
    /// Maximise Tr(W^T S W) − λ‖W‖₁ subject to W^T W = I_k via proximal
    /// gradient ascent with Nesterov momentum.  Step size γ = 1/(2·λ_max)
    /// satisfies the Lipschitz condition for the gradient 2·S·W.
    pub fn fit(&self, data: &Mat<f64>) -> SparsePCAResult {
        let v = self.config.verbose;

        // --- Stage 0: Drop all-zero rows and columns ---
        let col_nz: Vec<usize> = (0..data.ncols())
            .filter(|&j| (0..data.nrows()).any(|i| data.read(i, j) != 0.0))
            .collect();
        let row_nz: Vec<usize> = (0..data.nrows())
            .filter(|&i| (0..data.ncols()).any(|j| data.read(i, j) != 0.0))
            .collect();
        let data_cow;
        let data: &Mat<f64> = if col_nz.len() == data.ncols() && row_nz.len() == data.nrows() {
            data
        } else {
            data_cow = Mat::from_fn(row_nz.len(), col_nz.len(), |i, j| {
                data.read(row_nz[i], col_nz[j])
            });
            &data_cow
        };

        let (n, p) = (data.nrows(), data.ncols());

        // --- Stage 1: Biwhitening (Algorithm 1) ---
        // Run on the original non-negative data so that Sinkhorn-Knopp sees
        // only non-negative values, which is the domain the algorithm was
        // derived for.  Centring happens in Stage 2 after scaling.
        if v { eprint!("[biwhitening]  {n}×{p} matrix ... "); }
        let t = Instant::now();
        let bw = Biwhitener { max_iter: self.config.bw_max_iter, damp: self.config.bw_damp, ..Biwhitener::default() };
        let (c, d, bw_iters, bw_ok, bw_res) = bw.compute(data);

        // Fallback: if biwhitening stagnated badly, use per-gene standardisation.
        let xw = if !bw_ok && bw_res > 1e-2 {
            if v {
                eprintln!(
                    "stagnated after {bw_iters} iters (residual={bw_res:.2e}) — \
                     falling back to per-gene standardisation ({:.2}s)",
                    t.elapsed().as_secs_f64()
                );
            }
            let col_vars: Vec<f64> = (0..p)
                .map(|j| {
                    let mean = (0..n).map(|i| data.read(i, j)).sum::<f64>() / n as f64;
                    let var = (0..n).map(|i| (data.read(i, j) - mean).powi(2)).sum::<f64>() / n as f64;
                    var.sqrt().max(1e-10)
                })
                .collect();
            Mat::from_fn(n, p, |i, j| data.read(i, j) / col_vars[j])
        } else {
            if v {
                if bw_ok {
                    eprintln!("converged in {bw_iters} iters ({:.2}s)", t.elapsed().as_secs_f64());
                } else if bw_iters < bw.max_iter {
                    eprintln!(
                        "stagnated at iter {bw_iters}/{} residual={bw_res:.2e} ({:.2}s)",
                        bw.max_iter, t.elapsed().as_secs_f64()
                    );
                } else {
                    eprintln!(
                        "WARNING: hit max_iter={}, residual={bw_res:.2e} ({:.2}s)",
                        bw.max_iter, t.elapsed().as_secs_f64()
                    );
                }
            }
            Biwhitener::apply(data, &c, &d)
        };

        // --- Stage 2: Centre the biwhitened data ---
        // S = X_wc^T X_wc / n is the covariance only when X_wc is zero-mean.
        // Centring after biwhitening keeps Stage 1 in the non-negative domain.
        let col_means: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| xw.read(i, j)).sum::<f64>() / n as f64)
            .collect();
        let xwc = Mat::from_fn(n, p, |i, j| xw.read(i, j) - col_means[j]);

        // --- Stage 3: Sample covariance ---
        // S = X_wc^T X_wc / n  (1/n convention, consistent with MP theory).
        if v { eprint!("[covariance]   Xᵀ X / n  ({p}×{p}) ... "); }
        let t = Instant::now();
        let s = sample_covariance(&xwc);
        if v { eprint!("done ({:.2}s)  ", t.elapsed().as_secs_f64()); }

        // --- Stage 4: Full eigenspectrum (for KS diagnostic) ---
        // All p eigenvalues of S; compared against the MP CDF to verify the
        // bulk noise floor matches theory (arXiv:2509.15429, Section II).  O(p³).
        if v { eprint!("[eigenspectrum] full EVD ({p}×{p}) ... "); }
        let t = Instant::now();
        let rmt_pre = RmtTheory { q: p as f64 / n as f64 };
        let evd = SelfAdjointEigendecomposition::new(s.as_ref(), Side::Lower);
        let raw_eigs: Vec<f64> = (0..p).map(|i| evd.s().column_vector().read(i)).collect();
        if v { eprint!("done ({:.2}s)  ", t.elapsed().as_secs_f64()); }

        // --- Stage 4b: Biwhitening scale normalisation (Algorithm 1, final step) ---
        // After Sinkhorn converges, Algorithm 1 rescales by σ so that the
        // biwhitened covariance is on the canonical MP scale (median eigenvalue
        // of the bulk = median of MP).  Specifically:
        //   σ² = ℓ_med / λ_med
        // where ℓ_med is the median bulk eigenvalue of S and λ_med is the
        // median of the MP distribution with q = p/n.
        // Dividing S by σ² centres the bulk on the standard MP distribution,
        // which is what the RMT formulas (λ+, BBP, predicted overlap) assume.
        let lambda_med_mp = rmt_pre.mp_median();
        let lplus_pre = rmt_pre.lambda_plus();
        let mut bulk_eigs: Vec<f64> = raw_eigs.iter().cloned()
            .filter(|&e| e > 0.001 && e <= lplus_pre)
            .collect();
        bulk_eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let sigma_sq = if !bulk_eigs.is_empty() {
            let l_med = bulk_eigs[bulk_eigs.len() / 2];
            l_med / lambda_med_mp
        } else { 1.0 };
        if v && (sigma_sq - 1.0).abs() > 1e-3 {
            eprintln!("[normalisation] σ² = {sigma_sq:.4}  (bulk median / MP median)");
        } else if v {
            eprintln!("[normalisation] σ² = {sigma_sq:.4}  (≈1, scale already correct)");
        }
        let s = Mat::from_fn(p, p, |i, j| s.read(i, j) / sigma_sq);
        let s_eigenvalues: Vec<f64> = raw_eigs.iter().map(|&e| e / sigma_sq).collect();

        // --- Stage 5: RMT thresholding + subspace initialisation ---
        // λ+ = (1 + √q)² is the MP bulk edge (Eq. 3).  Eigenvalues above λ+
        // are signal outliers via the BBP phase transition (Section 2.2).
        //
        // Subspace iteration on S converges to the top-k_max eigenvectors.
        // Rayleigh quotients rq_j = v_j^T S v_j approximate the eigenvalues
        // without a full O(p³) decomposition.
        if v { eprint!("[RMT/subspace] subspace iteration (k_max=20) ... "); }
        let t = Instant::now();
        let rmt = rmt_pre;  // reuse — same q = p/n
        let lambda_plus = rmt.lambda_plus();
        let k_max = 20_usize.min(p).min(n);
        let v_cand = subspace_iteration(&s, k_max, 100);
        // Rayleigh quotients  rq_j = v_j^T S v_j
        let sv_cand = mat_mul(&s, &v_cand);
        let rq: Vec<f64> = (0..k_max)
            .map(|j| (0..p).map(|i| v_cand.read(i, j) * sv_cand.read(i, j)).sum())
            .collect();
        let lmax = rq.iter().cloned().fold(0.0_f64, f64::max);
        // k = number of eigenvalues above the BBP threshold; at least 1.
        let k = rq.iter().filter(|&&r| r > lambda_plus).count().max(1);
        let v_init = Mat::from_fn(p, k, |i, j| v_cand.read(i, j));
        if v {
            eprintln!(
                "λ_max = {lmax:.4}  λ+ = {lambda_plus:.4}  components = {k}  ({:.2}s)",
                t.elapsed().as_secs_f64()
            );
        }

        // --- Stage 6: FISTA Sparse PCA (Algorithm 2) ---
        // Step size γ = 1/(2·λ_max): the Lipschitz constant of ∇Tr(W^T S W)
        // with respect to W is 2·‖S‖₂ = 2·λ_max (Section 4.1).
        // λ is either supplied directly or as a fraction of λ+ (so that the
        // sparsity level scales with the noise floor rather than the signal).
        let gamma = if lmax > 1e-14 { 1.0 / (2.0 * lmax) } else { 0.5 };
        let lambda = match self.config.lambda_frac {
            Some(frac) => frac * lambda_plus,
            None => self.config.lambda,
        };
        if v { eprintln!("[FISTA]        λ = {lambda:.4e}  γ = {gamma:.4e}  (γλ = {:.4e})  max_iter = {}  tol = {}  tol_obj = {}",
            gamma * lambda, self.config.max_iterations,
            self.config.tolerance, self.config.tol_obj); }
        let t = Instant::now();
        let components = fista_sparse_pca(
            &s,
            &v_init,
            gamma,
            lambda,
            self.config.max_iterations,
            self.config.tolerance,
            self.config.tol_obj,
            v,
        );
        if v { eprintln!("[FISTA]        done ({:.2}s)", t.elapsed().as_secs_f64()); }

        // Rayleigh quotients for the k signal components (used in validate.rs
        // for the predicted-overlap table via BBP formula, Eq. 9).
        let eigenvalues: Vec<f64> = rq[..k].to_vec();

        SparsePCAResult { components, eigenvalues, s_eigenvalues, lambda_plus, q: rmt.q }
    }
}

/// Output of the Sparse PCA pipeline.
pub struct SparsePCAResult {
    /// Sparse loading matrix W (p × k).  Each column is one component;
    /// most entries are zero due to the L1 penalty.
    pub components: Mat<f64>,
    /// Rayleigh quotients v_j^T S v_j for the k signal components (descending).
    /// Used to compute predicted overlaps via the BBP formula (Eq. 9).
    pub eigenvalues: Vec<f64>,
    /// All p eigenvalues of the biwhitened covariance S (ascending order).
    /// Used for the Kolmogorov-Smirnov test against the MP distribution.
    pub s_eigenvalues: Vec<f64>,
    /// MP bulk edge λ+ = (1 + √q)² computed from the *filtered* (n, p).
    pub lambda_plus: f64,
    /// Aspect ratio q = p/n used for RMT, after zero-row/col filtering.
    pub q: f64,
}

// ---------------------------------------------------------------------------
// Algorithm 2: FISTA Sparse PCA
// ---------------------------------------------------------------------------

/// FISTA proximal gradient ascent for sparse PCA (Algorithm 2 in the paper).
///
/// Solves:  max_W  Tr(W^T S W) − λ‖W‖₁   s.t.  W^T W = I_k
///
/// Each iteration:
/// 1. **Gradient step**: Z = Y + 2γ·S·Y
///    (gradient of Tr(Y^T S Y) is 2·S·Y; step size γ = 1/(2·λ_max))
/// 2. **Proximal step**: soft-threshold each entry of Z with threshold γλ
///    (prox operator of the L1 penalty: sign(z)·max(|z|−γλ, 0))
/// 3. **Orthonormalise**: modified Gram-Schmidt to enforce W^T W = I_k
/// 4. **Nesterov momentum**: extrapolate between consecutive iterates with
///    coefficient β_t = (t−1)/(t+1) where t follows the FISTA update rule
///
/// Stops when either the iterate change ‖W_{t+1}−W_t‖_F < `tol` **or** the
/// relative objective change falls below `tol_obj` (OR criterion prevents
/// getting stuck when rotational ambiguity keeps ‖ΔW‖ large).
pub fn fista_sparse_pca(
    s: &Mat<f64>,
    v_init: &Mat<f64>,
    gamma: f64,
    lambda: f64,
    max_iter: usize,
    tol: f64,
    tol_obj: f64,
    verbose: bool,
) -> Mat<f64> {
    let (p, k) = (v_init.nrows(), v_init.ncols());
    let mut w = v_init.clone();
    let mut y = v_init.clone();
    let mut t = 1.0_f64;

    let mut obj_prev = trace_wt_s_w(s, v_init);

    for iter in 0..max_iter {
        let prev_w = w.clone();

        // Step 1: gradient ascent  Z = Y + 2γ S Y
        let sy = mat_mul(s, &y);
        let mut z = Mat::from_fn(p, k, |i, j| y.read(i, j) + 2.0 * gamma * sy.read(i, j));

        // Step 2: soft-threshold (proximal operator of λ‖·‖₁)
        // threshold = γλ; entries with |z_ij| ≤ γλ are zeroed out.
        for i in 0..p {
            for j in 0..k {
                let v = z.read(i, j);
                let shrunk = v.abs() - lambda * gamma;
                z.as_mut().write(i, j, if shrunk > 0.0 { shrunk * v.signum() } else { 0.0 });
            }
        }

        // Step 3: orthonormalise columns to enforce W^T W = I_k
        z = orthonormalize(&z);

        // Stopping criteria
        let obj_new = trace_wt_s_w(s, &z);
        let dw = frobenius_diff(&z, &prev_w, p, k);
        let dvar = (obj_new - obj_prev).abs();
        let iterate_ok = dw < tol;
        let obj_ok = dvar / (obj_prev.abs() + 1e-10) < tol_obj;

        if verbose && (iter + 1) % 50 == 0 {
            eprintln!(
                "[FISTA]        iter {:>4}  Var={:.6}  ΔVar={:.2e}  ΔW={:.2e}",
                iter + 1, obj_new, dvar, dw
            );
        }

        obj_prev = obj_new;

        if iterate_ok || obj_ok {
            w = z;
            break;
        }

        // Step 4: Nesterov momentum update.
        // Standard FISTA schedule: t_{k+1} = (1 + √(1 + 4t_k²)) / 2.
        // Note: the paper (Algorithm 2, arXiv:2509.15429) uses the formula
        // t_{k+1} = (1/20 + 1 + 4·t_k²) / 2, which causes t to grow
        // super-exponentially and β = (t_k−1)/t_{k+1} → 0 quickly,
        // effectively reducing to gradient ascent after a few steps.
        // We use the standard schedule as it is more numerically stable
        // and gives the classical O(1/k²) convergence guarantee.
        let next_t = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let beta = (t - 1.0) / next_t;
        y = Mat::from_fn(p, k, |i, j| z.read(i, j) + beta * (z.read(i, j) - prev_w.read(i, j)));
        w = z;
        t = next_t;
    }

    w
}

/// Tr(W^T S W) — total variance explained by loading matrix W.
fn trace_wt_s_w(s: &Mat<f64>, w: &Mat<f64>) -> f64 {
    let sw = mat_mul(s, w);
    let (p, k) = (w.nrows(), w.ncols());
    let mut acc = 0.0_f64;
    for i in 0..p {
        for j in 0..k {
            acc += w.read(i, j) * sw.read(i, j);
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// Dense matrix product C = A B  (BLAS-backed via faer).
fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a.as_ref() * b.as_ref()
}

/// Sample covariance S = X^T X / n  (p×p, 1/n convention, BLAS-backed).
fn sample_covariance(x: &Mat<f64>) -> Mat<f64> {
    let inv_n = 1.0 / x.nrows() as f64;
    let s: Mat<f64> = x.as_ref().transpose() * x.as_ref();
    Mat::from_fn(s.nrows(), s.ncols(), |i, j| s.read(i, j) * inv_n)
}

/// Modified Gram-Schmidt orthonormalisation of the columns of A (in-place).
///
/// Enforces W^T W = I_k after the soft-threshold step, as required by the
/// orthogonality constraint in Algorithm 2.
fn orthonormalize(a: &Mat<f64>) -> Mat<f64> {
    let (p, k) = (a.nrows(), a.ncols());
    let mut q = a.clone();
    for j in 0..k {
        for jj in 0..j {
            let dot: f64 = (0..p).map(|i| q.read(i, jj) * q.read(i, j)).sum();
            for i in 0..p {
                let v = q.read(i, j) - dot * q.read(i, jj);
                q.as_mut().write(i, j, v);
            }
        }
        let norm: f64 = (0..p).map(|i| q.read(i, j).powi(2)).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for i in 0..p {
                let v = q.read(i, j) / norm;
                q.as_mut().write(i, j, v);
            }
        }
    }
    q
}

/// Simultaneous (block) power iteration: returns top-k eigenvectors of S.
///
/// More efficient than sequential deflation for k > 1: each iteration
/// applies S to all k vectors simultaneously (one BLAS gemm), then
/// orthonormalises.  Used to warm-start FISTA and to compute Rayleigh
/// quotients for the BBP threshold test.
fn subspace_iteration(s: &Mat<f64>, k: usize, n_iter: usize) -> Mat<f64> {
    let p = s.nrows();
    let mut v = Mat::from_fn(p, k, |i, j| if i == j { 1.0 } else { 0.0 });
    for _ in 0..n_iter {
        let sv = mat_mul(s, &v);
        v = orthonormalize(&sv);
    }
    v
}

/// Frobenius distance ‖A − B‖_F, used for the iterate-change stopping test.
fn frobenius_diff(a: &Mat<f64>, b: &Mat<f64>, p: usize, k: usize) -> f64 {
    (0..p)
        .flat_map(|i| (0..k).map(move |j| (a.read(i, j) - b.read(i, j)).powi(2)))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthonormalize_produces_orthonormal_columns() {
        let a = Mat::from_fn(4, 2, |i, j| (i * 2 + j + 1) as f64);
        let q = orthonormalize(&a);
        for i in 0..2 {
            for j in 0..2 {
                let dot: f64 = (0..4).map(|k| q.read(k, i) * q.read(k, j)).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "q^T q [{i},{j}] = {dot}");
            }
        }
    }

    #[test]
    fn fista_recovers_top_eigenvector() {
        // S = 3·v₀v₀^T + 0.01·I,  v₀ = [1,1,1,1]/2
        // λ_max ≈ 3·‖v₀‖²·4 = 3.01; FISTA with λ=0 should recover v₀.
        let p = 4;
        let v0 = vec![0.5, 0.5, 0.5, 0.5];
        let mut s = Mat::from_fn(p, p, |i, j| 3.0 * v0[i] * v0[j] + if i == j { 0.01 } else { 0.0 });
        let _ = &mut s;

        let v_init = Mat::from_fn(p, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let lmax = 3.01_f64;
        let gamma = 1.0 / (2.0 * lmax);

        let w = fista_sparse_pca(&s, &v_init, gamma, 0.0, 500, 1e-8, 1e-8, false);

        let dot: f64 = (0..p).map(|i| w.read(i, 0) * v0[i]).sum::<f64>();
        assert!(dot.abs() > 0.99, "alignment = {dot}");
    }
}
