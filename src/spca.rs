use std::time::Instant;

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Col, Mat, Side};

use crate::biwhitening::Biwhitener;
use crate::rmt::RmtTheory;
use crate::sparse::CsrMatrix;

/// Controls how the pipeline estimates σ² (the bulk covariance scale factor)
/// and whether the full eigenspectrum is computed.
///
/// The default is `Full`, which is the only safe choice for production use.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum EigensolverMode {
    /// Full O(p³) symmetric EVD (default).
    ///
    /// Computes all p eigenvalues of the biwhitened covariance S.
    /// - σ² = median(bulk eigenvalues) / median(MP distribution) — exact and
    ///   robust: filtered to bulk range, unaffected by signal outliers.
    /// - KS diagnostic available (requires `compute_ks = true`).
    #[default]
    Full,

    /// ⚠️  APPROXIMATE — validate against `Full` before use in production.
    ///
    /// Skips the O(p³) EVD entirely.  Estimates σ² from the matrix trace:
    ///
    ///   σ²_fast = Tr(S) / min(n−1, p)
    ///
    /// This is O(p) (sum of diagonal entries) and gives the mean eigenvalue
    /// rather than the median.  The bias relative to the exact σ² is:
    ///
    ///   bias ≈ k × (λ̄_signal − σ²_true) / p_eff
    ///
    /// For typical scRNA-seq (k=5, λ̄≈5, p=5000): bias ≈ 0.4%, shifting λ+
    /// by the same amount — generally harmless.  For small p or strong signal
    /// (k=5, λ̄=15, p=1000): bias ≈ 7%, which can cause missed or spurious
    /// signal components.
    ///
    /// **KS diagnostic is unavailable in this mode** (`ks_distance` is `None`).
    /// `s_eigenvalues` in the result is empty.
    ///
    /// Typical speedup on large matrices: 5–15× over `Full`.
    /// Always compare `sigma_sq` from both modes on your data before switching.
    Fast,

    /// σ² and the KS diagnostic from the exact spectrum of a **cell subsample**,
    /// via its m×m Gram matrix: O(m²p + m³) instead of O(p³).
    ///
    /// Takes `n_cells` = m evenly spaced cells (deterministic: stratified on
    /// files sorted in blocks, e.g. by sample; avoid if rows repeat with a period
    /// that aliases the stride), centred with the
    /// full-data means. The non-zero eigenvalues of Y_subᵀY_sub/(m−1) equal those
    /// of the Gram Y_sub Y_subᵀ/(m−1); their bulk is MP with q' = p/m, and the
    /// noise scale σ² is the same per-entry quantity as on the full data. So σ²
    /// (bulk median / MP(q') median) and KS (against MP(q')) are exact for the
    /// subsample and consistent estimators for the whole matrix.
    ///
    /// λ+ and the signal count k still use the full q = p/n. `s_eigenvalues`
    /// holds the subsample spectrum; `top_eigenvectors` is empty.
    /// With `n_cells ≥ n` this reproduces `Full`'s σ² and KS to round-off.
    Subsample { n_cells: usize },
}

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
    /// **Eq. 8 penalty selection** (the paper's procedure; overrides `lambda`
    /// and `lambda_frac` when set). `Some(scale)` root-finds λ* such that the
    /// distance between the sparse subspace W and the outlier eigenspace V,
    /// `Σ_j (1 − σ_j²)` with σ_j the singular values of WᵀV, equals the
    /// RMT-predicted `Σ_j (1 − overlap_j)` (BBP squared overlaps of the k
    /// retained components), then fits at `scale × λ*`. The reference uses
    /// `scale = 0.6`. Each root-finding step is one FISTA run (~8–12 steps),
    /// so this costs about ten fits. See PLAN.md: a fixed `lambda_frac` has no
    /// relation to the spectrum and collapses components to single genes.
    pub auto_penalty: Option<f64>,
    /// Print per-stage timing and FISTA progress to stderr.
    pub verbose: bool,
    /// Maximum Sinkhorn-Knopp iterations for biwhitening (default 1000).
    pub bw_max_iter: usize,
    /// Sinkhorn under-relaxation factor α ∈ (0, 1].  Default 1.0 (undamped,
    /// as in the reference).  α < 1 is never needed for convergence and only
    /// slows it; see `Biwhitener::damp`.
    pub bw_damp: f64,
    /// Eigensolver mode controlling σ² estimation and eigenspectrum computation.
    /// Default: `EigensolverMode::Full` (exact, safe for production).
    /// Set to `EigensolverMode::Fast` to skip the O(p³) EVD — read its
    /// documentation carefully before use.
    pub eigensolver: EigensolverMode,
    /// Whether to compute the Kolmogorov-Smirnov goodness-of-fit statistic
    /// against the Marchenko-Pastur distribution (default: true).
    /// Only available with `EigensolverMode::Full`.
    /// Set to `false` to skip the KS test when the diagnostic is not needed;
    /// the eigenspectrum is still computed for σ² in `Full` mode regardless.
    pub compute_ks: bool,
    /// **P2.A** — precomputed biwhitening factors (c, d). When set, the
    /// Sinkhorn-Knopp iteration is skipped and these factors are used directly.
    /// Lengths must match the matrix shape *after* Stage 0 zero-row/col filtering;
    /// the driver is responsible for subsetting `c` to the cells that survive
    /// filtering, and likewise `d` for genes. When `None` (default), biwhitening
    /// runs as usual.
    pub precomputed_factors: Option<(Col<f64>, Col<f64>)>,
    /// **P2.B** — how many top eigenvectors of the σ̂²-rescaled covariance to
    /// return in `SparsePCAResult::top_eigenvectors` (default 20). Has no
    /// effect in `EigensolverMode::Fast` (no full EVD).
    pub top_eigvec_count: usize,
    /// Ceiling on how many signal components are fitted (default 20).
    ///
    /// In `EigensolverMode::Full` the number of eigenvalues above λ+ is known
    /// exactly from Stage 4 and is always reported in
    /// `SparsePCAResult::n_above_threshold`; `k_max` only caps how many of them
    /// FISTA fits (cost O(p² · k · iters)), and `verbose` says when it did.
    /// The reference (`BiwhitenedCovarianceEstimator.fit` in
    /// `spcarmt/scrna/rmt/_covariance.py`) has no cap.
    ///
    /// In `Fast` and `Subsample` there is no full EVD: Stage 5 runs subspace
    /// iteration on `k_max` vectors, so k = `count(ritz value > λ+)` is bounded
    /// by `k_max` and a result with k = `k_max` is a ceiling, not a count.
    /// Was a hard-coded 20, which bound on every real dataset tested (be1,
    /// 1715×2000, has 72 eigenvalues above λ+).
    pub k_max: usize,
    /// One S-product per FISTA iteration instead of two (default false).
    ///
    /// The objective needs S·W_t and the next gradient needs S·Y_{t+1}, with
    /// Y_{t+1} = W_t + β(W_t − W_{t−1}). By linearity S·Y_{t+1} = S·W_t +
    /// β(S·W_t − S·W_{t−1}), both already computed, so the O(p²k) gradient
    /// product is dropped. Equal in exact arithmetic; in floating point the
    /// iterates differ at round-off level.
    pub fista_reuse_products: bool,
}

impl Default for FistaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            tol_obj: 1e-6,
            lambda: 0.1,
            lambda_frac: None,
            auto_penalty: None,
            verbose: false,
            bw_max_iter: 1000,
            bw_damp: 1.0,
            eigensolver: EigensolverMode::Full,
            compute_ks: true,
            precomputed_factors: None,
            top_eigvec_count: 20,
            k_max: 20,
            fista_reuse_products: false,
        }
    }
}

/// Output of Stages 0–3: the biwhitened covariance and the biwhitening state.
struct Front {
    s: Mat<f64>,
    n: usize,
    p: usize,
    c: Col<f64>,
    d: Col<f64>,
    bw_iters: usize,
    bw_ok: bool,
    bw_res: f64,
    /// `EigensolverMode::Subsample`: (subsample spectrum ascending, q' = p/m).
    sub_spectrum: Option<(Vec<f64>, f64)>,
}

/// The Sparse PCA pipeline runner.
///
/// Construct with [`SparsePCA::new`] and call [`SparsePCA::fit`].
pub struct SparsePCA {
    config: FistaConfig,
}

impl SparsePCA {
    /// Create a new pipeline runner with the given configuration.
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
    /// If biwhitening hits `bw_max_iter` with X_w's variances still moving by
    /// more than 1e-2 per iteration, falls back to per-gene standardisation
    /// (divide each column by its standard deviation).
    ///
    /// **Stage 2 — Mean centring**
    /// Subtract column means from X_w.  Required so that
    /// S = X_wc^T X_wc / (n−1) equals the sample covariance (Section 2.3).
    ///
    /// **Stage 3 — Sample covariance**
    /// Compute the p×p covariance S = X_wc^T X_wc / (n−1) using a BLAS-backed
    /// matrix product.  Uses the 1/(n−1) convention matching the Python reference.
    ///
    /// **Stage 4 — Full eigenspectrum (diagnostic)**
    /// Compute all p eigenvalues of S via symmetric eigendecomposition.
    /// Used for the Kolmogorov-Smirnov goodness-of-fit test against the
    /// Marchenko-Pastur distribution (Section 3.1).
    ///
    /// **Stage 5 — RMT thresholding + FISTA start**
    /// Compute the MP bulk edge λ+ = (1 + √q)² (Eq. 3).  Count eigenvalues
    /// above λ+ as signal (BBP phase transition, Section 2.2); their
    /// eigenvectors initialise FISTA, as in the reference's
    /// `BiwhitenedCovarianceEstimator.fit` / `_fista_spca`.  In `Full` mode
    /// these come from Stage 4's exact EVD; `Fast` and `Subsample` use
    /// subspace iteration on `k_max` vectors instead (approximate near λ+).
    ///
    /// **Stage 6 — FISTA Sparse PCA (Algorithm 2)**
    /// Maximise Tr(W^T S W) − λ‖W‖₁ subject to W^T W = I_k via proximal
    /// gradient ascent with Nesterov momentum.  Step size γ = 0.5/(2·λ_max),
    /// matching the Python reference (half the theoretical 1/L optimum).
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
        //
        // P2.A path: if precomputed factors were provided in the config, use
        // them directly and skip Sinkhorn-Knopp.  Caller is responsible for
        // shape-matching to the (post-filter) (n, p) — we assert it here.
        if v {
            eprint!("[biwhitening]  {n}×{p} matrix ... ");
        }
        let t = Instant::now();
        let bw = Biwhitener {
            max_iter: self.config.bw_max_iter,
            damp: self.config.bw_damp,
            ..Biwhitener::default()
        };
        let (c, d, bw_iters, bw_ok, bw_res) = match &self.config.precomputed_factors {
            Some((c_in, d_in)) => {
                assert_eq!(
                    c_in.nrows(),
                    n,
                    "precomputed_factors c length {} != n {} (after zero-row/col filter)",
                    c_in.nrows(),
                    n
                );
                assert_eq!(
                    d_in.nrows(),
                    p,
                    "precomputed_factors d length {} != p {} (after zero-row/col filter)",
                    d_in.nrows(),
                    p
                );
                if v {
                    eprint!("[precomputed factors, skip Sinkhorn-Knopp] ");
                }
                (c_in.clone(), d_in.clone(), 0_usize, true, 0.0_f64)
            }
            None => bw.compute(data),
        };

        // Fallback: if biwhitening did not converge and X_w's variances were
        // still moving by > 1e-2 per iteration, use per-gene standardisation.
        let xw = if !bw_ok && bw_res > 1e-2 {
            if v {
                eprintln!(
                    "not converged after {bw_iters} iters (residual={bw_res:.2e}) — \
                     falling back to per-gene standardisation ({:.2}s)",
                    t.elapsed().as_secs_f64()
                );
            }
            let col_vars: Vec<f64> = (0..p)
                .map(|j| {
                    let mean = (0..n).map(|i| data.read(i, j)).sum::<f64>() / n as f64;
                    let var = (0..n)
                        .map(|i| (data.read(i, j) - mean).powi(2))
                        .sum::<f64>()
                        / n as f64;
                    var.sqrt().max(1e-10)
                })
                .collect();
            Mat::from_fn(n, p, |i, j| data.read(i, j) / col_vars[j])
        } else {
            if v {
                if bw_ok {
                    eprintln!(
                        "converged in {bw_iters} iters ({:.2}s)",
                        t.elapsed().as_secs_f64()
                    );
                } else {
                    eprintln!(
                        "WARNING: hit max_iter={}, residual={bw_res:.2e} ({:.2}s)",
                        bw.max_iter,
                        t.elapsed().as_secs_f64()
                    );
                }
            }
            Biwhitener::apply(data, &c, &d)
        };

        // --- Stage 2: Centre the biwhitened data ---
        // S = X_wc^T X_wc / (n-1) is the covariance only when X_wc is zero-mean.
        // Centring after biwhitening keeps Stage 1 in the non-negative domain.
        let col_means: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| xw.read(i, j)).sum::<f64>() / n as f64)
            .collect();
        let xwc = Mat::from_fn(n, p, |i, j| xw.read(i, j) - col_means[j]);

        // --- Stage 3: Sample covariance ---
        // S = X_wc^T X_wc / (n-1)  (unbiased estimator, matches Python reference).
        if v {
            eprint!("[covariance]   Xᵀ X / (n-1)  ({p}×{p}) ... ");
        }
        let t = Instant::now();
        let s = sample_covariance(&xwc);
        if v {
            eprint!("done ({:.2}s)  ", t.elapsed().as_secs_f64());
        }

        let sub_spectrum = self
            .subsample_rows(n)
            .map(|rows| gram_spectrum(&Mat::from_fn(rows.len(), p, |t, j| xwc.read(rows[t], j))));
        self.decompose(Front {
            s,
            n,
            p,
            c,
            d,
            bw_iters,
            bw_ok,
            bw_res,
            sub_spectrum,
        })
    }

    /// Stages 4–6 on a covariance built elsewhere, e.g. streamed with
    /// [`crate::streaming::CovAccumulator`].
    ///
    /// `s` is the p×p biwhitened, centred covariance of `n` cells; `c`, `d` are
    /// the factors it was built with (reported back in the result). With
    /// `EigensolverMode::Subsample`, pass the subsample spectrum from
    /// [`crate::streaming::subsample_spectrum`]; it is ignored otherwise.
    pub fn fit_covariance(
        &self,
        s: Mat<f64>,
        n: usize,
        c: Col<f64>,
        d: Col<f64>,
        sub_spectrum: Option<(Vec<f64>, f64)>,
    ) -> SparsePCAResult {
        let p = s.nrows();
        assert_eq!(s.ncols(), p, "S must be square");
        assert_eq!(d.nrows(), p, "d length != p");
        assert_eq!(c.nrows(), n, "c length != n");
        let sub_spectrum = match self.config.eigensolver {
            EigensolverMode::Subsample { .. } => Some(
                sub_spectrum
                    .expect("EigensolverMode::Subsample needs sub_spectrum in fit_covariance"),
            ),
            _ => None,
        };
        self.decompose(Front {
            s,
            n,
            p,
            c,
            d,
            bw_iters: 0,
            bw_ok: true,
            bw_res: 0.0,
            sub_spectrum,
        })
    }

    /// Evenly spaced cell indices for `EigensolverMode::Subsample`, else `None`.
    fn subsample_rows(&self, n: usize) -> Option<Vec<usize>> {
        match self.config.eigensolver {
            EigensolverMode::Subsample { n_cells } => {
                let m = n_cells.clamp(2, n);
                Some((0..m).map(|t| t * n / m).collect())
            }
            _ => None,
        }
    }

    /// [`SparsePCA::fit`] for a sparse (CSR) cells × genes matrix.
    ///
    /// Stages 0–3 run on the sparse matrix: Stage 0 drops empty rows/cols,
    /// Sinkhorn-Knopp uses sparse products, and the covariance is formed as
    /// (YᵀY − nμμᵀ)/(n−1) with Y = diag(c) X diag(d), in Σᵢ nnzᵢ² work.
    /// No n×p dense matrix exists at any point: memory is O(nnz + p²).
    /// Stages 4–6 are the same code as the dense path.
    ///
    /// Non-convergence fallback: per-gene standardisation, as in `fit`. Here the
    /// returned `bw_c`/`bw_d` are the factors actually applied (c = 1,
    /// d = 1/sd); the dense path reports the unconverged Sinkhorn factors.
    pub fn fit_csr(&self, data: &CsrMatrix) -> SparsePCAResult {
        let v = self.config.verbose;
        let (data, _, _) = data.drop_empty();
        let (n, p) = (data.nrows(), data.ncols());
        let xt = data.transpose();

        if v {
            eprint!("[biwhitening]  {n}×{p} CSR, nnz={} ... ", data.nnz());
        }
        let t = Instant::now();
        let (c, d, bw_iters, bw_ok, bw_res) = match &self.config.precomputed_factors {
            Some((c_in, d_in)) => {
                assert_eq!(
                    c_in.nrows(),
                    n,
                    "precomputed_factors c length != n (after zero-row/col filter)"
                );
                assert_eq!(
                    d_in.nrows(),
                    p,
                    "precomputed_factors d length != p (after zero-row/col filter)"
                );
                (c_in.clone(), d_in.clone(), 0_usize, true, 0.0_f64)
            }
            None => Biwhitener {
                max_iter: self.config.bw_max_iter,
                damp: self.config.bw_damp,
                ..Biwhitener::default()
            }
            .compute_csr_t(&data, &xt),
        };
        let (c, d) = if !bw_ok && bw_res > 1e-2 {
            if v {
                eprintln!("not converged after {bw_iters} iters (residual={bw_res:.2e}) — falling back to per-gene standardisation");
            }
            let sd = Col::from_fn(p, |j| {
                let (s1, s2) = xt
                    .row(j)
                    .fold((0.0, 0.0), |(a, b), (_, x)| (a + x, b + x * x));
                let mean = s1 / n as f64;
                (s2 / n as f64 - mean * mean).max(0.0).sqrt().max(1e-10)
            });
            (
                Col::from_fn(n, |_| 1.0),
                Col::from_fn(p, |j| 1.0 / sd.read(j)),
            )
        } else {
            if v {
                eprintln!(
                    "{bw_iters} iters, converged={bw_ok} ({:.2}s)",
                    t.elapsed().as_secs_f64()
                );
            }
            (c, d)
        };

        if v {
            eprint!("[covariance]   sparse YᵀY − nμμᵀ  ({p}×{p}) ... ");
        }
        let t = Instant::now();
        let s = data.biwhitened_covariance(&xt, &c, &d);
        if v {
            eprint!("done ({:.2}s)  ", t.elapsed().as_secs_f64());
        }
        let sub_spectrum = self.subsample_rows(n).map(|rows| {
            let mu = data.biwhitened_mean(&xt, &c, &d);
            gram_spectrum(&data.centred_rows(&rows, &c, &d, &mu))
        });
        drop((data, xt));

        self.decompose(Front {
            s,
            n,
            p,
            c,
            d,
            bw_iters,
            bw_ok,
            bw_res,
            sub_spectrum,
        })
    }

    /// Stages 4–6 (σ², RMT threshold, FISTA), shared by the dense and CSR paths.
    fn decompose(&self, f: Front) -> SparsePCAResult {
        let v = self.config.verbose;
        let Front {
            s,
            n,
            p,
            c,
            d,
            bw_iters,
            bw_ok,
            bw_res,
            sub_spectrum,
        } = f;
        // q of the spectrum σ² and KS are computed from (subsample q' or full q)
        let mut spectrum_q = p as f64 / n as f64;

        // --- Stage 4 / 4b: σ² estimation and eigenspectrum ---
        //
        // Two modes controlled by `config.eigensolver`:
        //
        // Full (default): O(p³) symmetric EVD → exact bulk-median σ², full
        //   eigenspectrum for KS diagnostic.
        //
        // Fast: O(p) trace estimator → approximate σ², no KS diagnostic.
        //   See `EigensolverMode::Fast` documentation for bias analysis.
        let rmt_pre = RmtTheory {
            q: p as f64 / n as f64,
        };

        // P2.B: capture the top-K eigenvectors when running Full EVD. faer's
        // SelfAdjointEigendecomposition returns eigenvalues sorted ascending
        // in `evd.s()` and eigenvectors as columns of `evd.u()`. So the top-K
        // (largest) are the LAST K columns of u(), in reverse order.
        let top_eigvec_count = self.config.top_eigvec_count.min(p);
        let mut top_eigenvectors: Vec<Vec<f64>> = Vec::new();
        // Full mode keeps the eigenvectors (columns ascending with `evd.s()`):
        // Stage 5 takes k and FISTA's start from them, as the reference does.
        let mut eigvecs: Option<Mat<f64>> = None;

        let (sigma_sq, s_eigenvalues): (f64, Vec<f64>) = match self.config.eigensolver {
            EigensolverMode::Full => {
                if v {
                    eprint!("[eigenspectrum] full EVD ({p}×{p}) ... ");
                }
                let t = Instant::now();
                let evd = SelfAdjointEigendecomposition::new(s.as_ref(), Side::Lower);
                let raw_eigs: Vec<f64> = (0..p).map(|i| evd.s().column_vector().read(i)).collect();
                let u = evd.u();
                // last K columns, reversed to be sorted descending
                for k in 0..top_eigvec_count {
                    let col_idx = p - 1 - k;
                    let col: Vec<f64> = (0..p).map(|i| u.read(i, col_idx)).collect();
                    top_eigenvectors.push(col);
                }
                eigvecs = Some(u.to_owned());
                if v {
                    eprint!("done ({:.2}s)  ", t.elapsed().as_secs_f64());
                }

                let lambda_med_mp = rmt_pre.mp_median();
                let lplus_pre = rmt_pre.lambda_plus();
                let mut bulk_eigs: Vec<f64> = raw_eigs
                    .iter()
                    .cloned()
                    .filter(|&e| e > 0.001 && e <= lplus_pre)
                    .collect();
                bulk_eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let sq = if !bulk_eigs.is_empty() {
                    let l_med = bulk_eigs[bulk_eigs.len() / 2];
                    l_med / lambda_med_mp
                } else {
                    1.0
                };
                (sq, raw_eigs)
            }
            EigensolverMode::Subsample { .. } => {
                let (eigs, q_sub) = sub_spectrum.expect("subsample spectrum computed in front end");
                spectrum_q = q_sub;
                let rmt_sub = RmtTheory { q: q_sub };
                let lplus_sub = rmt_sub.lambda_plus();
                let mut bulk: Vec<f64> = eigs
                    .iter()
                    .cloned()
                    .filter(|&e| e > 0.001 && e <= lplus_sub)
                    .collect();
                bulk.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let sq = if bulk.is_empty() {
                    1.0
                } else {
                    bulk[bulk.len() / 2] / rmt_sub.mp_median()
                };
                if v {
                    eprint!("[eigenspectrum] subsample Gram (q'={q_sub:.3}) ... ");
                }
                (sq, eigs)
            }
            EigensolverMode::Fast => {
                // σ² ≈ Tr(S) / min(n−1, p)
                // Tr(S) = sum of diagonal = sum of per-gene variances after centring.
                // Dividing by the matrix rank (min(n-1, p)) gives the mean eigenvalue,
                // which equals σ² for pure MP noise.  Signal outliers inflate this by
                // k×(λ̄_signal − σ²)/p_eff — see EigensolverMode::Fast docs for analysis.
                if v {
                    eprint!("[normalisation] Fast σ² from Tr(S) ... ");
                }
                let t = Instant::now();
                // Tr(S) = Σⱼ S_jj ≈ p·σ² for any aspect ratio q = p/n.
                // For q ≤ 1: all p eigenvalues are non-zero and each ≈ σ².
                // For q > 1: the p-n zero eigenvalues contribute 0 to the trace,
                //   but each of the p diagonal entries is still ≈ σ² because
                //   biwhitening normalises per-gene variance (Σᵢ x²ᵢⱼ/n ≈ 1).
                //   Tr(S) = Σⱼ Σᵢ x²ᵢⱼ/(n-1) ≈ n·p/(n-1)·σ² and dividing by
                //   p gives σ² × n/(n-1) ≈ σ².  Using p (not rank = n-1) is correct.
                let trace: f64 = (0..p).map(|i| s.read(i, i)).sum();
                let sq = trace / p as f64;
                if v {
                    eprint!("done ({:.4}s)  ", t.elapsed().as_secs_f64());
                }
                (sq, vec![])
            }
        };

        if v && (sigma_sq - 1.0).abs() > 1e-3 {
            eprintln!(
                "[normalisation] σ² = {sigma_sq:.4}{}",
                if self.config.eigensolver == EigensolverMode::Fast {
                    "  ⚠️  (fast/approximate)"
                } else {
                    "  (bulk median / MP median)"
                }
            );
        } else if v {
            eprintln!("[normalisation] σ² = {sigma_sq:.4}  (≈1, scale already correct)");
        }
        // P2.B: preserve the pre-σ̂²-rescale eigenvalues alongside the
        // rescaled ones. The rescaled version is what existing consumers see.
        let s_eigenvalues_unrescaled: Vec<f64> = s_eigenvalues.clone();
        let s = Mat::from_fn(p, p, |i, j| s.read(i, j) / sigma_sq);
        let s_eigenvalues: Vec<f64> = s_eigenvalues.iter().map(|&e| e / sigma_sq).collect();

        // --- Stage 5: RMT thresholding + FISTA start ---
        // λ+ = (1 + √q)² is the MP bulk edge (Eq. 3).  Eigenvalues above λ+
        // are signal outliers via the BBP phase transition (Section 2.2).
        //
        // Reference: `BiwhitenedCovarianceEstimator.fit` (spcarmt
        // `scrna/rmt/_covariance.py`) takes `np.linalg.eigh(S)` and counts
        // `eigs > lplus + 0.01` as `n_comps_`; `_adjust_penalty` and
        // `_fista_spca` (`scrna/prc/_spca.py`) start FISTA from the top
        // `n_comps` columns of the same `eigh`, descending, with
        // `lmax = max(s)`. In `Full` mode that spectrum is Stage 4's, so k, the
        // eigenvalues and the start come from it exactly (threshold λ+ itself:
        // the reference's +0.01 is an absolute margin with no scaling argument).
        //
        // Without a full EVD (`Fast`, `Subsample`), subspace iteration on k_max
        // vectors stands in: its Ritz values approximate the top eigenvalues,
        // but near λ+ the spectrum is dense and convergence is slow, so the
        // tail count is approximate and bounded by k_max.
        let k_max = self.config.k_max.min(p).min(n);
        let t = Instant::now();
        let rmt = rmt_pre; // reuse — same q = p/n
        let lambda_plus = rmt.lambda_plus();
        // rq: top candidate eigenvalues, descending; v_cand their vectors;
        // n_above: eigenvalues above λ+ (exact in Full, else among the k_max).
        let (rq, v_cand, n_above): (Vec<f64>, Mat<f64>, usize) = match &eigvecs {
            Some(u) => {
                // s_eigenvalues is ascending (faer); the top m are the last m.
                let n_above = s_eigenvalues.iter().filter(|&&e| e > lambda_plus).count();
                let m = n_above.clamp(1, k_max);
                let rq: Vec<f64> = (0..m).map(|j| s_eigenvalues[p - 1 - j]).collect();
                let v_cand = Mat::from_fn(p, m, |i, j| u.read(i, p - 1 - j));
                (rq, v_cand, n_above)
            }
            None => {
                if v {
                    eprint!("[RMT/subspace] subspace iteration (k_max={k_max}) ... ");
                }
                let v_cand = subspace_iteration(&s, k_max, 100);
                // Ritz values θ_j = v_j^T S v_j, descending after Rayleigh–Ritz
                let sv_cand = mat_mul(&s, &v_cand);
                let rq: Vec<f64> = (0..k_max)
                    .map(|j| (0..p).map(|i| v_cand.read(i, j) * sv_cand.read(i, j)).sum())
                    .collect();
                let n_above = rq.iter().filter(|&&r| r > lambda_plus).count();
                (rq, v_cand, n_above)
            }
        };
        let lmax = rq[0];
        // k = number of eigenvalues above the BBP threshold, capped by k_max; at least 1.
        let k = n_above.clamp(1, k_max);
        let v_init = Mat::from_fn(p, k, |i, j| v_cand.read(i, j));
        // In Full mode the count is exact, so k < n_above is a cap; in the
        // subspace modes a k that fills k_max is a ceiling, not a count.
        let capped = if eigvecs.is_some() {
            n_above > k
        } else {
            k == k_max
        };
        if v {
            eprintln!(
                "λ_max = {lmax:.4}  λ+ = {lambda_plus:.4}  above λ+ = {n_above}  components = {k}{}  ({:.2}s)",
                if capped { " [CAPPED by k_max]" } else { "" },
                t.elapsed().as_secs_f64()
            );
            if capped {
                eprintln!(
                    "[RMT/subspace] WARNING: k_max={k_max} caps the components fitted; {} \
                     Raise FistaConfig::k_max.",
                    if eigvecs.is_some() {
                        format!("{n_above} eigenvalues are above λ+.")
                    } else {
                        format!("the true count is >= {k_max} and NOT determined by this run.")
                    }
                );
            }
        }

        // --- Stage 6: FISTA Sparse PCA (Algorithm 2) ---
        // Step size γ = 0.5/(2·λ_max) = 1/(4·λ_max), matching the Python
        // reference (`t = 0.5/(2*lmax)` in `_fista_spca`).
        //
        // Note: the theoretically optimal step is 1/L = 1/(2·λ_max) since the
        // Lipschitz constant of ∇Tr(W^T S W) is L = 2·λ_max.  The Python
        // implementation uses half this value (a conservative choice that
        // still guarantees convergence).  We match Python for reproducibility.
        let gamma = if lmax > 1e-14 {
            0.5 / (2.0 * lmax)
        } else {
            0.25
        };
        // Eq. 8 target: Σ_j (1 − predicted overlap) over the k retained
        // components, the overlap being the BBP squared cosine between the
        // signal eigenvector and the outlier eigenvector. A component at or
        // below λ+ has no prediction and counts as fully unresolved (1).
        let target_distance: f64 = rq[..k]
            .iter()
            .map(|&r| {
                if r > lambda_plus {
                    1.0 - rmt.predicted_overlap(r).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .sum();
        let run = |lambda: f64, verbose: bool| {
            fista_sparse_pca(
                &s,
                &v_init,
                gamma,
                lambda,
                self.config.max_iterations,
                self.config.tolerance,
                self.config.tol_obj,
                verbose,
                self.config.fista_reuse_products,
            )
        };
        let (lambda, lambda_star) = match (self.config.auto_penalty, self.config.lambda_frac) {
            (Some(scale), _) => {
                let star = select_penalty_eq8(
                    |lam| subspace_distance(&run(lam, false), &v_init),
                    target_distance,
                    lambda_plus,
                    v,
                );
                (scale * star, Some(star))
            }
            (None, Some(frac)) => (frac * lambda_plus, None),
            (None, None) => (self.config.lambda, None),
        };
        if v {
            eprintln!("[FISTA]        λ = {lambda:.4e}  γ = {gamma:.4e}  (γλ = {:.4e})  max_iter = {}  tol = {}  tol_obj = {}",
            gamma * lambda, self.config.max_iterations,
            self.config.tolerance, self.config.tol_obj);
        }
        let t = Instant::now();
        let components = run(lambda, v);
        let subspace_distance = subspace_distance(&components, &v_init);
        let singletons = (0..k)
            .filter(|&j| {
                (0..p)
                    .map(|i| components.read(i, j).abs())
                    .fold(0.0, f64::max)
                    > 0.9
            })
            .count();
        if v {
            eprintln!(
                "[FISTA]        done ({:.2}s)  subspace distance Σ(1−σ²) = {subspace_distance:.3} \
                 (Eq. 8 target {target_distance:.3}){}",
                t.elapsed().as_secs_f64(),
                match lambda_star {
                    Some(star) => format!("  λ* = {star:.4e}"),
                    None => String::new(),
                }
            );
            if singletons > 0 {
                eprintln!(
                    "[FISTA]        WARNING: {singletons}/{k} components collapsed onto a single gene \
                     (max|w| > 0.9): the penalty is too large for them. Use `auto_penalty` (Eq. 8) \
                     or lower λ; see PLAN.md."
                );
            }
        }

        // Eigenvalues (Full) / Ritz values (other modes) of the k signal
        // components: the predicted-overlap table uses them via BBP (Eq. 9).
        let eigenvalues: Vec<f64> = rq[..k].to_vec();

        // KS diagnostic: only available in Full mode when compute_ks = true.
        let ks_distance =
            if self.config.compute_ks && self.config.eigensolver != EigensolverMode::Fast {
                Some(crate::verification::calculate_bulk_ks(
                    &s_eigenvalues,
                    spectrum_q,
                ))
            } else {
                None
            };

        // P2.B: expose biwhitening factors as plain Vec<f64> for ergonomic
        // (de)serialisation by callers.
        let bw_c: Vec<f64> = (0..c.nrows()).map(|i| c.read(i)).collect();
        let bw_d_tmp: Vec<f64> = (0..d.nrows()).map(|j| d.read(j)).collect();
        let (collapsed_cells, max_gene_factor_ratio) = biwhitening_collapse(&bw_c, &bw_d_tmp);
        if v && collapsed_cells > 0 {
            eprintln!(
                "[biwhitening]  WARNING: {collapsed_cells} cells collapsed (c < 1e-3·median), \
                 max d/median = {max_gene_factor_ratio:.1e}: genes detected in too few cells \
                 (≲ n/p = {:.1}) have no finite biwhitening. Keep genes in ≳ {:.0} cells.",
                n as f64 / p as f64,
                (2.0 * n as f64 / p as f64).ceil()
            );
        }
        let bw_d: Vec<f64> = (0..d.nrows()).map(|j| d.read(j)).collect();

        SparsePCAResult {
            components,
            eigenvalues,
            n_above_threshold: n_above,
            s_eigenvalues,
            lambda,
            lambda_star,
            subspace_distance,
            target_distance,
            lambda_plus,
            q: rmt.q,
            sigma_sq,
            ks_distance,
            sk_iters: bw_iters,
            sk_converged: bw_ok,
            sk_residual: bw_res,
            bw_c,
            bw_d,
            s_eigenvalues_unrescaled,
            top_eigenvectors,
            collapsed_cells,
            max_gene_factor_ratio,
        }
    }
}

/// Output of the Sparse PCA pipeline.
pub struct SparsePCAResult {
    /// Sparse loading matrix W (p × k).  Each column is one component;
    /// most entries are zero due to the L1 penalty.
    pub components: Mat<f64>,
    /// Eigenvalues of the σ̂²-rescaled S for the k signal components,
    /// descending: exact in `EigensolverMode::Full`, Ritz values from subspace
    /// iteration otherwise. Used to compute predicted overlaps via the BBP
    /// formula (Eq. 9).
    pub eigenvalues: Vec<f64>,
    /// Eigenvalues above λ+. Exact in `Full` mode (the reference's `n_comps_`,
    /// up to its +0.01 margin), so `eigenvalues.len() < n_above_threshold`
    /// means `FistaConfig::k_max` capped the fit. In `Fast` / `Subsample` it is
    /// the count among the `k_max` Ritz values, hence at most `k_max`.
    pub n_above_threshold: usize,
    /// All p eigenvalues of the biwhitened covariance S (ascending order).
    /// Empty when `EigensolverMode::Fast` is used.
    /// Used for the Kolmogorov-Smirnov test against the MP distribution.
    pub s_eigenvalues: Vec<f64>,
    /// L1 penalty actually used by FISTA (absolute, covariance-eigenvalue units).
    pub lambda: f64,
    /// Eq. 8 root λ* when `FistaConfig::auto_penalty` was set (`lambda` is
    /// then `scale × lambda_star`); `None` for a fixed `lambda` / `lambda_frac`.
    pub lambda_star: Option<f64>,
    /// Achieved distance Σ_j (1 − σ_j²) between the sparse subspace and the
    /// outlier eigenspace (σ_j = singular values of WᵀV). 0 = W spans V.
    pub subspace_distance: f64,
    /// RMT-predicted distance Σ_j (1 − overlap_j) over the k components (Eq. 8
    /// target). A `subspace_distance` far above it means over-penalised;
    /// each single-gene component adds ≈ 1.
    pub target_distance: f64,
    /// MP bulk edge λ+ = (1 + √q)² computed from the *filtered* (n, p).
    pub lambda_plus: f64,
    /// Aspect ratio q = p/n used for RMT, after zero-row/col filtering.
    pub q: f64,
    /// Scale factor σ² applied to the biwhitened covariance.
    ///
    /// In `Full` mode: exact median of bulk eigenvalues / MP median.
    /// In `Fast` mode: Tr(S) / min(n−1, p) — approximate, see `EigensolverMode::Fast`.
    /// Should be close to 1.0 for well-biwhitened data.
    pub sigma_sq: f64,
    /// Kolmogorov-Smirnov distance between the bulk eigenspectrum and the
    /// Marchenko-Pastur CDF.  `None` if `compute_ks = false` or
    /// `EigensolverMode::Fast` was used.
    pub ks_distance: Option<f64>,
    /// Number of Sinkhorn-Knopp iterations executed during biwhitening.
    /// Zero when precomputed factors were supplied (P2.A).
    pub sk_iters: usize,
    /// Whether Sinkhorn-Knopp reached its convergence tolerance.
    /// `true` when precomputed factors were supplied.
    pub sk_converged: bool,
    /// Final 95th-percentile relative change in c at the last SK iteration.
    /// Zero when precomputed factors were supplied.
    pub sk_residual: f64,
    /// **P2.B** — biwhitening cell-scaling factors (length = n after Stage 0 filter).
    /// These are the factors actually used (either from Sinkhorn-Knopp or from
    /// `FistaConfig::precomputed_factors`). Persist them to reuse across runs
    /// (P2.A) or to do downstream analyses (e.g. compute per-cell biwhitened
    /// residuals).
    pub bw_c: Vec<f64>,
    /// **P2.B** — biwhitening gene-scaling factors (length = p after Stage 0 filter).
    pub bw_d: Vec<f64>,
    /// **P2.B** — eigenvalues of the biwhitened sample covariance *before* the
    /// σ̂² rescaling (i.e. `s_eigenvalues × sigma_sq`). Empty in `Fast` mode.
    /// Useful when the consumer wants to apply a custom σ² (e.g. estimated
    /// from a noise-only run) or to verify the σ̂² estimator.
    pub s_eigenvalues_unrescaled: Vec<f64>,
    /// **P2.B** — top eigenvectors of the σ̂²-rescaled covariance, sorted by
    /// eigenvalue descending. `top_eigenvectors[k]` is a column of length p
    /// (after Stage 0 filter). Length = `config.top_eigvec_count` (default 20),
    /// capped to p. Empty in `Fast` mode.
    pub top_eigenvectors: Vec<Vec<f64>>,
    /// Cells whose biwhitening factor collapsed: c_i < 1e-3 · median(c).
    ///
    /// Non-zero means Sinkhorn had no finite solution and drifted c_i → 0,
    /// d_j → ∞, erasing those cells (see [`biwhitening_collapse`]). Healthy
    /// runs have min(c)/median(c) ≈ 0.1, so the threshold is far from normal.
    pub collapsed_cells: usize,
    /// max(d) / median(d). Healthy runs: ~10–100; a collapse shows as ≫ 1e3.
    pub max_gene_factor_ratio: f64,
}

/// Biwhitening collapse diagnostic: (cells with c_i < 1e-3·median(c),
/// max(d)/median(d)).
///
/// A gene detected in k cells (counts ≈ 1) needs (c_i d_j)² ≈ n/k for unit
/// variance, which spends n/(k·p) of each such cell's unit row variance. When
/// that exceeds 1 — roughly k < n/p, or several rare genes in one cell — no
/// finite factors exist; Sinkhorn drifts c_i → 0, d_j → ∞ while X_w's variances,
/// its stopping rule, barely move. The cells are erased from the covariance.
/// Rare cell types carrying rare genes are hit first. Remedy: keep genes
/// detected in ≳ 2–3·n/p cells (a detection filter, not variance-based).
pub fn biwhitening_collapse(c: &[f64], d: &[f64]) -> (usize, f64) {
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let (mc, md) = (median(c), median(d));
    let cells = c.iter().filter(|&&x| x < 1e-3 * mc).count();
    let ratio = d.iter().cloned().fold(0.0, f64::max) / md;
    (cells, ratio)
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
///    (gradient of Tr(Y^T S Y) is 2·S·Y; step size γ = 0.5/(2·λ_max))
/// 2. **Proximal step**: soft-threshold each entry of Z with threshold γλ
///    (prox operator of the L1 penalty: sign(z)·max(|z|−γλ, 0))
/// 3. **Orthonormalise**: modified Gram-Schmidt to enforce W^T W = I_k
/// 4. **Nesterov momentum**: extrapolate between consecutive iterates with
///    coefficient β_t = (t−1)/t_{k+1} using the paper's momentum schedule
///
/// Stops when either the iterate change ‖W_{t+1}−W_t‖_F < `tol` **or** the
/// relative objective change falls below `tol_obj` (OR criterion prevents
/// getting stuck when rotational ambiguity keeps ‖ΔW‖ large).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fista_sparse_pca(
    s: &Mat<f64>,
    v_init: &Mat<f64>,
    gamma: f64,
    lambda: f64,
    max_iter: usize,
    tol: f64,
    tol_obj: f64,
    verbose: bool,
    reuse_products: bool,
) -> Mat<f64> {
    let (p, k) = (v_init.nrows(), v_init.ncols());
    let mut w = v_init.clone();
    let mut y = v_init.clone();
    let mut t = 1.0_f64;

    // S·W for the current W; with `reuse_products` also S·Y (see FistaConfig).
    let mut sw = mat_mul(s, v_init);
    let mut obj_prev = trace_with(v_init, &sw);
    let mut sy_next = sw.clone();

    for iter in 0..max_iter {
        let prev_w = w.clone();
        let s_prev_w = sw;

        // Step 1: gradient ascent  Z = Y + 2γ S Y
        let sy = if reuse_products {
            sy_next
        } else {
            mat_mul(s, &y)
        };
        let mut z = Mat::from_fn(p, k, |i, j| y.read(i, j) + 2.0 * gamma * sy.read(i, j));

        // Step 2: soft-threshold (proximal operator of λ‖·‖₁)
        // threshold = γλ; entries with |z_ij| ≤ γλ are zeroed out.
        for i in 0..p {
            for j in 0..k {
                let v = z.read(i, j);
                let shrunk = v.abs() - lambda * gamma;
                z.as_mut().write(
                    i,
                    j,
                    if shrunk > 0.0 {
                        shrunk * v.signum()
                    } else {
                        0.0
                    },
                );
            }
        }

        // Step 3: orthonormalise columns to enforce W^T W = I_k
        z = orthonormalize(&z);

        // Stopping criteria
        let sz = mat_mul(s, &z);
        let obj_new = trace_with(&z, &sz);
        let dw = frobenius_diff(&z, &prev_w, p, k);
        let dvar = (obj_new - obj_prev).abs();
        let iterate_ok = dw < tol;
        let obj_ok = dvar / (obj_prev.abs() + 1e-10) < tol_obj;

        if verbose && (iter + 1) % 50 == 0 {
            eprintln!(
                "[FISTA]        iter {:>4}  Var={:.6}  ΔVar={:.2e}  ΔW={:.2e}",
                iter + 1,
                obj_new,
                dvar,
                dw
            );
        }

        obj_prev = obj_new;

        if iterate_ok || obj_ok {
            w = z;
            break;
        }

        // Step 4: Nesterov momentum update.
        //
        // The paper (Algorithm 2, arXiv:2509.15429) and the Python reference
        // use a modified FISTA schedule with constants p=1/20, q=1, r=4:
        //
        //   t_{k+1} = (1/20 + √(1 + 4·t_k²)) / 2
        //
        // This differs from the standard FISTA schedule (1 + √(1+4t²))/2
        // only in the leading constant (1/20 vs 1).  The effect is slightly
        // weaker momentum in early iterations: at t=1, β_standard ≈ 0.38
        // while β_paper ≈ 0.31.  Both schedules produce β → 1 asymptotically
        // and preserve the O(1/k²) convergence guarantee.
        let next_t = (1.0 / 20.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let beta = (t - 1.0) / next_t;
        y = Mat::from_fn(p, k, |i, j| {
            z.read(i, j) + beta * (z.read(i, j) - prev_w.read(i, j))
        });
        // S·Y by linearity from S·Z and S·W_prev (only used with reuse_products)
        sy_next = if reuse_products {
            Mat::from_fn(p, k, |i, j| {
                sz.read(i, j) + beta * (sz.read(i, j) - s_prev_w.read(i, j))
            })
        } else {
            Mat::new()
        };
        w = z;
        sw = sz;
        t = next_t;
    }

    w
}

/// Distance Σ_j (1 − σ_j²) between the column spaces of W and V (both p×k with
/// orthonormal columns), σ_j the singular values of WᵀV. Since Σ σ_j² = ‖WᵀV‖_F²
/// this needs no SVD. 0 when W spans V; each column of W orthogonal to V adds 1.
pub fn subspace_distance(w: &Mat<f64>, v: &Mat<f64>) -> f64 {
    let z = w.as_ref().transpose() * v.as_ref();
    let fro2: f64 = (0..z.nrows())
        .flat_map(|i| (0..z.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| z.read(i, j).powi(2))
        .sum();
    w.ncols() as f64 - fro2
}

/// Eq. 8 penalty selection (paper's `_adjust_penalty`): the λ* at which
/// `distance(λ) = target`, where `distance` runs FISTA at λ and measures the
/// subspace distance to the outlier eigenspace. Bisection on log λ, since the
/// distance is monotone in λ up to solver noise: 0 at λ → 0 (dense
/// eigenvectors) and ≈ k once every column has collapsed. Bracket
/// [1e-4, λ+], widened upward if needed; stops at 5 % relative width.
pub fn select_penalty_eq8<F: Fn(f64) -> f64>(
    distance: F,
    target: f64,
    lambda_plus: f64,
    verbose: bool,
) -> f64 {
    let mut evals = 0usize;
    let mut f = |lam: f64| {
        let d = distance(lam) - target;
        evals += 1;
        if verbose {
            eprintln!("[Eq. 8]        eval {evals:>2}  λ = {lam:.4e}  distance − target = {d:+.4}");
        }
        d
    };
    let mut lo = 1e-4_f64;
    let mut hi = lambda_plus.max(1e-3);
    if f(lo) >= 0.0 {
        if verbose {
            eprintln!("[Eq. 8]        distance already at/above target at λ = {lo:.1e}; using it");
        }
        return lo;
    }
    let mut f_hi = f(hi);
    while f_hi < 0.0 && hi < 64.0 * lambda_plus {
        hi *= 2.0;
        f_hi = f(hi);
    }
    if f_hi < 0.0 {
        if verbose {
            eprintln!("[Eq. 8]        target not reached by λ = {hi:.3e}; using it");
        }
        return hi;
    }
    while hi / lo > 1.05 {
        let mid = (lo * hi).sqrt();
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo * hi).sqrt()
}

/// Tr(W^T S W) — total variance explained by loading matrix W, given SW.
fn trace_with(w: &Mat<f64>, sw: &Mat<f64>) -> f64 {
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

/// Eigenvalues (ascending) of the Gram Y Yᵀ/(m−1) of an m×p centred row block,
/// i.e. the non-zero spectrum of the block's covariance, with q' = p/m.
pub(crate) fn gram_spectrum(y: &Mat<f64>) -> (Vec<f64>, f64) {
    let (m, p) = (y.nrows(), y.ncols());
    let inv = 1.0 / (m - 1) as f64;
    let g: Mat<f64> = y.as_ref() * y.as_ref().transpose();
    let g = Mat::from_fn(m, m, |i, j| g.read(i, j) * inv);
    let mut eigs = g.selfadjoint_eigenvalues(Side::Lower);
    eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    (eigs, p as f64 / m as f64)
}

/// Dense matrix product C = A B  (BLAS-backed via faer).
fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a.as_ref() * b.as_ref()
}

/// Sample covariance S = X^T X / (n−1)  (p×p, unbiased estimator, BLAS-backed).
///
/// Uses the 1/(n−1) convention to match the Python reference implementation
/// (both `_fista_spca` and `BiwhitenedCovarianceEstimator`).  Asymptotically
/// the distinction between 1/n and 1/(n−1) vanishes, but for small n the
/// difference shifts all eigenvalues by n/(n−1), which affects the BBP
/// threshold test.
fn sample_covariance(x: &Mat<f64>) -> Mat<f64> {
    let inv = 1.0 / (x.nrows() - 1) as f64;
    let s: Mat<f64> = x.as_ref().transpose() * x.as_ref();
    Mat::from_fn(s.nrows(), s.ncols(), |i, j| s.read(i, j) * inv)
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
    // Deterministic pseudo-random start (xorshift64). The previous start, the
    // first k coordinate vectors, has zero overlap with any signal direction
    // supported outside genes 0..k, so on a block-structured covariance the
    // iteration never finds it (a 3-spike test found 1 spike). Column order
    // and hence the eigenvector ordering do not depend on the start.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut v = Mat::from_fn(p, k, |_, _| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    });
    for _ in 0..n_iter {
        let sv = mat_mul(s, &v);
        v = orthonormalize(&sv);
    }
    // Rayleigh–Ritz: rotate within span(V) to the eigenvectors of VᵀSV, so the
    // columns are the best approximations the subspace holds and their Ritz
    // values come out in descending order (plain iteration leaves the slowly
    // converging tail mixed and unordered).
    let h: Mat<f64> = v.as_ref().transpose() * mat_mul(s, &v).as_ref();
    let evd = SelfAdjointEigendecomposition::new(h.as_ref(), Side::Lower);
    let q = evd.u();
    // faer's eigenvalues are ascending: reverse the columns of Q
    let q_desc = Mat::from_fn(k, k, |i, j| q.read(i, k - 1 - j));
    mat_mul(&v, &q_desc)
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
    fn subspace_distance_is_zero_for_same_span_and_one_per_orthogonal_column() {
        let v = Mat::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        assert!(subspace_distance(&v, &v).abs() < 1e-12);
        // rotate within the span: still 0
        let r = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 0.6,
            (1, 0) => 0.8,
            (0, 1) => -0.8,
            _ => 0.0,
        });
        assert!(subspace_distance(&r, &v).abs() < 1e-12);
        // one column outside the span: distance 1
        let w = Mat::from_fn(4, 2, |i, j| {
            if (i, j) == (0, 0) || (i, j) == (2, 1) {
                1.0
            } else {
                0.0
            }
        });
        assert!((subspace_distance(&w, &v) - 1.0).abs() < 1e-12);
    }

    /// Spiked covariance used by the Eq. 8 tests: three signal directions on
    /// disjoint 60-gene supports with eigenvalues 40, 12, 4 above unit noise.
    fn spiked_three(p: usize) -> (Mat<f64>, Mat<f64>) {
        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 10_000) as f64 / 10_000.0
        };
        let supports = [(0usize, 60usize), (70, 130), (140, 200)];
        let spikes = [40.0, 12.0, 4.0];
        let mut u = Mat::<f64>::zeros(p, 3);
        for (j, &(a, b)) in supports.iter().enumerate() {
            let mut col: Vec<f64> = (a..b).map(|_| 0.5 + next()).collect();
            let nrm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            col.iter_mut().for_each(|x| *x /= nrm);
            for (t, i) in (a..b).enumerate() {
                u.as_mut().write(i, j, col[t]);
            }
        }
        let s = Mat::from_fn(p, p, |i, j| {
            let sig: f64 = (0..3)
                .map(|c| spikes[c] * u.read(i, c) * u.read(j, c))
                .sum();
            sig + if i == j { 1.0 } else { 0.0 }
        });
        (s, u)
    }

    fn max_abs_col(w: &Mat<f64>, j: usize) -> f64 {
        (0..w.nrows())
            .map(|i| w.read(i, j).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn eq8_root_hits_target_and_sits_on_the_collapse_cliff() {
        let p = 200;
        let (s, _) = spiked_three(p);
        let gamma = 0.5 / (2.0 * 41.0);
        let v_init = subspace_iteration(&s, 3, 200);
        let target = 0.3;
        let fista =
            |lam: f64| fista_sparse_pca(&s, &v_init, gamma, lam, 2000, 1e-7, 1e-9, false, false);
        let distance = |lam: f64| subspace_distance(&fista(lam), &v_init);
        let star = select_penalty_eq8(distance, target, 2.0, false);
        let achieved = distance(star);
        assert!(
            (achieved - target).abs() < 0.08,
            "achieved {achieved} vs target {target} at λ*={star}"
        );
        // The distance is a cliff in λ: ~0 (dense eigenvectors) below the
        // root, collapsing columns above it. That is why the reference fits
        // at 0.6 λ*, not at λ*. Below the cliff every column stays spread;
        // well above it the weakest spike is a single gene.
        let w_scaled = fista(0.6 * star);
        assert!(
            (0..3).all(|j| max_abs_col(&w_scaled, j) < 0.5),
            "0.6 λ* must keep all columns spread"
        );
        assert!(subspace_distance(&w_scaled, &v_init) < target);
        let w_big = fista(3.0 * star);
        assert!(
            max_abs_col(&w_big, 2) > 0.9,
            "expected the weakest spike to collapse at 3 λ*"
        );
    }

    #[test]
    fn auto_penalty_through_fit_covariance_reports_and_avoids_singletons() {
        let (p, n) = (200usize, 1000usize);
        let (s, u) = spiked_three(p);
        let fixed = SparsePCA::new(FistaConfig {
            lambda_frac: Some(1.0),
            k_max: 8,
            compute_ks: false,
            ..FistaConfig::default()
        })
        .fit_covariance(
            s.clone(),
            n,
            Col::from_fn(n, |_| 1.0),
            Col::from_fn(p, |_| 1.0),
            None,
        );
        let auto = SparsePCA::new(FistaConfig {
            auto_penalty: Some(0.6),
            k_max: 8,
            compute_ks: false,
            ..FistaConfig::default()
        })
        .fit_covariance(
            s,
            n,
            Col::from_fn(n, |_| 1.0),
            Col::from_fn(p, |_| 1.0),
            None,
        );
        assert_eq!(
            auto.eigenvalues.len(),
            3,
            "RMT should retain the three spikes"
        );
        assert!(fixed.lambda_star.is_none() && auto.lambda_star.is_some());
        let star = auto.lambda_star.unwrap();
        assert!((auto.lambda - 0.6 * star).abs() < 1e-12);
        assert!(
            auto.subspace_distance <= auto.target_distance + 0.05,
            "{} vs {}",
            auto.subspace_distance,
            auto.target_distance
        );
        // fixed λ = λ+ collapses at least the weakest spike; Eq. 8 collapses none
        let k = auto.components.ncols();
        let singles = |w: &Mat<f64>| (0..k).filter(|&j| max_abs_col(w, j) > 0.9).count();
        assert!(
            singles(&fixed.components) >= 1,
            "fixed λ+ did not collapse anything; test is too weak"
        );
        assert_eq!(
            singles(&auto.components),
            0,
            "Eq. 8 run has single-gene components"
        );
        // and each Eq. 8 component lives on the support of its spike
        for j in 0..k {
            let on: f64 = (0..p)
                .map(|i| auto.components.read(i, j).abs() * (u.read(i, j) > 0.0) as u8 as f64)
                .sum();
            let all: f64 = (0..p).map(|i| auto.components.read(i, j).abs()).sum();
            assert!(
                on / all > 0.95,
                "component {j} leaks off its support: {on}/{all}"
            );
        }
    }

    #[test]
    fn subspace_iteration_finds_spikes_on_disjoint_supports() {
        let (s, u) = spiked_three(200);
        let v = subspace_iteration(&s, 3, 300);
        let sv = &s * &v;
        for j in 0..3 {
            let rq: f64 = (0..200).map(|i| v.read(i, j) * sv.read(i, j)).sum();
            let cos: f64 = (0..200)
                .map(|i| v.read(i, j) * u.read(i, j))
                .sum::<f64>()
                .abs();
            assert!((rq - [41.0, 13.0, 5.0][j]).abs() < 1e-6, "rq[{j}] = {rq}");
            assert!(cos > 0.999, "column {j} is not the spike: |cos| = {cos}");
        }
    }

    /// Cross-validation of the Eq. 8 root against the Python reference
    /// (`scripts/generate_eq8_vector.py` ports `_adjust_penalty`'s root-find
    /// and `_fista_spca` onto `test_vectors/spiked_model.json`).
    #[test]
    fn eq8_root_matches_python_reference() {
        #[derive(serde::Deserialize)]
        struct Vec8 {
            k: usize,
            target: f64,
            gamma_star: f64,
            distance_at_star: f64,
        }
        #[derive(serde::Deserialize)]
        struct Spiked {
            #[serde(rename = "X")]
            x: Vec<Vec<f64>>,
        }
        let v8: Vec8 = serde_json::from_str(
            &std::fs::read_to_string("test_vectors/eq8_penalty.json").unwrap(),
        )
        .unwrap();
        let sp: Spiked = serde_json::from_str(
            &std::fs::read_to_string("test_vectors/spiked_model.json").unwrap(),
        )
        .unwrap();
        let (n, p) = (sp.x.len(), sp.x[0].len());
        let x = Mat::from_fn(n, p, |i, j| sp.x[i][j]);
        let s = sample_covariance(&x);
        let v_init = subspace_iteration(&s, v8.k, 500);
        let sv = mat_mul(&s, &v_init);
        let lmax = (0..v8.k)
            .map(|j| {
                (0..p)
                    .map(|i| v_init.read(i, j) * sv.read(i, j))
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let gamma = 0.5 / (2.0 * lmax);
        // reference stopping rule: ‖ΔW‖_F < 1e-3 only (tol_obj = 0 disables the objective test)
        let distance = |lam: f64| {
            subspace_distance(
                &fista_sparse_pca(&s, &v_init, gamma, lam, 10_000, 1e-3, 0.0, false, false),
                &v_init,
            )
        };
        let star = select_penalty_eq8(distance, v8.target, 2.25, false);
        let rel = (star - v8.gamma_star).abs() / v8.gamma_star;
        assert!(
            rel < 0.15,
            "crate λ* = {star}, reference γ* = {} ({:.0}% off)",
            v8.gamma_star,
            100.0 * rel
        );
        let d = distance(star);
        assert!(
            (d - v8.target).abs() < 0.05,
            "distance at λ* = {d}, target {}, reference achieved {}",
            v8.target,
            v8.distance_at_star
        );
    }

    /// Direct cross-validation of the FISTA iteration against the Python
    /// reference (`_fista_spca`, ported in `scripts/generate_eq8_vector.py`):
    /// same S, same exact top-k eigenvectors V, same γ = 0.5/(2λ_max), same
    /// stopping rule (‖ΔW‖_F < 1e-10, objective test off), so both reach the
    /// same fixed point and W agrees to round-off (observed 4e-16), support included.
    #[test]
    fn fista_matches_python_reference_at_fixed_penalties() {
        #[derive(serde::Deserialize)]
        struct Case {
            penalty: f64,
            tol: f64,
            #[serde(rename = "W")]
            w: Vec<Vec<f64>>,
            nnz: usize,
        }
        #[derive(serde::Deserialize)]
        struct Vec8 {
            k: usize,
            #[serde(rename = "V")]
            v: Vec<Vec<f64>>,
            lmax: f64,
            fista: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Spiked {
            #[serde(rename = "X")]
            x: Vec<Vec<f64>>,
        }
        let v8: Vec8 = serde_json::from_str(
            &std::fs::read_to_string("test_vectors/eq8_penalty.json").unwrap(),
        )
        .unwrap();
        let sp: Spiked = serde_json::from_str(
            &std::fs::read_to_string("test_vectors/spiked_model.json").unwrap(),
        )
        .unwrap();
        let (n, p, k) = (sp.x.len(), sp.x[0].len(), v8.k);
        let s = sample_covariance(&Mat::from_fn(n, p, |i, j| sp.x[i][j]));
        let v = Mat::from_fn(p, k, |i, j| v8.v[i][j]);
        let gamma = 0.5 / (2.0 * v8.lmax);
        assert!(!v8.fista.is_empty());
        for case in &v8.fista {
            for reuse in [false, true] {
                let w = fista_sparse_pca(
                    &s,
                    &v,
                    gamma,
                    case.penalty,
                    50_000,
                    case.tol,
                    0.0,
                    false,
                    reuse,
                );
                let mut max_err = 0.0_f64;
                let mut nnz = 0usize;
                for j in 0..k {
                    // FISTA from the same V is deterministic, but align signs anyway
                    let dot: f64 = (0..p).map(|i| w.read(i, j) * case.w[i][j]).sum();
                    let sgn = if dot < 0.0 { -1.0 } else { 1.0 };
                    for i in 0..p {
                        let got = sgn * w.read(i, j);
                        max_err = max_err.max((got - case.w[i][j]).abs());
                        nnz += (got.abs() > 1e-12) as usize;
                    }
                }
                assert!(
                    max_err < 1e-12,
                    "penalty {}: max |W_rust − W_python| = {max_err:.2e} (reuse={reuse})",
                    case.penalty
                );
                assert_eq!(nnz, case.nnz, "penalty {}: support differs", case.penalty);
                eprintln!(
                    "penalty {:.4}: max err {max_err:.1e}, nnz {nnz} (reuse={reuse})",
                    case.penalty
                );
            }
        }
    }

    #[test]
    fn fista_recovers_top_eigenvector() {
        // S = 3·v₀v₀^T + 0.01·I,  v₀ = [1,1,1,1]/2
        // λ_max ≈ 3·‖v₀‖²·4 = 3.01; FISTA with λ=0 should recover v₀.
        let p = 4;
        let v0 = [0.5, 0.5, 0.5, 0.5];
        let mut s = Mat::from_fn(p, p, |i, j| {
            3.0 * v0[i] * v0[j] + if i == j { 0.01 } else { 0.0 }
        });
        let _ = &mut s;

        let v_init = Mat::from_fn(p, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let lmax = 3.01_f64;
        let gamma = 1.0 / (2.0 * lmax);

        for reuse in [false, true] {
            let w = fista_sparse_pca(&s, &v_init, gamma, 0.0, 500, 1e-8, 1e-8, false, reuse);
            let dot: f64 = (0..p).map(|i| w.read(i, 0) * v0[i]).sum::<f64>();
            assert!(dot.abs() > 0.99, "alignment = {dot} (reuse={reuse})");
        }
    }
}
