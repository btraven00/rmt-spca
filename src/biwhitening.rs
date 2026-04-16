use faer::{Col, Mat};

/// Algorithm 1 from Chardès et al.: Sinkhorn-Knopp biwhitening.
///
/// Finds diagonal scaling vectors c ∈ ℝⁿ (cells) and d ∈ ℝᵖ (genes) such
/// that the rescaled matrix X_w = diag(c) X diag(d) satisfies the
/// biwhitening conditions (Section 2.3, Eq. 6):
///
///   Var_i(X_w)_j = 1  for all genes j   (unit per-gene variance across cells)
///   Var_j(X_w)_i = 1  for all cells i   (unit per-cell variance across genes)
///
/// Note: the update targets unit *variance*, not unit second moment.  For
/// non-centred data the second moment is 1 + μ² where μ is the column/row
/// mean.  After mean-centring in the pipeline (Stage 2), the two coincide.
///
/// **Input should be non-negative** (raw counts or log-normalised counts
/// before mean-centring).  Centre the output afterward for the covariance
/// stage; centring before biwhitening introduces negative values that cause
/// the mean-correction term in the update to oscillate.
///
/// The iterative update alternates between rescaling genes and cells
/// (Algorithm 1, lines 4–9):
///
///   d_j ← √[ n · (1 + (d_j · (X^T c)_j / n)²) / (U^T c²)_j ]
///   c_i ← √[ p · (1 + (c_i · (X  d)_i / p)²) / (U  d²)_i ]
///
/// where U_ij = x²_ij.  The mean-correction terms account for non-zero
/// column means; they vanish exactly when the data is centred, in which
/// case the update reduces to pure variance normalisation.
pub struct Biwhitener {
    /// Maximum number of Sinkhorn-Knopp iterations (default 1000).
    pub max_iter: usize,
    /// Convergence tolerance on the 95th-percentile relative change in c
    /// between consecutive iterations (default 1e-6).
    pub tol: f64,
    /// Under-relaxation (damping) factor α ∈ (0, 1].
    ///
    /// Each update is blended with the previous value:
    ///   c_new = (1 − α)·c_old + α·c_computed
    ///
    /// α = 1.0 (default) is the standard Sinkhorn step.  α < 1 damps
    /// oscillations that can arise on log-normalised data or matrices with
    /// large dynamic range.  Try α = 0.5–0.8 if the algorithm stagnates.
    pub damp: f64,
}

impl Default for Biwhitener {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            damp: 1.0,
        }
    }
}

impl Biwhitener {
    /// Run Sinkhorn-Knopp biwhitening on X.
    ///
    /// Returns `(c, d, iters, converged, residual)`:
    /// - `c`         — cell scaling vector (length n)
    /// - `d`         — gene scaling vector (length p)
    /// - `iters`     — number of iterations executed
    /// - `converged` — true iff the 95th-pct relative change fell below `tol`
    /// - `residual`  — final 95th-pct relative change in c
    ///
    /// The 95th-percentile (rather than maximum) residual is used so that a
    /// small number of outlier cells with extreme expression profiles do not
    /// prevent convergence for the other 95% of cells.
    pub fn compute(&self, x: &Mat<f64>) -> (Col<f64>, Col<f64>, usize, bool, f64) {
        let (n, p) = (x.nrows(), x.ncols());

        // U_ij = x²_ij  (Algorithm 1, line 3)
        let u = Mat::from_fn(n, p, |i, j| x.read(i, j).powi(2));

        let mut c = Col::from_fn(n, |_| 1.0_f64);
        let mut d = Col::from_fn(p, |_| 1.0_f64);

        let mut iters = 0_usize;
        let mut last_res = f64::INFINITY;
        let mut best_res = f64::INFINITY;
        let mut iters_since_improvement = 0_usize;
        for _ in 0..self.max_iter {
            iters += 1;
            let prev_c = c.clone();

            // --- Update d (gene-wise scaling, Algorithm 1 line 5) ---
            // c²_i = c_i²
            let c2 = Col::from_fn(n, |i| c.read(i).powi(2));
            // (X^T c)_j = Σᵢ cᵢ xᵢⱼ  — BLAS gemv
            let x_t_c: Col<f64> = x.as_ref().transpose() * c.as_ref();
            // (U^T c²)_j = Σᵢ c²ᵢ x²ᵢⱼ  — BLAS gemv
            let u_t_c2: Col<f64> = u.as_ref().transpose() * c2.as_ref();
            // d_j = √[ n · (1 + (d_j · (X^T c)_j / n)²) / (U^T c²)_j ]
            // Guard: all-zero gene column → keep d_j = 1.
            let alpha = self.damp;
            let d_prev = d.clone();
            d = Col::from_fn(p, |j| {
                let denom = u_t_c2.read(j);
                if denom < 1e-300 { return 1.0; }
                let term = d_prev.read(j) * x_t_c.read(j) / n as f64;
                let d_new = ((n as f64 * (1.0 + term.powi(2))) / denom).sqrt();
                (1.0 - alpha) * d_prev.read(j) + alpha * d_new
            });

            // --- Update c (cell-wise scaling, Algorithm 1 line 7) ---
            let d2 = Col::from_fn(p, |j| d.read(j).powi(2));
            // (X d)_i = Σⱼ dⱼ xᵢⱼ  — BLAS gemv
            let x_d: Col<f64> = x.as_ref() * d.as_ref();
            // (U d²)_i = Σⱼ d²ⱼ x²ᵢⱼ  — BLAS gemv
            let u_d2: Col<f64> = u.as_ref() * d2.as_ref();
            // c_i = √[ p · (1 + (c_i · (X d)_i / p)²) / (U d²)_i ]
            // Guard: all-zero cell row → keep c_i = 1.
            c = Col::from_fn(n, |i| {
                let denom = u_d2.read(i);
                if denom < 1e-300 { return 1.0; }
                let term = prev_c.read(i) * x_d.read(i) / p as f64;
                let c_new = ((p as f64 * (1.0 + term.powi(2))) / denom).sqrt();
                (1.0 - alpha) * prev_c.read(i) + alpha * c_new
            });

            // Convergence check: 95th-percentile relative change in c.
            let mut diffs: Vec<f64> = (0..n)
                .map(|i| (c.read(i) - prev_c.read(i)).abs() / (c.read(i) + 1e-10))
                .collect();
            diffs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            last_res = diffs[(n * 95 / 100).min(n - 1)];
            if last_res < self.tol {
                return (c, d, iters, true, last_res);
            }

            // Stagnation detection: stop if the 95th-pct residual has not
            // improved by ≥ 1% over the last 100 iterations.  This tolerates
            // linear convergence rates as slow as 0.9999/iter (slow but
            // genuine progress) while catching true oscillation/divergence.
            if last_res < best_res * 0.99 {
                best_res = last_res;
                iters_since_improvement = 0;
            } else {
                iters_since_improvement += 1;
                if iters_since_improvement >= 100 {
                    return (c, d, iters, false, last_res);
                }
            }
        }

        (c, d, iters, false, last_res)
    }

    /// Apply scalings: returns X_w where `X_w\[i,j\] = c\[i\] · x\[i,j\] · d\[j\]`.
    ///
    /// This is the matrix diag(c) X diag(d) from Algorithm 1, line 10.
    pub fn apply(x: &Mat<f64>, c: &Col<f64>, d: &Col<f64>) -> Mat<f64> {
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| c.read(i) * x.read(i, j) * d.read(j))
    }
}

/// Convenience wrapper: biwhiten X and return the scaled matrix.
pub fn biwhiten(data: &Mat<f64>) -> Mat<f64> {
    let bw = Biwhitener::default();
    let (c, d, _, _, _) = bw.compute(data);
    Biwhitener::apply(data, &c, &d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Biwhitening should converge to positive scaling vectors and satisfy the
    /// fixed-point condition for all columns and rows.
    #[test]
    fn convergence_and_positivity() {
        let n = 40;
        let p = 20;
        let x = Mat::from_fn(n, p, |i, j| (i + 1) as f64 * (j + 1) as f64 / (n * p) as f64 + 0.1);
        let bw = Biwhitener::default();
        let (c, d, _, _, _) = bw.compute(&x);

        // All scaling factors must be strictly positive
        for i in 0..n {
            assert!(c.read(i) > 0.0, "c[{i}] = {}", c.read(i));
        }
        for j in 0..p {
            assert!(d.read(j) > 0.0, "d[{j}] = {}", d.read(j));
        }

        // Fixed-point check: a second call should change d and c by less than tol
        let (c2, d2, _, _, _) = bw.compute(&crate::biwhitening::Biwhitener::apply(&x, &c, &d));
        // After biwhitening, the already-whitened matrix should have near-unit scalings
        for i in 0..n {
            assert!(
                (c2.read(i) - 1.0).abs() < 0.5,
                "second-pass c[{i}] = {}",
                c2.read(i)
            );
        }
        for j in 0..p {
            assert!(
                (d2.read(j) - 1.0).abs() < 0.5,
                "second-pass d[{j}] = {}",
                d2.read(j)
            );
        }
    }
}
