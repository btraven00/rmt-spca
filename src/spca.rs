use faer::{Col, Mat};

use crate::biwhitening::Biwhitener;
use crate::rmt::RmtTheory;

pub struct FistaConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    /// L1 sparsity penalty λ [cite: 166, 613]
    pub lambda: f64,
}

impl Default for FistaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            lambda: 0.1,
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

    /// Full pipeline: biwhiten → covariance → RMT thresholding → FISTA.
    pub fn fit(&self, data: &Mat<f64>) -> SparsePCAResult {
        let (n, p) = (data.nrows(), data.ncols());

        // Biwhiten
        let bw = Biwhitener::default();
        let (c, d) = bw.compute(data);
        let xw = Biwhitener::apply(data, &c, &d);

        // Sample covariance S = X_w^T X_w / n  (p × p)
        let s = sample_covariance(&xw);

        // Number of signal components: count eigenvalues above RMT bulk edge
        let rmt = RmtTheory { q: p as f64 / n as f64 };
        let lambda_plus = rmt.lambda_plus();
        let k = estimate_n_components(&s, lambda_plus).max(1);

        // Initial eigenvectors via subspace iteration
        let v_init = subspace_iteration(&s, k, 100);

        // Step size γ = 1 / (2 λ_max(S))  [cite: AGENTS.md]
        let lmax = power_iteration_max_eigenvalue(&s, 50);
        let gamma = if lmax > 1e-14 { 1.0 / (2.0 * lmax) } else { 1.0 };

        let components = fista_sparse_pca(
            &s,
            &v_init,
            gamma,
            self.config.lambda,
            self.config.max_iterations,
            self.config.tolerance,
        );

        SparsePCAResult { components }
    }
}

pub struct SparsePCAResult {
    pub components: Mat<f64>,
}

// ---------------------------------------------------------------------------
// Algorithm 2: FISTA Sparse PCA [cite: 166, 613]
// ---------------------------------------------------------------------------

/// FISTA with soft-thresholding and orthogonalization.
///
/// Maximises Tr(W^T S W) subject to W^T W = I_k and an L1 penalty on W.
pub fn fista_sparse_pca(
    s: &Mat<f64>,
    v_init: &Mat<f64>,
    gamma: f64,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Mat<f64> {
    let (p, k) = (v_init.nrows(), v_init.ncols());
    let mut w = v_init.clone();
    let mut y = v_init.clone();
    let mut t = 1.0_f64;

    for _ in 0..max_iter {
        let prev_w = w.clone();

        // Gradient ascent step: Z = Y + 2γ S Y  [cite: gradient of Tr(Y^T S Y)]
        let sy = mat_mul(s, &y);
        let mut z = Mat::from_fn(p, k, |i, j| y.read(i, j) + 2.0 * gamma * sy.read(i, j));

        // Proximal operator: column-wise soft-threshold  [cite: 258, 613]
        for i in 0..p {
            for j in 0..k {
                let v = z.read(i, j);
                let shrunk = v.abs() - lambda * gamma;
                z.as_mut().write(i, j, if shrunk > 0.0 { shrunk * v.signum() } else { 0.0 });
            }
        }

        // Orthogonalise via modified Gram-Schmidt (Löwdin-style polar factor)  [cite: 258, 613]
        z = orthonormalize(&z);

        // Convergence on ‖W_new − W_old‖_F
        let diff = frobenius_diff(&z, &prev_w, p, k);
        if diff < tol {
            w = z;
            break;
        }

        // FISTA momentum update
        let next_t = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let beta = (t - 1.0) / next_t;
        y = Mat::from_fn(p, k, |i, j| z.read(i, j) + beta * (z.read(i, j) - prev_w.read(i, j)));
        w = z;
        t = next_t;
    }

    w
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// Dense matrix–matrix product C = A B.
fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let (m, inner) = (a.nrows(), a.ncols());
    let n = b.ncols();
    Mat::from_fn(m, n, |i, j| (0..inner).map(|l| a.read(i, l) * b.read(l, j)).sum::<f64>())
}

/// Matrix–column product y = A x.
fn mat_col_mul(a: &Mat<f64>, x: &Col<f64>) -> Col<f64> {
    let (m, inner) = (a.nrows(), a.ncols());
    Col::from_fn(m, |i| (0..inner).map(|l| a.read(i, l) * x.read(l)).sum::<f64>())
}

/// Top eigenvector of symmetric S via power iteration.
fn top_eigenvector(s: &Mat<f64>, n_iter: usize) -> Col<f64> {
    let p = s.nrows();
    let mut v = Col::from_fn(p, |i| if i == 0 { 1.0 } else { 0.0 });
    for _ in 0..n_iter {
        let sv = mat_col_mul(s, &v);
        let norm: f64 = (0..p).map(|i| sv.read(i).powi(2)).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }
        v = Col::from_fn(p, |i| sv.read(i) / norm);
    }
    v
}

/// Largest eigenvalue of symmetric S via power iteration + Rayleigh quotient.
fn power_iteration_max_eigenvalue(s: &Mat<f64>, n_iter: usize) -> f64 {
    let p = s.nrows();
    let v = top_eigenvector(s, n_iter);
    let sv = mat_col_mul(s, &v);
    (0..p).map(|i| v.read(i) * sv.read(i)).sum()
}

/// Sample covariance S = X^T X / n  (p × p).
fn sample_covariance(x: &Mat<f64>) -> Mat<f64> {
    let (n, p) = (x.nrows(), x.ncols());
    let inv_n = 1.0 / n as f64;
    Mat::from_fn(p, p, |i, j| {
        (0..n).map(|k| x.read(k, i) * x.read(k, j)).sum::<f64>() * inv_n
    })
}

/// Modified Gram-Schmidt orthonormalisation of the columns of A.
fn orthonormalize(a: &Mat<f64>) -> Mat<f64> {
    let (p, k) = (a.nrows(), a.ncols());
    let mut q = a.clone();
    for j in 0..k {
        // Subtract projections onto already-orthonormalised columns
        for jj in 0..j {
            let dot: f64 = (0..p).map(|i| q.read(i, jj) * q.read(i, j)).sum();
            for i in 0..p {
                let v = q.read(i, j) - dot * q.read(i, jj);
                q.as_mut().write(i, j, v);
            }
        }
        // Normalise
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

/// Subspace iteration: returns top-k eigenvectors of symmetric S.
fn subspace_iteration(s: &Mat<f64>, k: usize, n_iter: usize) -> Mat<f64> {
    let p = s.nrows();
    // Initialise with leading columns of the identity
    let mut v = Mat::from_fn(p, k, |i, j| if i == j { 1.0 } else { 0.0 });
    for _ in 0..n_iter {
        let sv = mat_mul(s, &v);
        v = orthonormalize(&sv);
    }
    v
}

/// Count eigenvalues of S exceeding lambda_plus using deflation.
fn estimate_n_components(s: &Mat<f64>, lambda_plus: f64) -> usize {
    let p = s.nrows();
    let k_max = 20_usize.min(p);
    let mut s_work = s.clone();
    let mut k = 0;

    for _ in 0..k_max {
        let lmax = power_iteration_max_eigenvalue(&s_work, 60);
        if lmax <= lambda_plus {
            break;
        }
        k += 1;
        // Deflate: S ← S − λ_max * v v^T
        let ev = top_eigenvector(&s_work, 60);
        s_work = Mat::from_fn(p, p, |i, j| s_work.read(i, j) - lmax * ev.read(i) * ev.read(j));
    }
    k
}

/// Frobenius distance ‖A − B‖_F.
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
        // q^T q should be close to I_2
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
        // Build S = v v^T + 0.01 I where v = [1,1,1,1]/2
        let p = 4;
        let v0 = vec![0.5, 0.5, 0.5, 0.5];
        let mut s = Mat::from_fn(p, p, |i, j| 3.0 * v0[i] * v0[j] + if i == j { 0.01 } else { 0.0 });
        // lambda_max ~ 3 * 0.25 * 4 + 0.01 = 3.01
        let _ = &mut s; // suppress warning

        let v_init = Mat::from_fn(p, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let lmax = 3.01_f64;
        let gamma = 1.0 / (2.0 * lmax);

        let w = fista_sparse_pca(&s, &v_init, gamma, 0.0, 500, 1e-8);

        // The leading component should align with v0
        let dot: f64 = (0..p).map(|i| w.read(i, 0) * v0[i]).sum::<f64>();
        assert!(dot.abs() > 0.99, "alignment = {dot}");
    }
}
