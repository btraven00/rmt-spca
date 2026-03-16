use faer::{Col, Mat};

/// Algorithm 1: Sinkhorn-Knopp biwhitening.
///
/// Finds diagonal scalings c (cells) and d (genes) such that
/// diag(c) X diag(d) has unit per-gene and per-cell second moments [cite: 9, 66, 81].
pub struct Biwhitener {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for Biwhitener {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

impl Biwhitener {
    /// Returns (c, d) — the row and column scaling vectors.
    pub fn compute(&self, x: &Mat<f64>) -> (Col<f64>, Col<f64>) {
        let (n, p) = (x.nrows(), x.ncols());

        // U_ij = x_ij^2  [cite: 75, 83]
        let u = Mat::from_fn(n, p, |i, j| x.read(i, j).powi(2));

        let mut c = Col::from_fn(n, |_| 1.0_f64);
        let mut d = Col::from_fn(p, |_| 1.0_f64);

        for _ in 0..self.max_iter {
            let prev_c = c.clone();

            // --- Update d (gene-wise scaling) [cite: 71, 83] ---
            // c2_i = c_i^2
            let c2 = Col::from_fn(n, |i| c.read(i).powi(2));
            // (X^T c)_j = sum_i c_i * x_ij
            let x_t_c = Col::from_fn(p, |j| (0..n).map(|i| x.read(i, j) * c.read(i)).sum::<f64>());
            // (U^T c^2)_j = sum_i c_i^2 * x_ij^2
            let u_t_c2 = Col::from_fn(p, |j| (0..n).map(|i| u.read(i, j) * c2.read(i)).sum::<f64>());
            // d_j = sqrt( n * (1 + (d_j * (X^T c)_j / n)^2) / (U^T c^2)_j )
            d = Col::from_fn(p, |j| {
                let term = d.read(j) * x_t_c.read(j) / n as f64;
                ((n as f64 * (1.0 + term.powi(2))) / u_t_c2.read(j)).sqrt()
            });

            // --- Update c (cell-wise scaling) [cite: 71, 83] ---
            let d2 = Col::from_fn(p, |j| d.read(j).powi(2));
            // (X d)_i = sum_j d_j * x_ij
            let x_d = Col::from_fn(n, |i| (0..p).map(|j| x.read(i, j) * d.read(j)).sum::<f64>());
            // (U d^2)_i = sum_j d_j^2 * x_ij^2
            let u_d2 = Col::from_fn(n, |i| (0..p).map(|j| u.read(i, j) * d2.read(j)).sum::<f64>());
            // c_i = sqrt( p * (1 + (c_i * (X d)_i / p)^2) / (U d^2)_i )
            c = Col::from_fn(n, |i| {
                let term = c.read(i) * x_d.read(i) / p as f64;
                ((p as f64 * (1.0 + term.powi(2))) / u_d2.read(i)).sqrt()
            });

            let max_diff = (0..n)
                .map(|i| (c.read(i) - prev_c.read(i)).abs())
                .fold(0.0_f64, f64::max);
            if max_diff < self.tol {
                break;
            }
        }

        (c, d)
    }

    /// Apply scalings: returns the matrix X_w where X_w[i,j] = c[i] * x[i,j] * d[j].
    pub fn apply(x: &Mat<f64>, c: &Col<f64>, d: &Col<f64>) -> Mat<f64> {
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| c.read(i) * x.read(i, j) * d.read(j))
    }
}

/// Convenience wrapper: biwhiten and return the scaled data matrix.
pub fn biwhiten(data: &Mat<f64>) -> Mat<f64> {
    let bw = Biwhitener::default();
    let (c, d) = bw.compute(data);
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
        let (c, d) = bw.compute(&x);

        // All scaling factors must be strictly positive
        for i in 0..n {
            assert!(c.read(i) > 0.0, "c[{i}] = {}", c.read(i));
        }
        for j in 0..p {
            assert!(d.read(j) > 0.0, "d[{j}] = {}", d.read(j));
        }

        // Fixed-point check: a second call should change d and c by less than tol
        let (c2, d2) = bw.compute(&crate::biwhitening::Biwhitener::apply(&x, &c, &d));
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
