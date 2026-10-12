//! Streaming pieces: fit the pipeline from row blocks without holding X.
//!
//! The n-dependent work is a sum over cells, and everything after the p×p
//! covariance depends only on p. So a driver can:
//!
//! 1. fit gene factors `d` by Sinkhorn on a cell subsample
//!    ([`crate::biwhitening::Biwhitener::compute_csr`]);
//! 2. reset factors of genes seen in too few subsample cells
//!    ([`reset_rare_genes`]), then refine `d` on all cells with R ≥ 1
//!    streaming passes of [`cell_factors`] + [`GeneMoments`] (warm-started
//!    full-data Sinkhorn);
//! 3. stream all cells once more: [`cell_factors`] gives each cell's `c`
//!    in closed form, and [`CovAccumulator`] sums YᵀY and the column sums;
//! 4. run [`crate::spca::SparsePCA::fit_covariance`] on the finished S, with
//!    σ²/KS from the subsample spectrum ([`subsample_spectrum`]) or a full EVD.
//!
//! Memory is O(p² + one block); n is unbounded.

use faer::{Col, Mat};
use rayon::prelude::*;

use crate::sparse::CsrMatrix;

/// Cell factors for fixed gene factors `d`: the exact Sinkhorn fixed point
/// per row. From c_i = √[p(1 + (c_i m_i / p)²) / u_i] with m = X d, u = (X∘X) d²:
///
///   c_i² (u_i − m_i²/p) = p   ⇒   c_i = √(p / (u_i − m_i²/p))
///
/// u_i/p − (m_i/p)² is the variance of the d-scaled row, so the root exists for
/// any non-constant row; constant (e.g. empty) rows get c_i = 1, as in Sinkhorn.
pub fn cell_factors(x: &CsrMatrix, d: &Col<f64>) -> Col<f64> {
    let p = x.ncols() as f64;
    let m = x.mul_vec(d, false);
    let d2 = Col::from_fn(d.nrows(), |j| d.read(j).powi(2));
    let u = x.mul_vec(&d2, true);
    Col::from_fn(x.nrows(), |i| {
        let denom = u.read(i) - m.read(i).powi(2) / p;
        if denom > 1e-300 {
            (p / denom).sqrt()
        } else {
            1.0
        }
    })
}

/// Non-zeros per gene (column) of `x`.
pub fn gene_nnz(x: &CsrMatrix) -> Vec<usize> {
    let mut nnz = vec![0usize; x.ncols()];
    for i in 0..x.nrows() {
        for (j, v) in x.row(i) {
            if v != 0.0 {
                nnz[j] += 1;
            }
        }
    }
    nnz
}

/// Reset gene factors fitted from fewer than `min_nnz` subsample cells to the
/// median of the others; returns how many were reset.
///
/// A gene seen in one subsample cell has a column target (unit variance over m
/// cells) and that cell's row target (unit variance over p genes) that both
/// fix the product c_i·d_j, at different values. Sinkhorn then drifts along
/// c_i↓ d_j↑ while X_w's variances (its stopping rule) barely move, and d_j can
/// reach 1e11. Resetting and then refining on all cells ([`GeneMoments`])
/// recomputes such genes from every cell that expresses them.
pub fn reset_rare_genes(d: &mut Col<f64>, nnz: &[usize], min_nnz: usize) -> usize {
    let mut ok: Vec<f64> = (0..d.nrows())
        .filter(|&j| nnz[j] >= min_nnz)
        .map(|j| d.read(j))
        .collect();
    if ok.is_empty() {
        return 0;
    }
    ok.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ok[ok.len() / 2];
    let mut reset = 0;
    for (j, &k) in nnz.iter().enumerate() {
        if k < min_nnz {
            d.write(j, med);
            reset += 1;
        }
    }
    reset
}

/// Gene-factor refinement over row blocks: the gene half of Sinkhorn in closed
/// form, mirroring [`cell_factors`]. With m_j = Σᵢ cᵢ x_ij and u_j = Σᵢ cᵢ² x_ij²:
///
///   d_j² (u_j − m_j²/n) = n   ⇒   d_j = √(n / (u_j − m_j²/n))
///
/// One streaming pass (c from [`cell_factors`] per block, then `add`) is one
/// warm-started full-data Sinkhorn step; it corrects the noise in a d fitted on
/// a cell subsample. Genes with zero variance keep d_j = 1.
pub struct GeneMoments {
    m: Vec<f64>,
    u: Vec<f64>,
    n: usize,
}

impl GeneMoments {
    pub fn new(p: usize) -> Self {
        Self {
            m: vec![0.0; p],
            u: vec![0.0; p],
            n: 0,
        }
    }

    pub fn add(&mut self, block: &CsrMatrix, c: &Col<f64>) {
        // ponytail: serial O(nnz) scatter; per-thread partial sums if it ever shows in profiles
        for i in 0..block.nrows() {
            let ci = c.read(i);
            for (j, x) in block.row(i) {
                self.m[j] += ci * x;
                self.u[j] += ci * ci * x * x;
            }
        }
        self.n += block.nrows();
    }

    pub fn finish(self) -> Col<f64> {
        let n = self.n as f64;
        Col::from_fn(self.m.len(), |j| {
            let denom = self.u[j] - self.m[j].powi(2) / n;
            if denom > 1e-300 {
                (n / denom).sqrt()
            } else {
                1.0
            }
        })
    }
}

/// Running sums for S = (YᵀY − nμμᵀ)/(n−1), Y = diag(c) X diag(d), over row blocks.
pub struct CovAccumulator {
    gram: Mat<f64>,
    colsum: Vec<f64>,
    n: usize,
}

impl CovAccumulator {
    pub fn new(p: usize) -> Self {
        Self {
            gram: Mat::zeros(p, p),
            colsum: vec![0.0; p],
            n: 0,
        }
    }

    /// Add a block of rows with their cell factors `c` (block-local indexing).
    pub fn add(&mut self, block: &CsrMatrix, c: &Col<f64>, d: &Col<f64>) {
        self.add_with_t(block, &block.transpose(), c, d);
    }

    /// As [`CovAccumulator::add`] with the block's transpose precomputed.
    ///
    /// Column j of the block's YᵀY is Σ_{i: x_ij≠0} y_ij · Y[i,:]: Σᵢ nnzᵢ² work,
    /// columns filled in parallel without sharing. Each entry sums the same
    /// products in the same (increasing i) order from either side, so the
    /// accumulated Gram stays exactly symmetric.
    pub fn add_with_t(&mut self, block: &CsrMatrix, bt: &CsrMatrix, c: &Col<f64>, d: &Col<f64>) {
        assert_eq!(block.ncols(), self.colsum.len(), "block width != p");
        assert_eq!(c.nrows(), block.nrows(), "c length != block rows");
        for (j, sum) in self.colsum.iter_mut().enumerate() {
            *sum += bt
                .row(j)
                .map(|(i, x)| c.read(i) * x * d.read(j))
                .sum::<f64>();
        }
        self.gram
            .as_mut()
            .par_col_chunks_mut(1)
            .enumerate()
            .for_each(|(j, mut colmat)| {
                let col = colmat.as_mut().col_mut(0).try_as_slice_mut().unwrap();
                let dj = d.read(j);
                for (i, xij) in bt.row(j) {
                    let ci = c.read(i);
                    let a = ci * xij * dj;
                    for (k, xik) in block.row(i) {
                        col[k] += a * (ci * xik * d.read(k));
                    }
                }
            });
        self.n += block.nrows();
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Finish: (S, μ). Needs n ≥ 2.
    pub fn finish(self) -> (Mat<f64>, Vec<f64>) {
        let n = self.n;
        assert!(n >= 2, "need at least 2 cells");
        let mu: Vec<f64> = self.colsum.iter().map(|s| s / n as f64).collect();
        let inv = 1.0 / (n - 1) as f64;
        let mut s = self.gram;
        s.as_mut()
            .par_col_chunks_mut(1)
            .enumerate()
            .for_each(|(j, mut colmat)| {
                let col = colmat.as_mut().col_mut(0).try_as_slice_mut().unwrap();
                // n·(μⱼμₖ), not (n·μⱼ)·μₖ: the latter rounds differently per side
                for (k, v) in col.iter_mut().enumerate() {
                    *v = (*v - n as f64 * (mu[j] * mu[k])) * inv;
                }
            });
        (s, mu)
    }
}

/// Eigenvalues (ascending) of the Gram YYᵀ/(m−1) of an m×p centred row block,
/// with q' = p/m: the input `EigensolverMode::Subsample` needs in
/// [`crate::spca::SparsePCA::fit_covariance`].
pub fn subsample_spectrum(y: &Mat<f64>) -> (Vec<f64>, f64) {
    crate::spca::gram_spectrum(y)
}
