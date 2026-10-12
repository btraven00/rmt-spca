//! Minimal CSR matrix for the sparse pipeline ([`crate::spca::SparsePCA::fit_csr`]).
//!
//! Only what the pipeline needs: validated construction, zero-row/col
//! filtering, transposition, (squared) matrix-vector products, and the
//! biwhitened covariance. Memory stays O(nnz + p²); no n×p dense matrix is formed.

use faer::{Col, Mat};
use rayon::prelude::*;

/// Canonical CSR matrix: strictly increasing column indices within each row.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CsrMatrix {
    /// Validated constructor. Rejects bad offsets, out-of-range or unsorted /
    /// duplicate column indices, and non-finite values. Duplicates are rejected
    /// rather than summed: biwhitening squares entries, and (a+b)² ≠ a²+b².
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self, String> {
        if indptr.len() != nrows + 1 || indptr[0] != 0 || indptr[nrows] != indices.len() {
            return Err("indptr does not match nrows / nnz".into());
        }
        if indices.len() != data.len() {
            return Err("indices and data lengths differ".into());
        }
        for i in 0..nrows {
            if indptr[i] > indptr[i + 1] {
                return Err(format!("indptr not monotone at row {i}"));
            }
            let row = &indices[indptr[i]..indptr[i + 1]];
            if row.iter().any(|&j| j >= ncols) {
                return Err(format!("column index out of range in row {i}"));
            }
            if row.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("row {i}: column indices not strictly increasing"));
            }
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err("non-finite value".into());
        }
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        })
    }

    /// CSR copy of a dense matrix (exact zeros dropped).
    pub fn from_dense(x: &Mat<f64>) -> Self {
        let (n, p) = (x.nrows(), x.ncols());
        let mut indptr = Vec::with_capacity(n + 1);
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        indptr.push(0);
        for i in 0..n {
            for j in 0..p {
                let v = x.read(i, j);
                if v != 0.0 {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        Self {
            nrows: n,
            ncols: p,
            indptr,
            indices,
            data,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Entries (column, value) of row i.
    #[inline]
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.indptr[i]..self.indptr[i + 1];
        self.indices[r.clone()]
            .iter()
            .copied()
            .zip(self.data[r].iter().copied())
    }

    /// Row-major transpose (i.e. the CSC of `self`). Row indices come out in
    /// increasing order, so the result is canonical.
    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.ncols + 1];
        for &j in &self.indices {
            counts[j + 1] += 1;
        }
        for j in 0..self.ncols {
            counts[j + 1] += counts[j];
        }
        let indptr = counts.clone();
        let mut next = counts;
        let mut indices = vec![0usize; self.nnz()];
        let mut data = vec![0.0f64; self.nnz()];
        for i in 0..self.nrows {
            for (j, v) in self.row(i) {
                indices[next[j]] = i;
                data[next[j]] = v;
                next[j] += 1;
            }
        }
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            indptr,
            indices,
            data,
        }
    }

    /// Keep rows and columns with at least one non-zero value (Stage 0).
    /// Returns the filtered matrix and the kept (row, col) indices.
    pub fn drop_empty(&self) -> (Self, Vec<usize>, Vec<usize>) {
        let mut col_nz = vec![false; self.ncols];
        let mut row_nz = vec![false; self.nrows];
        for (i, nz) in row_nz.iter_mut().enumerate() {
            for (j, v) in self.row(i) {
                if v != 0.0 {
                    col_nz[j] = true;
                    *nz = true;
                }
            }
        }
        let rows: Vec<usize> = (0..self.nrows).filter(|&i| row_nz[i]).collect();
        let cols: Vec<usize> = (0..self.ncols).filter(|&j| col_nz[j]).collect();
        if rows.len() == self.nrows && cols.len() == self.ncols {
            return (self.clone(), rows, cols);
        }
        let mut remap = vec![usize::MAX; self.ncols];
        for (new, &old) in cols.iter().enumerate() {
            remap[old] = new;
        }
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        indptr.push(0);
        for &i in &rows {
            for (j, v) in self.row(i) {
                if v != 0.0 {
                    indices.push(remap[j]);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        let m = Self {
            nrows: rows.len(),
            ncols: cols.len(),
            indptr,
            indices,
            data,
        };
        (m, rows, cols)
    }

    /// y = X v, or (X∘X) v when `squared`.
    pub fn mul_vec(&self, v: &Col<f64>, squared: bool) -> Col<f64> {
        let out: Vec<f64> = (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                self.row(i)
                    .map(|(j, x)| if squared { x * x } else { x } * v.read(j))
                    .sum()
            })
            .collect();
        Col::from_fn(self.nrows, |i| out[i])
    }

    /// Biwhitened, centred covariance from `self` = X (n×p) and `xt` = Xᵀ:
    ///
    ///   S = (Yᵀ Y − n μ μᵀ) / (n − 1),   Y = diag(c) X diag(d),   μ = colmeans(Y)
    ///
    /// Column j of YᵀY is Σ_{i: x_ij≠0} y_ij · Y[i,:], so the work is Σᵢ nnzᵢ²
    /// instead of n·p², and columns are filled in parallel without sharing.
    /// Each entry sums the same products in the same (increasing i) order from
    /// either side, so S is exactly symmetric.
    pub fn biwhitened_covariance(&self, xt: &Self, c: &Col<f64>, d: &Col<f64>) -> Mat<f64> {
        let mut acc = crate::streaming::CovAccumulator::new(self.ncols);
        acc.add_with_t(self, xt, c, d);
        acc.finish().0
    }

    /// Drop rows with no non-zero value, keeping all columns (for streamed
    /// blocks, whose column indexing must stay global). Returns kept row indices.
    pub fn drop_empty_rows(&self) -> (Self, Vec<usize>) {
        let rows: Vec<usize> = (0..self.nrows)
            .filter(|&i| self.row(i).any(|(_, v)| v != 0.0))
            .collect();
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        indptr.push(0);
        for &i in &rows {
            for (j, v) in self.row(i).filter(|&(_, v)| v != 0.0) {
                indices.push(j);
                data.push(v);
            }
            indptr.push(indices.len());
        }
        let m = Self {
            nrows: rows.len(),
            ncols: self.ncols,
            indptr,
            indices,
            data,
        };
        (m, rows)
    }
}

impl CsrMatrix {
    /// Column means μ of Y = diag(c) X diag(d), from `xt` = Xᵀ.
    pub fn biwhitened_mean(&self, xt: &Self, c: &Col<f64>, d: &Col<f64>) -> Vec<f64> {
        let n = self.nrows as f64;
        (0..self.ncols)
            .map(|j| {
                xt.row(j)
                    .map(|(i, x)| c.read(i) * x * d.read(j))
                    .sum::<f64>()
                    / n
            })
            .collect()
    }

    /// Dense rows `rows` of the centred biwhitened matrix Y − 1μᵀ.
    pub fn centred_rows(&self, rows: &[usize], c: &Col<f64>, d: &Col<f64>, mu: &[f64]) -> Mat<f64> {
        let mut y = Mat::from_fn(rows.len(), self.ncols, |_, j| -mu[j]);
        for (t, &i) in rows.iter().enumerate() {
            for (j, x) in self.row(i) {
                y.write(t, j, c.read(i) * x * d.read(j) - mu[j]);
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Mat<f64> {
        // 4×3 with an empty row (2) and an empty column (1)
        Mat::from_fn(4, 3, |i, j| match (i, j) {
            (_, 1) | (2, _) => 0.0,
            _ => (i * 3 + j + 1) as f64,
        })
    }

    #[test]
    fn validation_rejects_bad_input() {
        assert!(CsrMatrix::new(1, 2, vec![0, 2], vec![1, 0], vec![1.0, 1.0]).is_err());
        assert!(CsrMatrix::new(1, 2, vec![0, 2], vec![1, 1], vec![1.0, 1.0]).is_err());
        assert!(CsrMatrix::new(1, 2, vec![0, 1], vec![2], vec![1.0]).is_err());
        assert!(CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![f64::NAN]).is_err());
        assert!(CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).is_ok());
    }

    #[test]
    fn transpose_and_matvec_match_dense() {
        let x = small();
        let a = CsrMatrix::from_dense(&x);
        let t = a.transpose();
        let v = Col::from_fn(3, |j| (j + 1) as f64 * 0.5);
        let w = Col::from_fn(4, |i| 1.0 - i as f64);
        let xv: Col<f64> = x.as_ref() * v.as_ref();
        let xtw: Col<f64> = x.as_ref().transpose() * w.as_ref();
        let av = a.mul_vec(&v, false);
        let tw = t.mul_vec(&w, false);
        let sq = a.mul_vec(&v, true);
        for i in 0..4 {
            assert_eq!(av.read(i), xv.read(i));
            let want: f64 = (0..3).map(|j| x.read(i, j).powi(2) * v.read(j)).sum();
            assert_eq!(sq.read(i), want);
        }
        for j in 0..3 {
            assert_eq!(tw.read(j), xtw.read(j));
        }
    }

    #[test]
    fn drop_empty_removes_zero_row_and_col() {
        let (m, rows, cols) = CsrMatrix::from_dense(&small()).drop_empty();
        assert_eq!((m.nrows(), m.ncols()), (3, 2));
        assert_eq!(rows, vec![0, 1, 3]);
        assert_eq!(cols, vec![0, 2]);
        assert_eq!(m.row(2).collect::<Vec<_>>(), vec![(0, 10.0), (1, 12.0)]);
    }
}
