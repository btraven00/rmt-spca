//! RMT-guided Sparse PCA for single-cell RNA-seq.
//!
//! Implements the pipeline from Chardès et al., "A statistical physics approach
//! to characterise single-cell data" (2025), arXiv:2509.15429:
//!
//! 1. **Biwhitening** ([`biwhitening`]) — Sinkhorn-Knopp diagonal scaling so
//!    each gene and cell has unit variance; places the noise covariance on the
//!    canonical Marchenko-Pastur scale.
//! 2. **RMT thresholding** ([`rmt`]) — Marchenko-Pastur bulk edge λ+ = (1+√q)²;
//!    eigenvalues above λ+ are genuine signal (BBP phase transition).
//! 3. **Sparse PCA** ([`spca`]) — FISTA proximal gradient maximising
//!    Tr(W^T S W) − λ‖W‖₁  subject to W^T W = I_k.
//! 4. **Validation** ([`verification`]) — Kolmogorov-Smirnov test of the bulk
//!    eigenspectrum against the MP distribution.
//! 5. **Marker genes** ([`markers`]) — top genes by loading weight per component.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use rmt_spca::spca::{FistaConfig, SparsePCA};
//! # use faer::Mat;
//! # let data = Mat::<f64>::zeros(10, 5);
//!
//! let result = SparsePCA::new(FistaConfig {
//!     lambda_frac: Some(0.3),
//!     ..FistaConfig::default()
//! }).fit(&data);
//!
//! println!("σ² = {:.4}  components = {}", result.sigma_sq, result.eigenvalues.len());
//! ```

pub mod biwhitening;
pub mod markers;
pub mod rmt;
pub mod spca;
pub mod verification;
