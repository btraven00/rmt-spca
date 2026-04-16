# rmt-spca

Rust implementation of RMT-guided Sparse PCA for single-cell RNA-seq, following
[Chardès et al. (2025), arXiv:2509.15429](https://arxiv.org/abs/2509.15429).

This crate is the core algorithm library.  For file I/O (`.h5ad`), log-normalisation,
UMAP plotting, and a ready-to-use CLI, see the companion crate
[`rmt-spca-io`](rmt-spca-io).

## Quick start (library)

```rust
use rmt_spca::spca::{FistaConfig, SparsePCA};

let config = FistaConfig {
    lambda_frac: Some(0.3),   // λ = 0.3 × λ+; targets ~200–500 genes/component
    verbose: true,
    ..FistaConfig::default()
};
let result = SparsePCA::new(config).fit(&data); // data: faer::Mat<f64>, cells × genes
println!("σ² = {:.4}  KS = {:.4}", result.sigma_sq, result.ks_distance.unwrap_or(f64::NAN));
```

## Quick start (CLI)

The validate CLI lives in `rmt-spca-io`:

```bash
cd ../rmt-spca-io
cargo run --release --example validate -- data.h5ad --lambda-frac 0.3
```

## Pipeline

| Stage | Description | Cost |
|-------|-------------|------|
| 0 | Zero filtering — drop empty cells/genes | O(np) |
| 1 | **Biwhitening** (Algorithm 1) — Sinkhorn-Knopp scaling for unit per-gene and per-cell variance | O(np · iters) |
| 2 | Mean-centring | O(np) |
| 3 | Sample covariance S = X_w^T X_w / (n−1) | O(np²) |
| 4 | Full EVD → bulk-median σ² normalisation + KS diagnostic | **O(p³)** |
| 5 | Subspace iteration → signal components above λ+ = (1+√q)² | O(p² · k · iters) |
| 6 | **FISTA Sparse PCA** (Algorithm 2) — proximal gradient with Nesterov momentum | O(p² · k · iters) |

Stage 4 dominates for large p (76% of total time at p=5000).  See `EigensolverMode::Fast`
for an approximate O(p) alternative (read its documentation before use).

## Configuration

Key `FistaConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `lambda_frac` | `None` | λ = frac × λ+; overrides `lambda`; ~0.3 targets 200–500 genes/component |
| `lambda` | `0.1` | L1 penalty (absolute); ignored when `lambda_frac` is set |
| `eigensolver` | `Full` | `EigensolverMode::Full` (exact) or `::Fast` (approximate, 4× faster) |
| `compute_ks` | `true` | Compute KS goodness-of-fit; only available with `Full` eigensolver |
| `bw_max_iter` | `1000` | Max Sinkhorn-Knopp iterations |
| `bw_damp` | `1.0` | Sinkhorn under-relaxation ∈ (0,1]; try 0.5–0.8 if biwhitening oscillates |

## Output

`SparsePCAResult` fields:

| Field | Description |
|-------|-------------|
| `components` | p×k sparse loading matrix |
| `eigenvalues` | Rayleigh quotients for the k signal components |
| `s_eigenvalues` | All p covariance eigenvalues (empty in `Fast` mode) |
| `lambda_plus` | MP bulk edge λ+ |
| `sigma_sq` | Noise scale σ² (≈1 for well-biwhitened data) |
| `ks_distance` | KS distance from MP, or `None` |

## License

GPLv3
