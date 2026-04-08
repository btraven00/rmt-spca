# rmt-spca

Rust implementation of RMT-guided Sparse PCA for single-cell RNA-seq, following
[Chardès (2025), arXiv:2509.15429](https://arxiv.org/abs/2509.15429).

## Quick start

```bash
cargo run --release --example validate -- data.h5ad
```

## Validate CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-cells N` | all | Use first N cells (rows) |
| `--lambda L` | 0.1 | L1 sparsity penalty (absolute) |
| `--lambda-frac F` | — | Set λ = F × λ+ (overrides `--lambda`); ~0.3 targets 200–500 genes/component |
| `--bw-max-iter N` | 1000 | Max Sinkhorn-Knopp iterations for biwhitening |
| `--bw-damp F` | 1.0 | Sinkhorn under-relaxation α ∈ (0,1]; try 0.5–0.8 if biwhitening oscillates |
| `--top-markers N` | 10 | Top marker genes printed per component; `0` to suppress |
| `--no-log` | — | Skip log-normalisation (use with raw counts) |
| `--umap` | — | Project cells onto components and save a UMAP scatter-plot |
| `--umap-out PATH` | `umap.png` | Output path for the UMAP PNG |

## Pipeline

1. **Biwhitening** (Algorithm 1) — Sinkhorn-Knopp scaling so each gene and cell has unit second moment; followed by median-eigenvalue normalisation to place the bulk on the canonical MP scale.
2. **Sample covariance** — S = X_w^T X_w / n
3. **RMT thresholding** — bulk edge λ+ = (1+√q)²; eigenvalues above λ+ are signal components (BBP phase transition).
4. **FISTA Sparse PCA** (Algorithm 2) — proximal gradient ascent with Nesterov momentum; step size γ = 1/(2λ_max).
5. **KS validation** — Kolmogorov-Smirnov distance between the bulk eigenspectrum and the MP distribution; < 0.10 indicates good biwhitening.

## License

GPLv3
