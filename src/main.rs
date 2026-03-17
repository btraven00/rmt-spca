use faer::Mat;
use rmt_spca::biwhitening::biwhiten;
use rmt_spca::rmt::RmtTheory;
use rmt_spca::spca::{FistaConfig, SparsePCA};
use rmt_spca::verification::verify_mp_fit;

fn main() {
    // -----------------------------------------------------------------------
    // Synthetic example: n=80 cells, p=40 genes, rank-2 signal embedded in noise
    // -----------------------------------------------------------------------
    let n = 80_usize;
    let p = 40_usize;
    let k_true = 2_usize;

    // Build a low-rank signal + noise matrix.
    // Signal: two orthogonal loading vectors with known structure.
    let signal_strength = 3.0_f64;
    let x = Mat::from_fn(n, p, |i, j| {
        let signal = if j < k_true {
            signal_strength * ((i + j) as f64 / (n + p) as f64)
        } else {
            0.0
        };
        // Deterministic pseudo-noise using a simple hash-like formula
        let noise = ((i * 7 + j * 13) as f64 * 0.6931).sin() * 0.3;
        signal + noise
    });

    // -----------------------------------------------------------------------
    // Biwhitening
    // -----------------------------------------------------------------------
    let xw = biwhiten(&x);
    println!("Biwhitened matrix: {}×{}", xw.nrows(), xw.ncols());

    // -----------------------------------------------------------------------
    // RMT threshold
    // -----------------------------------------------------------------------
    let rmt = RmtTheory { q: p as f64 / n as f64 };
    println!(
        "MP bulk edge λ+ = {:.4}  (q = {:.3})",
        rmt.lambda_plus(),
        rmt.q
    );

    // -----------------------------------------------------------------------
    // Sparse PCA via FISTA
    // -----------------------------------------------------------------------
    let config = FistaConfig {
        lambda: 0.05,
        max_iterations: 500,
        tolerance: 1e-7,
        ..FistaConfig::default()
    };
    let spca = SparsePCA::new(config);
    let result = spca.fit(&x);
    let (out_p, out_k) = (result.components.nrows(), result.components.ncols());
    println!("Sparse PCA: {out_k} component(s), each of length {out_p}");

    // Sparsity: fraction of near-zero entries
    let nnz = (0..out_p)
        .flat_map(|i| (0..out_k).map(move |j| (i, j)))
        .filter(|&(i, j)| result.components.read(i, j).abs() > 1e-6)
        .count();
    let total = out_p * out_k;
    println!(
        "Non-zero loadings: {nnz}/{total} ({:.1}%)",
        100.0 * nnz as f64 / total as f64
    );

    // -----------------------------------------------------------------------
    // Verify MP fit on biwhitened covariance eigenvalues (diagonal as proxy)
    // -----------------------------------------------------------------------
    let diag_eigs: Vec<f64> = (0..p.min(n))
        .map(|i| xw.read(i % n, i % p).powi(2))
        .collect();
    let ks = verify_mp_fit(&diag_eigs, p, n);
    println!("KS distance from MP bulk: {ks:.4}");
}
