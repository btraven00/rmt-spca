use faer::Mat;
use rmt_spca::spca::{FistaConfig, SparsePCA};

fn main() {
    // Synthetic example: n=80 cells, p=40 genes, rank-2 signal in noise.
    let n = 80_usize;
    let p = 40_usize;
    let k_true = 2_usize;
    let signal_strength = 3.0_f64;

    let x = Mat::from_fn(n, p, |i, j| {
        let signal = if j < k_true {
            signal_strength * ((i + j) as f64 / (n + p) as f64)
        } else {
            0.0
        };
        let noise = ((i * 7 + j * 13) as f64 * 0.6931).sin() * 0.3;
        signal + noise
    });

    let result = SparsePCA::new(FistaConfig {
        lambda: 0.05,
        max_iterations: 500,
        tolerance: 1e-7,
        verbose: true,
        ..FistaConfig::default()
    }).fit(&x);

    let (out_p, out_k) = (result.components.nrows(), result.components.ncols());
    let nnz = (0..out_p)
        .flat_map(|i| (0..out_k).map(move |j| (i, j)))
        .filter(|&(i, j)| result.components.read(i, j).abs() > 1e-6)
        .count();

    println!("Components: {out_k}  non-zero loadings: {nnz}/{}", out_p * out_k);
    println!("σ² = {:.4}  KS = {:.4}",
        result.sigma_sq,
        result.ks_distance.unwrap_or(f64::NAN));
}
