/// Criterion benchmarks for the rmt-spca pipeline.
///
/// Synthetic data generator produces a log-normal matrix with:
///   - k_sig "signal" components planted above the MP bulk edge
///   - Poisson-like sparsity (~80% zeros) matching typical scRNA-seq counts
///
/// Run with:
///   cargo bench --bench pipeline
///   cargo bench --bench pipeline -- --output-format verbose
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::Mat;

use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::spca::{FistaConfig, SparsePCA};

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

/// Cheap deterministic pseudo-random float in [0, 1) via xorshift64.
fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Box-Muller normal sample from two uniform draws.
fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate an n×p synthetic scRNA-seq matrix:
/// - k_sig planted signal components with SNR `snr` above the MP bulk edge
/// - remaining variance is i.i.d. log-normal noise
/// - ~80 % zeros (sparse count-like)
///
/// The matrix has non-negative entries (absolute values) so biwhitening
/// behaves the same way as on real count data.
fn synthetic_scrna(n: usize, p: usize, k_sig: usize, snr: f64, seed: u64) -> Mat<f64> {
    let mut rng = seed;

    // Build k_sig signal directions (random unit vectors in R^p)
    let mut dirs: Vec<Vec<f64>> = Vec::with_capacity(k_sig);
    for _ in 0..k_sig {
        let mut v: Vec<f64> = (0..p)
            .map(|_| {
                let u1 = xorshift(&mut rng).max(1e-300);
                let u2 = xorshift(&mut rng);
                box_muller(u1, u2)
            })
            .collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        dirs.push(v);
    }

    // Bulk eigenvalue scale: MP upper edge for this aspect ratio
    let q = p as f64 / n as f64;
    let lambda_plus = (1.0 + q.sqrt()).powi(2);
    let signal_scale = (snr * lambda_plus).sqrt();

    Mat::from_fn(n, p, |i, _j| {
        // Noise: log-normal (mean 1, sd ~1 after log)
        let u1 = xorshift(&mut rng).max(1e-300);
        let u2 = xorshift(&mut rng);
        let noise = (box_muller(u1, u2) * 0.5).exp(); // log-normal

        // Signal contribution from planted components
        let signal: f64 = dirs.iter().map(|v| signal_scale * v[i % p]).sum();

        // Sparsity: ~80 % zeros
        let drop = xorshift(&mut rng) < 0.80;
        if drop { 0.0 } else { (noise + signal).abs() }
    })
}

fn sample_covariance(x: &Mat<f64>) -> Mat<f64> {
    let inv_n = 1.0 / x.nrows() as f64;
    let s: Mat<f64> = x.as_ref().transpose() * x.as_ref();
    Mat::from_fn(s.nrows(), s.ncols(), |i, j| s.read(i, j) * inv_n)
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_biwhitening(c: &mut Criterion) {
    let mut group = c.benchmark_group("biwhitening");
    for &(n, p) in &[(500, 2_000), (1_000, 5_000), (3_000, 10_000)] {
        let data = synthetic_scrna(n, p, 3, 2.0, 42);
        group.bench_with_input(
            BenchmarkId::new("sinkhorn-knopp", format!("{n}x{p}")),
            &data,
            |b, x| {
                b.iter(|| {
                    let bw = Biwhitener::default();
                    black_box(bw.compute(x))
                })
            },
        );
    }
    group.finish();
}

fn bench_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance");
    for &(n, p) in &[(500, 2_000), (1_000, 5_000), (3_000, 10_000)] {
        let data = synthetic_scrna(n, p, 3, 2.0, 42);
        let bw = Biwhitener::default();
        let (cv, d, _) = bw.compute(&data);
        let xw = Biwhitener::apply(&data, &cv, &d);
        group.bench_with_input(
            BenchmarkId::new("XtX/n", format!("{n}x{p}")),
            &xw,
            |b, x| b.iter(|| black_box(sample_covariance(x))),
        );
    }
    group.finish();
}

fn bench_fista(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    for &(n, p) in &[(500, 2_000), (1_000, 5_000)] {
        let data = synthetic_scrna(n, p, 3, 2.0, 42);
        group.bench_with_input(
            BenchmarkId::new("SparsePCA::fit", format!("{n}x{p}")),
            &data,
            |b, x| {
                b.iter(|| {
                    black_box(
                        SparsePCA::new(FistaConfig { verbose: false, ..FistaConfig::default() })
                            .fit(x),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_biwhitening,
    bench_covariance,
    bench_fista,
);
criterion_main!(benches);
