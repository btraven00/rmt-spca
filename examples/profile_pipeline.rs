/// Run the full pipeline with verbose timing output to see where time goes.
use faer::Mat;
use rmt_spca::spca::{EigensolverMode, FistaConfig, SparsePCA};

fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}
fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn synthetic_scrna(n: usize, p: usize, k_sig: usize, snr: f64, seed: u64) -> Mat<f64> {
    let mut rng = seed;
    let q = p as f64 / n as f64;
    let lp = (1.0 + q.sqrt()).powi(2);
    let signal_scale = (snr * lp).sqrt();
    let mut dirs: Vec<Vec<f64>> = (0..k_sig).map(|_| {
        let mut v: Vec<f64> = (0..p).map(|_| {
            let u1 = xorshift(&mut rng).max(1e-300);
            let u2 = xorshift(&mut rng);
            box_muller(u1, u2)
        }).collect();
        let norm: f64 = v.iter().map(|x| x*x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }).collect();
    Mat::from_fn(n, p, |i, _j| {
        let u1 = xorshift(&mut rng).max(1e-300);
        let u2 = xorshift(&mut rng);
        let noise = (box_muller(u1, u2) * 0.5).exp();
        let signal: f64 = dirs.iter().map(|v| signal_scale * v[i % p]).sum();
        if xorshift(&mut rng) < 0.80 { 0.0 } else { (noise + signal).abs() }
    })
}

fn main() {
    for (n, p) in [(500, 2000), (1000, 5000)] {
        let data = synthetic_scrna(n, p, 3, 2.0, 42);

        eprintln!("\n=== {n}×{p}  [Full EVD] ===");
        let t0 = std::time::Instant::now();
        let r_full = SparsePCA::new(FistaConfig {
            verbose: true, lambda_frac: Some(0.3), ..FistaConfig::default()
        }).fit(&data);
        eprintln!("[total]        {:.2}s  σ²={:.4}  KS={:.4}",
            t0.elapsed().as_secs_f64(),
            r_full.sigma_sq,
            r_full.ks_distance.unwrap_or(f64::NAN));

        eprintln!("\n=== {n}×{p}  [Fast σ²] ===");
        let t0 = std::time::Instant::now();
        let r_fast = SparsePCA::new(FistaConfig {
            verbose: true, lambda_frac: Some(0.3),
            eigensolver: EigensolverMode::Fast,
            ..FistaConfig::default()
        }).fit(&data);
        eprintln!("[total]        {:.2}s  σ²={:.4}  (KS unavailable)",
            t0.elapsed().as_secs_f64(),
            r_fast.sigma_sq);

        // Deviation report
        let sigma_rel_err = (r_fast.sigma_sq - r_full.sigma_sq).abs() / r_full.sigma_sq;
        let lp_full = (1.0 + (p as f64 / n as f64).sqrt()).powi(2);
        let lp_shift = lp_full * sigma_rel_err;
        eprintln!("\n  σ² deviation:  full={:.4}  fast={:.4}  rel_err={:.4} ({:.2}%)",
            r_full.sigma_sq, r_fast.sigma_sq, sigma_rel_err, 100.0 * sigma_rel_err);
        eprintln!("  λ+ shift:      {lp_shift:.4} (absolute, {:.2}%)",
            100.0 * sigma_rel_err);
        eprintln!("  components:    full={}  fast={}",
            r_full.eigenvalues.len(), r_fast.eigenvalues.len());
    }
}
