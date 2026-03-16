use crate::rmt::RmtTheory;

/// Kolmogorov-Smirnov distance between the empirical eigenspectrum and the
/// theoretical Marchenko-Pastur CDF.
///
/// Returns the KS statistic max_x |F_empirical(x) - F_MP(x)|.
/// A small value (≲ 0.05 for large samples) indicates the bulk matches noise.
pub fn verify_mp_fit(eigenvalues: &[f64], p: usize, n: usize) -> f64 {
    let rmt = RmtTheory { q: p as f64 / n as f64 };
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = sorted.len();

    let mut ks = 0.0_f64;
    for (i, &lambda) in sorted.iter().enumerate() {
        let f_theory = rmt.mp_cdf(lambda);
        // ECDF has a jump at lambda: check both the left and right limits
        let f_right = (i + 1) as f64 / m as f64;
        let f_left = i as f64 / m as f64;
        ks = ks.max((f_right - f_theory).abs());
        ks = ks.max((f_left - f_theory).abs());
    }
    ks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_noise_small_ks() {
        // Eigenvalues of a Wishart matrix (p=10, n=100) should follow MP closely.
        // We approximate with 30 quantile-spaced values from the MP CDF as a
        // self-consistency smoke test.
        let rmt = crate::rmt::RmtTheory { q: 10.0 / 100.0 };
        let m = 30;
        let eigenvalues: Vec<f64> = (1..=m)
            .map(|i| {
                // Binary-search inversion of the CDF
                let target = i as f64 / (m + 1) as f64;
                let lm = rmt.lambda_minus();
                let lp = rmt.lambda_plus();
                let mut lo = lm + 1e-9;
                let mut hi = lp - 1e-9;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if rmt.mp_cdf(mid) < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (lo + hi) / 2.0
            })
            .collect();

        let ks = verify_mp_fit(&eigenvalues, 10, 100);
        assert!(ks < 0.05, "KS = {ks}");
    }
}
