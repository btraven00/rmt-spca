use crate::rmt::RmtTheory;

/// Kolmogorov-Smirnov distance between the **bulk** eigenspectrum of the
/// biwhitened covariance and the Marchenko-Pastur CDF.
///
/// This is the goodness-of-fit test described in Section 3.1 of Chardès et al.
/// A small KS distance (≲ 0.10) confirms that the noise floor of S is
/// consistent with the MP distribution, validating that biwhitening produced
/// a properly normalised matrix and that the RMT assumptions hold.
///
/// Only bulk eigenvalues are included; the following are filtered out:
/// - Eigenvalues **above** λ+ (BBP signal outliers — not noise)
/// - Eigenvalues **≤ 0.001** (zero-spike from structural zeros / unexpressed genes)
///
/// The two-sided ECDF statistic is computed:
///   KS = max_i  max( |F_i^right − F_MP(λ_i)|, |F_i^left − F_MP(λ_i)| )
///
/// where F_i^right = i/m, F_i^left = (i−1)/m are the right and left limits
/// of the empirical CDF at the i-th sorted eigenvalue.
pub fn calculate_bulk_ks(eigenvalues: &[f64], q: f64) -> f64 {
    let rmt = RmtTheory { q };
    let l_plus = rmt.lambda_plus();

    let mut bulk: Vec<f64> = eigenvalues
        .iter()
        .cloned()
        .filter(|&e| e <= l_plus && e > 0.001)
        .collect();
    bulk.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let m = bulk.len();
    if m == 0 { return 0.0; }

    let mut ks = 0.0_f64;
    for (i, &val) in bulk.iter().enumerate() {
        let f_theory = rmt.mp_cdf(val);
        let f_right = (i + 1) as f64 / m as f64;
        let f_left = i as f64 / m as f64;
        ks = ks.max((f_right - f_theory).abs());
        ks = ks.max((f_left - f_theory).abs());
    }
    ks
}

/// KS distance between the **full** empirical eigenspectrum and the MP CDF.
///
/// Unlike `calculate_bulk_ks`, this includes all eigenvalues (signal outliers
/// and zero-spike).  Kept for reference; prefer `calculate_bulk_ks` for
/// validating biwhitening quality.
pub fn verify_mp_fit(eigenvalues: &[f64], p: usize, n: usize) -> f64 {
    let rmt = RmtTheory { q: p as f64 / n as f64 };
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = sorted.len();

    let mut ks = 0.0_f64;
    for (i, &lambda) in sorted.iter().enumerate() {
        let f_theory = rmt.mp_cdf(lambda);
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
        // Approximate with 30 quantile-spaced values from the MP CDF as a
        // self-consistency smoke test.
        let rmt = crate::rmt::RmtTheory { q: 10.0 / 100.0 };
        let m = 30;
        let eigenvalues: Vec<f64> = (1..=m)
            .map(|i| {
                let target = i as f64 / (m + 1) as f64;
                let lm = rmt.lambda_minus();
                let lp = rmt.lambda_plus();
                let mut lo = lm + 1e-9;
                let mut hi = lp - 1e-9;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if rmt.mp_cdf(mid) < target { lo = mid; } else { hi = mid; }
                }
                (lo + hi) / 2.0
            })
            .collect();

        let ks = verify_mp_fit(&eigenvalues, 10, 100);
        assert!(ks < 0.05, "KS = {ks}");
    }
}
