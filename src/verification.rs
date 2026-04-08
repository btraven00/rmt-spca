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
/// Uses `mp_cdf_bulk` (the conditional CDF normalised to [0, 1] over [λ-, λ+])
/// rather than the full `mp_cdf`, matching Python's `score()` method which
/// normalises the MP density by its total integral before calling `kstest` on
/// non-zero eigenvalues.  For q > 1, `mp_cdf` would have a point mass of
/// (1 − 1/q) below λ-, so the empirical and theoretical CDFs would be offset
/// by that amount; `mp_cdf_bulk` removes this offset.
///
/// The two-sided ECDF statistic is computed:
///   KS = max_i  max( |F_i^right − F_bulk(λ_i)|, |F_i^left − F_bulk(λ_i)| )
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
        // mp_cdf_bulk: conditional CDF normalised to [0,1] over non-zero support.
        // Matches Python BiwhitenedCovarianceEstimator._get_cdf() normalisation.
        let f_theory = rmt.mp_cdf_bulk(val);
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

    /// Generate quantile-spaced eigenvalues from mp_cdf_bulk (the conditional
    /// CDF normalised to [0, 1]).  These perfectly follow the bulk distribution,
    /// so calculate_bulk_ks should return a near-zero KS statistic.
    fn mp_bulk_quantiles(rmt: &crate::rmt::RmtTheory, m: usize) -> Vec<f64> {
        (1..=m)
            .map(|i| {
                let target = i as f64 / (m + 1) as f64;
                let lm = rmt.lambda_minus();
                let lp = rmt.lambda_plus();
                let mut lo = lm + 1e-9;
                let mut hi = lp - 1e-9;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if rmt.mp_cdf_bulk(mid) < target { lo = mid; } else { hi = mid; }
                }
                (lo + hi) / 2.0
            })
            .collect()
    }

    // Python reference: scipy.stats.kstest(eigs_bulk, _get_cdf()) ≈ 0 when
    // eigenvalues are drawn from the MP distribution.  Test for q < 1 and q > 1.
    #[test]
    fn pure_noise_small_ks_q_lt_1() {
        // q = 0.1 (p=10, n=100): standard case, mp_cdf_bulk == mp_cdf.
        let rmt = crate::rmt::RmtTheory { q: 10.0 / 100.0 };
        let eigenvalues = mp_bulk_quantiles(&rmt, 30);
        let ks = calculate_bulk_ks(&eigenvalues, rmt.q);
        assert!(ks < 0.05, "KS (q=0.1) = {ks}");
    }

    // Python reference: same test but for q > 1 (p > n), the common case in
    // scRNA-seq.  Before the mp_cdf_bulk fix, calculate_bulk_ks would return
    // KS ≈ (1 − 1/q) because it compared bulk eigenvalues (empirical CDF: 0→1)
    // against mp_cdf (theoretical CDF: (1−1/q)→1), producing a constant offset.
    #[test]
    fn pure_noise_small_ks_q_gt_1() {
        for &q in &[2.0_f64, 5.0, 10.0] {
            let rmt = crate::rmt::RmtTheory { q };
            let eigenvalues = mp_bulk_quantiles(&rmt, 50);
            let ks = calculate_bulk_ks(&eigenvalues, q);
            assert!(ks < 0.05, "KS (q={q}) = {ks}");
        }
    }
}
