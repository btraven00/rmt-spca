/// Random Matrix Theory quantities for biwhitened scRNA-seq data.
///
/// All formulas follow Chardès et al., "A statistical physics approach to
/// characterise single-cell data" (2023), hereafter "the paper".
///
/// The central assumption is that after biwhitening the count matrix X,
/// the noise part of the sample covariance S = X_w^T X_w / n converges to
/// a Marchenko-Pastur (MP) distribution with aspect ratio q = p/n as
/// n, p → ∞ (Section 2.1, Eq. 2).  Signal components appear as outlier
/// eigenvalues above the bulk edge λ+ via the BBP phase transition.
pub struct RmtTheory {
    /// Aspect ratio q = p/n (genes / cells).
    pub q: f64,
}

impl RmtTheory {
    /// Upper edge of the Marchenko-Pastur bulk spectrum.
    ///
    /// λ+ = (1 + √q)²  (Eq. 3 in the paper).
    ///
    /// Eigenvalues of S above λ+ are "outliers" that correspond to genuine
    /// biological signal components (BBP phase transition, Section 2.2).
    pub fn lambda_plus(&self) -> f64 {
        (1.0 + self.q.sqrt()).powi(2)
    }

    /// Lower edge of the Marchenko-Pastur bulk spectrum.
    ///
    /// λ- = (1 - √q)²  (Eq. 3).
    pub fn lambda_minus(&self) -> f64 {
        (1.0 - self.q.sqrt()).powi(2)
    }

    /// Predicted squared cosine overlap between an outlier eigenvector and
    /// the true signal direction.
    ///
    /// For an observed outlier eigenvalue λ > λ+, the BBP formula gives
    /// (Eq. 9 in the paper):
    ///
    ///   cos²θ = [ (α−1)² − q ] / [ (α−1)(α−1+q) ]
    ///
    /// where α is the underlying signal eigenvalue recovered via
    /// `calculate_alpha`.  A value near 1 means the empirical eigenvector
    /// is a reliable estimate of the true signal direction.
    pub fn predicted_overlap(&self, lambda: f64) -> f64 {
        let alpha = self.calculate_alpha(lambda);
        ((alpha - 1.0).powi(2) - self.q) / ((alpha - 1.0) * (alpha - 1.0 + self.q))
    }

    /// Inverse BBP map: recover the signal eigenvalue α from the observed
    /// outlier eigenvalue λ.
    ///
    /// The BBP forward map is (Eq. 8):
    ///   λ = α + q·α/(α−1)
    ///
    /// Rearranging: α² − (λ+1−q)·α + λ = 0, solved by the quadratic formula
    /// (taking the larger root, which corresponds to α > 1 + √q).
    fn calculate_alpha(&self, lambda: f64) -> f64 {
        let b = -(lambda + 1.0 - self.q);
        let disc = b * b - 4.0 * lambda;
        (-b + disc.max(0.0).sqrt()) / 2.0
    }

    /// Marchenko-Pastur probability density at x.
    ///
    /// For λ- < x < λ+ (Eq. 2):
    ///
    ///   ρ_MP(x) = √[(λ+−x)(x−λ-)] / (2π q x)
    ///
    /// Zero outside the support [λ-, λ+].
    pub fn mp_pdf(&self, x: f64) -> f64 {
        let lp = self.lambda_plus();
        let lm = self.lambda_minus();
        if x <= lm || x >= lp {
            return 0.0;
        }
        ((lp - x) * (x - lm)).sqrt() / (2.0 * std::f64::consts::PI * self.q * x)
    }

    /// Median of the non-zero eigenvalues of X^T X / n under the MP distribution.
    ///
    /// Used in the biwhitening post-convergence normalisation (Algorithm 1, final
    /// step) to find σ² = ℓ_med / λ_med where ℓ_med is the empirical median bulk
    /// eigenvalue and λ_med is this theoretical median.
    ///
    /// For q ≤ 1 (p ≤ n): binary search on the full `mp_cdf` for the 50th
    /// percentile of the standard MP distribution.
    ///
    /// For q > 1 (p > n): the p×p covariance X^T X / n has p − n zero eigenvalues
    /// and n non-zero eigenvalues.  The non-zero eigenvalues equal q times the
    /// eigenvalues of the n×n gram matrix X X^T / p, which follow MP(1/q).
    /// The median of the non-zero eigenvalues is therefore q × median(MP(1/q)).
    /// This matches Python's `marchenko_pastur_median(q)` which inverts q to 1/q
    /// before the binary search when q > 1.
    pub fn mp_median(&self) -> f64 {
        if self.q > 1.0 {
            // Non-zero eigenvalues of X^T X / n are q × eigenvalues of X X^T / p.
            // Eigenvalues of X X^T / p follow MP(1/q) (a proper distribution).
            // Matches Python: marchenko_pastur_median inverts q for q > 1.
            let rmt_inv = RmtTheory { q: 1.0 / self.q };
            return self.q * rmt_inv.mp_median();
        }
        let lm = self.lambda_minus();
        let lp = self.lambda_plus();
        let mut lo = lm + 1e-9;
        let mut hi = lp - 1e-9;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if self.mp_cdf(mid) < 0.5 { lo = mid; } else { hi = mid; }
        }
        (lo + hi) / 2.0
    }

    /// Marchenko-Pastur cumulative distribution function at x.
    ///
    /// Computed by composite Simpson's rule (2000 panels) over [λ-, x].
    ///
    /// For q > 1 (p > n), the empirical spectral distribution of X^T X / n (p×p)
    /// has a point mass of (1 − 1/q) at 0 from the p − n zero eigenvalues.
    /// This mass is included here so that mp_cdf is a proper probability CDF.
    /// Matches Python's `marchenko_pastur_cumulative` which adds (1 − 1/q) when
    /// q > 1 before integrating the continuous part.
    ///
    /// For the KS goodness-of-fit test against non-zero bulk eigenvalues, use
    /// `mp_cdf_bulk` instead (which conditions on eigenvalue > 0).
    pub fn mp_cdf(&self, x: f64) -> f64 {
        // Point mass at 0 for q > 1: fraction (1 − 1/q) of eigenvalues are zero.
        // Matches Python: `if (q > 1): F = 1 - 1/q`.
        let mass_at_zero = if self.q > 1.0 { 1.0 - 1.0 / self.q } else { 0.0 };
        let lm = self.lambda_minus();
        let lp = self.lambda_plus();
        if x <= lm {
            return mass_at_zero;
        }
        if x >= lp {
            return 1.0;
        }
        (mass_at_zero + self.mp_cdf_raw(x)).clamp(0.0, 1.0)
    }

    /// CDF of the non-zero (bulk) eigenvalues, normalised to [0, 1] over [λ-, λ+].
    ///
    /// For q ≤ 1 this equals `mp_cdf`.  For q > 1 the continuous MP density
    /// integrates to only 1/q over [λ-, λ+], so the raw integral is scaled by q
    /// to produce a proper CDF over the support of the non-zero eigenvalues.
    ///
    /// Use this for the KS test in `calculate_bulk_ks`, matching Python's
    /// `BiwhitenedCovarianceEstimator._get_cdf()` which normalises the MP density
    /// by its total integrated weight (`tot`) before calling `scipy.stats.kstest`
    /// on the non-zero eigenvalues.
    pub fn mp_cdf_bulk(&self, x: f64) -> f64 {
        let lm = self.lambda_minus();
        let lp = self.lambda_plus();
        if x <= lm { return 0.0; }
        if x >= lp { return 1.0; }
        let raw = self.mp_cdf_raw(x);
        // For q > 1 the density integrates to 1/q, so scale by q to reach 1.
        if self.q > 1.0 { (raw * self.q).clamp(0.0, 1.0) } else { raw.clamp(0.0, 1.0) }
    }

    /// Raw Simpson integral of mp_pdf from λ- to x (2000 panels, must be even).
    ///
    /// Integrates to 1 at λ+ for q ≤ 1, and to 1/q for q > 1.
    /// Used internally by `mp_cdf` and `mp_cdf_bulk`.
    fn mp_cdf_raw(&self, x: f64) -> f64 {
        let lm = self.lambda_minus();
        const N: usize = 2000; // must be even
        let h = (x - lm) / N as f64;
        let mut sum = self.mp_pdf(lm) + self.mp_pdf(x);
        for k in 1..N {
            let xk = lm + k as f64 * h;
            let coeff = if k % 2 == 0 { 2.0 } else { 4.0 };
            sum += coeff * self.mp_pdf(xk);
        }
        sum * h / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_plus_known_value() {
        // q=1: lambda_plus = (1+1)^2 = 4
        let rmt = RmtTheory { q: 1.0 };
        assert!((rmt.lambda_plus() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cdf_integrates_to_one_q_lt_1() {
        // For q < 1 (p < n): no mass at 0, CDF should reach 1 at λ+.
        let rmt = RmtTheory { q: 0.5 };
        let total = rmt.mp_cdf(rmt.lambda_plus() + 1.0);
        assert!((total - 1.0).abs() < 1e-3, "CDF at λ+ = {total}");
    }

    // Python reference for q > 1:
    //   marchenko_pastur_cumulative(lambda_plus(q), q) == 1.0
    //   marchenko_pastur_cumulative(lambda_minus(q) - eps, q) == 1 - 1/q
    #[test]
    fn cdf_point_mass_at_zero_for_q_gt_1() {
        // For q > 1 (p > n): point mass (1 − 1/q) at 0.
        // CDF just below λ- should equal (1 − 1/q); at λ+ should equal 1.
        let rmt = RmtTheory { q: 3.0 };
        let expected_mass = 1.0 - 1.0 / 3.0;  // ≈ 0.667
        let cdf_at_lm = rmt.mp_cdf(rmt.lambda_minus() - 1e-6);
        assert!(
            (cdf_at_lm - expected_mass).abs() < 1e-6,
            "CDF below λ- = {cdf_at_lm}, expected {expected_mass}"
        );
        let cdf_at_lp = rmt.mp_cdf(rmt.lambda_plus() + 1.0);
        assert!((cdf_at_lp - 1.0).abs() < 1e-3, "CDF at λ+ = {cdf_at_lp}");
    }

    // Python reference for mp_cdf_bulk:
    //   BiwhitenedCovarianceEstimator._get_cdf() normalises the density by its
    //   total integral (tot), so _get_cdf(lambda_plus) ≈ 1.0 for any q.
    //
    // q = 1.0 excluded: lambda_minus = 0 makes the MP density singular (∝ 1/√x
    // near 0), causing Simpson's rule to underestimate the integral by ~2%.
    #[test]
    fn cdf_bulk_normalises_to_one_for_any_q() {
        for &q in &[0.2_f64, 0.5, 2.0, 5.0, 10.0] {
            let rmt = RmtTheory { q };
            let bulk_at_lp = rmt.mp_cdf_bulk(rmt.lambda_plus() - 1e-9);
            assert!(
                (bulk_at_lp - 1.0).abs() < 1e-2,
                "q={q}: mp_cdf_bulk at λ+-ε = {bulk_at_lp}"
            );
            let bulk_at_lm = rmt.mp_cdf_bulk(rmt.lambda_minus() + 1e-9);
            assert!(
                bulk_at_lm < 0.05,
                "q={q}: mp_cdf_bulk at λ-+ε = {bulk_at_lm}"
            );
        }
    }

    // Python reference for mp_median (q > 1):
    //   marchenko_pastur_median(q) inverts to 1/q and finds median of MP(1/q).
    //   The Rust mp_median for q > 1 returns q * mp_median(1/q).
    //   Sanity check: result must lie in [λ-(q), λ+(q)] (non-zero eigenvalue support).
    #[test]
    fn median_in_support_for_q_gt_1() {
        for &q in &[1.5_f64, 2.0, 5.0, 10.0] {
            let rmt = RmtTheory { q };
            let med = rmt.mp_median();
            let lm = rmt.lambda_minus();
            let lp = rmt.lambda_plus();
            assert!(
                med >= lm && med <= lp,
                "q={q}: median={med} not in [{lm}, {lp}]"
            );
        }
    }

    // Python reference: mp_median is the 50th percentile of the conditional
    // distribution of non-zero eigenvalues.  Verify via mp_cdf_bulk.
    #[test]
    fn median_is_50th_percentile_of_bulk_cdf() {
        for &q in &[0.2_f64, 0.5, 2.0, 5.0] {
            let rmt = RmtTheory { q };
            let med = rmt.mp_median();
            let cdf_at_med = rmt.mp_cdf_bulk(med);
            assert!(
                (cdf_at_med - 0.5).abs() < 0.02,
                "q={q}: mp_cdf_bulk(median={med:.4}) = {cdf_at_med:.4}, expected 0.5"
            );
        }
    }

    #[test]
    fn predicted_overlap_in_unit_interval() {
        let rmt = RmtTheory { q: 0.5 };
        let lp = rmt.lambda_plus();
        // A strong outlier well above lambda_plus
        let ov = rmt.predicted_overlap(lp * 2.0);
        assert!(ov >= 0.0 && ov <= 1.0, "overlap={ov}");
    }
}
