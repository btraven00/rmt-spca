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

    /// Median of the Marchenko-Pastur distribution.
    ///
    /// Found by binary search on `mp_cdf`.  Used in the biwhitening
    /// post-convergence normalisation (Algorithm 1, final step).
    pub fn mp_median(&self) -> f64 {
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
    /// Used in the KS goodness-of-fit test to check that the bulk
    /// eigenspectrum of S is consistent with pure noise (Section 3.1 of
    /// Chardès et al., arXiv:2509.15429).
    pub fn mp_cdf(&self, x: f64) -> f64 {
        let lm = self.lambda_minus();
        let lp = self.lambda_plus();
        if x <= lm {
            return 0.0;
        }
        if x >= lp {
            return 1.0;
        }
        const N: usize = 2000; // must be even
        let h = (x - lm) / N as f64;
        let mut sum = self.mp_pdf(lm) + self.mp_pdf(x);
        for k in 1..N {
            let xk = lm + k as f64 * h;
            let coeff = if k % 2 == 0 { 2.0 } else { 4.0 };
            sum += coeff * self.mp_pdf(xk);
        }
        (sum * h / 3.0).clamp(0.0, 1.0)
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
    fn cdf_integrates_to_one() {
        let rmt = RmtTheory { q: 0.5 };
        let total = rmt.mp_cdf(rmt.lambda_plus() + 1.0);
        assert!((total - 1.0).abs() < 1e-3);
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
