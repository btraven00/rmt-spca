/// Random Matrix Theory helpers for biwhitened scRNA-seq data.
pub struct RmtTheory {
    pub q: f64, // p/n (genes / cells)
}

impl RmtTheory {
    /// Upper edge of the Marchenko-Pastur bulk [cite: 87, 152]
    pub fn lambda_plus(&self) -> f64 {
        (1.0 + self.q.sqrt()).powi(2)
    }

    /// Lower edge of the Marchenko-Pastur bulk
    pub fn lambda_minus(&self) -> f64 {
        (1.0 - self.q.sqrt()).powi(2)
    }

    /// Predicted squared cosine overlap of an outlier eigenvector with the true signal [cite: 156, 382]
    pub fn predicted_overlap(&self, lambda: f64) -> f64 {
        let alpha = self.calculate_alpha(lambda);
        ((alpha - 1.0).powi(2) - self.q) / ((alpha - 1.0) * (alpha - 1.0 + self.q))
    }

    /// Inverse BBP map: recover signal eigenvalue alpha from observed outlier lambda [cite: 387, 391]
    /// lambda = alpha + q * alpha/(alpha-1)  =>  alpha^2 - (lambda+1-q)*alpha + lambda = 0
    fn calculate_alpha(&self, lambda: f64) -> f64 {
        let b = -(lambda + 1.0 - self.q);
        let disc = b * b - 4.0 * lambda;
        (-b + disc.max(0.0).sqrt()) / 2.0
    }

    /// Marchenko-Pastur density at x
    pub fn mp_pdf(&self, x: f64) -> f64 {
        let lp = self.lambda_plus();
        let lm = self.lambda_minus();
        if x <= lm || x >= lp {
            return 0.0;
        }
        ((lp - x) * (x - lm)).sqrt() / (2.0 * std::f64::consts::PI * self.q * x)
    }

    /// Marchenko-Pastur CDF via composite Simpson's rule (2000 panels, even)
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
