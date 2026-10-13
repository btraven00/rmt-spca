use faer::Mat;

/// A single marker gene for one sparse PCA component.
pub struct MarkerGene {
    pub name: String,
    /// Loading weight (signed).  Positive = co-expressed with the component;
    /// negative = anti-correlated.
    pub weight: f64,
}

/// Return the top-`top_n` marker genes for component `k_idx`.
///
/// Filters out zero loadings (|weight| ≤ 1e-6) and sorts by absolute weight
/// descending.  `gene_names` must have length == `loadings.nrows()`.
pub fn get_top_markers(
    loadings: &Mat<f64>,
    gene_names: &[String],
    k_idx: usize,
    top_n: usize,
) -> Vec<MarkerGene> {
    let mut markers: Vec<MarkerGene> = (0..gene_names.len())
        .map(|g| MarkerGene {
            name: gene_names[g].clone(),
            weight: loadings.read(g, k_idx),
        })
        .filter(|m| m.weight.abs() > 1e-6)
        .collect();

    markers.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap());
    markers.truncate(top_n);
    markers
}

/// Print a formatted marker-gene table for all components.
///
/// ```text
/// Component 0  (λ=34.21, overlap=0.994)
///   Rank  Gene              Weight
///      1  S100A9           +0.1823
///      2  LYZ              +0.1701
///      ...
/// ```
pub fn print_marker_table(
    loadings: &Mat<f64>,
    gene_names: &[String],
    eigenvalues: &[f64],
    overlaps: &[f64],
    top_n: usize,
) {
    let k = loadings.ncols();
    for c in 0..k {
        let lambda = eigenvalues.get(c).copied().unwrap_or(f64::NAN);
        let overlap = overlaps.get(c).copied().unwrap_or(f64::NAN);
        println!("\nComponent {c}  (λ={lambda:.4}, overlap={overlap:.4})");
        println!("  {:>4}  {:<20}  {:>8}", "Rank", "Gene", "Weight");
        println!("  {}  {}  {}", "-".repeat(4), "-".repeat(20), "-".repeat(8));
        let markers = get_top_markers(loadings, gene_names, c, top_n);
        if markers.is_empty() {
            println!("  (no non-zero loadings)");
        }
        for (rank, m) in markers.iter().enumerate() {
            println!("  {:>4}  {:<20}  {:>+8.4}", rank + 1, m.name, m.weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_markers_sorted_by_abs_weight_zero_filtered_truncated() {
        let w = [0.1, -0.5, 0.0, 0.3, 1e-9];
        let loadings = Mat::from_fn(5, 1, |i, _| w[i]);
        let names: Vec<String> = ["a", "b", "c", "d", "e"].map(String::from).to_vec();
        let m = get_top_markers(&loadings, &names, 0, 2);
        let got: Vec<(&str, f64)> = m.iter().map(|g| (g.name.as_str(), g.weight)).collect();
        assert_eq!(got, vec![("b", -0.5), ("d", 0.3)]);
        assert_eq!(get_top_markers(&loadings, &names, 0, 10).len(), 3);
    }
}
