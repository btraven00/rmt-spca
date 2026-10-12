# Biwhitening collapse from rare genes

**Summary.** When genes are detected in too few cells, Sinkhorn biwhitening has no
finite solution. It then drives some cell factors cᵢ → 0 and gene factors dⱼ → ∞,
which erases those cells from the covariance, while its stopping rule still reports
convergence. The problem appears once n ≳ p. Rare cell types are hit first.

`SparsePCAResult::collapsed_cells` detects it. A detection filter of
**genes in ≥ 2–3·n/p cells** removes it on Zheng 2017 at 50k and 94k cells and
improves every downstream number.

## Mechanism

Take a gene j detected in k cells with counts ≈ 1. Unit variance across cells needs

    (cᵢ dⱼ)² · (k/n) ≈ 1   ⇒   (cᵢ dⱼ)² ≈ n/k.

Each of those k cells also has a unit variance across p genes to meet, and gene j
alone uses up

    (cᵢ dⱼ)² / p ≈ n / (k·p)

of it. If n/(k·p) > 1, or several rare genes share a cell and their shares add up
past 1, no finite (c, d) satisfies both sets of constraints. Sinkhorn then drifts
along cᵢ ↓, dⱼ ↑. The stopping rule watches the change in X_w's variances, which
barely moves, so the drift continues until the factors over- or underflow:
`max d / median d` reaches 1e30–1e107 in the runs below.

A collapsed cell has every other gene multiplied by cᵢ ≈ 0, so it drops out of S.
The cells that carry rare genes go first, and those are disproportionately rare
cell types.

The bound depends on n/p:

- Small n/p (10k cells, all genes; n/p ≈ 0.65): safe.
- Large n, or HVG selection (p = 2000 gives n/p = 5 at 10k): unsafe.

Small n/p is **not** sufficient on its own. With very few cells the per-gene
bound is easy, but a cell carrying several rare genes adds up their shares.

- Duò kumar, 245 cells × 2000 HVGs (n/p = 0.12): the ≥ 3·n/p filter keeps every
  gene (⌈0.37⌉ = 1), and one cell still collapsed (max d/median 2.5e3).
- So the filter needs an absolute floor as well: genes in
  ≥ max(5, ⌈3·n/p⌉) cells.

HVG selection makes it worse in a second way: the variance ranking favours
high-dispersion genes, and those include rare ones.

The reference Python implementation (spcarmt) uses the same update and stopping
rule, so it should share the failure. This is untested.

## Diagnostic

```rust
let r = SparsePCA::new(cfg).fit_csr(&x);
r.collapsed_cells        // cells with cᵢ < 1e-3·median(c)
r.max_gene_factor_ratio  // max(d)/median(d); healthy ≈ 10–100, collapse ≫ 1e3
```

The same pair is available as `spca::biwhitening_collapse(&c, &d)`. In verbose
mode a warning gives the suggested minimum detection count. The threshold has a
wide margin: healthy runs have min(c)/median(c) ≈ 0.1.

Tests (`tests/sparse_precision.rs`, section 6):

- A fixture with one gene in 4 cells (n/(k·p) = 2.5) is flagged, and exactly the
  4 expressing cells are the collapsed ones. This holds on both the dense and CSR
  paths.
- The same fixture after a ≥ 2·n/p detection filter, and a feasible fixture,
  both report 0.

## Does it occur on real data, and does the filter fix it?

Zheng 2017 FACS-sorted PBMC, all genes. Biwhitening is the default in-memory
Sinkhorn. σ²/KS come from a 5000-cell subsample and FISTA uses product reuse;
neither affects biwhitening. "ARI" is against the sort labels, using spectral
clustering on the RMT precision graph (`prec`) and kNN + Leiden at the true
cluster count (`kNN`). Reproduce with `sc-rmt-lapl/exp_degeneracy.py`.

| n | gene filter | p | collapsed cells | max d/median | KS | k | ARI prec | ARI kNN |
|---|---|---|---|---|---|---|---|---|
| 10k | ≥ 3 cells (n/p < 1, so the filter changes nothing) | 15,311 | 0 | 31 | 0.020 | 19 | 0.325 | 0.292 |
| 50k | ≥ 3 cells | 18,288 | **334** | 1e30 | 0.046 | 28 | 0.268 | 0.271 |
| 50k | ≥ 6 (= 2·n/p) | 16,881 | 0 | 65 | 0.016 | 34 | 0.315 | 0.272 |
| 50k | ≥ 9 (= 3·n/p) | 16,066 | 0 | 32 | **0.010** | 37 | **0.331** | 0.270 |
| 94k | ≥ 3 cells | 19,444 | **4,991** | 7e96 | 0.069 | 33 | 0.238 | **0.002** |
| 94k | ≥ 10 (= 2·n/p) | 17,055 | 0 | 93 | 0.015 | 46 | 0.300 | 0.216 |
| 94k | ≥ 15 (= 3·n/p) | 16,270 | 0 | 40 | **0.010** | 50 | **0.313** | 0.213 |
| 94k | ≥ 10, streamed | 17,055 | 0 | 75 | 0.014 | 47 | 0.322 | 0.217 |

At 94k with ≥ 3 cells, the 4,991 collapsed cells (5.3%) include **291 of 518
CD14 monocytes (56%)** and 1,821 CD34 cells.

What the filter does, at both sizes:

- It removes the collapse.
- It brings the bulk spectrum back to Marchenko–Pastur (KS ≈ 0.01).
- It uncovers more signal components.
- It improves ARI.

The cost is 8–17% of genes, all detected in fewer than ~3·n/p cells. 3·n/p was at
least as good as 2·n/p on every metric.

Earlier results from collapsed runs should not be reused. That covers:

- 50k and 94k with the ≥ 3-cell filter;
- every HVG-2000 run at 10k: Zheng had 1,484 collapsed cells (15%), splatter 135.

## Recommendation

- Check `collapsed_cells == 0` before trusting any run with n ≳ p.
- Filter genes to those detected in ≥ max(5, ⌈3·n/p⌉) cells, computing p after
  the filter and iterating once if it moves much. The floor matters on small
  datasets; see kumar above. This is a detection filter, independent of
  gene variance, so it doesn't bias the noise spectrum the way HVG selection does.
- Not done here: putting the filter inside `fit`/`fit_csr`. Dropping genes would
  need the result to return the kept gene indices; Stage 0's zero-gene drop
  already has that gap.
- The alternative of a regularised Sinkhorn (a variance floor that keeps every
  gene) changes the method and is untested.
