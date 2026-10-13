# PLAN — per-component L1 penalty for sparse PCA

Written 2026-10-12 from the A549 EMT analysis (`~/phd/rmt-a549-emt`, PLAN.md
§ "rmt-spca L1 collapse"; worklog `rmt-a549-emt`). Another agent should be
able to implement and test this from here without re-deriving it.

## Problem

With `lambda_frac` set (the documented way to use the crate), `fit()` passes
**one absolute penalty** `λ = lambda_frac × λ+` to `fista_sparse_pca`, which
soft-thresholds every column of W with the same `γλ` and then re-normalises
each column (modified Gram-Schmidt). Per column this is the fixed-point
iteration of

    max_w  w'Sw − λ‖w‖₁   s.t. ‖w‖₂ = 1

On A549 (GSE147405, TGFB1 arm, 3568 × 13226 after QC, `--lambda-frac 0.3`,
`--k-max 80`) the result is 50 of 53 columns with a single non-zero gene
(max |w| > 0.9, everything else ~1e-4); the G2/M mode (RQ λ = 73, predicted
overlap 0.95) came out as CENPA alone. EGF: 70/80, TNF: 64/73.

Why, numerically (reproduced in `~/phd/rmt-a549-emt`, scratch script
`collapse_spec.py`; eigenvalues of the σ²-scaled biwhitened covariance match
the crate's Rayleigh quotients to 2 decimals):

1. **Biwhitening puts every gene's variance at ≈ 1.** diag(S) quantiles
   50/90/99.9 % are all 1.1. So the single-gene solution e_g scores
   `S_gg − λ ≈ 1 − λ` for every gene and every component: a flat floor.
2. **Signal eigenvectors are spread**: effective genes 1/IPR = 1000–2600 per
   component (√m_eff ≈ 40), ‖v_j‖₁ = 72–87. The dense solution scores
   `λ_j − λ‖v_j‖₁`. Under a uniform-loading model the objective over a
   support of m genes is `λ_j m/m_eff − λ√m`, convex in √m, so the optimum
   is an endpoint: **dense or singleton, nothing between**.
3. **Survival rule**: a column stays spread iff `λ_j − 1 > λ(√m_eff − 1)`,
   roughly `λ_j > 40 λ`. Checked against the fitted columns (TGFB1):

   | lambda_frac | λ | predicted survivors | observed spread columns |
   |---|---|---|---|
   | 0.3 | 2.57 | 0, 1 | 0, 1, 2 |
   | 0.1 | 0.86 | 0–5 | 0–4, 6, 8 |
   | 0.05 | 0.43 | 0–11, 13, 14 | 19 of 53 |

4. **The spectrum spans λ+ = 8.6 to λ_max = 206.** For the last signal
   component to survive you need λ < (8.6 − 1)/39 ≈ 0.19, i.e.
   `lambda_frac < 0.023`, and at that λ component 0 keeps ~all of its 1800
   effective genes. One absolute λ cannot both sparsify the head of the
   spectrum and keep the tail alive. The paper's calibration ("0.3–0.7 × λ+
   gives 200–500 genes per component") is a property of Zheng2017's
   marker-dominated programs and narrower spectrum, not of the method.
5. Dynamics make it all-or-nothing: subtract a constant from every entry,
   then re-normalise, is a rich-get-richer map; below the survival line the
   only fixed point is a coordinate vector.

The FISTA iteration itself matches the Python reference (`spcarmt`,
`_fista_spca`: one scalar penalty, soft-threshold, Gram-Schmidt). **What the
port dropped is the penalty selection.** The reference never takes a fixed
penalty: `RMTSparsePCA._transform` calls `_adjust_penalty`, which
root-finds γ* (`scipy.optimize.root_scalar`, bracket (1e-4, 4)) so that the
distance between the sparse subspace W and the top-k outlier eigenspace V,

    Σ_j (1 − σ_j²),   σ_j = singular values of WᵀV,

equals the RMT-predicted `distance_ = Σ_j (1 − proj_j)` with `proj_j` the
BBP squared overlap ((α−1)² − q)/((α−1)(α−1+q)) — the same formula as this
crate's `RmtTheory::predicted_overlap` — and then uses `0.6 × γ*`
(`scale = 0.6`). That is the paper's Eq. 8 ("select the sparsity level so
that the inferred subspace and the outlier subspace approximately match the
angle predicted by RMT"), the thing that makes the method "nearly
parameter-free", and the paper says in so many words that "applying sparse
PCA with an overestimated penalty invariably destroys the biological
signal" with "a sharp performance drop for penalty parameters above γ*".
`lambda_frac × λ+` and the "~0.3 targets 200–500 genes" comment in
`sc_spca` are this crate's shortcut, not the paper's.

Checked on TGFB1 (k = 53, `scratchpad` script, eigenvectors from an SVD of
the centred biwhitened matrix):

| run | observed Σ(1 − σ²) | Eq. 8 target Σ(1 − proj) | singletons |
|---|---|---|---|
| lambda_frac 0.3 | 50.4 | 26.6 | 50/53 |
| lambda_frac 0.1 | 47.4 | 26.6 | 45/53 |
| lambda_frac 0.05 | 43.5 | 26.6 | 34/53 |
| dense eigenspace | 0.0 | 26.6 | — |

Every run is far on the over-penalised side of the criterion; a collapsed
column contributes ≈ 1 to the distance, so Eq. 8 cannot be satisfied with
more than a handful of singletons. The reference's root-finder would have
driven γ well below 0.05 λ+ here. The collapse is a property of the
objective; the paper guards against it with Eq. 8, and the port removed the
guard.

## Status, 2026-10-12 evening

**Fix 0 is implemented and tested** (uncommitted, `src/spca.rs`):

- `FistaConfig::auto_penalty: Option<f64>` — `Some(scale)` selects λ by Eq. 8
  and fits at `scale × λ*`; overrides `lambda` / `lambda_frac`; default
  `None`, so every existing path and cross-validation test is unchanged.
- `select_penalty_eq8` (bisection on log λ, bracket [1e-4, λ+] widened as
  needed, 5 % relative width) and `subspace_distance` (= k − ‖WᵀV‖_F², no
  SVD) are public helpers.
- `SparsePCAResult` gained `lambda`, `lambda_star`, `subspace_distance`,
  `target_distance`; verbose mode logs each Eq. 8 evaluation, the achieved
  vs target distance, and warns on single-gene components.
- `sc_spca --auto-penalty [SCALE]` (default 0.6) in rmt-spca-io; the
  components JSON carries the four new fields.
- Tests (all green with clippy `-D warnings` and fmt):
  `subspace_distance_is_zero_for_same_span_and_one_per_orthogonal_column`;
  `eq8_root_hits_target_and_sits_on_the_collapse_cliff` (3 disjoint spikes
  40/12/4: the distance is a staircase in λ, one step per collapsed column,
  the root sits just before the first step, 0.6 λ* keeps all columns spread,
  3 λ* collapses the weakest);
  `auto_penalty_through_fit_covariance_reports_and_avoids_singletons`
  (public path: fixed λ = λ+ collapses a spike, Eq. 8 collapses none and
  each component stays on its support);
  `eq8_root_matches_python_reference` (`scripts/generate_eq8_vector.py`
  ports `_fista_spca` + the `_adjust_penalty` root-find onto
  `test_vectors/spiked_model.json`; crate λ* within 15 % of γ* = 0.7407,
  same distance at the root).

**Found and fixed on the way: `subspace_iteration` started from the first k
coordinate vectors.** That start has zero overlap with any signal direction
supported outside genes 0..k, so on a block-structured covariance the
iteration could never find it (the 3-spike test found 1 spike and reported
k = 1). Now a deterministic xorshift start; eigenvector order is unchanged.
On A549 the old start had converged (RQs matched an SVD to 2 decimals), so
no earlier result is affected, but any dataset where the first k genes are
uninformative was at risk. Test: `subspace_iteration_finds_spikes_on_disjoint_supports`.

A549 TGFB1 with `--auto-penalty 0.6` (`~/phd/rmt-a549-emt/results/_spca/TGFB1_eq8.{log,json}`):
10 evaluations, 85 min; λ* = 0.135 (= 0.016 λ+), fit at 0.081; **0/52
single-gene components** (was 50/53), effective genes per component
843–1579, subspace distance 10.3 vs target 25.6. The acceptance criterion
("≤ 5 singletons, G2/M carries TOP2A/CENPF/UBE2C/PLK1/AURKA") is met:
component 4 is CENPF/TOP2A/… at equal weights. Caveats: (i) the loadings are
dense (~1300 effective genes of 13k), so on a spectrum this wide Eq. 8 with
one scalar λ gives a mildly sparsified PCA, not 10²-gene programs — Fix 1
(per-component penalty) is still the route to those; (ii) the Eq. 8 trace
shows the tail cliff between λ = 0.12 and 0.25 (distance −3.5 → +11.8), so
`scale` matters: 0.6 lands below the cliff here, 1.0 would not;
(iii) cost: each evaluation at small λ runs FISTA near `max_iterations`;
a warm start from the previous W or a lower cap during the search would cut
the 85 min by several ×.

A549 TNF with `--auto-penalty 0.6` (12,911 × 13,130, k = 73): 10 evaluations, 46 min; λ* = 0.051 = 0.013 λ+ (λ+ = 4.03), fit at 0.031; **0/73 single-gene components**, effective genes 756–1412; distance 13.3 vs target 30.0. Note the absolute λ* differs 2.6× between the two arms while λ*/λ+ is 0.016 vs 0.013 — a fixed absolute λ (0.081, borrowed from TGFB1) still collapsed 7/73 on TNF, so the selection must be per dataset, which is the point of Eq. 8.

## Fix 0 (first, it is the paper): implement Eq. 8 penalty selection

- Target: `distance = Σ_j (1 − predicted_overlap(rq_j))` over the k
  retained components. Everything needed is already computed in `fit()`
  Stage 5 (`rq`, `RmtTheory::predicted_overlap`, `v_init` = the outlier
  eigenspace from subspace iteration).
- Objective of the root-find: `f(γ) = Σ_j (1 − σ_j²) − distance`, with
  σ_j the singular values of `Wᵀ v_init` for W = FISTA(γ). Monotone
  increasing in γ (more penalty, further from the eigenspace); bracket
  (1e-4, 4) × λ+ scale, `xtol` 5e-3 as in the reference, Brent. Each
  evaluation is one FISTA run, warm-started from the previous W to keep it
  cheap (the reference cold-starts; ~10–15 evaluations).
- Return `0.6 × γ*` as the working penalty (reference default `scale`),
  expose `scale` in `FistaConfig`; keep `lambda` / `lambda_frac` as explicit
  overrides but make the Eq. 8 path the default when neither is set, and
  log γ*, the achieved distance, and the singleton count.
- Test: cross-validate `γ*` and `distance_` against `spcarmt` on the
  existing test vectors (the reference prints both); A549 TGFB1 regression
  as below.

Fix 1 (per-component penalty) remains worth doing **after** Fix 0, for the
reason in point 4 above: Eq. 8 constrains a *sum* over components with one
scalar γ, so on a spectrum spanning 8.6–206 the head will be left nearly
dense to satisfy the criterion while some of the tail may still sit at the
singleton floor. Whether that happens in practice is an empirical question
Fix 0 answers; Fix 1 is the refinement if it does.

## Fix 1 (refinement beyond the paper): per-component penalty

Threshold column j with its **own Rayleigh quotient** times the fraction:

    λ_j = lambda_frac × rq_j          instead of    λ = lambda_frac × λ+

`rq` is already computed in `fit()` (Stage 5, `rq: Vec<f64>` from the
subspace iteration) and `v_init`'s columns are in the same order, so the
vector exists at the call site. The survival rule becomes

    lambda_frac < (1 − 1/λ_j) / (√m_eff − 1)

which is nearly uniform across the spectrum: one `lambda_frac` then means
the same thing for every component (the same fraction of each component's
variance is traded for sparsity), and the paper's "genes per component"
calibration transfers between datasets.

Implementation:

- `fista_sparse_pca(…, lambda: f64, …)` → `lambda: &[f64]` (length k); the
  soft-threshold loop uses `lambda[j] * gamma`. One line in the loop, one in
  the signature. Keep allocations out of the loop (AGENTS.md).
- `FistaConfig`: add

      /// Penalty mode. `Global` is the published algorithm (one λ for all
      /// columns); `PerComponent` scales λ by each component's Rayleigh
      /// quotient, so `lambda_frac` buys the same relative sparsity in every
      /// component. See PLAN.md.
      pub penalty: PenaltyMode,   // enum PenaltyMode { Global, PerComponent }

  default `Global` so every existing cross-validation test against the
  Python reference stays bit-identical. `PerComponent` applies to both the
  `lambda` and `lambda_frac` paths (absolute `lambda` becomes
  `lambda × rq_j / λ+`, so the number keeps its documented units).
- `sc_spca` example: `--penalty per-component`.
- Verbose log: print `λ_j` range (min, max) instead of the single λ, and
  print the singleton count after FISTA (`max|w_j| > 0.9`) with a warning.
  A collapsed component must not be silent.

Tests:

- `fista_per_component_matches_global_for_flat_spectrum`: synthetic S with
  k equal spikes → both modes give the same W.
- `per_component_keeps_tail_alive`: synthetic S with spikes at 200, 50, 10
  on a 2000-gene uniform-ish support, noise at λ+ ≈ 8. `Global` at 0.3
  collapses the 10 and 50 spikes to coordinate vectors (assert
  max|w| > 0.9); `PerComponent` at the same `lambda_frac` leaves all three
  spread (assert max|w| < 0.5 and support ≥ 100 genes).
- A549 regression (`test_vectors/` if a small slice fits): singleton count
  at 0.3 goes from 50/53 to ≤ 5.

## Fix 2 (heavier): deflate and target a support size

Fit one component at a time: FISTA on S, deflate `S ← S − λ_j v_j v_j'`,
bisection on λ until `#{|w| > ε}` hits a requested `genes_per_component`.
Gives the paper's "200–500 genes" literally, costs k FISTA runs, and
deflation with sparse (non-eigen) vectors leaks variance into later
components. Only if Fix 1 turns out not to give a usable support-size
control.

## Fix 3 (guard only)

After FISTA, if `max|w_j| > 0.9`, halve λ_j for that column and re-run that
column. Cheap, catches the symptom, does not change the fact that the
answer depends on where the component sits in the spectrum. Do this as the
warning in Fix 1 regardless; do not do it instead of Fix 1.

## Not fixes

- Lowering `lambda_frac` (see 4 above).
- Eigenvalue clipping / shrinkage instead of sparse PCA: dense loadings,
  fine for the downstream spectral work, not programs.
- Varimax on surviving columns: rotates 2–3 programs, cannot recreate the
  50 that collapsed.

## Done when

`sc_spca` with Eq. 8 selection (no `--lambda-frac`) on the A549 TGFB1 slice
achieves the predicted distance within tolerance and gives ≤ 5 single-gene
columns of 53, the G2/M component contains
TOP2A/CENPF/UBE2C/PLK1/AURKA, and all existing cross-validation tests pass
unchanged in `Global` mode.
