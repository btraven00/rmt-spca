# Known issues

## Biwhitened counts keep a sequencing-depth axis (2026-10-15)

**What.** Sinkhorn biwhitening equalises per-cell and per-gene *variances*, not
means. A cell's library size survives as a common factor on its whitened row
(roughly sqrt(depth)), i.e. a rank-1 direction in cell space. On GSE147405
(A549 TGFB1 / TNF, DU145, per time window and pooled) it is the largest
eigenvalue of the biwhitened covariance in every window, 130–225 against an MP
edge of ~26, and `sc_spca` reports it as component 0 (top loadings HSP90AA1,
ENO1, UBB, RAN; BBP overlap² 0.88). Every gene correlates with it.

**Consequence.** Any gene selector or component interpretation built on the
biwhitened matrix sees depth as signal. In a planted-signal check (Poisson
counts with a per-cell depth factor, 120 planted programme genes), a
Randomly-style per-gene test at nominal FDR 5 % selected ~20 % of the null genes
with the axis left in and 0.1 % with it projected out (106/120 planted
recovered; K = 0 and nothing selected when no programme is planted). Log-library
normalisation never sees the axis, which is why HVG-type selection does not have
this problem.

**Fix that worked.** After biwhitening and gene-centring, project the per-cell
mean direction of the whitened matrix out of cell space before the SVD / before
counting components above the edge. Reference implementation and the self-check:
`~/phd/rmt-a549-emt/18_rmt_union_vs_hvg.py` (`rmt_select`, `--keep-depth-axis`
to reproduce the failure, `--self-check` for the planted test). Whether this
belongs in the crate (an option in `FistaConfig`, or at least a warning when the
top eigenvalue tracks the row sums) is open.

**Related.** The Eq. 8 (`--auto-penalty`) loadings are dense: on the same data
every one of ~13k genes has a non-zero loading in some retained component
(~1200 effective genes per component), so "non-zero loading" is not a usable
gene-selection rule; a per-component budget or a loadings null is needed.

## Build: `c++` on hosts with a stray old compiler (2026-10-15)

Not a crate bug, recorded because it cost time: on a host where `c++` resolves
to GCC 7 while `g++` is GCC 11, builds that pass `-fstack-clash-protection`
(e.g. extension modules of a uv-managed Python) fail; set `CC=gcc CXX=g++`.
The `sc_spca` binary built here links libpython3.12 and glibc 2.38+, so it does
not run on Ubuntu 22.04 hosts; build there instead.

## σ² bulk median: outliers in or out (design difference vs the reference, 2026-10-19)

**What.** `spca.rs` (`Full` and `Subsample`) estimates the noise scale as
median(bulk) / MP median with `bulk = eigenvalues in (0.001, λ+]`, i.e. the k
outliers are dropped first. Chardès' reference (`biwhitening_fast` →
`_get_scaling_marchenko_pastur`, vchz/spcarmt c6a125b) takes the median of
*all* n non-zero eigenvalues (of the uncentred Sinkhorn matrix). Everything
else is identical: on 821 Ex nuclei × 9,218 promoters of the C. elegans
multiome (ATAC promoter counts) the Sinkhorn factors of the two
implementations agree to corr(log c) = corr(log d) = 1.00000 and the
unrescaled spectra are the same (51 eigenvalues above (1 + √q)²).

**Consequence.** σ² = 0.892 (port) vs 0.928 (reference) on those nuclei; the
ATAC spectrum has ~16 eigenvalues within ±2 % of the edge, so k = 98 vs 80
(reference's rule applied to the port's spectrum: 79; the last 1 is centred
vs uncentred). Negligible when k/n is small (RNA, paul15: ~2 %), large for
ATAC-like data (6 %). Neither rule is unbiased once spikes are present: in the
`sigma_sq_bulk_median_estimators` test (30 strong spikes planted in 300 × 900
unit noise) the all-eigenvalues median overestimates σ² by 5–8 %, the
below-edge median underestimates it by 0.2–3.5 %; with no spikes both give
the same value. An order-statistics argument that the reference's rule is
unbiased (because outliers sit above the median) is wrong: interlacing lifts
the bulk eigenvalues as well. The real lesson is that k is ill-conditioned
on such spectra (~4 components per 1 % of σ²); the cross-fit null at the
subspace level is the robust quantity.

**Done.** `FistaConfig::sigma_bulk_all` / `sc_spca --sigma-bulk-all`
reproduces the reference's rule (79–80 vs 80 on the nuclei above); the default
is unchanged. Scripts: `~/phd/quivers/celegans/scripts/13_spike_sigma.py`,
`14_spike_biwhitening.py`; write-up `~/phd/quivers/celegans/spikes/2026-10-19-atac-rank.md`.

**Related.** The agreement gate for using the port instead of Chardès in the
quivers project (`celegans/PLAN.md`, MP rank within 1) is met in
`--sigma-bulk-all` mode, which is then a faithful fast implementation of the
registered method.

## h5ad reader rejects CSR with unsorted column indices (2026-10-20)

**What.** `sc_spca` fails at load time with `cannot read csr matrix: Minor
indices are not monotonically increasing within each lane` on an `.h5ad`
whose CSR `X` has unsorted column indices within a row. Such matrices are
valid CSR and come out of ordinary scipy operations: a sparse product
(`P @ L.T`, promoter activity from peak counts) and column fancy-indexing
(`X[:, idx]`) both produce them; `scipy.sparse.csr_matrix.has_sorted_indices`
is `False` and nothing in scipy or anndata warns. The refusal is in the
`anndata-hdf5` reader (rmt-spca-io), not in this crate.

**Consequence.** A caller gets a hard failure on some matrices and not on
others from the same pipeline (Exx's promoter matrix happened to be sorted,
Exxx's not), which looked like a non-deterministic crash until stderr was
captured. Cost ~2 h on the C. elegans agreement gate (2026-10-20).

**Workaround.** `X.sort_indices()` before `write_h5ad`
(`~/phd/quivers/celegans/scripts/16_gate_port_vs_chardes.py::port_fit`).

**Fixed (rmt-spca-io 2d87b26, 2026-10-20).** CSR `X` is now read straight
from HDF5 (`RawCsr` in `src/reader.rs`), bypassing anndata 0.3, whose row
slicing also `unwrap()`-panicked on such files under `--stream-block`. Lanes
are sorted on load and duplicates summed; all three loaders (dense, `--sparse`,
`--stream-block`) go through it. Test: `raw_csr_tests::unsorted_lanes_load`.
The `sort_indices()` workaround is no longer needed.
