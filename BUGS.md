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
