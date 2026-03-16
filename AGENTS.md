# AGENTS.md

## Build Commands

```bash
cargo build --release
```

## Test Commands

```bash
cargo test
```

## Lint Commands

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Key Implementation Notes

### Memory Layout
- Use `faer` reborrow patterns: `mat.as_ref()`, `mat.reborrow()`
- Avoid allocations in FISTA loops

### RMT Thresholds
- `lambda_plus = (1 + sqrt(p/n))^2` for biwhitened data
- `gamma = 1 / (2 * lambda_max(S))` for sparse PCA convergence

### Verification
- KS-distance between empirical eigenspectrum and theoretical MP distribution
