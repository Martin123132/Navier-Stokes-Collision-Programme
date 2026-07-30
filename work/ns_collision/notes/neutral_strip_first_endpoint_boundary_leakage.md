# Certified first boundary-leakage endpoint

## Target

At `t=3/8`, this stage compares the exact stored modified-chain point-source
semigroup with the exact Galerkin semigroup on the frozen 240-column trial
space. Both are observed through the exact common-circle modified boundary
map.

This is the source-oriented quantity identified by the floating pilot. It
includes the initial source-projector mismatch and subsequent dynamical
leakage without replacing either by the global two-block state norm.

## Full-state side

The full state is loaded from the hash-bound degree-320 recurrence cache. Its
error already contains:

1. exact-to-computational generator transfer;
2. Chebyshev recurrence and accumulation roundoff;
3. SciPy coefficient implementation error;
4. the exact infinite polynomial tail;
5. point-source construction roundoff.

## Reduced Galerkin side

For the stored trial matrix `V`, define

```text
R=V^T M V,       K=V^T A V,       L=R^(-1)K.
```

The prior directed projection errors for `R` and `K` are combined with the
fresh Cholesky residual. A generalized-semigroup Duhamel bound uses the
certified low floor `2.36` and the square-root condition numbers of the exact
and central restricted mass matrices.

Source-coordinate solve error, trial embedding error, triangular solves,
symmetric eigensystem reconstruction, interval exponentials, and final dense
actions are charged separately.

## Boundary map

The modified boundary map is

```text
O=sqrt(a^(-1)) B^T M^(-1/2).
```

The exact inverse-arc interval from the common-circle geometry certificate is
combined with binary64 construction error. Sparse output multiplication is
then enclosed entrywise and converted to a directed boundary `L2` norm.

## Scope

This closes one finite stored-chain endpoint. It does not control the
supremum between endpoint times, later repeated steps, the post-time-6 tail,
continuum Ritz transfer, or polygon-to-circle domain perturbation. The
production screen remains unchanged.

The executable is
`scripts/neutral_strip_first_endpoint_boundary_leakage_certificate.py`.
