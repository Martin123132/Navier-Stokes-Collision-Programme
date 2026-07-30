# Sparse inertia and indexed exact-polygon spectrum

## Purpose

The residual audit produced 241 disjoint intervals, each containing a
generalized eigenvalue of the stored `h=0.06` pencil. Existence in disjoint
intervals did not by itself prove that these were the first 241 eigenvalues.
This stage closes that counting gap and transfers the indexed intervals to
the exact Gaussian-weighted finite-element forms on the same stored polygon.

It does not transfer eigenvectors or reach the continuum circle problem.

## Generalized inertia count

Let `A_s` and `M_s` be the exactly stored binary64 stiffness and mass
matrices. The row-lumped coercivity audit proves that `M_s` is positive
definite. Hence, for a real shift `sigma`, the number of generalized
eigenvalues below `sigma` equals the negative inertia of

```text
A_s - sigma M_s.
```

The checker obtains the `MMD_AT_PLUS_A` symmetric permutation and structural
factor pattern from SuperLU. It does not use SuperLU's numerical pivots in
the certificate. Independently, it verifies:

1. stored mass and stiffness are bitwise symmetric;
2. row and column permutations agree;
3. the lower and upper factor patterns are transposes;
4. the exact pencil pattern is contained in the proposed lower pattern;
5. all 10,485,740 symbolic descendant-pair closure conditions hold.

It then recomputes the scalar LDL recurrence from the exact stored binary
entries with directed Decimal interval arithmetic. Every completed pivot
interval excludes zero, so diagonal congruence and Sylvester inertia give
the count.

The two shifts are

```text
retained-gap shift       106.71699046840874
post-interval shift      107.02775717792179.
```

At precision 220 the complete 15,211-pivot rows give

```text
shift                 negative   positive   minimum pivot margin
retained gap               240      14971   1.9201342690948274e-6
post interval              241      14970   6.908046900791498e-6.
```

All proximity intervals with indices `0,...,239` lie below the first shift,
and interval 240 lies above it. All 241 intervals lie below the second
shift. The counts therefore identify the intervals as the first 241
indexed stored-pencil eigenvalue intervals.

## Precision reproduction

A separate precision-260 run writes separate result and pivot caches. The
cross-check compares every pivot, not only the aggregate inertia:

```text
pivot signs compared                         30,422
all signs equal                              yes
both permutations equal                      yes
both shifts bitwise equal                    yes
every p260 interval nested in p220 interval  yes.
```

The worst relative pivot widths contract as follows:

```text
shift                 precision 220       precision 260
retained gap          6.45608e-25         6.85947e-65
post interval         9.92195e-2          1.05813e-41.
```

The p220 post-interval row is decisive but visibly affected by interval
dependency. The nested p260 reproduction is therefore an essential
stability check.

Lower precision runs failed closed and were not promoted:

```text
arithmetic / precision   first failed pivot(s)
binary64                 12065
Decimal 50               14378 / 14389
Decimal 100              14795 / 14855
Decimal 180              15118 / 15180.
```

These are conditioning diagnostics, not conflicting inertia results.

## Exact-polygon form transfer

Let `a_e,m_e` be the exact Gaussian-weighted P1 forms on the stored polygon.
The assembly audit proves

```text
|m_e(v,v)-m_s(v,v)| <= eta_M m_s(v,v)
|a_e(v,v)-a_s(v,v)| <= eta_A m_s(v,v),
```

with

```text
eta_M = 5.492564204190142e-13
eta_A = 5.431182629562088e-9
m_e >= 0.9999999999994507 m_s.
```

For every nonzero vector, the exact Rayleigh quotient is enclosed by

```text
(R_s-eta_A)/(1+eta_M) <= R_e
R_e <= (R_s+eta_A)/(1-eta_M).
```

Applying min-max to each indexed stored interval gives 241 indexed exact
polygon intervals. All formula endpoints are outward rounded and contain
an independent 100-digit Decimal evaluation. The resulting guards are

```text
minimum adjacent exact interval separation       8.21109564412836e-5
exact index-239 upper                             106.4162237645287
exact index-240 lower                             107.01775717228844
retained/complement separation                    0.6015334077597457.
```

Thus the exact-polygon complement after the retained 240-dimensional block
has generalized eigenvalue lower bound

```text
lambda_240 >= 107.01775717228844.
```

## Remaining gates

This closes indexed eigenvalue counting for the stored pencil and exact
finite-element forms on the stored polygon. It does not close:

1. exact-polygon generalized eigenvector or projector enclosures;
2. Riesz, Gram, push, cross, and endpoint propagation errors;
3. off-block leakage in the required trace norm;
4. continuum Ritz projector transfer;
5. polygon-to-circle domain perturbation;
6. the full Navier-Stokes composition.

## Artifacts

The primary checker is
`scripts/neutral_strip_common_circle_sparse_inertia_audit.py`.
The precision comparator is
`scripts/neutral_strip_sparse_inertia_precision_crosscheck.py`.
The exact-form transfer is
`scripts/neutral_strip_exact_polygon_indexed_spectrum_transfer.py`.

The principal results are
`results/neutral_strip_h006_q12_k240_sparse_inertia_audit_v1.json`,
`results/neutral_strip_h006_q12_k240_sparse_inertia_precision_crosscheck_v1.json`,
and
`results/neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json`.

