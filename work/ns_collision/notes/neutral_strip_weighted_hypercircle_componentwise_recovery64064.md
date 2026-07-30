# Componentwise Residual Recovery Through 64,064 Pivots

## Scope

This checkpoint repairs the standalone congruence-residual certificate at the
previously obstructed 64,064-pivot state-region boundary. The repair keeps the
same frozen interval pencil, ordering, positive scale, natural-order binary64
`L,D` reference, and source/reference hashes. It changes only the theorem used
to bound the transformed residual.

It certifies no pivot beyond 64,063, full 123,816-pivot inertia, weighted
global Ritz constant, continuum spectral capture, Navier-Stokes regularity,
or Clay-prize claim.

## Exact old-bound failure

A hash-bound precision-60 bisection followed by adjacent precision-100
replays locates the first failure of the old separated norm product:

| Prefix length | Last included pivot | Old ratio | Decision |
| ---: | ---: | ---: | --- |
| 64,039 | 64,038 | `0.4506887` | certified |
| 64,040 | 64,039 | `3.5757143` | fail-closed |

The ratio is nondecreasing over these nested natural-order factors. The full
63,680 leading factor and the adjacent 64,039 leading factor are both
bit-for-bit preserved when the next pivot is appended.

The triggering pivot is edge-metric pivot 64,039, original index 21. Its
reference diagonal is large and positive:

```text
10703.980598536496472661383450031280517578125
```

It is not a near-zero pivot. Its nine strict-lower entries include the
coefficient

```text
-3623.21038762363377827568911015987396240234375
```

back to the earlier delicate edge pivot 63,629. That one coupling supplies
almost the entire new inverse-row recurrence. Adding pivot 64,039 increases:

- the inverse infinity majorant by `2.1889x`;
- the inverse one majorant by `1.2849x`;
- the residual infinity norm by `2.8211x`; and
- the old transformed bound by `7.9339x`.

The maximum residual entry simultaneously moves to the new diagonal
coordinate `(64039, 64039)`.

## Sharper theorem

For the fixed reference

```text
A = L (D + L^-1 (A - L D L^T) L^-T) L^T,
```

let `R` be the symmetric nonnegative componentwise magnitude bound for
`A-LDL^T`, and let

```text
Q = (I - |L-I|)^-1.
```

Because `L-I` is strictly lower triangular,

```text
|L^-1| <= Q.
```

Therefore

```text
|L^-1 (A-LDL^T) L^-T| <= Q R Q^T.
```

The matrix `Q R Q^T` is symmetric and nonnegative. Its maximum row sum bounds
its spectral radius and hence the transformed residual spectral norm. The
row-sum vector is evaluated directly as

```text
Q R Q^T 1.
```

The old estimate instead multiplied three independent global maxima:

```text
||Q||_inf ||R||_inf ||Q||_1.
```

That product combined large values occurring on different paths. The direct
componentwise propagation preserves their geometry and is always no larger.
A three-dimensional synthetic regression independently matches the recurrence
against an explicitly formed dense `Q R Q^T`.

## Certified recovery

The componentwise calculation closes at both decimal precisions 60 and 100.
Every precision-100 upper bound nests inside its precision-60 counterpart,
all standalone provenance and binary-reference hashes match, and the old
separated result is reproduced exactly inside each new artifact.

| Quantity | 64,040 pivots | 64,064 pivots |
| --- | ---: | ---: |
| Negative reference diagonals | 32,491 | 32,500 |
| Positive reference diagonals | 31,549 | 31,564 |
| Zero reference diagonals | 0 | 0 |
| Minimum absolute diagonal | `0.000815079692889409` | same |
| Componentwise transformed bound | `1.71623e-5` | `2.20370e-4` |
| Bound/minimum ratio | `0.0210560` | `0.2703661` |
| Certified safety factor | `47.4924` | `3.69869` |
| Improvement over old bound | `169.819x` | `353.408x` |

Thus the complete seven-transition state-region checkpoint through pivot
64,063 is independently certified as `32500/31564/0`.

The componentwise bound grows by `12.8403x` between prefix lengths 64,040 and
64,064, so the recovery does not justify a long extrapolation. The next
symbolic transition is distant, at pivot 76,921, but that is not an admitted
target.

## Next bounded gate

The next target is exactly 64,128 pivots, only 64 beyond the certified
endpoint. It is a local componentwise-bound growth diagnostic and crosses no
new symbolic transition. It must be replayed at precisions 60 and 100 with
all provenance and upper-bound nesting checks.

No full-pencil or continuum stage is admitted.
