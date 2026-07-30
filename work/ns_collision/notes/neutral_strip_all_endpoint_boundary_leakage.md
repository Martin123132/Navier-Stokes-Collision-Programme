# Certified boundary leakage at all production endpoints

## One-step operator error

The source-specific recurrence result is upgraded to an arbitrary-state
one-step operator bound. The sparse action uses

```text
|| |X| ||_2
 <= sqrt(|| |X| ||_1 || |X| ||_infinity),
```

which reduces to the directed absolute row sum because the scaled matrix is
symmetric. Local recurrence errors are propagated with the same
`U`-polynomial identity used by the first-step certificate. Coefficient,
tail, accumulation, and exact-to-computational generator errors are then
added to obtain `epsilon_step`.

If `S=exp(-hH)` and `P` is the computed polynomial step,

```text
||P^n-S^n||
 <= n epsilon_step
    max(||P||,||S||)^(n-1).
```

The exact contraction is bounded by `exp(-1.9h)`, while
`||P||` is at most that contraction plus `epsilon_step`.

## Reduced state

The first-endpoint exact Galerkin transfer bound is evaluated at later times
with the direct Duhamel factor

```text
n exp(-2.36 h (n-1)).
```

The dense repeated-step error receives the same conservative factor. Initial
point-source construction error is propagated through the computed full-state
contraction.

## Endpoint output

For each of the 16 hash-bound central pilot rows, the checker adds:

1. repeated full-state action error;
2. exact reduced-form and dense reduced-action error;
3. exact boundary-operator propagation;
4. output construction and sparse multiplication error;
5. final 112-component norm-evaluation roundoff.

The endpoint-only weighted sum is reported as a diagnostic. It is not charged
to the production screen because strict decrease at the endpoints does not
prove monotonicity between endpoints.

## Scope

There are 15 finite `3/8` windows from `t=3/8` through `t=6`. The first 15
endpoint values anchor those windows; the 16th endpoint at `t=6` anchors the
separate post-terminal tail. All 15 within-window suprema and the post-time-6
tail remain open. The result is finite stored-chain evidence only and does
not perform continuum Ritz or polygon-to-circle transfer.

The executable is
`scripts/neutral_strip_all_endpoint_boundary_leakage_certificate.py`.
