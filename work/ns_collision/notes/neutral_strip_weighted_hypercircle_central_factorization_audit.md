# Central factorization audit for the hypercircle pencil

## Scope

This stage asks whether the completed `h=0.06` threshold pencil is
computationally and numerically suitable for a later verified inertia
calculation. It does not perform interval LDL and does not certify inertia.

The central pencil is

```text
K(beta) =
[ P   N^T   0    0        ]
[ N    0    0    D        ]
[ 0    0    A   -B        ]
[ 0    D   -B^T -beta^2 W ]
```

with exact decimal `beta=0.045`. The positive-exponential archive supplies
`P/N/D/B/W`. The completed Gaussian contribution checkpoint supplies `A`
and its aggregated radii. Both artifacts have mesh fingerprint

`174d325adf2b1a7f6c70a023982060bc492dbb279d267e4cdc2a2a85e9270835`.

The Gaussian stiffness reconstruction has 105,129 nonzeros and reproduces
fingerprint

`de55b4a7910cec423f36b8760fc164e6154c7271227f739c84b0516d0fdf268f`.

## Complete interval pencil

The assembled center is exactly symmetric, has dimension 123,816 and exactly
798,384 nonzeros. The nonnegative symmetric radius matrix has 612,660
nonzeros. Before scaling,

`||K_center||_infinity = 9679.63547316441`

and

`||K_radius||_infinity = 9.76561535541853e-10`.

The graph has median degree 5 and maximum degree 19. Reverse
Cuthill-McKee reduces bandwidth from 77,386 to 1,137, confirming that the
large dimension comes from a sparse local graph rather than a dense block.

## Symmetric scaling

A positive diagonal congruence is essential because it preserves inertia
mathematically while balancing the block scales. The audit applies symmetric
Ruiz scaling for 2, 4, 6, 8, and 10 iterations.

At ten iterations the row-maximum range is
`[0.9921399059444368, 1.0000000000000007]`. The scaled norms are

`||S K_center S||_infinity = 3.9428572190040034`

and

`||S K_radius S||_infinity = 2.4788832576715813e-13`.

The selected scale fingerprint is

`bef73a763a5ee24b85651a3c761b940b0fde2f8e04294a7d7e9b0c085c7ca9c2`.

## Ordering comparison

`MMD_AT_PLUS_A` is the only viable ordering tested. Its unscaled and all five
scaled cases have equal row/column permutations, numerical
`U = D L^T` defects below `8e-13`, and the same central pivot counts:

`61,908 positive, 61,908 negative, 0 zero`.

The selected ten-step case has:

- 5,630,594 entries in `L+U`;
- fill ratio `7.052488526824185`;
- factor storage `69,548,192` bytes;
- central factor time about `1.24` seconds;
- minimum absolute U diagonal `0.00013584258942400673`;
- maximum absolute U diagonal `16943.13617423965`;
- numerical LDL relation defect `6.559176562535337e-13`; and
- deterministic solve backward error `3.2072043711054002e-12`.

Its fixed elimination fingerprints are:

- permutation:
  `fc613374bffd7bba84293e3c302e56d0ef945a0530443b04dde5ba079adb36db`;
- L/U symbolic pattern:
  `dc941205f68286d3f318a58670fd0ddf14bf63afdaf48202dcc9e0291238103b`;
- U diagonal:
  `c3cf0559f421039b208855c6ba2a3bab9d887d4a8459ffc71c76b381bdfbf140`.

In contrast, scaled `MMD_ATA` and `COLAMD` have fill ratios above 33,
different row and column permutations, minimum pivots at about
`5.55e-17`, LDL-relation defects above `0.33`, and deterministic solve
residuals above `0.02`. Their U signs are not inertia diagnostics and both
orderings are rejected.

## Interval and roundoff verdict

For the selected case, the floating inverse-one-norm estimate times the input
radius norm is

`2.2617286555267673e-7`.

This is encouraging evidence that the entry intervals themselves are small.
It is not a proof because `onenormest` is not a verified upper bound.

The deliberately conservative global factor-roundoff proxy is
`0.30838519365602146`, or

`2270.1657481915036`

times the smallest central pivot. Therefore the global norm route does not
close. More RAM or repeating binary64 factorization cannot repair that proof
gap.

## Decision

The full central factorization is resource-feasible and the target pivot
count is highly stable under the admissible MMD ordering and symmetric
scaling. The verified inertia flag nevertheless remains false.

The next bounded stage should freeze the ten-step scale, MMD permutation, and
symbolic pattern, then test one of two proof mechanisms on a pivot prefix:

1. componentwise directed sparse LDL with adaptive precision; or
2. a verified residual/inverse bound strong enough to prove nonsingularity of
   the entire interval segment from the center.

The first pilot must expose per-pivot radii and cancellation growth. It must
not launch an uncheckpointed 123,816-pivot arbitrary-precision run. Full
inertia, `kappa_h<0.045`, the global Ritz constant, continuum spectral
capture, and every Navier-Stokes regularity claim remain false.
