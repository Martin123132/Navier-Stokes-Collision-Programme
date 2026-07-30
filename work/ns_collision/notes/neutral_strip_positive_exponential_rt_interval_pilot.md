# Positive-exponential RT0/P0 interval pilot

## Scope

This stage implements the missing local entry enclosures for the weighted
hypercircle pencil:

- P0 source mass `W_T=integral_T exp(+x^2/2)`;
- RT0 flux mass `P_T=integral_T exp(+x^2/2) psi_i.psi_j`;
- triangle area `D_T`; and
- P1-P0 load entries `B_T=D_T/3`.

The expensive analytic enclosure is tested on 512 deterministically spread
triangles. The fixed-`beta` geometry budget is evaluated on all 30,954
triangles. Complete-mesh `P/W/D/B` assembly and full-pencil inertia remain
open.

## Analytic enclosure

For `F(x)=exp(+x^2/2)`, Taylor coefficients about a binary center `c` obey

`(n+1)a_(n+1)=c a_n+a_(n-1)`.

The order-`n` derivative divided by `n!` is bounded by

`exp(max|x|^2/2) *
 sum_(k=0..floor(n/2)) max|x|^(n-2k)/(2^k k! (n-2k)!)`.

Multiplication by `max|x-c|^n` gives the directed Taylor remainder used by
the established simplex-moment engine. Degree 22 is used.

On a triangle with determinant magnitude `d`, the local outward RT0 basis is

`psi_i(X)=(X-v_i)/d`.

Writing `X=sum_a lambda_a v_a` reduces every RT0 entry to a directed linear
combination of the six barycentric quadratic moments

`integral_T F(x) lambda_a lambda_b`,

divided by the directed interval for `d^2`. No interval numerical quadrature
is used in the proof enclosure.

## Distributed pilot

The sample consists of 512 unique integer-linspace indices over the stored
triangle order, including indices 0 and 30,953. Its index SHA-256 is

`0fe8d7d58396bc0a3bd02606d0d2c560bbd4cf116fce5d8b6f39dec162125c50`.

All 8,192 checks pass:

- every q12 and independent q18 P0 source mass is enclosed;
- every q12 and q18 symmetric RT0 entry is enclosed;
- every binary area/load diagnostic agrees after its explicit arithmetic
  roundoff guard; and
- all sampled P0 masses and RT0 diagonal entries have positive lower bounds.

The maximum interval widths are:

- area: `3.686287386450715e-18`;
- load: `1.5178830414797062e-18`;
- P0 source mass: `1.3997691894473974e-12`; and
- RT0 mass entry: `4.042703949380666e-10`.

The largest q12/q18 differences are `5.329070518200751e-15` for source mass
and `1.659827830735594e-11` for RT0 mass. These quadratures are independent
diagnostics, not proof inputs.

## Directed geometry budget

All 30,954 triangles are checked with directed edge lengths, weight
oscillation, exponential, and pi intervals. The exact decimal threshold is
enclosed as

`beta=0.045 in [0.04499999999999999,0.04500000000000001]`.

The resulting intervals are

`alpha in
[1.1696923667355057,1.1696923667355128]`,

`C_data in
[0.030354018922231274,0.030354018922231562]`,

and

`C_data+alpha*beta in
[0.08299017542532901,0.08299017542532967]`.

Against the directed strict target lower bound
`0.08557115750643675`, the certified geometry headroom is at least

`0.0025809820811070745`.

Thus, once a separate inertia certificate proves `kappa_h<0.045`, the
geometry part of the continuum Ritz gate is already sufficient.

## Remaining obligation

The same local analytic enclosure must now run over every triangle with:

1. atomic resumable checkpoints;
2. globally oriented RT0 assembly;
3. aggregated sparse entry-error bounds;
4. exact integer divergence topology;
5. reuse of the existing directed P1 stiffness intervals; and
6. a final hash-bound `P/W/D/B` checkpoint.

Only after that complete assembly passes should the 123,816-dimensional
directed sparse LDL be attempted.

The complete-mesh matrix-entry flag, full inertia, `kappa_h` upper bound,
global Ritz constant, continuum spectral capture, point-source bridge,
conormal output, polygon-to-circle transfer, and Navier-Stokes regularity
claim all remain false.
