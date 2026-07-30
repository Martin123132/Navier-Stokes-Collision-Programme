# Reversible finite-time semigroup certificate

## Purpose

The preceding reversible FEM audit replaced the fitted post-`T` decay, but
two empirical allowances remained: five percent on sampled time-window
maxima and one percent on scalar quadrature. This stage replaces both for the
three stored `rho=0`, spacing-`0.12` matrices.

The result is an upper enclosure for each symmetrized stored floating matrix.
It is not an enclosure of the continuum strip problem.

## Contractive uniformization

With lumped mass `M`, use the canonical stored symmetric generator

```text
S = sym(M^(1/2) G M^(-1/2)),       A=-S.             (1)
```

The previous 80-digit Barta lower bound gives `A >= lambda_1 I`. A
Gershgorin upper bound `L` encloses the other end of its spectrum. Choose

```text
nu >= max(max_i A_ii, L/2),   P=I+S/nu.              (2)
```

Then `P` is entrywise nonnegative and its spectrum lies in `[-1,1]`. Hence

```text
exp(hS)z = exp(-nu h) sum_(k>=0) (nu h)^k P^k z/k!. (3)
```

Every step truncates (3) only after a geometric upper bound on its Poisson
tail is below `1e-18`. Since `||P||_2<=1`, that scalar tail times `||z||_2`
is an operator-action remainder. Sparse products, coefficient recurrence,
and positive summation receive IEEE operation-count roundoff allowances;
errors are propagated with the Barta contraction `exp(-lambda_1 h)`.

An independent regression check compares (3) with SciPy's Krylov
`expm_multiply` at four entry angles and agrees inside the certified error
plus `2e-11`.

## Time-window maxima

Write the physical boundary vector as

```text
q(t)=C z(t),
C=D_Gamma^(-1/2) R^T M^(1/2).                        (4)
```

The same Gershgorin bound as before encloses `||C||`. On a slab `[a,a+h]`,
Taylor's formula and semigroup contraction give

```text
sup ||q(a+s)||
 <= sum_(j=0)^2 h^j ||C S^j z(a)||/j!
    + h^3 ||C|| ||S^3 z(a)||/3!.                    (5)
```

The propagated state error and sparse-product roundoff are added to (5).
The explicit axial factor is bounded without sampling by taking the
increasing `exp(t)` at the right endpoint and its decreasing Gaussian factors
at the left endpoint.

The interval `[0,0.005]` is handled separately because the axial `L2` factor
is singular at zero. Entry nodes are eight to eleven graph steps from the
inner boundary on the production meshes. Uniformization therefore has no
boundary term below that order. Fifty subintervals bound the product of the
Poisson onset with the axial singularity; the first uses the explicit
`t^(d-1/4)` majorant.

The later deterministic schedule has 590 slabs. All slab endpoints align
with 16 windows of width `3/8` on `[0,6]`. Taking the largest bound in each
window and then adding the previous Barta post-`T` geometric sum certifies the
complete interval factor.

## Scalar Darboux sum

Let `c` be total inner flux in mass coordinates and solve

```text
-S v=c.                                               (6)
```

The Barta lower bound converts the numerical residual in (6) into an `L2`
error bound for `v`. The unweighted absorption in a slab is exactly

```text
integral_a^b c^T z(t) dt = v^T[z(a)-z(b)].            (7)
```

The scalar axial weight is positive. On each slab its increasing exponential
is evaluated at `b` and its decreasing error-function factor at `a`.
Multiplying that upper endpoint product by the upper enclosure of (7) gives
an upper Darboux sum, not a stressed trapezoid rule. The existing spectral
tail pays the remaining interval `[6,infinity)`.

## Production results

At spacing `0.12`, with exactly the same `3/8` window and 591-slab schedule,

```text
X      states   max state error    certified response
4.20    3738       4.54e-12          0.619681476353
5.25    4804       4.66e-12          0.620016242618
6.30    5820       5.01e-12          0.620371178275.   (8)
```

For the worst entry angle at each width, the certified finite-window sum is
about `1.60%` above the dense node maximum sum. The certified scalar finite
parts are `0.603022413464`, `0.603392283101`, and `0.603634473005`. The
largest post-`T` payments remain below `3.76e-5` in the interval factor and
`4.93e-5` in scalar gain. The response spread is

```text
0.000689701921.                                      (9)
```

The certified values are lower than the old explicitly stressed values
because proved local enclosures replace blunt five-percent and one-percent
multipliers.

## Remaining gates

This stage closes the finite-time sampling and scalar-quadrature obligations
for the three symmetrized stored matrices. It does not:

1. remove the absorbing sides `|x|=X` analytically;
2. enclose polygonal-circle or weighted FEM consistency error;
3. extend the estimate uniformly over `0<=rho<=1`;
4. certify the wall-migration-child-return composite;
5. address the remaining Navier-Stokes residual and localization gates.

The next stage is an analytic `x`-exit correction. The calculation is
reproduced by
`scripts/neutral_strip_reversible_finite_time_certificate.py`.

**Later development.** The side-before-wall probability now has an
interval-enclosed Kummer expansion, and a positive renewal inequality pays
all later side excursions. At `X=6.3` the correction changes the stored
response by about `5.32e-8`. Continuum FEM consistency remains open. See
`neutral_strip_x_exit_correction.md`.
