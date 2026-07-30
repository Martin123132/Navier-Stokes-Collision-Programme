# Reversible spectral tail and x-width audit

## Purpose

The reversible boundary-FEM pilot produced a stable physical-face response,
but it still used a fitted terminal decay and one artificial `x` width. This
note removes the fitted tail for each stored finite matrix and measures the
response at three increasing widths.

It certifies the post-`T` tail of the floating finite-dimensional models. It
does not certify the finite-time maxima, remove the artificial boundary
analytically, or prove a continuum trace bound.

## Symmetric killed generator

Let `G` be the backward killed generator and `M=diag(m_i)` its positive
invariant lumped mass. Exact detailed balance gives

```text
S=M^(1/2) G M^(-1/2)=S^T.                           (1)
```

The tiny stored asymmetry is included by using the symmetric part of `S`.
For `A=-sym(S)`, the principal vector is positive. Barta's inequality gives

```text
min_i (A v)_i/v_i <= lambda_1 <= max_i (A v)_i/v_i. (2)
```

The ratios in (2) are accumulated at 80 decimal digits. They enclose the
principal decay of each stored floating matrix without relying on the
`eigsh` residual alone.

At spacing `0.12`, the results are

```text
X       Barta lower       Barta upper       width
4.20    2.362570516834     2.362570521559    4.72e-9
5.25    2.325345933743     2.325345934288    5.45e-10
6.30    2.324559286697     2.324559334353    4.77e-8.   (3)
```

## Boundary operator

For a forward probability column `p`, set `z=M^(-1/2)p`. If `R` maps live
states to the inner dual faces and `D_Gamma` contains their true arclengths,
then

```text
||h(t)||_L2(S1)
 =||D_Gamma^(-1/2) R^T M^(1/2) z(t)||_2.            (4)
```

The norm of the matrix in (4) is bounded by the square root of the maximum
absolute row sum of its positive Gram matrix. The resulting analytic upper
bounds are

```text
X=4.20: 77.688163116868,
X=5.25: 55.171527283710,
X=6.30: 58.745071683357.                            (5)
```

## Finite-patch tail

The earlier all-axial-space estimate leaves growth `exp(t/2)` at `rho=0`.
That is valid but unnecessarily weak for this return law, which is restricted
to `|z|<H`. With `V(t)=exp(2t)-1` and `H=3/4`, `erf(x)<=2x/sqrt(pi)` gives,
for `t>=T`,

```text
exp(t)||g_t 1_|z|<H||_2
 <=sqrt(H/pi)/sqrt(1-exp(-2T)).                     (6)
```

Combining (3)-(6),

```text
rho(t)<=A_T ||C|| exp[-lambda_1(t-T)] ||z(T)||.     (7)
```

For an interval width `ell`, the complete post-`T` sum is therefore at most

```text
A_T ||C|| ||z(T)||/[1-exp(-lambda_1 ell)].          (8)
```

The scalar branch mass gets a separate tail payment using

```text
exp(t) P(|Z_t|<H)
 <=sqrt(2/pi)H/sqrt(1-exp(-2T))                     (9)
```

and `||h||_1<=sqrt(2 pi)||h||_2`.

At `T=6`, the largest contribution of (8) to the optimized interval factor
is `3.758716458935e-5`. The largest scalar tail payment is
`4.929230751102e-5`. No fitted terminal slope is used.

## Width sweep

The explicit truncation branch and stressed response are

```text
X       states   truncation probability   response with spectral tail
4.20     3738     1.664244080768e-3        0.623248235706
5.25     4804     1.320720817810e-5        0.623596956316
6.30     5820     3.374752264798e-8        0.623957665810. (10)
```

The response spread is `0.000709430104`. This is controlled numerical width
dependence, not an analytic removal of the `x` boundary. The larger value
than the earlier `0.621054704444` also includes a one-percent scalar
quadrature stress, the five-percent finite-window stress, and the explicit
spectral tail payments.

## Remaining gates

This stage closes the fitted-tail obligation for each stored FEM matrix. The
following remain open:

1. bound paths lost through `|x|=X` analytically;
2. enclose polygonal-circle and weighted FEM continuum errors;
3. extend uniformly over `0<=rho<=1`;
4. rebuild the composite wall branch only after the return constant is a
   continuum upper bound.

The calculation is reproduced by
`scripts/neutral_strip_reversible_spectral_tail_width_audit.py`.

**Later development.** Contractive uniformization, Taylor slab bounds, and
an upper scalar Darboux sum now close the first two items for the symmetrized
stored matrices. See `neutral_strip_reversible_finite_time_certificate.md`.
