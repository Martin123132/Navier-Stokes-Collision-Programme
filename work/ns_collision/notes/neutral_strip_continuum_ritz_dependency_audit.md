# Continuum Ritz dependency audit

## Purpose

The complete stored finite-chain screen is below one, but it is not a
continuum estimate. This stage determines exactly what can be transferred
from the exact Gaussian-weighted P1 forms on the stored polygon and rejects
an invalid eigenvalue-direction shortcut.

The conclusion is fail-closed: the stored polygon P1 space is conforming and
a viable cutoff-resolvent transfer theorem now has explicit numerical
thresholds, but no continuum response flag is promoted.

## Continuum and Galerkin operators

Let `D_p` be the stored binary polygonal domain, let
`mu=exp(-x^2/2)`, and put

```text
H=L2(D_p,mu),  V=H0^1(D_p),
m(u,v)=int_Dp mu u v,
a(u,v)=int_Dp mu grad(u).grad(v).                  (1)
```

The directed assembly audit encloses the exact P1 forms on the reconstructed
mesh. The topology audit independently verifies:

```text
triangle components                                  1
boundary components                                  2
Euler characteristic                                 0
inner boundary edges                               112
triangles                                         30954
state degrees of freedom                          15211. (2)
```

Every incidence-one edge is a marked boundary segment, every boundary vertex
is removed from the state basis, and the rebuilt mesh fingerprint equals the
directed assembly checkpoint fingerprint. Hence the continuous P1 state
space `V_h` really is a subspace of `V` on the stored polygon.

Let `T` and `T_h` be the continuum and Galerkin solution operators:

```text
a(Tf,v)=m(f,v),       v in V,
a(T_h f,v_h)=m(f,v_h), v_h in V_h.                 (3)
```

Both are positive self-adjoint compact operators on `H`, with `T_h` extended
by zero on the orthogonal complement of `V_h`.

## The eigenvalue direction

Conformity and min-max give

```text
lambda_j(D_p) <= lambda_h,j.                        (4)
```

Thus a conforming finite-element eigenvalue is a continuum *upper* bound.
The certified exact-P1 omitted-mode lower endpoint

```text
lambda_h,241 >= 107.01775717228844                  (5)
```

cannot be substituted as a lower bound for `lambda_241(D_p)`.

The available Li-Yau bound after conjugating to
`-Delta+x^2/4-1/2` is

```text
lambda_241(D_p) >= 46.61644988218182.               (6)
```

The retained exact-P1 upper endpoint is `106.4162237645287`, so the
available analytic lower bound falls short by `59.79977388234688`. This does
not say the true continuum gap is absent; it says the current bound does not
prove one.

## Cutoff-resolvent route

The high-mode theorem already uses the continuum cutoff `Lambda=60`.
It is unnecessary to identify 240 individually labelled continuum
eigenvectors. It is enough to prove that every continuum mode below `60` is
captured by the 240-mode exact-P1 block.

For compact self-adjoint operators, Weyl's inequality gives

```text
mu_241(T) <= mu_241(T_h)+||T-T_h||,
mu_j=1/lambda_j.                                    (7)
```

Using (5), the inverse spectral separation at cutoff `60` is

```text
d = 1/60-1/107.01775717228844
  > 0.00732242299699141.                            (8)
```

Therefore

```text
||T-T_h||_(H->H) < d
  => lambda_241(D_p)>60.                            (9)
```

For `P=1_[1/60,infinity)(T)` and `Q` the first 240 exact-P1 modes, the
separated-spectrum residual theorem also gives

```text
||(I-Q)P|| <= ||T-T_h||/d.                         (10)
```

Let `R_h` be the energy projection. A weighted projection estimate

```text
||u-R_hu||_m <= C_h ||u-R_hu||_a                   (11)
```

implies by Galerkin orthogonality and duality that

```text
||T-T_h||_(H->H) <= C_h^2.                         (12)
```

Consequently the first missing numerical target is

```text
C_h < sqrt(d) = 0.08557115750643678.                (13)
```

This is substantially less demanding than proving a continuum lower bound
above `106.4162`. A weighted hypercircle or equilibrated-flux source-problem
bound is the natural way to compute `C_h` on this mildly nonconvex polygon.
Projection-based cluster estimates and hypercircle constants are developed
for the Laplacian by Liu and Vejchodsky, while the weighted form here still
requires a checked adaptation.

## Singular source and conormal output

Equation (9) is not the whole transfer.

A point mass is not in `H`, and point evaluation is not a bounded functional
on `H0^1` in two dimensions. Hence a time-zero expression such as a raw Ritz
projection of `delta_z` is not justified. Heat smoothing makes
`exp(-tau P)delta_z` an `H` state for every `tau>0`, and the existing killed
kernel diagonal majorant bounds its norm, but a positive-time semigroup
comparison with the nodal P1 source is still required.

Likewise, the conormal trace is not a bounded map from an arbitrary `H`
state to boundary `L2`. The existing Rellich estimate and half-time
factorization provide the correct smoothing mechanism. The remaining
certificate must propagate an equilibrated parabolic residual through that
mechanism. Elliptic reconstruction is a plausible architecture because it
supports residual-based parabolic estimates on nonconvex polyhedral domains,
but its hypotheses and constants must be specialized to the weighted point
source and conormal goal functional.

Finally, the exact common-circle Riesz and pushforward geometry compares
boundary measures only. The polygon-versus-circle domain semigroups still
require a separate perturbation theorem.

## Certified status

This stage certifies:

1. exact mesh identity and conformity of the stored polygon P1 space;
2. the correct one-sided Rayleigh-Ritz eigenvalue direction;
3. rejection of the exact-P1 complement lower-bound substitution;
4. the cutoff-resolvent rank and projector theorem;
5. the exact sufficient thresholds in (8) and (13);
6. the dependency order for the source, conormal, and domain gates.

It does not certify `C_h`, continuum spectral capture, the positive-time
point-source bridge, continuum conormal response, polygon-to-circle domain
transfer, a Navier-Stokes estimate, or a regularity proof.

The executable is
`scripts/neutral_strip_continuum_ritz_dependency_audit.py`.

## Literature anchors

- X. Liu and T. Vejchodsky, *Projection error-based guaranteed L2 error
  bounds for finite element approximations of Laplace eigenfunctions*,
  arXiv:2211.03218.
- A. Demlow, O. Lakkis, and C. Makridakis, *A posteriori error estimates in
  the maximum norm for parabolic problems*, arXiv:0711.3928.

These references support the architecture. They are not treated as direct
certificates for the weighted singular-source problem.
