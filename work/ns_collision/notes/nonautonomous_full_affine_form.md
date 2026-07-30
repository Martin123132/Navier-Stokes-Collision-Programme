# Nonautonomous full-affine form gate

## Purpose

The fixed entry strain makes constant-spectrum orientation motion consume
the drift budget even though the amplitude envelope does not move. The
natural repair is to update the complete affine reference continuously,
without continuously changing the cubic cell label.

This removes the temporal errors exactly and retains a strong uniform
interior form floor. It does not yet prove the required boundary visit norm.

## Instantaneous affine reference

Use the mollified coefficients

```text
A_L(t)=S_L(t)+W_L(t),
U_L(t)=int rho_(kL)b(t).                              (1)
```

Let the centre follow `U_L` and let the orthogonal frame remove `W_L`. In
that spin-following frame, retain the complete measurable symmetric matrix
`S_L(t)` as the baseline; do not diagonalize it. Define

```text
b_ref(t,x)=U_L+W_L(x-c)+S_L(t)(x-c),
lambda_ref(t)=lambda_max(S_L(t)).                     (2)
```

Then the physical errors reduce exactly to

```text
e=R_L=b-U_L-A_L(x-c),

q_+=[lambda_max(S(x,t))-lambda_max(S_L(t))]_+.        (3)
```

The temporal quantities `T` and `G` in the fixed-reference gate are both
zero. No strain eigenvector or eigenvector derivative is used. In
particular, the constant-spectrum rotation example has zero reference error
under (2), rather than matrix error `2|sin(theta)|`.

## Uniform interior coercivity

Normalize the current amplitude level by

```text
lambda_max(S_L)L^2/nu<=1.                             (4)
```

On the compact cylinder

```text
D={r<2, |z|<0.75},                                    (5)
```

let `v` have zero trace on the complete boundary. For every measurable
symmetric trace-free matrix `S_L(t)`, integration by parts gives

```text
int_D v S_L(t)y dot grad(v)=0.                        (6)
```

Indeed the affine drift is divergence-free and the boundary term vanishes.
Thus the real form of the nonautonomous reference operator is

```text
Re a_t[v]
 =||grad v||_2^2-lambda_max(S_L(t))||v||_2^2.         (7)
```

The exact cylinder Poincare floor is

```text
lambda_1(D)
 =j_(0,1)^2/2^2+pi^2/(4(0.75)^2)
 =5.83228730.                                         (8)
```

Combining (4), (7), and (8) gives the orientation-independent bound

```text
Re a_t[v]>=4.83228730 ||v||_2^2.                      (9)
```

This is a rigorous homogeneous interior estimate for arbitrary measurable
orientation changes. It also implies that the normalized trace-free strain
has operator norm at most two: if its ordered top eigenvalue is one, its
bottom eigenvalue cannot be below minus two.

## Reduced physical gate

With (3), the compact sector condition loses both temporal charges:

```text
sqrt(S_3) C_3+(1+d)S_3 P<d,
d=0.130483946925.                                    (10)
```

The one-error thresholds are

```text
C_3<0.30540,
P<0.63228.                                            (11)
```

This is materially less restrictive than aborting when the fitted strain
rotates. Spatial non-affinity remains a genuine error and is not hidden in
the baseline.

## Remaining boundary theorem

The existing compact visit norm `0.55681307` was computed for a static
affine matrix by an elliptic weighted Poisson solve. It cannot simply be
inserted into the nonautonomous problem. The needed object is now a causal
outer-to-inner parabolic boundary operator, uniform over measurable
`S_L(t)` satisfying (4).

Homogeneous interior coercivity (9) is necessary but does not alone give the
calibrated boundary trace norm. A proof must provide one of:

1. a nonautonomous boundary-control estimate with norm below the renewal
   threshold;
2. a comparison theorem showing that the worst coefficient history is a
   static affine endpoint already audited;
3. accumulated same-level decay sufficient to pay coefficient changes.

This is now a narrower and better-posed gate than a probability estimate
from Leray energy. The decomposition, cylinder floor, reduced thresholds,
and constant-spectrum repair are reproduced by
`scripts/nonautonomous_full_affine_form_audit.py`.
