# Leray-compatible mollified cell frame

## Purpose

The stopping-time moving-cylinder identity assumes absolutely continuous
centres and frames. Pointwise velocity trajectories and time derivatives of
strain eigenvectors are not available for a general Leray solution. This
note replaces both by fixed-scale smooth local averages.

The construction is Galilean covariant, preserves orthogonality, removes the
mean local spin, and uses no eigenvalue-gap denominator during a visit.

## Mollified affine data

Let `rho` be a fixed smooth, nonnegative, radial mollifier of integral one and

```text
rho_L(x)=L^(-3)rho(x/L).                               (1)
```

For a divergence-free velocity define

```text
U_L(c,t)=integral rho_L(x-c)u(x,t)dx,

A_L(c,t)=integral rho_L(x-c)grad u(x,t)dx
        =-integral u(x,t) tensor grad rho_L(x-c)dx,    (2)

S_L=(A_L+A_L^T)/2,
W_L=(A_L-A_L^T)/2.                                    (3)
```

The distributional form in (2) is the useful one at Leray regularity. Since
`div u=0`,

```text
trace A_L=trace S_L=0.                                (4)
```

No projection back onto trace-free matrices is needed.

## Centre and spin-following frame

Solve the coupled ordinary differential equations

```text
c'=U_L(c,t),
O'=W_L(c,t)O,       O(tau) in SO(3).                  (5)
```

For fixed smooth `rho_L`, convolution makes `U_L` and `W_L` continuous and
locally Lipschitz in `c`. Their bounds use `||u(t)||_2` and fixed derivatives
of the kernel. For a Leray solution these coefficients are measurable in time
with integrable bounds, so the Caratheodory ODE gives an absolutely continuous
centre and frame.

Skewness gives the exact identity

```text
d(O^T O)/dt=O^T(W_L^T+W_L)O=0.                       (6)
```

Hence an initial proper rotation remains in `SO(3)`.

At a buffered entry time `tau`, diagonalize the single symmetric trace-free
matrix `S_L(c(tau),tau)` and use that orientation to initialize the tapered
affine shell. During the visit, evolve `O` by (5). Do not continuously
diagonalize `S_L` and do not divide by an eigenvalue gap. Repeated entry
eigenvalues merely leave freedom in choosing the initial basis.

## Residual

In the moving frame define

```text
e=u-U_L-W_L(x-c)-b_ref.                               (7)
```

Every subtracted field is divergence-free. For a radial mollifier,
`integral rho_L(x-c)(x-c)dx=0`, so

```text
integral rho_L e dx=0.                                (8)
```

The weighted mean skew part of `grad u-W_L` also vanishes. Thus (7) begins
with nonlinear affine variation and drift mismatch, not arbitrary local
translation or mean vorticity.

Under a Galilean change `u->u+V(t)`,

```text
U_L->U_L+V,
A_L, S_L, W_L unchanged,
e unchanged.                                          (9)
```

This supplies the exact invariance that the fixed-centre potential estimate
lacked.

## Relation to the affine spectrum

At entry, order the eigenvalues of `S_L` as

```text
lambda_1<=lambda_2<=lambda_3,
lambda_1+lambda_2+lambda_3=0.                         (10)
```

When `lambda_3>0`, normalization by `lambda_3` gives

```text
(lambda_1,lambda_2,lambda_3)/lambda_3
 =(-1-t,t,1),       -1/2<=t<=1.                       (11)
```

This is exactly the spectrum interval already covered by the divergence-free
shell pilot. The level rule chooses `L` so that the normalized amplitude lies
under the working Reynolds envelope.

## What this closes

The geometric objects in the stopping-time visit can now be defined without
pointwise values:

1. local translation is the smooth mean velocity `U_L`;
2. frame spin is the smooth mean antisymmetric gradient `W_L`;
3. the entry affine strain is the symmetric trace-free `S_L`;
4. pressure-driven rotation of the strain eigenvectors is not separately
   differentiated; its effect appears as growth of the measurable residual
   (7), which can trigger the declared coherence-abort rule.

## Remaining gates

This does not prove that the residual satisfies the small sector budget.
The next tasks are:

1. relate the entry tensor and envelope-selected `L` to the tapered reference
   with explicit `L^3` and `L^(3/2)` residuals;
2. show that a failure of those residual thresholds is paid by the split or
   bad-occupation bounds in `coherence_abort_renewal.md`;
3. write the stopped construction for smooth approximants and pass it stably
   to Leray limits, including measurability of entry choices.

The convolution identities, Galilean test, trace-free spectrum, spin removal,
and `SO(3)` preservation are reproduced by
`scripts/leray_mollified_cell_frame_audit.py`.
