# Projected interval and two-block transfer

## Purpose

The indexed-spectrum stage certified the first 241 exact-polygon finite
element eigenvalue intervals but left the common-circle Riesz algebra,
reduced generator, source metric, endpoint propagation, and off-block
leakage open. This stage closes the finite projected algebra on the frozen
240-dimensional trial space and derives the damped two-block theorem needed
for leakage.

It does not substitute the exact-reference complement floor into the
modified-chain leakage problem.

## Exact boundary geometry

For the regular `N=112` inner polygon, the trace mass, pushed P1 Gram, and
modified/reference cross Gram are real symmetric circulants. If
`alpha=pi/N`, `ell=2 sin(alpha)`, `a=2 alpha`, `q=tan(alpha)`, and
`c=cos(alpha)`, their nonzero coefficients are

```text
M_Gamma: diag 2 ell/3, neighbors ell/6,
P_Gamma: diag 2 p_d, neighbors p_o,
H:       diag (ell/a) 3/4, neighbors (ell/a) 1/8,

p_d=c^2(2q/3+4q^3/15),
p_o=c^2(q/3+q^3/15).
```

The checker evaluates these quantities with 100-digit interval
transcendentals. The exact trace lower eigenvalue and pushed-Gram upper
eigenvalue are therefore explicit. In particular,

```text
||P_Gamma^(1/2) M_Gamma^(-1)||_2
 <= sqrt(lambda_max(P_Gamma))/lambda_min(M_Gamma).
```

Every stored geometry coefficient is compared with its exact interval, and
the circulant templates are checked entry by entry. This certifies the
finite Riesz, Gram, push, and cross geometry rather than treating the stored
binary arrays as exact.

## Reduced exact-form generator

Let `V` be the cached 240-column stored-pencil basis and `Lambda` its frozen
diagonal. On the exact Gaussian-weighted polygon forms put

```text
G=V^T M_e V,                 K=V^T A_e V,
L_e=G^(-1)K.
```

The assembly audit gives

```text
|m_e-m_s| <= eta_M m_s,
|a_e-a_s| <= eta_A m_s.
```

Combining these bounds with the directed stored Gram defect and block
residual gives

```text
||G-I|| <= gamma_e,
||M_e^(-1/2)(A_e V-M_e V Lambda)|| <= epsilon_e,
||L_e-Lambda|| <= sqrt(1+gamma_e) epsilon_e/(1-gamma_e).
```

No individual eigenvector perturbation is used. The reduced semigroup
comparison follows directly from Duhamel, with the `G`-condition factor.
The exact source coefficients are `G^(-1)V^T e_z`, so the same Gram bound
also encloses the source-metric correction.

The exact boundary load map on this trial space is

```text
C_e=B_e^T V+B_(m,e)^T V L_e.
```

Certified Frobenius assembly errors for both boundary couplings, the reduced
generator bound, and the exact common-circle Riesz factor enclose
`C_e-C_s`. The checker propagates this, the semigroup difference, the source
metric, and the small exact/stored boundary-geometry difference through all
15 later windows and a geometric post-`6` tail.

## Low projector

Normalize the trial basis in exact mass:

```text
W=M_e^(1/2)V G^(-1/2).
```

The off-subspace residual is at most

```text
||(I-WW^T)T_e W|| <= epsilon_e/sqrt(1-gamma_e).
```

The indexed exact complement begins at
`107.01775717228844`. The exact reduced Ritz maximum is bounded by the
frozen retained maximum plus `||L_e-Lambda||`, leaving a gap above `0.6`.
The standard residual subspace theorem then bounds the sine of the angle
between `W` and the exact first-240 spectral subspace. This certifies the
subspace projector, not 240 individually labeled eigenvectors.

The resulting projector source-state error is recorded, but its boundary
trace composition remains a separate gate.

## Damped two-block theorem

Let a nonnegative self-adjoint operator have orthogonal block form

```text
H=[[A,B^*],[B,C]],       A>=alpha I, C>=beta I,
||B||<=epsilon.
```

For a solution starting in the low block, the component norms are dominated
by the positive two-dimensional system

```text
d/dt [x;y] <= [[-alpha,epsilon],
               [epsilon,-beta]] [x;y].
```

Writing

```text
m=(alpha+beta)/2,
d=(beta-alpha)/2,
s=sqrt(d^2+epsilon^2),
```

gives the explicit high-component bound

```text
y(t) <= epsilon exp(-m t) sinh(s t)/s.
```

The low feedback is the first component of the same exponential minus
`exp(-alpha t)`. This retains complementary damping and strictly improves
the gap-free `t epsilon` estimate when `beta` is useful.

The theorem is implemented and regression-tested. It is not yet charged to
the production screen: the certified `107.01775717228844` floor belongs to
the exact reference polygon operator, while the measured coupling
`6.343703098841749` belongs to the modified chain. A certified modified
high-block floor or a valid form transfer between those two operators is
still required.

## Scope

This stage certifies:

1. exact common-circle Riesz/Gram/push/cross geometry;
2. the exact-form reduced reference generator on the frozen trial space;
3. the source metric and its endpoint/time-slab propagation;
4. the exact polygon first-240 subspace projector angle;
5. the abstract damped two-block comparison theorem.

It does not certify modified-chain off-block leakage, the boundary trace of
the projector source mismatch, continuum Ritz transfer, polygon-to-circle
domain perturbation, or the full Navier-Stokes composition.

The executable is
`scripts/neutral_strip_projected_interval_two_block_transfer.py`.

## Validation

A fresh below-normal-priority deterministic replay completed all 16 proof
predicates in `62.32359589997213` seconds. The atomic output writer completed,
all premise hashes matched, and the two independent focused tests passed.
