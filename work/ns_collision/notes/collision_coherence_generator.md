# Collision-Coherence Generator

Date: 2026-07-17

Status: exact smooth-coefficient generator combining independent-replica
separation and deformation growth in one positive observable. It gives a
strict local collision-zone damping and a precise occupation-time target,
but the Leray energy estimate alone does not control that target uniformly.

## 1. Separation and Deformation State

Run two independent backward histories from a common observation endpoint:

```text
dQ_i=-u(Q_i,t-tau)d tau+sqrt(2*nu)dW_i,
R=Q_1-Q_2,
g=|R|,
theta=R/g.                                        (1.1)
```

Put

```text
delta u=u(Q_1)-u(Q_2),
sigma=theta dot delta u/g,
b=(I-theta tensor theta)delta u/g.                (1.2)
```

The relative generator is

```text
G_R=2*nu*[partial_gg+(2/g)partial_g
          +(1/g^2)Delta_S2]
    -g*sigma*partial_g-b dot grad_S2.             (1.3)
```

This contains both the radial Bessel drift and angular diffusion. In SDE
form the radial part is

```text
dg=(4*nu/g-g*sigma)d tau+2*sqrt(nu)d beta.        (1.4)
```

Along each history let a deformation vector satisfy

```text
z_i'=A_i z_i,
A_i=grad u(Q_i,t-tau).                            (1.5)
```

Writing `a_i=|z_i|`, `n_i=z_i/a_i`, and `S_i=(A_i+A_i^T)/2` gives

```text
(log a_i)'=lambda_i,
lambda_i=n_i dot S_i n_i,
n_i'=(I-n_i tensor n_i)A_i n_i.                  (1.6)
```

Thus separation orientation, vorticity orientation, and deformation growth
belong to one Markov state. Formula (1.6) is unchanged at the norm level if
the chosen backward convention uses `A_i^T` instead.

## 2. Positive Collision-Coherence Weight

For `epsilon>0` and `0<q<=1`, define

```text
w_(q,epsilon)(g)
 =[epsilon/(g^2+epsilon)]^(q/2),
V_(q,epsilon)=w_(q,epsilon)(g)*a_1*a_2.           (2.1)
```

The weight lies in `(0,1]` and equals one when the histories coincide. It
measures the part of the deformation correlation carried by histories that
remain within distance comparable to `sqrt(epsilon)`.

Direct radial differentiation gives

```text
(Delta_R w)/w
 =q*[(q-1)g^2-3*epsilon]/(g^2+epsilon)^2.         (2.2)
```

Combining (1.3), (1.6), and (2.2) yields the exact logarithmic generator

```text
G V/V=Gamma_(q,epsilon),                          (2.3)

Gamma_(q,epsilon)
 =lambda_1+lambda_2
  +q*sigma*g^2/(g^2+epsilon)
  -2*nu*q*[(1-q)g^2+3*epsilon]
      /(g^2+epsilon)^2.                           (2.4)
```

No stretching term has been expanded around heat flow. The Bessel boundary
effect and both deformation rates occur in the same scalar drift.

## 3. Strict Collision-Zone Damping

At collision, the longitudinal increment term is multiplied by `g^2` and
drops out:

```text
Gamma_(q,epsilon)(g=0)
 =lambda_1+lambda_2-6*nu*q/epsilon.               (3.1)
```

Hence viscosity dominates every finite smooth stretching rate sufficiently
near the collision diagonal.

For `0<q<1`, the damping remains strict away from collision:

```text
D_(q,epsilon)
 =2*nu*q*[(1-q)g^2+3*epsilon]
    /(g^2+epsilon)^2>0.                           (3.2)
```

For `q=1`,

```text
D_(1,epsilon)=6*nu*epsilon/(g^2+epsilon)^2.       (3.3)
```

This tends to zero at every fixed nonzero gap but remains concentrated at
the collision boundary. It is the regularized strict-local-martingale defect
of the Newtonian reciprocal gap. Values `0<q<1` add a genuine bulk damping
to that boundary loss.

## 4. Optimal Fractional Exponent

Write

```text
chi=g^2/epsilon.                                  (4.1)
```

Apart from the factor `2*nu/epsilon`, the damping is

```text
d_q(chi)=q*[(1-q)chi+3]/(1+chi)^2.                (4.2)
```

Its optimizer is

```text
q=1,                              0<=chi<=3,
q=(chi+3)/(2*chi),                chi>3.          (4.3)
```

Thus the Newtonian exponent is optimal in the innermost collision layer,
while the best far-field exponent tends to `q=1/2`. The optimized far-field
damping is asymptotic to

```text
nu/(2*g^2).                                       (4.4)
```

This gives a principled multi-exponent envelope rather than an arbitrary
choice of inverse-gap power.

## 5. Stopped Supermartingale Criterion

Let `tau_safe` be the first time at which

```text
lambda_1+lambda_2
 +q*sigma*g^2/(g^2+epsilon)
 >D_(q,epsilon).                                  (5.1)
```

Up to `tau_safe`, the positive process `V_(q,epsilon)` is a local
supermartingale. More generally, localization and Ito's formula give the
exact target

```text
E V(T)
 <=V(0)+E integral_0^T
       [Gamma_(q,epsilon)]_+ V d tau.             (5.2)
```

For two Cauchy histories,

```text
|E[Z_1 dot Z_2]|
 <=E[a_1*a_2]
 =E[w*a_1*a_2]+E[(1-w)*a_1*a_2].                 (5.3)
```

Equation (5.2) targets the first, common-history term. The second term is the
separated-history decorrelation problem. This is an exact implementation of
the common-residence/separated-history split proposed earlier.

The missing theorem is now specific:

```text
Bound the positive occupation integral in (5.2), and the separated term in
(5.3), using energy-class data without inserting an L^1_t L^infinity_x
strain bound.                                     (5.4)
```

## 6. Relation to the Degree-Two Strain Kernel

For a spherical harmonic `Y_l` and a singular radial power,

```text
2*nu*Delta_R[g^(-p)Y_l]
 =2*nu*g^(-p-2)
   *[p(p-1)-l(l+1)]Y_l.                           (6.1)
```

The Newtonian strain channel has `p=3,l=2`, so the coefficient vanishes.
Softening to `p=3-delta` gives

```text
p(p-1)-6=-delta*(5-delta)<0,
0<delta<5.                                        (6.2)
```

This recovers strict angular-radial damping but leaves a signed tensor. The
positive scalar weight (2.1) is the norm-level counterpart that retains the
boundary defect without claiming a sign for the strain kernel itself.

## 7. Affine Stress Test

Take the uniform strain

```text
S=diag(-a,-a,2a),       a>0,                      (7.1)
```

and align both deformation vectors with the stretching direction. Then

```text
lambda_1+lambda_2=4a.                             (7.2)
```

At collision, (3.1) is nonpositive when

```text
epsilon<=3*nu*q/(2*a).                            (7.3)
```

On the shell `g^2=epsilon`, even the worst longitudinal strain is controlled
only when

```text
epsilon
 <=nu*q*(4-q)/[2*a*(4+q)].                       (7.4)
```

The local collision layer is therefore real. It is not a global affine
bound. The transverse replica variance grows like `exp(2*a*tau)`, so for
`0<q<1`

```text
E[w_(q,epsilon)] is asymptotic to exp(-q*a*tau),
a_1*a_2=exp(4*a*tau),
E[V_(q,epsilon)] grows like exp((4-q)*a*tau).     (7.5)
```

Radial separation does not defeat spatially uniform stretching. This affine
field has infinite energy and is not periodic, so it remains a local stress
test rather than a Clay-admissible counterexample. It proves that global
control must use finite-energy localization and exit, not only the pointwise
collision drift.

## 8. Why Leray Energy Alone Does Not Close (5.2)

The energy inequality gives strain at the level

```text
S in L^2_t L^2_x.                                 (8.1)
```

For a three-dimensional heat kernel,

```text
||p_t||_2 is proportional to t^(-3/4).            (8.2)
```

Applying spatial and temporal Cauchy-Schwarz to a uniform endpoint
occupation estimate would require

```text
integral_0^T t^(-3/2)dt<infinity,                 (8.3)
```

which is false. Equivalently, the parabolic scaling index of
`L^2_t L^2_x` strain is

```text
2/2+3/2=5/2>2.                                   (8.4)
```

Thus a naive heat-kernel/Khasminskii estimate cannot upgrade Leray energy to
uniform common-residence control. The collision killing in (3.2), angular
cancellation, or a capacity estimate must be used essentially.

## 9. Outcome and Next Gate

Established:

```text
1. exact joint separation-orientation-deformation generator;
2. positive collision-coherence observable;
3. strict Newtonian boundary damping and fractional bulk damping;
4. optimized near/far inverse-gap exponent;
5. exact stopped-supermartingale occupation criterion;
6. affine and energy-class obstructions to naive global promotion.
```

Open:

```text
1. control of the unsafe occupation integral in (5.2);
2. control or cancellation of the separated term in (5.3);
3. justification for rough Leray drifts and deformation histories.
```

The next calculation should turn (5.2) into a killed two-particle Poisson
problem and stress-test it on a localized finite-energy strain tube. That is
the smallest model capable of deciding whether viscous exit plus collision
killing can beat coherent stretching without relying on the inadmissible
uniform affine field.
