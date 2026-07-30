# Trajectory-Level Collision Defect

Date: 2026-07-17

Status: exact Navier-Stokes Taylor coefficients and exact leading weakly
nonlinear Duhamel response for the two-mode sign-indefinite family. These
results isolate, but do not solve, the high-Reynolds cumulative problem.

## 1. Setup

Use the two independent Fourier modes

```text
k=K*(1,0,0),       m=K*(1,1,0),
u_k=A*(0,-1,1),    u_m=A*(-1,1,sigma),
```

with conjugate coefficients at the negative modes. Here `sigma=1` is the
positive quartic-transfer companion and `sigma=-1` is the exact negative
counterexample. Define

```text
x=exp(-s*K^2),
p=x^3+2*x^2+3*x,
R=A/(nu*K).
```

The Fourier normalization supplies only a common positive volume factor and
does not affect any sign statement below.

## 2. Exact Navier-Stokes Jet

Write the de-aliased Fourier solution as

```text
u(t)=u_0+t*u_1+t^2*u_2+t^3*u_3+...
```

and generate its coefficients recursively from

```text
(n+1)*u_(n+1)
 =nu*Delta*u_n-sum_(i+j=n) P[(u_i dot grad)u_j].  (2.1)
```

For the positive companion, the defect begins

```text
D_s(t)
 =A^4*K^4*c_1*t
  -nu*A^4*K^6*c_2*t^2
  +(A^6*K^6*c_3E+nu^2*A^4*K^8*c_3nu)*t^3
  +O(t^4),                                        (2.2)
```

where

```text
c_1=2*(1-x)^2*(29*p+66)/5,
c_2=2*(1-x)^2*(203*p+472)/5.                     (2.3)
```

Every coefficient of the degree-eleven polynomial `c_3E/(1-x)^2` is
strictly positive, and

```text
c_3nu=(1-x)^2*(4292*p+10128)/15>0.               (2.4)
```

Two nontrivial cancellations occur:

```text
the pure Euler A^5*K^5 term at order t^2 is zero;
the mixed nu*A^5*K^7 term at order t^3 is zero.  (2.5)
```

The discriminant of the quadratic `D_s(t)/t` remains negative even if the
positive Euler part of `c_3` is omitted:

```text
-4/75*(125309*p^2+579072*p+668544)<0.             (2.6)
```

Thus the third-order Taylor truncation is positive for every `t>0`. Section 3
shows why this local fact does not determine the eventual sign.

## 3. Resummed Weakly Nonlinear Response

Expand in amplitude rather than time:

```text
u=A*u^(1)+A^2*u^(2)+O(A^3).
```

The linear field `u^(1)` is the exact heat evolution. The quadratic field
`u^(2)` is obtained by one exact Duhamel integration. Put

```text
z=exp(-nu*K^2*t).
```

For the positive companion, the complete order-`A^4` defect is

```text
D_s^(4)(t)
 =A^4*K^2/(5*nu)*(1-x)^2*z^4*(1-z^2)
  *[z^2*(29*p+71)-5].                             (3.1)
```

It starts positive and crosses zero exactly once at

```text
z_*^2=5/(29*p+71),
nu*K^2*t_*=(1/2)*log((29*p+71)/5).                (3.2)
```

Viscosity therefore reverses the instantaneous leading defect, a feature no
finite short-time truncation above detected. It does not cancel the signed
time integral:

```text
integral_0^infinity D_s^(4)(t)dt
 =A^4*(1-x)^2*(29*p+61)/(120*nu^2)>0.             (3.3)
```

For the negative companion,

```text
D_s^(4)(t)
 =A^4*K^2/(5*nu)*(1-x)^2*z^4*(1-z^2)
  *[z^2*(p-1)-5]<0                                (3.4)
```

for every `t>0`, and

```text
integral_0^infinity D_s^(4)(t)dt
 =A^4*(1-x)^2*(p-11)/(120*nu^2)<0.                (3.5)
```

The integrated signs reproduce the initial quartic-transfer signs. The late
negative lobe in (3.1) is not large enough to erase the early positive lobe.

## 4. Exact Perturbative Absorption Scale

For the linear heat field,

```text
integral_0^infinity Q(t)dt=8*A^2*K^2/nu.          (4.1)
```

Combining (3.3) and (4.1) gives

```text
[integral D_s^(4)dt]/[nu*integral Qdt]
 =R^2*(1-x)^2*(29*p+61)/960.                     (4.2)
```

Hence viscosity absorbs the positive channel perturbatively when `R` is
small. This is consistent with standard small-data regularity and is not a
new global theorem.

Equation (4.2) also identifies the exact limitation. The weak expansion is
not uniform when `R` is order one or larger, precisely where the ratio ceases
to be small. It cannot decide the high-Reynolds regime by extrapolation.

## 5. Current Gate

These calculations rule out two misleading conclusions:

```text
1. The negative t^2 viscous curvature does not establish eventual control.
2. The eventual pointwise sign reversal does not establish cumulative
   cancellation.
```

The surviving question is now explicitly nonperturbative:

```text
Can the signed helical interference terms be controlled along a full
high-R Navier-Stokes trajectory by viscosity, endpoint motion of J_s,
or transfer between successive receiving shells?
```

The next empirical audit should use a de-aliased Galerkin solver with exact
energy and primitive-identity residual checks. Its purpose is to identify a
candidate trajectory inequality, not to substitute finite computation for a
proof.
