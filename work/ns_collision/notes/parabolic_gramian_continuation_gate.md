# Parabolic Gramian Continuation Gate

Date: 2026-07-25

Status: the local restart algebra and a hierarchy of sufficient critical
`L^3` continuation moments are proved in the classical regime. No
unconditional bound for those moments is proved.

Nothing here proves global regularity.

## 1. Local Windows

Let `X_{s,r}(a)` be one common-noise stochastic trajectory beginning at
label `a` and time `s`. Condition on that common path and put

```text
A(r)=grad u(X_{s,r}(a),r),
Phi(r,q)=the fundamental matrix of A,
J_st=Phi(t,s).
```

The tangent auxiliary noise from the `rho->1` replica limit defines

```text
F_st=4nu integral_s^t Phi(t,r)Phi(t,r)^T dr,             (1.1)
B_st=4nu integral_s^t Phi(s,r)Phi(s,r)^T dr,             (1.2)
H_st=4nu integral_s^t Phi(t,r)Phi(s,r)^T dr.             (1.3)
```

For `tau=t-s`, normalize the radial variances by

```text
f_st=tr(F_st)/(4nu tau),
b_st=tr(B_st)/(4nu tau).                                (1.4)
```

These quantities are dimensionless under Navier-Stokes parabolic scaling.
The expectation over the common path is not included in (1.1)-(1.4);
`F`, `B`, and `H` are already conditional covariances over the auxiliary
antisymmetric tangent noise. The continuation criterion below subsequently
averages over the common path.

## 2. Exact Restart Laws

For `s<m<t`, the fundamental-matrix cocycle gives

```text
J_st=J_mt J_sm,                                          (2.1)
F_st=F_mt+J_mt F_sm J_mt^T,                             (2.2)
B_st=B_sm+J_sm^(-1) B_mt J_sm^(-T),                    (2.3)
H_st=J_mt H_sm+H_mt J_sm^(-T).                          (2.4)
```

These are exact pathwise identities. They permit a proof or calculation to
restart on a shrinking parabolic window without retaining the full history
from time zero.

The audit verifies all four identities on a noncommuting trajectory of an
exact periodic Navier-Stokes solution.

## 3. What Moment Actually Controls Critical Velocity

The restarted Constantin-Iyer velocity formula is

```text
u(t)=E_plus P[(grad A_st)^T (u(s) o A_st)],              (3.1)
```

where `A_st=X_st^(-1)` and `P` is the Leray projection. Here `E_plus` is
expectation over the common Brownian driver `W_plus`.
The auxiliary antisymmetric driver `W_minus` has already been integrated
out in the conditional Gramians. At `x=X_st(a)`,

```text
grad A_st(x)=J_st(a)^(-1).
```

Every stochastic flow realization is volume preserving. Boundedness of
`P` on `L^3`, Minkowski/Jensen, and the change of variables `x=X_st(a)`
therefore give

```text
||u(t)||_3
 <=||P||_(3->3)
   [integral E_plus |J_st(a)^(-T)u(s,a)|^3 da]^(1/3).   (3.2)
```

Define the exact directional moment

```text
Gamma_J(s,t)
 =integral E_plus |J_st(a)^(-T)u(s,a)|^3 da.            (3.3)
```

Uniform boundedness of (3.3) as `t` approaches a putative first singular
time gives `u in L^infinity_t L^3_x`, hence a standard critical continuation
criterion applies.

The cubic power is not arbitrary: it is the absolute-value moment produced
by the critical `L^3` norm. Lower moments would require additional
cancellation before taking absolute values.

## 4. Tensor and Scalar Sufficient Conditions

The Gramian congruence gives

```text
||J_st^(-1)||_2^2
 <=lambda_max(B_st)/lambda_min(F_st).                   (4.1)
```

Thus

```text
Q_st=
 [lambda_max(B_st)/lambda_min(F_st)]^(3/2)              (4.2)
```

is a tensor-spectral sufficient cubic weight:

```text
|J_st^(-T)v|^3<=Q_st |v|^3.                             (4.3)
```

The determinant-floor argument from the previous checkpoint gives the
scalar radial fallback

```text
R_st=[b_st f_st^2/4]^(3/2)
    =b_st^(3/2) f_st^3/8,                               (4.4)

Q_st<=R_st.                                             (4.5)
```

Consequently there is a strict hierarchy:

```text
exact directional moment
 <= tensor-spectral moment
 <= scalar radial moment.                               (4.6)
```

Any uniformly bounded weighted spatial integral in this hierarchy is a
sufficient classical continuation condition. None is presently bounded
from the Leray energy inequality.

## 5. Infinitesimal Meaning of the Radial Traces

For a constant trace-free velocity gradient `A=S+Omega`, with `S`
symmetric and `Omega` skew,

```text
tr[exp(Ar)exp(A^T r)]
 =3+2||S||_F^2 r^2+O(r^3).
```

After integration over a window of length `tau`,

```text
f_st=3+(2/3)||S||_F^2 tau^2+O(tau^3),
b_st=3+(2/3)||S||_F^2 tau^2+O(tau^3).                  (5.1)
```

Thus the first radial excess detects strain, not rigid rotation. This is
encouraging, but it does not preserve which eigendirection created each
part of the excess.

## 6. Burgers-Vortex-Axis Falsifier

Consider the standard axial linearization

```text
S=diag(-a/2,-a/2,a)
```

with an arbitrary commuting rotation in the transverse plane. This is a
local Burgers-vortex-type model, not a finite-energy periodic solution.

The tensor ratio (4.2) pairs the expanding inverse-time covariance with the
matching contracted forward direction and captures the inverse deformation
sharply. The scalar expression (4.4) instead multiplies:

- the axial forward expansion;
- the transverse inverse-time expansion;
- without preserving their eigendirection labels.

At `a tau=8`, the audit finds that the scalar radial cubic bound exceeds
the tensor bound by more than `10^12`. The scalar condition remains
mathematically sufficient but is not a credible primary closure target.

This is a useful negative result: the collision programme must retain at
least tensorial directional pairing at the continuation gate.

## 7. Exact Periodic Navier-Stokes Tests

### Decaying finite-Fourier shear

The field

```text
u(x,t)=
 [U exp(-nu k^2 t) sin(k y),0,0]                        (7.1)
```

is divergence free, has zero nonlinear advection, and solves the unforced
periodic Navier-Stokes equation exactly. Along the deterministic path
`y=0`,

```text
J=I+K E_12,
K=U[1-exp(-nu k^2 T)]/(nu k).                           (7.2)
```

The numerical window integration reproduces (7.2), and the exact, tensor,
and radial hierarchy remains correctly ordered. Nonnormal shear creates a
real loss even without exponential eigenvalue stretching.

### Decaying ABC Beltrami flow

The `A=B=C=1` ABC field satisfies

```text
div u_0=0,
curl u_0=u_0,
Delta u_0=-u_0.
```

Therefore

```text
u(x,t)=exp(-nu t)u_0(x)
```

is an exact smooth finite-Fourier unforced periodic Navier-Stokes solution,
with its nonlinear term absorbed into pressure. Four deterministic
trajectories pass the Gramian hierarchy, determinant, inverse, and
volume-preservation checks. A split at the middle of the interval verifies
all restart laws (2.1)-(2.4) for a spatially varying noncommuting gradient.

These exact regular solutions validate the formulation. They do not supply
the missing a priori estimate near a hypothetical singularity.

## 8. Surviving Theorem Target

The primary target is no longer a bound on `f` and `b` separately. It is a
bound on either

```text
integral |u(s,a)|^3 E_plus Q_st(a) da                   (8.1)
```

or, preferably, the directional expression (3.3). The proof must exploit
the two-point Navier-Stokes generator, pressure/vorticity structure, or a
sign/cancellation before taking operator norms.

The scalar radial moment (4.4) remains a correct fallback and diagnostic,
but the Burgers-axis calculation shows why using it as the main theorem
target is likely to recreate the standard supercritical loss.

Still open:

- an unconditional bound for (3.3) or (8.1);
- a low-regularity construction of the inverse-time probe;
- an exceptional-set upgrade;
- global regularity.

Reproduce with

```text
python work/ns_collision/scripts/parabolic_gramian_continuation_audit.py \
  --output work/ns_collision/results/parabolic_gramian_continuation_audit_v1.json
python -m unittest \
  work/ns_collision/tests/test_parabolic_gramian_continuation.py
```
