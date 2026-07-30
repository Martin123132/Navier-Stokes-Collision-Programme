# Correlated-Replica Tangent Gramian Bridge

Date: 2026-07-25

Status: an exact smooth-flow bridge from a correlated two-replica limit to
the stochastic-flow deformation tensor is proved below. The scale-critical
Navier-Stokes estimate needed to use the bridge is open.

Nothing in this note proves three-dimensional global regularity.

The finite-dimensional Gramian identities themselves are standard
control-theoretic algebra. No novelty claim is made for those identities in
isolation. The research question is whether their realization as the
`rho->1` limit of the replica system exposes an estimate using the specific
Navier-Stokes structure. The stochastic-flow conventions are tied to
`references/constantin_iyer/detsns.tex`, especially `e:vort-trans`.

## 1. Purpose

The independent-replica calculation has an exact three-dimensional radial
drift, but two labels in one Constantin-Iyer flow use common noise. The
missing representation question is whether independent-replica separation
can be continued to the common-noise endpoint without discarding the
deformation information.

The continuation parameter is the correlation of two Wiener drivers. Its
singular endpoint produces a tangent diffusion. Two time orientations of
that tangent diffusion give Gramians whose exact algebra recovers the flow
Jacobian.

## 2. Exact Correlation Homotopy

Let `W_plus` and `W_minus` be independent standard `d`-dimensional Wiener
processes. For `-1<=rho<=1`, put

```text
W_1=sqrt((1+rho)/2) W_plus+sqrt((1-rho)/2) W_minus,
W_2=sqrt((1+rho)/2) W_plus-sqrt((1-rho)/2) W_minus.
```

Then each driver is standard and

```text
d<W_1^i,W_2^j>=rho delta_ij dt.
```

For

```text
dX_i=u(X_i,t)dt+sqrt(2nu)dW_i,
R=X_1-X_2,  G=|R|^2,  epsilon=1-rho,
```

the exact relative equation is

```text
dR=[u(X_1,t)-u(X_2,t)]dt+2sqrt(nu epsilon)dW_minus.       (2.1)
```

Ito's formula gives

```text
dG=2R dot delta_u dt+4nu d epsilon dt
   +4sqrt(nu epsilon) R dot dW_minus,                    (2.2)

d log G=
  2(R dot delta_u)/G dt
  +4nu(d-2)epsilon/G dt
  +4sqrt(nu epsilon) R/G dot dW_minus.                  (2.3)
```

Thus in `d=3` the independent endpoint `rho=0` has the known
`4nu/G` drift, while that direct radial drift vanishes continuously at
the common-noise endpoint `rho=1`.

The joint two-point generator is

```text
L_rho =
 u(x) dot grad_x+u(y) dot grad_y
 +nu(Delta_x+Delta_y+2rho grad_x dot grad_y).            (2.4)
```

For centre `q=(x+y)/2` and difference `r=x-y`, its diffusion part is

```text
nu(1+rho)/2 Delta_q+2nu(1-rho)Delta_r.                  (2.5)
```

The degeneration at `rho=1` is therefore explicit and one-dimensional in
the parameter `epsilon`.

## 3. The Common-Noise Tangent Limit

Start the replicas at a common point and use the symmetric coupling above.
Set

```text
Q_epsilon=(X_1+X_2)/2,
Y_epsilon=(X_1-X_2)/sqrt(epsilon).
```

For a smooth drift, the symmetric Taylor expansions are

```text
[u(q+sqrt(epsilon)y/2)-u(q-sqrt(epsilon)y/2)]
  /sqrt(epsilon)
 =grad u(q)y
  +epsilon D^3u(q)[y,y,y]/24+O(epsilon^2|y|^5),         (3.1)

[u(q+sqrt(epsilon)y/2)+u(q-sqrt(epsilon)y/2)]/2
 =u(q)+epsilon D^2u(q)[y,y]/8+O(epsilon^2|y|^4).        (3.2)
```

Consequently, on every interval where the required derivatives of `u` are
bounded, moment estimates and Gronwall stability give convergence on finite
time intervals to

```text
dX=u(X,t)dt+sqrt(2nu)dW_plus,
dY=grad u(X,t)Ydt+2sqrt(nu)dW_minus.                    (3.3)
```

The symmetric coupling removes an order-`sqrt(epsilon)` drift error. The
first nonlinear correction is order `epsilon`. Without the smooth bounded
derivatives available before a putative first singular time, this limiting
statement is not yet justified.

Equation (3.3) is not the deterministic label derivative. It is a
noise-driven tangent probe around the common stochastic path. Conditioned
on that path, it is a linear Gaussian system.

## 4. Forward, Inverse-Time, and Cross Gramians

Fix a conditioned smooth base path and write

```text
A(s)=grad u(X_s,s).
```

Let `Phi(t,s)` be the fundamental matrix

```text
partial_t Phi(t,s)=A(t)Phi(t,s),  Phi(s,s)=I,
J=Phi(T,0).
```

Define

```text
F=4nu integral_0^T Phi(T,s)Phi(T,s)^T ds,               (4.1)
B=4nu integral_0^T Phi(0,s)Phi(0,s)^T ds,               (4.2)
H=4nu integral_0^T Phi(T,s)Phi(0,s)^T ds.               (4.3)
```

`F` is the conditional covariance of the forward tangent replica. `B` is
the corresponding inverse-time tangent covariance. `H` is their covariance
when the auxiliary tangent noise is shared. The low-regularity stochastic
construction of the inverse-time probe remains a separate gate; the
following finite-time smooth-flow algebra is pathwise.

The cocycle identity gives

```text
Phi(T,s)=Phi(T,0)Phi(0,s)=J Phi(0,s).
```

Substitution into (4.1)-(4.3) proves

```text
F=J B J^T,                                               (4.4)
H=J B,                                                   (4.5)
J=H B^(-1),                                              (4.6)
F=H B^(-1) H^T.                                         (4.7)
```

Because `B` is positive definite for `T>0`, (4.6) is an exact recovery of
the flow deformation, including its orientation. This closes the algebraic
part of the independent-to-common-noise representation gap while the flow
is smooth.

## 5. Incompressibility Converts Radial Data to Tensor Bounds

If `div u=0`, Liouville's formula gives

```text
det Phi(t,s)=1.
```

Equation (4.4) therefore implies

```text
det F=det B.                                             (5.1)
```

Minkowski's determinant inequality, applied to the positive matrices
`Phi(T,s)Phi(T,s)^T`, gives in three dimensions

```text
det F>=(4nu T)^3,  det B>=(4nu T)^3.                    (5.2)
```

This is a volume consequence of incompressibility, not a numerical
observation.

Put

```text
f=tr(F)/(4nu T),  b=tr(B)/(4nu T).                      (5.3)
```

For a positive `3 by 3` matrix `M`,

```text
lambda_min(M)>=4 det(M)/tr(M)^2.
```

Combining this inequality with (4.4), (5.2), and
`lambda_max(M)<=tr(M)` yields the dimensionless bounds

```text
||J||_2^2     <= f b^2/4,                               (5.4)
||J^(-1)||_2^2<= b f^2/4.                               (5.5)
```

Both `f` and `b` are invariant under the Navier-Stokes parabolic scaling:
the Gramians and `nu T` scale by the same length-squared factor.

This is the first exact route in the programme from radial second moments of
two tangent replicas to full deformation. Noncollision alone is still too
weak: what is required is an upper estimate for both normalized radial
variances.

## 6. Adversarial Tests

The reproducible audit checks zero flow, planar strain, simple shear, rigid
rotation, and a noncommuting two-stage trace-free strain.

### Planar strain

For

```text
A=diag(a,-a,0),
```

all covariance eigenvalues remain positive, but at fixed `a>0`

```text
||J^(-1)||=exp(aT).
```

At `aT=8` the audit has strictly positive endpoint covariance and
deformation above `1000`. This disproves any inference from covariance
positivity or radial noncollision alone.

### Simple shear

For

```text
A_12=k,  A^2=0,
```

the normalized radial traces are

```text
f=b=3+(kT)^2/3.
```

They detect nonnormal deformation, but the trace-only estimate (5.4) loses
more than two orders of magnitude at `kT=12`. The tensorial ratio from
(4.4) is sharper, and the cross covariance (4.6) is exact.

### Rigid rotation

For skew `A`,

```text
F=B=4nu T I,
```

so endpoint covariance tensors contain no orientation. The cross covariance
is

```text
H=4nu T J
```

and recovers the rotation exactly. This separates amplification control from
orientation-sensitive vorticity correlation.

## 7. What Is Proved and What Is Not

Established in the smooth regime:

- the exact `rho`-dependent pair and logarithmic-gap identities;
- the symmetric `rho->1` tangent limit and its order-`epsilon` drift defect;
- the forward/inverse-time Gramian congruence;
- exact Jacobian recovery from the shared-probe cross covariance;
- determinant balance and the incompressible determinant floor;
- scale-invariant radial-trace bounds for `J` and `J^(-1)`;
- explicit failure of one-sided noncollision as a deformation estimate.

Not established:

- construction of the inverse-time probe at Leray regularity;
- an unconditional Navier-Stokes bound for `f` and `b`;
- a sufficient averaged moment estimate that controls `L^infinity_t L^3_x`;
- an exceptional-set or capacity upgrade;
- global regularity.

## 8. Next Theorem Target

For a parabolic window `[t-r^2,t]`, define the conditioned local versions of
`F`, `B`, and `H` along a common stochastic trajectory. The next target is
one of the following:

```text
1. a scale-critical bound on moments of f b^2 and b f^2;
2. a sharper tensorial estimate using F, B, and H before taking traces;
3. a counterexample showing that Leray energy permits these quantities to
   diverge independently of every currently retained collision functional.
```

The estimate must use the actual Navier-Stokes pressure, vorticity, or triad
structure. A bound containing

```text
integral ||grad u||_infinity dt
```

is circular and does not pass the gate.

Reproduce the audit with

```text
python work/ns_collision/scripts/correlated_replica_tangent_gramian_audit.py \
  --output work/ns_collision/results/correlated_replica_tangent_gramian_audit_v1.json
python -m unittest \
  work/ns_collision/tests/test_correlated_replica_tangent_gramian.py
```
