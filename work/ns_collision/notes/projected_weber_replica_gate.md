# Projected Weber Replica Gate

Date: 2026-07-25

Status: the common-path tangent/covector generator and a smooth tensor
proxy are derived. Two exact projection losses show that neither the
unprojected Gramian moment nor a positive projected Jensen moment is the
right primary closure target. A signed projected replica estimate remains
open.

Nothing here proves global regularity.

## 1. Joint Tangent-Covector Generator

Along a smooth common stochastic trajectory, write

```text
A=grad u(X,t),  S=(A+A^T)/2.
```

Couple the common path, the `rho->1` tangent replica, an inverse covector,
and the inverse Cauchy-Green tensor by

```text
dX=u(X,t)dt+sqrt(2nu)dW_plus,
dY=A Ydt+2sqrt(nu)dW_minus,
dZ=-A^T Zdt,
dC=(-A^T C-C A)dt.                                (1.1)
```

Here `C=J^(-T)J^(-1)` when `C(s)=I`. The time-inhomogeneous generator is

```text
L=partial_t+u dot grad_X+nu Delta_X
  +(A Y) dot grad_Y+2nu Delta_Y
  -(A^T Z) dot grad_Z
  +(-A^T C-C A):grad_C.                           (1.2)
```

For `n=Z/|Z|`,

```text
L|Z|^3/|Z|^3=-3 n^T S n.                          (1.3)
```

The auxiliary collision diffusion `2nu Delta_Y` does not act on the bare
directional cubic. Thus the exact Gramian bridge is algebraically
sufficient for continuation, but its positive cubic moment does not
automatically inherit collision damping.

## 2. Smooth Tensor-Spectral Proxy

For finite `p>=1`, define

```text
Psi_p(C)=[tr(C^p)]^(3/(2p)).                       (2.1)
```

It is a smooth upper approximation to
`lambda_max(C)^(3/2)=||J^(-1)||_2^3`. Cyclicity of trace gives

```text
d tr(C^p)/dt=-2p tr(S C^p),

L Psi_p/Psi_p=-3 mu_p,
mu_p=tr(S C^p)/tr(C^p).                           (2.2)
```

This retains tensorial directional pairing, but again has no direct
positive viscous term.

## 3. Coupling One Collision Weight

For

```text
w_q,epsilon(Y)
 =[epsilon/(|Y|^2+epsilon)]^(q/2),
sigma=(Y/|Y|)^T S(Y/|Y|),
```

direct differentiation gives

```text
L(w|Z|^3)/(w|Z|^3)
 =-3mu-q sigma |Y|^2/(|Y|^2+epsilon)
  -2nu q[(1-q)|Y|^2+3epsilon]
       /(|Y|^2+epsilon)^2.                        (3.1)
```

For `0<q<=1`, the final term is nonpositive and gives strict collision
damping. It still cannot make the generator pointwise nonpositive.

Take

```text
S=diag(-a/2,-a/2,a),  Z parallel e_1.
```

Then `mu=-a/2`. If `Y` points along the expanding axis, the far-field rate
is

```text
(3/2-q)a >= a/2,  0<q<=1.                         (3.2)
```

If `Y` points along the same contracting axis, the rate is
`(3/2+q/2)a`. A single globally superharmonic collision factor therefore
does not close the cubic inverse-deformation generator. This is a sign
obstruction, not a Navier-Stokes counterexample.

## 4. Averaging Before the Norm

Let

```text
W=(grad A_st)^T[u(s) composed with A_st]
```

be the unprojected stochastic Weber integrand and put `m=E W`. The exact
SPDE in the Constantin-Iyer source yields

```text
partial_t m+u dot grad m-nu Delta m+(grad u)^T m=0,
m(s)=u(s),                                           (4.1)

u(t)=P m(t).                                         (4.2)
```

Consequently

```text
||u(t)||_3
 <=||P||_(3->3)||m(t)||_3
 <=||P||_(3->3)
   [integral E|W|^3]^(1/3).                          (4.3)
```

The deterministic mean is a weaker sufficient target than the pathwise
positive moment. It also has a genuine viscous balance:

```text
(1/3)d_t||m||_3^3+nu D_3(m)
 =-integral |m| m^T S m.                             (4.4)
```

Away from `m=0`, the nonnegative dissipation is

```text
D_3(m)=sum_k integral [
 |m||partial_k m|^2
 +|m|^(-1)(m dot partial_k m)^2].                    (4.4a)
```

At every window reset, `m=u` and incompressibility gives

```text
integral |u|u^T S u
 =(1/3)integral u dot grad(|u|^3)=0.                  (4.5)
```

This cancellation is exact. It is not persistent.

Writing `m=u+grad q`, the Leray gauge satisfies

```text
q_t+u dot grad q-nu Delta q=p-|u|^2/2,
q(s)=0.                                               (4.6)
```

Taking an absolute value in (4.4) and applying the three-dimensional
Gagliardo-Nirenberg inequality gives only

```text
|right side|
 <=epsilon D_3(m)
   +C epsilon^(-3)||S||_2^4||m||_3^3.                (4.7)
```

The Leray inequality controls the time integral of `||S||_2^2`, not its
fourth power. The naive absolute-value closure is supercritical.

## 5. Exact Periodic Shear Falsifier

Consider the smooth exact Navier-Stokes shear

```text
u=(c+A exp(-nu k^2 tau)sin(k y),0,0),  c>|A|.          (5.1)
```

Its nonlinear advection and pressure vanish. The canonical solution of
(4.1), reset at `tau=0`, is

```text
m_1=c+A exp(-nu k^2 tau)sin(k y),

m_2=-c A k tau exp(-nu k^2 tau)cos(k y)
    -A^2/[4nu k] exp(-2nu k^2 tau)
      [1-exp(-2nu k^2 tau)]sin(2k y),

m_3=0.                                                 (5.2)
```

The transverse component is a pure gradient, so

```text
P m=u.                                                  (5.3)
```

For `c=2`, `A=k=1`, the audit finds:

```text
nu=0.25: max ||m||_3^3 / ||u(0)||_3^3 = 2.3479,
nu=0.10: max ||m||_3^3 / ||u(0)||_3^3 = 18.2123,
nu=0.03: ratio exceeds 400 on 0<=tau<=20,
```

while the physical velocity cubic norm never exceeds its initial value.
The reset gauge variation is already adverse:

```text
d/dtau [integral |m|m^T S m] at tau=0
 =-integral (c+A sin y)^3(A cos y)^2 dy <0.             (5.4)
```

Thus the mean-magnetization improvement still pays a potentially enormous
gradient component that the actual Leray projection removes.

## 6. Projection Before Jensen Is Still Insufficient

Put `v_W=P W_W`, so `E v_W=u`. For the zero-mean oscillatory part of the
same shear, projection removes every nonzero transverse Fourier mode but
leaves a random harmonic transverse component

```text
c_W=(A_s^2 k/2) integral_0^tau
     exp(-nu k^2 r)
     sin(k sqrt(2nu)W_r)dr.                            (6.1)
```

Symmetry gives `E c_W=0`, but

```text
Var(c_W)
 =A_s^4/[96nu^2 k^2]
  {4-(3+12x)exp(-2x)-exp(-6x)},
x=nu k^2 tau.                                          (6.2)
```

The bracket is positive and increases from
`16x^3+O(x^4)` to `4`. Therefore

```text
E|c_W|^3 >= Var(c_W)^(3/2)>0,                          (6.3)
```

although its signed contribution to the physical velocity is exactly zero.
Applying Leray before a positive Jensen moment retains a second avoidable
loss: cancellation across common-noise realizations.

## 7. Signed Replica Identities

Let `v_1,v_2,v_3` be independent copies of the projected stochastic Weber
velocity. Independence and `E v_j=u` give the exact two-copy identity

```text
|u|^3=|u| E[v_1 dot v_2].                              (7.1)
```

With `n=u/|u|` on `{u!=0}`, one also has the cubic identity

```text
|u|^3
 =E product_(j=1)^3(v_j dot n).                        (7.2)
```

For the shear harmonic in (6.1), independent signed pairing gives
`E[c_1 c_2]=0`, while every positive moment sees the variance. Equations
(7.1)-(7.2) retain both structures discarded by the preceding bounds:

```text
1. Leray projection before taking a norm;
2. sign across independent common-noise realizations.
```

## 8. Revised Theorem Target

The unprojected directional and tensor Gramian moments remain correct
sufficient continuation conditions. The exact shear calculation shows why
they should no longer be the primary closure target.

The next target is the signed projected two- or three-replica generator.
Its pressure transfer must remain in divergence or antisymmetric flux form
until replicas and spatial cells have been paired. Only after that pairing
should an absolute value be taken.

Still open:

- a scale-critical bound for the signed projected replica functional;
- a low-regularity construction of the projected stochastic flow;
- an exceptional-set upgrade;
- global regularity.

Reproduce with

```text
python work/ns_collision/scripts/projected_weber_replica_gate_audit.py \
  --output work/ns_collision/results/projected_weber_replica_gate_audit_v1.json
python -m unittest \
  work/ns_collision/tests/test_projected_weber_replica_gate.py
```
