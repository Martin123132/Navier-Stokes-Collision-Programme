# Radial-payoff Bellman gate

## Why the payoff matters

The all-exit exponential moment was a useful volume majorant, but it throws
away the strongest feature of the compact cylinder: an axial-cap exit has
payoff zero. The actual one-history visit kernel has scalar payoff

```text
m_B(t,y)=E_(t,y)[exp(int_t^tau lambda(s)ds);
                 the radial wall is hit before an axial cap].       (1)
```

Here `lambda<=1`, `r<2`, `|z|<0.75`, and the entry interface is `r=1`.
The all-exit Bellman problem nearly exhausts the renewal allowance because a
position-dependent controller can confine every particle. That observation
does not apply to (1): confinement increases the chance of eventually being
killed at a cap.

## Exact control Hamiltonian

After normalizing by maximal stretching and following the spin frame, the
backward affine drift matrix satisfies

```text
B=B^T,       tr B=0,       B>=-I.                       (2)
```

The time-dependent affine history is an open-loop choice in this set. Enlarge
it to a feedback control that may choose `B` separately at every `(t,y)`.
For `M=sym(y tensor p)`, write `C=B+I`. Then

```text
C>=0,       tr C=3,
B:M=C:M-tr M<=3 lambda_max(M)-tr M.                    (3)
```

The eigenvalues of `M` are

```text
(y.p+|y||p|)/2,  0,  (y.p-|y||p|)/2.                  (4)
```

Consequently the control Hamiltonian is exactly

```text
sup_B (B y).p=(y.p)/2+3|y||p|/2.                      (5)
```

Equality is attained by `B=3v tensor v-I`, with `v` a top eigenvector of
`M`. Since every measurable common affine history is among the controls,
the Bellman value is a pointwise upper bound for (1).

## Axisymmetric HJB

The cylinder and Hamiltonian are axisymmetric. Its radial-payoff value solves
in the viscosity sense

```text
Delta u+(y.grad u)/2+3|y||grad u|/2+u=0,              (6)
u=1 on r=2,       u=0 on |z|=0.75.                    (7)
```

The `+u` term is the worst case `lambda=1`; smaller nonnegative stretching is
covered by the same supersolution. A centered finite-difference policy
iteration gives the following inner-interface maxima:

| radial x axial intervals | `max_(|z|<0.75) u(1,z)` |
|---:|---:|
| `40 x 30` | `0.68692812` |
| `60 x 46` | `0.68726239` |
| `80 x 60` | `0.68737389` |
| `120 x 90` | `0.68746110` |

All maxima occur at `z=0`. These values are a converged pilot, not a
comparison-certified enclosure. They nevertheless show that the true
radial-payoff Bellman problem has far more room than the all-exit majorant.

## Explicit supersolution

Let

```text
s=r/2,       c=cos(2 pi z/3),

U(r,z)
 =0.89945 s^2+0.10055 s^16
  +1.3479(1-s^2)^(13/20)c^(7/20).                    (8)
```

All coefficients and exponents in (8) are rational except the displayed
cosine frequency. The boundary inequalities are exact:

```text
U(2,z)=1,
U(r,+/-0.75)=0.89945 s^2+0.10055 s^16>=0.            (9)
```

The maximum on the inner interface is explicitly

```text
U(1,0)=1.342878684567>1.232133608495.                 (10)
```

Put

```text
L=Delta U+(y.grad U)/2+U,
G=L^2-(9/4)|y|^2|grad U|^2.                           (11)
```

The inequalities `L<0` and `G>0` imply that the HJB residual in (6) is
negative. An independent grid with 1,700 radial points and 1,300 points on
the axial half-domain reports

```text
max residual=-0.01018214,
max L       =-0.57053157,
min G       = 0.02402878.                             (12)
```

The fractional powers make the second-derivative contribution tend to
negative infinity at both boundary pieces, while the first-derivative term
has a weaker singular order. The numerically delicate minimum of `G` is in
the interior near `(r,z)=(1.460,0.359)`.

Equation (12) first supplied falsification evidence. The sign has now been
certified on the whole open half-cylinder in three pieces.

1. On `0<=r<=1.9`, `0<=z<=0.7`, outward-rounded `mpmath.iv` mean-value
   enclosures certify the stronger residual with `1.005U` in 4,336
   evaluated boxes, with 2,192 certified leaves and no unresolved boxes.
2. Put `delta=10^-3`. On the three finite collars ending at the deliberately
   overlapping cutoffs `r=1.999` and `z=0.74953`, the coarser sufficient
   residual

   ```text
   Delta U+1.005U+2|y|(|U_r|+|U_z|)                  (13)
   ```

   is interval-certified negative in 339 radial, 41 axial, and 61 corner
   boxes, again with no unresolved boxes.
3. In the final open strips use

   ```text
   x=1-r^2/4,       a=cos(2 pi z/3).                  (14)
   ```

   Since `13/20+7/20=1`, if `a>=x` the negative radial curvature is of
   order `x^(-1)(a/x)^(7/20)` and absorbs the radial gradient, while the
   axial-gradient ratio is at most one. If `x>=a`, the symmetric axial
   statement applies. Directed interval arithmetic gives lower residual
   margins `251.81214885` and `1305.47147084` in the two cases.

SymPy independently verifies all eleven factorized derivative identities
used by the interval evaluator. Thus (8) is now a computer-assisted
supersolution certificate, not a mesh inference.

## Certified consequence

For any progressively measurable admissible affine history, apply Ito to
`exp(s)U(Y_s)` and stop before cylinder exit. The certified HJB residual
makes this a nonnegative supermartingale. Localization and Fatou then give,
for every entry point and hence every entry law `mu` on `r=1`,

```text
||K1||_(L2(mu))<=1.342878685.                         (15)
```

The square-tilted dynamic kernel identity then yields

```text
C_dynamic
 <=0.658695038668(1.342878685)^2
 =1.187840020>1.                                      (16)
```

Thus this first barrier remains a valid computer-assisted HJB certificate
but no longer closes after the cubic recentering cost is charged. The later
finite-energy barrier with gain `1.145614144998` replaces it in the live
cycle. The old `0.930737669` value is the legacy bare-halving calibration.

## Common Doob killing

The strengthened certificate is

```text
Delta U+sup_B (B y).grad U+1.005 U<=0.               (17)
```

Writing an actual payoff as `u=Uv` therefore gives a common transformed
killing rate at least `0.005` for every affine history. In particular, a
smooth perturbation with divergence-free drift error `e` and adverse
potential `q_+` is still controlled by the same pointwise barrier whenever

```text
q_+ + e.grad(log U) <=0.005.                          (18)
```

The absolute sufficient condition replaces the signed drift term by
`|e||grad log U|`. This is an exact smooth-approximation abort criterion, but
it is not the desired Leray-level result: `grad log U` is singular at the
absorbing boundary and critical `L^3/L^(3/2)` control does not imply (18).
The next analytic problem is a weighted form version of (17)-(18), or a
stopping construction that pays failures of (18) without assuming pointwise
regularity.

The Hamiltonian pilot and explicit barrier diagnostics are reproduced by
`scripts/radial_payoff_bellman_pilot.py` and
`scripts/radial_payoff_supersolution_audit.py`. The full interval and
asymptotic certificate is reproduced by
`scripts/radial_payoff_interval_certificate.py`.
