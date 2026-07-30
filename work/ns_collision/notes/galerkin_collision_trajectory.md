# Galerkin Collision Trajectories

Date: 2026-07-17

Status: validated finite-Galerkin evidence for the exact two-mode positive and
negative collision-defect channels. This is a diagnostic experiment, not a
proof of a Navier-Stokes estimate.

## 1. Numerical Contract

The solver uses the symmetric Fourier cube

```text
max(|k_1|,|k_2|,|k_3|)<=N
```

and evaluates the projected convolution directly. There is no pseudospectral
aliasing. Every trajectory checks:

```text
div u=0,
u_(-k)=conjugate(u_k),
(1/2)||u(t)||_2^2+nu*integral ||grad u||_2^2=(1/2)||u_0||_2^2,
partial_t J_s=-nu*D_s+X_s.                       (1.1)
```

The maximum residuals over the completed sweep were

```text
relative kinetic-energy balance       1.43e-15,
differential primitive identity        8.53e-14,
integrated primitive balance           3.25e-6.  (1.2)
```

The integrated residual is larger because `D_s` and `X_s` are sampled at 101
output times and integrated by Simpson quadrature. The differential identity
is evaluated algebraically at each snapshot.

A case is called resolved here only when

```text
relative change from the preceding truncation <3 percent,
maximum boundary-shell energy fraction <1e-3.    (1.3)
```

The raw append-only data are in
`results/galerkin_trajectory_sweep.jsonl`. An interrupted parent shell left
its Python child alive during the first sweep, producing eight duplicate
rows. Every duplicate is bit-for-bit identical. The runner now takes an
exclusive output lock, and `galerkin_sweep_analysis.py` deduplicates by the
full parameter key.

## 2. Positive Initial Channel

The positive companion was evolved at

```text
nu=1,       s=0.5,
T=min(2,2/R).
```

Thus the higher-Reynolds runs cover two initial nonlinear times. The best
available truncations give

```text
 R     N    integral D_s    integral D_s/(nu*integral Q)   status
0.25   3     0.000711844             0.00142493             resolved
0.50   3     0.0113099               0.00559390             resolved
1.00   3     0.176063                0.0208283              resolved
2.00   4     2.54112                 0.0674019              resolved
4.00   4    29.7468                  0.160147               marginal
```                                                     

At `R=2`, the `N=3` and `N=4` integrals differ by `0.47 percent`, and the
`N=4` boundary fraction is `5.00e-5`. At `R=4`, the change is `4.06 percent`
and the boundary fraction is `1.24e-3`, so that row is retained only as a
marginal trend.

All sampled positive-channel cumulative defects are positive. The resolved
ratio remains well below one through `R=2`, but it rises monotonically over
this family. A single trajectory cannot establish a uniform `theta<1`.

The weak-response prediction from `trajectory_collision_defect.md` is

```text
integral D_s/(nu*integral Q)
 =0.02283179438*R^2+O(R^3).                       (2.1)
```

The measured value divided by the leading prediction is

```text
R=0.25: 0.9986,
R=0.50: 0.9800,
R=1.00: 0.9122,
R=2.00: 0.7380,
R=4.00: 0.4384.                                  (2.2)
```

This independently validates the exact Duhamel calculation at small `R` and
shows a subquadratic departure as nonlinear transfer becomes important.

## 3. Negative Initial Channel

The exact negative quartic-transfer field does not define a dynamically
invariant cumulative cone. At `N=3`, its signed integral is

```text
R=0.25:  -3.79588e-5,
R=0.50:  -4.49239e-4,
R=1.00:   1.59965e-3,
R=2.00:   3.35383e-1.                             (3.1)
```

The change of sign is resolved in the bracket

```text
R=0.922: integral D_s=-1.70794e-6,
R=0.940: integral D_s= 2.82630e-4,                (3.2)
```

with boundary fractions near `3e-6`. The location depends on this initial
field, heat scale, and observation interval; it is not a universal Reynolds
constant.

Identity (1.1) explains the transition. Since `J_s(0)=0`,

```text
nu*integral D_s=integral X_s-J_s(T).              (3.3)
```

At `R=0.922`, the endpoint primitive still exceeds the accumulated transfer:

```text
J_s(T)=2.63427e-5,
integral X_s=2.46348e-5.                          (3.4)
```

At `R=0.94`, the transfer has overtaken the endpoint correction:

```text
J_s(T)=2.90899e-5,
integral X_s=3.11720e-4.                          (3.5)
```

Thus nonlinear evolution converts the initially helpful interference into a
net positive cumulative defect by transfer recurrence, not by violating the
exact primitive identity.

## 4. Consequence

The experiments close two possible shortcuts:

```text
1. A negative initial X_s does not remain a negative cumulative reservoir.
2. Viscous sign reversal of D_s does not force its time integral negative.
```

They do not disprove the cumulative criterion. In every resolved positive
case sampled here, `integral D_s/(nu*integral Q)<0.07`. The missing theorem is
uniformity over arbitrary smooth trajectories and arbitrarily long times.

The next structural target is now

```text
control integral X_s by endpoint motion of J_s plus a strict fraction of
viscous palinstrophy, while retaining the off-diagonal helical phases.       (4.1)
```

The next diagnostic should decompose `X_s(t)` into the exact parity/helical
blocks from `quartic_transfer_helical_matrix_audit.py` and identify which
off-diagonal channel causes the transition in (3.2).
