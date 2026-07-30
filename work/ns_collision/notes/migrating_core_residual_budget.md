# Migrating-core residual budget

## Purpose

The smooth wall-tracking interpolation has favorable scalar arithmetic, but
that arithmetic does not yet describe an actual moving Navier-Stokes core.
This note isolates the exact geometric part of the mapped PDE and states the
remaining perturbation obligation without treating a model response constant
as a theorem.

## Moving decomposition

Use transverse coordinates

```text
x=c(t)+L(t)O(t)y,              |y|<1,
ell=L'/L<0,                    R_*=A L^2/nu=1/2.
```

Let `v_c` be the baseline comoving velocity and choose the tracked centre law

```text
c'=v_c-L ell eta O n,          eta=2.                 (1)
```

Decompose the physical drift relative to `v_c` into the reference affine
part, an amplitude deficit, and a genuinely nonaffine residual. In mapped
coordinates the error relative to `A y` is

```text
e=e_res+(a-A)y+ell(eta n-y)-Omega y,                 (2)
Omega=O^T O'.
```

Write the stretching error as

```text
delta_s=2(a-A)+delta_s,res.                           (3)
```

The constant-Reynolds radial gauge puts all perturbations into

```text
q=delta_s-1/2 div(e)-(R_*/2)y dot e.                 (4)
```

Substitution in (4) gives the exact split

```text
q=q_geom+q_amp+q_res,                                (5)

q_geom=ell[1+(R_*/2)(|y|^2-eta y dot n)],
q_amp=(a-A)(1-R_*|y|^2/2),
q_res=delta_s,res-1/2 div(e_res)-(R_*/2)y dot e_res.
```

Pure rotation contributes zero because `div(Omega y)=0` and
`y dot Omega y=0`. If `A>=a`, `R_*<=2`, and `|y|<=1`, then `q_amp<=0`.
At the working parameters the geometric bracket is

```text
3/4+[(y dot n-1)^2+|y-(y dot n)n|^2]/4>=3/4.         (6)
```

Thus only `q_res` can be adverse under these assumptions. This does not say
that `q_res` is small; it identifies exactly what must be estimated.

## Sharp scalar allowance

For each entry angle let `a_R` and `a_S` be the return and tracked-wall
one-history gains after the one physical-to-gauge conversion. The working
pair condition is

```text
C(theta)=a_R(theta)^2+a_S(theta)^2,
max_theta C(theta)=0.672126890291... .                (7)
```

If a residual multiplies both one-history gains by `exp(E)`, (7) remains
contractive exactly when

```text
E<-(1/2)log(max C)=0.1986540657... .                  (8)
```

Equivalently, a common multiplicative error must be below

```text
1/sqrt(max C)=1.2197599361... .                       (9)
```

Branch-resolved actions have the sharper condition

```text
a_R^2 exp(2E_R)+a_S^2 exp(2E_S)<1.                  (10)
```

The audit also solves the anglewise quadratic for additive gain errors. If
the return and wall responses are bounded by `K_R F` and `K_S F`, the sharp
calibrated ceiling is the smallest positive root over the entry circle of

```text
(a_R+K_R F)^2+(a_S+K_S F)^2=1.                      (11)
```

This is the right place for the PDE trace theorem to enter; setting
`K_R=K_S=1` is only a unit-response calibration.

At that unit calibration, the worst entry angle is the transverse axis and

```text
common additive F allowance =0.129665885385,
return-only gain allowance  =0.207444861994,
wall-only gain allowance    =0.255907964394.          (12)
```

## Critical-norm calibration

The certified global finite-energy barrier supplies

```text
F=0.798968551320 ||q_res||_(3/2)
 +3.072840583265 ||e_res||_3,                        (13)

alpha=0.220329037686 ||q_res||_(3/2).
```

With `K_R=K_S=1`, the one-error calibrations are therefore

```text
||q_res||_(3/2)<0.162291601054,                      (14)
||e_res||_3      <0.042197400702.                    (15)
```

Combining (11) and (13) is conditional on proving the relevant wall-stopping
boundary response constants `K_R,K_S`. The audit reports the resulting
unit-response pure-potential and pure-drift numbers so later estimates can be
compared on the correct scale, but it does not label them Navier-Stokes
budgets.

## Remaining theorem

Three linked facts are still missing:

1. construct an adapted, mollified centre and frame through the wall-triggered
   interpolation and identify its actual `e_res` and `delta_s,res`;
2. derive `K_R,K_S` for the unnormalized wall-stopping law using a space-time
   trace estimate, including the unbounded migration-centre moment;
3. prove that the resulting critical norms fit (11) through every generation
   without conditioning away stopped mass.

The exact algebra and angle-resolved numerical allowances are reproduced by
`scripts/migrating_core_residual_budget_audit.py`.
