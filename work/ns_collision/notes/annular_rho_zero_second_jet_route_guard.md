# Annular rho-zero second-jet route guard

## 1. Result

For the static-optimal annular restart family, the complete restart-time
second derivative of

```text
g(u,lambda)
 =integral[
    p(u)u dot grad lambda
    -nu lambda|grad u|^2
    -nu lambda|grad lambda|^2]
```

has now been decomposed exactly. The calculation retains the
Navier-Stokes acceleration, both pressure Hessian terms, and the second
derivative of the backward weight. A fully dealiased small-carrier replay
and an independent central second difference verify the formula.

One asymptotic component can already be decided. The pressure contribution
with two velocity-heat derivatives obeys

```text
g''_(heat,pressure)(0)/N^7
 -> 1.3259462572683687e-6 > 0.                    (1.1)
```

This positive curvature is large enough that the quadratic model formed
from it and the certified negative first jet changes slope near

```text
N^2 t = 0.07875390524396945.                      (1.2)
```

Equation (1.2) is not a prediction for the full flow. Nonlinear second-jet
channels are unresolved, and the four-high/one-low inviscid pressure branch
can follow a candidate scale above `N^7`. Thus neither the leading curvature
nor the finite `T/N^2` window is known.

## 2. Exact chain rule

Write

```text
u_1
 =-P[(u dot grad)u]+nu Delta u,

lambda_1
 =-u dot grad lambda-nu Delta lambda.
```

Differentiating the coupled equations once more gives

```text
u_2
 =-P[(u_1 dot grad)u+(u dot grad)u_1]
   +nu Delta u_1,

lambda_2
 =-u_1 dot grad lambda-u dot grad lambda_1
   -nu Delta lambda_1.                            (2.1)
```

The exact second generator jet is

```text
g''
 =D_uu g[u_1,u_1]
  +2D_u_lambda g[u_1,lambda_1]
  +D_lambda_lambda g[lambda_1,lambda_1]
  +D_u g[u_2]
  +D_lambda g[lambda_2].                          (2.2)
```

For velocity directions `v,w` and weight directions `mu,eta`,

```text
D_uu g[v,w]
 =integral[
    p''[v,w] u dot grad lambda
    +p'[u;v] w dot grad lambda
    +p'[u;w] v dot grad lambda
    -2nu lambda grad v:grad w],                   (2.3)

D_u_lambda g[v,mu]
 =integral[
    p'[u;v] u dot grad mu
    +p v dot grad mu
    -2nu mu grad u:grad v],                       (2.4)

D_lambda_lambda g[mu,eta]
 =-2nu integral[
    mu grad lambda dot grad eta
    +eta grad lambda dot grad mu
    +lambda grad mu dot grad eta],                (2.5)

p''[v,w]=p[v,w]+p[w,v].                           (2.6)
```

The audit expands (2.2) into 20 labelled channels by splitting

```text
u_1=u_E+u_nu,
lambda_1=lambda_A+lambda_nu.
```

It then reconstructs (2.2) directly from the summed first directions and
accelerations. At the fixed-amplitude `N=3` replay, the expanded/direct
residual is

```text
1.14e-13.
```

A Richardson-extrapolated central second difference along

```text
u(t)=u+t u_1+(t^2/2)u_2,
lambda(t)=lambda+t lambda_1+(t^2/2)lambda_2
```

has relative residual `2.23e-11`.

## 3. Dealiasing ledger

If `K` is the one-field velocity carrier radius and the weight has fixed
radius `L=1`, then

```text
support(u_E)                    <=2K,
support(u_nu)                   <=K,
support(lambda_A)               <=K+L,
support(lambda_nu)              <=L,

support(D E[u_E])               <=3K,
support(D E[u_nu])              <=2K,
support(nu Delta u_E)           <=2K,
support(nu Delta u_nu)          <=K,
support(lambda_2)               <=2K+L.           (3.1)
```

Every coefficient of order `t^2` therefore has total support at most
`5K+O(L)`. Each rectangular grid length is chosen beyond ten times the
corresponding one-field carrier maximum. Replaying every channel with
dealias factors ten and twelve changes the largest channel by only
`4.27e-14`; the total changes by `2.28e-13`.

## 4. Double-heat pressure identity

Let `B_(2),N` be the original resonant HHL pressure load with the multiplier

```text
(|k_1|^2+|k_2|^2+|ell|^2)^2
```

inserted into every monomial. The exact finite-carrier identity is

```text
D_uu g_pressure[u_nu,u_nu]
 +D_u g_pressure[nu Delta u_nu]
 =-nu^2 a_N t_N B_(2),N.                         (4.1)
```

The direct Hessian/acceleration replay at `N=3` differs from (4.1) by
`1.43e-14`; the nondegenerate `N=5` replay is exact to displayed precision.

For heat order `m`,

```text
B_(m),N/N^(2m+1)
 -> (sqrt(2)/20) 2^m
    integral_D S(xi)^2 |xi|^(2m)
      (V_y(xi)^2-V_z(xi)^2) dxi.                 (4.2)
```

At `m=2`,

```text
B_(2),N/N^5 -> -0.22213447028743452.              (4.3)
```

The sign is analytic. On

```text
D=[2,3] x [-1/2,1/2]^2
```

one has

```text
V_z^2-V_y^2>=255/13718,
|xi|^4>=16,
integral_D S^2=1/8.
```

Thus the absolute value in (4.3) is at least

```text
51sqrt(2)/6859.
```

Since

```text
a_N/N -> |b_0|/nu,
t_N/N -> |b_0|sqrt(8/(3q))/nu,
```

the two factors of viscosity in (4.1) cancel and yield (1.1). The sparse
finite rows remain positive after normalization:

```text
N       B_(2),N/N^5       g''_(heat,pressure)/N^7
25     -0.2460734513       1.6297526149e-6
33     -0.2401419312       1.6873414739e-6
49     -0.2341720391       1.5937422607e-6
65     -0.2311738855       1.5281250828e-6
```

## 5. What remains

The support and predecessor power bounds triage the exact channels as
follows:

```text
double velocity heat, pressure       certified order N^7, positive
double velocity heat, Fisher         subcritical route
one heat and one Euler direction     at most N^6 route
fixed-weight anti-diffusion alone    no carrier N^2 gain
```

Two groups are not yet closed:

```text
H_uu[u_E,u_E] + D_u[D E[u_E]]
  in the nonlinear velocity-pressure sector,

pure transport and mixed pressure terms involving
u_E, lambda_A, and lambda_2.                         (5.1)
```

The earlier route count assigned these groups a possible `N^7` scale. That
count is not valid for the four-high Euler-pressure branch: differentiating
the already-summed first-jet estimate does not control the new internal
pressure correlations. Branch projection subsequently exposes a candidate
fixed-amplitude coefficient permitted at `N^7`, which becomes a candidate
optimized `N^9` contribution after the factors `a_N t_N=O(N^2)`. The next
theorem must resolve that branch by its
finite pressure outputs before any full leading coefficient is assigned. A
production `N>=25` full second-jet FFT remains deliberately deferred.

Nothing here proves a uniform Taylor remainder, a positive finite-window
gain, critical `L^3` control, blowup, or global regularity.
