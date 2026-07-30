# First-window maximum-bridge certificate

## Purpose

The first Brownian-majorant theorem proves a complete first-window flux
bound, but its closed-form interval factor is `2.51383`. A direct mode pilot
suggests that this loss is concentrated in the binary radial-excursion
split, not in the analytic high-mode tail. This note replaces that binary
split by an integral over the maximum radius of the first-passage bridge.

The resulting bound is positive and nonoscillatory. It certifies an isolated
first-window interval factor below one without inverse-Laplace inversion.

## Maximum-radius integration

Let one-dimensional Brownian motion with generator `Delta` start at two,
let `tau` be its first hit of one, and let

```text
M_tau=max_(0<=s<=tau) R_s,
q_d(t)=d/(2sqrt(pi)t^(3/2))exp(-d^2/(4t)).           (1)
```

For the angular mode `n>=1`, put

```text
kappa_n=n^2-1/4,
f_n(M)=exp(-kappa_n t/M^2).                          (2)
```

The Bessel absolute-continuity identity from the preceding note gives

```text
c_n(t)<=2^(-1/2) E[f_n(M_tau);tau in dt]/dt.         (3)
```

For every increasing differentiable `f`, layer-cake integration gives

```text
E[f(M_tau);tau in dt]/dt
 =f(2)q_1(t)+int_2^infinity f'(M)
                   P(M_tau>M,tau in dt)/dt dM.       (4)
```

A path counted by the tail in (4) must first travel from two to `M` and then
from `M` to one. The strong Markov property and the stable first-passage
convolution therefore give

```text
P(M_tau>M,tau in dt)/dt
 <=q_(M-2)*q_(M-1)(t)=q_(2M-3)(t).                  (5)
```

Substitution in (3)-(4) yields

```text
c_n(t)<=2^(-1/2) q_1(t) [exp(-kappa_n t/4)
 +int_2^infinity 2kappa_n t/M^3
   exp(-kappa_n t/M^2)
   (2M-3)exp(-((2M-3)^2-1)/(4t)) dM].               (6)
```

Unlike the exact Bessel inversion, every term in (6) is nonnegative.

## Radius enclosure

The certificate partitions `2<=M<=5` into intervals of width `0.01`. On a
time-radius box `[t_0,t_1]x[M_0,M_1]`, the integrand ratio in (6) is bounded
above by

```text
2 kappa_n t_1 (2M_1-3)/M_0^3
 exp[-kappa_n t_0/M_1^2
     -((2M_0-3)^2-1)/(4t_1)].                       (7)
```

At the first radius endpoint the second exponent is clipped to its exact
nonnegative lower bound zero. Every arithmetic operation in the rectangle
sum is rounded outward with `nextafter`; positive floating sums include the
standard `gamma_N` accumulation allowance.

For `M>=M_*`, the ratio

```text
R(M,t)=(2M-3)exp(-((2M-3)^2-1)/(4t))                (8)
```

is decreasing in `M` throughout `0<t<=3/8`. Since the integral of `f_n'` is
at most one, the complete omitted-radius contribution is at most
`R(M_*,t_1)`.

## Time and mode enclosure

The range `0.02<=t<=0.375` is divided into slabs of width `0.001`. The
one-dimensional density `q_1` and the zeroth Bessel mode have their exact
critical times inserted whenever they lie in a slab. The OU likelihood is
evaluated at the right endpoint.

For the axial factor, the certificate uses `erf<=1`:

```text
A(t)<=exp(t)/sqrt(2sqrt(pi)sqrt(exp(2t)-1)).         (9)
```

The right side decreases and then increases, so its slab maximum is attained
at an endpoint. Modes `1<=n<=96` use (6)-(8). Modes `n>=97` use the geometric
squared-mode tail already proved in the Brownian-majorant note; because
`97*0.02>1`, its high-mode hypothesis holds on every slab.

For `0<t<=0.02`, the earlier closed-form bound is increasing and gives

```text
sup rho_raw(t)<=0.003929361777.                      (10)
```

Thus all positive times in the first window are covered.

## Certified result

The production partition gives

```text
peak time slab                         [0.139,0.140]
Brownian angular-L2 upper               0.692031306564
complete raw spatial-L2 upper           1.636563918201
complete interval-factor upper          0.952384193963
maximum omitted squared-mode sum        1.90335e-9.     (11)
```

An independent refinement to time width `0.0005` and radius width `0.005`
tightens the raw bound to `1.632354689927` and the interval factor to
`0.949934670036`. A separate 128-mode, `M_*=6` run gives
`0.952384193389`, changing the production factor by less than `6e-10` while
reducing the omitted squared-mode tail below `1.6e-12`. The production
theorem retains the coarser value in (11).

This closes the isolated first-window response budget with about `4.76%`
headroom. It does not certify the exact inverse-Laplace mode values and it
does not finish the complete return theorem. The retained continuum spectral
block, polygon-to-circle conormal map, and later-window low modes remain to
be composed.

The calculation is reproduced by
`scripts/neutral_strip_first_window_maximum_bridge_certificate.py`.
