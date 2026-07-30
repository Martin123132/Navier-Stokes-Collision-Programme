# Neutral-strip first-window Brownian majorant

## Purpose

The parabolic spectral split pays every continuum high mode after
`ell=3/8`, but its diagonal-kernel argument deliberately leaves
`0<t<=ell` open. This note uses the unit gap from the entry circle `r=2` to
the absorbing circle `r=1`. It proves a finite, explicit first-window flux
bound. The first closed-form constant is conservative and does not yet close
the response budget.

## OU-to-Brownian domination

The transverse diffusion has generator

```text
L=Delta-x partial_x.                                (1)
```

Against planar Brownian motion with generator `Delta`, its stopped Girsanov
density is

```text
Z_t=exp((x_0^2-X_t^2+2t-int_0^t X_s^2 ds)/4).      (2)
```

Every entry point has `x_0^2<=4`, so

```text
Z_t<=exp(1+t/2).                                    (3)
```

This remains valid on the inner-circle first-hit event. Removing the strip
walls and finite sides only enlarges that positive hitting measure. Hence the
OU time-angle hitting density is pointwise dominated by `exp(1+t/2)` times
the unrestricted planar Brownian density.

## Exact Brownian modes

Let `h(t,theta)` be the Brownian first-hit density on the unit circle,
starting at radius two, and write

```text
h(t,theta)=(1/(2pi))[c_0(t)+2 sum_(n>=1)c_n(t)cos(ntheta)]. (4)
```

The standard disk hitting formula gives

```text
Laplace[c_n](p)=K_n(2sqrt(p))/K_n(sqrt(p)),          (5)
||h(t)||_2^2=(c_0(t)^2+2 sum_(n>=1)c_n(t)^2)/(2pi). (6)
```

Equation (5) follows equivalently from the skew-product representation and
the Bessel first-passage transform. References are Hamana and Matsumoto,
*Brownian Hitting to Spheres* (2023), `arXiv:2301.03756`, and Hamana and
Matsumoto,
*The probability densities of the first hitting times of Bessel processes*
(2012), `arXiv:1206.2120`.

## Bessel absolute continuity

Use one-dimensional Brownian motion with generator `Delta`, started at two
and stopped on first reaching one, as reference measure. Put

```text
A_t=int_0^t R_s^(-2) ds,
q_d(t)=d/(2sqrt(pi)t^(3/2)) exp(-d^2/(4t)).          (7)
```

Absolute continuity from this process to the Bessel process of index `n`
has stopped density

```text
(R_t/2)^(n+1/2) exp((1/4-n^2)A_t).                  (8)
```

The factor `2^n` connecting the Bessel hitting law to (5) cancels the
index-dependent endpoint power. Therefore

```text
c_0(t)<=2^(-1/2)e^(t/4)q_1(t),                     (9)
c_n(t)=2^(-1/2)q_1(t)
         E[exp(-(n^2-1/4)A_t) | tau_1=t], n>=1.    (10)
```

The inequality in (9) uses `A_t<=t`, since `R_s>=1` before absorption.

## Excursion split

Fix `M>2`. On paths staying below `M`, `A_t>=t/M^2`. A path that crosses
`M` before hitting one must travel first from two to `M` and then from `M`
to one. The strong Markov property and the stable first-passage convolution

```text
q_(M-2) * q_(M-1)=q_(2M-3)                         (11)
```

give the explicit mode bound

```text
c_n(t)<=2^(-1/2)[q_1(t)exp(-(n^2-1/4)t/M^2)
                  +q_(2M-3)(t)].                   (12)
```

This is the key gain: the angular clock controls ordinary paths, while a
large-radius path pays its extra radial distance.

Choose

```text
M_n(t)=2+sqrt(nt).                                  (13)
```

Writing the two ratios in (12) as `g_n` and `b_n` gives

```text
g_n=exp(-(n^2-1/4)t/(2+sqrt(nt))^2),
b_n=(1+2sqrt(nt))exp(-sqrt(n/t)-n).                 (14)
```

Since `(g_n+b_n)^2<=2(g_n^2+b_n^2)`, all squared modes are summable. For
`nt<=1`,

```text
g_n^2<=exp(-n^2t/6),                                (15)
```

and for `nt>1` (which forces `n>=3` in this window),

```text
g_n^2<=exp(1/54)exp(-2n/9).                         (16)
```

Also

```text
b_n^2<=(2+8nt)exp(-2n).                             (17)
```

The Gaussian integral and two geometric series now bound the complete
infinite sum in (6).

## Uniform first-window bound

After collecting (9) and (15)-(17), the audit obtains an explicit constant
`C_mode` such that

```text
||h(t)||_2
 <= C_ang t^(-7/4) exp(-1/(4t)),  0<t<=3/8.        (18)
```

The axial patch factor satisfies

```text
A(t)=exp(t)||g_t 1_(|z|<3/4)||_2
 <= C_ax exp(t)t^(-1/4).                            (19)
```

Combining (3), (18), and (19) gives

```text
rho_raw(t)
 <= e C_ang C_ax t^(-2)exp(-1/(4t)+3t/2).          (20)
```

The right side has its unique first-window maximum at

```text
t_*=(4-sqrt(10))/6.                                 (21)
```

Thus (20)-(21) prove a complete continuum first-window spatial-`L2` flux
bound without an FEM limit or inverse-Laplace calculation. The scalar mass
is bounded separately using (9), the global axial scalar factor, and

```text
int_0^ell q_1(t)dt=erfc(1/(2sqrt(ell))).             (22)
```

For `ell=3/8`, the resulting closed-form values are

```text
t_*                                      0.139620389972
uniform raw spatial-L2 upper             4.319730317023
first-window interval-factor upper       2.513829634370
first-window scalar-gain upper           0.736571860339. (23)
```

As a stress check, explicitly summing the path-split bounds through at least
mode 96 and appending the proved tail gives a sampled peak

```text
time                                      0.137522161766
path-split raw spatial-L2 upper           1.857153964351
path-split Brownian angular-L2 upper      0.809457944717
squared omitted-mode tail                 1.79404e-9.     (24)
```

At the same time, Stehfest orders 14 and 16 applied to the exact transform
(5) give Brownian angular norms `0.584985505111` and `0.585041554310`, a
relative spread of `9.58e-5`. These inversion values are diagnostics, not
enclosures.

## What remains

The analytic uniformization deliberately replaces all low angular modes by
coarse path bounds. Its interval factor is above one, so it certifies the
first window but does not close the return-response inequality. The sampled
path-split sum is much sharper but is not a supremum enclosure.

The companion maximum-bridge certificate retains (16)-(17) as the rigorous
high-mode tail and replaces the binary split (12) by positive integration
over the bridge maximum. It encloses the complete time supremum and lowers
the isolated first-window interval factor below one without promoting the
inverse-Laplace diagnostics. The next open gate is the retained continuum
spectral block and polygon-to-circle conormal map.

The calculation is reproduced by
`scripts/neutral_strip_first_window_brownian_majorant_audit.py`.
