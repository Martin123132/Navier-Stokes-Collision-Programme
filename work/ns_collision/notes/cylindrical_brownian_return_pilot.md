# Cylindrical Brownian return pilot

## Purpose

The corrected averaged-entry theorem prefers a spatial `L2` envelope for the
unnormalized return density. This note computes the exact mode representation
for the Brownian exterior of the infinite unit cylinder, then stress-tests
the resulting time envelope numerically.

This is the correct local geometry for a lateral return from `r=2` to `r=1`.
It includes axial spreading and the logarithmic recurrence of the transverse
two-dimensional exterior. It does not include Navier-Stokes drift or
deformation, and the numerical inversion is not yet an enclosure.

## Exact mode representation

Start Brownian motion with generator `Delta` at

```text
r=2,       theta=0,       z=0,                       (1)
```

and stop on the infinite cylinder `r=1`. Write `h_n(t)` for the angular
Fourier coefficients of the planar exterior-disk hitting density. Separation
of the resolvent gives the exact Laplace transforms

```text
int_0^infinity exp(-lambda t)h_n(t)dt
 =K_n(2sqrt(lambda))/K_n(sqrt(lambda)).              (2)
```

The axial Brownian motion is independent, so the complete surface density is

```text
k(t,theta,z)=h(t,theta)(4pi t)^(-1/2)
              exp[-z^2/(4t)].                       (3)
```

Parseval and the full-line axial Gaussian give

```text
||h(t)||_L2(S1)^2
 =[h_0(t)^2+2 sum_(n>=1)h_n(t)^2]/(2pi),             (4)

||g(t)||_L2(R)=2^(-3/4)(pi t)^(-1/4).                (5)
```

Using the full axial line in (5) is conservative for the finite entry patch
`|z|<0.75`.

## Inversion pilot

The audit inverts (2) with real-axis Gaver-Stehfest formulas, exponentially
scaled Bessel functions, angular modes `0..40`, orders `12,14,16`, and 241
logarithmic times from `0.02` to `10,000`.

Orders 14 and 16 differ by at most `1.60%` for `t>=0.03`, with much smaller
spread near the maximum. The raw order envelope has

```text
peak time                         0.12835,
peak spatial L2 density           0.4313 approximately. (6)
```

The long-time behavior is consistent with

```text
||k(t)||_2=O[t^(-5/4)/log(t)^2].                    (7)
```

This follows heuristically from the planar radial first-hit density
`O[t^(-1)/log(t)^2]`, angular mixing, and the axial `L2` factor `t^(-1/4)`.
The exponent `5/4>1` is just sufficient for the interval-supremum series.

## Inflated stress envelope

For a deliberately conservative diagnostic, the audit uses

```text
max{1.03 max(order 14,order 16),
    1.5 half-space L2 envelope}                      (8)
```

on the sampled interval. Beyond `t=10,000`, it uses `1.5` times the maximum
sampled coefficient in (7). The factors in (8) are safety stresses, not
proved error bounds.

With the certified `H1 -> L4(Sigma)` trace constant, this gives

```text
optimal time window                0.42242,
stress time-energy factor          0.79648,
conditional ||q||_(3/2) budget     0.14239,
conditional ||e||_3 budget         0.03842.           (9)
```

Thus the recurrent cylindrical tail does not numerically exhaust the new
barrier allowance. The physical difficulty remains the deformation-weighted
kernel, not bare Brownian return.

## Finite axial-patch mass

The total probability of hitting `r=1` inside `|z|<H` has the exact Fourier
representation

```text
p_H=(2/pi)int_0^infinity [sin(kH)/k]K_0(2k)/K_0(k)dk. (10)
```

At `H=3/4`, direct quadrature gives `p_H=0.310135151371`. The centered axial
start is worst because the axial Poisson kernel is an even decreasing
Gaussian mixture. This probability materially improves the branch-resolved
pilot, but it is not a weighted Navier-Stokes return estimate.

## Certification gate

The Brownian baseline can be upgraded without Monte Carlo. A certificate
must bound:

1. Gaver-Stehfest replacement error or, preferably, a contour/spectral
   inversion of every retained mode;
2. the angular mode tail uniformly in time;
3. the short-time curved-boundary comparison;
4. the `t^(-5/4)/log(t)^2` long-time coefficient and crossover.

Only then should the Navier-Stokes perturbation be added. The weighted
extension must preserve a summable spatial `L2` envelope; a total return-mass
bound by itself cannot do that.

The exact transforms, inversion convergence, stress envelope, and budget
calibration are reproduced by
`scripts/cylindrical_brownian_return_pilot.py`.
