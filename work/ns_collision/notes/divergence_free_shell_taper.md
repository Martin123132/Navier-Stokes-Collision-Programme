# Divergence-free localized affine shell

## Purpose

The reversible-shell rigidity theorem showed that a smooth, incompressible,
gradient shell cannot taper a fitted affine strain. This note takes the other
branch: retain exact incompressibility and localization, but allow the shell
operator to be non-self-adjoint.

The construction is exact. Its optimized taper and boundary-transfer
constants are numerical pilots rather than certified enclosures.

## Streamfunction construction

Write the ordered affine spectrum as `(-1-t,t,1)` and put

```text
s=t+1/2,                    0<=s<=3/2.
```

The backward affine drift decomposes into an axisymmetric part and a
two-dimensional traceless part:

```text
b_core=(x/2,y/2,-z)+s(x,-y,0).                         (1)
```

For a radial taper `f(r)`, define

```text
psi=s f(r)xy,
b=(x/2+partial_y psi,
   y/2-partial_x psi,
   -z).                                                (2)
```

The streamfunction contribution has zero transverse divergence, while the
axisymmetric transverse divergence `1` cancels `partial_z(-z)=-1`. Therefore

```text
div b=0                                                (3)
```

for every radial `f`, without a shell interface source. Conditions

```text
f(1)=1, f'(1)=0, f(r_t)=0, f'(r_t)=0                  (4)
```

match the affine core continuously to the axisymmetric outer drift.

## Exact strain reduction

The symmetric transverse gradient of the streamfunction is traceless. Its
two eigenvalues are `+-s sigma(r,theta)`. Direct differentiation gives the
exact angular maximum

```text
sigma_*(r)=max(
  |f+r f'|,
  |f+(3/4)r f'+(1/4)r^2 f''|).                        (5)
```

The forward transverse eigenvalue is at most `-1/2+s sigma_*`, while the
axial eigenvalue remains `1`. At the worst spectrum `s=3/2`, the stretching
excess is therefore

```text
delta_c=(3/2)[sigma_*-1]_+.                            (6)
```

Core matching forces `sigma_*(1)>=1`, so one is an absolute lower bound for
the minimax problem.

## Polynomial minimax taper

Use normalized radius `q=(r-1)/(r_t-1)` and parameterize

```text
f(q)=1-3q^2+2q^3
     +q^2(1-q)^2 sum_k c_k T_k(2q-1).                 (7)
```

Both quantities in (5) are linear in the coefficients. Minimizing their
sampled `L-infinity` norm is consequently a linear program. Degree refinement
at the old taper radius `1.91` converges only to about

```text
sigma_* = 2.14,
delta_c approximately 1.72.                            (8)
```

The old narrow buffer cannot localize the most anisotropic affine field
without creating a large strain overshoot.

Widening the taper gives:

| taper radius | optimized `sigma_*` | worst `delta_c` |
|---:|---:|---:|
| 1.91 | 2.1434 | 1.7152 |
| 2.00 | 1.88 | 1.32 |
| 2.25 | 1.3779 | 0.5668 |
| 2.50 | 1.0986 | 0.1479 |
| 2.60 | 1.0145 | 0.0218 |
| 2.65 | 1.0000011 | 0.00000165 |

The selected taper ends at `r_t=2.65`, followed by an axisymmetric collar to
the payoff boundary `eta=2.75`. Its degree-16 profile is numerically monotone
and reaches the theoretical strain floor to about six decimal places. This
is dense sampling, not interval certification.

## Nonsymmetric visit

After the axial OU separation, the transverse strong operator is

```text
A_t=-Delta-b_perp dot grad+zeta-c_t.                   (9)
```

A piecewise-linear triangular finite-element solve retains the nonsymmetric
advection matrix in (9). At `H/L=1.2`, `zeta_0=1.26030185`, the uniform
angular boundary `L^2` results are:

| `t` | visit norm | generation criterion |
|---:|---:|---:|
| -0.50 | 0.77741 | 0.24075 |
| 0.00 | 0.78069 | 0.24279 |
| 0.50 | 0.78973 | 0.24845 |
| 1.00 | 0.80264 | 0.25664 |

All seven sampled spectra close, with `t=1` worst. At `t=-1/2`, mesh
refinement converges to the independent exact Kummer visit gain
`0.77692977`. Even `H/L=1.5` remains below one at the worst spectrum, though
its generation criterion `0.8483` leaves much less room.

## Nonreversible Doob measure

The finite-element visit matrix is strictly positive, so it has Perron
right/left vectors `h,l` and multiplier `rho`. The conjugation

```text
P=diag(h)^(-1) B diag(h)/rho                           (11)
```

is row-stochastic. Its stationary law is proportional to `l h`. Jensen then
makes `P` contractive in the stationary `L^2` space without requiring
detailed balance. Equivalently, `B` has exact norm `rho` in the observable
measure proportional to `l/h`.

At `t=1`, `H/L=1.2`, the pilot gives

```text
rho=0.78895979,
C_Doob=0.24795909,
||B||_(uniform L2)=0.80264472.                         (12)
```

The Markov row-sum and stationarity residuals are below `3e-15`, and its
stationary `L^2` norm is one to roundoff. The natural observable measure and
uniform angular measure have one-history round-trip mismatch `1.41008`; the
natural cycle permits `2.00821`. Paying the measured conversion still leaves

```text
1.41008^2 C_Doob approximately 0.493<1.               (13)
```

This does not identify the actual Navier-Stokes entry law, but it removes
reversibility as a prerequisite for a canonical contractive visit measure.

## What changed

The earlier apparent choice between an unphysical localized reversible shell
and a physical but unlocalized full-affine shell was incomplete. A third
option exists:

```text
localized + incompressible + non-gradient.            (10)
```

Numerically, a sufficiently wide version of (10) preserves both the strain
ceiling and renewal contraction. It therefore deserves analytic development.

The remaining geometric obligations are a verified polynomial enclosure,
physical entry/exit hitting measures instead of uniform angular `L^2`, and a
conservative transfer between translating and rotating cells. The identities
and minimax sweep are reproduced by
`scripts/divergence_free_shell_taper_audit.py`; the boundary solve is in
`scripts/divergence_free_taper_transfer_pilot.py`.
