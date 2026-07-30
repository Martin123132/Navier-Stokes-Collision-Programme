# Radial cubic partition and collar optimization

## Purpose

The complete Poisson theorem requires the adverse non-affine potential to be
supported inside a radial collar. It does not require an axial cutoff: the
form estimate already permits arbitrary axial dependence throughout the
finite cylinder. This makes a transverse two-dimensional partition the
correct static localization object for one already selected cylinder.

A global partition into finite cylinders still needs an axial window or an
equivalent visit decomposition. Both versions are calculated below. The
transverse-only budget applies inside one fixed visit; the full tensor budget
is the conservative global fixed-frame option.

A naive tensor cutoff with support square `[-1.5L,1.5L]^2` does not fit in the
radial core `rho<=1.5L`; its corners reach `sqrt(2)*1.5L`. This note constructs
a compact partition whose complete square support fits inside the radial
core and optimizes the tradeoff between its IMS cost and the Poisson collar
constant.

## Cardinal cubic partition

Let `N_3` be the cardinal cubic B-spline, supported on `[0,4]`, and choose a
knot spacing `h`. In the transverse plane define

```text
phi_jk(x,y)=N_3(x/h-j) N_3(y/h-k),
chi_jk=sqrt(phi_jk).                                  (1)
```

The cardinal partition identity gives

```text
sum_jk phi_jk=1,       sum_jk chi_jk^2=1.             (2)
```

At most `4^2=16` weights are nonzero at one point. Each support is a square
of half-width `2h`. It lies in `rho<=rho_s L` provided

```text
2 sqrt(2) h=rho_s L.                                  (3)
```

The linear weights `phi_jk` therefore preserve the exact pressure partition
and neighbor-flux cancellation.

## Exact IMS bound

On one unit knot cell, four cubic splines are active. Direct simplification
gives

```text
I_3(x)=(1/4) sum_j N_3'(x-j)^2/N_3(x-j)
      =-3(9x^4-18x^3+2x^2+7x+2)
        /[2(3x^3-6x^2+4)(3x^3-3x^2-3x-1)].           (4)
```

The exact polynomial root check in the audit proves on `0<=x<=1` that

```text
I_3(x)<157/200.                                        (5)
```

The sampled sharp maximum is `0.784204769`; the rational upper `0.785` leaves
a visible verification margin. Tensorization and (3) give

```text
L^2 sum_jk |grad chi_jk|^2
 <=2(157/200)(L/h)^2
 =314/(25 rho_s^2).                                   (6)
```

This is a pointwise IMS bound, not an average cutoff estimate.

For a full finite-cylinder tensor partition, choose axial knot spacing

```text
h_z/L=3/4,
```

so the axial cubic support has half-width `2h_z=1.5L`. Its additional
dimensionless IMS cost is

```text
(157/200)(L/h_z)^2=314/225=1.39555556.                 (7)
```

## Collar optimization

At `R_*=0.5`, the available dimensionless transverse form margin before IMS
is

```text
m L^2/nu=5.29680963.                                   (8)
```

The support radius changes both sides of the calculation. Increasing
`rho_s` lowers (6), but narrows the unperturbed collar and reduces the
allowable Poisson relative-form parameter. The audited rows are:

| `rho_s` | transverse IMS | transverse `Q/nu` | full IMS | full `Q/nu` |
|---:|---:|---:|---:|---:|
| 1.50 | 5.582222 | 0 | 6.977778 | 0 |
| 1.60 | 4.906250 | 0.381872 | 6.301806 | 0 |
| 1.70 | 4.346021 | 0.581211 | 5.741576 | 0 |
| 1.75 | 4.101224 | 0.597517 | 5.496780 | 0 |
| 1.80 | 3.876543 | 0.581507 | 5.272099 | 0.023897 |
| 1.90 | 3.479224 | 0.465174 | 4.874780 | 0.214009 |
| 1.91 | 3.442888 | 0.446228 | 4.838444 | 0.215900 |
| 1.92 | 3.407118 | 0.425429 | 4.802674 | 0.215151 |

The transverse-only optimum is

```text
rho_s=1.75,       taper radius=2,
h/L=1.75/(2 sqrt(2)),
Q/nu<0.5975.                                           (9)
```

After adding (7), the full three-dimensional tensor optimum moves to

```text
rho_s=1.91,       taper radius=2,
h_perp/L=1.91/(2 sqrt(2)),       h_z/L=3/4,
total IMS cost=4.83844363,
residual margin=0.45836599,
Q/nu<0.2159.                                          (10)
```

Each final budget recomputes the sharp Sobolev relative-form conversion using
the appropriate residual margin and then applies the profile-specific
Poisson `alpha`. These are sufficient scale-invariant budgets for the
positive perturbation, not assertions that Navier-Stokes already supplies
the required smallness.

## Dyadic compatibility

The cubic spline has the positive two-scale identity

```text
N_3(x)=sum_(r=0)^4 [binom(4,r)/8] N_3(2x-r).          (11)
```

Thus a fixed uniform level and its dyadic child level both have exact
quadratic and pressure partitions, with no moving centers. Equation (11) is
the right algebra for a conservative parent/child transfer.

It does not by itself finish the time-dependent cover. A proof must still
show that monotone level changes can be charged in one common physical or
ground-state norm without paying a fixed gauge comparison or a complete
Poisson conversion at every generation. Spatially adaptive coarse/fine
interfaces need the same care; simply tensoring three simultaneous interface
transitions can spend more than (8).

## Status

The fixed-frame full tensor partition is now explicit and passes all four
immediate gates:

1. exact quadratic partition;
2. compact support inside the radial and axial cylinder boundaries;
3. absorbable pointwise IMS cost;
4. exact linear pressure partition.

The live localization problem has narrowed from constructing an unspecified
broad cutoff to proving a time-coherent parent/child visit transfer for the
positive refinement mask, followed by deriving the actual local
`L^(3/2)` perturbation bound from Navier-Stokes coherence. A spatially varying
cylinder axis is not automatic; its frame variation remains part of the
non-affine error.

The identities, rational Fisher bound, refinement relation, profile sweep,
and optimized budget are reproduced by
`scripts/radial_cubic_partition_audit.py`.
