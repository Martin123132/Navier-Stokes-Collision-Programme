# Radial collar trace calibration

## Purpose

The critical collar-to-form theorem reduces the perturbation gate to

```text
chi_dyn=C_col(d)sqrt(h[zeta U]/m_0)/g_0.              (1)
```

This note asks whether a useful value such as `chi_dyn<2` is numerically
plausible for a concrete protected core. It gives three finite-element
pilots: the minimum cutoff energy, the complete stationary axisymmetric
collar trace norm, and a time-harmonic stress test.

None of these calculations is an enclosure or a replacement for the full
nonautonomous theorem.

## Protected support

For `0<d<0.75`, take

```text
E_d={r<=1-d, |z|<=0.75-d}.                            (2)
```

Thus the perturbation has a radial collar before the entry surface `r=1`
and an axial collar before the absorbing caps. This support is a proposed
localization geometry; it has not yet been derived from a Leray solution or
the moving cubic partition.

## Optimal cutoff energy pilot

Among all `v=zeta U` with `zeta=1` on `E_d` and zero trace on the absorbing
boundary, the least possible comparison energy is the variational problem

```text
E_d^cut=min h[v],
h[v]=int_D(|grad v|^2-v^2),
v=U on E_d, v=0 on partial D.                         (3)
```

The continuous minimizer solves `(-Delta-1)v=0` outside `E_d`. Symmetry and
uniqueness make it axisymmetric and even, so the two-dimensional solve does
not discard a cheaper non-axisymmetric cutoff. The finest `240x180`
axisymmetric `Q1` pilot gives:

| `d` | `E_d^cut` | `sqrt(E_d^cut/m_0)/g_0` |
|---:|---:|---:|
| 0.10 | 62.2880 | 2.67355 |
| 0.20 | 41.7547 | 2.18897 |
| 0.30 | 30.1486 | 1.86003 |
| 0.40 | 21.9759 | 1.58804 |

The medium-to-fine relative changes are below `1.9e-4` in every row. These
values are converged pilots, not certified upper bounds.

## Complete stationary collar norm

For the axisymmetric affine family

```text
B_a=diag(a,a,-2a),        -1<=a<=1/2,                 (4)
```

solve

```text
(-Delta-B_a y dot grad-1)w=0                         (5)
```

on `D\E_d`, with zero absorbing trace and arbitrary data on the complete
inner interface `partial E_d`. Instead of sampling interface profiles, the
pilot computes the exact discrete operator norm

```text
C_stat(d,a)
 =sup |w(r=1,z=0)|/||w||_(L2(D\E_d)).                (6)
```

The optimization permits signed data, so it is an upper benchmark for the
nonnegative overshoot class. Its optimizer is in fact sign-changing. The
worst sampled strain is the radial-contraction endpoint `a=-1`:

| `d` | worst `C_stat` | cutoff factor | `chi_stat` |
|---:|---:|---:|---:|
| 0.10 | 1.62060 | 2.67355 | 4.33276 |
| 0.20 | 0.857872 | 2.18897 | 1.87786 |
| 0.30 | 0.609009 | 1.86003 | 1.13278 |
| 0.40 | 0.483039 | 1.58804 | 0.767083 |

Coarse-to-fine trace changes are below `0.61%`. The thin `d=0.1` collar is
not viable at the desired condition-number scale. Every sampled collar
`d>=0.2` remains below two.

## Temporal frequency stress test

For each fixed matrix in (4), the harmonic problem is

```text
(-Delta-B_a y dot grad-1+i omega)w_omega=0.           (7)
```

At `d=0.2`, `a=-1`, the static-worst claim is false: the trace norm rises
from `0.857872` at zero frequency to approximately `0.860722` near
`omega=10`, then falls. The corresponding combined value is

```text
chi_harmonic=1.88409<2.                               (8)
```

The sampled `d=0.3` and `d=0.4` families are largest at zero frequency.
A coarse scan through the complete interval `-1<=a<=1/2` keeps the maximum
at `a=-1`; it does not reveal an interior strain maximum.

Equation (8) is encouraging but limited. A supremum over harmonic
frequencies for fixed `B_a` is not the causal `L^infinity_t L^2_x` norm for
switching, rotating, or non-axisymmetric measurable histories.

## Conditional budget at the tight pilot

Inserting `chi=1.88409` into the exact sector margin gives the illustrative
one-error tolerances

```text
||q_+||_(3/2)<0.08635,
||e||_3<0.03761.                                      (9)
```

With equal relative sector shares, the corresponding masses are

```text
||q_+||_(3/2)<0.04359,
||e||_3<0.01862.                                      (10)
```

These are conditional normalized budgets, not Navier-Stokes hypotheses
already obtained from the solution.

## Revised gate

The numerical evidence supports `d>=0.2` and rejects spending effort on the
thin `d=0.1` geometry. The next analytic target is now sharply defined:

1. prove an explicit causal collar trace bound for every admissible
   measurable full-affine history, with target `C_col(0.2)<0.914` or use the
   more generous `d=0.3` geometry;
2. enclose the variational cutoff energy rather than relying on mesh
   convergence;
3. realize `E_d` through the stopped moving partition without hiding the
   errors in the collar or paying a larger IMS/restart cost.

The cutoff, stationary operator, and frequency calculations are reproduced
by `scripts/radial_barrier_cutoff_energy_pilot.py`,
`scripts/radial_collar_trace_pilot.py`, and
`scripts/radial_collar_frequency_pilot.py`.
