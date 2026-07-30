# Anisotropic affine Poisson transfer pilot

## Purpose and status

The general affine spectral floor shows that constant trace-free eigenvalue
mismatch does not destroy the localized Dirichlet gap. The next question is
whether the complete boundary visit and cutoff-form constants survive when
the transverse affine drift is anisotropic.

This note gives a converged finite-element pilot over the complete ordered
spectrum. It is evidence for a uniform transfer theorem, not a rigorous
enclosure and not a Navier-Stokes regularity result.

## Reversible continuation

Normalize the ordered forward strain spectrum as

```text
(-1-t,t,1),                    -1/2<=t<=1,
kappa=(1+2t)/4,                0<=kappa<=3/4.          (1)
```

In the unit core, the transverse reversible potential is

```text
Phi_t(r,theta)=r^2/4+kappa r^2 cos(2 theta).           (2)
```

For this pilot, hold (2) constant along radial rays in the shell. The full
transverse weight is therefore

```text
w_t=exp[(1/4)min(r^2,1)
        +kappa min(r^2,1)cos(2 theta)].                (3)
```

Equation (3) is continuous at `r=1`, returns exactly to the old radial weight
at `t=-1/2`, and induces the same normalized angular measure on the inner and
outer circular interfaces. It couples angular modes but leaves the axial OU
modes separated.

There is an important scope limitation. The ray-constant shell drift is a
reversible transfer model, not a divergence-free continuation of the affine
Navier-Stokes velocity. The old Brownian-radial shell with retained axial
drift had the same physical idealization. A final proof must replace this
shell by an actual incompressible transport or charge its discrepancy to a
controlled transfer term.

## Discretization

The transverse disk `r<2` is triangulated by concentric polygonal rings. Mesh
nodes are placed exactly at `r=1` and at the cutoff support `r=1.91`; the
short collar `1.91<r<2` is refined independently. Piecewise-linear finite
elements assemble the weighted form

```text
a_t(u,v)=integral w_t[grad u dot grad v
                      +(zeta-1_(r<1))uv].             (4)
```

For each outer angular boundary vector, the sparse Dirichlet problem gives
the inner visit matrix `B`. The audit computes its singular norm in the
weighted angular `L^2` measure, the inner Green trace norm `D_i`, and the
cutoff energy operator.

For a baseline-harmonic `u`, the cutoff energy is evaluated through

```text
a_t(zeta_c u,zeta_c u)
 =integral |grad zeta_c|^2 u^2 w_t,                   (5)
```

which avoids cancellation and resolves the narrow collar directly.

## Axisymmetric calibration

At the principal axial eigenvalue, the three mesh levels give:

| angular nodes | `||B||` | `||D_i||` | `E_zeta` | `chi_P` |
|---:|---:|---:|---:|---:|
| 32 | 0.857832 | 0.514676 | 25.1660 | 4.19539 |
| 48 | 0.857268 | 0.515581 | 26.0574 | 4.27561 |
| 64 | 0.857071 | 0.515898 | 26.5614 | 4.31908 |

These move toward the independently certified radial values

```text
||B||=0.85681683,       ||D_i||=0.51630576,
chi_P=4.35602663.                                      (6)
```

The remaining error is dominated by the narrow collar energy. The monotone
calibration trend is useful, but it is not an upper/lower enclosure theorem.

## Spectrum stress test

On the finest pilot mesh, seven equally spaced `t` values give:

| `t` | `||B||` | `||D_i||` | `E_zeta` | `chi_P` | allowable `alpha` |
|---:|---:|---:|---:|---:|---:|
| -0.50 | 0.857071 | 0.515898 | 26.5614 | 4.31908 | 0.126251 |
| -0.25 | 0.856853 | 0.513759 | 26.6649 | 4.31960 | 0.126310 |
| 0.00 | 0.856206 | 0.507420 | 26.9767 | 4.32116 | 0.126487 |
| 0.25 | 0.855153 | 0.497116 | 27.5004 | 4.32369 | 0.126776 |
| 0.50 | 0.853727 | 0.483214 | 28.2421 | 4.32712 | 0.127167 |
| 0.75 | 0.851972 | 0.466186 | 29.2106 | 4.33136 | 0.127649 |
| 1.00 | 0.849940 | 0.446572 | 30.4173 | 4.33629 | 0.128207 |

The condition number worsens by only about `0.4%`. The baseline visit norm
improves enough that the admissible relative perturbation fraction also
improves. On this sampled family, the axisymmetric endpoint remains worst.

## Axial mode check

At both `t=-1/2` and `t=1`, the first five axial modes were tested. The visit
norm, inner Green norm, and cutoff energy are all maximized at the principal
mode. This is the finite-element counterpart of the old form-ordering
argument. The per-mode ratio `sqrt(D_i E)/B` grows for high modes only because
`B` becomes very small; the full abstract estimate uses the separate operator
suprema, all attained at mode zero in this test.

## Consequence and next gate

The numerical evidence says that extending the affine reference from the
axisymmetric spectrum to the full trace-free spectrum costs essentially no
Poisson budget for the reversible model. In particular, it gives no sign of
an interior bad spectrum hidden between the two endpoint geometries.

What is not yet certified:

1. the finite-element values are converged approximations, not interval or
   complementarity bounds;
2. only seven spectrum values are sampled;
3. the shell continuation is not an incompressible physical construction;
4. translating and rotating cell labels remain unresolved.

The next analytic step is to derive a parameter-uniform form comparison or a
verified eigenvalue enclosure for (3), while separately designing an
incompressible shell/frame transfer. The complete pilot is reproduced by
`scripts/anisotropic_poisson_transfer_pilot.py`.

The exact obstruction to a smooth reversible incompressible taper, and the
compact full-affine alternative forced by that obstruction, are developed in
`reversible_shell_rigidity.md`.
