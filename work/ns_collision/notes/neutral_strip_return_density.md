# Neutral-strip return density pilot

## Purpose

The wall-stopping trace theorem reduces the return response to the raw
space-time `L2` density of the weighted inner-boundary flux. This note
computes that density for the working static neutral strip at `rho=0` using
the same finite-state generator as the scalar branch audit.

It is a convergence pilot, not a continuum certificate.

**Later correction.** The fixed 32-bin comparison below does not control
angular refinement. At fixed interior mesh the fitted-edge boundary law is
atomic, and its histogram response diverges like `B^(1/4)` after the atoms
separate. Therefore the reported `K_R` and thresholds are coarse finite-state
diagnostics, not continuum upper bounds. See
`neutral_strip_boundary_density_discretization_no_go.md`.

## Boundary-resolved generator

The existing monotone generator records aggregate killing rates at the inner
circle and strip walls. Each killed edge is now also recorded with its source
node, rate, and exact fitted-boundary intersection. Summing the resolved
inner and wall edges reproduces the original aggregate rate arrays exactly.

The generator matrix itself is unchanged.

For `B=32` angular bins, let `R_B` be the sparse matrix of rates from interior
states into inner-circle bins. If `P_theta` is the bilinear initial state for
an entry point on `r=2`, the transverse absorption density is

```text
h_theta(t)=P_theta^T exp(tG) R_B.                    (1)
```

Near the top of the entry circle, bilinear interpolation touches the
absorbing strip wall. Its return data are zero, so that interpolation weight
is omitted without renormalization; it belongs to the competing wall branch.

## Axial and deformation factors

At `rho=0`, the independent axial process has variance

```text
V(t)=exp(2t)-1.                                      (2)
```

For its Gaussian density `g_t`, restricted to `|z|<H`, `H=3/4`,

```text
int_(-H)^H g_t(z)^2 dz
 =erf(H/sqrt(V))/(2sqrt(pi)sqrt(V)).                 (3)
```

The raw weighted return-density norm is therefore

```text
rho_theta(t)
 =exp(t)||h_theta(t)||_L2(S1)
  ||g_t 1_|z|<H||_L2(R).                            (4)
```

The scalar time integral using the corresponding Gaussian patch mass is
checked against the older implicit-time patched-return resolvent.

## Interval factor

For each of 32 entry angles, the pilot propagates (1) over `0<t<=12`, using
fine sampling around the peak and coarser sampling in the tail. It evaluates

```text
J_raw(theta)
 =inf_ell [ell+1/m_0]
  sum_n sup_(t in I_n) rho_theta(t),                 (5)

m_0=4.832287335665.
```

The sampled envelope is inflated by five percent. The uncomputed tail is
bounded in the pilot by an exponential fitted to the terminal samples and
then slowed by twenty percent. These are numerical stresses, not rigorous
tail bounds.

With scalar branch gain `p_R(theta)`, the finite-state response is

```text
K_R(theta)
 =sqrt[p_R(theta) C_4 J_raw(theta)],
C_4=0.674148137961.                                  (6)
```

## Working result

The `30`- and `40`-interval meshes give

```text
maximum change in J_raw       0.00196814,
maximum change in K_R         0.00114174.             (7)
```

The worst return response occurs at the transverse-axis entry. Inserting
the `40`-interval response into the existing wall-fixed pair criterion gives
the conditional return-only thresholds

```text
||q_res||_(3/2)<0.366023205860,
||e_res||_3      <0.107962714456.                    (8)
```

For the potential row, the interval factor is conservatively inflated by
`(1-alpha)^(-3)`, where

```text
alpha=0.220329037686 ||q_res||_(3/2).                (9)
```

The numbers in (8) are materially smaller than the borrowed normalized
Brownian-envelope calibration, but still leave visible model margin.

## Scope and next gate

This pilot does not certify (8) for the continuum strip. Boundary angular
binning, time-window maxima, the fitted tail, and mesh convergence need
enclosures. It also covers only the static `rho=0` affine endpoint and no
nonaffine Navier-Stokes drift.

The immediate sequence is:

1. refine boundary bins and time windows and certify the `rho=0` flux;
2. extend the envelope uniformly over `0<=rho<=1`;
3. propagate wall flux through migration and one child storage return to
   obtain the composite `K_S` law on `r=1`.

The boundary-resolved semigroup and density arithmetic are reproduced by
`scripts/neutral_strip_return_density_pilot.py`.
