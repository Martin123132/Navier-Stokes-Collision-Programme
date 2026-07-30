# Neutral-strip branch resolvent pilot

## Purpose

The stopped strip supplies a positive weighted survival-tail margin, but that
does not determine how much mass reaches the inner return circle and how much
reaches the outer walls. This note resolves those events separately in the
static affine model before attempting the harder boundary-density norm.

## Competing resolvents

On

```text
Omega_Y={x^2+y^2>1, |y|<Y},       Y=2.1,
```

let `tau` be the first hit of either `r=1` or `|y|=Y`. For

```text
L_rho=Delta-x partial_x-rho y partial_y,
0<=rho<=1,                                             (1)
```

define

```text
p_R=E[1_{r_tau=1}],       p_S=E[1_{|y_tau|=Y}],

m_R=E[exp(c_rho tau)1_{r_tau=1}],
m_S=E[exp(c_rho tau)1_{|y_tau|=Y}],
c_rho=(1-rho)/2.                                      (2)
```

The functions `p_j` solve `L_rho p_j=0` with complementary Dirichlet data.
The residual moments solve

```text
(L_rho+c_rho)m_j=0                                    (3)
```

with the same data. The exact strip spectral bound proves that (3) is below
the first pole for every `rho` in this interval.

## Discretization

The pilot uses a centered monotone nearest-neighbor continuous-time Markov
generator on Cartesian grids. At a coordinate edge crossing the circle, an
unequal-step Shortley-Weller stencil uses the exact circle intersection
rather than moving absorption to the neighboring node center. The disk and
strip walls are absorbing; the artificial boundaries `x=+-X` are reflecting.
The working calculations use

```text
Y intervals       30, 40, 50,
mesh widths       0.14, 0.105, 0.084,
X                 4.2,
entry angles      64,
rho               0, .25, .5, .75, 1.                (4)
```

Additional `X=3.15,4.2,5.25` rows stress the artificial truncation. Boundary
values are included in bilinear interpolation to evaluate the exact `r=2`
entry circle rather than snapping it to grid nodes.

The discrete partition identity

```text
p_R+p_S=1                                             (5)
```

holds to linear-solve precision, and all computed resolvents are
nonnegative. Mesh and truncation spreads are reported rather than hidden.

## Scalar stress diagnostic

For orientation only, insert (2) into

```text
C_scalar=g_H^2[m_R^2+(s_cubic m_S)^2],                (6)

g_H=1.145614144998,
s_cubic=0.639292608019.
```

Equation (6) is not the final renewal norm. It treats the residual moment as
a scalar branch multiplier, does not include the finite axial return patch,
and provisionally assigns the cubic split factor to every wall hit.

The important outcome is qualitative and adverse: entries near the wall are
split dominated and can lie below one, while entries on the strongly
returning axis produce `C_scalar>1` in the weakly transverse-contracting
rows. Therefore the positive strip spectral margin alone is insufficient.
The finite axial patch and the dynamic boundary-density norm cannot be
dropped from the next calculation.

The exact axial patch and the complete wall deformation moment are inserted
in `neutral_strip_axial_patch_branch_pilot.md`. That sharper scalar pilot is
positive; it does not erase this raw-stress negative control.

## Next gate

The next pilot must propagate the killed semigroup and record both boundary
fluxes as functions of time and boundary position. The inner flux must be
composed with the exact outward-OU probability of landing in `|z|<3/4`.
Both branches must then be measured in physical space-time `L2` density
norms. Only those gains may be inserted into

```text
a_S^2+a_R^2<1.                                       (7)
```

The finite-state construction, convergence rows, and scope checks are
reproduced by `scripts/neutral_strip_branch_resolvent_pilot.py`.
