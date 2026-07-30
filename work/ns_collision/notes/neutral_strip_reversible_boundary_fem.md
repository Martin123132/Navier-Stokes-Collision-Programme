# Reversible boundary-fitted neutral-strip pilot

## Purpose

The fitted Shortley-Weller chain computes scalar hitting probabilities but
does not supply a continuum boundary-`L2` density: its killed-edge law is
atomic at fixed mesh, and the curved stencil is not exactly reversible. This
note replaces that measuring discretization for the static `rho=0` strip.

The replacement passes its structural and coupled-refinement gates. It is
still a floating-point pilot, not a continuum trace certificate.

## Body-fitted mesh

The polygonal domain approximates

```text
{(x,y): x^2+y^2>1, |y|<2.1, |x|<4.2}.              (1)
```

A constrained Delaunay triangulation preserves three marked boundary pieces:

1. the inner unit-circle polygon;
2. the physical strip walls `|y|=2.1`;
3. the artificial sides `|x|=4.2`.

The artificial sides are an explicit absorbing truncation branch. They are
not silently reflected. Uniform points on `r=2` are inserted as exact mesh
vertices, so entry-angle refinement is coupled to each mesh.

The implementation uses `triangle==20250106`, a Python wrapper for the
Triangle constrained-Delaunay mesher.

## Positive reversible generator

For `rho=0`, the transverse affine generator has divergence form

```text
L=mu^(-1) div(mu grad),       mu(x,y)=exp(-x^2/2).   (2)
```

Let `m_i` be the invariant-weighted lumped P1 mass at a live vertex. For an
edge `ij`, let `w_ij` be the constrained-Delaunay cotangent conductance and
evaluate `mu` at the edge midpoint. The chain rates are

```text
c_ij=mu_ij w_ij,
q_ij=c_ij/m_i.                                      (3)
```

All retained `c_ij` are positive. Cocircular zero edges below `1e-12` are
discarded as roundoff. Symmetry of `c_ij` gives exact discrete detailed
balance

```text
m_i q_ij=m_j q_ji.                                  (4)
```

The largest relative error in (4) over three meshes is below `3.5e-16`.
The algebraic and resolvent hitting partitions include inner, wall, and
truncation loss and close within `5e-15` at entry points.

## Physical inner faces

The absorbed rate into each inner Dirichlet vertex is assigned to the dual
arc formed by half of each adjacent true-circle arc. These dual faces are
positive, disjoint, and sum exactly to

```text
sum_b |Gamma_b|=2 pi.                               (5)
```

Thus the discrete boundary norm has the fixed physical meaning

```text
||h_h(t)||_L2(S1)^2
  =sum_b flux_b(t)^2/|Gamma_b|.                     (6)
```

There is no independent histogram-bin parameter and no fixed-mesh atomic
refinement limit.

## Coupled refinement

Using the same axial OU factors, time schedule, and conservative five-percent
interval stress as the earlier diagnostic gives

```text
spacing   states   inner faces   maximum response
0.16       2076        40         0.618929914899
0.12       3738        56         0.620117729331
0.09       6625        72         0.621054704444.    (7)
```

The finest response change is

```text
0.000936975112.                                     (8)
```

The finest terminal state at `t=12` is `1.1311e-15`. The maximum artificial
`x`-truncation probability is stable near `0.00166`. The maximum return
probability is stable near `0.7761`; the maximum wall probability is near
`0.91746`, at a different entry direction.

The worst response remains on the transverse axis. Small mesh asymmetry can
select angle `0` or `pi`; the maximum value itself is stable.

## What this does and does not repair

This construction repairs the two defects isolated by the previous no-go:

- the generator is positive and reversible;
- the boundary flux is measured on disjoint physical faces under coupled
  refinement.

It does not yet prove that `0.621054704444` is a continuum upper bound. The
following remain open:

1. polygonal-circle and weighted mass/stiffness consistency errors;
2. certified maxima inside every time window;
3. a rigorous semigroup tail replacing the fitted tail;
4. removal or enclosure of the `x` truncation;
5. uniformity over `0<=rho<=1`;
6. propagation through the wall-migration-child-return composite.

The next stage should stress mesh and `x` width, then exploit exact mass
reversibility to enclose the finite-dimensional time tail spectrally. The
construction and audit are reproduced by
`scripts/neutral_strip_reversible_boundary_fem_pilot.py`.

**Later development.** The fitted tail has now been replaced by a Barta and
boundary-operator spectral bound for each stored matrix, and an `x`-width
sweep is stable to `0.00070943`. A later uniformization certificate also
encloses finite-time maxima and scalar quadrature for the symmetrized stored
matrices. Continuum certification remains open. See
`neutral_strip_reversible_spectral_tail_width.md` and
`neutral_strip_reversible_finite_time_certificate.md`.
