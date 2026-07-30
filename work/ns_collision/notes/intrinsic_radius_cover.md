# Intrinsic Lipschitz radius and comparable-cell cover

## Purpose

The adaptive envelope gives a safe diffusion radius at each point, while the
pressure partition identity says only neighboring weight differences should
be paid. A raw radius can oscillate arbitrarily in space, so adjacent cells
need not have comparable gauges. This note gives a canonical spatial
regularization that preserves the Reynolds cap and time monotonicity.

## Lipschitz minorant

Let `A(x,t)>=a(x,t)>=0` be a pointwise nondecreasing amplitude envelope and
define the raw intrinsic radius

```text
ell(x,t)=sqrt(R_* nu/A(x,t)),       0<R_*<=2.
```

For `kappa>0`, set

```text
rho_kappa(x,t)
 =inf_y [ell(y,t)+kappa distance(x,y)].                  (1)
```

This infimal convolution has three elementary properties:

```text
rho_kappa<=ell,
|rho_kappa(x)-rho_kappa(z)|<=kappa distance(x,z),
rho_kappa(x,t_2)<=rho_kappa(x,t_1)  for t_2>=t_1.        (2)
```

The first follows by choosing `y=x`; the second follows from the triangle
inequality; the third follows because the infimum preserves pointwise order.

Define the inflated reference envelope

```text
A_tilde(x,t)=R_* nu/rho_kappa(x,t)^2.                    (3)
```

Then

```text
A_tilde>=A>=a,
partial_t A_tilde>=0,
a rho_kappa^2/nu<=R_*.
```

Consequently `(A_tilde,rho_kappa)` can replace `(A,L)` in the monotone
envelope gauge. The favorable amplitude/scale sign is retained even though
the raw envelope was not spatially regular.

## Neighbor comparability

Consider balls with radii `rho_i=rho_kappa(x_i)`. Suppose their
`q`-enlargements overlap. Then

```text
distance(x_i,x_j)<=q(rho_i+rho_j).
```

Writing `M=max(rho_i,rho_j)`, `m=min(rho_i,rho_j)`, Lipschitz continuity gives

```text
M-m<=kappa q(M+m).
```

If `theta=kappa q<1`,

```text
M/m<=(1+theta)/(1-theta),                                (4)

max(A_tilde_i,A_tilde_j)/min(A_tilde_i,A_tilde_j)
 <=[(1+theta)/(1-theta)]^2.                              (5)
```

For `kappa=1/8` and `q=2`, the radius ratio is at most `5/3` and the reference
amplitude ratio is at most `25/9`.

A maximal disjoint Vitali subfamily of intrinsic balls therefore covers the
target set after fixed enlargement, and the enlarged family has bounded
overlap by the usual packing argument. The overlap constant depends only on
dimension, `q`, and `kappa`, not on the strain amplitude.

## Numerical nonuniform audit

The audit uses a one-dimensional envelope with two unequal peaks and then
adds a larger peak at a later time. It computes (1) directly on 401 points.
The checks verify that:

1. the regularized radius stays below the raw diffusion radius;
2. its discrete Lipschitz slope is at most `kappa`;
3. it remains nonincreasing after the envelope grows;
4. (3) dominates the raw and actual strain amplitudes;
5. the actual local Reynolds number remains at most two;
6. every pair of overlapping doubled balls obeys (4) and (5).

The example is numerical only to stress the construction on a highly
nonuniform profile; properties (2), (4), and (5) are exact metric
inequalities.

## The kappa tradeoff

Taking `kappa` small makes neighboring radii and reference amplitudes nearly
equal, which is favorable for the pressure edge-flux mismatch. It also lets a
very small raw radius depress `rho_kappa` over a wider spatial region. That
creates more small cells and can worsen covering and renewal costs.

Taking `kappa` larger keeps the radius more local but allows larger gauge
jumps between overlapping cells. The parameter must therefore be chosen by
balancing pressure-flux mismatch against the number and return cost of the
intrinsic cells; neither limit is free.

## Remaining construction

This lemma supplies a spatial cover at each fixed time. A proof still needs a
time-coherent selection of centres and a subordinate partition satisfying:

1. controlled centre velocities and reselection times;
2. cutoff gradients of order `rho_i^(-1)` with bounded overlap;
3. neighboring gauge weights compatible with the pressure edge identity;
4. a shrinking-core return estimate uniform across cell births, deaths, and
   mergers;
5. summability over all cells near a hypothetical first singular time.

`monotone_dyadic_cover.md` replaces arbitrary moving selections by a fixed
dyadic grid. Unsafe cells only split, a 2:1 balance controls neighboring
reference amplitudes, and continuous centre-velocity errors disappear. The
new live issue is transferring deformation and renewal weights from a parent
to its children without accumulating a cost at every generation.

The minorant, monotonicity, Reynolds cap, and comparability checks are
reproduced by `scripts/intrinsic_radius_cover_audit.py`.
