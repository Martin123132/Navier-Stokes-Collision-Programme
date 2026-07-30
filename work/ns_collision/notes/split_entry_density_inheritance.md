# Split-entry density inheritance

## Purpose

A true dyadic split changes labels at the same physical point. It is
therefore too strong to demand that the split independently generate a new
space-time density, but it is also wrong to treat the split as smoothing.
This note states exactly what a pointwise Markov split preserves and isolates
the deterministic-time obstruction.

## Density inheritance

Let `f_i(s,x)>=0` be the joint density before a split, indexed by parent
labels. Let `P_ij(s,x)` be the pointwise child-label kernel, with

```text
P_ij>=0,             sum_j P_ij=1.                  (1)
```

The child densities are

```text
g_j(s,x)=sum_i f_i(s,x)P_ij(s,x).                   (2)
```

Hence the physical marginal is exactly preserved:

```text
sum_j g_j(s,x)=sum_i f_i(s,x).                      (3)
```

In particular, every pointwise, spatial, or interval-supremum bound on the
physical density passes unchanged through the label split. Also,

```text
sum_j g_j(s,x)^2<=[sum_j g_j(s,x)]^2,               (4)
```

so the child `ell2` density is bounded by the physical marginal. A child
coordinate map with surface Jacobian `J>=J_min>0` adds at most the factor
`J_min^(-1/2)` to a spatial `L2` density norm.

Equations (1)-(4) are inheritance statements. If the incoming law is a
delta, the child law is still a delta.

## Temporal atom obstruction

The averaged surface theorem controls

```text
int h[w(t)]dt,
```

not `h[w(t_*)]` at one prescribed time. The family

```text
w_N(t,x)=sqrt(N)1_[t_*,t_*+1/N](t)v(x),   h[v]=1,   (5)
```

has unit integrated energy but `h[w_N(t_*)]=N`. Therefore a deterministic
global level change represented by `delta_(t_*)` cannot be inserted into the
averaged surface-trace theorem. Markov relabeling does not repair this.

If a split is downstream of a random exit law that already has an absolute
space-time density, and the split map does not collapse time or geometry,
(3) preserves the existing envelope. No second smoothing theorem is then
needed. The construction must prove that this is the actual ordering.

## Fixed-time volume alternative

The positive-part energy estimate also gives the uniform bound

```text
sup_t ||w(t)||_2^2<=F^2/[(1-alpha)^2 m_0].           (6)
```

Thus a deterministic split time can be handled if its unnormalized child
law is a volume law satisfying

```text
dnu/dx<=M_V/|D|.                                    (7)
```

Indeed,

```text
int_D |w(t,x)|^2dnu
 <=M_V F^2/[|D|(1-alpha)^2m_0].                     (8)
```

For the normalized cylinder `|D|=6pi`, and assigning the full current
split-only allowance `0.08730757` while retaining the legacy return
baseline, the conditional one-error thresholds are:

| `M_V` | `||q||_(3/2)` | `||e||_3` |
|---:|---:|---:|
| 1 | 0.848047 | 0.271168 |
| 2 | 0.634378 | 0.191745 |
| 4 | 0.467720 | 0.135584 |
| 8 | 0.341021 | 0.0958724 |
| 16 | 0.246565 | 0.0677921 |
| 32 | 0.177167 | 0.0479362 |

These values are implications of (7), not physical Navier-Stokes bounds.
The strong numbers reflect that a bounded volume density pairs with the
uniform `L2` estimate rather than a boundary trace.

## Revised split gate

There are now two legitimate routes:

1. prove that every true split inherits an existing unnormalized absolute
   space-time density, including the child-coordinate Jacobian;
2. at deterministic split times, prove a bounded child-volume density and
   use (8).

A bare pointwise relabel supplies neither route. Zero-lag level cascades and
surface-supported child entries must be isolated explicitly. This is the
remaining geometric/probabilistic split gate.

The exact identities, random finite-dimensional stress tests, temporal-atom
counterexample, and volume-density table are reproduced by
`scripts/split_entry_density_inheritance_audit.py`.
