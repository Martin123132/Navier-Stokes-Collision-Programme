# Geometry-triggered migrating child

## Purpose

The neutral-strip scalar pilot closes only if a wall branch starts a genuine
smaller-scale generation. The existing envelope split cannot be assigned to
that event. This note asks what a new geometric half-scale transition would
actually require.

## Nested-child obstruction

Let the complete parent visit start on radius `eta L`, with `eta=2`, and let
a dyadic child have scale `L/2`. A cardinal-cubic transverse support of radius
`rho_s L` permits a direct child-centre displacement of at most

```text
rho_s L/(2 sqrt(2)).
```

The child's outer entry radius is `eta L/2`, so its farthest direct reach is

```text
L[eta/2+rho_s/(2 sqrt(2))].                         (1)
```

If the parent support remains inside its complete buffer, `rho_s<=eta`, then

```text
reach/(eta L)<=(1+1/sqrt(2))/2<1.                  (2)
```

Thus a support-contained half-scale child cannot even reach the parent entry
surface, much less an admissible strip wall `Y L` with `Y>eta`. This is
independent of tuning `eta`.

At `Y=2.1`, direct capture would require

```text
rho_s>=2 sqrt(2)(Y-1)=3.111269837...>2,             (3)
```

outside the complete parent buffer. The existing `rho_s=1.91` construction
misses by

```text
2.1-1-1.91/(2sqrt(2))=0.4247130239... .             (4)
```

## The inward alternative

An inner hit at radius `L` is geometrically aligned with a concentric child:
it lies at radius `2` in units of `L/2`. Even granting the bare one-history
halving factor `1/2` to this branch, however, the honest scalar criterion is

```text
g_H^2[(k_R/2)^2+k_S^2].                              (5)
```

The wall branch remains unpaid. Across the admissible strip-width sweep, (5)
stays above one; the sampled minimum is about `1.15`. Moving the split from
the wall to the inward return therefore does not repair the architecture.

## Migrating child

Write a general transverse wall point in parent units as

```text
w=(x,+/-Y),        r_w=sqrt(x^2+Y^2).
```

The closest centre of a half-scale child whose outer entry circle contains
that point is displaced by

```text
d(w)=(1-1/r_w)w,       |d(w)|=r_w-1.                 (6)
```

The wall point is then at radius `1` in parent units, or radius `eta=2` in
child units. For child coordinate `z`, the endpoint parent-minus-child gauge
exponent before the factor `R_*/4` is

```text
|d+z/2|^2-|z|^2.
```

Its unrestricted maximum is `4|d|^2/3`, so the point-dependent shrink-paid
one-history factor is

```text
s_mig(x,Y)=exp[R_*(sqrt(x^2+Y^2)-1)^2/3]/2.          (7)
```

At `R_*=1/2`, `Y=2.1`, the axial wall point `x=0` has factor `0.6117`, but
this is not a uniform wall bound: (7) grows with `|x|`. The correct wall gain
therefore puts (7) inside the stopping resolvent,

```text
k_S^mig=E[exp(tau) 1_S s_mig(X_tau,Y)],
C_mig=g_H^2[k_R^2+(k_S^mig)^2].                      (8)
```

Although the boundary payoff is unbounded, the inward `x`-OU law makes the
sampled resolvent finite. At `Y=2.1`, (8) is `0.762477784893`; it is below one
on every sampled width in `2.02<=Y<=2.20`, with mesh, timestep, and truncation
stresses. Thus the full wall geometry, gauge arithmetic, and static scalar
branches remain mutually compatible.

The existing one-history physical-to-gauge condition-number stress is
`exp(R_*/4)`. Charging this once on the migrating wall branch gives
`0.845167851953` at `Y=2.1` and remains below one throughout the sampled
admissible window. The minimum additional one-history wall-transfer mismatch
allowed there is `1.313557337073`; this is the numerical ceiling that any
complete transfer theorem must respect.

## Smooth boundary tracking

There is a uniform alternative to the direct endpoint comparison. During a
smooth halving, move the centre so the retained wall point stays at normalized
radius `eta=2`. If `ell=L'/L<0` and `n` is its radial direction, then

```text
c'/L=-ell eta n.
```

The centre drift and scale terms in the gauged error combine exactly as

```text
q_geom=ell[1+(R_*/2)(|y|^2-eta y dot n)].             (9)
```

On the unit core, at `R_*=1/2`, `eta=2`, the bracket is

```text
3/4+[(y dot n-1)^2+|y-(y dot n)n|^2]/4>=3/4.         (10)
```

Therefore one smooth tracked halving contributes the uniform one-history
factor

```text
s_track<=2^(-3/4)=0.5946035575... .                  (11)
```

This identity is independent of the wall coordinate `x`. Both its scalar
criterion and its one-conversion stress are below one throughout the sampled
admissible widths. It is still conditional on a legitimate moving coherent
core during the interpolation: (9) controls the geometry terms, not the
nonaffine Navier-Stokes errors swept up by that motion.

## Exact physical transfer and the remaining gap

A cardinal-cubic partition at any spacing and translation is positive and
sums to one. At a fixed physical point one may therefore resample a translated
fine label by

```text
P(j|i,x)=phi_j^fine(x),                              (12)
```

independent of the old label. Summing the old weights shows that (12) has the
correct fine marginal, and the replica-pair product remains Markov. This gives
an exact nonlocal transfer in the common physical probability norm.

For a fixed wall-sign branch, a complete global partition switch also creates
no distributional jump after labels are summed:

```text
sum_j phi_j^child-sum_i phi_i^parent=1-1=0.          (13)
```

Likewise its signed Laplacian pressure commutators cancel exactly,

```text
sum_j [Delta,phi_j]p=[Delta,1]p=0.                  (14)
```

So positivity, Markov mass, the full-partition jump, and the fixed-branch
pressure algebra are not the remaining obstruction.

## Stopping-state cascade

The wall sign determines an adapted extended-state transition:

```text
L_child=L/2,
c_child=c+(1-1/|w|)wL,
P(j|wall state)=phi_j^child(x).                       (15)
```

The stopping point is then at child radius `eta=2`. Its child neutral
coordinate has absolute value at most `2`, so it lies strictly inside the
next strip. Its normalized distance to the next wall is at least

```text
Y-eta=0.1                                             (16)
```

at the working geometry. Continuity therefore excludes a literal zero-time
second wall hit. The complete one-conversion scalar pair factor is below one,
so its conditional `N`-generation terminal bound decays geometrically. Sign
branching creates no count factor because (15) is Markov.

There is no deterministic cumulative centre-travel bound: `x` is unbounded
on the wall, even though its geometry-weighted resolvent is finite. A complete
cascade theorem must propagate that wall-centre moment, not replace it by the
axial `x=0` benchmark.

It is not yet the required theorem. The wall sign and stopping time make the
fine translation branch dependent. We have not proved that the resulting
mixed-level family defines one adapted PDE localization, or that a geometry-
triggered migration earns the envelope-derived shrink payment without a
compensating branch. Migration-centre moments and finite-time generation
accumulation also remain open beyond the conditional scalar product.

The result is therefore positive but conditional: the scalar mechanism
survives a concrete wall-following geometry, while the live obstruction is a
nonlocal physical/pressure transfer theorem. The exact identities and finite-
state stresses are reproduced by
`scripts/geometry_triggered_migrating_child_pilot.py`.
