# Monotone balanced dyadic cover

## Purpose

The intrinsic Lipschitz radius supplies a safe cell size at every point, but
an arbitrary maximal-ball cover can change its selected centres when the
radius evolves. Repeated reselection could create uncontrolled centre motion
or temporal chattering.

A fixed dyadic grid replaces moving centres by monotone refinement.

## Stopping rule

Let `rho_kappa(x,t)` be the nonincreasing intrinsic radius. A dyadic cube `Q`
is safe when

```text
side(Q)<=inf_(x in Q) rho_kappa(x,t).                    (1)
```

Start with coarse cubes covering the domain. Whenever (1) fails, replace `Q`
by its dyadic children. Never merge children back into their parent. Because
`rho_kappa` is nonincreasing in time, a safe cube can later become unsafe but
an unsafe parent can never become necessary again. The cover therefore has
only births by subdivision, with no merge/split chatter.

For every active cube, (1) and `rho_kappa<=ell` imply

```text
a(x,t) side(Q)^2/nu<=R_*                                (2)
```

throughout `Q`, up to the fixed dimensional conversion between side and the
chosen ball or cylinder radius.

## Balancing

Refine a coarse neighbor whenever two adjacent active cubes differ by more
than one dyadic level. This standard 2:1 balancing also uses only splits. It
gives

```text
1/2<=side(Q)/side(Q')<=2
```

for adjacent cubes. With the cell reference amplitude

```text
A_Q=R_* nu/side(Q)^2,
```

neighboring reference amplitudes differ by at most a factor four. Fixed cube
centres have zero continuous centre velocity, so one entire class of the
moving-tube errors disappears between refinement events.

## Dynamic audit

The audit evolves a strongly nonuniform one-dimensional envelope through five
snapshots, increasing one peak from zero to 640. At each snapshot it computes
the Lipschitz intrinsic radius, splits unsafe cells, and enforces 2:1 balance.
It verifies that:

1. active cells cover the domain exactly;
2. every side is below the safe radius on its complete cell;
3. every cell obeys the local Reynolds cap;
4. neighboring sides have ratio at most two;
5. cell count and maximum level never decrease;
6. later amplitude growth causes refinement but no merge.

The calculation is a one-dimensional combinatorial audit. The same dyadic
stopping and balancing rules apply in three dimensions, with a
dimension-dependent packing constant and `2^3` children per split.

## Parent-child transition gate

Refinement replaces one reference amplitude by children with

```text
A_child=4 A_parent.
```

The scalar envelope lemma regards increasing reference amplitude and
shrinking scale as favorable. That does not automatically make the
bookkeeping free: a localized deformation norm or stochastic visit operator
must transfer from the parent partition weight to the child weights.

The correct transition should preserve

```text
sum_children phi_child=phi_parent                         (3)
```

throughout a short interpolation. Equation (3) preserves pressure-flux
conservation and avoids multiplying mass merely because a cell split. The
gauges at parent and child scale are uniformly comparable because `R_*` is
fixed, but paying a fixed comparison constant at every generation would
still diverge as `side(Q)->0`.

The remaining gate is therefore a telescoping or contractive parent-child
transfer estimate, not centre selection. It must show that either:

1. the partitioned observable is exactly conserved through a split; or
2. the killed spectral decay accumulated before the split pays its transition
   cost; or
3. all split costs sum as a logarithmic variation already controlled by the
   envelope-shrink term.

## Status

The dyadic construction removes arbitrary centre trajectories and temporal
cover chatter while preserving the Reynolds and neighboring-scale bounds. It
does not yet prove summability over infinitely many generations approaching a
hypothetical singular time.

`dyadic_gauge_transition.md` computes the transition constant. For a genuine
envelope-driven radius halving, the favorable shrink term more than pays the
worst parent-child recentering cost at `R_*<=2` in three dimensions. The
remaining problem is branching transfer across balance-only child interfaces,
not a single nested lineage.

The tree evolution, safety, balance, and reference-amplitude checks are
reproduced by `scripts/monotone_dyadic_cover_audit.py`.
