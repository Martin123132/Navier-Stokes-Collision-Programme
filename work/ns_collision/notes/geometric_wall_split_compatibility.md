# Geometric wall versus true split

## Purpose

The axial-patch strip pilot closes its scalar branch criterion only after a
wall hit receives the cubic true-split factor. This note checks whether the
current scale construction permits that assignment. It does not.

## Two different stopping rules

The audited cubic factor is earned at a global amplitude event:

```text
running adverse envelope reaches A_n=R_* nu/L_n^2,
L_(n+1)=L_n/2.                                        (1)
```

The strip event is geometric and path dependent:

```text
|Y_tau|=2.1L.                                         (2)
```

Neither condition contains the other.

## Exact counterexample

Take the zero solution `u=0` and a fixed frame. The global adverse envelope
is zero and never reaches a positive threshold in (1). The stochastic path
is Brownian. Start its transverse component at `(x,y)=(0,2)` and consider the
slab

```text
1.5<y<2.1.                                            (3)
```

While (3) holds, the path cannot hit the unit disk because `y>1`. The exact
one-dimensional gambler's-ruin probability of reaching `2.1` before `1.5`
is

```text
(2-1.5)/(2.1-1.5)=5/6.                               (4)
```

Thus wall exit occurs with positive probability while no envelope split
occurs. This disproves

```text
wall exit => true cubic level change.                 (5)
```

## Consequence

The pointwise cubic refinement mask can still relabel a path conservatively
at a wall. But the existing theorem explicitly gives arbitrary extra
refinements no radius-halving payment. In particular, the factor

```text
s_cubic=0.639292608019                                (6)
```

cannot be inserted merely because (2) occurred.

There is also a direct geometric mismatch. The largest audited transverse
parent-to-child center displacement is

```text
delta_max/L=1.91/(2sqrt(2))=0.6753... .               (7)
```

After halving, the child's `r=2` outer entry radius is `L` in parent units.
At the working wall the remaining gap is

```text
2.1-1.91/(2sqrt(2))-1=0.4247...>0.                   (8)
```

Thus the wall point is not in any directly audited child visit. In fact, the
requirements

```text
Y>2                         (contain the full entry circle),
Y<=1+1.91/(2sqrt(2))<1.676  (direct child capture)    (9)
```

are incompatible for every strip width. A farther recentering would require
a new nonlocal transfer and gauge estimate.

This invalidates promotion of the favorable `approximately 0.65` axial-patch
criterion under the current architecture. Without (6), the companion pilot's
same-level wall criterion exceeds one near wall-dominated entries.

There are two honest continuations:

1. close the wall branch as a same-scale branch, without split payment;
2. redesign the scale hierarchy so geometric exit triggers a transition,
   then rederive its gauge, Markov, pressure, and many-generation factors
   rather than borrowing (6).

The counterexample does not rule out the second architecture. It rules out
identifying the two events for free. The exact event logic and probability
are reproduced by
`scripts/geometric_wall_split_compatibility_audit.py`.
