# Dyadic gauge transition and shrink payment

## Purpose

A monotone dyadic cover avoids moving-centre chatter, but a path can pass from
a parent cell to a recentered child gauge at every generation. Bounding each
gauge comparison separately produces a constant greater than one per split
and is therefore unusable near infinitely fine scales.

For a genuine nested core whose physical radius halves, the monotone-envelope
shrink term supplies a larger contraction. This note computes both constants
exactly and states the limitation of that payment.

## Parent-child gauge cost

For a cube `Q` of side `h`, use the normalized radial gauge

```text
g_Q(x)=exp[-R_* |x-c_Q|^2/(4h^2)].
```

Let a child have side `h/2` and centre offset `sigma h/4` in each coordinate,
where `sigma` is a sign. Write a point in the child as

```text
x=c_child+(h/2)y,       -1/2<=y_j<=1/2.
```

In one coordinate, the parent exponent minus the child exponent before the
factor `R_*/4` is

```text
(sigma/4+y/2)^2-y^2.
```

Its maximum is `1/12`, attained at `y=sigma/6`. Consequently, in dimension
`d`,

```text
sup_(x in child) g_child(x)/g_parent(x)
 =exp(R_* d/48).                                         (1)
```

At `R_*=2`, `d=3`, this is `exp(1/8)`, approximately `1.13315`. Paying (1)
at every generation grows exponentially, so uniform equivalence of adjacent
gauges is not enough.

## Genuine radius-halving payment

Along the monotone-envelope tube,

```text
ell=L'/L<=0.
```

During envelope growth, the scale part of the ideal effective error obeys

```text
q_scale=ell(1+R_*|y|^2/2)<=ell.
```

If a genuine nested core shrinks from radius `L` to `L/2`,

```text
integral ell dt=log(1/2)=-log 2.
```

The resulting one-history contraction is at least `1/2`. Combining it with
the worst recentering cost (1) gives

```text
gamma_1=exp(R_* d/48)/2.                                 (2)
```

For two independent histories the factor is

```text
gamma_2=exp(R_* d/24)/4=gamma_1^2.                       (3)
```

At `R_*=2`, `d=3`,

```text
gamma_1 approximately 0.56657,
gamma_2 approximately 0.32093.
```

Thus true nested radius-halving transitions are strictly contractive in the
ideal gauge model, even before adding the killed spectral margin. More
generally, (2) is below one whenever

```text
R_* d<48 log 2.
```

The programme operates far inside this range.

## Why this is conditional

The shrink payment applies to a history retained in one physical core while
that core radius actually halves. A balanced dyadic partition also creates
children for combinatorial reasons. The union of those children equals the
parent, so partition refinement alone has not removed physical histories or
earned a free `1/2` contraction.

Paths crossing child interfaces create transfer and renewal terms. Claiming
the payment (2) for every balance-only split would amount to obtaining
arbitrary damping by refining a partition, which is false. Those refinements
must instead be handled by an exact flux identity, inherited parent weights,
or a charge to a nearby genuine safety split.

## Refined transition gate

The scalar nested-core lineage is no longer the problem: its gauge transition
is paid with a strict margin. The unresolved obligation is a branching one:

1. preserve `sum_children phi_child=phi_parent` so pressure flux remains
   conservative;
2. route histories that cross child interfaces through a transfer operator
   rather than treating each as a fresh uncontrolled return;
3. charge balance-only refinements to a bounded number of nearby true splits;
4. retain the contraction (2) after summing over all child branches.

The exact exponent optimization and generation stress test are reproduced by
`scripts/dyadic_gauge_transition_audit.py`.

The branching problem is continued in `branching_transfer_operator.md`.
Child count does not amplify a positive replica observable in the physical
probability norm; the surviving issue is compatibility between that
conservative norm, local gauges, and signed pressure edge weights.
