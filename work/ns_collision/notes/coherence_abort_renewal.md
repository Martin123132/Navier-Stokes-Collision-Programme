# Coherence-abort renewal budget

## Purpose

The stopping-time moving visit removes the continuous Fisher cost, but it
creates a precise renewal question. What happens if the fitted affine or
sector budget fails before the path reaches the geometric exit boundary?

Conservative relabeling is norm one. It preserves mass, but that fact alone
cannot pay arbitrarily many early restarts. This note derives the exact
operator slack available for such aborts.

## Restart resolvent

Let `G` be the operator for a successfully completed generation and let `R`
be an abort followed by re-entry into the same entry space. Summing over any
number of aborts gives

```text
T=G+GR+GR^2+...
 =G(I-R)^(-1).                                        (1)
```

If

```text
||G||<=g,       ||R||<=r<1,                           (2)
```

then

```text
||T||<=g/(1-r).                                       (3)
```

Strict renewal contraction follows from the transparent condition

```text
g+r<1.                                                (4)
```

The current unperturbed generation criterion is

```text
C_0=0.256635660491,
g_0=sqrt(C_0)=0.506592203346.                         (5)
```

Therefore the total restart allowance before sector perturbations is

```text
r<1-g_0=0.493407796654.                               (6)
```

A conservative relabel has norm one, so free same-scale coherence restarts
do not satisfy (6). This is a no-free-restart statement, not a failure of the
stopping-time construction.

## Abort paid by a true split

The existing cubic level audit gives the replica-pair true-halving factor

```text
r_split=0.408695038668.                               (7)
```

Thus an abort that genuinely triggers an envelope-driven dyadic halving does
fit inside (6). The restart may carry at most the additional multiplicative
mismatch

```text
(1-g_0)/r_split=1.20727620835.                        (8)
```

before the unperturbed Neumann closure is lost.

Sector errors consume most of this slack. With

```text
g(alpha,beta)
 =g_0[1+chi(alpha+beta)/(1-alpha)],
chi=4.76786626401,                                    (9)
```

the exact restart condition is

```text
g(alpha,beta)+r<1.                                    (10)
```

For `r=r_split`, (10) becomes

```text
beta<d_split-(1+d_split)alpha,
d_split=0.03507246210.                                (11)
```

Equal sector shares then permit only

```text
alpha=beta<0.01723401144,
||q_+||_(3/2)/nu<0.09440626175,
||e||_3/nu<0.04033607065.                             (12)
```

These are much tighter than the no-restart budgets. Unlimited split-paid
aborts are mathematically possible, but they leave little room for physical
non-affinity.

## Abort paid by probability

If an abort branch has conditional operator norm `M` and probability `p` in
the dynamic `L^2` law, its restart norm is at most

```text
r<=sqrt(p) M.                                         (13)
```

With `M=1` and no sector error, (6) requires

```text
p<(1-g_0)^2=0.243451253799.                           (14)
```

This is the quantitative occupation route: same-scale coherence failures
need not be impossible, but their conditional `L^2` branch mass must be less
than about `24%` before other errors, and smaller after the sector budget is
spent.

## Consequence

Every early visit termination must carry one of three explicit payments:

1. a genuine level halving with the true-split factor (7);
2. a probability or bad-occupation factor satisfying (13)-(14);
3. another strict operator decay that enters `r` in (4).

Pure conservative relabeling supplies none of them. This prevents a common
logical shortcut: Markov contraction guarantees no amplification at a label
change, but norm one is not enough to sum an unlimited restart series.

The next PDE task is to define coherence-failure stopping times from local
velocity/strain averages and prove that each failure falls into the true
split or small-occupation classes. The Neumann algebra and finite-dimensional
stress tests are reproduced by
`scripts/coherence_abort_renewal_audit.py`.
