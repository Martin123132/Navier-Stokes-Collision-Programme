# Conservative branching and replica-pair transfer

## Purpose

A three-dimensional dyadic split has eight children, so two independent
replicas have 64 possible child pairs. Counting those branches as 64 separate
worst cases would immediately destroy the true-split contraction. This note
keeps the probabilities and transfer structure exact.

The conclusion is useful and limited: branching and interface crossings are
nonexpansive in one common physical norm. Amplification enters only after
incompatible cell or gauge weights are imposed.

## Child branching

Let `p_i>=0`, `sum_i p_i=1`, be the conditional probabilities for one history
to enter the eight children. The forward branching map from parent mass to
child masses is the column

```text
C=(p_1,...,p_8)^T.
```

Its induced `l^1` norm is one. Independent replicas branch by

```text
C_pair=C tensor C,
```

whose 64 entries are `p_i p_j` and still sum to one. Equivalently, the
backward maps on bounded observables are contractions in `l^infinity`.

If a true radius-halving split contributes the previously audited factors

```text
gamma_1=exp(1/8)/2 approximately 0.566574,
gamma_2=gamma_1^2 approximately 0.321006,
```

then

```text
||gamma_1 C||_1=gamma_1,
||gamma_2 C_pair||_1=gamma_2.                            (1)
```

The 64 branches do not multiply (1) by 64. After `n` true pair splits, total
mass is at most `gamma_2^n`, independent of the number `64^n` of branch
labels.

## Interface crossings

Let `K` be any finite-state child-interface generator with nonnegative
off-diagonal rates and column sums zero. Then

```text
P(t)=exp(tK)
```

is column-stochastic and `||P(t)||_1=1`. For independent replicas,

```text
K_pair=K tensor I+I tensor K,
P_pair(t)=P(t) tensor P(t),
```

so `||P_pair(t)||_1=1` as well. The audit uses asymmetric rates on the cube
graph and verifies these claims numerically; symmetry or reversibility is not
required.

Therefore arbitrarily many child-interface crossings are harmless in the
physical probability norm. Treating every crossing as a new branch and
summing suprema would introduce a loss absent from the exact operator.

## Weighted norm and harmonic inheritance

Suppose child masses are measured with positive weights `w_i`. A parent of
weight `w_parent` has one-step transfer factor

```text
M=(sum_i p_i w_i)/w_parent.                              (2)
```

The transfer is exactly nonexpansive when the parent weight is inherited
harmonically:

```text
w_parent=sum_i p_i w_i.                                 (3)
```

For two independent replicas, product weights `w_i w_j` and parent weight
`w_parent^2` preserve the same identity. Thus compatible branching weights do
not spend any of the pair contraction `gamma_2`.

Arbitrary gauge weights do not satisfy (3). Conjugating the interface
semigroup by a nonconstant diagonal weight can also produce norm greater than
one even though the physical semigroup is stochastic. This is the precise
source of the branching-weight obstruction.

## Conditional closure criterion

Let the nonnegative factors accumulated between consecutive true pair splits
be

```text
M_balance, M_interface, M_pressure, M_renewal.
```

A sufficient per-generation criterion is

```text
gamma_2 M_balance M_interface M_pressure M_renewal<1.    (4)
```

Equivalently, the complete mismatch has logarithmic budget

```text
log(M_balance M_interface M_pressure M_renewal)
 <-log(gamma_2) approximately 1.136.
```

Balance-only refinements carry no guaranteed `gamma_2`. They must be exactly
conservative in the chosen norm, or a bounded number of them must be charged
to a nearby true split before (4) can be used.

## Connection to pressure

Pressure shell terms are signed conservative fluxes, not Markov
probabilities. Their unweighted partition sum is zero, while nonuniform cell
weights leave neighboring edge differences. The branching calculation says
that one should not take independent absolute values of all child fluxes.
Instead, a candidate norm should satisfy both:

1. harmonic parent-child inheritance such as (3) for positive replica mass;
2. controlled neighboring differences for the signed pressure edge flux.

The open construction is a partition/gauge weight with both properties. A
common physical weight gives perfect transfer conservation but must still
retain the localized spectral contraction. A raw cell gauge gives the
spectral estimate but can amplify transfer. Bridging those two norms is now
the exact gate.

The branching columns, asymmetric interface semigroups, weighted conjugation,
and 50-generation stress test are reproduced by
`scripts/branching_transfer_operator_audit.py`.

`interface_weight_no_go.md` proves that a closed irreducible interface graph
admits no nonconstant positive conservative weight. It replaces the search
for one universal weighted norm by a two-norm cycle: physical mass for
transfer and a local gauge only during complete buffered visits.
