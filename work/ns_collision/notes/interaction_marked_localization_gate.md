# Interaction-marked localization gate

## Purpose

The protected collar is numerically viable, while continuous cubic
square-root localization is not. A natural alternative is to keep the
physical solution global and assign each perturbation insertion to a label
whose support lies behind a collar.

Interaction marking is algebraically exact, but it does not create an
independent local collar theorem for free. The divergence-free cancellation
and the decoupled label structure pull in opposite directions.

## Exact Dyson marking

Let

```text
P=sum_j P_j.                                          (1)
```

Whenever the baseline Neumann series converges,

```text
(A_0-P)^(-1)
 =sum_(n>=0) sum_(j_1,...,j_n)
  A_0^(-1)P_(j_1)A_0^(-1)...P_(j_n)A_0^(-1).         (2)
```

Thus every perturbation interaction can be labelled without changing the
resolvent. The audit verifies every labelled word through order five against
the unlabelled matrix power.

Equation (2) is a bookkeeping identity. Taking the norm of every labelled
word separately loses cancellations and can be much larger than the norm of
their sum. More importantly, the baseline propagators in (2) remain the
same physical propagators. Merely naming `P_j` does not move its observation
surface or create distance between the physical entry point and the other
labels.

## Loss of local skew cancellation

Let `K^*=-K` be the global divergence-free drift form, and let positive
multipliers `Phi_j` satisfy

```text
sum_j Phi_j=I.                                        (3)
```

The exact one-sided split is

```text
K=sum_j Phi_j K.                                      (4)
```

Each piece has real part

```text
Sym(Phi_j K)
 =1/2(Phi_j K-K Phi_j).                               (5)
```

The commutators in (5) sum to zero, but they need not be small separately.
The two-coordinate example

```text
K=M[[0,1],[-1,0]],
Phi_1=diag(1,0), Phi_2=diag(0,1)                      (6)
```

has zero global real part while each labelled piece has maximum real
eigenvalue `M/2`. In the PDE,

```text
[Phi_j,e dot grad]=-(e dot grad Phi_j),               (7)
```

so (5) is exactly the partition-gradient term that labelwise energy bounds
would have to pay. Avoiding square-root IMS notation does not remove it.

## Direct-sum dichotomy

For a quadratic partition write `D_j=Phi_j^(1/2)` and define the isometry

```text
Jv=(D_1v,...,D_mv),       J^*J=I.                    (8)
```

The complete lift

```text
mathbb K=J K J^*                                      (9)
```

is skew, and compression recovers the physical operator exactly:

```text
J^* mathbb K J=K.                                    (10)
```

But (9) contains every cross-label block

```text
D_i K D_j.                                            (11)
```

Keeping (11) preserves the physical equation but does not give independent
local collar problems. Dropping (11) leaves the block diagonal
`diag(D_j K D_j)`, whose compression is

```text
sum_j Phi_j K Phi_j != K.                            (12)
```

The same failure occurs for a positive multiplication potential: diagonal
label blocks compress to `sum_j Phi_j^2 Q`, not `Q`. Random matrix tests
verify exact full compression and order-one errors after block
decoupling.

This is the interaction-label dichotomy:

```text
coupled labels:   exact generator and skewness, no independent collars;
decoupled labels: independent collars, changed generator and commutators.
                                                               (13)
```

## Consequence for the current programme

Naive interaction marking cannot be used to claim that the physical error
is supported behind the active collar. A surviving construction must prove
all of the following together:

1. retain the complete label sum before taking the real energy part;
2. reconstruct physical radial observations contractively;
3. control every cross-label collar transfer instead of discarding it;
4. make label changes in the physical Markov norm or at stopping times that
   carry an explicit renewal payment.

At present this is no simpler than the original boundary theorem. The more
credible next routes are therefore:

1. a genuinely stopped domain decomposition in which every relabel is paid
   and each completed segment earns killed contraction;
2. an averaged dynamic entry-law theorem that propagates the global critical
   form without pointwise collar support;
3. a stronger local Morrey/Kato estimate derived from Navier-Stokes
   coherence rather than from bare critical mass.

The exact split, commutator counterexample, direct-sum lift, and labelled
Dyson stress test are reproduced by
`scripts/interaction_marked_localization_audit.py`.
