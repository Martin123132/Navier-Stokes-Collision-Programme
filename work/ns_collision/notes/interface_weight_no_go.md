# Interface-weight no-go and two-norm route

## Purpose

Conservative branching removes the false `64^n` loss, but local gauges assign
different weights to neighboring cells. This note asks whether one
nonconstant positive weight can remain nonexpansive under every crossing of a
closed irreducible child-interface graph.

It cannot. The exact no-go points to a two-norm architecture: physical mass
for interface transfer, local gauged norms only during buffered coherent
visits.

## Logarithmic weighted growth

Let `K` be a forward finite-state Markov generator in column convention:
off-diagonal entries are nonnegative and every column sums to zero. For a
positive cell weight `w`, conjugate by `D_w=diag(w)`:

```text
B=D_w K D_w^(-1).
```

The logarithmic `l^1` norm is

```text
mu_1(B)
 =max_j [(K^T w)_j/w_j].                                (1)
```

Thus weighted interface transfer is infinitesimally nonexpansive only if

```text
K^T w<=0                                                 (2)
```

componentwise.

Let `pi>0` be the stationary probability of the irreducible closed graph,
so `K pi=0`. Multiplying (2) by `pi` gives

```text
sum_j pi_j (K^T w)_j=w^T K pi=0.
```

Every term is nonpositive and every `pi_j` is positive, hence `K^T w=0`.
Irreducibility makes the left kernel one-dimensional, spanned by the constant
weight. Therefore:

```text
the only positive nonexpansive weight on a closed irreducible
interface graph is a constant.                               (3)
```

For independent replicas with product weights, the pair logarithmic growth
is twice the one-history growth. The audit verifies (1)-(3) on the asymmetric
cube generator used in the branching test.

This no-go does not apply to a killed or open graph; boundary loss can support
nonconstant superharmonic weights. That is exactly why local gauges remain
useful inside killed visits.

## Two-norm architecture

Use two deliberately different norms:

1. the constant physical `l^1` mass norm during child branching, balance
   refinement, and interface crossings;
2. the local radial gauge only after a history enters a buffered coherent
   visit, returning to the physical norm when that visit ends.

The interface operator is exactly contractive in the first norm. The price is
one physical-to-gauge condition number per complete visit, rather than one
weight mismatch at every child crossing.

For the unit-ball gauge

```text
g(y)=exp(-R_*|y|^2/4),       |y|<=1,
```

one history has multiplier condition number at most `exp(R_*/4)`. A replica
pair therefore costs at most

```text
C_pair=exp(R_*/2).                                      (4)
```

Combining this conservative conversion bound with the genuine dyadic
radius-halving factor gives, in dimension `d`,

```text
Gamma_phys(R_*,d)
 <=(1/4) exp[R_* d/24+R_*/2].                           (5)
```

This is a sufficient bound and may overcount endpoint gauge comparisons; it
is not asserted to be sharp.

## Numerical budget

In three dimensions,

```text
Gamma_phys(R_*,3)=(1/4)exp(5R_*/8).
```

At the largest proposed value `R_*=2`,

```text
Gamma_phys approximately 0.87269,
-log Gamma_phys approximately 0.13629,
1/Gamma_phys approximately 1.14599.
```

Thus the conservative two-norm conversion still leaves strict contraction,
but only about fourteen percent multiplicative room for all additional
pressure, renewal, and coherence errors. The killed spectral margin has not
been included in (5), so a visit of positive duration can add room.

Smaller `R_*` improves the budget rapidly. The tradeoff remains the one found
by the intrinsic cover: smaller Reynolds cells create more interfaces and
potentially larger renewal cost.

## Revised closure gate

The desired cycle now has three phases:

```text
physical conservative transfer
 -> one buffered local gauged visit
 -> physical conservative transfer.
```

A proof must ensure that entry and exit maps incur (4) once per buffered
visit. Applying it separately at every balance refinement or child crossing
would reproduce exponential weight growth and violate (3).

Pressure remains signed rather than Markovian. Its partition flux should be
kept in the physical phase, where uniform weights cancel it exactly, or its
edge mismatch must fit within the remaining logarithmic budget in (5).

The nullspace calculation, logarithmic norms, pair doubling, and two-norm
budget are reproduced by `scripts/interface_weight_no_go_audit.py`.

`two_norm_generation_cycle.md` corrects the temporal bookkeeping: repeated
returns at one fixed scale must be renewed before the genuine radius-halving
factor is applied. The resulting model favors `R_*` near one rather than the
edge value two.
