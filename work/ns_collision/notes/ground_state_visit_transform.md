# Ground-state visit transform

## Purpose

The Gaussian boundary calculation diagonalizes the complete finite-cylinder
visit operator. This note uses its positive principal mode to separate the
visit into one scalar growth factor and an honest Markov mixing kernel.

This is the boundary-operator analogue of the local gauge used in the
collision argument, but it is exact for the ideal finite cylinder. It shows
that higher axial modes should not be charged as independent multiplicative
losses.

## Doob transform

Let `B` be the positive outer-to-inner boundary visit operator in
`L^2(mu_h)`, where

```text
dmu_h proportional exp(-R_*y^2)dy.
```

Its axial spectral decomposition is

```text
B phi_n=U_n phi_n,
U_0>U_1>U_2>...>0.                                    (1)
```

The principal eigenfunction `phi_0` is strictly positive in the open axial
interval. Define

```text
P f=(U_0 phi_0)^(-1) B(phi_0 f).                       (2)
```

Then positivity of `B` and (1) imply

```text
P1=1.                                                  (3)
```

Because `B` is self-adjoint in `mu_h`, `P` is reversible in the ground-state
probability measure

```text
dmu_0=phi_0^2 dmu_h.                                   (4)
```

Equivalently, in the self-adjoint oscillator variable,
`dmu_0=psi_0^2dy`. The exact factorization is

```text
B=U_0 M_(phi_0) P M_(phi_0)^(-1).                     (5)
```

Thus a complete ideal visit consists of one scalar factor `U_0` and one
reversible Markov step. For independent histories,

```text
B tensor B
 =U_0^2 times the independent pair Doob kernel.        (6)
```

The pair Markov part is conservative in every `L^p` space associated with
its evolving law.

## Mixing spectrum

The spectrum of `P` is

```text
1, U_1/U_0, U_2/U_0, ... .                             (7)
```

Consequently its mean-zero `L^2(mu_0)` contraction is `U_1/U_0`, and its
chi-square contraction factor is `(U_1/U_0)^2`.

| `R_*` | half-height `h` | `U_0` | `U_1/U_0` | chi-square factor |
|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 0.856819 | 0.256839 | 0.0659665 |
| 0.5 | 1.75 | 1.04334 | 0.293330 | 0.0860426 |
| 0.5 | 2.0 | 1.19705 | 0.326344 | 0.106501 |
| 1.0 | 1.0 | 0.631962 | 0.125540 | 0.0157604 |
| 1.0 | 1.2 | 1.00311 | 0.139501 | 0.0194606 |

The working `R_*=0.5`, `h=1.5` visit removes roughly `93.4%` of chi-square
deviation from its ground-state law in one complete ideal visit. Compact
`R_*=1` cores mix even more strongly.

## Kernel positivity and minorization

The audit constructs the discrete boundary operator from the complete
retained axial family and applies (2). For every working geometry it checks:

1. strict positivity of the visit and transformed kernels;
2. row sums equal to one;
3. invariance of `mu_0`;
4. detailed balance;
5. the spectral mixing ratio (7).

At `R_*=0.5`, `h=1.5`, refinement gives:

| grid/modes | minimum `dP/dmu_0` | maximum `dP/dmu_0` | `U_1/U_0` |
|---:|---:|---:|---:|
| 201/41 | 0.413731 | 3.78120 | 0.256855 |
| 401/61 | 0.413700 | 3.78187 | 0.256839 |
| 801/81 | 0.413692 | 3.78204 | 0.256836 |

The numerical kernel therefore has a Doeblin minorization

```text
P(y,dz) >= 0.4136 mu_0(dz)                             (8)
```

on the resolved cylinder, giving the total-variation contraction bound
`1-0.4136<0.5864`. Equation (8) is numerical evidence, not yet a certified
continuum lower bound. Positivity, Markov normalization, reversibility, and
the spectral statement follow analytically from (1)-(5).

## What this fixes

The pointwise boundary payoff combines all modes and is larger than `U_0`.
That led the full-mode audit to describe the principal mode as an optimistic
surrogate. The correct distinction is now:

```text
principal mode: incomplete for a pointwise constant-boundary value,
principal multiplier: exact for the Gaussian L2 operator norm and
                      exact scalar growth in the Doob factorization.
```

No axial mode has disappeared. The nonprincipal modes form the mixing
spectrum of `P` and contract relative fluctuations.

## Perturbative significance

For a non-affine potential `q`, the perturbed boundary operator `B_q` has a
new principal multiplier and ground state. A useful stability theorem would
control

```text
U_0(q)/U_0(0),
phi_0(q)/phi_0(0),
and the spectral/minorization gap of P_q                 (9)
```

from the critical interior form bound. This is more structured than bounding
the full pointwise Feynman-Kac payoff: only one scalar can spend the renewal
margin, while the normalized remainder is Markovian.

The ground state vanishes at the absorbing axial caps, so a raw global
supremum condition number is infinite. The transform must remain local to a
complete killed visit, exactly as required by the earlier interface-weight
no-go. Carrying `phi_0` as a weight through every closed child-interface
crossing would repeat the same forbidden construction.

## Remaining cell-transfer gate

In the ideal fixed cylinder, the reversible Gaussian and ground-state laws
are canonical. Actual Navier-Stokes cells move, rotate, deform, and split.
The required architecture is therefore:

```text
physical evolving law during inter-cell Markov transfer
 -> one local ground-state transform for a complete visit
 -> scalar U_0 times Markov mixing
 -> one return to the physical evolving law.            (10)
```

The open estimates are:

1. perturbative stability (9) under the non-affine critical form error;
2. one entry/exit comparison per complete visit between neighboring local
   ground-state laws, with no charge at balance-only crossings;
3. compatibility of those laws under dyadic rescaling and eigenframe change.

The transform, reversible kernel, mixing spectrum, and grid-refined kernel
density bounds are reproduced by
`scripts/ground_state_visit_transform_audit.py`.
