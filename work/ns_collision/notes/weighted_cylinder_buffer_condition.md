# Weighted-cylinder buffer condition

## Purpose

The off-diagonal form estimate leaves one baseline quantity,

```text
chi=sqrt(||D_i|| ||D_o||)/||B_0||.
```

This note computes it for internal radial traces of the reversible
finite-cylinder model, checks radial convergence, and verifies that angular
and axial modes cannot produce a larger baseline block.

The result is useful but deliberately localized: it controls arbitrary
three-dimensional form perturbations in the interior of a positive outer
collar. Perturbation of the collar-to-Dirichlet payoff map remains a separate
estimate.

## Radial weak operator

For axial OU eigenvalue `zeta` and angular mode `m`, the reversible radial
weight is

```text
w(rho)=rho exp[(R_*/2)min(rho^2,1)].                   (1)
```

The radial mode operator is

```text
A_(m,zeta)
 =-w^(-1)(w f')'
   +[zeta+m^2/rho^2-2R_* 1_(rho<1)]f.                 (2)
```

It has the regular natural condition at the axis for `m=0`, vanishing axis
trace for `m!=0`, and a Dirichlet condition at `rho=eta=2`.

For internal surfaces `rho_i=1` and `rho_o<2`, let

```text
D_i=G(rho_i,rho_i),
D_o=G(rho_o,rho_o),
B=G(rho_i,rho_o).                                      (3)
```

The audit assembles the exact weighted weak form with two-point Gaussian
quadrature and piecewise-linear radial elements. Surface evaluation is a
bounded functional because both traces are strictly internal.

## Working-geometry values

At `R_*=0.5`, axial half-height `h=1.5`, the 800-element results are:

| `rho_o` | outer collar | `chi` | allowable `alpha` |
|---:|---:|---:|---:|
| 1.10 | 0.90 | 1.08183350 | 0.366011 |
| 1.25 | 0.75 | 1.23463893 | 0.335929 |
| 1.50 | 0.50 | 1.62634755 | 0.277470 |
| 1.75 | 0.25 | 2.48383981 | 0.200926 |
| 1.90 | 0.10 | 4.11725624 | 0.131713 |

The `rho_o=1.5` choice leaves a half-radius outer collar and a rigorous
abstract relative-form budget of approximately `0.27747`. A thinner
quarter-radius collar still leaves `0.20093`.

## Radial convergence

For `rho_o=1.5`, the condition numbers are

```text
200 elements: 1.6263484004,
400 elements: 1.6263477198,
800 elements: 1.6263475497.
```

All five trace positions change by less than `10^(-6)` between the final two
grids. Every sampled Green cross value is positive.

The condition number grows as the outer trace approaches `rho=2`. This is
not merely discretization loss. A Dirichlet boundary source has divergent
self-energy in the limiting volume-resolvent realization: both `D_o` and
`B` vanish, but their ratio in (5) becomes singular. The form perturbation
must therefore be separated from rough payoff data by a positive collar, or
the boundary space must be strengthened from `L^2` to its natural fractional
Sobolev scale.

## Full angular and axial modes

Higher axial modes add killing. Angular mode `m` adds the nonnegative form
potential `m^2/rho^2`. Positivity and form ordering therefore force all three
baseline block norms in (3) into `m=0,n=0`.

The audit verifies this directly through `|m|=6` and the first eight axial
modes. At `rho_o=1.5`, the cross norms begin

| `|m|` | maximum cross norm |
|---:|---:|
| 0 | 0.19737335 |
| 1 | 0.10472757 |
| 2 | 0.05567197 |
| 3 | 0.03016794 |
| 4 | 0.01667809 |
| 5 | 0.00939599 |
| 6 | 0.00538507 |

Each angular block is itself maximized by the principal axial mode. Hence
the axisymmetric number `chi=1.62634755` is the baseline full-cylinder
condition number, not an axisymmetric-only guess.

## What is now conditional but genuinely three-dimensional

Let `Q` be any nonnegative interior multiplication/form perturbation,
including one that couples angular and axial modes, and suppose

```text
0<=Q<=alpha A_0.                                       (4)
```

The abstract off-diagonal theorem uses only the full baseline norms in (3).
Since those norms are controlled by `m=0,n=0`, the tabulated `chi` applies to
arbitrary mode coupling. At `rho_o=1.5`, condition (4) with
`alpha<0.27747` preserves the current Gaussian renewal bound for the internal
transfer.

This does not yet control a potential extending through the collar
`1.5<rho<2`, nor does it by itself identify the internal Green transfer with
the complete outer-boundary payoff operator. The remaining domain-
decomposition problem is to factor the collar Dirichlet-to-interface map and
bound its non-affine error without reintroducing pointwise endpoint control.

The weighted finite-element assembly, radial refinement, full mode stress
test, and renewal budgets are reproduced by
`scripts/weighted_cylinder_buffer_condition_audit.py`.
