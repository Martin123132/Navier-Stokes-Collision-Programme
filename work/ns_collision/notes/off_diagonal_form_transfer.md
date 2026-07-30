# Off-diagonal form transfer

## Purpose

The separable axial model turns a form perturbation into a scalar eigenvalue
shift. A genuinely three-dimensional potential couples modes, so the visit
must instead be treated as an off-diagonal block of a resolvent or Poisson
operator.

This note identifies the correct abstract inequality. A tempting purely
relative estimate is false: form order controls the complete positive
resolvent, but a perturbation can create a cross-boundary coupling that was
nearly absent in the baseline. The missing geometric quantity is a
diagonal-to-cross buffer condition number.

## Resolvent order

Let `A_0` be positive self-adjoint and let a nonnegative perturbation satisfy

```text
0 <= Q <= alpha A_0,       0<=alpha<1.                 (1)
```

Write

```text
R_0=A_0^(-1),
R_q=(A_0-Q)^(-1).
```

Conjugating by `A_0^(1/2)` gives

```text
0 <= R_q-R_0 <= alpha/(1-alpha) R_0.                  (2)
```

Let `F_i` and `F_o` be inner and outer trace-source maps and define

```text
B_0=F_i^* R_0 F_o,       B_q=F_i^* R_q F_o,
D_i=F_i^* R_0 F_i,       D_o=F_o^* R_0 F_o.           (3)
```

Applying Cauchy-Schwarz in the positive form `R_q-R_0`, then (2), gives

```text
||B_q-B_0||
 <= alpha/(1-alpha) sqrt(||D_i|| ||D_o||).             (4)
```

Put

```text
chi=sqrt(||D_i|| ||D_o||)/||B_0||.                    (5)
```

Positivity of the baseline block matrix gives `chi>=1`, and (4) becomes

```text
||B_q||/||B_0|| <= 1+chi alpha/(1-alpha).              (6)
```

Unlike a generic norm estimate, (4) retains the exact inner/outer Green
geometry. The task is reduced to one baseline condition number.

## Why the naive estimate fails

Take

```text
A_0=I_2,
Q=alpha u tensor u,       u=(1,1)/sqrt(2),
F_i=e_1,
F_o=epsilon e_1+sqrt(1-epsilon^2)e_2.
```

With `alpha=1/4` and `epsilon=10^(-3)`, the baseline cross transfer is
`0.001`, while the perturbed transfer is approximately `0.167833`.
Therefore

```text
actual relative amplification:             167.833,
naive bound 1/(1-alpha):                    1.333.
```

The perturbation has created a new coupling through the direction `u`. The
correct diagonal-envelope upper bound is `0.334333` and remains valid.

This is not a numerical edge case. Sending `epsilon` to zero makes the
relative amplification unbounded at fixed `alpha`. Eight independent random
positive-matrix trials also verify (2) and (4).

## Renewal budget

Let `C_0<1` be the baseline Gaussian `L^2` complete-generation criterion. If
(6) is the only perturbation loss, closure follows when

```text
[1+chi alpha/(1-alpha)]^2 C_0 < 1,
```

or equivalently

```text
alpha
 < [C_0^(-1/2)-1]/[chi+C_0^(-1/2)-1].                 (7)
```

For the preferred `R_*=0.5`, half-height `h=1.5` geometry,
`C_0=0.3789044474`. Equation (7) gives:

| `chi` | allowable `alpha` |
|---:|---:|
| 1 | 0.384448 |
| 1.5 | 0.293971 |
| 2 | 0.237967 |
| 3 | 0.172313 |
| 5 | 0.111041 |
| 10 | 0.0587844 |

These values are more conservative than the exact separable axial threshold
`0.8692`. That difference is expected: (4) permits arbitrary mode-creating
perturbations and uses only positive form order.

## Consequence for the cylinder

The full three-dimensional extension cannot be completed by writing
`B_q<=B_0/(1-alpha)`. It requires:

1. internal trace maps for which `D_i`, `D_o`, and `B_0` are bounded;
2. a certified finite-cylinder value of `chi`;
3. a positive collar separating rough payoff boundary data from the support
   of the form perturbation, or an equivalent boundary Sobolev norm.

The abstract theorem, counterexample, random stress tests, and renewal table
are reproduced by `scripts/off_diagonal_form_transfer_audit.py`.
