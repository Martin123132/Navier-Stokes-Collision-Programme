# Poisson cutoff form transfer

## Purpose

The internal Green-block theorem controls arbitrary three-dimensional mode
coupling, but it does not directly accept rough Dirichlet payoff data. This
note derives a form-to-boundary estimate for the complete outer Poisson visit
when the adverse perturbation is separated from the payoff boundary by a
positive collar.

The collar supplies exactly the missing smoothing. The correct conversion
constant is a cutoff energy of the baseline Poisson extension, not the
divergent self-energy of a boundary source.

## Same-boundary perturbation identity

Let `P_0 f=u_0` solve the baseline problem and let `u_q` solve the perturbed
problem with the same outer boundary data:

```text
A_0 u_0=0,
(A_0-Q)u_q=0,
u_q|boundary=u_0|boundary=f.                           (1)
```

Then `w=u_q-u_0` has zero boundary data and

```text
w=(A_0-Q)^(-1)Q u_0.                                  (2)
```

Assume the nonnegative perturbation has relative form bound

```text
0<=Q<=alpha A_0,       alpha<1,                        (3)
```

on the zero-boundary energy space. Let `T_i` be the inner trace and

```text
D_i=T_i A_0^(-1) T_i^*.                               (4)
```

Positive resolvent Cauchy-Schwarz gives

```text
||T_i w||
 <= sqrt[alpha ||D_i|| Q[u_0]]/(1-alpha).              (5)
```

## Cutoff energy

Suppose `Q` is supported in `rho<=rho_s`. Choose a radial cutoff `zeta` that
equals one there, decreases across an unperturbed collar, and vanishes before
or at the outer payoff boundary. Then

```text
Q[u_0]=Q[zeta u_0]
 <= alpha a_0[zeta u_0,zeta u_0].                      (6)
```

Define the baseline cutoff-Poisson energy

```text
E_zeta
 =sup_(||f||_boundary=1)
    a_0[zeta P_0f,zeta P_0f].                          (7)
```

Combining (5)-(7) yields the complete Poisson transfer estimate

```text
||B_q-B_0||
 <= alpha/(1-alpha) sqrt(||D_i|| E_zeta).              (8)
```

Thus

```text
chi_P=sqrt(||D_i|| E_zeta)/||B_0||                    (9)
```

plays the same role as the internal Green-block condition number, but now
`B_0` is the actual outer-boundary-to-inner visit operator.

## Working-geometry calibration

At `R_*=0.5`, half-height `h=1.5`, `eta=2`, the baseline values are

```text
||B_0||=0.85681683,
||D_i||=0.51630576,
C_0=0.37890445.
```

The radial finite-element visit gain differs from the exact Kummer/Bessel
value by less than `2e-6`. The calculated cutoff constants and renewal
budgets are:

| support `rho_s` | taper end | `chi_P` | allowable `alpha` | conservative `Q/nu` |
|---:|---:|---:|---:|---:|
| 1.00 | 1.50 | 1.27774 | 0.328318 | 1.51287 |
| 1.25 | 1.50 | 1.90110 | 0.247285 | 1.13948 |
| 1.25 | 1.75 | 1.43812 | 0.302790 | 1.39524 |
| 1.50 | 1.75 | 2.15836 | 0.224425 | 1.03414 |
| 1.50 | 1.90 | 1.78532 | 0.259166 | 1.19423 |
| 1.50 | 2.00 | 1.64890 | 0.274717 | 1.26589 |
| 1.60 | 2.00 | 1.89177 | 0.248202 | 1.14375 |
| 1.70 | 2.00 | 2.24442 | 0.217693 | 1.00314 |
| 1.75 | 2.00 | 2.49338 | 0.200311 | 0.92310 |
| 1.80 | 2.00 | 2.82798 | 0.180898 | 0.83358 |
| 1.90 | 2.00 | 4.11993 | 0.131639 | 0.60659 |
| 1.91 | 2.00 | 4.35602 | 0.125399 | 0.57783 |
| 1.92 | 2.00 | 4.63440 | 0.118761 | 0.54724 |
| 1.93 | 2.00 | 4.96961 | 0.111644 | 0.51445 |
| 1.95 | 2.00 | 5.91657 | 0.095482 | 0.43998 |

Widening the taper reduces its gradient energy. Allowing the cutoff to decay
across the entire strictly unperturbed collar is substantially better than a
sharp quarter-radius transition.

The `Q/nu` column uses the earlier sharp Sobolev form estimate with the
conservative transverse spectral margin. At `R_*=0.5`, relative form bound
one corresponds to

```text
Q/nu < 4.60795495.
```

Multiplying by the allowable `alpha` gives the displayed scale-invariant
budgets. These are sufficient bounds for arbitrary positive interior
potentials, not constant-potential calibrations.

The wider support rows are included for the radial partition optimization.
They lose form-transfer margin, but permit a wider compact partition with a
smaller IMS gradient cost. The combined optimum is not the row with the
largest standalone Poisson budget; see `radial_cubic_partition.md`.

## Full-mode maximum

The cutoff energy was searched through angular modes `|m|<=16` and the first
61 axial modes. Every profile is maximized by `m=0,n=0`. This agrees with the
form ordering: angular oscillation adds `m^2/rho^2`, axial oscillation adds
killing, and the unperturbed collar damps rough boundary data before it
reaches the perturbation support.

For the representative profiles `(rho_s,rho_t)=(1.25,1.75)` and `(1.5,2)`,
the condition numbers converge as follows:

| radial elements | `chi_P(1.25,1.75)` | `chi_P(1.5,2)` | `chi_P(1.91,2)` |
|---:|---:|---:|---:|
| 200 | 1.4381061 | 1.6488852 | 4.3559852 |
| 400 | 1.4381159 | 1.6488973 | 4.3560183 |
| 800 | 1.4381183 | 1.6489003 | 4.3560266 |

The small upward change is stable and well below the precision relevant to
the renewal margins.

## What this establishes

Conditionally on (3), the full Gaussian `L^2` Poisson visit remains
contractive under an arbitrary non-axisymmetric, mode-coupling perturbation
supported within the collar. In particular, support through `rho<=1.5` and

```text
alpha<0.2747
```

is sufficient at the working geometry. This is the desired
three-dimensional form-to-boundary theorem for a localized error.

## Remaining Navier-Stokes localization gate

The actual non-affine strain/frame error is not automatically supported in a
chosen radial core. A proof must use the intrinsic dyadic cover and a
quadratic partition so that every dangerous point lies in the protected
interior of some cell while collars are assigned to neighboring cells.

The resulting tasks are:

1. preserve the relative form budget under the partition;
2. control IMS/cutoff gradient terms and the pressure edge commutators;
3. ensure balance-only refinements do not pay the Poisson conversion
   repeatedly;
4. derive the required local `Q/nu<1.2659` bound from actual Navier-Stokes
   coherence rather than assuming it.

The variational bound, full-mode cutoff energies, exact-transfer comparison,
mass conversion, and radial convergence are reproduced by
`scripts/poisson_cutoff_form_transfer_audit.py`.
