# Full finite-cylinder axial mode expansion

## Purpose

The principal axial mode suggested that compact three-dimensional cores can
overcome transverse recurrence. This note evaluates the complete axial
boundary operator for the same piecewise ideal model rather than assigning
the outer radial payoff to one separated mode.

The full mode family is modestly worse than the principal surrogate, but the
compact-core route survives.

## Self-adjoint axial form

The inward axial OU generator is

```text
L_z=partial_yy-2R_* y partial_y,       -h<y<h.
```

With

```text
phi(y)=exp(R_* y^2/2) psi(y),
```

the eigenvalue problem `-L_z phi=zeta phi` becomes the Dirichlet oscillator

```text
H psi
 =[-partial_yy+R_*^2 y^2-R_*]psi
 =zeta psi.                                               (1)
```

This is self-adjoint in ordinary `L^2`. It is equivalent to self-adjointness
of `L_z` in the Gaussian weight `exp(-R_*y^2)dy`.

## Expansion of the radial-exit payoff

The outer radial boundary payoff is the constant function one on `-h<y<h`.
If `psi_n` are normalized eigenfunctions of (1), its weighted OU
coefficients are

```text
c_n=integral exp(-R_*y^2/2) psi_n(y)dy.                  (2)
```

For axial eigenvalue `zeta_n`, let `U(zeta_n)` be the exact Kummer/Bessel
radial transfer from the preceding audit. The complete inner-boundary visit
profile is

```text
u(1,y)
 =sum_n c_n U(zeta_n) exp(R_*y^2/2) psi_n(y).            (3)
```

Odd modes vanish from (2). The audit retains the full numerical family rather
than imposing parity by hand, diagonalizing the self-adjoint tridiagonal
oscillator and evaluating (3) directly.

## Verification

The calculation checks:

1. grid refinement from 201 to 801 axial points;
2. agreement of the finite-difference principal eigenvalue with the exact
   Kummer root;
3. decay of the final ten retained mode contributions;
4. reconstruction of the constant radial boundary data at the centreline;
5. location of the maximal visit gain.

For all threshold geometries, the maximum of (3) occurs at the axial centre,
as expected from symmetry and the inward drift.

## Corrected aspect ratios

At `eta=2`, `beta=1`, the maximum allowed half-heights are approximately:

| `R_*` | principal-mode `H/L` | full-mode `H/L` |
|---:|---:|---:|
| 0.5 | 2.4746 | 2.2558 |
| 1 | 1.3649 | 1.3055 |
| 2 | 0.9016 | 0.8828 |

The precise audited values retain finite-difference error and are stored in
the script output. `R_*=0.25` already closes without axial killing in the
transverse benchmark, so no finite-height threshold is required there.

The full boundary data therefore tightens but does not destroy the compact
core mechanism. At `R_*=1`, a full cylinder height of roughly `2.61L` is the
ideal-model threshold. A long tube remains excluded.

## What is now established in the model

For the piecewise ideal generator:

1. all same-scale core returns are included in one Feynman-Kac boundary
   problem;
2. the complete finite axial mode family is included;
3. inward axial trapping is included;
4. three-dimensional exterior return and true dyadic splitting are included
   through their exact operator factors.

Under those assumptions, sufficiently compact cores satisfy the complete
generation inequality.

## Remaining Navier-Stokes gate

The model still idealizes the actual PDE in decisive ways. It assumes an
axisymmetric affine core, a Brownian radial shell with no positive stretching,
fixed cylindrical geometry, and the benchmark exterior return factor. The
next estimate must perturb (3) by the genuine Navier-Stokes errors:

```text
delta_s-b_0 dot e/(2nu), pressure-driven geometry,
centre/frame motion, and weighted exterior deformation.
```

The compact geometry now supplies a numerical margin against which those
errors can be measured. The next useful calculation is to tabulate that
margin away from the threshold, preferably near `R_*=0.5`, `H/L` between
`1.5` and `2`, and `eta=2`, then compare it with the critical `L^(3/2)` form
budget already derived for non-affine coherence.

The full mode expansion, convergence checks, and corrected aspect ratios are
reproduced by `scripts/finite_cylinder_mode_audit.py`.
