# Axial form-to-boundary estimate

## Purpose

The Gaussian `L^2` architecture reduces the ideal visit growth to the
principal radial multiplier `U(zeta_0)`. This note proves that a critical
relative form bound propagates all the way to that boundary multiplier for a
nonconstant but separable class of adverse potentials.

The class is

```text
q_+=q_+(y),
```

acting throughout the radial cylinder. It is not yet the general
three-dimensional Navier-Stokes error. Within this class, however, the result
is exact and includes concentrated axial profiles.

## Perturbed oscillator

After the axial OU gauge transform, the unperturbed Dirichlet operator is

```text
H=-partial_yy+R_*^2 y^2-R_*,      -h<y<h.              (1)
```

An axial adverse potential gives

```text
H_q=H-q_+(y).                                           (2)
```

Because `q_+` has no radial dependence, separation survives. If
`zeta_n(q)` and `phi_n(q)` are the eigenpairs of (2), the complete boundary
visit operator satisfies

```text
B_q phi_n(q)=U(zeta_n(q)) phi_n(q).                    (3)
```

The radial gain `U` is positive and decreasing, so

```text
||B_q||_2=U(zeta_0(q)).                                (4)
```

The eigenfunction may change substantially; it enters the transformed
Markov kernel, not the scalar norm in (4).

## Relative form propagation

Assume

```text
<v,q_+v> <= alpha <v,Hv>       for every v in H_0^1,
alpha=||H^(-1/2) q_+ H^(-1/2)||<1.                    (5)
```

Then

```text
H_q >= (1-alpha)H,
zeta_0(q) >= (1-alpha)zeta_0(0).                       (6)
```

Combining monotonicity of `U` with (4)-(6) gives the explicit nonlinear
boundary estimate

```text
||B_q||_2
 <= U((1-alpha)zeta_0(0)).                             (7)
```

This is the form-to-boundary implication that failed in the pointwise
architecture. No endpoint Green evaluation occurs.

Let `zeta_req(R_*)` be the axial eigenvalue at which the complete generation
criterion equals one. Closure follows whenever

```text
(1-alpha)zeta_0(0)>zeta_req,

alpha < alpha_crit
      =1-zeta_req/zeta_0(0).                           (8)
```

## Relative form budgets

| `R_*` | half-height `h` | `zeta_0` | `zeta_req` | `alpha_crit` |
|---:|---:|---:|---:|---:|
| 0.5 | 1.5 | 0.669303 | 0.0875368 | 0.869212 |
| 0.5 | 1.75 | 0.403665 | 0.0875368 | 0.783145 |
| 0.5 | 2.0 | 0.242992 | 0.0875368 | 0.639755 |
| 1.0 | 1.0 | 1.59692 | 0.560615 | 0.648939 |
| 1.0 | 1.2 | 0.898203 | 0.560615 | 0.375848 |

The compact `R_*=0.5`, `h=1.5` geometry retains closure even under a very
large relative form perturbation. The near-threshold geometries again spend
substantial robustness before PDE errors are introduced.

## Concentration stress test

The audit diagonalizes `H-q` directly and computes the generalized
eigenvalue in (5). At the working geometry, it tests a constant, two centred
Gaussians, and twin Gaussians near the axial caps.

| profile | actual failure `alpha` | critical `Q/nu` |
|---|---:|---:|
| constant | 0.869211 | 6.52997 |
| centre Gaussian, width `0.15h` | 0.876588 | 5.32235 |
| centre Gaussian, width `0.35h` | 0.871969 | 4.69356 |
| twin cap Gaussians | 0.905902 | 24.2273 |

Here `Q` is the physical `L^(3/2)` mass of the potential extended uniformly
over the radial disk `rho<eta`. The universal sufficient value `0.869211` is
saturated by the constant potential and is close to the actual threshold for
the dangerous centre-concentrated profiles. Cap-localized error is much less
effective because the principal Dirichlet state is small there.

At half the universal budget,

```text
alpha=0.434606,
zeta_0(q)>=(1-alpha)zeta_0=0.378418,
C_q<=0.585394.                                         (9)
```

The actual generation criteria are approximately `0.58539`, `0.57480`,
`0.58147`, and `0.53103` for the four profiles, all below (9). This verifies
both the operator ordering and the nonlinear propagation through `U` and the
renewal formula.

## Relation to the full three-dimensional form bound

The earlier sharp Sobolev calculation supplies a sufficient relative form
bound for general `q_+(x)` in the volume. The present result shows exactly
what such an `alpha` buys once the perturbation remains separable:

```text
critical volume form control
 -> principal axial eigenvalue control
 -> Gaussian boundary operator control
 -> complete generation closure.                      (10)
```

For general `q_+(rho,theta,y)`, separation and the scalar formula (4) are
lost. Extending (10) requires a buffered Poisson/trace estimate or a
min-max comparison for the full boundary transfer operator. That extension
must preserve the ground-state Markov factorization closely enough that only
its principal scalar spends the renewal margin.

## Remaining Navier-Stokes gate

Two distinct tasks should not be conflated:

1. prove a full three-dimensional analogue of (7) for the weighted cylinder
   boundary operator;
2. derive an `alpha` below the tabulated threshold from actual local
   strain/eigenframe coherence using Leray-class quantities.

The first is now a concrete elliptic operator problem. The second remains the
regularity-critical Navier-Stokes problem and cannot be inserted as an
assumption.

The generalized eigenvalue, profile thresholds, critical masses, and
generation bounds are reproduced by
`scripts/axial_form_to_boundary_audit.py`.
