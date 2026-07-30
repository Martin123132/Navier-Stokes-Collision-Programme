# Modified-chain complementary floor by inertia and a constraint Schur matrix

## Purpose

The projected-transfer stage left one finite-dimensional mismatch: the
certified complement floor `107.01775717228844` belongs to the exact
reference polygon form, while the off-block coupling belongs to the modified
reversible chain. This stage obtains a floor from the modified chain itself.

Let

```text
H=M_tilde^(-1/2) A_tilde M_tilde^(-1/2),
W=M_tilde^(1/2)V,
Z=ker(W^T).
```

The required high block is the restriction of `H` to `Z`. No reference
eigenvalue is substituted for a modified-chain quantity.

## Inertia-Schur theorem

Fix `beta` and put `J=H-beta I`. Suppose:

1. `J` has exactly `k` negative eigenvalues;
2. `W` has `k` columns; and
3. `S=W^T J^(-1)W` is negative definite.

Then `H-beta I` is positive definite on `ker(W^T)`.

Indeed, the KKT matrix

```text
K=[[J,W],[W^T,0]]
```

has inertia `inertia(J)+inertia(-S)` by Schur complementation. In an
orthogonal basis adapted to `range(W)` and `ker(W^T)`, elementary block
elimination instead gives the inertia of the constrained restriction plus
`(k,k,0)`. Comparing the two identities shows that the constrained
restriction has no nonpositive direction.

The calculation is performed in generalized coordinates with

```text
J_gen=A_tilde-beta M_tilde,
S=(M_tilde V)^T J_gen^(-1)(M_tilde V).
```

This is congruent to the normalized formula above, so definiteness is
unchanged.

## Certification architecture

The target is `beta=102.7`. Directed Decimal sparse LDL at precision 220 is
run at `102.6` and `102.8`. Both rows must complete all `15211` pivots with
exactly `240` negative pivots. This proves a resolvent gap of at least `0.1`
around `beta`.

The central sparse solve forms the 240-by-240 Schur matrix. Its residual is
bounded in the `M_tilde^(-1)` norm, and the resolvent gap converts that
residual into a Schur operator error. Dense product roundoff, eigensystem
orthogonality, and reconstruction error are paid separately. The final
upper bound on the exact largest Schur eigenvalue must remain negative.

Each directed inertia row is written atomically with its full Decimal pivot
interval cache. A later run verifies and reuses a completed row, so the
two-shift calculation is resumable.

## Scope

The result certifies only the stored binary modified chain and the
`M_tilde`-orthogonal complement of the frozen 240-column reference trial
matrix. The next finite step is to combine this floor with directed low-block
and off-block bounds in the damped two-block theorem, including boundary
output smoothing. Continuum Ritz transfer and polygon-to-circle domain
perturbation remain open.

The executable is
`scripts/neutral_strip_modified_complement_inertia_schur_certificate.py`.

## Certified production row

At spacing `h=0.06`, precision 220 completes both `15211`-pivot rows:

```text
shift       negative    positive    minimum pivot margin
102.6          240        14971       1.9930368601895657e-6
102.8          240        14971       4.4158905347223125e-6
```

The resulting resolvent gap is at least `0.1`. The solve-error contribution
to the Schur bound is at most `1.1429334580132102e-7`, and the exact largest
Schur eigenvalue is at most `-0.009977535668957992`. Hence the modified
complementary floor `102.7` is certified. A checkpoint replay reuses and
hash-verifies both complete Decimal rows.
