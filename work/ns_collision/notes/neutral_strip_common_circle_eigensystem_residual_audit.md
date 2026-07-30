# Common-circle eigensystem residual audit

## Scope

The frozen common-circle calculation uses 240 retained modes from a cached
241-vector generalized eigensystem of the stored `h=0.06` reference mass and
stiffness matrices. It also diagonalizes a 240-dimensional transformed
modified stiffness matrix. This audit encloses the residual and orthogonality
arithmetic for both eigensystems relative to those exact stored binary64
matrices.

The original conclusion was deliberately narrower than an indexed eigenvalue
certificate. It proves small residuals, a stored-mass coercivity inequality,
and disjoint spectral proximity intervals. It does not prove that the
intervals contain eigenvalues numbered `1,...,241`, nor that the stored
matrices enclose the intended exact finite-element integrals.

## Directed residuals

For each cached reference pair `(lambda_k,v_k)`, let

```text
r_k=A v_k-lambda_k M v_k.                                (1)
```

Sparse products, column scalings, subtraction, Euclidean norms, and the
`M`-orthogonality Gram are enclosed with conservative IEEE binary64
`gamma_n` bounds and outward `nextafter` inflation. The matrices have at most
ten nonzeros per row and are exactly symmetric as stored.

The direct residual bound is not yet in the natural self-adjoint metric. To
obtain that metric without inverting the `15211 x 15211` mass matrix, the audit
proves a row-lumped coercivity inequality.

## Stored-mass coercivity

For each triangle, restrict its stored binary64 local mass matrix `B_T` to the
one, two, or three state vertices present on that element. Directed interval
evaluation of every leading principal minor proves

```text
B_T >= 0.15 diag(B_T 1).                                 (2)
```

The smallest outward-rounded leading-minor lower bounds for block sizes one,
two, and three are respectively

```text
1.3101229999230833e-8,
1.0156758961628055e-16,
5.919727470781511e-25.                                   (3)
```

At most nine local values contribute to a global mass entry. Paying the
positive duplicate-summation factor

```text
gamma = 5.773159728050848e-15
```

therefore gives, for the exact stored global matrix and its exact row-sum
diagonal `D`,

```text
M >= (0.15-gamma)/(1+gamma) D
  >= 0.14999999999999333 D.                              (4)
```

This is a certificate about the stored mass matrix. It is independent of a
sparse numerical inverse, but it does not certify quadrature consistency with
the exact weighted mass form.

## Inverse-mass residual and spectral proximity

Equation (4) implies

```text
||M^(-1/2) r_k||^2
 <= (1/alpha) sum_i |r_ki|^2/D_i,                        (5)
```

where every residual component and row-sum denominator is outward enclosed.
For an approximately `M`-normalized vector, the self-adjoint residual theorem
then gives an eigenvalue of the stored pencil `(A,M)` within

```text
eta_k/||v_k||_M
```

of `lambda_k`.

The production values are

```text
maximum direct residual L2 upper                7.595999996836146e-13
maximum inverse-mass residual upper             7.541469516727855e-11
maximum eigenvalue proximity radius             7.541469516753379e-11
reference M-orthogonality Frobenius upper        1.0926234959797678e-9
minimum adjacent proximity-interval separation  8.212190464007561e-5
retained/omitted cutoff interval separation      0.6015334187394926. (6)
```

All 241 proximity intervals are disjoint. Since each contains at least one
stored-pencil eigenvalue, they contain 241 distinct eigenvalues. This does not
exclude additional eigenvalues between the intervals, so it is not yet a
verified eigenvalue count or an indexed statement about the first 241 modes.
That companion count is now supplied by the directed sparse-inertia audit
documented in `neutral_strip_sparse_inertia_indexed_spectrum.md`. The
residual result itself remains a proximity theorem; the combined result
indexes all 241 stored-pencil intervals and transfers their eigenvalue
endpoints to the exact forms on the stored polygon.
A verified inertia computation or an equivalent block counting theorem is
still required at the retained cutoff.

## Modified projected chain

The 240-dimensional modified chain also passes its directed checks:

```text
maximum standard-eigenpair residual L2 upper       1.1384725498967043e-11
orthogonality Frobenius upper                       1.893421204981003e-12
restricted-mass Cholesky reconstruction upper      1.7417789381624251e-12
stiffness congruence reconstruction upper          2.133728032185990e-10.
```

These numbers confirm that ordinary linear-algebra residuals are not the
current scale obstruction. The endpoint effect of the residuals is still
unpromoted because the proof must either verify the retained spectral count
and apply a block perturbation theorem, or propagate the residual forcing
directly through a boundary-output Duhamel estimate.

## Artifacts and next obligation

The executable is
`scripts/neutral_strip_common_circle_eigensystem_residual_audit.py`; the full
241-row result is
`results/neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json`.

The first finite-input obligation below is now completed by
`neutral_strip_gaussian_weighted_assembly_interval_audit.md`:

1. The exact Gaussian-weighted mass, stiffness, boundary stiffness coupling,
   and boundary mass coupling are enclosed around the fingerprint-matched
   q12 matrices.
2. The remaining obligation is to verify the retained spectral count, or
   replace it by a block residual
   propagation theorem strong enough for the endpoint boundary output.

The Riesz/Gram/projected algebra and the endpoint effect of both assembly and
eigensystem errors still have to be enclosed before the residual scale in
(6) can be charged against the frozen time-slab screen.
