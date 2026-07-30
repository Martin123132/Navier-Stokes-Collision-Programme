# Common-circle endpoint roundoff audit

## Purpose and scope

The frozen finite-block time-slab certificate evaluates common-circle output
differences at every substep endpoint from `t=3/8` through `t=6`. Its original
`5e-11` relative arithmetic guard was intentionally conservative but had not
been derived operation by operation. This audit replaces that unsupported part
of the guard by an outward-rounded error budget.

The input matrices, generalized eigenvalues and modes, and source coefficients
are the stored binary64 numbers. They are treated as exact inputs to this
stage. Thus the conclusion is an endpoint-arithmetic enclosure for the frozen
binary problem, not an interval enclosure of finite-element assembly or of the
exact generalized eigensystems.

## Directed arithmetic model

For a nonnegative binary64 dot product with `n` elementary operations, the
implementation uses the conservative factor

```text
gamma_n = n eps / (1-n eps),                              (1)
```

where `eps=2^-52`; this is twice the usual unit roundoff. Every derived upper
bound is inflated toward `+infinity` with `nextafter`. Matrix products are
bounded by products of entrywise absolute values, so cancellation cannot make
the error estimate smaller.

For each stored binary64 eigenvalue `lambda` and endpoint `t`, both values are
first converted to their exact integer ratios. `mpmath.iv` at 60 decimal digits
then encloses

```text
exp(-lambda t).                                           (2)
```

The distance from the binary64 central value to the outward-rounded endpoints
of (2) is propagated through the mode scaling and dense source action.

## Riesz and Gram errors

Let `M_Gamma` be the stored polygon-boundary trace mass, `V` the stored Riesz
density modes, and `L` the stored conormal load modes. The computed residual
is enclosed entrywise, including its matrix-product and subtraction roundoff:

```text
R = M_Gamma V-L.                                          (3)
```

A Gershgorin lower bound `m_Gamma>0` gives

```text
||V-M_Gamma^(-1)L||_2 <= ||R||_F / m_Gamma.               (4)
```

The same absolute-product model encloses precomputation errors in the sparse
cross and pushed common-circle Gram actions. These errors are then propagated
through all 112 entry columns at every endpoint.

For the quadratic common-circle expression

```text
q^2 = m^T G_mm m - 2 m^T G_mr r + r^T G_rr r,             (5)
```

the audit pays separately for action error, multiplication and summation
roundoff, the final three-term combination, and the Riesz solve error (4).
Taking the square root only after adding the nonnegative squared-error budget
produces an upper norm; the solve term is then added by the triangle
inequality.

## Production result

The `h=0.06`, quadrature-order 12, 240-mode audit covers all 451 endpoints at
substep `0.0125`. Every check passes:

```text
maximum endpoint roundoff norm upper       3.010754666410437e-11
minimum margin under the existing guard    2.668598279252563e-11
worst endpoint and entry column             t=0.375, column 3
boundary-mass Gershgorin lower              1.8697504183912635e-2
Riesz residual Frobenius upper              6.424512520815097e-13
Riesz solve operator-error upper            3.436026786047070e-11. (6)
```

The Riesz solve residual is the dominant structural contribution. As an
independent check, the worst endpoint and column were reconstructed directly
at 80 decimal digits from the exact binary input values, applying the sparse
cross and pushed Grams after reconstructing the density. Its norm differs from
the binary64 central route by `2.2126397936084174e-14`, well inside (6).

The machine-readable result is
`results/neutral_strip_h006_q12_k240_endpoint_roundoff_audit_v1.json`; the
executable is
`scripts/neutral_strip_common_circle_endpoint_roundoff_audit.py`.

## What remains open

This stage proves that the endpoint arithmetic guard dominates the actual
roundoff budget for the stored frozen data. It does not prove that the stored
coefficient matrices enclose exact assembled integrals, that the generalized
eigenpairs enclose the exact discrete eigensystem, or that the discrete low
space encloses the continuum Ritz projector. Those input-level errors must be
bounded and propagated before the frozen finite-block result can be promoted
to a full discrete interval certificate.
