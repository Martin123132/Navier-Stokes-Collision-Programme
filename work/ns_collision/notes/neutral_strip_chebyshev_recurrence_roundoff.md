# Certified one-step Chebyshev recurrence roundoff

## Purpose

The scaling and scalar coefficients are certified independently. This stage
encloses the floating sparse matrix recurrence used for all 112 point-source
columns during one `3/8` step.

## Stable error propagation

Let the computed Chebyshev states satisfy

```text
T_hat_0=y,
T_hat_1=fl(X y),
T_hat_(n+1)=fl(2 X T_hat_n-T_hat_(n-1)).
```

Writing the first-action error as `e_1` and each later local arithmetic error
as `delta_r`, self-adjointness gives the exact identity

```text
e_n
 = U_(n-1)(X)e_1
   + sum_(r=2)^n U_(n-r)(X) delta_r.
```

The previous scaling certificate and the small directed construction error
place the computational scaled spectrum inside `[-1,1]`. Hence

```text
||U_j(X)|| <= j+1.
```

The checker applies these polynomial weights to directed local sparse-product,
doubling, and subtraction errors. It does not use the exponentially growing
bound obtained by replacing the recurrence with
`e_(n+1)<=2e_n+e_(n-1)+delta_n`.

## Other one-step charges

The result also includes:

1. sequential polynomial accumulation roundoff;
2. the certified SciPy coefficient implementation error;
3. the exact degree-320 polynomial tail;
4. point-source inverse-square-root construction roundoff;
5. Duhamel transfer from the exact normalized stored generator to the
   computational scaled generator.

The maximum of the resulting one-step state-action error is recorded across
all 112 sources.

## Scope

This stage certifies the full-state sparse action for one step. The reduced
240-dimensional semigroup, boundary multiplication, repeated 16-step error,
within-window suprema, and post-time-6 tail remain open. No screen value is
updated.

The executable is
`scripts/neutral_strip_chebyshev_recurrence_roundoff_certificate.py`.
