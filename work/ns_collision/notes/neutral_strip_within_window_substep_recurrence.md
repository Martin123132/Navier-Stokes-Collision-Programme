# Within-window substep recurrence

## Universal action bound

The `3/80` scalar coefficient certificate is combined with the already
certified scaled stored generator. Sparse matrix multiplication uses a
directed `gamma_n` error and the absolute scaled-operator norm. The recurrence
errors are propagated with

```text
e_n = U_(n-1)(X)e_1
      + sum_(r=2)^n U_(n-r)(X) delta_r,
```

and `||U_j(X)|| <= j+1` for self-adjoint spectrum in `[-1,1]`.
Coefficient implementation error, infinite polynomial tail, accumulation
roundoff, and exact-to-computational generator transfer are charged
separately.

## Repeated substeps

For exact substep `S` and computed polynomial `P`,

```text
||P^n-S^n||
 <= n epsilon_step max(||P||,||S||)^(n-1).
```

The certificate evaluates this bound for every `n=1,...,160`, rather than
using only its value at `t=6`.

## Scope

This closes the arbitrary-state sparse recurrence error for the time-slab
step. The actual 112-source grid, boundary output at each substep,
second-derivative interpolation charge, 15 finite-window suprema, and
post-time-6 tail remain open.

The executable is
`scripts/neutral_strip_within_window_substep_recurrence_certificate.py`.
