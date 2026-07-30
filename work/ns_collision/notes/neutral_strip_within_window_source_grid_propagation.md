# Within-window actual-source grid

## Atomic layout

The stored full state is propagated with the certified `3/80` Chebyshev
step. The 240-dimensional reduced state is evaluated directly from its
eigendecomposition at each grid time, avoiding an additional repeated dense
step error. Checkpoints are written only after blocks of ten substeps, so
every checkpoint lies on a production multiple of `3/8`.
The first block reaches `t=3/8`; the next 15 blocks cover the 15 finite
windows through `t=6`.

Each checkpoint consists of an fsynced NPZ state file and hash-bound JSON
metadata. The metadata records every certified boundary row, production
endpoint cross-check, and post-block CPU sample.

## Grid-point enclosure

For substep `n`, the full-state action error uses

```text
||P^n-S^n||
 <= n epsilon max(||P||,||S||)^(n-1).
```

The exact source-construction error is propagated by the computed contraction.
For `t >= 3/8`, the first-endpoint reduced-form and dense arithmetic errors
are extended by

```text
(t/(3/8)) exp(-2.36 (t-3/8)).
```

This factor dominates both the Duhamel generator term and the decaying
source, trial, and evaluation terms.

The common-circle output is evaluated through guarded sparse multiplication
for the full state and guarded dense multiplication for the reduced state.
The construction error of the precomputed reduced output matrix, subtraction,
112-component norm evaluation, exact output-map discrepancy, and all state
errors are charged separately. State norms used with the output-map
discrepancy are directed column norms; the reduced trial map uses a directed
Frobenius upper bound rather than an unguarded spectral norm.

At each tenth substep, the resulting certified interval must overlap the
independently certified production-endpoint interval.

## Scope

Completing this script certifies 151 grid points from `t=3/8` through `t=6`.
It does not assume monotonicity and does not yet control values between grid
points. The second-derivative interpolation charge, post-time-6 tail,
continuum transfer, and Navier-Stokes regularity remain open.

The executable is
`scripts/neutral_strip_within_window_source_grid_propagation.py`.
