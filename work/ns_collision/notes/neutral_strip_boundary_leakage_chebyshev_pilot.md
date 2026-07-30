# Source-oriented boundary leakage: Chebyshev feasibility gate

## Question

The certified two-block theorem controls state-space leakage, but multiplying
its state norm by the full boundary operator norm loses the available screen
headroom. The relevant source family is much narrower: 112 point sources on
the entry circle, evolved for at least one half-window.

This stage asks whether that directed family can be propagated through the
stored modified finite chain with enough numerical separation to justify a
full interval implementation.

## Directed quantity

In mass-normalized coordinates let

```text
H=M^(-1/2) A M^(-1/2),
Q=M^(1/2) V (V^T M V)^(-1/2),
T=Q^T H Q,
O=sqrt(a^(-1)) B^T M^(-1/2).
```

For each discrete entry point `z`, the pilot evaluates

```text
O [exp(-tH) M^(-1/2)e_z
   - Q exp(-tT) Q^T M^(-1/2)e_z].
```

Thus the diagnostic includes both the initial source-projector mismatch and
the later low/high dynamical leakage. It does not replace either term by a
global state norm.

## Polynomial action

The certified two-block constants imply a full stored-chain floor above
`1.9`. A central binary64 Gershgorin calculation gives an upper value below
`7988`, so the pilot scales the matrix to `[-1,1]` using `[1.9,8000]`.
For one window `h=3/8`,

```text
exp(-hH)
 = exp(-h c) [I_0(a) + 2 sum_(k>=1) (-1)^k I_k(a) T_k(X)],
```

where `c=(8000+1.9)/2`, `a=h(8000-1.9)/2`, and `X=(H-cI)/a*h`.
Scaled Bessel coefficients are evaluated with `scipy.special.ive`.

Degree 320 is used for the production pilot. The first step also records
degrees 200, 240, 280, and 320. Degree 280 and 320 differ by about
`1e-12` in the largest reported boundary norm, while the sampled scalar tail
at degree 320 is about `7e-17`.

## Fail-closed status

The calculation is deliberately a pilot. The following are still open:

1. a directed interval Gershgorin upper bound for the normalized matrix;
2. interval evaluation of all Bessel coefficients and the infinite tail;
3. accumulated sparse-recurrence and output-multiplication roundoff;
4. replacement of endpoint samples by suprema on every time window;
5. a certified tail after time 6.

Consequently no value from this file is charged to the production screen.
The existing certified screen and its headroom remain unchanged.

## Reproducibility

The executable is
`scripts/neutral_strip_boundary_leakage_chebyshev_pilot.py`. It checkpoints
the full and reduced state after every `3/8` step, hash-verifies that
checkpoint on resume, and runs below normal process priority.
