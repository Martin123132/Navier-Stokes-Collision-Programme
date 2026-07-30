# Directed-LDL prefix audit for the hypercircle pencil

## Scope

This stage tests a componentwise verified LDL mechanism on a bounded prefix
of the completed `h=0.06`, `beta=0.045` hypercircle threshold pencil. It
does not attempt all 123,816 pivots and does not certify full inertia,
`kappa_h<0.045`, continuum spectral capture, or Navier-Stokes regularity.

The run fixes the previously selected ten-step positive Ruiz congruence and
`MMD_AT_PLUS_A` ordering. The frozen fingerprints are:

- Ruiz scale:
  `bef73a763a5ee24b85651a3c761b940b0fde2f8e04294a7d7e9b0c085c7ca9c2`;
- raw SuperLU permutation:
  `fc613374bffd7bba84293e3c302e56d0ef945a0530443b04dde5ba079adb36db`;
- derived elimination order:
  `5f8adf72fa6fe8e3ea62d716c6a2f34df7252fa0a45bd5cafb55fbf7001d5f13`;
- stored `L/U` patterns:
  `dc941205f68286d3f318a58670fd0ddf14bf63afdaf48202dcc9e0291238103b`;
- central `U` diagonal:
  `c3cf0559f421039b208855c6ba2a3bab9d887d4a8459ffc71c76b381bdfbf140`.

## Permutation and symbolic-fill correction

Small independent symmetric examples establish the SuperLU convention
directly. If `p = perm_r = perm_c`, the factor relation is

```text
K[argsort(p), argsort(p)] = L U,
```

not `K[p,p] = L U`. The production relative infinity residual for the
`argsort(p)` interpretation is below `1e-10`.

The floating factors are numerically close to `U = D L^T`, with relative
defect `6.559176562535337e-13`, but their stored patterns are not exact
transposes:

- 4,502 `L` locations are absent from `U^T`;
- 3,924 `U^T` locations are absent from `L`.

Therefore the numerical factor pattern is not used to decide which interval
entries must be propagated. At each exact symbolic pivot `k`, the code forms
the descendant set from:

1. every later input-graph neighbor of `k`; and
2. every later descendant of each earlier column `j` for which `L[k,j]`
   has already been retained.

This is the standard no-pivot LDL clique recurrence. It is independent of
floating cancellation and retains even interval entries that evaluate to
zero. SuperLU values are comparison diagnostics only.

## Directed arithmetic and checkpoint

Each stored binary64 center and nonnegative radius is converted exactly to
`Decimal`. Lower operations use `ROUND_FLOOR`; upper operations use
`ROUND_CEILING`. The positive binary64 Ruiz factors are included through
directed interval multiplication. Every pivot uses

```text
D[k] = K[k,k] - sum_j L[k,j]^2 D[j]
```

and every lower entry uses

```text
L[i,k] =
  (K[i,k] - sum_j L[i,j] L[k,j] D[j]) / D[k].
```

A pivot is accepted only when its complete interval excludes zero. The
checkpoint records every pivot endpoint, every lower-factor endpoint needed
for resume, signs, centers, radii, margins, cancellation charges, symbolic
coordinates, precision attempts, and the full input/factor contract. It is
written atomically every 64 pivots and protected by a canonical state hash.

The adaptive test suite includes:

- empirical proof of the inverse-permutation convention;
- containment of nine sampled dense 100-digit reference factorizations;
- a near-cancelling 2-by-2 example that fails closed at 8 digits and
  succeeds after automatic escalation to 40 digits; and
- one-pivot parking, exact resume, and checkpoint-corruption rejection.

## Production prefix

The 50-digit run certifies pivots `0..1023`:

- 1,024 negative and 0 positive pivots;
- minimum interval distance from zero
  `0.9999999999999202387999618073`;
- maximum pivot-radius-to-margin ratio
  `7.9705831101496535969750527484872661277522158316282e-14`;
- maximum cancellation charge
  `1.0000000000001594116622029934148654773937601527886`;
- 2,540 retained symbolic lower entries;
- at most 3 descendants in any processed column; and
- 0 symbolic coordinates absent from the symmetric floating-factor
  envelope.

All 1,024 pivots belong to the negative source-triangle block. None has a
prior diagonal-update term. Thus this prefix rigorously certifies those easy
signs and validates the input, scaling, ordering, interval, symbolic,
checkpoint, and resume machinery, but it has not yet stress-tested
cancellation propagation.

## Independent precision replay

A fresh 80-digit run certifies the same 1,024 signs. Every 80-digit pivot
interval is nested in its 50-digit counterpart, every lower-factor interval
is nested, and the symbolic-coordinate hashes agree. The precision
cross-check therefore passes all five recorded comparisons.

This is a cross-precision replay of the same recurrence, not an independent
proof algorithm. The dense 100-digit sample test supplies a separate
small-matrix reference check.

## Decision

The bounded directed-LDL mechanism closes for the first 1,024 pivots. The
full-inertia and continuum flags remain false.

The permuted input graph first allows a prior-factor contribution at pivot
index 1,738. The next bounded stage should therefore rerun to exactly 2,048
pivots, preserving 64-pivot checkpoints and the `50,80,120` precision
schedule. That extension crosses the first genuine recurrence interaction
by only 310 pivots. It should be accepted only if:

1. every pivot interval still excludes zero;
2. an 80-digit replay nests inside the 50-digit intervals;
3. cancellation and lower-interval widths remain controlled; and
4. the full 123,816-pivot inertia flag remains false.

No full directed run should begin until that interaction-bearing prefix has
closed and its first nontrivial update profile has been reviewed.
