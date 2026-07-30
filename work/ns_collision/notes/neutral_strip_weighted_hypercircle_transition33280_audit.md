# Weighted Hypercircle Transition-33280 Audit

## Scope

This checkpoint certifies only the leading 33,280 pivots of the stored
123,816-dimensional weighted-hypercircle threshold pencil. It uses the frozen
entrywise interval input, ten-step positive Ruiz congruence, and
`MMD_AT_PLUS_A` elimination order.

It does not certify full inertia, the weighted global Ritz constant,
continuum spectral capture, Navier-Stokes regularity, or a Clay-prize claim.

## Two-route certificate

Componentwise directed-Decimal LDL certified all 33,280 pivots at precision
schedules `50,80,120` and `80`. The result is:

- 31,971 negative and 1,309 positive pivots;
- zero unresolved pivots;
- minimum margin `0.003304358585502960725532827806`;
- 131,628 symbolic lower entries; and
- maximum pivot-radius/margin ratio
  `5.9196727264835606493920324436e-10`.

Every precision-80 pivot interval and lower interval nests in its
precision-50 enclosure. The fresh 33,280 run also reproduces every stored
precision-50 diagnostic from the prior 32,064 prefix.

The independent congruence-residual method certified the same signs at
precisions 60 and 100. Its precision-60 transformed-residual/minimum-diagonal
ratio is

```text
2.9399878879713527131431104875e-8.
```

All six upper bounds are no larger at precision 100. The directed recurrence
and congruence-residual proof therefore both close, while sharing only the
frozen input, scaling, and order.

## Newly added segment

The 1,216 new pivots, indices 32,064--33,279, contain:

| Quantity | Value |
| --- | ---: |
| Negative pivots | 485 |
| Positive pivots | 731 |
| Edge-metric pivots | 486 |
| Triangle-constraint pivots | 730 |
| State pivots | 0 |
| Exact-zero input diagonals | 730 |
| Maximum diagonal terms | 4 |
| Maximum off-diagonal recurrence terms | 10 |
| Maximum descendants | 6 |

## Delicate pivot 32,849

The new minimum occurs before the predicted fill transition. Pivot 32,849 has
an exact-zero input diagonal and is generated entirely by recurrence:

```text
precision-50 interval:
[0.00330435858550296072553282780603931492200306444314,
 0.00330435858941510500495756248256247532827734921432]
```

Its precision-80 interval nests strictly inside this enclosure. The pivot has
two diagonal terms, six off-diagonal recurrence terms, five descendants,
cancellation charge at most `422.5558991531053`, and radius/margin ratio below
`5.92e-10`.

Relative to the 32,064 checkpoint:

- the minimum margin is smaller by a factor of about `44.32`;
- maximum cancellation is larger by about `260.22`;
- maximum radius/margin is larger by about `811.56`; and
- the independent residual ratio is larger by about `348.44`.

Both certificates still close decisively. These changes are nevertheless a
real warning that numerical difficulty is not scaling linearly with pivot
count.

## Fill transition at 33,224

The symbolic descendant count first rises from five to six at pivot 33,224.
The 56 certified pivots from 33,224 through 33,279 are all positive. Their
minimum margin is `0.1660553289129427032568830705`, and their largest
radius/margin ratio is below `2.351e-13`.

Thus the predicted fill transition itself is benign at this boundary. The
more important new phenomenon is the cancellation-sensitive zero-diagonal
pivot at 32,849.

## Next gate

The next unprocessed symbolic transition is at pivot 62,972, and the first
state pivot is 63,644. The intended bounded state-entry target remains 63,680.

The existing congruence-residual program still requires a matching certified
directed-LDL audit for provenance and sign comparison. Calling it
"residual-only" at 63,680 would therefore be inaccurate. The next stage is:

1. implement a standalone residual mode whose input contract is independently
   hash-bound to the frozen pencil, scale, order, and prefix length;
2. test it against the existing 2,304, 32,064, and 33,280 certificates;
3. run only the 63,680 state-entry pilot at precisions 60 and 100; and
4. keep full directed LDL, full inertia, and all continuum claims false.
