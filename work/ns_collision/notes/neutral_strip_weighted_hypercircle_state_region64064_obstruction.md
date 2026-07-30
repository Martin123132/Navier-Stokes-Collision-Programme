# Standalone Residual Obstruction at 64,064 Pivots

> **Resolved at this boundary:** this note records the failure of the original
> separated global-norm product. The later componentwise residual theorem
> certifies the same 64,064-pivot interval family. See
> `neutral_strip_weighted_hypercircle_componentwise_recovery64064.md`.

## Scope

This checkpoint tests the hash-bound standalone congruence-residual route
through pivot 64,063 of the stored 123,816-dimensional weighted-hypercircle
threshold pencil. It crosses the seven symbolic transitions at pivots

```text
63733, 63735, 63900, 64043, 64044, 64049, 64056.
```

The calculation does not use a directed-LDL audit. It does not certify the
64,064-pivot interval-family inertia, any later prefix, full inertia, a
weighted Ritz constant, continuum spectral capture, Navier-Stokes regularity,
or a Clay-prize claim.

## Fail-closed result

Standalone runs at decimal precisions 60 and 100 agree on the reference
factor, diagonal signs, minimum diagonal, and every provenance hash. Every
higher-precision upper bound is no larger than its precision-60 counterpart.
The precision crosscheck fails only its required condition that both residual
certificates close.

The numerical reference data are:

| Quantity | Value |
| --- | ---: |
| Negative reference diagonals | 32,500 |
| Positive reference diagonals | 31,564 |
| Zero reference diagonals | 0 |
| Minimum absolute reference diagonal | `0.0008150796928894088466677203542` |
| Transformed residual upper bound | `0.07788046272932349` |
| Bound/minimum-diagonal ratio | `95.54950688716327` |
| Reference `L` nonzeros | 102,873 |

These are signs of the fixed central reference, not a certified inertia for
the interval family. The bound exceeds the smallest reference diagonal by a
factor of about 95.55, so the standalone proof is correctly fail-closed.

## What changed

An independent reconstruction proves that the complete 63,680 factor is
bit-for-bit equal to the leading block of the 64,064 factor. The previously
certified prefix has not been altered by an inconsistent factorization.
The global minimum is also unchanged: it remains edge-metric pivot 63,629,
original index 20.

Relative to the certified 63,680 prefix:

| Quantity | Growth factor |
| --- | ---: |
| Minimum-diagonal reduction | `1` |
| Absolute inverse one-norm majorant | `130.6988` |
| Absolute inverse infinity-norm majorant | `99.5610` |
| Residual infinity norm | `29.8280` |
| Transformed residual bound | `388137.2430` |
| Bound/minimum-diagonal ratio | `388137.2430` |

The failure is therefore not caused by a new smaller diagonal. It is the
product of inverse-majorant amplification and a larger binary-reference
residual.

The 384 newly included pivots comprise 257 edge-metric pivots and 127 state
pivots. In the central reference, the edge pivots split 103 negative and 154
positive, while the state pivots split 5 negative and 122 positive. No new
triangle or source pivots occur in this interval.

## Localized amplification

The inverse-norm product develops in three stages:

| Prefix length | Inverse-norm product upper bound | Growth |
| --- | ---: | ---: |
| 63,680 | `9.0357e5` | baseline |
| 63,734 | `1.2884e6` | `1.426x` |
| 63,901 | `9.4197e7` | `73.112x` |
| 64,044 | `1.1758e10` | `124.820x` |
| 64,064 | `1.1758e10` | `1x` |

The final infinity-majorant maximum is edge pivot 64,040, original index 232.
Its largest recurrence contribution is the coefficient
`5167.42789596299` multiplying the earlier delicate edge pivot 63,629.
The final one-majorant maximum is triangle-constraint pivot 62,917. Its
dominant path runs through pivot 63,629 and then edge pivots 64,039 and
64,040.

The largest residual entry is also the `(64040, 64039)` edge pair, with upper
bound about `3.1335e-12`. Thus the residual defect and inverse amplification
are concentrated on the same local chain. The later symbolic transitions at
64,043, 64,044, 64,049, and 64,056 add no further inverse-majorant growth.
They are crossed by the run, but they are not the immediate source of the
final jump.

## Consequence

No larger-prefix calculation was admitted from this checkpoint. The next
finite task was to localize the earliest closure loss inside
`63680..64064`, starting with the bracket ending at 63,901 and then the
larger jump ending before 64,044.

The obstruction did remain localized to the 64,039--64,040 chain. The sharper
componentwise transformed-residual bound subsequently closed at both 64,040
and 64,064 pivots without changing the binary64 `L,D` reference. Merely
evaluating the old separated product with more Decimal digits did not help;
retaining componentwise path geometry did.
