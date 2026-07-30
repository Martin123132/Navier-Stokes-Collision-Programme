# Weighted Hypercircle Transition-32064 and Full-Workload Feasibility Audit

## Scope

This checkpoint advances the finite weighted-hypercircle threshold-pencil
calculation. It does not certify the full 123,816-dimensional inertia, the
weighted global Ritz constant, continuum spectral capture, Navier-Stokes
regularity, or a Clay-prize claim.

All arithmetic certificates use the stored entrywise interval family, the
frozen ten-step positive Ruiz congruence, and the frozen
`MMD_AT_PLUS_A` elimination order. The symbolic workload map uses the same
input graph and order but performs no interval arithmetic and certifies no
pivot sign.

## Transition at 2,304 pivots

The first bounded extension crossed the symbolic transition at pivots
2,270--2,274. Directed interval LDL certified all 2,304 pivots at Decimal
precisions 50 and 80:

- 1,768 negative and 536 positive pivots;
- minimum whole-prefix pivot margin
  `0.1464628437608216996667076129`;
- 6,410 stored lower intervals;
- identical signs and symbolic coordinates at both precisions; and
- every precision-80 pivot and lower interval nested in its precision-50
  enclosure.

The four new interaction pivots 2,270--2,273 are positive edge-metric pivots.
Their margins are about `1.518`, and the largest radius-to-margin ratio in the
new class is approximately `4.27e-13`. The transition therefore introduces
the predicted extra recurrence terms without causing visible interval
instability.

## Independent congruence-residual certificate

An independent interval-propagation route was added. Let the exactly
interpreted binary64 reference factorization be

```text
M = L D L^T
```

with unit lower triangular `L`, and let `A` range over the stored symmetric
entrywise interval family after the same positive scaling and permutation.
Writing `E = A - M` gives the exact congruence

```text
A = L (D + L^-1 E L^-T) L^T.
```

For symmetric `E`,

```text
||L^-1 E L^-T||_2
    <= ||L^-1||_1 ||L^-1||_inf ||E||_inf.
```

The implementation encloses the absolute inverse by the finite triangular
recurrence

```text
|L^-1| <= (I - |L-I|)^-1.
```

All residual, inverse-majorant, and norm operations are outward-rounded.
There are no interval pivot divisions. If the resulting transformed-residual
bound is strictly smaller than every `|D_ii|`, Weyl sign preservation and
Sylvester inertia certify the interval family.

At 32,064 pivots the precision-60 result is:

- reference signs: 31,486 negative, 578 positive, and zero zero pivots;
- minimum absolute reference diagonal:
  `0.1464628437608385402857180679`;
- residual infinity-norm upper bound:
  `1.8902378091947144049844896916709e-13`;
- transformed two-norm upper bound:
  `1.2357785285333108229672879287378e-11`; and
- transformed-bound/minimum-diagonal ratio:
  `8.4374882857746012204658263812e-11`.

A precision-100 replay reproduced the structure and signs and made all six
reported upper bounds no larger. This route shares the frozen interval input,
scale, and order with directed LDL, but its interval proof is independent of
the directed pivot recurrence.

## Directed LDL through 32,064 pivots

The componentwise directed-Decimal recurrence also certified the first 32,064
pivots at precisions 50 and 80:

- 31,486 negative and 578 positive pivots;
- minimum whole-prefix margin
  `0.1464628437608216996667076129` at pivot 1,564;
- 125,492 stored lower intervals;
- largest pivot-radius/margin ratio
  `7.294171315577793e-13`;
- largest lower-interval width
  `2.250003807527397e-12`; and
- complete precision-80-in-precision-50 nesting for all pivots and lower
  intervals.

There are 578 interaction-bearing pivots in this prefix: 532 negative
triangle-constraint pivots and 46 positive edge-metric pivots. In the newly
certified transition region 32,022--32,063, the weakest margin is
`0.6061445158051528443279307318` at pivot 32,051. No state pivot has yet
entered.

The two methods therefore independently certify the inertia of this bounded
leading problem. They do not certify any unprocessed pivot.

## Full symbolic workload

The exact potential-fill scan now covers the complete 123,816-pivot order.
Its principal totals are:

| Quantity | Total |
| --- | ---: |
| Symbolic lower entries | 2,699,822 |
| Diagonal recurrence terms | 2,699,822 |
| Off-diagonal common terms | 159,047,977 |
| Reference-product pair terms | 164,571,437 |
| Maximum descendants at one pivot | 419 |
| Maximum diagonal terms at one pivot | 1,531 |
| Maximum common terms at one lower entry | 1,409 |

The next local fill transition is at pivot 33,224, so 33,280 is the next
bounded target. The first state pivot is 63,644.

The workload is strongly backloaded. The final 9,128 pivots account for
approximately `95.3903%` of all off-diagonal common terms and `90.0494%` of
all reference-product pair terms. Runtime extrapolation from the 32,064
prefix is therefore invalid.

## Feasibility decision

The current certified boundary is 32,064 of 123,816 pivots, or approximately
`25.8965%` of the dimension. Disk and memory capacity are not the primary
blockers. The current monolithic checkpoint format is:

| Precision | Projected full checkpoint | Projected cumulative rewrites |
| --- | ---: | ---: |
| 50 | 519.26 MiB | 37.42 GiB |
| 80 | 723.39 MiB | 50.01 GiB |

With 512-pivot batches, running both precisions in that format would rewrite
about 87.4 GiB. A chunked or append-only hash-chained checkpoint is required
before a full launch.

A representative Decimal multiplication-and-accumulation kernel gave only a
lower bound of about 35.1 minutes at precision 50 and 31.7 minutes at
precision 80 for the full common-term count. These measurements deliberately
exclude set intersections, interval divisions, input/output, checkpoint
replay, and validation. They are not total-runtime predictions.

The full directed-LDL and full congruence-residual launches are therefore both
marked not ready. The recommended finite-stage sequence is:

1. Certify the 33,280 transition with both arithmetic routes.
2. Run a residual-only 63,680 state-entry pilot.
3. Replace monolithic checkpoint rewrites with chunked hash-chained records.
4. Only then decide whether a complete finite-inertia run is scientifically
   justified.

## Continuum dependency gates

Even a successful full finite inertia calculation would not prove continuum
capture. The remaining dependency order is:

1. certify the weighted global Ritz/solution-operator error, requiring strict
   bounds below `0.08557115750643675` and `0.007322422996991409`;
2. transfer the positive-time singular source;
3. transfer the smoothed conormal output;
4. control the polygon-to-circle domain perturbation; and
5. close the separate Navier-Stokes argument.

Every one of those continuum and Navier-Stokes flags remains false.
