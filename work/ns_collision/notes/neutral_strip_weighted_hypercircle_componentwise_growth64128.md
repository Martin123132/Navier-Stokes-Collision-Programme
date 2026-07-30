# Componentwise Growth Audit Through 64,128 Pivots

## Result

The standalone componentwise congruence-residual theorem certifies the frozen
interval pencil through pivot 64,127, a prefix of 64,128 pivots. Precision-60
and precision-100 replays agree under a formal upper-bound nesting and
provenance check.

The certified reference signs are

- 32,564 negative;
- 31,564 positive;
- zero exactly zero.

This is a bounded-prefix inertia result only. It does not certify pivot 64,128
or later, the 123,816-pivot pencil, a weighted Ritz constant, continuum
transfer, or Navier-Stokes regularity.

## Numerical certificate

The precision-100 quantities are

| Quantity | Certified value |
| --- | ---: |
| minimum absolute reference diagonal | `0.0008150796928894088466677203542` |
| componentwise transformed-residual bound | `0.00022036990173119582409433780800379684...` |
| bound/minimum-diagonal ratio | `0.27036608033994526316957344688294149...` |
| safety factor | `3.6986888249541076085380810828427011...` |
| improvement over separated bound | `353.40789335342631298958007239211...` |

The old separated product remains fail-closed at ratio `95.5495...`.
Componentwise propagation closes because it retains the path geometry in
`Q R Q^T` instead of multiplying three unrelated global maxima.

## Why the extension is flat

Every precision-100 control metric is exactly unchanged between prefix lengths
64,064 and 64,128:

- the componentwise bound and ratio;
- the minimum diagonal and its pivot;
- the residual infinity norm and largest residual entry;
- both absolute inverse majorants;
- all three componentwise controlling pivots.

The growth factor is therefore exactly `1`.

The structural replay explains this plateau:

- the 64 added pivots are all edge-metric variables;
- all 64 new reference diagonals are negative;
- the smallest new absolute diagonal is `3.0233069502126737...`, far above
  the old global minimum;
- the 64,064 factor is bitwise equal to the leading block of the 64,128
  factor;
- the extension adds 192 strict-lower factor entries;
- the controlling componentwise row remains pivot 64,040;
- the global minimum remains edge-metric pivot 63,629.

Both ordered source prefixes and reconstructed binary reference factors were
independently rehashed. The stored precision artifacts reproduce those hashes
exactly.

## Runtime provenance

The production runs used one below-normal worker after the user explicitly
made CPU capacity available while unrelated jobs remained active. The
`--skip-cpu-policy` flag disabled only the default launch/park sampling for
that authorized run. It did not disable any mathematical construction,
directed-rounding, hash, provenance, precision-nesting, or fail-closed claim
check.

## Next gate

The next admitted target is exactly 64,256 pivots. This doubles the last local
increment from 64 to 128 after an exact plateau, while remaining well before
the next symbolic transition at pivot 76,921.

The sequence remains fail-closed:

1. produce the 64,256 separated precision-60 reference;
2. run the componentwise precision-60 certificate;
3. continue to precision 100 only if the componentwise ratio is strictly
   below one;
4. do not infer anything about a larger prefix if that gate fails.

