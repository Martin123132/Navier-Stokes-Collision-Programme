# Standalone Residual Certificate Through First State Entry

## Scope

This checkpoint certifies the leading 63,680 pivots of the stored
123,816-dimensional weighted-hypercircle threshold pencil by a standalone
congruence-residual proof. It does not use a directed-LDL audit as an
assumption, sign oracle, or admission condition.

It does not certify full inertia, the weighted global Ritz constant,
continuum spectral capture, Navier-Stokes regularity, or a Clay-prize claim.

## Standalone proof contract

The congruence-residual theorem was already independent at the arithmetic
level:

```text
A = L (D + L^-1 (A - L D L^T) L^-T) L^T.
```

The old pilot wrapper nevertheless required a matching directed-LDL result.
The new runner removes that requirement and binds the calculation directly
to:

- SHA-256 hashes of the complete assembly result, interval matrix archive,
  Gaussian assembly result, and Gaussian checkpoint;
- hashes of the frozen Ruiz scale, raw permutation, elimination order,
  factor pattern, and full reference diagonal;
- hashes of the ordered prefix indices and positive scale;
- hashes of the exact ordered center and radius prefixes; and
- hashes of the exact binary64 reference `L`, `D`, and combined factor.

No directed audit path is accepted or loaded.

Standalone precision-60 replays at 2,304, 32,064, and 33,280 pivots reproduce
every historical residual norm, inverse bound, minimum diagonal, safety
ratio, and sign count exactly. This admits the first-state-entry pilot without
borrowing its conclusion from directed LDL.

## Certificate at 63,680 pivots

The standalone calculation closes at precisions 60 and 100:

| Quantity | Certified value |
| --- | ---: |
| Negative reference diagonals | 32,392 |
| Positive reference diagonals | 31,288 |
| Zero reference diagonals | 0 |
| Minimum absolute reference diagonal | `0.0008150796928894088466677203542` |
| Transformed residual upper bound | `2.00651867683450134e-7` |
| Bound/minimum-diagonal ratio | `0.0002461745390467909253534533381` |
| Reciprocal safety factor | `4062.158515141681` |
| Reference `L` nonzeros | 98,982 |

Every reported precision-100 upper bound is no larger than its precision-60
counterpart. Both runs have the same standalone contract, binary reference
factor hash, signs, and minimum diagonal.

## What entered

The first state pivot is 63,644. Eleven state pivots occur by pivot 63,679:

- all 11 are positive;
- their smallest absolute reference diagonal is
  `0.6366957100696649662907589118`; and
- none is responsible for the global small diagonal.

The global minimum occurs earlier, at edge-metric pivot 63,629. Thus the first
state variables themselves are well separated at this boundary. The delicate
feature is an edge-metric recurrence immediately before state entry.

The full 63,680 block/sign profile is:

| Block | Pivots | Negative | Positive |
| --- | ---: | ---: | ---: |
| Source triangle | 30,954 | 30,954 | 0 |
| Triangle constraint | 30,954 | 1,017 | 29,937 |
| Edge metric | 1,761 | 421 | 1,340 |
| State | 11 | 0 | 11 |

Between pivots 33,280 and 63,679, all 29,692 newly processed
triangle-constraint pivots are positive. The 697 new edge-metric pivots split
into 421 negative and 276 positive pivots.

## Risk growth

Relative to the 33,280 standalone certificate:

- the minimum diagonal is smaller by a factor of about `4.05`;
- the absolute inverse one-norm bound grows by about `38.92`;
- the absolute inverse infinity-norm bound grows by about `45.17`;
- the transformed residual bound grows by about `2065.43`; and
- the final bound/minimum-diagonal ratio grows by about `8373.32`.

The proof still closes by a factor above 4,062. The trend means that inverse
majorant growth, rather than state-pivot separation, is now the dominant
finite-dimensional risk. A long extrapolation is not justified.

## Next bounded gate

The next compact symbolic transition cluster is:

```text
63733, 63735, 63900, 64043, 64044, 64049, 64056.
```

The next target is therefore exactly 64,064 pivots at precisions 60 and 100.
This is a bounded state-region diagnostic, not permission for a full run.
Full inertia and all continuum claims remain false.
