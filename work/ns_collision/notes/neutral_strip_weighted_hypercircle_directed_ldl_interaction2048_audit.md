# Directed-LDL interaction audit through pivot 2,048

## Scope

This stage extends the verified componentwise directed-Decimal LDL prefix
from 1,024 to exactly 2,048 pivots. The stopping point was chosen to cross
the first input-graph-dependent pivot at index 1,738 without approaching the
full 123,816-pivot calculation.

The run retains:

- the exact stored binary64 center/radius interval family;
- the ten-step positive symmetric Ruiz congruence;
- the frozen `MMD_AT_PLUS_A` elimination order;
- dynamically generated symbolic fill from input edges and prior cliques;
- 64-pivot atomic checkpoints; and
- the adaptive precision schedule `50,80,120`.

The numerical SuperLU factor supplies only the fixed order and comparison
values. Its asymmetric stored `L/U^T` pattern is not used to decide
certification fill.

## Precision-50 result

The first precision attempt completes all 2,048 pivots:

- 1,516 negative and 532 positive pivots;
- minimum margin over the whole prefix
  `0.1464628437608216996667076129` at pivot 1,564;
- maximum pivot-radius-to-margin ratio
  `4.0904215736535162752778652906803267784255617245686e-13`;
- maximum cancellation charge
  `1.6238482964115980096664253286023819129619964955402`;
- 5,612 symbolic lower entries;
- at most 3 descendants in a processed column; and
- no symbolic coordinate missing from the symmetric floating-factor
  envelope.

The block inventory is:

```text
edge metric            532
triangle constraint    310
state                     0
source triangle       1,206
```

The minimum whole-prefix margin occurs at an independent positive edge
pivot with no diagonal update. It is not the worst recurrence-bearing
pivot.

## Interaction region

Exactly 310 pivots have prior diagonal terms. They are indices
`1738..2047`, all belong to the triangle-constraint block, and all are
strictly negative.

Within this interaction region:

- minimum pivot margin:
  `0.7879861706481846044216128619` at index 1,909;
- maximum radius-to-margin ratio:
  `4.0904215736535162752778652906803267784255617245686e-13`
  at index 1,896;
- maximum cancellation charge:
  `1.6238482964115980096664253286023819129619964955402`
  at index 1,887;
- maximum lower-interval width:
  `1.1777498480601589210277481733054195364e-12`
  at index 2,015;
- maximum diagonal recurrence terms per pivot: 2;
- maximum total off-diagonal recurrence terms per pivot: 3; and
- maximum descendants per pivot: 3.

Thus the first genuine recurrence interactions remain comfortably separated
from zero. No precision escalation is needed.

## Precision-80 replay and resource park

The first precision-80 attempt starts under an admissible daytime baseline,
then sees total CPU samples `83.0%` and `95.3%`. It atomically parks after
pivot 128 with 256 lower intervals and no failed pivot. No worker is left
running.

A later admissible baseline resumes that exact hash-bound state. Resume
validation reconstructs the symbolic coordinates and checks every stored
interval before continuing. The run then completes all 2,048 pivots.

The cross-precision checks all pass:

- both endpoint arrays have length 2,048;
- every precision-80 pivot interval is inside its precision-50 interval;
- every precision-80 lower interval is inside its precision-50 interval;
- all signs agree;
- symbolic-coordinate hashes agree; and
- interaction ranges and block counts agree.

This is a precision replay of the same verified recurrence, not an
independent analytic proof method.

## Next structural boundary

A symbolic-only scan through pivot 8,192 performs no interval arithmetic and
certifies no new signs. It locates the next fill-complexity transition:

- diagonal term count first rises from 2 to 4 at pivot 2,270;
- off-diagonal common-term count first rises from 1 to 2 at pivot 2,270;
- descendant count first rises from 3 to 4 at pivot 2,274.

At a bounded endpoint of 2,304 the symbolic profile would contain 6,410
lower entries, maximum diagonal term count 4, maximum descendant count 4,
and maximum off-diagonal common-term count 2. No state-block pivot appears
in the first 8,192 positions.

The next arithmetic stage should therefore stop at exactly 2,304, crossing
the new complexity boundary by only 34 pivots. It must use a separate
checkpoint and require another precision-80 nesting replay.

## Decision

The 2,048-pivot interaction-bearing directed-LDL certificate closes. This
is stronger than the 1,024-pivot infrastructure pilot because it verifies
310 nontrivial elimination pivots and their propagated lower intervals.

It still certifies only a prefix. Full inertia `61908/61908/0`,
`kappa_h<0.045`, the global Ritz projection constant, continuum spectral
capture, and every Navier-Stokes regularity claim remain false. No full
123,816-pivot run should begin from this result.
