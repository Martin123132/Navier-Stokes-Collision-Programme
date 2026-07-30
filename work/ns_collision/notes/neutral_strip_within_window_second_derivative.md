# Stored-chain within-window interpolation certificate

## Purpose

The 151 certified values on the `3/80` time grid do not by themselves
control the boundary discrepancy between adjacent grid times. This stage
closes that gap for the stored binary finite chain on

`3/8 <= t <= 6`.

It does not address the post-`t=6` tail or any continuum/domain transfer.

## Interpolation theorem

For a Hilbert-valued `C2` function `f` on an interval of length `h`,

`sup ||f|| <= max(||f(left)||, ||f(right)||) + h^2 sup ||f''|| / 8`.

The certificate uses `h=3/80` and the already-certified endpoint upper
bounds. No monotonicity between grid points is assumed.

## Direct derivative calculation

For each point source, the full stored-chain boundary response has second
derivative

`O H exp(-t H) H Y`.

The script evaluates this expression directly at every one of the fifteen
window starts. Each full-chain row uses the certified degree-320 `3/8`
semigroup action and the repeated-step operator error. The enclosure also
adds:

- exact-to-computational generator error;
- boundary-output construction and geometry error;
- point-source construction error;
- sparse generator-product roundoff;
- final dense boundary-product roundoff.

The reduced row is evaluated independently at the same time. Directed
scalar exponential intervals and dense arithmetic control the stored
orthonormal eigensystem path. A separate perturbation bound transfers that
path to the exact restricted mass/stiffness forms, source coordinates, and
boundary output.

An earlier provisional implementation evaluated the direct derivative only
at `t=3/8` and decayed that output norm. Review rejected that step: spectral
components can cancel differently at later times. The production result
contains fifteen direct full rows and fifteen direct reduced rows, and the
test requires every interpolation row to use its matching direct pair.

## Production result

Artifact:

`work/ns_collision/results/neutral_strip_h006_within_window_second_derivative_v1.json`

SHA-256:

`11341a59cbddd3c98747cefe558dc073f98f7c7964d8949d4326935f80e011b8`

Key values:

| Quantity | Certified upper |
| --- | ---: |
| First full second derivative | `13.756172035123853` |
| First reduced second derivative | `8.694119070777171` |
| First discrepancy second derivative | `22.450291105901027` |
| Maximum `h^2/8` interpolation charge | `0.003946340233459166` |
| Finite raw source-discrepancy sum | `0.005311161511479153` |
| Finite-window screen charge | `0.0030907844288031805` |
| Existing certified screen | `0.9700890192321616` |
| Finite-window combined diagnostic | `0.9731798036609649` |
| Remaining diagnostic headroom | `0.0268201963390351` |

All fifteen windows and all 150 subslabs pass. The direct derivative rows
happen to decrease strictly, but that observation is not used to replace
any direct calculation.

## Remaining gate

The result deliberately retains:

- `post_terminal_time_tail_certified = false`;
- `screen_updated = false`.

The next finite-chain obligation is a separate post-`t=6` boundary
discrepancy tail anchored at the certified sixteenth endpoint. Only after
that tail is enclosed may the finite-window and tail charges be installed
in the production screen.
