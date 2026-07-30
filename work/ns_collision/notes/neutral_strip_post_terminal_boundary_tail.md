# Stored-chain post-terminal boundary tail

## Purpose

This stage closes the boundary-discrepancy windows beginning at `t=6`.
Together with the fifteen certified finite windows, it completes the
boundary-leakage charge for the stored binary finite chain.

It does not perform continuum Ritz or polygon-to-circle transfer.

## Terminal-state enclosure

The hash-verified source-grid checkpoint stores the full state and reduced
coordinates at `t=6` for all 112 point sources. The certificate:

1. encloses each computed full-state norm and adds the certified repeated
   full-state error;
2. reconstructs each reduced physical state with directed dense-product
   arithmetic and adds the certified exact-form and dense-state errors;
3. composes both exact terminal state norms with the certified exact
   boundary-output norm.

This is deliberately state-oriented. It does not assume that the very small
full-minus-reduced boundary cancellation at `t=6` persists later.

## Geometric tail

For the tail window beginning `m` steps after `t=6`, the full and reduced
boundary amplitudes are bounded by

`A_full exp(-1.9 m (3/8))`

and

`A_reduced exp(-2.36 m (3/8))`.

The two directed window ratios are strictly below one. Their geometric sums
are combined source by source, multiplied by the global axial upper bound
valid for every `t>=6`, and then maximized over the 112 sources.

## Production result

Artifact:

`work/ns_collision/results/neutral_strip_h006_post_terminal_boundary_tail_v1.json`

SHA-256:

`74a6051a2c529f2c9ddfeccd305eb4a14e6bc7e75e439afaa8b1d464cef54858`

Key values:

| Quantity | Certified upper |
| --- | ---: |
| Full terminal boundary amplitude | `0.00015269290724115976` |
| Reduced terminal boundary amplitude | `0.00015265527907074635` |
| Post-terminal raw sum | `0.0002734111616928798` |
| Post-terminal screen charge | `0.00015910925687251309` |
| Finite-window screen charge | `0.0030907844288031805` |
| Complete stored-chain screen | `0.9733389129178374` |
| Complete stored-chain headroom | `0.026661087082162634` |

All fourteen tail and screen-composition checks pass.

## Remaining gates

The stored finite-chain boundary-leakage screen is complete, but it is not a
continuum estimate. The next obligations are the continuum Ritz-projector
transfer and polygon-to-circle domain transfer. Until those are certified,
this result is not a Navier-Stokes regularity estimate or proof.
