# Neutral-strip same-scale width sweep

## Question

Geometric wall exit does not earn the cubic true-split factor. Can the strip
be widened or narrowed until the complete same-scale wall branch closes
without that payment?

The static `rho=0` affine endpoint is enough to test this route. For every
strip half-width `Y>2`, compute

```text
k_R(Y)=E[exp(tau)1_R P_H(tau)],
k_S(Y)=E[exp(tau)1_S],

C_same(Y)=g_H^2[k_R(Y)^2+k_S(Y)^2].                  (1)
```

No split factor appears in (1).

## Competing effects

The two ends of the sweep fail for different reasons.

1. As `Y` approaches the entry radius two, top entries strike the wall almost
   immediately, so `k_S` approaches one and `g_H^2 k_S^2>1`.
2. As `Y` grows, wall hitting is delayed. The deformation weight `exp(tau)`
   then enlarges `k_S`; eventually its resolvent approaches the killed
   spectral pole at rate one.

An interior optimum is therefore plausible but not automatically below one.

## Pilot sweep

The boundary-fitted finite-state calculation samples `Y` from `2.02` to
`3.5`, adds a local grid near the apparent optimum, and uses 64 entry angles
for every width. It separately varies spatial resolution, semigroup time
step, and the artificial `x` truncation near the optimum.

The worst same-scale criterion falls from about `1.29` near `Y=2.02` to a
minimum around `Y=2.25-2.30`, but remains approximately `1.16`. It then rises
rapidly as the wall moment approaches its pole. The result is stable under
the recorded refinement stresses and does not depend on the conditionally
paid rows.

Thus simple width tuning does not repair the current global-level
architecture, at least in this static fitted-grid pilot.

## Scope

This is not a certified minimization over continuous `Y`; a proof-level
no-go still needs discretization enclosures and a continuous-width argument.
But the observed deficit is much larger than mesh, time, and truncation
movement, making further blind strip tuning low priority.

The honest alternatives are:

1. prove a new geometry-triggered scale transition, including its gauge,
   pressure, and many-generation accounting;
2. replace the strip wall by a storage boundary whose complete unpaid exit
   operator is contractive.

The sweep and diagnostics are reproduced by
`scripts/neutral_strip_same_scale_width_sweep.py`.
