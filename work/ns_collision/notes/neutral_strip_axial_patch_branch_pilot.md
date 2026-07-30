# Neutral-strip axial-patch branch pilot

## Purpose

The raw residual-moment stress for the stopped strip fails near the strongly
returning axis. That stress omits the finite axial acceptance patch on the
return branch. This note inserts the exact outward-OU patch probability and
also charges the complete affine deformation accumulated before a wall exit.

## Complete scalar branch gains

Let `tau` stop the transverse affine diffusion on either the inner circle
`R={r=1}` or the strip walls `S={|y|=2.1}`. The one-history stretching factor
is `exp(tau)`. Independently,

```text
dZ=(1+rho)Z dt+sqrt(2)dW,
V_rho(t)=[exp(2(1+rho)t)-1]/(1+rho).                  (1)
```

Starting at the centered axial point, the exact probability of landing in
the accepted patch `|Z_tau|<H`, `H=3/4`, is

```text
P_H(t)=erf[H/sqrt(2V_rho(t))].                        (2)
```

The complete scalar branch multipliers are therefore

```text
k_R=E[exp(tau)1_R P_H(tau)],
k_S=E[exp(tau)1_S].                                  (3)
```

The centered axial start maximizes (2), so it is conservative for the return
gain. The wall moment in (3) is the resolvent of `L_rho+1`; it is finite when
the killed transverse rate exceeds one.

If a wall hit is a genuine cubic split, the current scalar pair criterion is

```text
C_pair=g_H^2[k_R^2+(s_cubic k_S)^2],                 (4)

g_H=1.145614144998,
s_cubic=0.639292608019.
```

Both branch weights in (4) are unnormalized and each is charged once.

## Numerical method

The transverse generator uses the monotone finite-state scheme from the
branch-resolvent pilot. Circle crossings use the exact coordinate-line
distance in a Shortley-Weller stencil instead of absorbing at the center of
an interior disk node. The return flux is propagated with positivity-
preserving backward-Euler semigroup steps to time 30. As an internal check,
the same time sum with unit weight recovers the direct unweighted resolvent
to linear-solve precision.

The audit includes:

1. three base meshes over all five `rho` values;
2. fitted-boundary refinements at 60 and 80 strip intervals;
3. half-step temporal refinements at both affine endpoints;
4. artificial `x` widths `3.15,4.2,5.25`;
5. staircase-versus-fitted circle comparisons;
6. discrete principal killed-rate estimates.

## Pilot result

Unlike the raw all-`z` stress, the complete scalar branches complement each
other by orientation. Axis entries have larger `k_R` and smaller `k_S`; wall
entries have the reverse. Across the sampled static family `0<=rho<=1`, all
64 entry angles close (4). The worst row is near `rho=0`, with criterion
approximately `0.65`; the criterion decreases across the sampled family.

This positive number depends essentially on the true-split payment. Replacing
`s_cubic` by one, as required for a merely geometric same-level wall exit,
makes the sampled scalar criterion exceed one near the wall. The companion
compatibility audit proves that wall hitting alone does not trigger the
global envelope condition that earns `s_cubic`.

The fitted killed rates are approximately `2.34` at `rho=0` and `2.00` at
`rho=1`, comfortably above the wall-moment exponent one. These are
converging numerical values, not certified spectral enclosures.

## What this does not prove

The scalar calculation is a meaningful positive gate, but two essential
facts remain absent:

1. a geometric hit of `|y|=2.1` has not been proved to coincide with an
   actual dyadic amplitude/level split in the Navier-Stokes construction;
2. constant-payoff gains do not control the perturbative H1 response, which
   still needs physical space-time `L2` density bounds for both boundary
   kernels.

The model is also static, covers only `0<=rho<=1`, and assumes a fixed
eigenframe. Thus it is not a Navier-Stokes regularity result. The calculation
and convergence diagnostics are reproduced by
`scripts/neutral_strip_axial_patch_branch_pilot.py`.
