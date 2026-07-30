# Finite-height axial killing benchmark

## Purpose

The exact transverse buffered visit fails at moderate `R_*` because the
two-dimensional shell repeatedly returns to the stretching core. A finite
three-dimensional core can also exit through its axial ends. This note
quantifies how much axial escape is required and includes the adverse inward
axial drift of the ideal strain.

The first calculation is a separated-mode surrogate. It is not yet the full
finite-cylinder boundary operator.

## Constant axial killing in the radial problem

Add a dimensionless killing rate `zeta` to both the core and shell radial
equations. The regular core solution is proportional to

```text
M(1-zeta/(2R_*),1,-R_* rho^2/2),
```

and the shell basis is

```text
I_0(sqrt(zeta) rho), K_0(sqrt(zeta) rho).
```

Matching value and derivative at `rho=1` and imposing unit payoff at
`rho=eta` gives the exact one-mode visit gain. It decreases strictly with
`zeta` in every audited case and recovers

```text
1/[1-R_* log eta]
```

as `zeta` tends to zero.

For each `R_*`, the audit solves for the killing rate that makes the complete
generation criterion exactly one at `eta=2`, `beta=1`.

## Correct axial OU eigenvalue

The ideal backward strain drift is not Brownian in the axial direction. For

```text
S=diag(-a,-a,2a),
```

the backward axial drift is `-2az`, directed toward the centre. With
`h=H/L`, the dimensionless killed axial generator is

```text
partial_yy-2R_* y partial_y,       -h<y<h.
```

Its even principal eigenvalue `zeta` satisfies

```text
M(-zeta/(4R_*),1/2,R_* h^2)=0.                         (1)
```

The audit solves (1) directly. In every tested geometry,

```text
zeta_OU<pi^2/(4h^2).
```

Thus the usual Brownian axial eigenvalue overstates escape, severely for long
cores and large `R_*`.

## Geometric verdict

Axial killing can restore the model generation inequality, but only if the
core is genuinely finite. At `R_*=1`, the maximum permitted OU half-height is
less than twice the transverse radius. Long vortex tubes do not receive
enough axial killing because the inward drift traps histories near the middle.

At smaller `R_*`, a taller core is allowed. The audit records the required
`zeta`, optimistic Brownian aspect ratio, and corrected OU aspect ratio for
`R_*=0.25,0.5,1,2`.

This strengthens the earlier geometry distinction:

```text
finite three-dimensional concentration blob: potentially viable,
long transverse tube: recurrent buffered-visit obstruction persists.
```

## Limitation and next gate

Constant killing is exact for one separated axial mode. The actual cylinder
problem has unit payoff on the radial outer boundary and zero payoff at the
axial ends, so that boundary data excites the complete axial mode family.
The principal mode identifies the slowest escape and the right aspect-ratio
scale, but it is not by itself a pointwise upper bound for the full boundary
operator.

The next calculation should construct the complete Dirichlet axial expansion
with the inward OU eigenfunctions, propagate every mode through the radial
core-shell transfer, and bound the resulting operator in the physical norm.
Only then can finite axial killing be inserted into the renewal theorem.

The Kummer/Bessel transfer, axial OU roots, and aspect-ratio thresholds are
reproduced by `scripts/axial_killing_buffered_visit_audit.py`.

The complete boundary operator is evaluated in
`finite_cylinder_mode_expansion.md`. Higher axial modes tighten the maximum
allowed aspect ratios modestly, but the compact-core closure survives in the
ideal model.
