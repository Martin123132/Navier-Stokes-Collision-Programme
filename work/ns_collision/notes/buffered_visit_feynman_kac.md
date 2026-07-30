# Exact buffered-visit Feynman-Kac benchmark

## Purpose

The two-norm generation algebra left the complete buffered visit norm as an
input. Replacing it by a bare gauge condition number is too optimistic. This
note solves an ideal core-plus-shell visit exactly and reveals a logarithmic
transverse recurrence obstruction.

This corrects the provisional suggestion that `R_*` near one and buffer ratio
two might close without positive visit action.

## Piecewise radial model

Use a two-dimensional transverse radius `rho=r/L`. Inside the coherent core
`0<rho<1`, take the ideal backward generator and one-history stretching

```text
G_in=Delta+R_* rho partial_rho+2R_*.
```

In the transition shell `1<rho<eta`, use pure two-dimensional Brownian motion
with no stretching:

```text
G_shell=Delta.
```

This is deliberately favorable in the shell: positive shell stretching or
inward drift can only worsen the visit gain.

Let `u(rho)` be the one-history Feynman-Kac weight accumulated until the outer
surface `rho=eta`, where `u(eta)=1`. Regularity at zero and matching value and
derivative at `rho=1` give

```text
u_in(rho)=U exp[R_*(1-rho^2)/2],
u_shell(rho)=U[1-R_* log rho].                           (1)
```

The outer boundary condition yields the exact inner-boundary gain

```text
U=1/[1-R_* log eta].                                    (2)
```

The audit verifies both PDEs and both interface conditions symbolically.

## Moment threshold

The positive Feynman-Kac moment is finite only when

```text
R_* log eta<1.                                          (3)
```

At equality the repeated transverse returns to the core overwhelm outer
escape. This is not boundary chatter: the buffer is present, but the
two-dimensional shell retains logarithmic recurrence.

For independent replicas, the complete visit gain from the inner boundary is

```text
V=U^2=1/[1-R_* log eta]^2.                              (4)
```

This can be far larger than the pair gauge condition number used in the
provisional two-norm sweep.

## Renewal and generation closure

Keep the three-dimensional exterior pair return

```text
r=eta^(-2 beta)
```

and the true dyadic generation factor

```text
gamma=(1/4)exp(R_* d/24),       d=3.
```

The same-scale renewal converges only if `rV<1`. After summing those visits,
one complete generation closes precisely when

```text
V(gamma+r)<1.                                           (5)
```

For `beta=1`, `eta=2`, same-generation renewal alone requires

```text
R_*<(1-1/2)/log 2 approximately 0.72135.
```

Condition (5) is stricter. The audit solves its threshold numerically and
finds it between `0.35` and `0.45`. In particular:

```text
R_*=0.25: complete ideal generation closes,
R_*=0.5:  complete ideal generation fails,
R_*=1:    no tested or optimized buffer closes.
```

## Interpretation

The failure is useful. It locates the missing mechanism more precisely than a
generic demand for positive visit action. A purely transverse finite-width
tube repeatedly revisits the stretching core before reaching the outer
buffer. Even a shell with zero stretching does not suppress that recurrence
enough at moderate `R_*`.

At least one genuinely three-dimensional ingredient must improve the visit:

1. finite axial killing, already known to increase the principal escape rate;
2. a three-dimensional shell rather than a transverse annulus;
3. favorable outward shell drift;
4. collision or angular damping retained during core returns;
5. an occupation estimate grouping all core returns before applying the
   deformation weight.

The first candidate should be finite axial killing because it is already part
of the intended finite three-dimensional core and introduces no new physical
assumption.

## Correction to the provisional cycle

The algebra in `two_norm_generation_cycle.md` remains valid for a supplied
visit action `D`, but its zero-action `R_*=1`, `eta=2` row must not be read as
an estimate of the actual buffered visit. Equation (4) supplies a stricter
benchmark and overturns that provisional model recommendation.

The piecewise PDE, exact gain, parameter sweep, and optimized buffers are
reproduced by `scripts/buffered_visit_feynman_kac_audit.py`.

The first proposed repair, finite axial escape, is quantified in
`axial_killing_buffered_visit.md`. Including the actual inward axial OU drift
shows that the core must be genuinely finite; a long tube receives far less
axial killing than the Brownian eigenvalue suggests.
