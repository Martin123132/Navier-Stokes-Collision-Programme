# Shrinking-tube return and renewal gate

## Purpose

The monotone-envelope tube removes the adverse scale term during a coherent
core visit. A full stochastic history is not killed at core exit, however;
it can travel through the exterior and return. This note derives the exact
return barrier for a moving inner radius and records what shrinkage does and
does not solve.

## Moving Newtonian barrier

Let the inner radius be `L(t)`, let

```text
ell=L'/L,
h(t,r)=(L(t)/r)^beta,
```

and consider an exterior weighted generator in dimension `d`,

```text
G=partial_t+nu Delta+b dot grad+c.
```

Writing `b dot x` for the radial drift numerator gives the exact identity

```text
(G h)/h
 =beta ell
  +beta[nu(beta-d+2)-b dot x]/r^2
  +c.                                                     (1)
```

In three dimensions this becomes

```text
(G h)/h
 =beta ell
  +beta[nu(beta-1)-b dot x]/r^2
  +c.                                                     (2)
```

Thus `h` is a Feynman-Kac supersolution whenever

```text
c r^2
 <=beta[b dot x+nu(1-beta)-ell r^2].                     (3)
```

For a start on the moving outer surface `r=eta L(t)`, `h=eta^(-beta)`.
At a later hit of the moving inner surface, `h=1`. Optional stopping then
gives the one-history weighted return bound

```text
||R_one||<=eta^(-beta),
```

and two independent histories give `eta^(-2 beta)`. These factors are
uniform in the absolute value of `L`.

## Shrinkage is favorable

For the monotone envelope

```text
L^2=R_* nu/A,
ell=-A'/(2A)<=0.
```

The new term `beta ell` in (1) is nonpositive. In (3), shrinkage supplies the
additional positive deformation budget

```text
-beta ell=beta A'/(2A).
```

For pure three-dimensional Brownian motion and `beta=1`, all spatial terms
cancel and

```text
(G h)/h=c-A'/(2A).
```

Hence `c<=A'/(2A)` is sufficient for the weighted return bound `eta^(-1)`.
If the one-history deformation is the full ideal value `c=2a`, this channel
alone requires

```text
A'>=4aA.                                                  (4)
```

A running maximum does not generally satisfy (4). Shrinking therefore helps
the exterior problem but does not make arbitrary stretching harmless.

## Brownian capacity budget

With a static scale, zero radial drift, and `0<beta<1`, condition (3) permits
the dimensionless shell deformation

```text
c r^2/nu<=beta(1-beta).
```

The right side is maximized at `beta=1/2`, with value `1/4`. The price is a
weaker return contraction: one history gives `eta^(-1/2)` and two histories
give `eta^(-1)` rather than `eta^(-2)`.

This quantifies a genuine tradeoff. Taking `beta` near one maximizes geometric
return contraction but leaves little static capacity to pay deformation;
taking `beta=1/2` maximizes the deformation allowance but weakens renewal.
Shrinkage and outward radial drift add to this budget.

## Conditional renewal lemma

Let `V` be one complete buffered visit and `R` the weighted exterior return.
The adaptive core calculation gives a contractive ideal visit and no adverse
amplitude/scale error for `R_*<=2`. If the remaining coherence and shell
errors leave `||V||<=v`, while (3) holds with some `beta>0`, then

```text
||R||<=eta^(-2 beta)
```

for two histories. The complete visit series converges whenever

```text
v eta^(-2 beta)<1,
```

with bound

```text
v/[1-v eta^(-2 beta)].
```

This closure is uniform as the envelope radius shrinks. It is a theorem for
the stated model inequalities, not yet for Navier-Stokes.

## Remaining obstruction

The scalar amplitude and moving radius are no longer the unexplained pieces.
The live exterior obligation is to prove (3) from Navier-Stokes geometry.
More specifically, one needs a useful joint estimate on

```text
c r^2-beta b dot x
```

along replica excursions. The earlier affine return counterexample shows
that finite energy alone cannot provide it. Pressure-collision cancellation,
eigenframe persistence, or a signed trajectory average must enter here.

There is also a selection problem: the centre path, finite axial extent, and
buffer geometry must remain coherent as `L` shrinks. Failure in any of these
terms can defeat the model barrier even though the pure scale contribution is
favorable.

The moving-barrier identity, capacity optimization, and renewal arithmetic
are reproduced by `scripts/shrinking_tube_renewal_audit.py`.
