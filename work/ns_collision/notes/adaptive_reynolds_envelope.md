# Monotone-envelope adaptive strain tube

## Motivation

The pressure-shell audit identifies

```text
R_local=aL^2/nu
```

as the obstruction to fixed-scale absorption. Choosing the instantaneous
diffusion radius `L^2=R_* nu/a` caps this number, but the radius expands every
time `a` decreases. In the moving-tube gauge, that expansion is an adverse
potential and can recreate the term one hoped to remove.

A monotone amplitude envelope avoids this problem and produces an exact sign
lemma in the ideal transverse strain tube.

## Envelope and scale

Let `a(t)>=0` be the actual affine tube rate and choose

```text
A(t)=max(A_0,sup_(tau<=t) a(tau)),
L(t)^2=R_* nu/A(t),
0<R_*<=2.
```

Then `A>=a`, `A'>=0`, and

```text
ell=L'/L=-A'/(2A)<=0.
```

The tube can shrink but never expands. Its actual local Reynolds number is

```text
aL^2/nu=R_* a/A<=R_*.
```

For a smooth pre-singular solution, a running maximum of an absolutely
continuous `a` is again absolutely continuous. A smooth monotone majorant can
be used if a classical gauge is preferred.

## Exact moving-gauge sign

Use the reference operator with drift `A y`, stretching potential `2A`, and
gauge Reynolds number

```text
R=A L^2/nu=R_*.
```

The actual mapped affine drift is `a y-ell y`. Relative to the reference,

```text
e=[(a-A)-ell]y,
delta_s=2(a-A).
```

The moving-tube effective error is

```text
q=delta_s-1/2 div(e)-(R_*/2)y dot e+(R_*'/4)|y|^2.
```

Since `R_*'=0` and the transverse dimension is two, this factors exactly as

```text
q=(a-A)(1-R_*|y|^2/2)
  +ell(1+R_*|y|^2/2).                                  (1)
```

Inside the unit disk, `|y|<=1`. If `R_*<=2`, both parenthetical coefficients
are nonnegative. Since `a-A<=0` and `ell<=0`,

```text
q<=0                                                       (2)
```

pointwise. Thus neither a falling strain rate nor the adaptive scale motion
consumes any of the ideal killed spectral margin. For two independent
histories, the two copies of `q` remain nonpositive.

This cancellation is lost if the reference gauge follows `a` instant by
instant. Keeping `R=aL^2/nu` constant then gives

```text
q_inst=-(a'/(2a))(1+R|y|^2/2),
```

which is positive whenever `a` decreases and the tube expands.

## Direct cutoff sign

The same monotonicity is visible without the gauge. For a radial decreasing
cutoff `phi(y)` and `y=(x-c)/L(t)`, scale motion contributes

```text
partial_t phi=-ell y dot grad(phi).
```

Here `y dot grad(phi)<=0` and `ell<=0`, so `partial_t phi<=0`. A shrinking
cutoff removes localized mass instead of creating an adverse source. Centre
motion still has to be treated separately; radial rotation remains
energy-neutral.

## Spectral and pressure margins at the gate

At `R_*=2`, the existing exact Kummer calculation gives

```text
lambda_1/A=4,
single-history margin (lambda_1-2A)/A=2,
two-history margin/A=4.
```

Smaller `R_*` gives a larger escape margin. At the same time, the pressure
shell scaling ratio obeys

```text
aL^2/nu<=R_*.
```

Therefore the monotone envelope simultaneously keeps the ideal killed
operator in its robust Reynolds regime and removes the amplitude growth from
the shell pressure/dissipation ratio. This is a scaling statement, not yet a
pressure estimate: the constants, pressure tails, and non-affine source must
still be controlled.

## What this changes

The previous moderate/high-Reynolds split can be sharpened. For the modeled
affine amplitude there need be no high-Reynolds visit: the envelope scale
caps every visit at `R_*<=2` without paying a time-expansion error.

The unresolved bad class is now geometric rather than purely dimensional:

1. non-affine drift and stretching errors may violate the critical form
   budget even at small `L`;
2. the pressure collision defect has nonlocal tails not bounded by local
   scaling alone;
3. centre tracking and changing eigenframes add errors not present in (1);
4. a shrinking core increases the burden on the exterior weighted-renewal
   operator;
5. all candidate cores must be selected and summed without assuming the
   regularity one is trying to prove.

The next proof gate is therefore a shrinking-core renewal estimate. It must
couple the favourable sign (2) to an exterior return/deformation bound whose
constant is uniform as `L` decreases. If that fails, the failure should be
traceable to centre motion, coherence loss, pressure tails, or return
amplification rather than to the scalar strain amplitude.

The factorization, cutoff sign, scale history, and `R_*=2` spectral constants
are reproduced by `scripts/adaptive_reynolds_envelope_audit.py`.

The exterior side of the construction is continued in
`shrinking_tube_renewal.md`. Monotone shrinkage also improves the Newtonian
return barrier and yields a scale-uniform conditional renewal lemma, but it
does not by itself control arbitrary exterior deformation.
