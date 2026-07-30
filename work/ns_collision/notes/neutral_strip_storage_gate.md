# Neutral-strip storage gate

## Purpose

The finite axial patch cancels deformation for the axisymmetric affine
return model, but the anisotropic endpoint

```text
b_0=(-x,0,z)
```

has a neutral transverse direction and no exterior spectral gap. This note
tests a stopped storage geometry that turns excursions in that neutral
direction into a separate outer-exit branch.

## Stopped strip

For

```text
b_rho=(-x,-rho y,(1+rho)z),       0<=rho<=1,
```

use the transverse domain

```text
Omega_Y={(x,y):x^2+y^2>1, |y|<Y}.                    (1)
```

Stop on either the inner circle `r=1` or the walls `|y|=Y`. The first event
is the same-scale return candidate; the second is the outer-exit candidate.
The full `r=2` entry circle is strictly inside the strip whenever `Y>2`.

Conjugating the transverse generator gives

```text
H_rho=(-partial_xx+x^2/4-1/2)
      +(-partial_yy+rho^2 y^2/4-rho/2).              (2)
```

The first oscillator is nonnegative. Slicing the killed domain at fixed `x`
and applying the Dirichlet Poincare inequality on intervals of length at
most `2Y` gives

```text
lambda_perp(rho,Y)>=pi^2/(4Y^2)-rho/2.               (3)
```

This remains valid where removal of the unit disk splits a slice into two
shorter intervals.

## Exact exponent balance

Axial Gaussian `L2` dilution leaves residual deformation growth

```text
(1-rho)/2.                                           (4)
```

Combining (3) and (4), the stopped weighted survival semigroup has net tail
margin at least

```text
lambda_perp-(1-rho)/2
 >=pi^2/(4Y^2)-1/2.                                  (5)
```

The dependence on `rho` cancels exactly. Hence the interval

```text
2<Y<pi/sqrt(2)=2.221441469...                         (6)
```

both contains every point of the `r=2` entry circle and gives a positive
uniform margin for `0<=rho<=1`. At the working value `Y=2.1`,

```text
margin>=25 pi^2/441-1/2=0.059501... .                (7)
```

Thus the neutral endpoint is no longer an operator-level tail obstruction
once neutral excursions are stopped at a nearby wall.

## What remains open

Equation (5) is a survival-semigroup estimate, not yet the required
space-time `L2` estimate for either boundary flux. Three identifications are
still necessary:

1. prove short-time and boundary-trace estimates for the inner-return and
   wall-exit kernels;
2. prove that a hit of `|y|=Y` is a genuine physical cubic scale split, not
   merely a geometric exit assigned a convenient name;
3. cover time-dependent eigenframes and the `t<0` half of the complete
   trace-free affine spectrum.

If these hold, the two unnormalized kernels can be inserted directly into

```text
a_S^2+a_R^2<1,                                       (8)
```

without conditioning or counting either branch mass twice. The exact
spectral calculation and scope checks are reproduced by
`scripts/neutral_strip_storage_gate.py`.
