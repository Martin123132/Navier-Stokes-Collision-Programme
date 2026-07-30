# Anisotropic affine exterior tail gate

## Purpose

Finite axial extent repairs the axisymmetric affine return stress test in the
spatial `L2` density norm. This note checks whether that cancellation is
uniform over the complete trace-free affine spectrum. It is not.

## Return-aligned spectrum

After normalizing the largest returning deformation rate to one and
permuting coordinates, the static diagonal family is

```text
b_rho=(-x,-rho y,(1+rho)z),       0<=rho<=1.          (1)
```

The drift is trace-free. The radial deformation direction contributes
`exp(t)`. The outward axial OU Gaussian has `L2` decay

```text
exp[-(1+rho)t/2].                                    (2)
```

After axial compensation, the transverse first-hit density must therefore
pay the residual exponential rate

```text
exp[(1-rho)t/2].                                     (3)
```

If `lambda_perp(rho)` denotes the transverse killed-return rate, the desired
weighted `L2` tail requires

```text
lambda_perp(rho)>(1-rho)/2.                          (4)
```

At `rho=1`, (3) is neutral and the exact axisymmetric Tricomi calculation
works. The opposite endpoint is structurally different.

## Neutral-direction Weyl sequence

At `rho=0`,

```text
L_perp=Delta-x partial_x
```

on the exterior of the unit disk. Conjugation by the one-dimensional OU
Gaussian gives

```text
H_0=(-partial_xx+x^2/4-1/2)-partial_yy.              (5)
```

Let `g_0(x)` be the normalized zero-energy oscillator ground state. For
`R>1`, set

```text
psi_R(x,y)=g_0(x)sqrt(2/R)sin[pi(y-R)/R],
R<y<2R.                                               (6)
```

Its support misses the unit disk. The `x` contribution to the Rayleigh
quotient is zero and the `y` contribution is exactly

```text
<psi_R,H_0 psi_R>=pi^2/R^2 ->0.                      (7)
```

Since `H_0>=0`, its spectral bottom is exactly zero. There is no positive
transverse operator decay to pay the required rate `1/2` in (4).

This is an operator-level obstruction. Positivity and recurrence strongly
suggest the fixed outer-start kernel has the corresponding slow tail, but a
complete pointwise lower asymptotic is not proved here. The exact conclusion
is that no uniform all-entry spectral envelope over the full affine family
can be obtained from the present cylinder geometry.

## Consequence

The two affine endpoints now say different things:

1. `rho=1`: axial spreading exactly cancels deformation and the transverse
   killed OU tail is exponentially summable;
2. `rho=0`: axial spreading cancels only half the deformation, while a
   neutral transverse direction removes the spectral gap.

Therefore the axisymmetric calculation is a real positive mechanism but not
a comparison theorem for all affine spectra. A viable continuation needs at
least one new ingredient:

1. a storage or return geometry finite in every neutral direction;
2. a branch-dependent orientation/shape that converts neutral spreading into
   density dilution without losing the visit estimate;
3. a Navier-Stokes dynamical theorem excluding persistent return branches
   near the neutral endpoint;
4. a norm or coupled two-history estimate that uses more incompressibility
   than the current one-history surface `L2` bound.

The trace-free exponent balance and explicit Weyl quotients are reproduced
by `scripts/anisotropic_affine_exterior_tail_gate.py`.

The first exact geometry-level repair is developed in
`neutral_strip_storage_gate.md`: stopping the neutral coordinate at
`|y|=Y`, with `2<Y<pi/sqrt(2)`, restores a positive survival-semigroup tail
margin. Identifying that wall exit with a physical true split remains open.
