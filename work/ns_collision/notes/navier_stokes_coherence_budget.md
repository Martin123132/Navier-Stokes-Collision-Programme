# Explicit Navier-Stokes coherence budget

## Purpose

The cubic partition and level transfer reduce the unresolved local dynamics
to

```text
||q_+||_(3/2)/nu<0.2159003229.                         (1)
```

This note writes `q` in actual Navier-Stokes variables on the optimized
support, computes explicit scale-invariant sufficient conditions, and checks
Galilean invariance. The calculation identifies two obligations that cannot
be hidden inside the word coherence: local translation transport and affine
gradient/eigenframe control.

## Effective error

In a frame centered at `c(t)`, use the axisymmetric reference backward drift

```text
b_0(r)=B_0 r,       B_0=diag(a,a,-2a),
aL^2/nu=R_*=1/2.                                      (2)
```

Write the actual backward drift as

```text
b=c'(t)+b_0+e.                                         (3)
```

Both `b` and `b_0` are divergence-free. If `delta_s` is the excess of the
actual one-history stretching potential over the reference value `2a`, the
gauged form error is exactly

```text
q=delta_s-b_0 dot e/(2nu).                             (4)
```

The complete cubic support, in units of `L`, is

```text
Omega_hat=[-rho_s/sqrt(2),rho_s/sqrt(2)]^2
          x[-3/2,3/2],       rho_s=1.91.               (5)
```

Its volume is `6 rho_s^2=21.8886`, and

```text
sup_(Omega_hat)|B_0 r|/(aL)=sqrt(rho_s^2+9)
                           =3.556416736.                (6)
```

## Critical Campanato conditions

Define the scale-invariant quantities

```text
D=||[delta_s]_+||_(3/2)/nu,
E=||e||_(3/2)/(nu L).                                  (7)
```

Holder, (4), and (6) give the explicit sufficient condition

```text
D+0.889104183996 E<0.215900322919.                     (8)
```

If `D=0`, this permits only `E<0.2428290484`.

Choose the cell velocity `c'` so that `e` has zero mean. Applying an
elementary coordinatewise Poincare estimate on (5) gives

```text
||e||_(3/2)<=4.86345554596 L ||grad e||_(3/2).         (9)
```

Thus, for

```text
G=||grad b-B_0||_(3/2)/nu,                             (10)
```

a sufficient critical gradient condition is

```text
D+4.32411867460 G<0.215900322919.                      (11)
```

With no stretching excess, (11) requires `G<0.04992932414`. This is a
genuine scale-invariant affine-oscillation requirement. Finiteness of the
critical norm is not enough.

## Leray-level conversion

The sharp rectangular `L^2` Poincare constant is `3L/pi`, because the axial
side is the longest side. An exact polynomial integration gives

```text
[integral_(Omega_hat)(x^2+y^2+4z^2)^3]^(1/6)
 =3.99609415220.                                       (12)
```

Define

```text
F=L^(1/2)||[delta_s]_+||_2/nu,
H=L^(1/2)||grad b-B_0||_2/nu.                          (13)
```

Using `L^2 * L^6 -> L^(3/2)` Holder for the drift pairing gives

```text
1.67251362402 F+0.953997206074 H<0.215900322919.       (14)
```

The one-term thresholds are

```text
F<0.1290873329,       H<0.2263112738.                  (15)
```

If the two terms receive equal shares of the budget, the thresholds halve.
Equation (14) is useful because `F` and `H` are local scale-invariant
versions of quantities appearing in Leray energy estimates. Ordinary global
energy and dissipation do not make them uniformly small on every support at
every time. They may, however, bound the number or occupation of bad supports;
that is a possible probabilistic route not captured by a pointwise theorem.

## Galilean test

Suppose the physical drift differs from the reference core only by a constant
velocity `U`. If the cell is held fixed, (4) creates the artificial term

```text
q_U=-b_0 dot U/(2nu).                                  (16)
```

Writing `M=|U|L/nu`, its positive critical norms are

```text
Q_U/nu=0.9035600729 M   for transverse U,
Q_U/nu=2.0070579730 M   for axial U.                   (17)
```

The axial term exhausts (1) already at `M=0.10757055`. But adding a constant
velocity is a Galilean transformation and cannot affect regularity. Therefore
a fixed-center estimate that charges (16) to `q_+` cannot be the final
architecture.

Choosing `c'=U` removes (16) exactly. The unresolved task is to implement
that subtraction for many local cells through a conservative moving-label or
visit transfer while retaining `sum phi=1`. This is separate from the cubic
scale-halving transfer already proved.

Rigid rotation around the cylinder axis passes the same invariance check:
`b_0 dot (-omega y,omega x,0)=0`, so it creates no radial-gauge potential.

## Pressure and affine geometry

Pressure does not occur explicitly in the instantaneous identity (4). Its
trace-free Hessian drives the time evolution of the maximal strain,
eigenframe, and best local affine matrix, and therefore determines whether
`D`, `G`, or `F`, `H` remain small through a visit.

There are now two mathematically distinct routes:

1. prove (11) or (14) near every dangerous support using the strain and
   pressure evolution plus viscous frame coherence;
2. extend the finite-cylinder/Poisson certificate uniformly from the
   axisymmetric `B_0` to each locally fitted trace-free affine matrix, leaving
   only its smaller nonlinear remainder in `G` or `H`.

Neither follows from the standard Leray inequality alone. The second route
is attractive because it removes constant gradient mismatch instead of
trying to prove that all dangerous strain is nearly axisymmetric.

The symbolic effective error, support integrals, Poincare conversions,
Galilean stress test, and all numerical constants are reproduced by
`scripts/navier_stokes_coherence_budget_audit.py`.
