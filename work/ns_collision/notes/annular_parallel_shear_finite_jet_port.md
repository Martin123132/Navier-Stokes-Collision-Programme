# Annular parallel-shear finite jet port

## 1. Result

Let

```text
u_N=H_N-a_y U_yz-a_x U_xy,
lambda_N=t Phi_+++,
```

where the two low waves and their common polarization are

```text
ell_yz=(0,1,-1),       Uhat_yz(+ell_yz)=-i r,
ell_xy=(1,-1,0),       Uhat_xy(+ell_xy)=+i r,
r=(1,1,1)/sqrt(3).
```

The signs give

```text
U_yz+U_xy
 =2r[sin(ell_yz.x)-sin(ell_xy.x)].
```

This stage ports the complete finite first and second `rho=0` generator
jets to that field. It establishes the following.

1. The old 4-channel first derivative and 20-channel second derivative
   formulas replay with the new high packet and both low modes.
2. Independent central differences, tenfold/twelvefold padding, and
   weight-scale homogeneity all pass.
3. A degree-five bivariate interpolation in `(a_y,a_x)` enumerates every
   pure and mixed low-polarization term.
4. The heat-weighted HHL identities hold component by component.
5. The total first jet retains a strictly negative `N^5` limit.
6. The complete inviscid-pressure second jet has a strictly negative
   optimized `N^9` limit.

The last statement is not yet a theorem for the complete second jet.
Viscosity-bearing quartic Fisher and mixed projector channels still need a
uniform `o(N^9)` bound.

## 2. Why the two-mode low field remains simple

For arbitrary amplitudes,

```text
U(a_y,a_x)
 =2r[a_y sin(ell_yz.x)-a_x sin(ell_xy.x)]
 =r f.
```

Both low waves are perpendicular to `r`, so

```text
r dot grad f=0,
(U dot grad)U=r f(r dot grad f)=0,
p[U,U]=0.                                         (2.1)
```

They also have the same squared frequency:

```text
|ell_yz|^2=|ell_xy|^2=2,
Delta U=-2U.                                      (2.2)
```

Thus every point in the two-amplitude low plane is Euler stationary and
the viscous evolution preserves that plane by one scalar heat factor.
This is stronger than stationarity only on the diagonal ray `a_y=a_x`.

## 3. Exact finite jet

With the inessential outer replica factor removed, the generator is

```text
g(u,lambda)
 =integral[
    p(u)u dot grad lambda
    -nu lambda|grad u|^2
    -nu lambda|grad lambda|^2].
```

The first flow directions are

```text
u_E=-P[(u dot grad)u],       u_nu=nu Delta u,
lambda_A=-u dot grad lambda, lambda_nu=-nu Delta lambda.
```

The coupled accelerations are

```text
u_2
 =-P[(u_1 dot grad)u+(u dot grad)u_1]+nu Delta u_1,

lambda_2
 =-u_1 dot grad lambda-u dot grad lambda_1
  -nu Delta lambda_1.
```

The audit evaluates

```text
g'
 =D_u g[u_E]+D_u g[u_nu]
  +D_lambda g[lambda_A]+D_lambda g[lambda_nu],    (3.1)

g''
 =D_uu g[u_1,u_1]
  +2D_u_lambda g[u_1,lambda_1]
  +D_lambda_lambda g[lambda_1,lambda_1]
  +D_u g[u_2]+D_lambda g[lambda_2].               (3.2)
```

At the `N=3` validation point

```text
(a_y,a_x,t,nu)=(0.7,0.7,0.9,1),
```

the values are

```text
g  =  -0.5518548312622972,
g' =  +6.436909359253194,
g''= -551.1354708504964.
```

The Richardson relative residuals are

```text
first derivative   1.12e-13,
second derivative  1.97e-11.
```

Changing the padding from ten to twelve times one-field support changes the
largest labelled subterm by `1.14e-13` and the total second derivative by
the same amount. Every subterm scales exactly as either `t` or `t^3`.

At `N=5`, `(a_y,a_x,t)=(0.6,0.6,0.8)`, the corresponding values are

```text
g  =   -0.3419900308294758,
g' =   +4.020253270478951,
g''= -1010.5817110571942.
```

These fixed-amplitude rows are formula checks. The static optimizer does not
yet give positive escape at these small carriers.

## 4. Complete mixed-polarization enumeration

Every labelled subterm is homogeneous in the velocity with a known degree
at most five. The audit evaluates the jets on the 21-point integer simplex

```text
(a_y,a_x),  a_y>=0, a_x>=0, a_y+a_x<=5,
```

inverts the exact rational monomial matrix, and checks the resulting
polynomial at two additional signed points.

At `N=5`,

```text
maximum node reconstruction residual       2.39e-10,
maximum off-node relative residual          5.49e-12,
maximum forbidden-support relative term     1.15e-11.
```

For a quartic velocity term, first-coordinate incidence permits only an
even number of high legs. The complete first inviscid-pressure polynomial
therefore has the following rows:

```text
HHHH:
  +0.008604599054229193

HHHL:
  zero to 1.06e-15

HHLL:
  +0.011133040887229828 a_x^2
  +0.002752748120771920 a_y a_x
  -0.025464315019688430 a_y^2

HLLL:
  zero to 8.44e-15

LLLL:
  zero to 4.30e-15.                               (4.1)
```

The `LLLL` cancellation is the finite polynomial form of (2.1). The mixed
`a_y a_x` term in `HHLL` is real and nonzero; treating the two shears as
independent copies would miss it.

The inviscid-pressure second jet is quintic. Incidence and stationarity
leave only four-high/one-low and two-high/three-low rows:

```text
HHHHL:
  -0.21535533158146028 a_y
  +0.02337269983878652 a_x

HHLLL:
  +0.02332046902419491 a_y^3
  +0.02699494216521026 a_y^2 a_x
  +0.00016895136508277 a_y a_x^2
  +0.00134301522530456 a_x^3.                    (4.2)
```

The forbidden `HHHHH`, `HHHLL`, and `HLLLL` rows are at numerical
roundoff. The low-only `LLLLL` row is below `1.06e-14`.

On the equal-amplitude ray, (4.2) gives

```text
c1_parallel,5=-0.19198263174267377,
c3_parallel,5=+0.05182737777979250.              (4.3)
```

The sign in (4.3) is only a finite replay. The large-`N` sign comes from
the previously proved square and tail theorems.

## 5. Heat-weighted HHL identities

For either low component, let `B_(m),N` be its pressure HHL load after
inserting

```text
(|k_1|^2+|k_2|^2+|ell|^2)^m
```

in every monomial. Linearity in the low Fourier field gives the exact
finite identities

```text
D_u g_pressure[nu Delta u]
 =nu t[a_y B_(1),N^yz+a_x B_(1),N^xy],           (5.1)

g''_(double heat,pressure)
 =-nu^2 t[a_y B_(2),N^yz+a_x B_(2),N^xy].        (5.2)
```

Both identities replay below `4.0e-13` on the finite jet rows, including
the pressure-high-high and pressure-cross pieces separately.

For the continuum profile

```text
b(xi)
 =S(xi)(x^2+y^2)/(x|xi|^3)(-z,0,x),
```

define

```text
beta_m
 =(10sqrt(3))^-1 integral_D
   (2|xi|^2)^m |b(xi)|^2 dxi.                    (5.3)
```

The same missing-component square used in the static theorem gives

```text
B_(m),N/N^(2m+1) ->-beta_m<0.                    (5.4)
```

Order-64 Gauss-Legendre quadrature, used for values rather than signs,
gives

```text
beta_0=0.0011545988274271606,
beta_1=0.014358876443453603,
beta_2=0.18231326692566646.                      (5.5)
```

The finite normalized values at `N=49` are

```text
B_(0),49/49   =-0.0012368087555808364,
B_(1),49/49^3=-0.015252155078257962,
B_(2),49/49^5=-0.19223908711796092.
```

They approach (5.5) from the same side.

## 6. First-jet asymptotic

The pressure-only static optimizer has

```text
a_N/N -> 8 beta_0/(9nu),
t_N/N ->128 beta_0/(45nu).                       (6.1)
```

Combining (5.1), (5.4), and (6.1) gives

```text
g'_N/N^5
 ->-1024 beta_0^2 beta_1/(405nu)
  =-4.839802238667534e-8/nu<0.                   (6.2)
```

The predecessor first-jet remainder proof ports at the level of carrier
powers. The modified multiplier is smooth on the same compact annulus,
has the same parity-gauged sine cutoff and fixed support geometry, and the
parallel low field has finite Fourier `l1` mass. Thus all remaining
first-jet rows remain `O(N^4)=o(N^5)`.

The repaired static witness therefore still initially moves in the wrong
direction at the exact `N^5` scale required by the reset deficit.

## 7. Negative inviscid-pressure N9 branch

The complete inviscid-pressure second jet has the exact amplitude form

```text
J''_inv,N(a,t)
 =t[c1_parallel,N a+c3_parallel,N a^3]            (7.1)
```

on the equal-amplitude ray.

The parallel-shear square and fourteen-profile tail theorem give

```text
c1_parallel,N/N^7
 ->-gamma,

gamma=(sqrt(3)/10)||v_y||_2^2>0,                 (7.2)

|c1_parallel,N-D_parallel,N|
 <=70,657,920 N^6.
```

The two-high/three-low branch obeys

```text
c3_parallel,N=O(N^3).                            (7.3)
```

Indeed, after the three fixed low leaves are inserted, momentum determines
the second high wave up to a bounded shift. There are `O(N^3)` choices,
the two high coefficients contribute `O(N^-2)`, and the compact inviscid
forms have at most two net high-frequency multipliers. This gives (7.3).

Using (6.1),

```text
J''_inv,N/N^9
 ->-[1024 beta_0^2/(405nu^2)] gamma<0.            (7.4)
```

The positive double-heat pressure curvature is subleading:

```text
g''_(double heat,pressure)/N^7
 ->1024 beta_0^2 beta_2/405
  =6.145050142888524e-7>0.                       (7.5)
```

Thus the favorable positive heat curvature cannot determine the large
carrier second jet inside the inviscid-pressure sector. The strict
four-high square gives a larger negative `N^9` route.

## 8. Remaining theorem gate

Equation (7.4) is not yet the complete second-jet limit. The exact channel
enumeration contains viscosity-bearing quartic terms such as

```text
H_uu[E,E] weighted Fisher,
D_u[u2_EE] weighted Fisher,
2H_u_lambda[E,A] weighted Fisher,
```

and mixed heat/transport pressure-projector rows. Finite values do not
exclude an `N^9` contribution.

The next theorem must prove that all such rows are `o(N^9)`. The intended
route is:

```text
1. parity-gauge the four high coefficients;
2. apply the six/five/four compatible Phi stencils;
3. split every internal Leray or pressure output into a finite shell and
   dyadic shells;
4. retain the projector singularity explicitly rather than differentiating
   through zero;
5. sum all mixed yz/xy low leaves before assigning a sign.
```

Only after that exclusion can (7.4) be promoted to the complete second jet
and used in a uniform Taylor-remainder argument.

Nothing here proves a finite parabolic-window inequality, critical `L^3`
control, finite-time blowup, or global regularity.

## 9. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_parallel_shear_finite_jet_port_audit.py
```

The production record is
`results/annular_parallel_shear_finite_jet_port_audit_v1.json`.
