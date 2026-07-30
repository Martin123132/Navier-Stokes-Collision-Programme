# Annular rho-zero Euler-Maclaurin and Leray-cusp gate

## 1. Purpose

The complete four-high coefficient has already been reduced to

```text
c_1,N/N^7 -> L_EE,

L_EE
 =(sqrt(2)/20) integral (v_z^2-v_y^2)
 +(sqrt(2)/10) integral (g_y a_y-g_z a_z).
```

The remaining theorem-level gate is the sign of `L_EE`. This stage
replaces the earlier shifted-lattice extrapolation with a direct
tensor-trapezoid rule on the exact continuum packet box, identifies its
leading Euler-Maclaurin error, and isolates the only nonsmooth continuum
mechanism left after the packet-face correction.

The numerical candidate remains

```text
L_EE approximately -2.99386e-7.
```

No value in this note is promoted to an interval certificate.

## 2. Direct exact-box rule

On

```text
D=[2,3] x [-1/2,1/2]^2,
```

write

```text
S=sin(pi(x-2)) cos(pi y) cos(pi z),

a=S*(-xz,-yz,x^2+y^2)/(x^2+y^2+z^2)^(3/2).
```

Extend `a` evenly to `-D` and by zero elsewhere. For even `N`, sample
the exact endpoints with mesh width `h=1/N`. The profile vanishes
exactly on every face.

The two Euler symbols supply two powers of the continuum wave number.
The three independent three-dimensional sums supply `h^9`. Therefore
the finite FFT quartic sum has normalization

```text
h^11=N^-11.
```

The code fields have signs

```text
V_code=-i v,
G_code=-g.
```

Consequently the second continuum term is

```text
(sqrt(2)/10) h^11 (pair_z-pair_y),
```

where `pair_j=sum Re(G_code,j conjugate(a_j))`.

The direct rows are

```text
N       L_VV                 L_GH                 L_EE
8    1.521058480684e-7   -4.179748720853e-7   -2.658690240169e-7
16   1.670756620523e-7   -4.577510002340e-7   -2.906753381817e-7
32   1.709383746387e-7   -4.681240667008e-7   -2.971856920622e-7
64   1.719143662902e-7   -4.707488255523e-7   -2.988344592621e-7
```

The successive direct differences shrink by factors

```text
3.8103, 3.9486,
```

which exposes the expected second-order boundary error. The final
second-order Richardson diagnostic is

```text
-2.993840483287e-7.
```

This is a diagnostic, not an error bar.

## 3. Disjoint output bands

The two cancelling terms do not cancel pointwise.

The velocity energy is supported on `K+K`, hence

```text
|rho_x|<=1
or
4<=|rho_x|<=6.
```

The acceleration-profile pairing is supported on `K`, hence

```text
2<=|rho_x|<=3.
```

Thus the outer Leray projector in `L_GH` is uniformly smooth. The
projector at zero output occurs only in the mixed-sign velocity band.
This separation permits distinct error estimates for the two terms.

## 4. Explicit h^2 face measure

For a one-dimensional factor that vanishes at both endpoints,

```text
h sum f(kh)
 =integral f
  +h^2/12 [f'(b)-f'(a)]
  +O(h^4).
```

Tensoring the rule gives

```text
mu_h=mu+h^2 mu_2+O(h^4),

mu_2
 =(1/12) sum_faces orientation*partial_normal(a) dS.
```

Since `L_EE` is quartic in the packet measure, its leading coefficient
is the explicit directional derivative

```text
c_2=D L_EE[a](mu_2).
```

The finite face evaluations give

```text
N       c_2(N)              L_N-c_2(N)/N^2
8    1.984670391674e-6   -2.968794988868e-7
16   2.189131936093e-6   -2.992266348070e-7
32   2.242916357724e-6   -2.993760400678e-7
64   2.256532202164e-6   -2.993853704443e-7
```

The corrected differences shrink by factors `15.71` and `16.01`.
The differentiated Euler energy trace closes at roundoff in every row.
This identifies the observed `h^2` term structurally; it is not inferred
from a regression.

Evaluating the full quartic functional on

```text
mu_h_star=mu_h-h^2 mu_2,h
```

also converges at fourth order. Its `N=64` value is

```text
-2.993858424597e-7.
```

## 5. Higher packet-face correction

The next one-dimensional Euler-Maclaurin factor is

```text
C_j
 =-(1/720) sum_j_faces orientation*partial_j^3(a).
```

Tensoring

```text
S_j-h^2 A_j-h^4 C_j
```

also supplies the positive edge measures

```text
A_j A_k
 =(1/144) orientation_j orientation_k
   partial_j partial_k(a).
```

The corresponding corrected quartic rows are

```text
N       corrected quartic value
8      -2.990784108622e-7
16     -2.993723256764e-7
32     -2.993852780437e-7
64     -2.993859498977e-7
```

The rule is sixth order for an ordinary smooth linear packet integral.
For example, its successive errors for `integral a_z` shrink by a
factor close to `64`. The quartic functional retains a smaller
fourth-order term. This is explained by an internal Leray cusp, not by
an omitted packet-face term.

## 6. The internal Leray cusp

For the mixed-sign output, define the unprojected convolution matrix

```text
A(rho)
 =integral a(xi) tensor a(rho-xi) dxi.
```

The zero extension of `a` is Lipschitz and compactly supported, hence
each component belongs to `H^1`. Distributionally,

```text
partial_m partial_n A_jl
 =(partial_m a_l)*(partial_n a_j).
```

The right side is bounded and continuous by the `L^2*L^2 -> L^infinity`
convolution estimate. Thus `A` has a bounded Hessian. It is also even,
so its first derivative vanishes at zero. Taylor's formula now gives

```text
A(rho)=M+O(|rho|^2),

M=integral a tensor a,

v(rho)=P_rho M rho+O(|rho|^3).                   (6.1)
```

Parity makes `M` diagonal. The `N=64` direct covariance replay is

```text
M_xx=2.086463114127e-4,
M_yy=1.129052679769e-6,
M_zz=3.978555703143e-2,

max off-diagonal <=1.13e-21.
```

Across seven nonaxial directions,

```text
max |v(rho)-P_rho M rho|/|rho|^3
```

stabilizes from `0.0932` at `N=8` to `0.0981` at `N=64`.
Equation (6.1) therefore replays directly.

The homogeneous degree-one term `P_rho M rho` is continuous and
Lipschitz but direction dependent at first derivative order. It enters
`L_GH` linearly through the intermediate velocity. A codimension-three
degree-one cusp naturally leaves an `h^4` cubature defect after all
smooth packet-face terms have been removed.

## 7. Explicit small-cube budget

Let

```text
A_2=||a||_2^2.
```

Since

```text
|a|^2
 =S^2 (x^2+y^2)/(x^2+y^2+z^2)^2
 <=S^2/x^2,
```

and convexity gives

```text
1/x^2 <=(3-x)/4+(x-2)/9   on [2,3],
```

the separable sine integrals yield the exact elementary bound

```text
A_2<=13/288.
```

Young's inequality then gives

```text
|v(rho)|<=A_2 |rho|.
```

Take the internal cube

```text
C_delta={|rho_j|<=delta}, delta=1/20.
```

With `R=sqrt(19/2)` and `beta=sqrt(2)/10`, the part of `L_GH` in which
the intermediate velocity lies in `C_delta` is bounded by

```text
beta * 8 sqrt(3) R A_2^2 delta^4
 <=7.692e-8.
```

Removing the same velocity piece from `L_VV` changes it by at most

```text
24 (sqrt(2)/20) A_2^2 delta^5
 <=1.081e-9.
```

Thus the complete unknown small-cube contribution has the explicit
absolute bound

```text
7.800e-8.                                       (7.1)
```

This is already smaller than the observed negative margin.

## 8. Remaining certification obligation

The next audit must split the intermediate velocity into:

```text
the small cube C_delta, controlled by (7.1);
and its complement, where every internal projector is separated
from zero.
```

On the complement, apply the corrected tensor rule and produce an
expanded derivative ledger for the smooth remainder. At `N=64`, after
reserving the small-cube budget, it is enough to prove a regular-part
error below about `2.1e-7`. Equivalently, a sixth-order constant below
roughly `1.4e4` suffices. This is deliberately generous.

The floating FFT must also receive a directed roundoff enclosure.
Neither fit agreement nor agreement between corrected rules can replace
those two bounds.

This stage proves:

```text
the exact direct h^11 cubature and sign convention,
the disjoint output-band decomposition,
the explicit Euler-Maclaurin h^2 coefficient,
the internal Leray-cusp expansion,
and a closed elementary 7.800e-8 small-cube budget.
```

It does not prove:

```text
L_EE<0,
a nonzero optimized N^9 coefficient,
a parabolic-window Taylor estimate,
critical L^3 control,
finite-time blowup,
or global regularity.
```
