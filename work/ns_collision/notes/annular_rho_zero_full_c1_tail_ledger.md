# Annular rho-zero complete c1 tail ledger

## 1. Result

Let

```text
V=B(H,H),                 W=B(H,U),
G=B(H,V),
AH=C(H,Phi),              AU=C(U,Phi),
G1=-B(U,V)-2B(H,W),
L0=C(V,Phi)+C(H,AH),
L1=-2C(W,Phi)-C(U,AH)-C(H,AU).
```

The preceding fixed-output gate isolated

```text
D_N=-2T(V,V,U;Phi)-4T(G,H,U;Phi)
```

inside the complete four-high amplitude-one coefficient `c_1,N`. This
stage proves, for every odd `N>=5`,

```text
|c_1,N-D_N| <= 35,328,960 N^6.                  (1.1)
```

This is stronger than the checkpoint target

```text
|c_1,N-D_N| <= C N^6 log(2+N).
```

Combining (1.1) with the already proved bound

```text
|N^-7 D_N-L_EE| <= 250,000/N
```

gives, for odd `N>=128`,

```text
|c_1,N/N^7-L_EE| <= 35,578,960/N.               (1.2)
```

Therefore

```text
c_1,N/N^7 -> L_EE.                              (1.3)
```

No numerical fit is used in (1.1)-(1.3). The sign of `L_EE` is still
open because the continuum quadrature has not yet been converted into
an interval enclosure.

## 2. Exact amplitude-one expansion

Write

```text
u=H-aU,
E=V-2aW,
A=AH-aAU,
B(u,E)=G+aG1+O(a^2),
lambda2=L0+aL1+O(a^2).
```

The exact compact second-jet identity is

```text
6S(u,E,E;Phi)
+6S(u,u,E;A)
+6S(u,u,B(u,E);Phi)
+S(u,u,u;lambda2).
```

Its coefficient of `a` is

```text
-6S(V,V,U;Phi) -24S(V,W,H;Phi)

-6S(V,H,H;AU)
-12S(V,H,U;AH)
-12S(W,H,H;AH)

-12S(G,H,U;Phi) +6S(G1,H,H;Phi)

-3S(H,H,U;L0) +S(H,H,H;L1).                    (2.1)
```

Expanding the two dominant symmetrizations gives

```text
-6S(V,V,U;Phi)
 =-2T(V,V,U;Phi)-4T(V,U,V;Phi),

-12S(G,H,U;Phi)
 =-4T(G,H,U;Phi)
  -4T(G,U,H;Phi)
  -4T(H,U,G;Phi).                               (2.2)
```

The first term on each right-hand side is in `D_N`. The other three
permutations, together with the seven remaining forms in (2.1), are the
complete tail.

## 3. The only boundary regularity used

The high packet has

```text
#K_N=2N^3,
2N<=|k|<sqrt(19/2)N<4N,
|hhat_N(k)|<=1/(2N).
```

After removing its alternating parity, its coefficient is the product
of three sine cutoffs and

```text
m(k)=P_k(e_3)/|k|.
```

On the annulus,

```text
|grad m(k)|<=6/|k|^2<=3/(2N^2).
```

The one-dimensional sine difference is at most `pi/N`. Hence

```text
|Delta_j[parity(k) hhat_N(k)]|
 <=[(pi/2)+(3/2)]/N^2
 <4/N^2.                                        (3.1)
```

At a face of the zero-extended packet, the boundary sine is itself at
most `pi/N`, so (3.1) remains valid. This proof uses boundedness and one
Lipschitz difference only. It does not assert that the zero extension
is `C^2`, let alone `C^6`.

## 4. The exact vertex difference

For odd `N`, on either sign of the packet,

```text
alpha(k)=-(-1)^(k1+k2+k3).
```

A four-high resonance against one low shear mode and one vertex mode
must contain two positive-carrier and two negative-carrier waves.
Because the low wave `(0,+/-1,-/+1)` has even coordinate sum, the
product of the four packet parities is

```text
(-1)^(q1+q2+q3).                                 (4.1)
```

Thus the vertex coefficients become the tensor product of the signed
one-dimensional stencil

```text
(-1/4, 1/2, -1/4).                               (4.2)
```

There can be at most three scalar derivative factors containing `q`.
After expanding them, every coordinate monomial is `q^alpha` with
`|alpha|<=3`. In one dimension the signed stencils for powers
`0,1,2,3` are

```text
0: (-1/4, 1/2,-1/4),   sum 0,
1: ( 1/4,   0,-1/4),   sum 0,
2: (-1/4,   0,-1/4),   sum -1/2,
3: ( 1/4,   0,-1/4),   sum 0.
```

For every three-dimensional multiindex of total degree at most three,
some coordinate has exponent zero or odd. Its stencil therefore has
zero sum and factors one exact first difference. The residual tensor
stencil has `l1` norm at most `1/2`. Since the two low shear
coefficients have total `l1` norm `2`, the combined low/stencil factor
is at most one.

This is an exact finite-stencil statement, not an asymptotic smoothness
claim.

## 5. Why the two leading terms are exceptional

In

```text
T(x,y,z;phi)=mean[p[x,y] z dot grad phi],
```

the outer pressure multiplier is

```text
Q_r=r tensor r/|r|^2,    Q_0=0.
```

It has degree zero and is not continuous at zero. In the two terms

```text
T(V,V,U;Phi),   T(G,H,U;Phi),
```

the test side contains no high leaf. Changing the vertex mode changes
the bounded pressure output `r`, so a difference of `Q_r` gives no
factor `N^-1`. These are exactly the two terms retained in `D_N`.

Every tail permutation has at least one high leaf on the test side.
Choose that leaf as the dependent resonance variable. A unit change of
the vertex mode is absorbed by the chosen leaf while the outer pressure
output `r` is held fixed. Consequently only

```text
||Q_r||<=1
```

is used, at bounded and high output alike.

## 6. Internal Euler outputs need no shell split

Treat the projected Euler bilinear as one symbol:

```text
b_r(x,y)
 =-(i/2)P_r[(r dot x)y+(r dot y)x],    b_0=0.
```

It satisfies

```text
|b_r(x,y)|<=|r||x||y|.                            (6.1)
```

On the unit sphere its norm is at most one and its Lipschitz constant
is at most three. For arbitrary `r,s`,

```text
min(|r|,|s|)
 |r/|r|-s/|s|| <=2|r-s|.
```

The radial extension of the sphere symbol therefore obeys the global
bound

```text
|b_r(x,y)-b_s(x,y)|
 <=7|r-s||x||y|.                                 (6.2)
```

In particular (6.2) remains valid when one of the outputs is zero.
This is why an internal pressure/Leray shell logarithm is unnecessary:
the singular degree-zero projector is always multiplied by the Euler
output frequency, producing the globally Lipschitz degree-one symbol.

## 7. One atomic contraction

After expanding `B`, `C`, and the scalar derivatives, every atomic tail
contraction has:

```text
four high leaves,
one low shear leaf,
one vertex stencil,
at most two degree-one high-frequency symbols.
```

Every intermediate frequency is less than `20N`. Equations
(6.1)-(6.2) therefore give the deliberately coarse bounds

```text
|K_N|<=400N^2,
|Delta K_N|<=280N.                               (7.1)
```

Choosing three high waves determines the fourth, so there are at most

```text
(2N^3)^3=8N^9
```

resonant tuples. For one fixed coordinate monomial, (3.1) and (7.1)
give

```text
8*(1/8)*
[4*400+(1/2)*280] N^6
=1740N^6.                                        (7.2)
```

There are at most

```text
2^3 * 3^3 =216
```

terms after choosing the `q` or high-frequency part of each of three
linear factors and expanding their coordinates. The residual vertex
and low coefficient factor is at most one. Thus every atomic
contraction satisfies

```text
|atomic tail contraction|
 <=375,840N^6.                                   (7.3)
```

The constants are intentionally loose. Their role is to expose every
carrier power, not to give a useful finite-`N` error bar.

## 8. Complete ledger

The three nonleading permutations have absolute coefficient mass
`4+4+4=12`. Expanding `G1`, `L0`, and `L1`, the remaining forms give:

```text
expression                              absolute coefficient
-24S(V,W,H;Phi)                                  24
-6S(V,H,H;AU)                                     6
-12S(V,H,U;AH)                                   12
-12S(W,H,H;AH)                                   12
-6S(B(U,V),H,H;Phi)                               6
-12S(B(H,W),H,H;Phi)                             12
-3S(H,H,U;C(V,Phi))                               3
-3S(H,H,U;C(H,AH))                                3
-2S(H,H,H;C(W,Phi))                               2
-S(H,H,H;C(U,AH))                                 1
-S(H,H,H;C(H,AU))                                 1
```

Their absolute coefficient mass is `82`, and the complete tail mass is

```text
12+82=94.
```

Multiplying (7.3) by `94` gives

```text
94*375,840=35,328,960,
```

which proves (1.1).

## 9. Finite decomposition replay

The proof does not use finite scaling, but the stored branch rows give
an independent decomposition check:

```text
N    seven forms       three permutations   reconstructed tail
5    -0.0730617234      0.0017393860         -0.0713223374
9    -0.1755152604      0.0067401397         -0.1687751208
17   -0.4996713695      0.0276432072         -0.4720281623
25   -0.9835732022      0.0629261273         -0.9206470749
29   -1.2855101386      0.0859165785         -1.1995935601
```

Across all eight stored carriers

```text
N=5,7,9,13,17,21,25,29,
```

the reconstructed tail agrees with `c_1,N-D_N` to less than
`1e-10`.

## 10. Correct scope

This stage proves:

```text
the complete termwise tail ledger,
the explicit O(N^6) tail bound,
and c_1,N/N^7 -> L_EE.
```

It does not prove:

```text
L_EE<0 or L_EE!=0,
a nonzero optimized N^9 coefficient,
control of the complete viscous second jet,
a uniform Taylor remainder on a parabolic window,
critical L^3 control,
finite-time blowup,
or Navier-Stokes global regularity.
```

The next gate is a deterministic joint interval enclosure for the two
cancelling continuum integrals in `L_EE`.

## 11. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_rho_zero_full_c1_tail_ledger_audit.py
python -m pytest -q work/ns_collision/tests/test_annular_rho_zero_full_c1_tail_ledger.py
```

The production record is
`results/annular_rho_zero_full_c1_tail_ledger_audit_v1.json`.
