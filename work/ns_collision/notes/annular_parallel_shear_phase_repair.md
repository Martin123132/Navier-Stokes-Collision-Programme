# Annular parallel-shear phase repair

## 1. Result

The two-shear self-flux obstruction has an exact polarization repair.

Let

```text
ell_yz=(0,1,-1),
ell_xy=(1,-1,0),
r=(1,1,1)/sqrt(3).
```

Choose the low Fourier polarizations

```text
d_yz=+r,
d_xy=-r.
```

The resulting real field is

```text
U_*(x)=2r[sin(ell_yz.x)-sin(ell_xy.x)].           (1.1)
```

Both wave vectors are perpendicular to `r`, so `U_*=r f` with
`r.grad f=0`. Consequently,

```text
(U_*.grad)U_*=0,
p[U_*,U_*]=0,
div[(|U_*|^2/2)U_*]=0.                            (1.2)
```

The pressure and nonlinear self-advection vanish pointwise. The complete
local-energy flux vector need not vanish, but its divergence does; hence its
integrated load against every periodic gradient vanishes, not merely its
load at the `+++` partition vertex.

The polarization change retains the strict annular signs:

```text
B_parallel,N/N
 ->-||b||_L2(D)^2/(10sqrt(3))<0,

c1_parallel,N/N^7
 ->-(sqrt(3)/10)||v_y||_2^2<0.                   (1.3)
```

It also restores the finite static optimizer and the order-`N^5`
reset-window gate. This is not a finite-time blow-up theorem or a solution
of the Navier-Stokes regularity problem.

## 2. Why scalar phases do not work

First keep the old normalized polarizations and vary only their sine and
cosine quadratures:

```text
Uhat_j(+ell_j)=(q_j-i p_j)d_j,
Uhat_j(-ell_j)=(q_j+i p_j)d_j.
```

Exact four-mode enumeration gives the common interaction polynomial

```text
P=p1^2 p2+p1 p2^2+p1 q2^2+p2 q1^2.              (2.1)
```

The low self-fluxes are

```text
L_kinetic =-(sqrt(2)/32)P,
L_pressure=-(sqrt(2)/96)P,
L_complete=-(sqrt(2)/24)P.                       (2.2)
```

The weighted Fisher form is

```text
M=(1/16)[
  8p1^2+p1p2+8p2^2
 +8q1^2-q1q2+8q2^2
].                                                (2.3)
```

The strict-square sign requires `p1>0` and `p2>0`. Every term in (2.1) is
then nonnegative and the first two terms are strictly positive. No relative
scalar phase cancels the self-flux in that quadrant.

Thus the repair genuinely uses polarization freedom; it is not a hidden
phase choice in the old family.

## 3. Full polarization factorization

The real divergence-free sine polarizations for the two waves have the
forms

```text
s_yz=(a,b,b),
s_xy=(c,c,d).
```

Retaining the strict-square diagonal ratio sets

```text
c=-b,       b>0,
```

while `a` and `d` control transverse off-diagonal entries. With zero cosine
quadrature, exact Fourier enumeration gives

```text
L_pressure
 =-(a-b)(b+d)(a-2b-d)/24,                         (3.1)

L_complete
 =(a+2b)(2b-d)(a-2b-d)/24,                       (3.2)

M=(4a^2+ab+17b^2-bd+4d^2)/8.                    (3.3)
```

Both fluxes vanish on the common algebraic branch

```text
a=d+2b.                                           (3.4)
```

The point

```text
a=b,       d=-b
```

lies on (3.4). At this point the two sine polarizations are `+r` and `-r`
after choosing `b=1/sqrt(3)`. This is exactly the parallel shear (1.1).

There is also a useful no-go inside the exact diagonal subfamily `a=d=0`.
Allow arbitrary divergence-free cosine polarizations

```text
c_yz=(A,B,B),
c_xy=(C,C,D).
```

Then

```text
L_pressure
 =-b[(A-B)^2+(C-D)^2+2b^2]/24<0                 (3.5)
```

for every `b>0`. Cosine phases cannot repair the exact diagonal family.
The transverse sine components in (3.4) are essential.

## 4. Exact low-field costs

At the selected common-polarization point, each Fourier direction is unit
length and the four nonzero low coefficients have total Parseval mass

```text
||U_*||_2^2=4.                                    (4.1)
```

The `+++` weighted Fisher form is

```text
mean[Phi_+++ |grad U_*|^2]=9/8.                  (4.2)
```

The old two-shear field had mass `17/16`. The repair therefore costs only
an additional `1/16` in weighted Fisher mass while removing both cubic
self-fluxes exactly.

The two low wave sums are still even, their squared radii remain two, and
their Fourier `l1` mass remains four. These facts matter for the inherited
finite-difference and tail ledgers.

## 5. Stencil matrix

Exact rational/surd stencil enumeration gives

```text
Q_*=
sqrt(3) *
[
 [ 1/60,    1/1080, -1/540 ],
 [ 1/1080, -1/30,    1/1080],
 [-1/540,   1/1080,  1/60  ]
].
                                                        (5.1)
```

The combined low strain is

```text
S_*=(1/sqrt(3)) *
[
 [-1,  1/2, -1 ],
 [1/2, 2,    1/2],
 [-1,  1/2, -1 ]
].
                                                        (5.2)
```

Their diagonal parts retain the exact relation

```text
diag Q_*=-diag S_*/20
         =(1/(20sqrt(3)))diag(1,-2,1).            (5.3)
```

The repair adds off-diagonal entries, so it remains to prove that the
modified high profile cannot see them.

## 6. Reflection symmetry

On the even frequency support `K=D union (-D)`, the modified high profile
obeys

```text
b(R_x xi)= R_x b(xi),
b(R_y xi)= R_y b(xi),
b(R_z xi)=-R_z b(xi).                             (6.1)
```

Its energy tensor is therefore diagonal.

Let `C` be the second Euler energy-tensor curvature. It is quartic in the
initial profile:

```text
C=v tensor v+gamma tensor b+b tensor gamma.
```

Hence `C(-b)=C(b)`. Euler equivariance and (6.1) imply

```text
C=R_j C R_j,       j=x,y,z.                       (6.2)
```

All off-diagonal entries of `C` vanish. This is an analytic symmetry
argument. The FFT rows only replay it; their largest off-diagonal residual
is below `1e-22`.

## 7. Strict signs

Write the high-profile energy tensor as

```text
E=diag(E_x,0,E_z).
```

The static pressure-HH limit is

```text
B_parallel,0
 =-2 Q_*:E
 =-(E_x+E_z)/(10sqrt(3))
 =-||b||_L2(D)^2/(10sqrt(3))<0.                  (7.1)
```

For the four-high coefficient,

```text
L_parallel=2 Q_*:C.
```

The off-diagonal entries do not contribute. Euler energy conservation and
the missing high component give

```text
C_x+C_y+C_z=0,
C_y=||v_y||_2^2>0.
```

Thus

```text
L_parallel
 =(C_x-2C_y+C_z)/(10sqrt(3))
 =-(sqrt(3)/10)||v_y||_2^2<0.                   (7.2)
```

Strict nonvanishing of `v_y` is the same covariance argument already
proved for the modified profile.

## 8. Finite replay

The complete low-only flux is zero, HLL cannot reach the partition
stencil, and the high-low Fisher cross term is support-excluded. Therefore
for every real amplitude `x`,

```text
L_complete(h_N+xU_*)=x B_complete,N,
L_pressure(h_N+xU_*)=x B_pressure,N,

E_+++(h_N+xU_*)=D_N+(9/8)x^2.                    (8.1)
```

The complete dictionary replay at `N=3` verifies (8.1) to roundoff.
Selected finite pressure rows are

```text
N       B_pressure,N/N
3      -0.003103082254644
9      -0.001654084303565
17     -0.001402983694755
25     -0.001319681755857
49     -0.001236808755581
```

They approach the continuum reference

```text
-0.001154598827427155.
```

The finite complete and pressure optimizers are both positive from the
audited size `N=25`.

## 9. Restored optimizer

Let `B_N` denote either the complete or pressure HHL load and let

```text
D_N=mean[Phi_+++ |grad h_N|^2],
Q(delta_+++)=75/256.
```

The exact objective is now

```text
J_N(a,t)
 =t[a|B_N|-nu(D_N+(9/8)a^2)]
  -(nu/16)(75/256)t^3.                            (9.1)
```

There is no low cubic. Exact optimization gives

```text
a_N=8|B_N|/(9nu),

A_N=4|B_N|^2/(9nu)-nu D_N,

t_N=sqrt[16A_N/(3nu(75/256))],

g_N=(2/3)A_N t_N.                                 (9.2)
```

Write

```text
B_N/N ->-beta_*,
beta_*=0.001154598827427155....
```

Then

```text
a_N/N ->8beta_*/(9nu),

A_N/N^2 ->4beta_*^2/(9nu),

t_N/N ->128beta_*/(45nu),

g_N/N^3 ->1024beta_*^3/(1215nu^2)>0.             (9.3)
```

The original `Theta(N)` amplitude and coefficient scales are restored,
with new exact constants.

## 10. Reset deficit

Because `||U_*||_2^2=4`, the norm-only reset bound is

```text
Delta_s
 >=(1/2)(
   sqrt(||h_N||_2^2+4a_N^2)-5t_N/16
  )_+^3.                                          (10.1)
```

Using (9.3),

```text
lim [2a_N-5t_N/16]/N
 =8beta_*/(9nu)>0.
```

Therefore

```text
liminf Delta_s/N^3
 >=256beta_*^3/(729nu^3),                         (10.2)

liminf Delta_s/(3g_N)
 >=5/(36nu).                                      (10.3)
```

On a heat window `delta_N=T/N^2`, any positive penalized restart
contribution must satisfy

```text
average g_0/g_N(s)
 >=[5/(36nu T)]N^2+o(N^2).                       (10.4)
```

Thus the order-`N^5` average-generator gate is restored, with a stronger
constant than in the old one-shear norm bound.

## 11. Full c1 tail

Every one of the fourteen structural `c1` profiles has exactly one low
leaf. The dominant functional and every tail row are therefore linear in
the low field.

The parallel repair has the same:

```text
low Fourier l1 mass=4,
low wave radii squared=2,
even low-wave parity,
high multiplier and difference bounds.
```

Hence the existing tail constant is unchanged:

```text
|c1_parallel,N-D_parallel,N|
 <=70,657,920 N^6.                                (11.1)
```

Combining (11.1) with (7.2) gives

```text
c1_parallel,N/N^7
 ->-(sqrt(3)/10)||v_y||_2^2<0.                   (11.2)
```

## 12. Route decision

Adopt the common-polarization parallel shear as the canonical low field.
It simultaneously provides:

```text
pointwise low Euler stationarity,
zero complete and pressure self-flux,
strict static HHL sign,
strict complete c1 sign,
a finite static optimizer,
and a nonvacuous reset tax.
```

The next gate is the complete finite first- and second-jet port. The new
polarization changes mixed channels even though the low self-evolution
vanishes, so no old jet constant should be copied without enumeration.

The production record is

```text
results/annular_parallel_shear_phase_repair_audit_v1.json
```

and is reproduced by

```text
python work/ns_collision/scripts/annular_parallel_shear_phase_repair_audit.py
```
