# Annular eight-vertex heat-window gate

## 1. Result

The separable annular family defeats a static complete-HHL Fisher-Schur
estimate at the `+++` tensor vertex. This stage asks whether the exact
eight-cell partition identity or one parabolic interval removes that
obstruction.

The answer has three distinct parts.

1. Equal vertex weights cancel the complete load exactly, at every finite
   `N` and every time.
2. The load vector itself survives at order `N` in six of the seven
   nonconstant Walsh characters. Nonconstant compatible weights can retain
   it.
3. Heat damping preserves the local `N^4` load/Fisher loss over a fixed
   scaled time interval. A small-amplitude Navier-Stokes shadowing argument
   transfers this obstruction to universal homogeneous trajectory
   inequalities near the zero solution.

Thus global conservation is genuine, but it is not a free local absorption
theorem.

## 2. Eight tensor vertices

For `v in {-1,+1}^3`, write

```text
Phi_v(x)
 = product_(j=1)^3 (1+v_j cos x_j)/2.
```

Then

```text
sum_v Phi_v=1,
sum_v grad Phi_v=0.
```

For one common complete HHL flux `F_N`, define

```text
b_v(N)=mean[F_N dot grad Phi_v].
```

The partition identity immediately gives

```text
sum_v b_v(N)=0.                                    (2.1)
```

This is exact. It is not an estimate and does not require the annular
family.

For a nonempty subset `S` of `{1,2,3}`, let

```text
chi_S(v)=product_(j in S)v_j.
```

The vertex load has the Walsh representation

```text
b_v=sum_(S nonempty) chi_S(v) beta_S.              (2.2)
```

Equation (2.1) removes only the constant character. It does not force the
seven values `beta_S` to vanish.

## 3. Exact pressure incidence

The annular family, low wave, and polarization are

```text
k_abc=(2N+a-1,b-(N+1)/2,c-(N+1)/2),

hhat_N(k_abc)
 =(-1)^(a+b+c)
  product_j sin(pi index_j/(N+1))
  P_k(e_3)/|k|,

ell=(0,1,-1),

Uhat(ell)=-i(e_2+e_3)/sqrt(2).
```

For a fixed pressure difference `q`, summing the low signs and vertex
outputs gives a quadratic continuum matrix `A_v`. Its eight values sum to
zero exactly.

It is cleaner to record their Walsh transform. The following matrices are
the coefficients of `sqrt(2)`:

```text
A_x = 0,

A_y =
 [ 0       0       0    ]
 [ 0      1/10   -1/20  ]
 [ 0     -1/20   -1/10  ],

A_xy =
 [ 1/24    0       0    ]
 [ 0      -1/12    1/24 ]
 [ 0       1/24    1/24 ],

A_z =
 [ 0       0       0    ]
 [ 0       1/10    1/20 ]
 [ 0       1/20   -1/10 ],

A_xz =
 [-1/24    0       0    ]
 [ 0      -1/24   -1/24 ]
 [ 0      -1/24    1/12 ],

A_yz =
 [ 0       0       0   ]
 [ 0      -1/8     0   ]
 [ 0       0       1/8 ],

A_xyz =
 [ 0       0       0    ]
 [ 0       1/10    0    ]
 [ 0       0      -1/10 ].
```

All entries are exact elements of `Q(sqrt(2))`. The pure `x` character is
zero, while the other six matrices are nonzero.

The leading kinetic matrix is zero separately at every vertex. Therefore
the complete leading response is the pressure response above; cross
pressure remains lower order.

## 4. Continuum signs

On

```text
D=[2,3] x [-1/2,1/2]^2,
```

put

```text
S=sin(pi(x-2))sin(pi(y+1/2))sin(pi(z+1/2)),

V=P_xi(e_3)/|xi|.
```

Parity makes the covariance

```text
C=integral_D S^2 V V^T
```

diagonal. Its three diagonal entries are positive. The six surviving
Walsh limits are obtained from

```text
beta_S=sqrt(2) A_S:C.
```

Pointwise inequalities fix their signs:

```text
V_z^2>V_y^2,

V_x^2-2V_y^2+V_z^2>0,

2V_z^2-V_x^2-V_y^2>0.
```

Consequently,

```text
beta_y, beta_z, beta_xyz <0,

beta_xy, beta_xz, beta_yz >0,

beta_x=0.                                          (4.1)
```

Positive scalar damping preserves all these signs.

The numerical continuum vertex limits are:

```text
(-,-,-)   0.015472511324367
(-,-,+)  -0.007489595594444
(-,+,-)  -0.005169731852765
(-,+,+)  -0.002813183877158
(+,-,-)   0.002813183877158
(+,-,+)   0.000456635901550
(+,+,-)  -0.001863227840129
(+,+,+)  -0.001406591938579
```

They sum to zero. Their `l1` norm is approximately `0.03748466221`, so the
zero-sum vector is far from zero.

The nonnegative selector supported on its three positive entries retains
half that `l1` mass. More simply, reversing the low polarization makes the
`+++` entry positive, and the nonnegative coefficient vector
`w=delta_(+++)` retains it. Equal-weight cancellation therefore does not
control arbitrary nonconstant compatible coefficients.

## 5. Vertex Fisher geometry

Let

```text
F_N(a,b,c)=(-1)^(a+b+c) k_abc tensor hhat_N(k_abc).
```

After this alternating gauge, a plus vertex sign produces the
zero-boundary difference operator

```text
D f=(f_1,f_2-f_1,...,f_N-f_(N-1),-f_N),
```

while a minus sign produces the zero-boundary sum operator

```text
S f=(f_1,f_2+f_1,...,f_N+f_(N-1),f_N).
```

For `T_+=D` and `T_-=S`, the exact tensor identity is

```text
E_v(h_N)
 =1/32 ||T_(v_1)T_(v_2)T_(v_3)F_N||_2^2.          (5.1)
```

If `r(v)` is the number of minus signs, smooth-grid scaling in (5.1)
gives

```text
E_v(h_N)=Theta(N^(2r(v)-3)).                       (5.2)
```

Thus:

```text
r=0:  E_v=Theta(N^-3),
r=1:  E_v=Theta(N^-1),
r=2:  E_v=Theta(N),
r=3:  E_v=Theta(N^3).
```

At `N=65`, the scaled constants in explicit sign tuples are:

```text
(+,+,+)   -3   3.75259618
(+,+,-)   -1   1.51541489
(+,-,+)   -1   1.54089507
(-,+,+)   -1   1.56564718
(+,-,-)    1   0.62228351
(-,+,-)    1   0.63262526
(-,-,+)    1   0.64308159
(-,-,-)    3   0.25985150
```

The partition identity also gives

```text
sum_v E_v(h_N)=mean |grad h_N|^2.                 (5.3)
```

Global dissipation is dominated by the `---` cell and can pay an order-`N`
load. The local `+++` Fisher energy is smaller by six powers of `N`.
Borrowing global or neighboring dissipation is therefore a real additional
payment, not a proof of the original local estimate.

## 6. One heat window

Let the high and low modes evolve under the linear heat semigroup with
viscosity `nu>0`, and set

```text
tau=N^2 t.
```

The high continuum profile gains the positive multiplier

```text
exp(-nu tau |xi|^2).
```

At `+++`,

```text
B_+++(tau/N^2)/N
 -> sqrt(2)/20 integral_D
    S^2 exp(-2nu tau|xi|^2)(V_y^2-V_z^2)<0.       (6.1)
```

For any fixed scaled window `0<=tau<=T`,

```text
integral_0^(T/N^2) B_+++(t)dt =-Theta(N^-1).
```

The mixed-difference proof remains uniform after multiplication by the
smooth heat factor, so

```text
integral_0^(T/N^2) E_+++(t)dt=O(N^-5).
```

Therefore

```text
|integral B_+++ dt| / integral E_+++ dt
 >=c_(nu,T)N^4.                                   (6.2)
```

The pressure margin is analytic:

```text
|lim N integral_0^(T/N^2)B_+++(t)dt|
 >=T exp(-19nu T) 51sqrt(2)/438976.               (6.3)
```

For the audited values `nu=1` and `T=0.1`, continuum quadrature gives

```text
lim N integral B_+++ dt
 =-0.0000807957036071.
```

The lower bound in (6.3) is `2.45745e-6`. Equal-weight cancellation still
holds exactly at each heat time.

## 7. Small-amplitude Navier-Stokes transfer

Heat flow alone does not include nonlinear phase motion. A perturbative
argument nevertheless rules out using that motion to prove a universal
homogeneous estimate valid down to arbitrary amplitudes.

Fix `N`, set `f_N=U+h_N`, and use initial data

```text
u_epsilon(0)=epsilon f_N.
```

For `s>5/2`, the mild Navier-Stokes formula and the standard bilinear
`H^s` estimate give, on the fixed interval `[0,T/N^2]`,

```text
u_epsilon(t)
 =epsilon exp(nu t Delta)f_N+O_N(epsilon^2)
```

in `C_t H^s` as `epsilon->0`.

The localized HHL flux is cubic in its projected low/high arguments. Its
trajectory integral therefore equals

```text
epsilon^3 times the heat HHL integral +O_N(epsilon^4).
```

The low norm times the high Fisher integral has the same leading
`epsilon^3` homogeneity. If a one-vertex trajectory estimate held with one
constant independent of `epsilon` and `N`, divide by `epsilon^3` and first
send `epsilon` to zero for each fixed `N`. It would imply the heat estimate
contradicted by (6.2).

This excludes nonlinear phase rescue only for universal homogeneous local
estimates that apply from time zero at arbitrarily small amplitude. It does
not exclude:

- compensation requiring a large-amplitude regime;
- delayed estimates;
- nonhomogeneous remainder payments;
- coefficient restrictions imposed by an adaptive construction;
- the exact equal-weight global identity.

## 8. Finite replay

The static `+++` rows reproduce the predecessor:

```text
N    B_+++/N          N^3 E_+++    |B|/E
---  ----------------  -----------  ----------
  3  -0.003919922892    1.4481566       0.2193
  5  -0.002639667824    2.1553834       0.7654
  9  -0.002012123796    2.8202761       4.6809
 17  -0.001707573940    3.3002441      43.2145
 33  -0.001556452611    3.5917574     513.9072
 65  -0.001481302468    3.7525962    7046.3683
```

For `N=3`, all 32 component/vertex loads and all eight Fisher forms were
independently checked against the dictionary Fourier algebra. The maximum
load residual is below `3.5e-17`, and the maximum Fisher residual is below
`2.2e-14`.

Every finite equal-weight load sum is below `8e-17`, and every sum of the
eight vertex Fisher forms agrees with the global enstrophy form to
roundoff.

## 9. Route decision

Established:

- exact equal-weight cancellation of all eight complete HHL loads;
- exact six-channel Walsh incidence for the annular family;
- survival of nonconstant weighted loads at order `N`;
- exact vertex Fisher difference/sum identities;
- the four Fisher scaling classes `N^-3,N^-1,N,N^3`;
- persistence of the local `N^4` obstruction through a heat window;
- transfer to small-amplitude smooth Navier-Stokes trajectories for
  universal homogeneous estimates.

Open:

- a sharp inequality coupling coefficient edge variation to neighboring
  Fisher energies;
- whether an admissible adaptive coefficient construction can concentrate
  at `+++` without paying those neighbors;
- genuinely large-amplitude phase compensation;
- cross-shell HHL absorption;
- terminal dual control and critical `L^3`;
- Navier-Stokes global regularity.

The next gate should insert the exact six-channel vector and the scaling
law (5.2) into the twelve-edge compatible coefficient graph. The question
is no longer whether the load cancels globally. It is whether any
admissible nonconstant coefficient vector that retains it must pay enough
neighboring Fisher energy.

The deterministic audit is
`scripts/annular_eight_vertex_heat_window_gate_audit.py`; its production
result is
`results/annular_eight_vertex_heat_window_gate_audit_v1.json`.
