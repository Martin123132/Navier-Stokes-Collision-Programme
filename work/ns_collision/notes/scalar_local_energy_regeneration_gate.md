# Scalar local-energy regeneration gate

Status: the sharp `H^(5/2)` dense HHH tensor is absent from the ordinary
scalar local-energy trace at zero output, exactly as energy conservation
requires. It nevertheless survives in the complete time derivative of the
HHL local-energy transfer after the low velocity, kinetic flux, high-high
pressure, cross pressure, and Leray projection are all evolved together.
The surviving four-leg HHHL coefficient is isolated in a pure top-Walsh
channel. Its central real-conjugate quartet has exact asymptotic value
`3sqrt(2)/16`, while a directed interval certificate proves a uniform
positive lower bound over the complete continuous packet domain. The
weakest shell weight controlled by Leray energy and dissipation is exactly
`H^(-3/2)` on the forcing amplitude. This is a sharp smooth shell theorem,
not a global regularity proof.

Correction: an earlier version incorrectly said that the fixed-relative-
width packet coefficient approaches the central value. Since `R/M` was
fixed, the normalized offsets did not shrink. That limit claim is withdrawn
below and replaced by a continuous-domain positivity proof.

## 1. Two scalar statements that must not be confused

For a divergence-free Fourier field, let

```text
Rhat(q)=sum_(a+b=q) uhat(a) tensor uhat(b)          (1.1)
```

and let `Ghat(q)` be its nonlinear Navier-Stokes time derivative. The
ordinary scalar local energy sees only

```text
tr Rhat(q)=Fourier_q[|u|^2].                       (1.2)
```

The HHL transfer used in the shell/partition argument is a different
scalar. It pairs the full anisotropic stress with the gradient of a
localized low velocity. Differentiating that transfer can therefore see
the traceless part of `Ghat(0)`.

The previous dense-packet result proved

```text
||Ghat_H(0)|| approximately H^(5/2),               (1.3)

tr Ghat_H(0)=0.                                    (1.4)
```

Equation (1.4) settles the first scalar question. It does not settle the
time derivative of the HHL transfer.

## 2. Exact scalar trace identity

For three distinct divergence-free legs `(a,U)`, `(b,V)`, `(c,W)`, put

```text
q=a+b+c.                                           (2.1)
```

Let

```text
E(U,V,W)
 =[(U dot V)W+(U dot W)V+(V dot W)U]/6

  +[p(U,V)W+p(U,W)V+p(V,W)U]/3,                   (2.2)
```

where the single ordered pressure pair is

```text
p(U,V)
 =-[(a+b) dot U][(a+b) dot V]/|a+b|^2.            (2.3)
```

When `a+b=0`, define `p(U,V)=0`; this is the fixed zero-mean pressure
gauge. Formula (2.3) is used only at nonzero output.

The complete cubic flux coefficient is `F_abc=6E(U,V,W)`. Direct
contraction of the projected Navier-Stokes symbol gives

```text
tr G_abc(q)=-2i q dot F_abc(q).                    (2.4)
```

Thus

```text
tr G_abc(0)=0                                      (2.5)
```

for every triad, not merely after summation. Four exact Fourier replays
have maximum residual below `9.4e-13`.

Here `G` is the nonlinear/Euler contribution. This cancellation is not a
claim that full Navier-Stokes energy is conserved: viscosity contributes
the standard energy dissipation.

This proves that the carrier derivative cancels from the ordinary scalar
trace. For fixed nonzero `q`, (2.4) replaces the carrier derivative by the
low output `q`; dense multiplicity may still remain, but the
`H^(5/2)` zero-output tensor itself is invisible.

## 3. Complete HHL local-energy transfer

Let `h` be the high field and `z` a low velocity. The coefficient linear
in `z` of

```text
F(u)=[|u|^2/2+p[u,u]]u
```

at `u=h+epsilon z` is

```text
T(h,z)
 =(|h|^2/2)z
  +(z dot h)h
  +p[h,h]z
  +(p[z,h]+p[h,z])h.                              (3.1)
```

The last line contains both ordered cross-pressure terms.

Write the nonlinear projected evolutions

```text
N_0=N[h,h],

N_1=N[h,z]+N[z,h].                                (3.2)
```

Here `N_1` is the complete linearized evolution of the low perturbation;
it is not held fixed. Differentiating (3.1) gives

```text
dot T_K
 =(h dot N_0)z
  +(1/2)|h|^2 N_1
  +(N_1 dot h+z dot N_0)h
  +(z dot h)N_0,                                  (3.3)
```

and

```text
dot T_p
 =[p[N_0,h]+p[h,N_0]]z
  +p[h,h]N_1

  +[p[N_1,h]+p[z,N_0]+p[N_0,z]+p[h,N_1]]h

  +[p[z,h]+p[h,z]]N_0.                            (3.4)
```

Equations (3.3)-(3.4) retain every kinetic, high-high-pressure,
cross-pressure, and Leray-projection contribution before taking absolute
values.

The quartic coefficient can equivalently be written with the symmetric
trilinear form (2.2):

```text
Q_HHHL
 =6 E(h,z,N[h,h])
  +3 E(h,h,N[h,z]+N[z,h]).                        (3.5)
```

For the selected waves, the three high legs sum to zero and the low leg
has wave `-r`. Thus `Q_HHHL` itself has Fourier output `-r`. Pairing it
with the `+r` coefficient of the partition gradient produces the final
zero-frequency scalar integral. These two frequency statements are
distinct.

The audit independently reconstructs (3.5) by the 16-sign Walsh
polarization of

```text
DF(u)[N(u,u)].                                    (3.6)
```

The maximum vector disagreement is below `7.3e-13`.

Viscosity has not been omitted from this coefficient. The viscous vector
field is linear, so `DF(u)[nu Delta u]` is cubic. Its coefficient odd in
the three distinct high legs and the low leg is exactly zero. Equations
(3.3)-(3.5) are therefore the complete four-leg HHHL coefficient of the
Navier-Stokes time derivative.

## 4. Exact central survival

Use the dense-packet center waves and phased polarizations

```text
A=(1,0,0),       U_A=(0,-3,1),

B=(-1,1,0),      U_B=(-2,-2,2),

C=(0,-1,0),      U_C=i(-3,0,1),                   (4.1)
```

with `A+B+C=0`. Take

```text
r=(1,1,1),

k_low=-r,

Z=(1,-1,0)/sqrt(2),

zhat(k_low)=iZ.                                   (4.2)
```

At zero low wave, transversality makes the cross pressure vanish.
Linearized sweeping contributes zero at mean output because
`A+B+C=0`. The leading complete differentiated transfer is therefore

```text
Q_HHHL(0)=G_HHH(0)Z.                              (4.3)
```

For the exact integer matrix from the dense HHH gate,

```text
r^T G_HHH(0)Z=-6sqrt(2)H.                         (4.4)
```

The corner partition coefficient is `1/64`, and the real conjugate pair
contributes twice. Hence

```text
complete vertex load/H
 ->3sqrt(2)/16
  =0.265165042944955.                              (4.5)
```

Finite carriers give

```text
H       complete/H          stress prediction/H

16      0.283114680286       0.265165042945
32      0.269685767824       0.265165042945
64      0.266297271124       0.265165042945
128     0.265448227361       0.265165042945
256     0.265235847001       0.265165042945
512     0.265182744456       0.265165042945
1024    0.265169468354       0.265165042945.       (4.6)
```

The final relative error is below `1.67e-5`.

The component calculation is also informative. Two pressure pieces are
individually about fourteen times larger than the answer and cancel each
other. The linearized-low contribution tends to zero. After those genuine
cancellations, the stress-regeneration coefficient (4.5) remains.

## 5. Frequency-isolated dense packet

To ensure that neighboring partition frequencies cannot manufacture or
cancel the result, use

```text
4 B_M={4n:n in {-M,...,M}^3}.                     (5.1)
```

Take the explicit carrier

```text
R=16384M.                                         (5.2)
```

The high waves are `RA+4B_M`, `RB+4B_M`, `RC+4B_M` and their conjugates,
with the same projected polarizations and one global unit-energy
normalization.

Every high offset sum lies in `4Z^3`. After adding the low waves `+/-r`, a
quartic output in the partition support `{-1,0,1}^3` is possible only at
the matching corner `+/-r`. Mixed carrier signs stay far outside that
support. Consequently all eight vertex loads are exactly

```text
b_v=chi_123(v)b_+++,

chi_123(v)=v_1v_2v_3.                              (5.3)
```

### 5.1 Correct fixed-width asymptotics

Write two normalized offsets as `x=n/M`, `y=m/M` and the third as
`z=-x-y`. The normalized waves are

```text
a=A+x/4096,

b=B+y/4096,

c=C+z/4096.                                       (5.4)
```

The true continuum domain has

```text
x,y,z in [-1,1]^3,       x+y+z=0.                 (5.5)
```

It does not collapse to `(A,B,C)` as `M` increases. Therefore the packet
coefficient need not approach the central value `3sqrt(2)/32`; its natural
limit is a weighted polytope average of the continuous symbol.

### 5.2 Directed interval certificate

The audit evaluates the leading normalized coefficient on the larger
domain

```text
x,y in [-1,1]^3,       z=-x-y in [-2,2]^3,        (5.6)
```

which contains (5.5). It factors out the normalized low polarization and
obtains the outward-rounded interval

```text
-r^T G(a,b,c)Z_0/64
 in [0.14667827693662766,
     0.22832724863627277],                         (5.7)

Z_0=(1,-1,0).
```

After restoring `Z=Z_0/sqrt(2)`, the leading coefficient is at least

```text
0.10371720427464774.                               (5.8)
```

The complete finite-low-wave coefficient depends on `tau=1/R`. Directed
interval automatic differentiation over

```text
0<=tau<=1/16384
```

starts from the exact identity `S(0)=-r^T G(a,b,c)Z_0/64`. Indeed,

```text
N[U_j,Z_0]+N[Z_0,U_j]=-i(Z_0 dot a_j)U_j
```

at zero low wave. Symmetry and trilinearity of `E` make the sum of the
three low-evolution rows proportional to
`Z_0 dot (a+b+c)=0`. In the high-evolution rows, cross pressure vanishes
by transversality and the zero-output pressure gauge; the remaining
kinetic trace term vanishes after the partition contraction because
`r dot Z_0=0`. Thus the mean-value comparison below starts from the
leading stress value exactly, not from a numerical approximation.

The interval derivative is

```text
dS/dtau in
[-0.7940866835614854,
  0.7945951061371643].                             (5.9)
```

The mean-value correction is at most

```text
4.84982364585672e-5                               (5.10)
```

before low-vector normalization. Hence every complete positive quartet in
every packet `M>=1` satisfies

```text
S_complete>=0.1036829108427723>0.                 (5.11)
```

This is the missing uniform theorem. It does not infer positivity from
three finite rows and does not identify the polytope-average limit with
the central value.

The mode and coherent-triad counts remain

```text
N_mode=6(2M+1)^3,

N_triad=(3M^2+3M+1)^3.                            (5.12)
```

Unit energy supplies three factors of `M^(-3/2)`, coherent counting
supplies `M^6`, and the nonlinear derivative supplies `R`. Since
`R=16384M`,

```text
|dot B_HHL|>=c R^(5/2)                            (5.13)
```

with an explicit `c>0`. The complete signed differentiation therefore
does not remove the dense exponent.

## 6. Sharp negative shell norm

Let

```text
a_H(t)=||u_H(t)||_2,

E_*=sup_t sum_H a_H(t)^2,

D=integral sum_H H^2 a_H(t)^2 dt.                (6.1)
```

For any fixed finite low-output Fourier/Walsh channel, the annular
Bernstein estimate gives

```text
|G_H|<=C H^(5/2)a_H^3.                            (6.2)
```

Multiply its square by `H^(-3)`:

```text
H^(-3)|G_H|^2
 <=C H^2 a_H^6
 <=C E_*^2 H^2 a_H^2.                            (6.3)
```

Summing shells and integrating time proves

```text
sum_H H^(-3)||G_H||_(L2_t)^2
 <=C E_*^2 D.                                     (6.4)
```

Thus the forcing amplitude belongs to the shell-negative norm with weight

```text
H^(-3/2).                                         (6.5)
```

This exponent is sharp from the available inputs. The dense parabolic
pulse has weighted squared cost

```text
H^(3-2s)                                          (6.6)
```

under amplitude weight `H^(-s)`. It remains bounded exactly when
`s>=3/2`.

## 7. Viscosity leaves a summable half derivative

For the shell relaxation equation

```text
dot c_H+c nu H^2 c_H=G_H,       c_H(0)=0,         (7.1)
```

Young's inequality gives

```text
||c_H||_(L2_t)
 <=(c nu H^2)^(-1)||G_H||_(L2_t).                 (7.2)
```

For dyadic `H>=H_0`, write the right side as

```text
H^(-1/2)[H^(-3/2)||G_H||_(L2_t)]/(c nu).
```

Cauchy-Schwarz and `sum_(H>=H_0)H^(-1)<=2/H_0`
then give

```text
||sum_H c_H||_(L2_t)
 <=sqrt(2)/(c nu sqrt(H_0))
   [sum_H H^(-3)||G_H||_(L2_t)^2]^(1/2).          (7.3)
```

Combining (6.4) and (7.3) is the first route in this regeneration programme
that both survives the dense packet and is paid by the standard Leray
quantities. The two viscous derivatives exceed the sharp multiplicity
loss by half a derivative, which is dyadically summable.

Equation (7.3) controls the forced Duhamel component. Initial stress is a
separate heat term and remains part of the next complete shell theorem.

## 8. Scope and next gate

Established:

- exact scalar trace cancellation at zero output;
- the complete quartic HHHL derivative, including low evolution and every
  kinetic and pressure term;
- independent 16-sign reconstruction of that derivative;
- survival of the exact `3sqrt(2)/16` real-quartet coefficient;
- a frequency-isolated dense pure top-Walsh packet with a uniform
  continuous-domain interval lower bound;
- sharp `H^(5/2)` differentiated-transfer growth;
- the Leray-controlled, sharp `H^(-3/2)` shell forcing norm;
- a weighted viscous Duhamel estimate with a remaining summable half
  derivative.

Not established:

- the complete comparable-shell `HHH/HHL` filtered evolution;
- control of filter commutators and moving shell boundaries;
- a nonlinear low-regularity Duhamel theorem;
- passage to suitable weak solutions;
- exceptional-set removal or global regularity.

The next gate is no longer another cancellation search. It is to lift
(6.4)-(7.3) from one fixed low channel to the complete dyadic shell system,
including comparable-shell neighbors, the already-controlled HHL
commutator, filter commutators, and initial stress.

The identities and finite-mode replays are generated by
`scripts/scalar_local_energy_regeneration_gate_audit.py`.
