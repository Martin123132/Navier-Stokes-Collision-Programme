# Cross-shell modulated-wave gate

Status: an exact divergence-free two-sideband family falsifies every
carrier-separation gain for the isolated high-high-to-low pressure load.
The same family also has a nonzero limit for the complete signed cubic
high-high-low local-energy flux, after kinetic transport and cross pressure
are included. This is a scoped no-go for an instantaneous shell estimate,
not a Navier-Stokes blow-up or regularity result.

## 1. The channel left after self-shell closure

The preceding theorem controls the complete pressure generated and tested
by one annular shell. The first remaining nonlocal interaction has two
comparable high inputs and one low testing velocity:

```text
p_L=P_L R_iR_j(w_(H,i)w_(H,j)),

B_HHL^p
 =mean[p_L U_L dot grad Phi_v],

H>>L>=m.                                           (1.1)
```

A tempting target is a gain

```text
|B_HHL^p|
 <=C(L/H)^alpha [amplitude norms],

alpha>0.                                           (1.2)
```

This note proves that no such gain can hold when the remaining norms stay
fixed on the family below.

## 2. Exact two-sideband field

Take partition frequency `m=1` and integer carrier `H>=8`. Define

```text
a_H=(1,1,H),

b_H=(0,0,H),

q=a_H-b_H=(1,1,0).                                (2.1)
```

Let

```text
B=e_1,

A_H=P_(a_H perpendicular)e_1/
    |P_(a_H perpendicular)e_1|.                   (2.2)
```

Then

```text
a_H dot A_H=0,

b_H dot B=0,

A_H->e_1.                                         (2.3)
```

Use the real high field with Fourier coefficients

```text
what_H(a_H)=A_H,

what_H(b_H)=B,

what_H(-k)=conjugate(what_H(k)).                   (2.4)
```

It is exactly divergence free and lies in the annulus

```text
H<=|k|<=sqrt(H^2+2).                               (2.5)
```

Thus its Fourier amplitudes remain fixed while the high/low scale ratio
tends to infinity.

## 3. Fixed low velocity and vertex

Set

```text
k=(-2,-2,-1),

r=(1,1,1),

q+k+r=0.                                          (3.1)
```

The testing velocity is the real divergence-free field

```text
Uhat(k)=iC,

C=P_(k perpendicular)r
 =(-1/9,-1/9,4/9),                                (3.2)

Uhat(-k)=conjugate(Uhat(k)).
```

Use the all-cosine vertex

```text
Phi_+(x)
 =product_(j=1)^3 [1+cos(x_j)]/2.                 (3.3)
```

Its Fourier coefficient at `r` is `1/64`, so

```text
Fourier(grad Phi_+)(r)=ir/64.                     (3.4)
```

All low data in (3.1)-(3.4) are independent of `H`.

## 4. Surviving Reynolds stress

The two high sidebands generate the low tensor coefficient

```text
Rhat_H(q)
 =A_H tensor e_1+e_1 tensor A_H

 ->2e_1 tensor e_1.                               (4.1)
```

This is the standard slowly modulated Reynolds-stress mechanism in exact
finite Fourier form. It is anisotropic and does not decay with the carrier.

For the pressure convention

```text
phat_HH(q)
 =-q_iq_j Rhat_(H,ij)(q)/|q|^2,                   (4.2)
```

the polarization can be evaluated explicitly:

```text
q dot A_H
 =H^2/sqrt((H^2+1)(H^2+2)).                       (4.3)
```

Therefore

```text
phat_HH(q)
 =-H^2/sqrt((H^2+1)(H^2+2))

 ->-1.                                             (4.4)
```

The low pressure is order one, despite arbitrarily large frequency
separation.

## 5. Pressure-only no-go

Only the triples `(q,k,r)` and their conjugates enter the selected vertex
load. Equations (3.2), (3.4), and (4.4) give

```text
B_HHL^p
 =-phat_HH(q)/144

 ->1/144.                                          (5.1)
```

The high Fourier coefficients, low velocity, and partition weight are all
bounded independently of `H`. Hence any estimate of the form (1.2), with
an otherwise `H`-independent amplitude budget, is false.

Amplitude rescaling does not repair the missing carrier gain:
`w_H->a w_H` and `U->bU` multiply both the load and its natural cubic
amplitude budget by `a^2b`.

## 6. Complete cubic local-energy flux

Pressure alone is not the whole local energy interaction. For

```text
u=w_H+tU,
```

take the coefficient linear in `t` of

```text
F(u)=[|u|^2/2+p[u]]u.                              (6.1)
```

The exact high-high-low flux is

```text
F_HHL
 =(|w_H|^2/2)U
  +(U dot w_H)w_H
  +p[w_H,w_H]U
  +(p[U,w_H]+p[w_H,U])w_H.                        (6.2)
```

The final line includes both ordered cross-pressure terms.

At the low wave `q`, the first three terms have limiting coefficient

```text
[(tr R)/2 I+R+p_HH I]Uhat(k).                     (6.3)
```

Using

```text
R=2e_1 tensor e_1,

(tr R)/2=1,

p_HH=-1,                                          (6.4)
```

the isotropic kinetic term and pressure scalar cancel. The anisotropic
term remains:

```text
2e_1(e_1 dot Uhat(k)).                             (6.5)
```

For the cross pressure, its output frequency has size `H`. One symbol
contraction is with a polarization transverse to its own high carrier,
which gives at least `O(1/H)` decay. In this exact geometry the paired
terms improve to `O(1/H^2)`.

Equations (3.4)-(6.5) then give

```text
mean[F_HHL dot grad Phi_+]
 ->1/144.                                          (6.6)
```

Combining pressure and transport therefore does not restore a positive
power of `L/H`.

## 7. Independent cubic reconstruction

The audit forms (6.2) component by component. It independently recovers the
same object from the cubic polynomial identity

```text
F_HHL
 =[F(w_H+U)-F(w_H-U)]/2-F(U).                     (7.1)
```

The maximum Fourier-coefficient disagreement over the eight tested
carriers is below `2.3e-16`. This checks all combinatorial factors in
(6.2), including the two cross-pressure terms.

## 8. Finite-carrier replay

The exact sparse calculation gives:

```text
H       pressure       kinetic       cross p        complete HHL

8       0.00678561     1.166e-3     -3.215e-3       0.00473735
16      0.00690400     6.203e-4     -8.640e-4       0.00666029
32      0.00693429     3.183e-4     -2.186e-4       0.00703398
64      0.00694190     1.610e-4     -5.459e-5       0.00704832
128     0.00694381     8.095e-5     -1.361e-5       0.00701114
256     0.00694429     4.058e-5     -3.398e-6       0.00698147
512     0.00694440     2.032e-5     -8.486e-7       0.00696387
1024    0.00694443     1.017e-5     -2.120e-7       0.00695439. (8.1)
```

The common analytic limit is

```text
1/144=0.00694444444444.                            (8.2)
```

At `H=1024`, the pressure load is `0.99999857` of the limit and the
complete load is `1.00143192` of it. The pressure error decreases on every
row; the complete error decreases from `H=64` onward. The scaled
quantities satisfy

```text
max H|kinetic load|<0.01042,

max H^2|cross-pressure load|<0.22384.              (8.3)
```

Every velocity mode is divergence free to below `1e-12`, and every load is
real to the same tolerance.

## 9. What is falsified

Falsified:

- a universal positive carrier-separation power for the pressure-only
  high-high-to-low load when the remaining amplitude norms are fixed;
- restoration of such a power by combining all instantaneous cubic
  high-high-low kinetic and pressure terms;
- a route that tries to sum cross-shell interactions using frequency
  separation alone.

Not falsified:

- dyadic summation using actual shell-amplitude decay;
- conservative telescoping between adjacent scale boundaries;
- time-integrated compensation paid by viscosity;
- cancellation after coupling all eight partition vertices;
- a Carleson or square-function estimate that retains the shell sequence;
- the broader Navier-Stokes programme.

## 10. Route decision

The next object must retain shell amplitudes and conservative transfer
structure. Construct the exact three-shell local-energy identity, using
the fact that the two largest frequencies in every occupied triad are
comparable. Remove the self-shell pressure terms already controlled, then
write the surviving high-high-to-low Reynolds-stress channel as a signed
transfer between shell boundaries.

Before taking absolute values, test:

- telescoping in the low-shell index;
- antisymmetry between donating and receiving scales;
- cancellation across the eight partition vertices;
- time-integrated payment by shellwise viscous dissipation.

The certificate and exact sparse replay are generated by
`scripts/cross_shell_modulated_wave_gate_audit.py`.
