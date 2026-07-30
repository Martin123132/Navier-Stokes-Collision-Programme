# High-carrier coercivity and the weighted zero-face gate

Status: a rigorous linear carrier lower bound for unweighted enstrophy of
pure high-pass velocity fields, an exact support-only no-go for promoting
that bound to vertex-weighted Fisher energy, and a finite-Fourier
Navier-Stokes pressure pilot exhibiting the predicted zero-face
concentration mechanism. The pressure-packet asymptotics, a floor-free
intrinsic absorption theorem, mixed low/high interactions, and regularity
remain open.

## 1. Pure high-pass fields have a linear unweighted cost

Work on the normalized three-torus. Let

```text
u_hat(k)=0 for |k|<K,

div u=0,

p=-R_i R_j(u_i u_j).                                (1.1)
```

For a smooth scalar partition function `Phi`, define

```text
b_Phi=mean[p u dot grad Phi].                       (1.2)
```

Let `C_R` denote the `L^(3/2)` norm of the matrix-to-scalar double
Riesz map and `C_S` the mean-zero Sobolev constant

```text
||f||_6<=C_S||grad f||_2.
```

Holder and the pressure estimate give

```text
|b_Phi|
 <=||grad Phi||_infinity ||p||_(3/2)||u||_3

 <=C_R||grad Phi||_infinity||u||_3^3.               (1.3)
```

High-pass Poincare and interpolation give

```text
||u||_2<=K^(-1)||grad u||_2,

||u||_3^3
 <=||u||_2^(3/2)||u||_6^(3/2)

 <=C_S^(3/2)K^(-3/2)||grad u||_2^3.                 (1.4)
```

Combining (1.3) and (1.4),

```text
|b_Phi|
 <=C_R C_S^(3/2)||grad Phi||_infinity
   K^(-3/2)||grad u||_2^3.                          (1.5)
```

Therefore every nonzero fixed load obeys

```text
||grad u||_2^2
 >=K/C_S
   [|b_Phi|/(C_R||grad Phi||_infinity)]^(2/3).      (1.6)
```

For a frequency-`m` vertex partition,

```text
||grad Phi_(v,m)||_infinity<=sqrt(3)m/2.            (1.7)
```

Equations (1.6)-(1.7) prove linear carrier coercivity of the global
unweighted least-enstrophy problem for a pure high-pass velocity. This
extends the previous explicit-block result in one direction but does not
retain its power:

- coercivity is now global over all pure high-pass architectures;
- its universally proved carrier power is `K`;
- whether a stronger global lower power holds remains open.

The linear power is the natural sharp candidate. A three-dimensionally
localized packet at a point where the partition gradient is nonzero has

```text
fixed cubic load:       A^3 delta^3 approximately 1,

unweighted enstrophy:   A^2 delta approximately delta^(-1),

carrier:                K approximately delta^(-1). (1.8)
```

A rigorous sharpness construction still requires a pressure-active
band-limited packet with enclosed constants.

### 1a. The square-factor high-pass bridge

The failure of an unweighted-to-weighted transfer does not mean that the
zero face hides every useful high-pass quantity. Factor the vertex basis as

```text
Phi_(v,m)=psi_(v,m)^2,                              (1.9)
```

where every one-dimensional factor of `psi` is either
`sin(mx_j/2)` or `cos(mx_j/2)`. Its Fourier support consists of the eight
half-lattice shifts

```text
(+/-m/2,+/-m/2,+/-m/2),                            (1.10)
```

each with length `sqrt(3)m/2`. It is also an exact eigenfunction:

```text
Delta psi=-(3m^2/4)psi.                            (1.11)
```

Integration by parts gives the ground-state identity

```text
||grad(psi u)||_2^2
 =mean[Phi|grad u|^2]
  +(3m^2/4)||psi u||_2^2.                          (1.12)
```

If `u_hat(k)=0` for `|k|<K`, multiplication by `psi` leaves
`psi u` supported above

```text
K-sqrt(3)m/2.                                      (1.13)
```

Applying high-pass Poincare to `psi u` and then using (1.12) yields, for
`K>sqrt(3)m`,

```text
||psi u||_2^2
 <=mean[Phi|grad u|^2]/[K(K-sqrt(3)m)].             (1.14)
```

Moreover,

```text
u grad psi=grad(psi u)-psi grad u.
```

Equations (1.12)-(1.14) therefore give

```text
||u grad psi||_2
 <=[1+sqrt(1+(3m^2/4)/[K(K-sqrt(3)m)])]
   mean[Phi|grad u|^2]^(1/2).                      (1.15)
```

At `K=2sqrt(3)m`, the bracket is exactly

```text
1+3sqrt(2)/4.                                      (1.16)
```

This is the correct weighted uncertainty bridge. A zero face can hide
unweighted `L2` mass from weighted Fisher, but it cannot hide the weighted
mass `psi u` or the factor `u grad psi` once spectral separation is imposed.
The remaining obstruction is now the commutator generated when `psi` is
moved through the high-output pressure multiplier.

## 2. A zero face destroys support-only weighted coercivity

Consider the one-dimensional vertex factor

```text
phi_-(x)=sin(x/2)^2=(1-cos x)/2.                   (2.1)
```

For an integer `N>=2`, put

```text
f_N(x)
 =sum_(j=1)^N c_j exp(i(N+j)x),

c_j=sqrt[2/(N+1)] sin[pi j/(N+1)].                 (2.2)
```

The sine identity gives

```text
||f_N||_2^2=1,

supp f_hat_N contained in {N+1,...,2N}.            (2.3)
```

Write

```text
d_k=k f_hat_N(k),
```

with `d_k=0` outside the displayed band. Since multiplication by
`phi_-` shifts frequency by at most one,

```text
mean[phi_-|f_N'|^2]
 =1/4 sum_k |d_k-d_(k-1)|^2.                       (2.4)
```

The sine window vanishes at both spectral endpoints. For every interior
pair,

```text
|d_(k+1)-d_k|
 <=sqrt[2/(N+1)](1+2pi).                           (2.5)
```

The two endpoint terms obey the same elementary sine bound. Consequently,

```text
mean[phi_-|f_N'|^2]
 <=[5pi^2+2(1+2pi)^2]/4
 <38.86                                             (2.6)
```

uniformly in `N`. In contrast,

```text
||f_N'||_2^2>=(N+1)^2.                             (2.7)
```

For `N>=2`, the real shear

```text
u_N(x)=(0,sqrt(2) Re f_N(x_1),0)                   (2.8)
```

is divergence free. Its positive and negative frequency blocks do not
interact through the frequency-one weight, so (2.3)-(2.7) apply with the
same normalization.

The audit reaches:

```text
N     weighted Dirichlet     unweighted Dirichlet

8       5.0261137756             158.8923506
16      5.4363195797             609.6912777
32      5.6550082504            2387.8303130
64      5.7676069177            9450.2922550
128     5.8246864141           37599.9570717
256     5.8534157885          149998.2518992        (2.9)
```

Thus Fourier support alone cannot imply any positive-power carrier lower
bound for vertex-weighted Fisher energy.

The shear pressure is zero. Section 2 is an exact weighted-coercivity
no-go, not yet a pressure-load counterexample.

## 3. Zero-face concentration scaling

Let a band-limited packet have:

```text
width delta,

carrier K approximately delta^(-1),

velocity amplitude A.                              (3.1)
```

Center it one packet width from a simple quadratic zero face, so

```text
lambda approximately delta^2,

|grad lambda| approximately delta.                 (3.2)
```

If the packet is localized in `d` coordinate directions, then

```text
pressure load       approximately A^3 delta^(d+1),

weighted Fisher     approximately A^2 delta^d,

unweighted Fisher   approximately A^2 delta^(d-2). (3.3)
```

Normalizing the load to a fixed value `B` requires

```text
A approximately B^(1/3)delta^(-(d+1)/3).           (3.4)
```

The resulting powers are

```text
d   weighted Fisher                    unweighted Fisher

1   B^(2/3) delta^(-1/3)               B^(2/3) delta^(-7/3)

2   B^(2/3)                            B^(2/3) delta^(-2)

3   B^(2/3) delta^(1/3)                B^(2/3) delta^(-5/3). (3.5)
```

Full three-dimensional concentration can therefore make weighted Fisher
decrease while unweighted Fisher diverges.

Crucially, before fixing the load, the adverse ratio is always

```text
pressure load/(nu weighted Fisher)
 approximately A delta/nu
 =A/(nu K).                                         (3.6)
```

It is independent of the number of localized dimensions. The concentration
mechanism defeats carrier-only coercivity, but it does not defeat a theorem
whose threshold is intrinsic:

```text
K>=C A/nu.                                         (3.7)
```

For the fixed-load, fully concentrated family in (3.5),

```text
A/(nu K) approximately delta^(-1/3)/nu -> infinity.
```

It lies on the adverse, high-local-Reynolds side of (3.7).

## 4. An actual pressure-active finite-Fourier pilot

The audit tests whether the Section 3 mechanism survives incompressibility
and the pressure Poisson map.

For `N=2,3,4,5`, it:

1. builds three real carriers with leading wave vectors
   `(4N,0,0)`, `(0,4N,0)`, and `(-4N,-4N,0)`;
2. multiplies them by a peak-one three-dimensional Fejer window centered
   at `(1/N,0,0)`;
3. projects every Fourier coefficient onto its divergence-free plane;
4. computes `p=-R_iR_j(u_i u_j)` spectrally;
5. pairs `p u` with the exact vertex weight
   `Phi_(-,+,+)`;
6. rescales amplitude so the pressure load has magnitude one.

The grid has size `32N`. Field support is below `5N` per coordinate,
pressure support is below `10N`, and the cubic load is below `15N+1`.
Therefore the zero Fourier coefficient used by the mean is alias free.

The resulting rows are:

```text
N   K_min   amplitude   weighted Fisher   unweighted Fisher   U/K

2     7      11.27467       125.36562          864.75058       3.71271

3    10      18.49394       112.33376         1282.47786       4.22856

4    13      26.78695       103.51281         1882.21712       4.69801

5    16      35.86764        96.80227         2612.63892       5.10444. (4.1)
```

The largest relative divergence residual is below `5e-15`. Log-log fits
over these four deliberately small rows give

```text
amplitude             1.2625    predicted  4/3,

weighted Fisher      -0.2815    predicted -1/3,

unweighted Fisher     1.2029    predicted  5/3,

U/K                    0.3475    predicted  1/3.     (4.2)
```

The actual induced pressure load is nonzero before normalization and the
weighted/unweighted separation has the predicted direction. These are
binary64 finite-Fourier diagnostics, not interval enclosures or a proof of
the limiting exponents.

## 5. What changed

Established rigorously:

- pure high-pass fixed-load unweighted enstrophy grows at least linearly
  in carrier;
- the square-factor identity controls `psi_v u` and `u grad psi_v` by
  vertex-weighted Fisher for `K>sqrt(3)m`;
- the earlier quadratic growth remains proved only for the isolated block
  architecture and is not promoted globally;
- support alone gives no carrier coercivity for vertex-weighted Fisher;
- zero-face concentration preserves the dimensionless adverse ratio
  `A/(nu K)`.

Established numerically:

- a strict high-pass, divergence-free finite-Fourier packet and its induced
  pressure realize a nonzero vertex load;
- after fixed-load normalization, weighted Fisher decreases while
  unweighted Fisher and `U/K` increase over four alias-free carriers.

Still open:

- rigorous asymptotics or an interval enclosure for the pressure packet;
- a floor-free intrinsic inequality under `K>=C||u||_infinity/nu`;
- mixed low/high paraproduct terms;
- critical signed control and Navier-Stokes regularity.

## 6. Route decision

Carrier size by itself is the wrong theorem parameter at a partition zero
face. The exact uncertainty packet rules out that route before any pressure
estimate is attempted.

The surviving statement must compare carrier to amplitude. The
square-factor part is now exact; the next target is its pressure-commutator
companion. For the high-output double Riesz operator

```text
T=Q_H R_iR_j,
```

derive an estimate of the form

```text
||psi_v T(u_i u_j)||_2
 <=C||u||_infinity||psi_v u||_2
   +C||u||_infinity K^(-1)||u grad psi_v||_2.       (6.1)
```

Together with

```text
Phi_v=psi_v^2                                      (6.2)
```

and (1.14)-(1.15), this should prove or falsify

```text
pressure load
 -nu mean[Phi_v|grad u|^2]

 <=controlled weight-Fisher or low-frequency remainder               (6.3)
```

for pure high-pass fields satisfying

```text
K>=C||u||_infinity/nu.                             (6.4)
```

Only after (6.3) survives the exact sine packet and the pressure-active
Fejer family should the low/high and low/low pressure paraproducts be added.
