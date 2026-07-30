# Quantitative cost of realizing a compatible pressure load

Status: exact scaling and single-block optimization, together with an
explicit high-carrier family. Fixed bad loads have bounded `L2` and bounded
critical `L3` realization cost as carrier frequencies rise, but the
block-optimal velocity Fisher cost grows quadratically and is seen exactly
by every compatible vertex weight. This is a theorem for the constructed
finite-Fourier architecture, not a computation of the global least cost.
No critical pressure estimate or Navier-Stokes regularity conclusion is
claimed.

## 1. Amplitude and partition scaling

Let `b_m(u)` denote the compatible pressure-load vector formed with the
frequency-`m` partition `Phi_(v,m)(x)=Phi_v(mx)`, where `m` is a positive
integer. Under

```text
u_(a,m)(x)=a u(mx),

p_(a,m)(x)=a^2p(mx),                                (1.1)
```

the pressure transport and partition derivative give

```text
b_m[u_(a,m)]=a^3 m b_1[u].                          (1.2)
```

The relevant velocity costs scale as

```text
||u_(a,m)||_2^2
 =a^2||u||_2^2,

||grad u_(a,m)||_2^2
 =a^2m^2||grad u||_2^2,

||u_(a,m)||_3^3
 =a^3||u||_3^3.                                    (1.3)
```

To realize a fixed load of magnitude `B` at partition frequency `m`, take

```text
a=(B/m)^(1/3).
```

Consequently,

```text
L2 squared cost  scales as B^(2/3)m^(-2/3),

H1 squared cost  scales as B^(2/3)m^(4/3),

L3 cubed cost    scales as B/m.                     (1.4)
```

Thus

```text
m||u||_3^3/B                                      (1.5)
```

is the critical dimensionless realization cost.

## 2. Normalized isolated block

Use one block from the surjectivity construction:

```text
a=(N,0,0),

b=(0,2N,0),

k=n-a-b,

A perpendicular to a,
B perpendicular to b,
C perpendicular to k.                              (2.1)
```

Normalize the three polarizations to unit length. If their Fourier
coefficient magnitudes are `x,y,z`, the low pressure transport coefficient
is

```text
gamma=|kappa_N|xyz,                                 (2.2)
```

where `kappa_N` is the normalized geometric coupling. Put

```text
P=gamma/|kappa_N|.
```

The real field includes each mode and its conjugate, so

```text
||u||_2^2=2(x^2+y^2+z^2).                           (2.3)
```

AM-GM gives the exact minimum

```text
min_(xyz=P)||u||_2^2
 =6P^(2/3),                                         (2.4)

x=y=z=P^(1/3).
```

Likewise,

```text
||grad u||_2^2
 =2(|a|^2x^2+|b|^2y^2+|k|^2z^2).                  (2.5)
```

The constrained minimum is

```text
min_(xyz=P)||grad u||_2^2
 =6P^(2/3)(|a||b||k|)^(2/3),                       (2.6)
```

at

```text
|a|x=|b|y=|k|z
 =(P|a||b||k|)^(1/3).                              (2.7)
```

For the `L2` optimizer,

```text
||u||_infinity<=6P^(1/3).
```

On the normalized torus,

```text
||u||_3^3
 <=||u||_infinity||u||_2^2
 <=36P.                                             (2.8)
```

Equation (2.8) is an upper bound sufficient for constructing a bounded
critical-cost sequence. It is not the exact least `L3` cost.

## 3. Normalized coupling at high carrier

The unnormalized third polarization is

```text
C_raw=|k|^2q-(k.q)k,

q=a+b.
```

It points along the projection of `q` onto `k` perpendicular. Since

```text
q=n-k,
```

this is also the projection of the fixed low output `n`. After normalizing
`C_raw`, every coupling `kappa_N` has a finite nonzero limit as

```text
N -> infinity.                                      (3.1)
```

The seven limits are audited symbolically. Therefore a fixed transport
coefficient `gamma` requires bounded coefficient amplitudes as the carrier
rises.

This is the cancellation hidden by the original unnormalized construction,
whose integer `C_raw` made its displayed energy enormous. Polarization
normalization removes that irrelevant coordinate choice.

## 4. Uniform lacunary isolation

Let the seven block scales be

```text
N_S=M 4^index(S),  M>=8.                            (4.1)
```

Each actual mode is

```text
M times an integer leading mode
  plus a correction with coordinates at most one.  (4.2)
```

An exact finite enumeration of the `42` signed leading modes finds only
`14` zero-sum triples: the positive and negative triple from each block.

If a triple has nonzero leading sum, (4.2) and `M>=8` leave at least one
output coordinate of magnitude `5`, so it cannot enter the partition
stencil. If the leading sum is zero, it is one of the certified block
triples and outputs the intended subset wave.

Hence the seven Walsh load coordinates remain independent for every
`M>=8`, not only for the original numerical scales.

## 5. High-carrier realization costs

For each `M`, choose the exact `L2`-optimal amplitudes from (2.4) for the
fixed Hamming-profile load. The nonzero coupling limits imply:

```text
sup_(M large)||u_M||_2^2<infinity,                  (5.1)

sup_(M large)||u_M||_3^3<infinity.                  (5.2)
```

By contrast, (2.6) gives

```text
minimum block H1 cost
 =Theta(M^2).                                       (5.3)
```

Therefore neither `L2` nor critical `L3` realization cost is coercive under
carrier translation. The block `H1` cost is quadratically coercive.

This does not prove that the global least `H1` cost among every possible
velocity architecture grows quadratically. The claim in (5.3) is for the
isolated blocks whose surjectivity and support separation are certified.

## 6. Compatible weights see the Fisher cost

The gradient-energy density has Fourier outputs formed by sums of two
occupied signed velocity modes. A second exact leading-mode enumeration
finds `21` zero pairs, all exact conjugate opposites. Every nonzero leading
pair has, after the bounded correction, an output coordinate of magnitude
at least `M-2>=6`. It therefore finds no nonzero output in

```text
{-1,0,1}^3 minus {0}.                               (6.1)
```

Every vertex basis weight `Phi_v` has mean `1/8` and only the stencil modes
(6.1). Consequently,

```text
mean[Phi_v |grad u_M|^2]
 =1/8||grad u_M||_2^2                              (6.2)
```

for all eight vertices and every `M>=8`.

This matters because a zero face does not hide the high-carrier velocity
Fisher cost. Even the vertex-delta terminal coefficient that extremizes the
cubic graph sees exactly one eighth of the full enstrophy.

Combining (5.3) and (6.2), the negative velocity-Fisher contribution
eventually dominates the fixed pressure-load number along this explicit
family for any fixed positive viscosity. This is not yet an absorption
theorem for arbitrary fields or for the full optimized functional.

## 7. What changed

Established:

- exact load, `L2`, `H1`, and `L3` scaling;
- exact single-block `L2` and `H1` minima;
- nonzero normalized coupling limits;
- a uniformly isolated high-carrier load realization family;
- bounded `L2` and bounded critical `L3` upper cost;
- quadratic block-Fisher growth;
- exact one-eighth visibility through every vertex weight.

Still open:

- the global least realization cost beyond the block architecture;
- a uniform high-carrier absorption inequality for arbitrary compatible
  coefficient vectors and arbitrary velocity fields;
- the carrier band comparable to the partition frequency;
- low-regularity passage and global regularity.

## 8. Route decision

The pressure route should not discard velocity Fisher before optimizing the
compatible graph. Critical `L3` size alone allows bad load directions at
arbitrarily high carriers, but the retained Fisher term makes those same
realizations increasingly expensive.

The live obstruction is therefore concentrated near

```text
velocity carrier frequency
 comparable to partition frequency.                (8.1)
```

This converts an unbounded spectral search into a possible compact-ratio
problem, provided a general high-carrier absorption theorem can replace the
explicit block calculation.

## 9. Next theorem target

Derive, for arbitrary smooth divergence-free `u` and compatible
nonnegative coefficient vector `w`, a frequency-separated inequality of the
form

```text
high-carrier pressure load
 <=epsilon nu mean[lambda_w|grad u|^2]
   +controlled coefficient/low-frequency terms.    (9.1)
```

It must:

1. remain valid when vertex coefficients vanish;
2. retain the exact compatible graph projection;
3. use the intrinsic partition scale and pressure-tail decomposition;
4. give an explicit carrier threshold;
5. leave only a bounded carrier-ratio band for the finite sharp problem.
