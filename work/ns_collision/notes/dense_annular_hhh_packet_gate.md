# Dense annular HHH packet gate

Status: a unit-energy divergence-free annular packet can feed one fixed low
traceless tensor and top-Walsh channel at the sharp `H^(5/2)` Bernstein
rate. The construction has `O(H^3)` modes and `O(H^6)` coherent HHH
triples. A parabolic lifetime `H^(-2)` leaves its forcing `L2_t` cost of
order `H^3`, while its energy and enstrophy-time cost remain bounded.
Therefore the raw tensor regeneration norm cannot be derived from the two
standard Leray quantities alone. This is a functional-input no-go, not an
unforced Navier-Stokes solution or a blow-up construction.

## 1. Exact central HHH channel

Take the three carrier directions

```text
A=(1,0,0),

B=(-1,1,0),

C=(0,-1,0),                                       (1.1)

A+B+C=0.
```

Use the integer base vectors

```text
v_A=(-4,-3,1),

v_B=(-3,-1,2),

v_C=(-3,7,1).                                     (1.2)
```

Their divergence-free projections are exactly

```text
U_A=P_A v_A=(0,-3,1),

U_B=P_B v_B=(-2,-2,2),

U_C=P_C v_C=(-3,0,1).                             (1.3)
```

For the complete three-leg stress symbol

```text
G(A,B,C)
 =B(A,B) odot U_C
  +B(A,C) odot U_B
  +B(B,C) odot U_A,                               (1.4)
```

direct rational algebra gives

```text
G(A,B,C)=i M,                                     (1.5)

M=
[ 36    0  -30
   0   -36   30
 -30    30    0 ].                                (1.6)
```

The matrix is traceless and

```text
||M||_F=12sqrt(43)=78.689262291624.                (1.7)
```

Assign relative Fourier phases

```text
(phase_A,phase_B,phase_C)=(1,1,i).                (1.8)
```

Their product rotates `iM` to the real matrix `-M`. Define the unit
traceless channel

```text
T=-M/[12sqrt(43)].                                 (1.9)
```

Then the central channel pairing is exactly

```text
<T,G>=12sqrt(43)>0.                               (1.10)
```

This exact integer witness replaces any reliance on fitted random
polarizations.

## 2. Thick annular packet

Let

```text
B_M={-M,...,M}^3                                  (2.1)
```

and choose a carrier `R=K M`, with fixed sufficiently large integer `K`.
Place Fourier modes in

```text
R A+B_M,

R B+B_M,

R C+B_M,                                          (2.2)
```

and their full negative conjugates. At wave `k` in each positive cluster,
use

```text
P_k v_A,          P_k v_B,          i P_k v_C,    (2.3)
```

respectively. A single global coefficient normalizes the Fourier `L2`
energy to one.

Every coefficient is transverse to its own wave, so the packet is exactly
divergence free. It is a finite Fourier sum and therefore smooth. For fixed
large `K`, all modes lie in one bounded-ratio annulus of radius comparable
to `R`.

The only carrier-center triples among the six signed clusters that sum to
zero are

```text
A+B+C=0
```

and its full negative. Mixed sign choices cannot create a hidden competing
channel.

## 3. Coherent lattice count

For offsets `x,y,z in B_M`, a positive carrier triple has zero output
exactly when

```text
x+y+z=0.                                          (3.1)
```

In one dimension, the exact number of pairs `(x,y)` satisfying
`|x+y|<=M` is

```text
3M^2+3M+1.                                        (3.2)
```

The three-dimensional coherent triad count is therefore

```text
N_triad=(3M^2+3M+1)^3=O(M^6).                     (3.3)
```

The real packet has

```text
N_mode=6(2M+1)^3=O(M^3)                           (3.4)
```

modes.

The projected trilinear symbol is continuous in the three normalized wave
directions. By (1.10), there is an `epsilon>0` such that all triples in
sufficiently narrow boxes satisfy

```text
<T,G(a,b,c)>
 >=6sqrt(43) R [polarization factors]              (3.5)
```

with the same sign. Taking `M<=epsilon R/4` gives a constructive coherent
subfamily. Reality doubles the positive contribution rather than
cancelling it because of the phase choice (1.8).

## 4. Sharp scaling theorem

Unit energy distributes coefficients at size

```text
N_mode^(-1/2)=O(M^(-3/2)).                        (4.1)
```

Each HHH symbol contributes one derivative `R`. Combining (3.3)-(4.1),

```text
|<T,G_R(0)>|

 >=c R M^6 M^(-9/2)

 >=c_epsilon R^(5/2).                             (4.2)
```

The matching upper bound is the annular Bernstein estimate:

```text
|G_R(0)|
 <=2||u_R||_2||grad p_R||_2

 <=C||u_R||_2||u_R||_infinity||grad u_R||_2

 <=C R^(5/2)||u_R||_2^3.                          (4.3)
```

Thus

```text
|G_R(0)| asymptotically R^(5/2)                   (4.4)
```

is sharp. The extra `R^(3/2)` beyond the sparse-triad `O(R)` size is
precisely dense mode multiplicity.

## 5. Finite lattice replay

The audit uses `K=32` and exhaustively sums every coherent triple:

```text
M   R    modes   triads    channel forcing

1   32     162      343       23.88655220
2   64     750     6859       96.06839386
3   96    2058    50653      234.25350474
4  128    4374   226981      451.83642500.         (5.1)
```

After division by the exact coherent count and energy-normalization scale,
the channel coefficients are

```text
78.41495, 78.48700, 78.51034, 78.52186,            (5.2)
```

approaching the exact central value `78.68926`.

At `M=4`, the selected channel contains more than `0.999998` of the full
forcing tensor norm. Every one of the `226981` audited triples has positive
channel projection; the minimum normalized projection is `66.85719`.

## 6. Cell/Walsh coupling

The dense stress output is `q=0`. Choose

```text
r=(1,1,1),

k_low=-r,

U_low=(1,-1,0)/sqrt(2).                            (6.1)
```

The low mode is divergence free, and

```text
r^T T U_low=-1/sqrt(86) !=0.                       (6.2)
```

The cubic partition coefficient at `r` is

```text
phihat_v(r)=v_1v_2v_3/64.                         (6.3)
```

Hence the dense tensor obstruction survives in the pure top Walsh
character

```text
chi_123(v)=v_1v_2v_3.                              (6.4)
```

Equal cell weights still cancel, but a nonconstant nonnegative selector can
retain the channel exactly as in the earlier HHL witness.

## 7. Parabolic Leray-input no-go

Let `phi` be a fixed nonzero smooth compactly supported pulse and set

```text
u_R(t)=phi(R^2 t)u_R.                              (7.1)
```

Then

```text
sup_t ||u_R(t)||_2^2<=C,                           (7.2)

integral ||grad u_R(t)||_2^2 dt<=C,                (7.3)
```

because shell enstrophy `R^2` persists for time `R^(-2)`.

The cubic stress regeneration scales as `phi^3G_R`, so (4.2) gives

```text
integral |<T,G_R(t)>|^2 dt

 >=c R^5 R^(-2)

 =c R^3.                                          (7.4)
```

Therefore no universal estimate of the raw form

```text
sum_H ||f_H||_(L2_t)^2

 <=C(sup_t||u||_2^2,
     integral||grad u||_2^2 dt)                   (7.5)
```

can follow from those Leray inputs alone for arbitrary smooth
divergence-free paths.

This does not construct an unforced Navier-Stokes trajectory. Equation
(7.5) might conceivably hold on actual solutions because of temporal
correlation imposed by the equation. It cannot be proved by treating the
energy and enstrophy inequalities as the only inputs.

## 8. Route decision

Established:

- an exact rational nonzero HHH pressure-strain channel;
- a smooth real divergence-free unit-energy annular packet;
- `O(H^3)` modes and `O(H^6)` same-sign triples;
- the sharp `H^(5/2)` tensor-forcing theorem;
- survival in a fixed top-Walsh cell channel;
- failure of a raw tensor forcing bound from Leray inputs alone.

Not established:

- an unforced Navier-Stokes counterexample;
- obstruction of the trace/local-energy channel;
- failure of equation-specific temporal decorrelation;
- critical signed closure or global regularity.

The next object must retain the complete signed scalar local-energy
evolution. Its pressure-strain trace cancels, so it may remove the dense
`H^(5/2)` tensor obstruction. The alternative is a shell-weighted negative
norm that exactly pays the multiplicity without losing the signed flux.

The construction and exhaustive finite lattice replay are generated by
`scripts/dense_annular_hhh_packet_gate_audit.py`.
