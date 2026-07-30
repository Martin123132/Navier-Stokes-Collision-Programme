# Dense low-output HHH block gate

Status: certified. The outward-rounded interval audit is strictly positive,
the exact count replay passes, and all six focused tests pass. This is an
instantaneous spatial theorem, not a temporal persistence or regularity
theorem.

## 1. Question

The dense annular packet already gives a fixed traceless tensor channel of
size `H^(5/2)` at the single output `q=0`. The earlier channelwise envelope
analysis asked whether a positive fraction of the `O(H^3)` low Fourier
outputs can be populated at the same scale. The direct physical-space theorem
in `direct_h_minus_one_stress_tail_gate.md` now closes the actual
`H^(-1)` stress tail independently, so the present calculation is retained
as a sharp spatial diagnostic rather than an endpoint obstruction.

This gate asks only the spatial question:

```text
Can one unit-energy divergence-free packet produce

  <T,G_HHH(4q)> >= c H^(5/2)

for every q in an O(M)-wide output cube?                         (1.1)
```

It does not assume that this forcing persists for a parabolic time along an
unforced Navier-Stokes trajectory.

## 2. Packet and output variable

Use the certified carrier directions, base vectors, and phases

```text
A=(1,0,0),       v_A=(-4,-3,1),       phase 1,

B=(-1,1,0),      v_B=(-3,-1,2),       phase 1,

C=(0,-1,0),      v_C=(-3,7,1),        phase i.      (2.1)
```

Let

```text
R=16384M

k_A=RA+4n_A,
k_B=RB+4n_B,
k_C=RC+4n_C,                                  (2.2)
```

with every `n_j` in `[-M,M]^3`, together with the complete negative
conjugate packet. At each wave use the exact Leray projection of its base
vector and one global coefficient that normalizes the real field to unit
Fourier `L2` energy.

For an output index `q`, impose

```text
n_A+n_B+n_C=q.                                  (2.3)
```

Then the physical HHH output is `4q`. Introduce

```text
x=n_A/M,   y=n_B/M,   w=q/M,   z=w-x-y.          (2.4)
```

After division by `R`, the high waves are

```text
a=A+x/4096,
b=B+y/4096,
c=C+z/4096,                                     (2.5)
```

and their sum is `w/4096`.

## 3. Exact output multiplicity

In one dimension, for `|q|<=M`, the number of triples in
`[-M,M]^3` with sum `q` is exactly

```text
N_M(q)=3M^2+3M+1-q^2.                            (3.1)
```

This follows either from bounded-composition inclusion-exclusion or by
taking the coefficient of the corresponding Laurent polynomial. In three
dimensions the coordinates separate:

```text
N_M^(3)(q)=product_(j=1)^3 N_M(q_j).             (3.2)
```

Therefore, throughout the interior cube

```text
|q_j|<=floor(M/2),                               (3.3)
```

every output has at least

```text
[(11/4)M^2]^3=(11/4)^3 M^6                      (3.4)
```

coherent positive-cluster triples. The block contains

```text
(2 floor(M/2)+1)^3=O(M^3)                        (3.5)
```

outputs. Their physical frequencies lie on `4 Z^3`, a fixed-density
sublattice, so the block still has positive three-dimensional Fourier
volume.

## 4. Fixed tensor channel

At the carrier centre, the phased HHH symbol is the real matrix `-M_0`,
where

```text
M_0=
[ 36    0  -30
   0   -36   30
 -30    30    0 ],                               (4.1)

||M_0||_F=12sqrt(43).
```

Fix

```text
T=-M_0/[12sqrt(43)].                              (4.2)
```

For arbitrary normalized waves `(a,b,c)`, the complete projected HHH symbol
contains the three rows

```text
P_(a+b)[(U_a dot b)U_b+(U_b dot a)U_a] odot U_c,

P_(a+c)[(U_a dot c)U_c+(U_c dot a)U_a] odot U_b,

P_(b+c)[(U_b dot c)U_c+(U_c dot b)U_b] odot U_a. (4.3)
```

The phases in (2.1) turn all three rows into real contributions with the
same centre sign. The audit contracts (4.3) with the integer numerator
`-M_0` before dividing by `12sqrt(43)`.

## 5. Directed continuum enclosure

For the output block, the true normalized domain is

```text
x,y,z in [-1,1]^3,
w in [-1/2,1/2]^3,
z=w-x-y.                                         (5.1)
```

The interval calculation uses the larger box

```text
x,y in [-1,1]^3,
w in [-1/2,1/2]^3,
z in [-5/2,5/2]^3,                               (5.2)
```

while retaining the exact pair-output identities

```text
a+b=-C+(x+y)/4096,
a+c=-B+(w-y)/4096,
b+c=-A+(w-x)/4096.                               (5.3)
```

Every elementary binary64 endpoint is expanded by one ulp in the required
direction. Production returns the fixed-channel interval

```text
<T,G_HHH(a,b,c)>/R
 in [76.62863792138288,80.75262600052086].         (5.4)
```

Because (5.2) contains (5.1), strict positivity of `c_-` certifies every
lattice triple in every packet `M>=1`; finite sampling is not used as the
proof.

The all-negative conjugate branch at output `4q` is the conjugate of the
positive branch at `-4q`. The certified positive symbols are real, so the
two branches add. Mixed carrier signs cannot enter this block because their
carrier-centre sum is nonzero and `R` is much larger than the offset width.

## 6. Uniform scaling

The squared norms of the three base vectors sum to

```text
26+14+59=99.                                     (6.1)
```

Leray projection is contractive, so the real packet energy before
normalization is at most

```text
198(2M+1)^3.                                     (6.2)
```

If `alpha_M` is the unit-energy coefficient, then

```text
alpha_M >=[198(2M+1)^3]^(-1/2).                  (6.3)
```

Combining two conjugate branches, the interval lower bound, (3.4), and
`R=16384M` gives

```text
<T,G_HHH(4q)>
 >=2 alpha_M^3 R c_- (11/4)^3 M^6
 >=c_block R^(5/2)                               (6.4)
```

uniformly for every output in (3.3), with an explicit `c_block>0` recorded
by the audit.

Thus one unit-energy smooth divergence-free packet simultaneously reaches
the sharp per-channel `H^(5/2)` scale on `O(H^3)` low outputs.

## 7. What this settles

The three-dimensional output multiplicity in the old channelwise envelope is
not merely an abstract array: the exact projected Navier-Stokes nonlinearity
realizes it instantaneously. The complete stress nevertheless obeys the
trajectory-level direct `L2_t H_x^(-1)` tail theorem, so the dense derivative
must be compensated by its subsequent evolution.

It does not establish:

- persistence of the same signs and amplitudes for time `H^(-2)`;
- an endpoint failure for an actual unforced Navier-Stokes solution;
- a suitable-weak defect or a singular solution;
- finite-time blow-up or global regularity.

Temporal evolution of this packet remains a useful diagnostic, but it cannot
overturn the direct stress-tail theorem. The collision programme's proof
obligation instead returns to the signed time-integrated triad measure and
its quartic transfer/palinstrophy balance in
`collision_defect_dynamics.md`. Any use of the dense packet there must retain
the exact nonlinear deformation rather than insert parabolic persistence as
a closure assumption.

The deterministic certificate and finite lattice replay are implemented in
`scripts/dense_low_output_block_gate_audit.py`. The production result is
`results/dense_low_output_block_gate_audit_v1.json`.
