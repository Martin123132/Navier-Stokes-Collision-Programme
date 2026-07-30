# Joint primitive HHL incidence-Schur gate

## 1. Question

The isolated primitive-chain theorem controls one residue chain at a time:

```text
|B_chain| <= C_q (|Uhat|/m) E_lambda,chain.
```

It does not permit summing these estimates over primitive steps or residue
chains. The same signed Fisher edges can occur in several decompositions, and
charging them separately repeats energy which exists only once in the
physical field.

This gate therefore performs the missing finite assembly. It includes every
cube low wave, both low polarizations and phases, both high polarizations,
and all resonant high-mode differences in one matrix calculation. The
physical weighted Fisher matrix is formed once.

The outcome is fail-closed:

- the finite joint assembly and its direct Fourier reconstruction pass;
- pressure-dominated normalized witnesses grow strongly as transverse
  residue windows widen and as three-dimensional slabs lengthen;
- no window-uniform Schur bound is claimed;
- no asymptotic divergence theorem is claimed.

## 2. Fourier geometry

Use the single compatible tensor vertex

```text
lambda(x)
 = product_(j=1)^3 (1+cos x_j)/2.
```

Its Fourier support is the cube `{-1,0,1}^3`. For a real divergence-free
high field,

```text
hhat(k)
 = sum_(a=1)^2 c_(k,a) p_(k,a)/|k|,
```

where `p_(k,1),p_(k,2)` are a deterministic real orthonormal frame in
`k-perp`. The division by `|k|` makes the complex coordinates
gradient-normalized.

The high windows are

```text
K(H,L,R_y,R_z)
 = {H,...,H+L-1} x {-R_y,...,R_y} x {-R_z,...,R_z},
```

with `H=8`. This carrier excludes same-sign high-high aliasing into the
vertex cube.

The low basis contains:

- 13 cube waves modulo sign;
- two real transverse polarizations per wave;
- cosine and sine Fourier phases.

Thus all real low fields supported in the cube are represented by 52 real
coordinates.

## 3. Complete HHL blocks

For signed high legs `(a,h_a)`, `(b,h_b)` and a low leg `(s,U_s)`, one
ordered complete HHL symbol is

```text
T(a,b;s)
 = (h_a dot h_b) U_s/2
 + (U_s dot h_a) h_b
 + p[h_a,h_b] U_s
 + 2 p[U_s,h_a] h_b,
```

where

```text
p[v_a,w_b]
 = -[((a+b) dot v_a)((a+b) dot w_b)]/|a+b|^2
```

when `a+b` is nonzero, and is zero at zero frequency.

For each low coordinate `j`, both ordered opposite-sign high pairs and both
low signs are summed before pairing the output with `grad lambda`. This
gives a Hermitian matrix `A_j` such that

```text
B_j(h) = c* A_j c.
```

The production code assembles only resonant triples. It does not recover the
matrix by repeatedly evaluating a full cubic polynomial.

## 4. One physical Fisher matrix

The shared Fisher form is

```text
E_lambda(h) = c* G c,
```

with

```text
G_(i,j)
 = 2 lambdahat(k_i-k_j)
     (k_i dot k_j)
     (v_i dot v_j).
```

Here `v_i=p_i/|k_i|`. Positive and negative real-field modes account for the
factor two.

After diagonalizing `G`, only eigenvalues below the roundoff-scaled exact
null tolerance are quotiented. If `W=G^(-1/2)` on the retained range, define

```text
T_j = W* A_j W.
```

Two different diagnostics are retained:

```text
R_lower
 = max_(||z||=1) [sum_j (z* T_j z)^2]^(1/2),
```

computed by deterministic alternating Hermitian eigensteps, and

```text
R_upper
 = ||sum_j T_j^2||_op^(1/2).
```

The first is a physical lower witness: it supplies a high field with unit
Fisher energy and a low coordinate vector of Euclidean norm one. The second
is a valid square-function Schur upper bound. Neither finite quantity proves
uniform behavior as the window grows.

## 5. Independent reconstruction

The matrix construction is checked against the earlier dictionary-based
Fourier implementation in two ways.

For random high coefficients:

```text
c* G c
```

is compared with the direct signed Fourier Fisher sum, and every low
coordinate in the smallest window is compared with the complete component
flux

```text
(|h|^2/2)U
+(U dot h)h
+p[h,h]U
+(p[U,h]+p[h,U])h.
```

Each maximizing joint witness is then rebuilt as an actual real Fourier
high field and an actual real 52-coordinate low field. Its matrix load is
compared with both the component expansion and the direct cubic
polarization identity.

Across all eight windows:

```text
maximum matrix/direct witness-load residual < 6.7e-16,
maximum component/direct flux residual       < 7.2e-14.
```

The Fisher energies equal one and the low-coordinate norms equal one to
roundoff.

## 6. Spectral rows

```text
window       dim(G)  min eig(G)   Schur upper  joint lower
-----------  ------  -----------  -----------  -----------
axial_4           8  4.7746e-2      0.034197     0.026690
axial_8          16  1.5077e-2      0.038985     0.032168
strip_4x3        24  1.4181e-2      0.186729     0.166447
strip_4x5        40  6.6333e-3      0.367897     0.338844
slab_4x3x3       72  4.2683e-3      0.564609     0.389681
slab_4x5x3      120  1.9953e-3      1.212614     0.867563
slab_6x3x3      108  2.2004e-3      0.840629     0.607972
slab_8x3x3      144  1.3341e-3      1.137486     0.840068
```

The directional comparisons are more informative than the largest-to-
smallest ratio:

```text
axial length 8 / length 4 lower ratio     = 1.20524
strip width 5 / width 3 lower ratio       = 2.03575
slab width 5 / width 3 lower ratio        = 2.22634
```

For slab lengths `4,6,8`, the lower witnesses are

```text
0.3896807, 0.6079718, 0.8400677.
```

A descriptive linear fit over only these three finite rows has slope
`0.1125968` and `R^2=0.999687`. This is route evidence, not an asymptotic
fit theorem.

## 7. Pressure localization

The direct maximizing witnesses separate into kinetic, high-high pressure,
and cross-pressure loads:

```text
window       complete   kinetic    p[h,h]U   cross p
-----------  ---------  ---------  ---------  ----------
axial_4       0.026690   0.026683   0          0.000007
strip_4x3     0.166447   0.014228   0.152906  -0.000687
strip_4x5     0.338844   0.010416   0.329782  -0.001354
slab_4x3x3    0.389681   0.013217   0.377240  -0.000776
slab_4x5x3    0.867563   0.010403   0.858554  -0.001393
slab_6x3x3    0.607972   0.015062   0.593641  -0.000732
slab_8x3x3    0.840068   0.014962   0.825764  -0.000659
```

The axial rows remain pressure-free and nearly saturated, in agreement with
the isolated-chain Hardy theorem. The transverse and slab witnesses are
instead 92 to 99 percent high-high pressure. The new signal is therefore a
joint transverse pressure recombination mechanism, not a contradiction of
the one-chain estimate.

## 8. What is and is not closed

Closed:

- all 52 real low coordinates are assembled simultaneously;
- both high polarizations and every resonant difference in each finite
  window are included;
- the physical Fisher form is charged once;
- its numerical nullspace is handled explicitly;
- every finite matrix and maximizing witness has an independent direct
  Fourier replay;
- strong pressure-dominated finite-window growth is established.

Open:

- an explicit family indexed by window size;
- an analytic formula for its Fisher energy;
- an analytic formula or lower bound for its high-high pressure HHL load;
- proof that the normalized ratio diverges;
- proof of a hidden cancellation which instead restores a uniform bound;
- absorption of all cross-shell HHL terms;
- terminal dual control, critical `L^3`, and global regularity.

The next stage should not attempt a graph-coloring upper bound first. It
should extract the approximately separable Dirichlet profile visible in the
slab witnesses and compute both sides analytically. A divergent ratio would
close the proposed joint-Schur route as a rigorous no-go. A cancellation
would identify exactly what the finite optimizer is missing.

The deterministic audit is
`scripts/joint_primitive_hhl_incidence_schur_gate_audit.py`; its production
result is
`results/joint_primitive_hhl_incidence_schur_gate_audit_v1.json`.
