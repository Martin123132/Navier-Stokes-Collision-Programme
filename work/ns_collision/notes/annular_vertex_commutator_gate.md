# Annular vertex commutator gate

Status: a sharp residue-chain uncertainty theorem closes the smooth
high-output pressure commutator of one annular velocity shell at a single
partition vertex. This does not control the low-output high-high beat,
cross-shell paraproducts, a sharp output cutoff, or the full Navier-Stokes
pressure load. No regularity conclusion is claimed.

## 1. The obstruction being repaired

For a tensor partition vertex `v`, write

```text
Phi_v=psi_v^2,

psi_v(x)=product_(j=1)^3 phi_(v_j)(x_j),
```

where each `phi_(v_j)` is a sine or cosine at half-frequency `m/2`.
The previous pressure-Hamming audit proved the exact multiplier identity

```text
psi_v T f
 =sum_(S subset {1,2,3})
   tau_(v,S) A_S(D)[psi_(v xor S)f],               (1.1)

|tau_(v,S)|=1.
```

For a smooth multiplier at output scale `H`,

```text
||A_S||_(2->2)
 <=L_|S| (m/(2H))^|S|.                            (1.2)
```

The difficulty was that (1.1) involves every Hamming neighbor of `v`.
A lower carrier alone cannot compare those vertex masses: a packet with
unbounded bandwidth can concentrate arbitrarily sharply at the triple zero
of `psi_v`.

This note adds the missing upper bandwidth. Assume

```text
supp uhat subset {k in Z^3: K<=|k|<=Lambda K}.     (1.3)
```

The finite coordinate degree forced by (1.3) yields an exact comparison of
all eight vertex masses.

## 2. One-dimensional residue chains

Let

```text
f(x)=sum_(|n|<=L) a_n exp(inx),                    (2.1)
```

with `m` a positive integer. Multiplication by `sin(mx/2)` or
`cos(mx/2)` shifts an input coefficient to `n-m/2` and `n+m/2`.
Two shifted outputs overlap exactly when their input indices differ by
`m`. Thus the coefficients split orthogonally into residue chains modulo
`m`.

Order one such chain as

```text
n_1<n_2<...<n_N,  n_(j+1)-n_j=m.                  (2.2)
```

Ignoring phases of modulus one, the cosine output is the zero-boundary sum
matrix `S_N/2`, while the sine output is the zero-boundary difference matrix
`D_N/2`:

```text
S_N a=(a_1,a_1+a_2,...,a_(N-1)+a_N,a_N),

D_N a=(a_1,a_2-a_1,...,a_N-a_(N-1),-a_N).         (2.3)
```

The common Dirichlet sine eigenvectors

```text
a_j^(q)=sin(j q pi/(N+1)),  q=1,...,N             (2.4)
```

diagonalize both quadratic forms. Their eigenvalues, with the common
normalization suppressed, are

```text
lambda_D(q)=1-cos(q pi/(N+1)),

lambda_S(q)=1+cos(q pi/(N+1)).                    (2.5)
```

Consequently

```text
sup_(a!=0) ||S_N a||/||D_N a||
 =cot(pi/[2(N+1)]),                               (2.6)

sup_(a!=0) ||D_N a||/||S_N a||
 =cot(pi/[2(N+1)]).                               (2.7)
```

The first eigenvector attains (2.6), and the last attains (2.7).

Among the `2L+1` consecutive input frequencies, the longest residue chain
has length

```text
N_max=ceil((2L+1)/m).                             (2.8)
```

Since the cotangent in (2.6) increases with `N`, orthogonality of the
chains proves the sharp two-sided toggle theorem

```text
||cos(mx/2)f||_2
 <=C_(L,m)||sin(mx/2)f||_2,

||sin(mx/2)f||_2
 <=C_(L,m)||cos(mx/2)f||_2,                       (2.9)

C_(L,m)=cot(pi/[2(N_max+1)]).                     (2.10)
```

This argument also covers odd `m`: the shifted modes then lie on the
half-integer lattice, whose pairwise frequency differences remain integers
and hence orthogonal on a `2pi` interval.

## 3. Tensor Hamming collapse

Suppose a scalar, vector, or tensor field `g` on `T^3` has coordinate
Fourier degree at most `L`. If vertices `v` and `w` differ in one
coordinate, apply (2.9) in that coordinate while treating the other two
coordinates and the value space as a Hilbert-valued coefficient. Factors
of `psi_v` in the other coordinates do not change the active residue
chains.

Iterating along a shortest path on the vertex cube gives

```text
||psi_w g||_2
 <=C_(L,m)^d ||psi_v g||_2,                       (3.1)

d=Hamming(v,w).
```

The exponent is exact: tensor products of the one-dimensional extremizers
attain `C_(L,m)^d`.

## 4. Insertion into the Walsh multiplier identity

Let `T_H` have a matrix-to-scalar multiplier `M_H` satisfying

```text
sup_xi ||partial_S M_H(xi)||
 <=L_|S| H^(-|S|),  |S|<=3,                       (4.1)
```

where `partial_S` differentiates once in every coordinate in `S`.
A smooth high-output cutoff multiplying the double-Riesz symbol has these
bounds. A sharp cutoff does not.

Take `f=u tensor u`. Pointwise,

```text
|u tensor u|_F=|u|^2
 <=||u||_infinity |u|.                            (4.2)
```

Combining (1.1), (1.2), (3.1), and (4.2) yields

```text
||psi_v T_H(u tensor u)||_2
 <=C_ann ||u||_infinity ||psi_v u||_2,             (4.3)

C_ann
 =L_0+3L_1 theta+3L_2 theta^2+L_3 theta^3,        (4.4)

theta=(m/(2H))C_(L,m).                            (4.5)
```

Thus all distance-two and distance-three leakage is returned to the
original vertex without introducing a positive floor for `Phi_v`.

## 5. Uniformity on an annulus

Under (1.3), every coordinate frequency obeys

```text
|k_j|<=L,  L=floor(Lambda K).                     (5.1)
```

The elementary cotangent bound

```text
cot(pi/[2(N+1)])<2(N+1)/pi                        (5.2)
```

and `ceil x<=x+1` imply

```text
theta
 <[2L+1+2m]/(pi H).                               (5.3)
```

Set `H=K`. If `K>sqrt(3)m`, then, for integer `m>=1`,
`1+2m<2K`. Therefore

```text
theta<=2(Lambda+1)/pi.                            (5.4)
```

For a dyadic annulus, `Lambda=2`, this is

```text
theta<=6/pi<2.                                    (5.5)
```

The factor `C_(L,m)` itself grows like `K/m`, but the multiplier difference
contributes the reciprocal factor `m/K`. Their product is uniformly
bounded. This cancellation is the annular finite-type mechanism.

## 6. Single-vertex pressure absorption

Let

```text
E_v=mean[Phi_v |grad u|^2].
```

The established square-factor identities give, for `K>sqrt(3)m`,

```text
||psi_v u||_2
 <=E_v^(1/2)/sqrt(K(K-sqrt(3)m)),                 (6.1)

||u grad psi_v||_2
 <=gamma(K,m)E_v^(1/2),                           (6.2)

gamma(K,m)
 =1+sqrt(1+(3m^2/4)/[K(K-sqrt(3)m)]).             (6.3)
```

For the smooth high-output self-shell pressure

```text
p_H=T_H(u tensor u),
```

use `grad Phi_v=2psi_v grad psi_v`, then (4.3), (6.1), and
(6.2):

```text
|mean[p_H u dot grad Phi_v]|
 <=2||psi_v p_H||_2||u grad psi_v||_2

 <=[2 gamma C_ann ||u||_infinity
    /sqrt(K(K-sqrt(3)m))] E_v.                    (6.4)
```

Hence this pressure component is absorbed by `nu E_v` whenever

```text
nu
 >=2 gamma C_ann ||u||_infinity
   /sqrt(K(K-sqrt(3)m)).                           (6.5)
```

For the convenient threshold `K>=2sqrt(3)m`,

```text
sqrt(K(K-sqrt(3)m))>=K/sqrt(2),

gamma<=1+3sqrt(2)/4.                              (6.6)
```

Thus (6.5) follows from an explicit intrinsic carrier condition
`K>=C_Lambda||u||_infinity/nu`, together with
`K>=2sqrt(3)m`.

## 7. Shellized counterexample stress

The numerical audit also shifts the previous curl-Fejer packet farther into
the positive octant. At order `N`, every positive velocity coordinate lies
between `2N+1` and `4N-1`; the conjugate packet occupies the negative
octant. Therefore

```text
K_min=sqrt(3)(2N+1),

K_max=sqrt(3)(4N-1),

K_max/K_min<2.                                    (7.1)
```

Alias-free grids of size `18N` were used for `N=3,...,7`. The relative
divergence residual stays below `2.4e-15`. The discarded diagonal ratio
from the broad-band no-go is now

```text
N      ratio
3      0.000395606
4      0.000372754
5      0.000362375
6      0.000356704
7      0.000353218.                               (7.2)
```

It decreases rather than growing, with total variation factor below
`1.121`. This finite-Fourier calculation is a stress test of the mechanism,
not the proof; the proof is (2.1)-(6.5).

## 8. What is established

Established in this stage:

- the sharp one-dimensional sine/cosine toggle inequality for arbitrary
  coordinate degree and partition frequency;
- its exact tensor Hamming-distance extension;
- a uniform, floor-free diagonal commutator estimate for one annular shell
  and a smooth high-output pressure multiplier;
- an explicit single-vertex absorption condition for that pressure
  component;
- a shellized replay showing that the prior broad-band counterexample no
  longer grows.

Not established:

- a corresponding theorem for a sharp high-output cutoff;
- control of the low-output pressure generated by opposite high carriers;
- comparable-shell or separated-shell paraproduct summation;
- control of the full pressure of a multi-shell velocity;
- the terminal signed critical estimate;
- low-regularity passage, exceptional-set removal, or global regularity.

## 9. Next gate

For a single shell `u_K`, decompose

```text
R_iR_j(u_(K,i)u_(K,j))
 =P_(<K)[...] + P_(>=K)[...].                     (9.1)
```

The second term is covered by this note when the output projection is
smooth. The next obstruction is the first term: the low-output high-high
beat created by nearly opposite carriers.

The next stage should retain the signed eight-cell load before taking
absolute values and test whether incompressibility, opposite-carrier
geometry, and the zero-sum pressure-load identity provide a summable gain.
Only after the self-shell low output is settled should separated shell
pairs and mixed paraproducts be added.

The exact certificate and finite-Fourier replay are generated by
`scripts/annular_vertex_commutator_gate_audit.py`.
