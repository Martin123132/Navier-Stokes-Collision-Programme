# Pressure Hamming commutator and the lower-carrier no-go

Status: exact eight-shift multiplier identity, exact Hamming-cube leakage
bound, and a smooth divergence-free two-scale family falsifying the proposed
bandwidth-independent diagonal commutator estimate. The surviving target is
annular and shellwise. No critical signed estimate or Navier-Stokes
regularity conclusion is claimed.

## 1. The question left by the square-factor bridge

For a vertex `v` of the tensor partition, write

```text
Phi_v=psi_v^2.
```

The preceding square-factor argument proved, for a velocity supported above
carrier `K>sqrt(3)m`,

```text
||psi_v u||_2^2
 <=E_v/[K(K-sqrt(3)m)],

||u grad psi_v||_2
 <=gamma(K,m) E_v^(1/2),                           (1.1)

E_v=mean[Phi_v|grad u|^2].
```

This suggested the high-output pressure estimate

```text
||psi_v Q_H R_iR_j(u_i u_j)||_2
 <=C||u||_infinity[
      ||psi_v u||_2
      +K^(-1)||u grad psi_v||_2].                  (1.2)
```

If (1.2) held with only the lower carrier as a spectral hypothesis, then
(1.1) would give a single-vertex intrinsic absorption theorem. This note
shows that (1.2) is false in that scope.

The failure is informative rather than terminal. The exact multiplier
calculus identifies the omitted quantities, and the counterexample explains
which spectral freedom creates them.

## 2. Exact eight-shift multiplier identity

Each one-dimensional factor of `psi_v` is a sine or cosine at frequency
`m/2`. Hence

```text
psi_v(x)
 =sum_(epsilon in {+1,-1}^3)
   c_(v,epsilon) exp(i(m/2)epsilon dot x).          (2.1)
```

Put `h=m/2`. Let `T=M(D)` be a translation-invariant scalar or
matrix-to-scalar multiplier. On the half-lattice,

```text
[psi_v T f]^hat(k)
 =sum_epsilon c_(v,epsilon)
   M(k-h epsilon) fhat(k-h epsilon).                (2.2)
```

For each output frequency, take the Walsh transform of the eight multiplier
values:

```text
A_S(k)
 =2^(-3)sum_epsilon epsilon_S M(k-h epsilon),

epsilon_S=product_(j in S)epsilon_j.                (2.3)
```

Walsh inversion and differentiation of (2.1) give the exact identity

```text
psi_v T f
 =sum_(S subset {1,2,3})
   i^(-|S|)A_S(D)[h^(-|S|) partial_S psi_v f].      (2.4)
```

Differentiating a sine toggles it to a cosine, and conversely. Therefore

```text
partial_S psi_v
 =sigma(v,S)h^|S| psi_(v xor S),                    (2.5)
```

where `v xor S` toggles the vertex signs in `S`. Equation (2.4) becomes

```text
psi_v T f
 =sum_S tau(v,S) A_S(D)[psi_(v xor S) f],           (2.6)

|tau(v,S)|=1.
```

This is the exact pressure commutator geometry. Derivative order is Hamming
distance on the eight-cell cube.

For the concrete double-Riesz symbol

```text
M_12(xi)=-xi_1 xi_2/|xi|^2,
```

the rational audit evaluates (2.3) at
`k=(11/2,13/2,15/2)` and `h=1/2`. Its distance-two and distance-three Walsh
coefficients are nonzero. They are genuine multiplier terms, not artifacts
of a loose Taylor remainder.

## 3. Coupled Hamming bound

For a bounded multiplier,

```text
||A_S||_(2->2)<=sup_xi ||M(xi)||.                  (3.1)
```

If the multiplier is smooth at output scale `H`, iterated central
differences sharpen this to

```text
||A_S||_(2->2)
 <=h^|S| sup_xi||partial_S M(xi)||
 <=L_|S|(m/(2H))^|S|.                              (3.2)
```

Take `f=u tensor u` and use

```text
|u tensor u|_F=|u|^2
 <=||u||_infinity |u|.
```

Equations (2.6)-(3.2) yield

```text
||psi_v T(u tensor u)||_2
 <=||u||_infinity
   sum_w B_(v,w)||psi_w u||_2,                     (3.3)

B_(v,w)
 =L_d(m/(2H))^d,

d=Hamming(v,w).                                    (3.4)
```

The smooth prototype with `L_d=1` is the tensor cube

```text
B=[[1,a],[a,1]] tensor 3,

a=m/(2H),
```

whose exact operator norm and row sum are both

```text
(1+a)^3.                                           (3.5)
```

Without smoothness, the bounded double-Riesz multiplier gives the fallback
row sum `8`. Thus the multiplier is controlled after all eight vertex
weights are retained.

What (3.3) does not provide is a diagonal estimate charged only to `E_v`.
The distance-two and opposite-vertex masses can sit where `Phi_v` and
`|grad psi_v|^2` both vanish.

## 4. Why a lower carrier is not a bandwidth

A condition

```text
uhat(k)=0 for |k|<K                               (4.1)
```

does not prevent concentration on a scale much smaller than `1/K`. A
trigonometric packet may start at mode `K` and extend through modes of size
`N`, with `N/K` arbitrarily large. A one-sided frequency shift makes that
statement exact while retaining spatial concentration.

This distinction did not affect the earlier sine packet, whose lower carrier
and bandwidth were both of order `N`. It is decisive at a triple partition
zero.

## 5. Exact divergence-free two-scale family

Fix the peak-one Fejer factor

```text
F_N(x)
 =[sin(Nx/2)/(N sin(x/2))]^2.                     (5.1)
```

Its Fourier support is `{-(N-1),...,N-1}`. Define

```text
a_N(x)
 =F_N(x_1)F_N(x_2)F_N(x_3)
  cos((N+2)(x_1+x_2+x_3)),                         (5.2)

u_N
 =N^(-1)(partial_2 a_N,-partial_1 a_N,0).          (5.3)
```

Equation (5.3) is a curl, so

```text
div u_N=0                                          (5.4)
```

exactly. The positive Fourier cube of `a_N` has each coordinate in

```text
{3,...,2N+1},                                      (5.5)
```

and the real conjugate occupies the full negative cube. Consequently

```text
uhat_N(k)=0 for |k|<K_0,

K_0=3sqrt(3),                                      (5.6)
```

for every `N`, while

```text
0<c<=||u_N||_infinity<=C.                          (5.7)
```

Thus the lowest carrier and amplitude remain fixed, but the packet width
and maximum frequency tend to infinity.

Use the triple-zero square root

```text
psi(x)
 =sin(x_1/2)sin(x_2/2)sin(x_3/2).                 (5.8)
```

Let `chi_H` be a fixed smooth high-output cutoff, zero for `|xi|<=K_0`
and one for `|xi|>=2K_0`, and put

```text
p_N^H
 =-chi_H(D)R_iR_j(u_(N,i)u_(N,j)).                 (5.9)
```

## 6. Asymptotic no-go

The standard Fejer estimate

```text
F_N(x)
 <=C min(1,[N dist(x,2pi Z)]^(-2))                (6.1)
```

and its differentiated version show that the velocity is concentrated at
scale `1/N`. Near the triple zero,

```text
psi(x)=O(|x|^3),

grad psi(x)=O(|x|^2).                              (6.2)
```

Scaling (6.1)-(6.2) gives

```text
||psi u_N||_2
 <=C N^(-9/2),                                     (6.3)

||u_N grad psi||_2
 <=C N^(-7/2).                                     (6.4)
```

The pressure has a slower nonlocal component. In distributions,

```text
N^3 u_N tensor u_N
 ->A delta_0,                                      (6.5)
```

where `A` is the nonzero positive Reynolds-stress matrix of the rescaled
curl profile. The fixed high-pass pressure kernel is smooth away from zero.
On every compact set avoiding zero,

```text
N^3 p_N^H
 ->-chi_H(D)R_iR_j(A_ij delta_0)                   (6.6)
```

uniformly. The limit is not identically zero: otherwise all retained
Fourier coefficients

```text
chi_H(k) k^T A k/|k|^2
```

would vanish, forcing the positive matrix `A` to be zero. Since the zero
set of `psi` has empty interior, there is an open set on which both `psi`
and the limiting pressure kernel are nonzero. Therefore

```text
||psi p_N^H||_2
 >=c N^(-3).                                       (6.7)
```

Combining (5.7) and (6.3)-(6.7) gives

```text
 ||psi p_N^H||_2
 -----------------------------------------------
 ||u_N||_infinity[
   ||psi u_N||_2+K_0^(-1)||u_N grad psi||_2]

 >=c N^(1/2) ->infinity.                           (6.8)
```

This proves that (1.2) is false under a lower-carrier hypothesis alone.
Amplitude rescaling multiplies both sides of (1.2) by the same quadratic
factor. Hence one may make `||u_N||_infinity` arbitrarily small while
preserving (6.8); the condition

```text
K_0>=C||u_N||_infinity/nu                          (6.9)
```

does not repair the failed estimate.

## 7. Finite-Fourier stress

The audit evaluates (5.1)-(5.9) without sampling its Fourier support:

1. grid size is `10N`;
2. the largest product frequency remains below Nyquist;
3. the curl is formed spectrally;
4. the smooth cutoff transitions from zero at `K_0` to one at `2K_0`;
5. the full induced Poisson pressure is used.

For `N=4,6,8,10,12,14`, the measured lowest velocity mode remains exactly
`3sqrt(3)`, and the relative divergence residual stays below `3e-15`.
The diagonal ratio rises monotonically:

```text
N      ratio
4      0.0040953
6      0.0345588
8      0.0878861
10     0.1529055
12     0.2203505
14     0.2854904.                                  (7.1)
```

Over the same rows, `N^3||psi p_N^H||_2` increases toward its nonzero
asymptotic scale. The finite rows are a binary64 mechanism check; the
counterexample conclusion comes from (5.1)-(6.8), not from extrapolating
the fitted slope.

## 8. What this result does and does not settle

Established:

- the exact eight-shift Walsh identity;
- genuine distance-two and distance-three pressure leakage;
- a coupled eight-cell multiplier bound;
- a smooth, finite-Fourier, divergence-free family falsifying the
  lower-carrier-only diagonal estimate;
- failure of amplitude-relative intrinsic scaling to rescue that estimate.

Not established:

- failure of a bounded-annulus or dyadic-shell diagonal estimate;
- summability of shell interactions;
- control of mixed low/high pressure paraproducts;
- the critical signed terminal dual;
- low-regularity passage or Navier-Stokes regularity.

## 9. Route decision

The next theorem must include an upper bandwidth. For

```text
K<=|k|<=Lambda K,                                  (9.1)
```

triple-zero concentration cannot occur on scales arbitrarily below `1/K`.
The immediate target is a finite-type annular uncertainty estimate of the
form

```text
K^(-2)sum_(d(v,w)=2)||psi_w u||_2
 +K^(-3)||psi_(-v)u||_2

 <=C_Lambda[
   ||psi_v u||_2
   +K^(-1)||u grad psi_v||_2],                     (9.2)
```

with the powers adjusted for the exact multiplier differences. Equation
(9.2) must be proved with a uniform constant, not inferred from finite
matrices.

If it survives, combine it with (2.4), prove the comparable-shell pressure
estimate, and then treat separated shells by paraproduct decay. If it
fails, retain the full Hamming-coupled cell vector and use signed
coefficient differences before taking absolute values.

The result file and finite-Fourier replay are generated by
`scripts/pressure_hamming_commutator_gate_audit.py`.
