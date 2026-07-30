# Separable annular pressure-Schur no-go

## 1. Result

The finite joint incidence-Schur gate found pressure-dominated slab
witnesses, but it did not establish their asymptotic behavior. This stage
extracts an explicit divergence-free tensor family and proves the
asymptotic result.

For one fixed compatible tensor vertex `lambda` and one fixed low Fourier
wave `U`, there are real high fields `h_N`, each supported in a single
annulus of uniformly bounded shell ratio, for which

```text
E_lambda(h_N) = O(N^-3),

B_complete_HHL(h_N,U)
  = -c_* N + o(N),        c_*>0.
```

Consequently,

```text
|B_complete_HHL(h_N,U)| / E_lambda(h_N) >= c N^4
```

for all sufficiently large odd `N`. There is therefore no constant
independent of `N` in the proposed static estimate

```text
|B_complete_HHL(h,U)| <= C ||Uhat|| E_lambda(h).
```

This is an analytic no-go theorem for that estimate. It is not a
Navier-Stokes regularity theorem.

## 2. Explicit family

Let `N>=3` be odd and let `1<=a,b,c<=N`. Define

```text
k_abc
 = (2N+a-1, b-(N+1)/2, c-(N+1)/2),

alpha_abc
 = (-1)^(a+b+c)
   sin(pi a/(N+1))
   sin(pi b/(N+1))
   sin(pi c/(N+1)).
```

With `P_k` the orthogonal projection onto `k-perp`, set

```text
hhat_N(k_abc)
 = alpha_abc P_k e_3 / |k_abc|,

hhat_N(-k_abc)
 = conjugate(hhat_N(k_abc)).
```

The coefficients are real, the field satisfies the Fourier reality
condition, and

```text
k dot hhat_N(k)=0.
```

All positive modes satisfy

```text
2N <= |k| < sqrt(19/2) N.
```

Thus every `h_N` occupies one annulus with shell ratio below
`sqrt(19/8)<2`. Same-sign high pairs cannot alias into the vertex cube.

Fix

```text
lambda(x)=product_(j=1)^3 (1+cos x_j)/2,

ell=(0,1,-1),

Uhat(ell)=-i(e_2+e_3)/sqrt(2),

Uhat(-ell)=conjugate(Uhat(ell)).
```

The low field is real, divergence-free, and independent of `N`.

## 3. Exact Fisher identity

Write

```text
D_k = k tensor hhat_N(k).
```

The positive-frequency Fisher form has the three-dimensional tensor
Toeplitz kernel generated in each coordinate by

```text
diag = 1/2,     nearest neighbor = 1/4.
```

Conjugating by the alternating sign in `alpha_abc` changes the
nearest-neighbor entry to `-1/4`. In one dimension, with zero Dirichlet
extension,

```text
sum_i (1/2)f_i^2 - sum_i (1/2)f_i f_(i+1)
 = (1/4) sum_(i=0)^N (f_(i+1)-f_i)^2.
```

Tensoring this identity and including the negative-frequency reality copy
gives the exact formula

```text
E_lambda(h_N)
 = (1/32) sum_(a,b,c=0)^N
   ||Delta_a Delta_b Delta_c F_N(a,b,c)||_F^2,
```

where

```text
F_N(a,b,c)=(-1)^(a+b+c)D_(k_abc).
```

This is not an asymptotic approximation.

At the grid points `t=(a,b,c)/(N+1)`, the gauged tensor is the restriction
of

```text
f_N(t)
 = sin(pi t_1) sin(pi t_2) sin(pi t_3) M(Xi_N(t)),

M(xi)=xi tensor P_xi(e_3)/|xi|,

Xi_N(t)
 = (
     2-1/N+(1+1/N)t_1,
     (1+1/N)(t_2-1/2),
     (1+1/N)(t_3-1/2)
   ).
```

The sine factors give the zero boundary extension. A mixed finite
difference is the integral of `partial_123 f_N` over its grid cell.
Cauchy-Schwarz on each cell therefore gives

```text
sum ||Delta_a Delta_b Delta_c F_N||_F^2
 <= (N+1)^-3 integral_[0,1]^3 ||partial_123 f_N||_F^2.
```

For `N>=3`, every `Xi_N([0,1]^3)` lies in a common compact set separated
from the origin. The function `M` is smooth there, so the derivative
integrals have one finite uniform upper bound. Hence

```text
E_lambda(h_N) <= C_F (N+1)^-3.
```

## 4. Pressure limit

Only finitely many high-high differences can be seen by the fixed low
wave and vertex gradient. For each nonzero such `q`,

```text
phat_N(q)
 = -(2/|q|^2)
   sum_(k,k-q in K_N)
   (q dot hhat_N(k))(q dot hhat_N(k-q)).
```

Put

```text
D = [2,3] x [-1/2,1/2]^2,

S(x,y,z)
 = sin(pi(x-2)) sin(pi(y+1/2)) sin(pi(z+1/2)),

V(xi)=P_xi(e_3)/|xi|.
```

The shifted products are ordinary Riemann sums. The boundary sine factors
remove the edge discrepancy, and

```text
phat_N(q)/N
 -> -2(-1)^(q_1+q_2+q_3)/|q|^2
    integral_D S^2(q dot V)^2.
```

The complete exact sum over the two low signs and every compatible vertex
output reduces over `Q(sqrt(2))` to

```text
A
 = sum_q L_q [-2(-1)^sum(q)/|q|^2] q q^T
 = diag(0,sqrt(2)/20,-sqrt(2)/20).
```

It follows that

```text
B_pressure_HH(h_N,U)/N
 -> (sqrt(2)/20)
    integral_D S^2(V_y^2-V_z^2).
```

For `r^2=x^2+y^2+z^2`,

```text
V_y = -zy/r^3,

V_z = (x^2+y^2)/r^3.
```

The sign is pointwise, not numerical:

```text
V_z^2-V_y^2
 = ((x^2+y^2)^2-y^2 z^2)/r^6
 >= (16-1/16)/(19/2)^3
 = 255/13718.
```

Also,

```text
integral_D S^2 = 1/8.
```

Therefore

```text
lim_(N->infinity) B_pressure_HH(h_N,U)/N
 <= -51 sqrt(2)/438976
 < 0.
```

The production Gauss-Legendre replay gives the sharper corroborative value

```text
-0.00140659193857883.
```

The quadrature is not used to prove the sign or its margin.

## 5. Complete HHL flux

The no-go concerns the complete HHL flux, not an isolated pressure term.
Two remaining estimates are needed.

For the kinetic terms, division by `N` turns each finite resonant sum into
a Riemann sum. The pointwise leading quadratic matrix, summed exactly over
both low signs and all vertex outputs, is the zero matrix:

```text
K_lead = 0_(3x3).
```

Thus

```text
B_kinetic(h_N,U)=o(N).
```

For a cross-pressure coefficient, divergence-free cancellation gives

```text
p[U_s,h_k]
 = -[(k dot U_s)(s dot hhat_N(k))]/|s+k|^2.
```

Here the numerator factors are `O(N)` and `O(N^-1)`, while the denominator
is `O(N^2)`. Each pressure coefficient is `O(N^-2)`. Multiplication by the
second `O(N^-1)` high coefficient costs `O(N^-3)` per resonant pair, and
there are `O(N^3)` pairs. Hence

```text
B_cross_pressure(h_N,U)=O(1)=o(N).
```

The complete load consequently has the same nonzero limit as the
high-high pressure load:

```text
B_complete_HHL(h_N,U)/N
 -> (sqrt(2)/20) integral_D S^2(V_y^2-V_z^2)<0.
```

Combining this with the Fisher upper bound proves the `N^4` divergence.

## 6. Numerical replay

The finite rows are checks on the formulas, not the asymptotic proof.

```text
N    E_lambda       B_complete      |B|/E       (|B|/E)/N^4
---  -------------  --------------  ----------  -------------
  3  5.363543e-2    -1.175977e-2       0.2193   2.70684e-3
  5  1.724307e-2    -1.319834e-2       0.7654   1.22469e-3
  9  3.868692e-3    -1.810911e-2       4.6809   7.13449e-4
 17  6.717371e-4    -2.902876e-2      43.2145   5.17408e-4
 33  9.994594e-5    -5.136294e-2     513.9072   4.33340e-4
 49  3.143080e-5    -7.380927e-2    2348.3097   4.07353e-4
 65  1.366444e-5    -9.628466e-2    7046.3683   3.94741e-4
```

At `N=65`,

```text
N^3 E_lambda                  = 3.75259618,
B_pressure_HH/N               = -0.00148130245,
B_kinetic/N                   = -2.07e-11,
B_cross_pressure/N            =  3.95e-13.
```

For `N=3`, the sparse formulas were independently compared with the older
dictionary Fourier algebra. Fisher and all three component loads agree to
better than `1.3e-16`, and the complete component expansion agrees with
the direct cubic polarization identity.

A fixed `3x3` transverse control with only its longitudinal length growing
does not diverge for this fixed low wave. Its ratios decrease from
`0.3589` at length 8 to `0.2523` at length 256. The obstruction genuinely
uses a three-dimensional mode population inside a bounded annulus; a
short finite longitudinal fit was not an asymptotic argument.

## 7. Route decision

Closed:

- an explicit real divergence-free annular family;
- an exact mixed-difference Fisher identity;
- a uniform `O(N^-3)` Fisher upper bound;
- an exact strictly negative high-high pressure limit;
- cancellation of the leading kinetic limit;
- lower order of cross pressure;
- `N^4` divergence for the complete HHL/Fisher ratio;
- falsification of the proposed uniform static joint-Schur estimate.

Not closed:

- every possible pressure/Fisher estimate;
- a signed sum over all eight vertices;
- a time-dependent or parabolically weighted estimate;
- cross-shell absorption;
- terminal dual control;
- critical `L^3`;
- Navier-Stokes global regularity.

The isolated primitive-chain Hardy theorem remains valid. The new family
escapes by coherent recombination across a genuinely three-dimensional set
of transverse residues. The next useful gate is to apply the family to the
signed all-vertex local-energy identity and determine whether that larger
structure cancels it. If it survives, the static localization route must be
replaced by an evolution-dependent payment.

The deterministic audit is
`scripts/separable_annular_pressure_schur_no_go_audit.py`; its production
result is
`results/separable_annular_pressure_schur_no_go_audit_v1.json`.
