# Fourier pressure-load surjectivity

Status: exact finite-Fourier construction proving that the instantaneous
Navier-Stokes pressure-load map reaches every zero-sum eight-cell load
vector. In particular, it reaches all eight Hamming-profile rays identified
by the compatible cubic graph audit. This closes the proposed algebraic
exclusion route, not the quantitative pressure estimate. No critical bound
or Navier-Stokes regularity conclusion is claimed.

## 1. Fourier pressure map

Use normalized periodic means and

```text
u(x)=sum_(k in Z^3) u_hat(k)e^(ik.x),

u_hat(-k)=conj(u_hat(k)),

k.u_hat(k)=0.                                       (1.1)
```

The mean-zero pressure determined by incompressibility satisfies

```text
p_hat(q)
 =-|q|^(-2) sum_(a+b=q)
   [q.u_hat(a)][q.u_hat(b)],

q!=0.                                               (1.2)
```

Define the scalar pressure transport

```text
g=-u.grad p.                                        (1.3)
```

Then

```text
g_hat(n)
 =-i sum_(q+k=n)[q.u_hat(k)]p_hat(q).               (1.4)
```

Equations (1.2)-(1.4) are a homogeneous cubic map from divergence-free
velocity coefficients to scalar Fourier coefficients.

## 2. Seven Walsh load coordinates

At partition frequency one and center zero,

```text
phi_s(x)=(1+s cos x)/2,

Phi_v=product_(j=1)^3 phi_(v_j),

v in {-1,+1}^3.
```

The pressure work carried by the vertex coefficient `w_v` is

```text
b_v
 =mean[p u.grad Phi_v]
 =-mean[Phi_v u.grad p]
 =mean[Phi_v g].                                    (2.1)
```

For a nonempty coordinate subset `S`, let

```text
chi_S(v)=product_(j in S)v_j,

C_S(x)=product_(j in S)cos x_j,

G_S=mean[g C_S].                                    (2.2)
```

The product partition has the exact Walsh expansion

```text
Phi_v
 =1/8 sum_(S subset {1,2,3})chi_S(v)C_S.            (2.3)
```

Because

```text
mean g=-mean div(pu)=0,
```

equations (2.1)-(2.3) give

```text
b_v
 =1/8 sum_(S nonempty)chi_S(v)G_S,                  (2.4)

sum_v b_v=0.                                        (2.5)
```

Thus the compatible pressure load is equivalent to seven real Walsh
coordinates. Surjectivity of the map `u -> (G_S)` is exactly surjectivity
onto the zero-sum load hyperplane.

## 3. One isolated block

Fix a nonzero stencil wave

```text
n in {0,1}^3.
```

For an integer `N`, define

```text
a=(N,0,0),

b=(0,2N,0),

k=n-a-b,

q=a+b.                                              (3.1)
```

Choose the integer polarizations

```text
A=(0,1,0),

B=(1,0,0),

C=|k|^2 q-(k.q)k.                                   (3.2)
```

They are divergence free:

```text
a.A=0,

b.B=0,

k.C=0.                                              (3.3)
```

Moreover,

```text
q.C=|k|^2|q|^2-(k.q)^2=|k cross q|^2>0              (3.4)
```

for the seven stencil waves used below.

Put Fourier coefficients on the six modes `+/-a`, `+/-b`, and `+/-k`:

```text
u_hat(a)=alpha A,

u_hat(b)=alpha B,

u_hat(k)=i sigma alpha C,

u_hat(-r)=conj(u_hat(r)),                           (3.5)
```

where `sigma` is `+1` or `-1`.

Only the three pairings

```text
(a,b;k),

(a,k;b),

(b,k;a)
```

contribute to `g_hat(n)`. Direct substitution into (1.2)-(1.4) gives

```text
g_hat(n)=sigma alpha^3 K(n,N),                      (3.6)
```

with the exact nonzero rational coefficient

```text
K(n,N)
 =-2 sum_(r in {a+b,a+k,b+k})
   [(r.A)(r.B)(r.C)]/|r|^2.                         (3.7)
```

Changing `sigma` changes the sign, and changing `alpha` changes the
magnitude cubically.

## 4. Seven spectrally isolated blocks

Use the seven representative subset waves

```text
(1,0,0),
(0,1,0),
(0,0,1),
(1,1,0),
(1,0,1),
(0,1,1),
(1,1,1),                                           (4.1)
```

with scales

```text
N=8,32,128,512,2048,8192,32768.                    (4.2)
```

The resulting real velocity has `42` occupied Fourier modes.

An exhaustive exact integer enumeration checks every unordered triple of
these signed modes. Exactly `14` triples reach the partition stencil

```text
{-1,0,1}^3 minus {0}.
```

They are precisely the positive triple and negative triple from each of the
seven blocks. There is:

- no mixed-block low interaction;
- no repeated-mode low interaction;
- no unintended stencil output.

Therefore cross terms in the cubic pressure map are invisible to all seven
Walsh coordinates.

## 5. Surjectivity theorem

For a subset wave `n_S`, a real pair

```text
g_hat(+n_S)=g_hat(-n_S)=gamma_S
```

contributes

```text
G_S=gamma_S/2^(|S|-1).                              (5.1)
```

Given any prescribed real seven-vector `(G_S)`, choose the `S` block by

```text
alpha_S^3
 =2^(|S|-1)|G_S|/|K_S|,                            (5.2)
```

and select `sigma_S` so that

```text
sigma_S K_S=sign(G_S)|K_S|.                        (5.3)
```

Equations (3.6), (5.1), and the support certificate then give the prescribed
`G_S` exactly, independently for all seven subsets.

Hence:

```text
The instantaneous map

  smooth real divergence-free trigonometric polynomials
       -> their Poisson pressure
       -> compatible eight-cell load b

is onto {b in R^8: sum_v b_v=0}.                    (5.4)
```

Every constructed field is smooth initial data for a local smooth
Navier-Stokes solution. The statement is instantaneous and does not assert
that the load direction persists in time.

## 6. Realizing the saturating Hamming ray

The compatible graph audit identified the positive-vertex load profile

```text
Hamming distance 0:  225/256,

Hamming distance 1:  -45/256,

Hamming distance 2:  -27/256,

Hamming distance 3:   -9/256.                      (6.1)
```

Its Walsh coordinates are

```text
G_{1}=G_{2}=G_{3}=27/32,

G_{12}=G_{13}=G_{23}=9/8,

G_{123}=9/8.                                       (6.2)
```

The required transport coefficients are

```text
g_hat(e_j)=27/32,

g_hat(e_j+e_k)=9/4,

g_hat(1,1,1)=9/2.                                  (6.3)
```

Applying (5.2)-(5.3) produces a `42`-mode field. Direct floating sparse
convolution of the complete field gives:

```text
maximum target transport residual <4.2e-13,

maximum unintended stencil mode =0,

maximum load residual <4.1e-14.                    (6.4)
```

This proves that the bad compatible load direction itself is
PDE-realizable.

It does not prove that the actual conditional edge functions generated by
this velocity equal the abstract pointwise functions

```text
e_j=3c v_j Phi_(v_hat_j)^2
```

from the previous sharpness construction. Accordingly, it does not show
that the actual velocity saturates the full directionwise `L^(3/2)`
envelope.

## 7. Independent benchmarks

### Taylor-Green

The sparse pressure map gives zero on every partition-stencil mode for

```text
u=(sin x cos y,-cos x sin y,0).
```

Thus all eight compatible loads vanish, independently reproducing the
earlier mode-three cancellation.

### Seed 81

The existing seed-81 pressure adversary has `116` velocity modes and `696`
induced pressure modes. Computing its seven loads by sparse convolution,
then pairing them with the stored eight partition coefficients, gives

```text
sparse pressure work =1.280453496113644,

stored grid work     =1.280453496113639,

absolute residual   <4.9e-15.                      (7.1)
```

This cross-checks the signs, Fourier normalization, partition center, and
pressure convention against an independent existing calculation.

## 8. What this rules out

The following proposed hope is false:

```text
Incompressibility and the pressure Poisson law algebraically exclude the
worst compatible graph load directions.
```

They do not. The instantaneous finite-Fourier map reaches every direction.
No sign cone, missing ray, or algebraic codimension is available at this
level.

This does not rule out a quantitative estimate. The lacunary construction
uses large frequencies and pays a substantial velocity norm. Its success
leaves open bounds involving:

- velocity `L^3`, energy, or enstrophy cost;
- the partition frequency relative to occupied velocity frequencies;
- time persistence under Navier-Stokes evolution;
- cross-level cancellation on an intrinsic balanced cover;
- Carleson control of repeated load realization across scales.

## 9. Next theorem target

For a prescribed nonzero load vector `b`, define realization costs such as

```text
E_0(b)
 =inf{||u||_2^2: u divergence free and b(u)=b},

E_1(b)
 =inf{||grad u||_2^2: u divergence free and b(u)=b},

E_c(b)
 =inf{||u||_3^3: u divergence free and b(u)=b}.      (9.1)
```

The cubic homogeneity `b(alpha u)=alpha^3b(u)` fixes their amplitude scaling.
The next bounded stage should:

1. derive the spatial and amplitude scaling laws exactly;
2. optimize a single block before optimizing seven-block assemblies;
3. compare the Hamming ray with Taylor-Green and seed-81;
4. test whether a critical realization cost remains coercive when carrier
   frequencies are translated upward;
5. connect any surviving coercivity to the intrinsic pressure/Fisher budget.
