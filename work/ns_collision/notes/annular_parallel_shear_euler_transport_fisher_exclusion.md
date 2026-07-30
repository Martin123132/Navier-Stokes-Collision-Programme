# Annular parallel-shear Euler-transport Fisher exclusion

## 1. Result

Let

```text
F(u,lambda)=mean[lambda |grad u|^2],
E=-P[(u dot grad)u],
A=-u dot grad lambda.
```

The five pure Euler/transport terms arising from the velocity Fisher part of
the generator combine exactly:

```text
H_uu[E,E] weighted Fisher
+D_u[u2_EE] weighted Fisher
+2H_u_lambda[E,A] weighted Fisher
+D_lambda[lambda2_E0] velocity Fisher
+D_lambda[lambda2_0A] velocity Fisher

=-nu d^2/ds^2 F(u(s),lambda(s)).                 (1.1)
```

Here

```text
u_s=E,             u_ss=D E[u]E,
lambda_s=A,        lambda_ss=-E dot grad lambda-u dot grad A.
```

For the repaired parallel-shear restart family and its static optimizer,
the complete block in (1.1) obeys

```text
|block_N|<=C nu N^8=o(N^9).                      (1.2)
```

The companion transported weight-self block is also an exact second
material derivative and obeys

```text
|weight-self block_N|<=C nu N^6=o(N^9).           (1.3)
```

Together, (1.2)-(1.3) close every pure `E,A` viscosity-bearing Fisher row.
They do not yet close rows containing one or more viscous/antidiffusive
directions `V,D`.

## 2. Why all five rows must be combined

Differentiating the weighted Dirichlet functional twice gives

```text
F''
=mean[
  2lambda |grad E|^2
 +2lambda grad u:grad(D E[u]E)
 +4A grad u:grad E
 +(-E dot grad lambda-u dot grad A)|grad u|^2
].                                               (2.1)
```

Multiplication by `-nu` reproduces the factors and signs of all five audit
rows in (1.1). In particular, the original pair

```text
H_uu[E,E], D_u[u2_EE]
```

is not closed under the chain rule. Estimating it without the three
transported-weight rows discards the useful cancellation.

Along Euler velocity and transported weight,

```text
partial_s u+u dot grad u+grad p=0,
partial_s lambda+u dot grad lambda=0.
```

Incompressibility and periodicity imply

```text
d/ds mean[lambda f]=mean[lambda D_t f],
D_t=partial_s+u dot grad.
```

Consequently,

```text
F''=mean[lambda D_t^2 |grad u|^2].               (2.2)
```

Every derivative of `lambda` in (2.1) has disappeared. The Fourier vertex
is therefore `Phi`, not `grad Phi` or `D^2 Phi`.

## 3. Material matrix reduction

Set

```text
M=grad u,          P=Hess p,
Q=M^2+P.
```

The Euler equation gives

```text
D_t M=-Q,
D_t^2 |M|^2=2|Q|^2-2M:D_t Q.                    (3.1)
```

The pressure satisfies

```text
-Delta p=tr(M^2).
```

Thus (3.1) is quartic in `u` and has total differential order four. Hessian
pressure factors are degree-two symbols and are regular at zero output.
When `D_t P` is expressed through `Hess(D_t p)`, one degree-zero outer
Riesz projector remains. It is always paired with the exterior factor
`M`. On an HHHH contribution that factor is high, so it can be chosen as
the dependent resonance leaf: the vertex shift is absorbed there while
the outer output and projector are held fixed.

This is the same fixed-output protection used in the complete `c1` tail
ledger. No derivative through the zero-output pressure convention is
taken.

## 4. The two-difference vertex lemma

After parity gauging, each coordinate of the `Phi` vertex has stencil

```text
(-1/4, 1/2, -1/4).
```

For a coordinate power `q^m`, its vanishing order is

```text
m        0  1  2  3  4
order    2  1  0  1  0.                          (4.1)
```

The quartic kernel has at most four scalar derivative factors, so every
vertex monomial has multiindex `alpha` with `|alpha|<=4`. Exhausting all
35 such multiindices gives

```text
sum_j order(alpha_j)>=2.                          (4.2)
```

The minimum is attained, for example, at permutations of `(2,2,0)` and
`(2,1,1)`. Therefore every atomic quartic contraction retains at least two
exact compatible lattice differences after all explicit vertex powers have
been consumed.

## 5. Boundary-safe packet differences

For the one-dimensional zero-extended sine sequence

```text
s_n=sin(pi n/(N+1)),  1<=n<=N,
```

write `theta=pi/(N+1)`. Direct summation gives

```text
||s||_1=cot(theta/2)<=N+1,
||Delta s||_1=2 max s<=2,
||Delta^2 s||_1=4 sin(theta)<=4pi/(N+1).          (5.1)
```

The parity-gauged polarization multiplier is homogeneous of degree `-1`
and smooth on a fixed annulus separated from zero. Its first two unit
differences are therefore `O(N^-2)` and `O(N^-3)`. Applying the discrete
product rule to the three sine factors and this multiplier yields

```text
||h_N||_infinity=O(N^-1),   ||h_N||_1=O(N^2),
||Delta h_N||_1=O(N),       ||Delta^2 h_N||_1=O(1). (5.2)
```

Mixed second differences have the same `O(1)` bound.

Equation (5.2), rather than a pointwise `C^2` claim for the zero extension,
is what is used below. The boundary layer in a pure second difference has
one fewer free lattice coordinate, exactly compensating for its larger
pointwise value.

## 6. HHHH bound

At fixed vertex output:

```text
free four-high tuples                    O(N^9),
product of four high coefficients        O(N^-4),
quartic differential kernel              O(N^4).
```

Without a compatible difference this is the candidate `O(N^9)` scale. The
two differences from (4.2) reduce it to `O(N^7)`:

```text
two profile differences:
  O(N^-2) from the l1 bounds (5.2);

one profile and one kernel difference:
  O(N^-1) times O(N^3) instead of O(N^4);

two regular kernel differences:
  O(N^2) instead of O(N^4).                       (6.1)
```

There is one case not covered by the last line of (6.1): two differences
may hit an internal degree-one Euler symbol whose output is near zero. This
symbol is globally Lipschitz but is not falsely declared `C^2`.

Split its output into a finite shell and dyadic shells

```text
K<=|r|<2K, 1<=K<=CN.
```

For a fixed output there are at most `O(N^6)` choices of the two high
pairs, and there are `O(K^3)` outputs in the shell. The second difference
of the degree-one symbol is `O((1+K)^-1)`. Restoring the four high
coefficients and the remaining degree-three kernel gives

```text
shell contribution
 <=C N^6 K^3 N^-4 N^3/(1+K)
 <=C N^5 K^2.                                    (6.2)
```

The finite shell is `O(N^5)`. The dyadic sum of (6.2) is dominated by its
largest shell and is `O(N^7)`. Hence the fixed-weight HHHH coefficient is
`O(nu N^7)`. Since `t_N=O(N)`,

```text
HHHH optimized contribution=O(nu N^8).           (6.3)
```

## 7. HHLL and LLLL bounds

With two fixed low leaves, one high frequency determines the other up to a
bounded shift. There are `O(N^3)` choices, the high coefficient product is
`O(N^-2)`, and the order-four kernel is at most `O(N^4)`. Thus

```text
HHLL fixed-amplitude coefficient=O(nu N^5).
```

The optimizer contributes `a_N^2 t_N=O(N^3)`, so

```text
HHLL optimized contribution=O(nu N^8).           (7.1)
```

For the low field `U=r f`, one has

```text
r dot grad f=0,
r dot grad(grad f)=0.
```

The Euler velocity is stationary and `D_t grad U=0`; hence the LLLL part of
(2.2) is exactly zero.

Equations (6.3) and (7.1) prove (1.2).

## 8. Transported weight-self companion

Set

```text
W(lambda)=mean[lambda |grad lambda|^2].
```

The seven labelled subterms in

```text
H_lambda_lambda[A,A] weight self
+D_lambda[lambda2_E0+lambda2_0A] weight self
```

combine exactly as

```text
-nu W''=-nu mean[lambda D_t^2 |grad lambda|^2].  (8.1)
```

Indeed, `D_t lambda=0` gives

```text
D_t grad lambda=-(grad u)^T grad lambda,

D_t^2 grad lambda
 =Q^T grad lambda+((grad u)^T)^2 grad lambda.     (8.2)
```

The block is therefore quadratic in velocity, cubic in the weight, and has
at most two high-frequency derivatives.

The HH branch has `O(N^3)` high pairs, coefficient product `O(N^-2)`, and
kernel `O(N^2)`, hence fixed-weight size `O(nu N^3)`. Restoring
`t_N^3=O(N^3)` gives

```text
HH weight-self=O(nu N^6).                         (8.3)
```

The HL branch vanishes by first-coordinate incidence. The LL branch has
only fixed Fourier support and costs

```text
a_N^2 t_N^3=O(N^5).                              (8.4)
```

Equations (8.3)-(8.4) prove (1.3).

## 9. Finite replay

The five independently labelled finite rows sum at `N=5` to

```text
HHHH:
 -0.05166846710339710

HHLL:
 -0.11097887044609589 a_x^2
 -0.02142596988848933 a_y a_x
 -0.02090803769355087 a_y^2.
```

At the stored fixed-amplitude rows their complete sums are

```text
N=3: -0.202940186956581,
N=5: -0.085488882554977.
```

These values check the channel assembly only. Neither their signs nor a
finite scaling fit is used in the proof.

The seven weight-self subterms independently reconstruct

```text
HH:
 -0.00004715074226175247

LL:
 -0.05192057291666574 a_x^2
 -0.03184000651041721 a_y a_x
 -0.05192057291666444 a_y^2.
```

Again, these coefficients check assembly and do not supply the bound.

## 10. Scope

This stage proves

```text
Euler/transport velocity-Fisher block=O(N^8)=o(N^9).
Euler/transport weight-self block=O(N^6)=o(N^9).
```

It does not yet prove:

```text
all viscosity-bearing second-jet rows are o(N^9),
the complete second-jet N^9 limit,
a uniform parabolic-window Taylor estimate,
critical L3 control,
finite-time blowup,
or Navier-Stokes global regularity.
```

The next gate is to close every mixed `V,D` one-heat block and then the
strictly lower two-heat rows.

## 11. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_parallel_shear_euler_transport_fisher_exclusion_audit.py
python -m pytest -q work/ns_collision/tests/test_annular_parallel_shear_euler_transport_fisher_exclusion.py
```

The production record is
`results/annular_parallel_shear_euler_transport_fisher_exclusion_audit_v1.json`.
