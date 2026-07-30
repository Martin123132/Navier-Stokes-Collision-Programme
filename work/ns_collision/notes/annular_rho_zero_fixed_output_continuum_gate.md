# Annular rho-zero fixed-output continuum gate

## 1. Result

The two `N^7`-saturating fixed-output contractions in the
four-high/one-low coefficient have a genuine continuum limit. They are
not an unresolved collection of hundreds of bounded Fourier modes.

Exactly 36 pressure outputs can contribute at leading order. After their
signs and pressure projectors are combined, their entire low-frequency
geometry is the single matrix

```text
Q=(sqrt(2)/40) diag(0,-1,1).                     (1.1)
```

Their normalized sum `D_N` therefore converges:

```text
D_N/N^7 -> L_EE.                                 (1.2)
```

where `L_EE` is an explicit pair of fixed-domain convolution integrals.
The finite rows suggest

```text
L_EE approximately -3.0e-7,
```

but that sign is not certified here. A polynomial fit in `1/N` is a
diagnostic, not an interval proof.

## 2. The low test stencil

Let

```text
ell=(0,1,-1),
d=(0,1,1)/sqrt(2).
```

The low field has coefficients

```text
Uhat(+ell)=-i d,   Uhat(-ell)=+i d.
```

The weight coefficient at `s in {-1,0,1}^3` is

```text
w_s=product_j (1/2 if s_j=0, 1/4 otherwise).
```

Thus the Fourier coefficient of `U dot grad Phi` is

```text
A_q
 =(d dot q)[w_(q-ell)-w_(q+ell)]
 =(q_y+q_z)[w_(q-ell)-w_(q+ell)]/sqrt(2).        (2.1)
```

There are exactly 36 nonzero values. Every one satisfies

```text
|q|^2<=6<16.                                     (2.2)
```

This explains why the numerical signal was concentrated inside `|q|<4`;
the true leading support is smaller and exact.

## 3. Parity gauge

For odd `N`, a positive packet wave is

```text
k=(2N+i-1, j-(N+1)/2, l-(N+1)/2).
```

Its alternating coefficient obeys

```text
(-1)^(i+j+l)=-(-1)^(k_x+k_y+k_z).
```

Set

```text
sigma_k=(-1)^(k_x+k_y+k_z).
```

The packet can then be written exactly as

```text
Hhat_N(k)=-sigma_k N^-1 a_N(k/N),                (3.1)
```

with the even convention `a_N(-xi)=a_N(xi)`.

On the positive continuum packet

```text
D=[2,3] x [-1/2,1/2]^2,
```

define

```text
S(xi)
 =sin(pi(x-2)) sin(pi(y+1/2)) sin(pi(z+1/2)),

a(xi)=S(xi) P_xi(e_3)/|xi|.                      (3.2)
```

Extend `a` evenly to `-D` and by zero elsewhere. The sine factor vanishes
on every face, so this zero extension is Lipschitz. It is not necessary to
claim that the extension is `C^2`, `C^6`, or smooth across the boundary.

The sample arguments differ from (3.2) by at most

```text
1/N, 1/(2N), 1/(2N)
```

in the three sine factors. Together with

```text
|P_xi(e_3)/|xi||<=1/2,
```

this gives the coarse explicit estimate

```text
epsilon_N
 :=||a_N-a||_L1+||a_N-a||_L2
 <=64/N                                           (3.3)
```

for odd `N>=5`, using centred lattice cells.

## 4. Continuum Euler fields

Write

```text
V=B(H,H),
G=B(H,V).
```

For divergence-free Fourier coefficients,

```text
B(x,y)^hat(r)
 =-i/2 P_r sum_k [
    (r dot xhat(k)) yhat(r-k)
   +(r dot yhat(k)) xhat(r-k)].                   (4.1)
```

Define the continuum convolutions

```text
v(rho)
 =P_rho integral
   (rho dot a(xi)) a(rho-xi) dxi,                 (4.2)

g(rho)
 =1/2 P_rho integral [
    (rho dot a(xi))v(rho-xi)
   +(rho dot v(xi))a(rho-xi)] dxi.                (4.3)
```

The parity factors in (3.1) are constant across each convolution:

```text
Vhat_N(r)=-i sigma_r N^2 v_N(r/N),

Ghat_N(r)=   sigma_r N^5 g_N(r/N).                (4.4)
```

The limiting profiles satisfy

```text
a(-rho)= a(rho),
v(-rho)=-v(rho),
g(-rho)= g(rho).                                  (4.5)
```

## 5. Which symmetrized terms survive

For

```text
T(x,y,z;Phi)=integral p[x,y] z dot grad Phi,
```

the first dominant form is

```text
-6S(V,V,U;Phi)
 =-2T(V,V,U;Phi)-4T(V,U,V;Phi).                  (5.1)
```

Only the first term can have fixed-output order `N^7`. In the second term,
bounded pressure output forces both occurrences of `V` to bounded
frequencies. Each is only `O(N)`, so the contribution is `O(N^2)`.

The second dominant form is

```text
-12S(G,H,U;Phi)
 =-4T(G,H,U;Phi)
  -4T(G,U,H;Phi)
  -4T(H,U,G;Phi).                                 (5.2)
```

At bounded pressure output and `N>=5`, the last two terms vanish by the
annular support gap. Thus the two `N^7` contractions are exactly

```text
-2T(V,V,U;Phi),
-4T(G,H,U;Phi).                                   (5.3)
```

## 6. Fixed-output pressure limits

For fixed nonzero `q`, equations (4.4)-(4.5) give

```text
N^-7 p[V,V]^hat(q)
 ->-sigma_q integral (e_q dot v)^2,               (6.1)

N^-7 p[G,H]^hat(q)
 -> sigma_q integral (e_q dot g)(e_q dot a),      (6.2)
```

where `e_q=q/|q|`.

Now combine every `q` before deciding the sign. From (2.1),

```text
Q
 :=sum_(q!=0) sigma_q A_q q q^T/|q|^2
  =(sqrt(2)/40) diag(0,-1,1).                    (6.3)
```

This identity is exact rational arithmetic after factoring out
`sqrt(2)`. It yields

```text
L_VV
 =(sqrt(2)/20) integral (v_z^2-v_y^2),            (6.4)

L_GH
 =(sqrt(2)/10) integral (g_y a_y-g_z a_z),        (6.5)

L_EE=L_VV+L_GH.                                   (6.6)
```

The continuum energy identity gives a useful independent trace check:

```text
integral |v|^2+2 integral g dot a=0.              (6.7)
```

Matrix (6.3) is traceless, so this gate probes only the anisotropic part
of (6.7).

## 7. Quantitative convergence

For compactly supported profiles `f,g`, define the continuum version of
(4.1) by `B_c(f,g)`. If their support radii are `R_f,R_g`, Young's
inequality gives, for `p=1,2`,

```text
||B_c(f,g)||_p
 <=(R_f+R_g)/2 [
    ||f||_1||g||_p+||g||_1||f||_p].              (7.1)
```

Apply (7.1) twice, first to `v=B_c(a,a)` and then to
`g=B_c(a,v)`. Equation (3.3), the finite shifts `q/N`, and
`max|q|=sqrt(6)` give the deliberately coarse bound

```text
|N^-7 D_N-L_EE|<=250000/N                         (7.2)
```

for odd `N>=128`, where `D_N` is the sum of the two contractions in
(5.3). The size of this constant is irrelevant to the sign computation;
its purpose is to make convergence quantitative without using a fit.

It remains to relate `D_N` to the complete `c_1,N`. At bounded nonactive
output, the support incidence loses at least one carrier power. At high
pressure output, discrete summation by parts moves one compatible
difference onto either:

- the zero-extended Lipschitz packet profile, costing `C/N`; or
- a pressure/Leray multiplier on a dyadic shell, costing `C/|r|`.

Using only this first difference, rather than unsupported higher boundary
smoothness, suggests the target bound

```text
|c_1,N-D_N|
 <=C_stencil N^6 log(2+N),                        (7.3)
```

where `C_stencil` must be instantiated as the finite sum of the
compact-support `L1`, `L2`, and first-difference constants for the seven
nonleading amplitude-one terms and the three nonleading permutations in
(5.1)-(5.2). This audit has not yet written and checked that termwise
constant ledger.

Once (7.3) is established term by term, combining it with (7.2) will give

```text
|c_1,N/N^7-L_EE|
 <=[250000+C_stencil log(2+N)]/N.                 (7.4)
```

Equation (7.2) proves (1.2). Equation (7.4), and therefore convergence of
the complete `c_1,N/N^7`, remains conditional on the missing ledger.

## 8. Finite replay

Projecting the stored exact FFT rows onto the 36 modes gives

```text
N     active sum/N^7
5    -2.4530460e-6
9    -1.0133663e-6
17   -5.8122475e-7
21   -5.1391652e-7
25   -4.7225460e-7
29   -4.4401498e-7
```

At `N=29`,

```text
active 36-mode sum       =-7659.203530...
full c_1,N               =-7660.403124...
full minus active        =   -1.199594...
```

Recent quadratic and cubic fits in `1/N` give negative candidates between
roughly `-3.12e-7` and `-2.96e-7`. These values motivate the next
calculation but certify nothing about the limiting sign.

## 9. Correct scope

This stage proves:

```text
the exact 36-mode support,
the exact signed projector matrix,
the fixed-domain formula for L_EE,
and convergence D_N/N^7 -> L_EE with a quantitative remainder.
```

It does not prove:

```text
the termwise tail bound needed to replace D_N by the full c_1,N,
L_EE<0 or even L_EE!=0,
a nonzero optimized N^9 coefficient,
control of the complete viscous second jet,
a uniform Taylor remainder on a parabolic window,
critical L^3 control, blowup, or global regularity.
```

The next proof gate has two parts. First, write the ten omitted
permutation/term bounds with explicit constants and prove (7.3). Second,
give deterministic convolution quadrature for (6.4)-(6.5) an interval
error narrower than the candidate `3e-7` margin. The two pieces must be
combined before the sign is certified.

## 10. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_rho_zero_fixed_output_continuum_gate_audit.py
```

The production record is
`results/annular_rho_zero_fixed_output_continuum_gate_audit_v1.json`.
