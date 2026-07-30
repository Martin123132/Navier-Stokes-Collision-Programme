# Joint scale-cell coherence gate and viscous occupation

Status: the cumulative low-output Reynolds stress has the expected
`L1-to-L2` shell-energy bound, but the exact sideband family falsifies a
pointwise `ell2` improvement based on high-shell orthogonality. Every shell
collapses onto the same low Fourier mode and the same cell Walsh character.
Retaining time changes the answer: Stokes damping gives a weighted
`L2_t ell2_H` theorem and a finite occupation bound. Extending that theorem
to Navier-Stokes requires control of nonlinear Duhamel regeneration, which
is not proved here. No regularity conclusion is claimed.

## 1. Cumulative low-output stress

For a low dyadic scale `L`, define schematically

```text
R_L
 =P_(<=cL) sum_(H,H'>=4L, H~H')
    (u_H tensor u_H').                            (1.1)
```

Comparable high neighbors are included because the exact cubic atlas only
forces the largest two frequencies to be comparable, not equal.

The low-pass kernel has `L2` norm `C_c L^(3/2)`. Young's inequality gives

```text
||P_(<=cL)(u_H tensor u_H')||_2
 <=C_c L^(3/2)||u_H||_2||u_H'||_2.                (1.2)
```

Finite shell overlap and `2ab<=a^2+b^2` therefore imply

```text
||R_L||_2
 <=C_c L^(3/2) sum_(H>=4L)||u_H||_2^2.            (1.3)
```

The low-frequency double Riesz transform is `L2` bounded, so the same
estimate controls the associated low pressure. Pairing with a low velocity
and `grad W`, where `W` is a cell-weighted partition, recovers the previous
HHL envelope.

The right side of (1.3) is an `ell1` sum of shell energies. The proposed
improvement would replace it, at least in the dangerous signed channel, by

```text
[sum_H ||u_H||_2^4]^(1/2).                         (1.4)
```

That replacement needs actual orthogonality after low-frequency and cell
projection. It is the gate tested below.

## 2. Exact common Fourier channel

For every integer carrier `H`, retain the two high sidebands

```text
a_H=(1,1,H),       b_H=(0,0,H),

q=a_H-b_H=(1,1,0).                                (2.1)
```

Their divergence-free polarizations are `A_H` and `e_1`. The quadratic
Reynolds-stress coefficient at `q` is

```text
Rhat_H(q)
 =A_H tensor e_1+e_1 tensor A_H.                  (2.2)
```

The first polarization component is

```text
(A_H)_1=sqrt[(H^2+1)/(H^2+2)].                    (2.3)
```

Consequently

```text
(Rhat_H(q))_11
 =2sqrt[(H^2+1)/(H^2+2)]
 >=2sqrt(2/3),                                    (2.4)

||Rhat_H(q)||_F<=2.                               (2.5)
```

Take dyadically separated carriers `H_j`. A cross pair from distinct
carriers has vertical output `H_j-H_k`, so it cannot equal `q`. Thus

```text
Fourier_q[R(sum_j u_Hj tensor sum_k u_Hk)]
 =sum_j Rhat_Hj(q)                                (2.6)
```

exactly.

## 3. The pointwise orthogonality no-go

Equations (2.4)-(2.5) give

```text
||sum_(j=1)^N Rhat_Hj(q)||_F
 --------------------------------
 [sum_(j=1)^N ||Rhat_Hj(q)||_F^2]^(1/2)

 >=sqrt(2/3)sqrt(N).                              (3.1)
```

No carrier-independent constant can bound the coherent stress sum by its
shell square function.

The cell variable does not rescue the estimate. For the complete signed
HHL local-energy flux, all eight vertex loads satisfy

```text
b_(H,v)=chi_123(v)b_H,                             (3.2)
```

and

```text
b_H->1/144>0.                                     (3.3)
```

All carriers therefore occupy the single joint channel

```text
(low Fourier mode q, Walsh mask 123).             (3.4)
```

It follows that

```text
|sum_(j=1)^N b_Hj|
 -----------------------
 [sum_(j=1)^N |b_Hj|^2]^(1/2)

 ~sqrt(N).                                        (3.5)
```

The finite replay through eight carriers reconstructs both sums exactly
and gives ratios greater than two.

This is a scoped no-go. It does not falsify:

- the `ell1` shell-energy estimate (1.3);
- a bound using differences between low scales or cells;
- a spacetime Carleson estimate;
- a theorem using viscous duration rather than instantaneous magnitude.

It only removes high-shell-label orthogonality after projection to the
common Fourier-Walsh channel.

## 4. A Stokes thought experiment

Now retain time. Evolve the entire finite Fourier field by Stokes heat flow.
The two high modes have squared frequencies

```text
|a_H|^2=H^2+2,       |b_H|^2=H^2,                 (4.1)
```

and the fixed low testing mode has squared frequency `9`. Every resonant
complete HHL load contains one copy of each. Hence

```text
b_H(t)
 =b_H(0) exp[-nu(2H^2+11)t].                      (4.2)
```

This identity is exact, not an asymptotic high-carrier approximation. The
sparse replay evolves every Fourier coefficient independently and
reconstructs (4.2) at three dimensionless times per carrier.

At `t=0`, `N` coherent shells produce a load of order `N`. The highest
shells, however, persist for progressively shorter times of order
`H^(-2)`. That duration is the gain absent from the pointwise estimate.

## 5. Viscous Gram theorem

Let `c_j` take values in any Hilbert space and set

```text
F(t)=sum_j exp(-2nu mu_j^2 t)c_j,                  (5.1)

mu_(j+1)>=rho mu_j,       rho>1.                   (5.2)
```

Direct integration gives

```text
||F||_(L2_t)^2
 =1/(2nu) sum_(j,k)
   <c_j,c_k>/(mu_j^2+mu_k^2).                     (5.3)
```

Write `x_j=c_j/mu_j`. The normalized Gram kernel is

```text
K_jk=mu_j mu_k/(mu_j^2+mu_k^2).                   (5.4)
```

Its diagonal is `1/2`. At shell distance `d>=1`,

```text
|K_(j,j+d)|<=rho^(-d).                            (5.5)
```

The Schur row sum is at most

```text
C_rho=1/2+2/(rho-1).                              (5.6)
```

Therefore

```text
||F||_(L2_t)^2
 <=C_rho/(2nu)
   sum_j ||c_j||^2/mu_j^2.                        (5.7)
```

For the exact HHL heat law,

```text
mu_j=sqrt(H_j^2+11/2).                            (5.8)
```

Dyadic `H_j` give a lacunarity ratio close to two. In particular, adding
arbitrarily many unit-size coherent shells makes `F(0)` unbounded while
the right side of (5.7) remains bounded by a geometric series.

## 6. Occupation consequence

Chebyshev applied to (5.7) gives

```text
measure{t:||F(t)||>=Lambda}
 <=C_rho/(2nu Lambda^2)
   sum_j ||c_j||^2/mu_j^2.                        (6.1)
```

This is the precise surviving viscous-occupation statement. It does not
claim that the nonlinear Navier-Stokes channel follows free heat decay.

## 7. Forced relaxation gate

The natural nonlinear replacement of (5.1) is

```text
dot c_j+2nu mu_j^2 c_j=f_j.                       (7.1)
```

For zero initial data, Young's convolution inequality gives

```text
||exp(-2nu mu_j^2 t)*f_j||_(L2_t)
 <=[2nu mu_j^2]^(-1)||f_j||_(L2_t).               (7.2)
```

Summing and using `mu_j>=mu_0 rho^j` yields the conditional theorem

```text
||sum_j c_j||_(L2_t)

 <=1/[2nu mu_0^2 sqrt(1-rho^(-4))]
   [sum_j ||f_j||_(L2_t)^2]^(1/2).                (7.3)
```

The audit checks (7.3) on an independently integrated family of signed
exponential forcings.

For Navier-Stokes, `f_j` is the projected nonlinear regeneration of the
common low Fourier-Walsh stress channel. The standard Leray energy
inequality does not presently supply the `ell2_H L2_t` bound required by
(7.3). Proving such a bound, finding cancellation that weakens it, or
constructing a coherent forced no-go is the next genuine gate.

## 8. What changed

Established:

- the cumulative low-output Reynolds-stress envelope (1.3);
- exact collapse of dyadic sidebands to one Fourier-Walsh channel;
- failure of a pointwise high-shell `ell2` orthogonality gain;
- the exact complete-HHL Stokes damping law (4.2);
- the Hilbert-valued viscous Gram estimate (5.7);
- the threshold-occupation corollary (6.1);
- the conditional forced-relaxation estimate (7.3).

Still open:

- control of the actual Navier-Stokes regeneration terms in (7.1);
- a critical signed large-data estimate;
- passage to suitable weak solutions;
- exceptional-set removal and global regularity.

The identities and finite-mode replays are generated by
`scripts/joint_scale_cell_viscous_occupation_audit.py`.
