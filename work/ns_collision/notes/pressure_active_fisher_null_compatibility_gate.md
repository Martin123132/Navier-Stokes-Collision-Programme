# Pressure-active Fisher-null compatibility gate

Status: certified for one canonical affine residue chain. The complete
high-high-low local-energy symbol is controlled by one half of the full
signed weighted Fisher graph. This is a positive compatibility result, not
yet a theorem for arbitrary residue chains or the full multiband operator.

## 1. Why this gate is needed

For the compatible zero-face weight

```text
lambda_1(x_1)=sin(x_1/2)^2,
```

weighted velocity Fisher energy is a discrete Dirichlet form on frequency
residue chains. The preceding multiband no-go showed that the neighboring
edges of this form cannot be discarded or charged separately to dyadic
blocks.

The next possible obstruction is more serious. A pressure-active HHL symbol
could act nontrivially on a long Fisher near-null chain. If its first-order
skew part were not paired with either the chain derivative or the
transverse curvature floor, no joint pressure-Fisher estimate could hold.

This note resolves that question for a canonical chain, retaining every
signed Fisher edge.

## 2. Canonical two-polarization chain

Take

```text
k_n=(n,1,0),              n=N_0,...,N_1,
K_n=n^2+1.                                           (2.1)
```

The two divergence-free polarizations are

```text
uhat_s(k_n)=alpha_n e_3/sqrt(K_n),

uhat_t(k_n)=beta_n(-1,n,0)/K_n.                     (2.2)
```

Set the negative-frequency coefficients by complex conjugation. The
normalization in (2.2) makes each corresponding gradient matrix have
Frobenius norm `|alpha_n|` or `|beta_n|`.

Use the real low wave

```text
Uhat(0,-1,0)=Uhat(0,1,0)=e_1                       (2.3)
```

and the compatible weight

```text
lambda(x)
 =phi_-(x_1)phi_+(x_2)
 =[(1-cos x_1)/2][(1+cos x_2)/2].                  (2.4)
```

It is the sum of the two eight-cell vertices `(-,+,-)` and `(-,+,+)`.
There is no `x_3` dependence.

## 3. Exact full Fisher graph

Write

```text
c_n
 =[n(n+1)+1]/sqrt(K_n K_(n+1)).                    (3.1)
```

This is the cosine of the angle between `k_n` and `k_(n+1)`. The gradient
Gram factors for the two polarizations are

```text
gamma_(s,n)=c_n,

gamma_(t,n)=c_n^2.                                 (3.2)
```

For

```text
F_gamma(a,b)
 =|a|^2+|b|^2-2 gamma Re(b conjugate(a)),          (3.3)
```

direct Fourier expansion of (2.4) gives

```text
E_lambda(h)
 =integral lambda|grad h|^2

 =(1/4) sum_edges
   [F_(gamma_s)(alpha_n,alpha_(n+1))
    +F_(gamma_t)(beta_n,beta_(n+1))].              (3.4)
```

The sum includes the two endpoint edges to zero. Equation (3.4) is the
unsplit physical Fisher form. No annular absolute values have been taken.

The transverse residue lifts the exact constant-chain interior null:

```text
1-c_n^2=1/(K_n K_(n+1)).                           (3.5)
```

This small curvature term is precisely what will pay the skew HHL symbol.

## 4. Complete HHL edge symbol

For

```text
F_HHL
 =(|h|^2/2)U+(U dot h)h
  +p[h,h]U+(p[U,h]+p[h,U])h,                       (4.1)
```

the load against `grad lambda` has no diagonal or cross-polarization
terms. It is exactly

```text
B_HHL
 =sum_n C_s(n) Im(alpha_(n+1) conjugate(alpha_n))
  +sum_n C_t(n) Im(beta_(n+1) conjugate(beta_n)).
                                                               (4.2)
```

The shear coefficient is

```text
C_s(n)=-1/[4 sqrt(K_n K_(n+1))].                  (4.3)
```

For the in-plane polarization, the high-high pressure edge is genuinely
present:

```text
phat_HH(e_1)
 =-2 sum_n beta_(n+1)conjugate(beta_n)/
             [K_n K_(n+1)],                       (4.4)
```

and its selected load coefficient is

```text
C_(p,HH)(n)=+1/[2 K_n K_(n+1)].                   (4.5)
```

The anisotropic kinetic term `(U dot h)h` contributes exactly

```text
C_(K,anis)(n)=-1/[2 K_n K_(n+1)].                 (4.6)
```

Thus (4.5) and (4.6) cancel before cross pressure is added. The remaining
kinetic scalar and both ordered cross-pressure terms give

```text
C_t(n)
 =-[n^6+3n^5+14n^4+23n^3+35n^2+24n+4]

   /[4(n^2+1)(n^2+4)
       (n^2+2n+2)(n^2+2n+5)].                     (4.7)
```

This cancellation is part of the complete local-energy flux. It would be
missed by studying pressure alone.

## 5. Edge domination theorem

For `0<=gamma<1`,

```text
F_gamma(a,b)
 =|b-gamma a|^2+(1-gamma^2)|a|^2.                 (5.1)
```

Since

```text
Im(b conjugate(a))
 =Im((b-gamma a)conjugate(a)),
```

Young's inequality gives the sharp elementary estimate

```text
2 sqrt(1-gamma^2)
  |Im(b conjugate(a))|
 <=F_gamma(a,b).                                  (5.2)
```

For the shear coefficient, (3.5) and (4.3) give the exact identity

```text
C_s(n)^2=[1-gamma_(s,n)^2]/16.                    (5.3)
```

For the in-plane coefficient, exact symbolic reduction gives

```text
[1-gamma_(t,n)^2]/16-C_t(n)^2

 =Q(n)/
  [16(n^2+1)^2(n^2+4)^2
      (n^2+2n+2)^2(n^2+2n+5)^2],                 (5.4)
```

where

```text
Q(n)
 =n^12+6n^11+29n^10+90n^9+261n^8+570n^7
  +1275n^6+2142n^5+3538n^4+4024n^3
  +4096n^2+2368n+1184.                            (5.5)
```

Every coefficient of `Q` is positive. Therefore, for every integer
`n>=1`,

```text
|C_sigma(n)|
 <=(1/4)sqrt(1-gamma_(sigma,n)^2),

sigma in {s,t}.                                   (5.6)
```

Combining (5.2) and (5.6) controls each internal edge:

```text
|C_sigma(n) Im(z_n)|
 <=F_(gamma_sigma)(n)/8.                          (5.7)
```

Summing (5.7), retaining the nonnegative endpoint edges in (3.4), proves

```text
|B_HHL|<=E_lambda(h)/2.                            (5.8)
```

The constant is uniform in the chain endpoints and length.

## 6. Null and pressure-active modes

Real constant data and the real first Dirichlet mode

```text
a_j=sin[pi(j+1)/(N+1)]
```

have

```text
Im(a_(j+1)conjugate(a_j))=0.                       (6.1)
```

Their complete HHL load therefore vanishes exactly. The transverse
curvature and finite endpoints still make their Fisher energy positive;
"null" here refers to the HHL skew direction and to the limiting interior
Fisher null inherited from the axial chain.

Multiplying the chain data by a phase ramp makes the skew products
nonzero. The high-high pressure load then becomes nonzero as well. These
are pressure-active perturbations of the near-null direction, and (5.8)
pays them without splitting the physical Fisher graph.

The initial zero found with real coefficients was therefore a parity
cancellation in the selected local-energy load, not absence of generated
pressure. Equation (4.4) is nonzero even for a real constant chain; its
cosine output is orthogonal to the corresponding odd weight derivative.

## 7. Independent sparse replay

The production audit reconstructs (4.1) with sparse Fourier convolution,
including:

- both divergence-free polarizations;
- kinetic scalar and anisotropic terms;
- high-high pressure;
- both ordered low-high pressure terms;
- both compatible vertices in (2.4);
- the independent cubic polynomial reconstruction of the complete HHL
  coefficient.

Across 16 constant, first-Dirichlet, and phase-tilted rows:

```text
maximum analytic/sparse load residual       3.47e-18,
maximum complete Fourier residual           9.51e-16,
pressure-active rows                        8,
maximum observed |B_HHL|/E_lambda           0.01690.
```

Generalized finite-chain spectra for both polarizations, chain lengths
through 128, and starts `1`, `4`, and `16` have maximum absolute
generalized eigenvalue

```text
0.1870369331<1/2.                               (7.1)
```

The spectra are checks, not the proof; the proof is the exact edge
inequality (5.7).

Taylor-Green, seed-81, and the carrier-1024 modulated-wave HHL no-go replay
unchanged. In particular, this canonical compatibility theorem does not
invalidate the separate example whose complete HHL load tends to `1/144`.

## 8. Scope and next theorem

Proved:

- exact complete HHL symbol for the canonical affine residue chain;
- exact pressure/anisotropic-kinetic cancellation;
- a positive-polynomial certificate for the in-plane coefficient;
- the uniform two-polarization bound (5.8);
- compatibility of real constant and first-Dirichlet chain directions;
- Fisher control of pressure-active phase tilts.

Not proved:

- arbitrary primitive partition steps or transverse residues;
- arbitrary low waves, low polarizations, or translated partition phases;
- coupling between different residue chains;
- a uniform full multiband HHL Schur bound;
- terminal dual control, critical `L^3`, or global regularity.

The next stage should lift (4.2)-(5.8) to general primitive partition steps
and transverse residues, then assemble the finite low-wave/vertex block
without taking absolute values between residue chains. The modulated-wave
HHL adversary must remain an explicit block in that calculation.

The deterministic certificate is generated by
`scripts/pressure_active_fisher_null_compatibility_gate_audit.py`; its
production result is
`results/pressure_active_fisher_null_compatibility_gate_audit_v1.json`.
