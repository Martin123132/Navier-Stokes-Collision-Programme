# Primitive HHL chain Hardy envelope

Status: certified for one isolated primitive residue chain. Arbitrary
transverse residue, low polarization, low phase, compatible vertex sign,
and common vertex translation are allowed. The result supplies the scalar
chain envelope needed before joint low-wave/vertex block assembly. It does
not yet justify summing different primitive chains against the physical
Fisher form.

## 1. Setup

Let

```text
q in {-1,0,1}^3 minus {0}
```

be a primitive partition step, and let

```text
p=number of nonzero coordinates of q.              (1.1)
```

At partition scale `m`, consider the finite one-sided chain

```text
k_n=eta+n m q,

eta dot q=0,

n=N_0,...,N_1,     N_0>=3.                         (1.2)
```

The high field has arbitrary complex divergence-free coefficients at
`+/-k_n` and is extended by zero at both chain endpoints.

The low field is one real Fourier pair at `+/-m ell`, where

```text
ell in {-1,0,1}^3 minus {0},

ell dot Uhat=0.                                    (1.3)
```

Both the direction and complex phase of `Uhat` are arbitrary.

Use one translated compatible tensor vertex

```text
lambda_v(x)
 =product_(j=1)^3 [1+v_j cos(m(x_j-a_j))]/2,

v_j in {-1,+1}.                                    (1.4)
```

The theorem also holds for every nonnegative compatible mixture of these
vertices because both the load and Fisher form are linear in `lambda`.

The carrier condition `N_0>=3` excludes same-sign high-high aliases from
the partition Fourier cube. Only opposite-sign chain pairs remain.

## 2. Complete ordered HHL symbol

For one ordered high pair `(h_a,h_b)` and one low coefficient `U_c`, the
complete HHL local-energy coefficient is

```text
T(a,b,c)
 =1/2 (h_a dot h_b)U_c
  +(U_c dot h_a)h_b
  +p[h_a,h_b]U_c
  +2p[U_c,h_a]h_b.                                (2.1)
```

The last term contains both ordered cross pressures. Since every pressure
multiplier is an orthogonal scalar contraction,

```text
|p[X,Y]|<=|X||Y|.                                  (2.2)
```

Consequently,

```text
|T(a,b,c)|
 <=(1/2+1+1+2)|U_c||h_a||h_b|
 =(9/2)|U_c||h_a||h_b|.                           (2.3)
```

This estimate includes kinetic scalar transport, anisotropic kinetic
transport, high-high pressure, and both low-high pressure orders. No
component of the local-energy flux is omitted.

## 3. Exact finite resonance degree

Write an opposite-sign high pair output as

```text
k_n-k_l=(n-l)m q=d m q.                            (3.1)
```

For low sign `sigma in {-1,+1}`, the partition gradient can see the triple
only if

```text
d q+sigma ell in {-1,0,1}^3 minus {0}.             (3.2)
```

Exhausting all 26 primitive steps, all 26 low waves, both signs, and
`d=-3,...,3` gives

```text
maximum number of resonant (sigma,d) pairs =6.     (3.3)
```

The same maximum holds when the supports of `q` and `ell` are disjoint.
Larger `|d|` cannot satisfy (3.2), so the finite search is exhaustive.

For a fixed high mode, the directed partner degree is therefore at most
six. The inequality `2ab<=a^2+b^2`, together with the positive and
negative Fourier chains, gives

```text
sum_resonant_ordered_triples |h_a||h_b|
 <=12 sum_(n>0)|h_(k_n)|^2.                        (3.4)
```

## 4. Compatible vertex coefficient

For one tensor vertex, put

```text
lambda_0=lambdahat(0)=1/8.                         (4.1)
```

If a nonzero normalized partition wave `s` has `r` occupied coordinates,

```text
|lambdahat(ms)|=lambda_0/2^r.                      (4.2)
```

Hence

```text
|ms| |lambdahat(ms)|
 =m sqrt(r) lambda_0/2^r
 <=m lambda_0/2.                                   (4.3)
```

Combining (2.3), (3.4), and (4.3) proves the velocity-mass envelope

```text
|B_chain|
 <=27m lambda_0 |Uhat|
       sum_n |hhat(k_n)|^2.                        (4.4)
```

This part is uniform in all Fourier phases and polarizations.

## 5. Axial steps and matrix-valued Hardy

Suppose `p=1`. Set

```text
D_n=k_n tensor hhat(k_n).                          (5.1)
```

After a unit-modulus gauge that absorbs the vertex sign and translation
phase, the exact positive-plus-negative chain Fisher form is

```text
E_(lambda,chain)
 =lambda_0 sum_edges ||D_(j+1)-D_j||_F^2.          (5.2)
```

No neighboring interface has been discarded.

Since `eta dot q=0` and the chain is one-sided,

```text
|k_n|>=jm
```

after reindexing the first occupied mode as `j=1`. The Hilbert-valued
discrete Hardy inequality gives

```text
sum_(j>=1) ||D_j||_F^2/j^2
 <=4 sum_(j>=1)||D_j-D_(j-1)||_F^2,

D_0=0.                                             (5.3)
```

As `|hhat(k_j)|=||D_j||_F/|k_j|`, equations (5.2)-(5.3) imply

```text
sum_j |hhat(k_j)|^2
 <=4 E_(lambda,chain)/(m^2 lambda_0).              (5.4)
```

Substitution into (4.4) proves

```text
|B_chain|
 <=108 (|Uhat|/m) E_(lambda,chain),     p=1.       (5.5)
```

The constant is conservative. Its purpose is uniformity under all the
new phase and polarization freedoms, not sharpness.

## 6. Multi-coordinate steps

When `p>=2`, the tensor vertex Fourier coefficient at the chain step is
strictly smaller:

```text
|lambdahat(mq)|=lambda_0/2^p.                      (6.1)
```

Bounding the signed neighboring edge by `2ab<=a^2+b^2` leaves the exact
mass floor

```text
E_(lambda,chain)
 >=2lambda_0(1-2^(1-p))
   sum_n ||D_n||_F^2.                              (6.2)
```

Also `|k_n|>=m sqrt(p)`. Equations (4.4) and (6.2) therefore give

```text
|B_chain|
 <=[27/{2p(1-2^(1-p))}]
   (|Uhat|/m) E_(lambda,chain).                    (6.3)
```

In particular,

```text
p=2:  constant=27/2=13.5,

p=3:  constant=6.                                  (6.4)
```

Thus the uniform constant over all primitive cube steps is the axial
constant `108`.

## 7. The omitted low phase is not a no-go

The strongest new stress uses:

```text
q=e_1,   eta=0,   ell=e_2,

Uhat(-e_2)=i e_3.                                  (7.1)
```

The two high polarizations are `e_3` and `e_2`. Pressure and the kinetic
scalar term vanish in this geometry. The anisotropic kinetic term becomes
a weighted velocity-mass pairing between the two polarization sequences.

If `G` is the exact Fisher difference matrix and

```text
P=diag(1,1/2,...,1/N),
```

the scalar generalized block is

```text
(1/2)PGP relative to G.                            (7.2)
```

Its largest finite-chain generalized eigenvalue increases monotonically
toward

```text
2/3.                                               (7.3)
```

At length `16384`, the audit obtains

```text
0.6666446802217748<2/3.                            (7.4)
```

The sine-phase block is therefore not invisible, but it is
Hardy-controlled. A preliminary scratch extractor that assigned instead
of accumulated two polarizations at the same Fourier mode falsely removed
the diagonal mass term. The production audit combines the polarizations
before installing each real mode and independently reconstructs the full
cubic coefficient. This is now regression-tested.

For comparison, the raw discrete Hardy generalized eigenvalue reaches
`3.17187238746049` at length `16384` and remains below the certified
constant `4`.

## 8. Sparse primitive atlas

The production audit tests:

- axial, planar-diagonal, and three-coordinate primitive steps;
- zero, canonical, oblique, and tilted transverse residues;
- disjoint and overlapping low-wave supports;
- arbitrary complex low polarizations;
- both high polarizations simultaneously at every Fourier mode;
- translated vertices with varying signs;
- partition scales `m=1,2,4`;
- constant-free first-Dirichlet random chain envelopes.

All 21 rows pass. The maximum complete component-versus-direct Fourier
residual is below

```text
5.9e-16,
```

and the maximum observed normalized ratio is

```text
m|B_chain|/(|Uhat|E_lambda)
 =0.004257588884.                                  (8.1)
```

The numerical ratios are not used to establish the constants. The proof
is the exact budget (2.3)-(6.3).

The canonical `1/2` theorem, Taylor-Green, seed-81, and the modulated-wave
HHL no-go all replay unchanged. The coarse constant `108` extends the
allowed geometry; it does not replace the sharper canonical result.

## 9. Route decision

Proved:

- a uniform complete-HHL envelope for one isolated primitive chain;
- arbitrary transverse residue within that chain;
- arbitrary low polarization and complex phase;
- arbitrary compatible vertex sign and common translation;
- exact constants `108`, `27/2`, and `6` by step support;
- stability under partition rescaling.

Not proved:

- simultaneous assembly of different primitive steps;
- simultaneous assembly of different residue chains;
- a finite low-wave/vertex Schur bound against one physical Fisher matrix;
- permission to sum isolated-chain Fisher charges by absolute values;
- absorption of all cross-shell HHL interactions;
- terminal dual control, critical `L^3`, or global regularity.

The next calculation must form the joint primitive-step/low-wave incidence
matrix while retaining the physical Fisher matrix once. Reusing (5.5)
separately for every chain would repeat the multiband recombination error.

The deterministic certificate is generated by
`scripts/primitive_hhl_chain_hardy_envelope_audit.py`; its production
result is
`results/primitive_hhl_chain_hardy_envelope_audit_v1.json`.
