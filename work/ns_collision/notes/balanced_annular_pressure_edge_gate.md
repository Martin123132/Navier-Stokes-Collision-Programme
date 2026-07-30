# Floor-free balanced-annular pressure-edge gate

Status: certified. The production audit and all five focused tests pass.
This stage controls the complete self-pressure edge of one bounded annular
velocity band. It does not control a full multiband Navier-Stokes pressure
edge.

## 1. Balanced band and partition

Let a smooth divergence-free field on the normalized three-torus satisfy

```text
supp uhat contained in {k: K<=|k|<=Lambda K}.       (1.1)
```

For an integer partition frequency `m>=1`, write

```text
Phi_v=psi_v^2,       sum_v Phi_v=1,                 (1.2)
```

where each one-dimensional factor of `psi_v` is a sine or cosine at
half-frequency `m/2`. Put

```text
E_v=integral Phi_v |grad u|^2.                      (1.3)
```

If `L=floor(Lambda K)`, the sharp residue-chain toggle theorem has constant

```text
C=C_(L,m)=cot(pi/[2(N+1)]),

N=ceil((2L+1)/m).                                  (1.4)
```

It applies to Hilbert-valued trigonometric polynomials and to odd as well as
even `m`.

## 2. One vertex sees the global enstrophy

Apply the toggle theorem to `grad u`. A vertex at Hamming distance `d` from
`v` obeys

```text
||psi_w grad u||_2<=C^d||psi_v grad u||_2.          (2.1)
```

Since the number of cube vertices at distance `d` is `binomial(3,d)`,

```text
||grad u||_2^2
 =sum_w ||psi_w grad u||_2^2
 <=(1+C^2)^3 E_v.                                  (2.2)
```

The lower carrier in (1.1) gives the ordinary high-pass Poincare estimate

```text
||u||_2^2
 <=K^(-2)(1+C^2)^3 E_v.                            (2.3)
```

Unlike the earlier shifted-support argument, (2.3) does not require
`K>sqrt(3)m`. The upper bandwidth is what keeps `C` finite.

## 3. Complete pressure, without an output cutoff

Let

```text
p=-R_iR_j(u_i u_j),       M(0)=0.                  (3.1)
```

The exact eight-shift Hamming identity is

```text
psi_v p
 =sum_(S subset {1,2,3})
   tau_(v,S) A_S(D)[psi_(v xor S)(u tensor u)].     (3.2)
```

The matrix-to-scalar double-Riesz symbol has Frobenius operator norm one.
Every Walsh average `A_S` therefore has `L2` norm at most one. With
`U=||u||_infinity`,

```text
||psi_v p||_2
 <=U sum_w ||psi_w u||_2
 <=sqrt(8) U||u||_2.                               (3.3)
```

No smooth output projection is used. Thus low-output opposite-carrier beats
and high pressure outputs of this one band are both included.

The exact derivative toggle gives

```text
||u grad psi_v||_2^2
 =(m^2/4)sum_(d(v,w)=1)||psi_w u||_2^2
 <=(m^2/4)||u||_2^2.                               (3.4)
```

Consequently, for the vertex pressure load

```text
P_v=integral p u dot grad Phi_v,
```

equations (2.3)-(3.4) yield

```text
|P_v|
 <=2sqrt(2)(1+C^2)^3
   [m U/K^2] E_v.                                  (3.5)
```

## 4. Compatible weights and intrinsic absorption

For arbitrary nonnegative coefficients,

```text
lambda=sum_v w_v Phi_v,       w_v>=0,              (4.1)
```

linearity, (3.5), and nonnegativity give

```text
|integral p u dot grad lambda|

 <=2sqrt(2)(1+C^2)^3 [m U/K^2]
   integral lambda|grad u|^2.                      (4.2)
```

There is no division by a coefficient, by `lambda`, or by an edge face.
Zero vertex coefficients are allowed.

The pressure term is absorbed by the replica velocity Fisher term whenever

```text
nu K^2
 >=2sqrt(2)(1+C^2)^3 m U.                          (4.3)
```

The exact cubic terminal-weight Fisher energy remains as additional
nonnegative dissipation; it is not spent in (4.2).

If

```text
kappa_- m<=K<=kappa_+ m                            (4.4)
```

and `Lambda`, `kappa_-`, and `kappa_+` are fixed, then

```text
N<=ceil(2 Lambda kappa_+ +1).                      (4.5)
```

Thus (4.3) follows from a scale-uniform intrinsic condition

```text
m>=C_(Lambda,kappa_-,kappa_+) U/nu.                (4.6)
```

The constant is explicit, although the direct estimate is deliberately not
claimed sharp.

## 5. Adversarial checks

The production audit uses sparse Fourier convolution rather than a sampled
pressure field.

For Taylor-Green, all eight compatible frequency-one pressure loads vanish,
as in the earlier graph audit. For seed-81, the nonzero stored compatible
pressure load is reconstructed, every vertex load satisfies (3.5), and the
positive cubic graph Fisher term is retained. Under

```text
u_(a,m)(x)=a u(mx),                                (5.1)
```

the quantities scale as

```text
P ->a^3 m P,
E ->a^2 m^2 E,
U ->a U,
K ->m K.                                           (5.2)
```

Both sides of (3.5) therefore scale identically. This is the required
co-scaling stress: the theorem does not conceal the local Reynolds number.

## 6. Scope and next gate

Established by the derivation:

- a complete-output pressure bound for one bounded annular field;
- floor-free extension to every compatible nonnegative vertex weight;
- intrinsic absorption by that field's weighted velocity Fisher energy;
- retention of the cubic terminal-weight Fisher term.

Not established:

- comparison of a spectral component's weighted Fisher energy with the
  weighted Fisher energy of the full multiband velocity;
- cross-shell `HHL` or mixed pressure interactions;
- construction of the scale-adapted terminal partition;
- the terminal dual supremum, a critical `L3` bound, exceptional-set
  removal, or Navier-Stokes regularity.

The next gate is a finite-overlap multiband inequality. Multiplication by a
frequency-`m` partition couples only shell pieces whose Fourier frequencies
differ by the partition stencil. A useful extension must retain that signed
finite graph and prove that the sum of component weighted Fisher costs is
controlled by the physical weighted Fisher form, or exhibit an exact
cancellation counterexample. The seed-81 and amplitude co-scaling families
remain mandatory falsifiers.

The deterministic certificate is generated by
`scripts/balanced_annular_pressure_edge_gate_audit.py`; the production result
is `results/balanced_annular_pressure_edge_gate_audit_v1.json`, with SHA-256
`9a024a23381d62e7842d7d26406fcea2a5343a168f386d3bad85e5308cef99dd`.
