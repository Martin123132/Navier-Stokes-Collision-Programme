# Compatible-edge annular escape

## 1. Result

This stage puts the annular eight-vertex obstruction back into the complete
compatible coefficient objective. It retains all three adverse terms:

```text
complete pressure/transport flux,
weighted velocity Fisher dissipation,
exact twelve-edge coefficient Fisher penalty.
```

The resulting static coercivity statement is false. There is an explicit
smooth divergence-free family and a one-vertex compatible weight for which
the optimized instantaneous objective is positive from finite carrier size
and grows like a positive constant times `N^3`.

This is a no-go theorem for one instantaneous compatible-edge closure. It is
not a blow-up theorem, a failure of the actual adjoint evolution, or a
Navier-Stokes regularity result.

## 2. Exact normalized objective

At partition frequency `m=1`, write

```text
lambda_w=sum_v w_v Phi_v,               w_v>=0,

Q(w)=sum_j mean[H_j D_j^2].
```

After omitting the harmless common factor three, the normalized
instantaneous bracket is

```text
J(u,w)
 =mean[F(u) dot grad lambda_w]
  -nu mean[lambda_w |grad u|^2]
  -(nu/16)Q(w),                                    (2.1)
```

where `F(u)` is the complete local-energy flux used in the preceding HHL
audits. The coefficient `nu/16` and the cubic polynomial `Q` are exact.

Take a fixed nonnegative coefficient direction `z`, put `w=t z`, and write

```text
M(z)=sum_v z_v,

B_N(z)=sum_v z_v b_v(N),

D_N(z)=sum_v z_v E_v(h_N).
```

Orient the fixed low plane wave so its HHL load is positive and give it real
amplitude `a`. The complete objective becomes

```text
J_N(a,t;z)
 =t[a |B_N(z)|-nu(D_N(z)+a^2 M(z)/2)]
  -(nu/16)t^3 Q(z).                                (2.2)
```

Section 3 verifies that no omitted full-field term changes (2.2).

## 3. Full-field support check

The positive annular high modes have first coordinate in

```text
[2N,3N-1].
```

The gradient of every tensor vertex is supported in the cube
`[-1,1]^3` minus the origin. For `N>=3`:

```text
HHH: every mixed-sign output has |k_1|>=N+1>1;

HLL: every output has |k_1|>=2N>1;

high/low Fisher cross term: |k_1|>=2N>1.
```

Thus only HHL can reach the stencil. The low field is a single
divergence-free plane shear. Its local-energy flux has zero load at every
vertex because its Fourier flux is perpendicular to its wave vector.

The low weighted Fisher cost is exact and vertex independent. Its two modes
have wave norm squared `2`, coefficient norm squared `1`, and only the
constant Fourier coefficient `1/8` of `Phi_v` contributes. Hence

```text
E_v(U)=2 modes * 2 * (1/8)=1/2.                    (3.1)
```

Consequently,

```text
mean[lambda_(tz)|grad(h_N-aU)|^2]
 =t[D_N(z)+a^2M(z)/2].                             (3.2)
```

The audit independently expands the full cubic flux and full Fisher form at
`N=3` and checks (2.2) to roundoff. It also recomputes

```text
Q(delta_v)=75/256
```

at all eight vertices using the exact rational twelve-edge polynomial.

## 4. Exact two-variable optimization

For fixed `z`, first optimize (2.2) in `a`:

```text
a_N^*=|B_N(z)|/[nu M(z)],

A_N(z)
 =|B_N(z)|^2/[2nu M(z)]-nu D_N(z).                 (4.1)
```

The remaining scalar problem is

```text
sup_(t>=0) [t A_N-(nu/16)t^3Q(z)].
```

If `A_N<=0`, its supremum is zero at `t=0`. If `A_N>0`, then

```text
t_N^*
 =sqrt[16A_N/(3nu Q(z))],

max J_N
 =2A_N^(3/2)/[3sqrt(3nu Q(z)/16)].                 (4.2)
```

Equations (4.1)-(4.2) solve the radial coefficient optimization exactly.
The leading angular functional is

```text
|beta.z|^3/[M(z)^(3/2)sqrt(Q(z))].                 (4.3)
```

A global maximizer of (4.3) is unnecessary for the coercivity decision:
one certified positive direction falsifies the proposed universal bound.

## 5. Fixed-ray classification

The preceding annular theorem gives

```text
b_v(N)=N beta_v+o(N),

E_v(h_N)=Theta(N^(2r(v)-3)),
```

where `r(v)` is the number of minus signs in `v`.

For a fixed nonnegative ray with `z_--->0`, the `---` cell contributes
`Theta(N^3)` to `D_N(z)`, while the favorable square in (4.1) is only
`O(N^2)`. Such a ray is eventually suppressed.

If instead

```text
z_---=0,       beta.z !=0,
```

then `D_N(z)=O(N)` and the favorable term in (4.1) is `Theta(N^2)`. Every
such ray eventually escapes. Rays on this face with `beta.z=0` require
subleading analysis and are not classified here.

This explains both sides of the graph geometry. Neighboring Fisher energy
can suppress a coefficient direction, but nonnegativity does not force the
coefficient to put mass on the expensive `---` vertex.

## 6. The explicit `+++` escape

Choose

```text
z=delta_+++.
```

Then

```text
M=1,
Q=75/256,
D_N=E_+++(h_N)=Theta(N^-3),

B_N/N -> -beta_*,
beta_*=0.0014065919385788078... >0.
```

The sign is analytic:

```text
beta_+++
 =(sqrt(2)/20) integral_D S^2(V_y^2-V_z^2)<0.
```

For fixed `nu>0`,

```text
a_N^*/N
 -> beta_*/nu,

A_N/N^2
 -> beta_*^2/(2nu),

t_N^*/N
 -> 64 beta_*/(15sqrt(2)nu),

max J_N/N^3
 -> 32sqrt(2) beta_*^3/(45nu^2)>0.                (6.1)
```

Even imposing the bounded terminal coefficient `t=1` does not restore
nonpositivity:

```text
J_N(a_N^*,1)/N^2
 -> beta_*^2/(2nu)>0.                              (6.2)
```

For `nu=1`, the optimized coefficient first becomes positive among the
audited sizes at `N=25`. The bounded choice `t=1` is directly positive at
the audited size `N=137`.

## 7. What this closes

The following proposed implication is false:

```text
nonnegative compatible coefficients
+ exact twelve-edge cubic penalty
+ complete weighted velocity Fisher
=> universal nonpositive or uniformly coercive instantaneous bracket.
```

The obstruction includes the low-field Fisher price. It is not an artifact
of discarding neighboring cells, replacing the graph penalty by an edgewise
relaxation, or keeping only high-high pressure.

What remains open is substantially more structured:

- the actual backward-adjoint coefficient evolution;
- a coefficient law coupled to the velocity state;
- an explicit low-frequency, amplitude, or critical endpoint tax;
- nonhomogeneous controlled remainders;
- delayed trajectory compensation;
- critical `L^3` control and global regularity.

The next useful gate is therefore dynamic. Insert the escaping annular
family into the exact backward-adjoint evolution and determine whether its
required low amplitude and coefficient scale force a controlled endpoint
cost, or whether a positive restart-window contribution survives.

## 8. Reproducibility

Run:

```text
python work/ns_collision/scripts/compatible_edge_annular_escape_audit.py
```

The production record is
`results/compatible_edge_annular_escape_audit_v1.json`.
