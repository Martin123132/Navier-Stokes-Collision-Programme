# First-Principles Collision Audit

Date: 2026-07-17

Status: exact identities, explicit stress tests, and open proof gates. This is
not a proof of Navier-Stokes regularity.

## 1. Question

For the incompressible Navier-Stokes system

```text
partial_t u + (u dot grad)u = -grad p + nu Delta u,
div u = 0,
```

can finite-time singularity be reformulated as a collision or collapse, and
does viscosity produce a singular separation barrier analogous to

```text
g' = A/g + regular terms,  A>0?
```

The word `collision` must not be left informal. The first audit distinguishes:

1. collision of two deterministic labelled parcels;
2. rank loss of a deterministic material element;
3. collision of two labels within one stochastic flow;
4. collision of two independent stochastic replicas;
5. simultaneous collapse of an independent stochastic cluster;
6. collision of zeros or poles in a heat/Cole-Hopf representation.

These events are not interchangeable.

## 2. Deterministic Pair Identities

Let

```text
dX(a,t)/dt = u(X(a,t),t),
R = X(a,t)-X(b,t),
G = |R|,
H = |R|^2,
delta u = u(X(a,t),t)-u(X(b,t),t).
```

Then, before any loss of classical regularity,

```text
R' = delta u,
G' = (R/G) dot delta u,
H' = 2 R dot delta u.
```

Taking one more derivative and using the momentum equation gives the exact
force-level identity

```text
H'' = 2|delta u|^2
      - 2 R dot delta(grad p)
      + 2 nu R dot delta(Delta u).
```

Writing the velocity increment along the joining segment,

```text
delta u = A_R R,
A_R = integral_0^1 grad u(X(b)+theta R,t) dtheta,
```

gives

```text
G' = alpha_R G,
alpha_R = (R/G) dot A_R (R/G).
```

Thus deterministic pair collision is equivalent to the accumulated
longitudinal strain satisfying

```text
integral_0^t alpha_R(s) ds -> -infinity
```

along the pair. No universal `+A/G` term appears in this first-order
deterministic identity. This is a derived fact, not a verdict on the whole
programme: pressure, viscosity, global finite energy, and cluster geometry may
still constrain the accumulated strain.

## 3. Material Deformation and Volume

For the deformation gradient

```text
F(a,t)=grad_a X(a,t),
```

one has

```text
F' = grad u(X(a,t),t) F,
(det F)' = (div u)(X(a,t),t) det F = 0.
```

Hence `det F=1` while the classical flow exists. A full three-dimensional
material volume cannot collapse in that regime. This does not prevent one
singular value of `F` tending to zero while another tends to infinity. The
appropriate geometric event may therefore be anisotropic rank degeneration,
not collapse of total volume.

For a finite deterministic cluster with centre `Xbar`,

```text
V = sum_i |X_i-Xbar|^2,
V' = 2 sum_i (X_i-Xbar) dot (u_i-ubar).
```

Again there is no automatic positive constant in `V'`.

## 4. Exact Affine-Strain Stress Test

Let

```text
u(x,t)=A(t)x,
A(t)=diag(-a(t),-a(t),2a(t)).
```

This is divergence free. Since `Delta u=0`, it solves Navier-Stokes locally
with

```text
Hess p = -(A'+A^2).
```

Taking `a(t)=1/(T-t)` gives

```text
p(x,t)=-3*x_3^2/(T-t)^2,
X_1(t)=X_1(0)*(T-t)/T,
X_2(t)=X_2(0)*(T-t)/T,
X_3(t)=X_3(0)*(T/(T-t))^2.
```

Pairs separated only in a compressive direction collide at `T`, while the
expanding direction diverges. This flow has infinite energy on `R^3` and is
not periodic, so it is not a Clay-admissible counterexample. Its role is
precise: no purely local pointwise argument from the differential equations
can manufacture deterministic pair repulsion. Any successful deterministic
barrier must use global admissibility or a different collision functional.

## 5. Stochastic Lagrangian Separation

The Constantin-Iyer representation uses stochastic flows satisfying

```text
dX = u(X,t)dt + sqrt(2*nu)dW.
```

There are two different pairings.

### 5.1 Common noise inside one flow

Two labels in the same stochastic flow use the same `W`. Their difference
satisfies

```text
dR = (u(X)-u(Y))dt.
```

The noise cancels. There is no Bessel repulsion in the label separation. This
is the pairing relevant to the spatial derivative and inverse of one flow.

### 5.2 Independent replicas

For independent Brownian motions,

```text
dX = u(X,t)dt + sqrt(2*nu)dW_1,
dY = u(Y,t)dt + sqrt(2*nu)dW_2,
dR = delta u dt + 2*sqrt(nu)dB.
```

In spatial dimension `d`, Ito's formula for `G=|R|` gives

```text
dG = [(R/G) dot delta u + 2*nu*(d-1)/G]dt
     + 2*sqrt(nu)d beta.
```

In three dimensions this is

```text
dG = [(R/G) dot delta u + 4*nu/G]dt
     + 2*sqrt(nu)d beta.                         (5.1)
```

This is an exact singular repulsive term with the same `1/G` form and sign as
the zero-gap barrier.

For `H=G^2`,

```text
dH = [2 R dot delta u + 4*d*nu]dt
     + 4*sqrt(nu*H)d beta.
```

In dimension three the positive cluster source is `12*nu`.

## 6. Exact Bessel Reduction and Collision Criterion

Define the longitudinal increment rate

```text
alpha_t = R dot delta u / |R|^2
```

away from `R=0`. Then in three dimensions

```text
dH = [2*alpha_t*H + 12*nu]dt + 4*sqrt(nu*H)d beta.  (6.1)
```

Set

```text
A_t = integral_0^t alpha_s ds,
Y_t = exp(-2*A_t) H_t,
s(t) = integral_0^t exp(-2*A_r)dr.
```

After this integrating factor and time change, `Z=Y/(4*nu)` obeys

```text
dZ = 3 ds + 2*sqrt(Z)dB_s.                         (6.2)
```

This is a squared Bessel process of dimension three. Consequently, finite
accumulated compression leaves the independent replicas non-colliding. The
only way for the drift to defeat this mechanism at a finite boundary is for
the longitudinal strain integral itself to become singular.

The affine model shows this gate is real. With

```text
alpha_t=-1/(T-t),
```

the relative SDE has the explicit solution

```text
R_t=(T-t)[R_0/T + 2*sqrt(nu)*integral_0^t dB_s/(T-s)],
```

and `R_t -> 0` at `T`. Instantaneous `4*nu/G` repulsion does not beat a
non-integrable compressive strain.

## 7. Sobolev Remainder Structure

For a Sobolev vector field, the standard maximal-function difference estimate
has the form

```text
|u(x)-u(y)|
  <= C_d |x-y| [M|grad u|(x)+M|grad u|(y)]
```

outside a negligible set. Therefore (5.1) has the schematic lower drift

```text
dG >= [4*nu/G - K_t G]dt + 2*sqrt(nu)d beta,
K_t=C_d[M|grad u|(X_t)+M|grad u|(Y_t)].             (7.1)
```

For a periodic incompressible drift and particles initially uniform in space,
the one-particle law remains uniform. The Leray energy estimate and the
`L^2` boundedness of the maximal operator then suggest

```text
E integral_0^T K_t dt
  <= C |domain|^(1/2) T^(1/2)
       ||grad u||_(L^2([0,T]x domain)) < infinity.  (7.2)
```

This should yield almost-sure non-collision for almost every independently
sampled starting pair, after the rough-drift SDE and approximation passage are
justified. It is deliberately recorded as a candidate lemma, not yet as a
proved theorem in this programme.

Even a proof of (7.2) would not settle regularity. Leray energy already gives
average control, while a singularity may live on an exceptional set. The
needed upgrade is from almost-everywhere replica non-collision to a quantity
that controls the inverse flow, deformation weights, or a critical norm of
`u`.

## 8. Independent Cluster Identity

For `N` independent replicas in dimension `d`, let

```text
Xbar = (1/N) sum_i X_i,
V = sum_i |X_i-Xbar|^2.
```

Ito's formula gives

```text
dV = 2 sum_i (X_i-Xbar) dot (u_i-ubar) dt
     + 2*nu*d*(N-1)dt
     + dM_t.                                       (8.1)
```

The constant positive source is the stochastic counterpart of the positive
internal cluster-variance source in the zero-flow calculation. Without the
velocity drift, the cluster radius is a Bessel object of effective dimension
`d(N-1)` and complete coincidence is polar whenever `d(N-1)>=2`.

This does not prevent lower-rank simplex degeneration. Gram-matrix and
smallest-singular-value dynamics are therefore a required next calculation.

## 9. Heat and Cole-Hopf Sign Audit

If a polynomial `phi` solves the ordinary heat equation

```text
phi_t=nu*phi_xx,
```

its simple roots satisfy

```text
z_j'=-2*nu sum_(k!=j) 1/(z_j-z_k).
```

For real roots this is attractive, opposite to the increasing-Newman
anti-heat direction. For example,

```text
phi(x,t)=x^2-a^2+2*nu*t
```

has real roots that collide at `t=a^2/(2*nu)`.

The Cole-Hopf transform for physical one-dimensional viscous Burgers starts
from a strictly positive `phi`; the heat maximum principle preserves that
non-vanishing and prevents Burgers poles. Three-dimensional incompressible
Navier-Stokes has no known scalar Cole-Hopf transform of this kind. A possible
invention target is therefore a positive matrix, determinant, or multi-flow
object whose loss of positivity is equivalent to deformation collapse.

## 10. Gram-Matrix Boundary Audit

The positive cluster source in (8.1) prevents complete coincidence, but a
cluster can lose rank without all points meeting. This distinction can be
audited exactly.

Apply an orthonormal centring transform to `N` independent replicas and put
their relative coordinates into a `d` by `m` matrix `Q`, where `m=N-1`. For
zero velocity drift,

```text
dQ=sqrt(2*nu)dB,
G=Q^T Q.
```

The eigenvalues of `G` form a real Wishart process. Near zero, its smallest
eigenvalue has the effective squared-Bessel dimension

```text
delta_eff=d-m+1.                                  (10.1)
```

In `d=3` this yields:

| configuration | `m` | degeneracy | `delta_eff` | driftless boundary |
|---|---:|---|---:|---|
| pair | 1 | coincidence | 3 | non-attainable |
| triangle | 2 | collinearity | 2 | critical non-attainable |
| tetrahedron | 3 | coplanarity/zero volume | 1 | attainable |

The tetrahedron conclusion also has a direct local explanation. For square
`Q`, let `D=det Q`. Since the determinant is linear in every matrix entry,
its matrix Laplacian vanishes. Under pure independent Brownian forcing,

```text
dD=sqrt(2*nu) cof(Q):dB,
d[D]_t=2*nu |cof(Q)|_F^2 dt.                       (10.2)
```

At a generic rank-two boundary, `cof(Q)` is nonzero, so `D` has a
nondegenerate one-dimensional martingale component and can cross zero. There
is no determinant analogue of the pair's `4*nu/G` barrier.

This does not contradict deterministic material-volume preservation. Labels
inside one stochastic flow have common noise, while the matrix `Q` above is
built from independent replicas. It does show that cluster-variance
non-collision cannot by itself be promoted to nondegeneration of every
geometric configuration.

## 11. Proof Gates

The programme currently has one exact match and four open gates.

### Exact match

Independent three-dimensional viscous replicas have the exact gap drift
`4*nu/G`, and independent clusters have a positive constant variance source.

### Gate A: representation fidelity

Independent replicas are legitimate in finite-replica stochastic Lagrangian
approximations, but deformation within one stochastic flow uses common noise.
We must connect replica separation to the spatial inverse or deformation of a
single flow. One-point non-collision is not enough.

### Gate B: collision necessity

Prove that a hypothetical Navier-Stokes singularity forces one of the selected
collision functionals to vanish, or forces its accumulated longitudinal
strain to diverge negatively. Loss of differentiability without literal
particle coincidence must be covered.

### Gate C: exceptional-set upgrade

Energy-class estimates naturally give almost-everywhere trajectory control.
Regularity requires excluding an exceptional singular point. A capacity,
two-point kernel, or inverse-flow estimate may be needed.

### Gate D: critical deformation control

The Constantin-Iyer formula depends on an entire stochastic flow and its
spatial inverse, not only one-point motions. The barrier must ultimately bound
the deformation weights that amplify vorticity in three dimensions.

## 12. Next Derivations

1. Derive a two-copy equation for inverse-flow differences in the
   Constantin-Iyer representation.
2. Determine whether an averaged inverse-deformation determinant or smallest
   singular value satisfies a closed parabolic inequality.
3. Prove or disprove the periodic Leray-class version of (7.2) through smooth
   approximation, with all exceptional-set qualifications explicit.
4. Derive the two-point velocity and vorticity correlation equations along
   independent replicas, starting at a common endpoint. Test whether radial
   separation weakens the stretching contribution at the diagonal.
5. Search for a critical norm controlled by two-replica separation or
   decorrelation; this is the required bridge to regularity.

## 13. Claim Ledger

Established algebraically:

- deterministic pair and force-level identities;
- material determinant conservation while the flow is classical;
- affine incompressible collapse stress test;
- common-noise cancellation;
- independent-replica `4*nu/G` radial drift in three dimensions;
- squared-Bessel integrating-factor reduction;
- independent-cluster positive variance source;
- Wishart/Gram effective boundary dimensions for pair, triangle, and
  tetrahedron configurations;
- generic zero-volume crossing for an independently diffusing tetrahedron;
- ordinary heat-root sign is attractive.

Not established:

- global existence of a three-dimensional stochastic flow at Leray
  regularity with every operation above justified;
- uniform non-collision for every starting pair;
- control of single-flow deformation by independent-replica separation;
- necessity of particle collision for Navier-Stokes blow-up;
- global smoothness of three-dimensional Navier-Stokes.
