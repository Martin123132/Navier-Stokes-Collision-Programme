# Backward replica dual and the pressure-edge gate

## Scope and result

This stage starts from the signed projected replica balance in
`signed_projected_replica_generator.md`. It asks whether a terminal
Legendre weight can be propagated backward so that the favorable replica
cross-gradient term pays the critical pressure and strain flux.

The answer is useful but not yet a closure:

1. A backward advection-diffusion weight gives an exact restart inequality
   for the terminal `L^3` norm.
2. The cubic terminal penalty supplies a second, exact Fisher dissipation
   `lambda|grad lambda|^2`; this term must not be discarded.
3. At a replica reset, `rho=0` is instantaneously optimal. Positive
   correlation removes dissipation before it creates any new pathwise
   cancellation.
4. The flux cannot be universally nonpositive. An amplitude-scaled version
   of the stored smooth periodic pressure field has positive instantaneous
   `L^3` generator.
5. Pressure is nevertheless an exact antisymmetric transfer on a partition
   graph. The weight Fisher term becomes an exact cubic edge penalty. The
   unresolved gate is a degenerate reciprocal-weight edge remainder.

All identities are derived for classical smooth periodic Navier-Stokes
solutions and smooth projected Weber replicas reset at the left endpoint.
No low-regularity passage or global regularity result is claimed.

## 1. Replica restart data

Fix a smooth interval `[s,T]` and reset both projected replicas at time `s`:

```text
V_1(s)=V_2(s)=u(s).
```

For `0<=rho<=1`, the correlated replica quantities satisfy

```text
C_rho(s)=|u(s)|^2,
C_rho(T)>=C_0(T)=|u(T)|^2,
G_rho>=|grad u|^2.
```

The second and third statements follow from the spatial Wiener-chaos
expansions

```text
C_rho=sum_(n>=0) rho^n ||V_n||^2,
G_rho=sum_(n>=0) rho^n ||grad V_n||^2.
```

The weighted replica identity is

```text
d/dt integral lambda C_rho
 = integral(lambda_t+u dot grad lambda+nu Delta lambda)C_rho
   -2 integral lambda R_rho
   +integral grad lambda dot F_rho
   -2nu(1-rho) integral lambda G_rho.              (1.1)
```

## 2. Backward terminal weight

For arbitrary smooth terminal data `lambda_T>=0`, solve

```text
lambda_t+u dot grad lambda+nu Delta lambda=0,
lambda(T)=lambda_T.                                (2.1)
```

This is well posed in the backward time direction. With
`mu(tau)=lambda(T-tau)`,

```text
mu_tau=u(T-tau) dot grad mu+nu Delta mu.
```

The maximum principle preserves nonnegativity. Incompressibility and
integration by parts give the exact penalty identity

```text
integral lambda_T^3-integral lambda_s^3
 =6nu integral_s^T integral lambda|grad lambda|^2. (2.2)
```

Thus backward propagation contracts the `L^3` norm, and the amount of
contraction is itself a useful critical dissipation.

## 3. Exact restart dual inequality

The pointwise Legendre identity is

```text
|u(T)|^3
 =sup_(lambda_T>=0)[
    (3/2)lambda_T|u(T)|^2-(1/2)lambda_T^3].
```

For each terminal weight, use
`|u(T)|^2<=C_rho(T)`, propagate the first term by (1.1), use the reset
identity at `s`, and retain the exact penalty difference (2.2). This gives

```text
||u(T)||_3^3
 <= ||u(s)||_3^3
    +sup_(lambda_T>=0) integral_s^T Q_rho[lambda]dt, (3.1)
```

where

```text
Q_rho[lambda]
 = integral[
    -3lambda R_rho
    +(3/2)grad lambda dot F_rho
    -3nu(1-rho)lambda G_rho
    -3nu lambda|grad lambda|^2].                   (3.2)
```

The final term in (3.2) comes from the terminal dual penalty, not from the
replica equation. Dropping it would throw away half of the useful
rho-zero critical dissipation when `lambda` is close to `|u|`.

At zeros of `|u(T)|`, the exact optimizer may fail to be smooth. Formula
(3.1) is read first for smooth positive terminal approximations. Passing to
the exact supremum requires the corresponding standard density argument;
no positive lower bound uniform in the approximation is available.

## 4. Independent endpoint

At `rho=0`,

```text
R_0=u^T S u,
F_0=2(p-|u|^2/2)u,
G_0=|grad u|^2.
```

The strain term and the kinetic part of `F_0` cancel after spatial
integration. Hence (3.1) becomes

```text
||u(T)||_3^3
 <= ||u(s)||_3^3
    +3 sup_(lambda_T>=0) integral_s^T integral[
       p u dot grad lambda
       -nu lambda|grad u|^2
       -nu lambda|grad lambda|^2].                 (4.1)
```

For an infinitesimal interval and terminal optimizer `lambda=|u|`, (4.1)
recovers the exact classical cubic balance

```text
d/dt ||u||_3^3
 =3 integral[
    p u dot grad|u|
    -nu |u||grad u|^2
    -nu |u||grad|u||^2].                           (4.2)
```

Thus the adjoint formulation does not conceal the known pressure
obstruction. Its gain is that it remains valid for arbitrary terminal
weights over a finite restart interval and admits the replica interpolation.

## 5. Correlation ordering at a reset

At the reset instant, all replicas equal `u`, so `R_rho` and `F_rho` do not
yet depend on `rho`. Only the decorrelation dissipation changes. Therefore

```text
Q_rho[lambda]-Q_0[lambda]
 =3nu rho integral lambda|grad u|^2
 >=0.                                               (5.1)
```

Consequences:

- `rho=0` minimizes the instantaneous adverse generator.
- Positive correlation cannot improve a pointwise-in-time reset argument.
- Any benefit from `rho>0` must be genuinely finite-time: pathwise
  separation must reorganize `R_rho` or `F_rho` enough to repay the
  dissipation removed in (5.1).

This sharply narrows the role of the correlation homotopy.

## 6. Exact favorable stresses

### Positive periodic shear

For

```text
u=(c+A exp(-nu k^2 t)sin(ky),0,0),  c>A>0,
```

take terminal weight `lambda_T=|u(T)|=u_x(T)`. The exact backward solution is

```text
lambda(t,y)
 =c+A exp[-nu k^2(2T-t)]sin(ky).
```

Pressure vanishes. With

```text
c=2, A=0.7, nu=0.15, k=2, T=0.8,
```

the physical cubic falls from `9.47` to `8.562852542383414`. The initial
propagated dual is `9.190096425217607`, and the two exact Fisher payments
reconstruct the terminal value with zero floating-point residual.

### Unit ABC flow

For the unit Beltrami ABC field,

```text
Delta u=-u,  p=-|u|^2/2.
```

The pressure work in (4.2) is zero. On a `64^3` grid,

```text
integral |u||grad u|^2           =4.906706968455056,
integral |u||grad|u||^2          =0.9704816568734222,
integral |u|^3                   =5.877191402571214.
```

The Helmholtz balance residual is `2.78e-6`, with the error concentrated near
the nonsmooth speed at stagnation points.

## 7. Universal sign is false

The seed-81 finite-Fourier pressure adversary has, at its stored amplitude,

```text
P       =-mean(|u| u dot grad p)       about 40.5124,
D_u     = mean(|u||grad u|^2)          about 5151.13,
D_lambda= mean(|u||grad|u||^2)         about 1661.74.
```

Scale the velocity by a constant `alpha` while keeping the spatial scale
fixed. Then pressure scales by `alpha^2`, so the reset generator in (4.2)
is

```text
3 alpha^3[alpha P-D_u-D_lambda]        (nu=1).       (7.1)
```

Across grids `48,64,80,96`, its sign-change amplitude lies in

```text
[168.1671822,168.1678321].
```

At `alpha=170` and `alpha=200`, (7.1) is positive on every grid. The datum
is smooth and periodic, and local smooth Navier-Stokes evolution exists.
This supplies resolved evidence that the instantaneous cubic generator has
no universal nonpositive sign. It is not an interval certificate, a blow-up
example, or an obstruction to an integrated restart estimate.

For `rho=1`, the reset threshold drops to about `41.02` because the
velocity-gradient replica dissipation is absent, confirming (5.1).

## 8. Pressure as a scalar edge transfer

Let the tensor-product partition be generated in each coordinate by

```text
phi_+=(1+cos(x-x_*))/2,
phi_-=(1-cos(x-x_*))/2.
```

For eight nonnegative coefficients `w_b`, put

```text
lambda=sum_b w_b phi_b.
```

Define the scalar pressure-energy transfer in cell `b` by

```text
J_b=integral p u dot grad phi_b.
```

Because `sum_b grad phi_b=0`,

```text
sum_b J_b=0.                                       (8.1)
```

Pair cells that differ in one bit. Since the two one-dimensional
derivatives are opposite,

```text
integral p u dot grad lambda
 =sum_edges (w_0-w_1)E_edge.                       (8.2)
```

This is the scalar-energy analogue of the earlier Hessian-pressure
partition identity. Only neighboring weight differences survive.

The audit reconstructs scalar pressure from the stored pressure gradient to
`1.14e-13`. For a deterministic positive coefficient vector, direct,
cellwise, twelve-edge, and conditional representations of (8.2) agree to
better than `2e-12`.

The coefficient vector is chosen by reflecting an earlier positive vector
about a constant. Since constant coefficients are invisible to (8.2), this
reverses the pressure transfer while preserving strict positivity:

```text
min lambda=0.6,
integral p u dot grad lambda=1.280453496113641.
```

Scale both velocity and this smooth weight by `alpha`. The pressure,
velocity-Fisher, and weight-Fisher terms then give

```text
alpha^3[
  alpha*1.280453496113641
  -787.1192752552947
  -0.126477783203125].
```

Its sign changes at `alpha=614.8179183608784` and is positive at
`alpha=700`. This is a second resolved universal-sign falsifier whose weight
is smooth and bounded away from zero; it is independent of the cusp of
`lambda=|u|`.

## 9. Exact cubic edge penalty

Fix one coordinate direction and condition on the other two coordinates.
The partition weight has the one-dimensional form

```text
lambda=A phi_+ +B phi_-,
```

where `A,B>=0` are the two interpolated face weights. Let

```text
e=mean_normal_direction[p u_j partial_j phi_+].
```

Then the pressure transfer and adjoint Fisher term in this direction are
exactly

```text
P_j=mean_other[(A-B)e],                             (9.1)

D_j=mean_other[(A+B)(A-B)^2/16].                   (9.2)
```

Equation (9.2) is stronger than a bound based on a global minimum of
`lambda`: it remains meaningful when some vertex weights vanish. It is the
continuous cubic graph energy naturally paired with the pressure edge.

Pointwise Young optimization gives

```text
P_j-nu D_j
 <=(4/nu) mean_other[e^2/(A+B)].                   (9.3)
```

The convention at `A+B=0` requires the corresponding flux to vanish;
otherwise the right side is infinite.

For the audited positive eight-cell weight,

```text
exact pressure-Fisher-velocity flux = -785.9652995,
Young edge upper bound              = 2696.0167974.
```

Thus (9.3) is correct but destroys a large favorable cancellation on the
adversarial field. It does not close (4.1). The exact remaining gate is

```text
(4/nu) sum_j mean[e_j^2/(A_j+B_j)]
 ?<= nu(1-rho) integral lambda G_rho,              (9.4)
```

or a sharper signed substitute that does not separate the edge flux from
the replica velocity dissipation.

## 10. What this stage establishes

Established under the smooth assumptions:

- the exact backward terminal-weight penalty identity;
- the replica restart dual inequality (3.1)-(3.2);
- the physical rho-zero form (4.1);
- instantaneous optimality of `rho=0` at every reset;
- exact shear and ABC consistency checks;
- a smooth, resolved high-amplitude falsifier of universal flux sign;
- scalar pressure conservation on the partition graph;
- the exact conditional cubic Fisher edge form (9.2);
- the degenerate reciprocal-weight Young budget (9.3).

Still open:

- absorption of the edge remainder by replica velocity dissipation;
- a sharper signed estimate that preserves the observed cancellation;
- treatment of terminal weights approaching zero without a reciprocal loss;
- representation of the full terminal dual supremum by a scale-adapted
  multilevel partition;
- a finite-time mechanism by which `rho>0` can outperform `rho=0`;
- low-regularity construction and an exceptional-set upgrade;
- three-dimensional Navier-Stokes global regularity.

## 11. Next theorem target

The next bounded stage should attack (9.4) before introducing more
computation. Two routes remain live:

1. Derive a scale-adapted, degenerate edge estimate that couples `e_j`
   directly to `lambda G_rho` and avoids the loose reciprocal Young split.
2. Compute the first nontrivial finite-time expansion in `rho` after a
   replica reset. Since `rho>0` is worse at order zero, it must create a
   favorable correction in `R_rho` or `F_rho` at a later order to justify
   its use.

Every candidate must retain the terminal dual supremum and survive the
amplitude-scaled seed-81 pressure field.
