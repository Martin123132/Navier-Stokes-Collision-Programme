# Signed projected replica generator and the critical pressure gate

## Scope and status

This note completes the next analytic stage after
`projected_weber_replica_gate.md`. It derives the correlated two-replica
generator, its weighted form, a Gaussian correlation homotopy, and the
three-replica tensor generator while retaining the Leray projection before
taking a critical moment.

The identities below are exact for a classical smooth incompressible
Navier-Stokes solution and smooth projected stochastic Weber replicas on a
periodic domain, or under sufficient decay at infinity. They are not yet
justified at Leray regularity. No critical replica estimate, exceptional-set
upgrade, or global regularity theorem is claimed.

The main outcome is two-sided:

1. Independent replicas recover the exact Navier-Stokes energy equality.
   Correlating their Brownian drivers gives a monotone path from
   `|u|^2` to the pathwise projected second moment and exposes an exact
   decorrelation dissipation proportional to `1-rho`.
2. The scale-critical `L^3` lift has an exact weighted dual formulation, but
   its nonconstant optimizer makes the projected pressure flux reappear. A
   stored smooth periodic adversary has resolved, nonzero critical pressure
   work, so unweighted global pressure orthogonality cannot close the route.

## 1. Projected stochastic Weber equation

Let `V=V_W` be the pathwise Leray projection of a stochastic Weber covector.
For a fixed smooth deterministic velocity `u`, write its pressure correction
as `pi[V]`. With the convention used in this project,

```text
dV_i = [-u_j partial_j V_i + nu Delta V_i
        -partial_i u_j V_j - partial_i pi[V]] dt
       -sqrt(2nu) partial_k V_i dW_k,
div V = 0.
```

Taking a divergence gives the linear pressure equation

```text
Delta pi[V]
  = -partial_i(u_j partial_j V_i + partial_i u_j V_j).
```

Equivalently, after using

```text
(u dot grad)V + (grad u)^T V
  = grad(u dot V) - u cross curl V,
```

the projected equation is

```text
dV = [nu Delta V + P(u cross curl V)]dt
     -sqrt(2nu) grad V dW.
```

The stochastic Weber representation gives `E V=u`. Comparing its mean
equation with Navier-Stokes fixes the mean pressure gauge:

```text
E pi[V] = p - |u|^2/2
```

up to a function of time.

## 2. Correlated two-replica identity

Take two copies `V_1,V_2` whose Brownian drivers satisfy

```text
d<W_1^k,W_2^l>_t = rho delta_kl dt,   -1 <= rho <= 1.
```

Set `K=V_1 dot V_2`. Ito's product rule gives

```text
dK
 + [u dot grad K
    +2 V_1^T S V_2
    +div(pi_1 V_2 + pi_2 V_1)]dt
 = [nu Delta K
    -2nu(1-rho) grad V_1:grad V_2]dt
   +dM,
```

where `S=(grad u+(grad u)^T)/2` and

```text
dM = -sqrt(2nu)[
        V_2 dot partial_k V_1 dW_1^k
       +V_1 dot partial_k V_2 dW_2^k].
```

The coefficient `1-rho` is exact. The separate Laplacians contribute
`-2nu grad V_1:grad V_2` after rewriting them as `nu Delta K`, while the
cross quadratic variation returns `+2nu rho grad V_1:grad V_2`.

Define

```text
C_rho = E(V_1 dot V_2),
R_rho = E(V_1^T S V_2),
F_rho = E(pi_1 V_2 + pi_2 V_1),
G_rho = E(grad V_1:grad V_2).
```

Then

```text
partial_t C_rho + u dot grad C_rho
 +2R_rho + div F_rho
 = nu Delta C_rho -2nu(1-rho)G_rho.
```

On a periodic domain, or after all boundary terms vanish,

```text
d/dt E integral V_1 dot V_2
 = -2 E integral V_1^T S V_2
   -2nu(1-rho) E integral grad V_1:grad V_2.       (2.1)
```

This is a signed cross-energy balance. The strain term `R_rho` has no
universal sign. For the canonical construction in which both replicas are
the same functional of positively correlated Gaussian noises, Section 4
shows that `G_rho` is nonnegative and in fact bounded below by
`|grad u|^2`. That conclusion would not apply to unrelated replica
functionals or to arbitrary negative correlations.

## 3. The independent endpoint and reset law

At `rho=0`, the replicas are independent. Therefore

```text
C_0 = |u|^2,
R_0 = u^T S u,
G_0 = |grad u|^2,
F_0 = 2(p-|u|^2/2)u.
```

The strain term cancels after integration:

```text
integral u^T S u
 = integral u dot grad(|u|^2/2)
 = 0.
```

Consequently (2.1) becomes exactly

```text
d/dt ||u||_2^2 = -2nu ||grad u||_2^2.
```

This is an important consistency gate: the replica equation does not merely
resemble the energy equality; its independent endpoint is the equality.

At a reset time, when both pathwise replicas equal `u` before their future
noises separate,

```text
d/dt E integral(V_1 dot V_2)|reset
 = -2nu(1-rho)||grad u||_2^2.                      (3.1)
```

Thus decorrelation supplies viscosity continuously from the common-noise
endpoint `rho=1` to the independent endpoint `rho=0`.

Exact stress checks:

- For
  `u=(c+A exp(-nu k^2 t)sin(ky),0,0)`,
  the normalized reset derivative is
  `-nu(1-rho)A^2 k^2`.
- For unit ABC flow,
  `u(t)=exp(-nu t)u_ABC`,
  it is `-6nu(1-rho)`.
- For the algebraic strain
  `S=diag(-a/2,-a/2,a)`,
  the strain part `-2V_1^T S V_2` is `-2a` on `e_3` but `+a` on `e_1`.
  This last check is a local sign stress, not a claimed periodic solution.

## 4. Gaussian correlation homotopy

For `0<=rho<=1`, realize

```text
W_2 = rho W_1 + sqrt(1-rho^2) B
```

with `B` independent. For a square-integrable projected Weber functional,
expand at each spatial point in Wiener chaos:

```text
V(W,x) = sum_(n>=0) V_n(W,x).
```

The correlated pairing is

```text
C_rho(x)
 = E[V(W_1,x) dot V(W_2,x)]
 = sum_(n>=0) rho^n ||V_n(x)||_chaos^2.            (4.1)
```

It follows immediately that

```text
C_0 = |E V|^2 = |u|^2,
C_1 = E|V|^2,
C_rho is nondecreasing for 0<=rho<=1,
C_1-C_0 = Var(V).
```

Spatially differentiating the chaos expansion gives a second identity:

```text
G_rho
 = E[grad V(W_1):grad V(W_2)]
 = sum_(n>=0) rho^n ||grad V_n||_chaos^2
 >= ||grad V_0||^2
 = |grad u|^2,                                    (4.2)
```

for `0<=rho<=1`, assuming the spatially differentiated chaos series is
square summable. Hence the last term in (2.1) and (5.1) is genuinely
dissipative for nonnegative weights:

```text
-2nu(1-rho)G_rho
 <= -2nu(1-rho)|grad u|^2.
```

This is a useful coercive gain, not merely a formal interpolation.

When the functional is Malliavin differentiable,

```text
partial_rho C_rho
 = E <D V(W_1),D V(W_2)>_H,

Var(V) = integral_0^1 partial_rho C_rho d rho.     (4.3)
```

The audit independently checks (4.1) and (4.3) by Gauss-Hermite quadrature for a
three-component cubic Hermite functional. Its nonnegative chaos energies are

```text
1.25, 1.0625, 1.625, 0.3.
```

This gives a concrete projected tangent bridge. Unlike the raw deformation
Gramian, it includes the pathwise Leray projection and its pressure transfer.
It does not by itself bound the critical endpoint.

## 5. Weighted identity and exact critical dual

For a smooth deterministic weight `lambda(x,t)`, integrating the local
two-replica balance gives

```text
d/dt integral lambda C_rho
 = integral(lambda_t+u dot grad lambda+nu Delta lambda)C_rho
   -2 integral lambda R_rho
   +integral grad lambda dot F_rho
   -2nu(1-rho) integral lambda G_rho.              (5.1)
```

At `rho=0`, the kinetic part of `F_0` and the integrated strain term combine
exactly, leaving the standard weighted local-energy identity:

```text
d/dt integral lambda |u|^2
 = integral(lambda_t+u dot grad lambda+nu Delta lambda)|u|^2
   +2 integral p u dot grad lambda
   -2nu integral lambda |grad u|^2.                (5.2)
```

The critical cubic has the pointwise Legendre representation

```text
|u|^3
 = sup_(lambda>=0) [
     (3/2)lambda |u|^2 -(1/2)lambda^3].            (5.3)
```

Indeed, with `a=|u|`,

```text
a^3 - [(3/2)lambda a^2-(1/2)lambda^3]
 = (lambda-a)^2(lambda+2a)/2 >= 0,
```

and equality holds at `lambda=a`. The weight scales like velocity, so (5.3)
is exactly critical under Navier-Stokes scaling.

Equations (5.1)-(5.3) turn the target into a precise question:

> Can one choose a backward, adapted, or partitioned weight that controls the
> signed combination of strain and pressure flux while retaining enough of
> the `2nu(1-rho)G_rho` decorrelation term?

The pressure term is absent for a constant weight, but the critical optimizer
`lambda=|u|` is nonconstant. Global pressure orthogonality at the energy level
therefore does not answer the critical question.

## 6. Resolved pressure falsifier

The audit reuses the deterministic periodic finite-Fourier field from
`pressure_frame_pairing_audit.py` with seed `81` and velocity RMS `10`.
It resamples the exact stored trigonometric data onto grids
`48,64,80,96` and evaluates

```text
lambda_epsilon = sqrt(|u|^2+epsilon^2),
P_epsilon = -mean(lambda_epsilon u dot grad p),
N_epsilon = -mean(lambda_epsilon u dot (u dot grad)u).
```

For the critical limit `epsilon=0`,

```text
40.5122864 <= P_0 <= 40.5124923,
relative grid spread < 5.1e-6.
```

The largest absolute convective residual over the four grids is below
`2.99e-4`, and the `96^3` residual is about `5.12e-6`, consistent with the
exact periodic convective cancellation. In contrast, pressure work remains
positive. At the deliberately smooth weight `epsilon=1`, it still exceeds
`40.23` on every grid.

This rules out the easiest hoped-for statement:

```text
"global pressure cancellation at lambda=1 also cancels the critical
 lambda=|u| flux."
```

It does not rule out a joint pressure-strain estimate, a dynamically chosen
adjoint weight, or cancellation across partition edges and replica
correlations. The numbers are resolved floating-point stress evidence, not
an interval certificate.

## 7. Three-replica tensor generator

For three replicas with a positive-semidefinite correlation matrix
`(rho_rs)`, define

```text
T_ijk = E[V_1i V_2j V_3k]
```

and the one-replica drift operator

```text
B_u V = -u dot grad V -(grad u)^T V
        -grad pi[V] +nu Delta V.
```

Ito's rule gives

```text
partial_t T
 = E sum_(r=1)^3 slot_r(B_u V_r)
   +2nu sum_(r<s) rho_rs sum_k
      E[slot_r(partial_k V_r) slot_s(partial_k V_s)].  (7.1)
```

At the independent endpoint,

```text
T = u tensor u tensor u.
```

With `n=u/|u|`,

```text
(n tensor n tensor n):T = |u|^3.                  (7.2)
```

This is an exact signed cubic representation. It is not automatically cleaner
than the weighted two-replica equation: after contraction with `n`, pressure
is not a pure global divergence because derivatives hit `n` and the other
replica factors.

## 8. What this stage establishes

Established under the smooth assumptions:

- the projected stochastic Weber SPDE and linear replica pressure;
- the exact `rho`-dependent local and global two-replica balances;
- recovery of the energy equality at the independent endpoint;
- exact reset dissipation proportional to `1-rho`;
- the Wiener-chaos correlation homotopy from `|u|^2` to `E|V|^2`;
- the pointwise lower bound `G_rho>=|grad u|^2` for `0<=rho<=1`;
- the weighted replica identity and exact critical `L^3` dual;
- the three-replica tensor generator and independent cubic endpoint;
- a resolved smooth periodic falsifier for naive critical pressure
  cancellation.

Still open:

- a signed weighted pressure-strain estimate at critical scaling;
- an adjoint or partition weight that survives the pressure adversary;
- a quantitative use of the proved cross-gradient coercivity strong enough
  to pay the weighted pressure-strain flux;
- low-regularity construction of the projected replica system;
- conversion of a smooth a priori estimate into an exceptional-set theorem;
- global regularity of three-dimensional Navier-Stokes.

## 9. Next theorem target

The next bounded stage should derive the Euler-Lagrange or backward-adjoint
equation for `lambda` in (5.1), then connect its pressure term to the existing
partition-flux edge antisymmetry. A candidate is admissible only if it:

1. retains pressure and strain together before absolute values;
2. uses the proved
   `G_rho>=|grad u|^2` decorrelation dissipation quantitatively;
3. has critical scaling and a controlled terminal cost from (5.3);
4. survives exact shear, ABC flow, Burgers-strain sign stress, and the stored
   pressure adversary.

No large parameter search is warranted until that signed analytic target is
explicit.
