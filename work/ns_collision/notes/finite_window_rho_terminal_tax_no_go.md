# Finite-window positive-correlation no-go in the backward replica dual

Status: exact smooth-class theorem for the canonical projected Weber
replicas with a deterministic nonnegative terminal weight. This closes the
proposed search for a net positive-`rho` advantage near `h=0.0756` in that
specific dual class. It does not prove a critical Navier-Stokes estimate or
global regularity.

## 1. Why this check comes before a solver

The preceding short-time calculation found

```text
integral_s^(s+h)(Q_rho-Q_0)dt
 =rho[K_0 h+(1/2)K_1 h^2+O(h^3)],

K_0 = 2361.35782576588,
K_1 = -62459.6458230194.
```

The quadratic truncation vanishes near

```text
h=0.0756122707598066.
```

That suggested a nonperturbative replica computation. Before building it,
however, the full endpoint identity must be used. It determines the sign
for every finite window without solving the replica covariance PDE.

## 2. Weighted endpoint identity

Let the canonical replicas be reset at time `s`, so

```text
C_rho(s)=|u(s)|^2.
```

For a smooth deterministic terminal datum `lambda_T>=0`, propagate
`lambda` backward by

```text
lambda_t+u dot grad lambda+nu Delta lambda=0.
```

The weighted two-replica identity then gives

```text
(3/2)[
  integral lambda_T C_rho(T)
 -integral lambda_s |u(s)|^2]

=integral_s^T integral[
   -3lambda R_rho
   +(3/2)grad lambda dot F_rho
   -3nu(1-rho)lambda G_rho].
```

The exact cubic-weight contraction is

```text
integral lambda_T^3-integral lambda_s^3
 =6nu integral_s^T integral lambda|grad lambda|^2.
```

Consequently the complete dual generator integral

```text
J_rho[lambda_T]
=integral_s^T integral[
   -3lambda R_rho
   +(3/2)grad lambda dot F_rho
   -3nu(1-rho)lambda G_rho
   -3nu lambda|grad lambda|^2]
```

has the exact endpoint form

```text
J_rho
 =(3/2)[
    integral lambda_T C_rho(T)
   -integral lambda_s|u(s)|^2]
  -(1/2)[
    integral lambda_T^3
   -integral lambda_s^3].                         (2.1)
```

Every term in (2.1), except `C_rho(T)`, is independent of `rho`.
Therefore

```text
J_rho-J_0
 =(3/2)integral lambda_T[C_rho(T)-C_0(T)].         (2.2)
```

Equation (2.2) is the terminal variance tax. It is not an estimate.

## 3. Wiener-chaos sign

For two copies of the same square-integrable Wiener functional with
correlation `0<=rho<=1`,

```text
C_rho(T,x)
 =sum_(n>=0)rho^n ||V_n(T,x)||_chaos^2,

C_0(T,x)=|E V(T,x)|^2=|u(T,x)|^2.
```

Thus

```text
C_rho(T,x)-C_0(T,x)
 =sum_(n>=1)rho^n ||V_n(T,x)||_chaos^2>=0.         (3.1)
```

Since `lambda_T>=0`, (2.2)-(3.1) imply

```text
J_rho[lambda_T]>=J_0[lambda_T]                    (3.2)
```

for every admissible deterministic terminal weight. The ordering survives
optimization over any common terminal-weight class:

```text
sup_(lambda_T) J_rho
 >=sup_(lambda_T) J_0.                             (3.3)
```

The left side is an upper-bound contribution, so larger is worse.
Therefore `rho=0` is globally optimal in this class, not merely
instantaneously optimal at a reset.

More strongly, the tax is absolutely monotone in `rho`:

```text
partial_rho(J_rho-J_0)
 =(3/2)integral lambda_T
   sum_(n>=1)n rho^(n-1)||V_n(T)||_chaos^2>=0.
```

At `rho=1`, the tax is exactly

```text
(3/2)integral lambda_T Var[V(T)].
```

## 4. What happened to the formal crossover

At `rho=0`,

```text
partial_rho(J_rho-J_0)
 =(3/2)integral lambda_T||V_1(T)||_chaos^2>=0.     (4.1)
```

The earlier polynomial

```text
K_0 h+(1/2)K_1 h^2
```

is only a short-time Taylor approximation to the nonnegative quantity
(4.1). Its formal zero near `0.0756` cannot be a sign change of the exact
accumulated correction. The uncontrolled higher time orders must become
material before the truncation could turn negative.

A finite-window stochastic solver could still test discretization and
variance, but it cannot discover a net positive-correlation improvement
that contradicts (2.2)-(3.1). That production computation is therefore no
longer the next research target.

## 5. Scope boundary

The theorem uses:

- canonical replicas represented by the same Wiener functional;
- a common deterministic reset at `s`;
- `0<=rho<=1`;
- a deterministic backward weight;
- a smooth nonnegative terminal weight;
- enough regularity for the weighted balance and chaos expansion.

It does not automatically cover:

- a stochastic or path-adapted weight, whose Ito covariations must first be
  derived;
- a sign-changing terminal construction;
- a genuinely different multi-replica inequality;
- low-regularity limiting solutions.

Negative correlation does not evade the endpoint issue for free. The
majorization `C_rho>=C_0` can fail there, so the physical terminal
Legendre functional requires an explicit correction. Adding the exact
correction restores the `rho=0` endpoint value.

Correlated replicas may remain a useful representation for exposing
coercive quantities. What is closed is the narrower premise that their
complete deterministic backward-dual generator can become smaller than
the independent one over a finite restart window.

## 6. Route decision

The main proof effort returns to `rho=0`. The live obstruction is now:

> Prove or sharply falsify a pressure-tail estimate at the intrinsic
> local-Reynolds frequency `m` comparable to `a/nu`, while preserving
> adaptive overlap, zero-face degeneracy, and the full nonnegative
> terminal-weight supremum.

The finite-window correlation branch should be reopened only after a new
dual structure is derived that falls outside the assumptions above.
