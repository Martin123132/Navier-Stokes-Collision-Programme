# Deficit-retaining annular restart gate

## 1. Result

The compatible-edge annular family has a positive static generator even
after the velocity and coefficient Fisher terms are retained. That does not
yet make it a counterexample to the exact backward-adjoint restart formula.
The missing term is the reset-time Legendre deficit.

At `rho=0`, retaining this deficit gives the exact identity

```text
||u(T)||_3^3
 =||u(s)||_3^3
  +sup_(lambda_T>=0)[J_0[lambda_T]-Delta_s(lambda_s)].  (1.1)
```

For the static-optimal annular `+++` weight,

```text
Delta_s(lambda_s)>=c N^3.
```

The normalized pressure generator is also order `N^3`, but a parabolic
window has length `T/N^2`. Therefore the static witness can survive (1.1)
only if its time-averaged generator amplifies to order `N^5`, two powers of
`N` above its initial value.

This closes the claim that the positive static generator is by itself a
dynamic restart obstruction. It does not prove that the required nonlinear
amplification is impossible, control all terminal weights, or prove
Navier-Stokes regularity.

## 2. The exact deficit

For `r>=0` and `lambda>=0`, the Legendre identity has the pointwise
remainder

```text
r^3-[(3/2)lambda r^2-(1/2)lambda^3]
 =(r-lambda)^2(r+lambda/2).                       (2.1)
```

Define

```text
Delta(u,lambda)
 =integral (|u|-lambda)^2(|u|+lambda/2).
```

It is nonnegative and obeys

```text
Delta(u,lambda)
 >=(1/2)|| |u|-lambda ||_3^3.                     (2.2)
```

The exact optimizer is `lambda=|u|`; weights far from the speed pay a
critical cubic tax.

## 3. Deficit-retaining restart identity

Let `lambda` solve

```text
lambda_t+u dot grad lambda+nu Delta lambda=0,
lambda(T)=lambda_T>=0.
```

At `rho=0`, the strain term cancels the kinetic part of the replica flux.
The physical integrated generator is

```text
J_0[lambda_T]
 =3 integral_s^T integral[
    p u dot grad lambda
    -nu lambda|grad u|^2
    -nu lambda|grad lambda|^2].                   (3.1)
```

The weighted endpoint identity says that the terminal Legendre functional
equals its reset value plus (3.1). At the reset,

```text
L(u(s),lambda_s)
 =||u(s)||_3^3-Delta(u(s),lambda_s).
```

Taking the terminal supremum gives (1.1) exactly. The earlier stored restart
inequality discarded `-Delta_s` and then bounded the supremum of (3.1).
That inequality remains correct, but it is too loose for deciding whether
an arbitrary compatible weight is dynamically relevant.

## 4. A correction to the static load

The static annular audit used the complete local-energy HHL flux. The exact
`rho=0` generator uses pressure only:

```text
B_pressure
 =B_(p[h,h]U)+B_(p[h,U]+p[U,h])h.                 (4.1)
```

The kinetic HHL term must be removed because it has already cancelled the
strain term. The finite replay makes this correction explicitly.

It does not remove the static escape. At `N=25`,

```text
B_pressure                  =-0.0401680400630415,
E_+++(h_N)                  = 0.000223377700152842,
optimized normalized g_0   = 0.0000400774625053296.
```

The leading kinetic incidence is exactly zero, and the finite kinetic
correction rapidly vanishes. Pressure-only and complete loads therefore
have the same nonzero continuum limit.

## 5. Exact partition norms

For

```text
Phi=product_j (1+cos x_j)/2,
lambda_T=t Phi,
```

the exact moments are

```text
mean phi_+^3=5/16,

mean Phi^3=(5/16)^3=125/4096,

||lambda_T||_3=5t/16,

mean Phi|grad Phi|^2=75/4096.                     (5.1)
```

The low plane wave has two unit Fourier coefficients at waves of squared
norm two, so

```text
||aU||_2^2=2a^2.                                  (5.2)
```

The high and low Fourier supports are disjoint.

Backward propagation contracts the weight's `L^3` norm:

```text
||lambda_s||_3<=||lambda_T||_3.
```

Using (2.2), reverse triangle, `||u||_3>=||u||_2`, and Parseval gives

```text
Delta_s
 >=(1/2)(
    sqrt(||h_N||_2^2+2a^2)-5t/16
   )_+^3.                                         (5.3)
```

This bound does not require solving the backward weight equation.

## 6. Static-optimal annular scaling

Write

```text
B_N/N -> -beta_*,

beta_*=0.0014065919385788078...,

Q(delta_+++)=75/256.
```

The pressure-only static optimizer has

```text
a_N/N
 -> beta_*/nu,

t_N/N
 -> 64 beta_*/(15sqrt(2)nu),

g_N/N^3
 -> 32sqrt(2)beta_*^3/(45nu^2).                  (6.1)
```

Even if the high-field contribution in (5.3) is discarded,

```text
sqrt(2)a_N-5t_N/16
 =[sqrt(2)beta_*/(3nu)]N+o(N).
```

Therefore

```text
liminf Delta_s/N^3
 >=sqrt(2)beta_*^3/(27nu^3)>0.                   (6.2)
```

The ratio of this lower bound to three times the static normalized
generator is exact:

```text
Delta_s/(3g_N)
 >=5/(288nu)+o(1).                                (6.3)
```

At currently accessible finite sizes the high-field `L^2` term makes the
tax much larger than the asymptotic low-only bound. At `N=25`, (5.3) gives

```text
Delta_s>=0.5598052404531734.
```

## 7. The parabolic amplification gate

Take a restart window

```text
delta_N=T/N^2.
```

If the exact penalized contribution in (1.1) is positive, then necessarily

```text
average_(s,T) g_0
 >Delta_s/(3delta_N)
 =Omega(N^5).                                     (7.1)
```

Relative to the initial static value in (6.1), the required amplification
is

```text
average g_0/g_N(s)
 >=[5/(288nu T)]N^2+o(N^2).                       (7.2)
```

Thus heat-scale persistence of an order-`N^3` generator is insufficient.
The generator must grow by two additional carrier powers during the
window.

For `nu=1`, `T=0.1`, and `N=25`, the rigorous full-`L^2` lower bound
requires an average amplification exceeding `2.9e7`. This is a necessary
condition, not a proof that the actual dynamics cannot produce it.

## 8. Consequences

Established:

- the exact deficit-retaining `rho=0` restart identity;
- the pressure-only correction to the static load;
- persistence of the pressure-only static escape;
- an exact reset-deficit lower bound using backward contraction;
- an order-`N^3` tax for the static-optimal annular weight;
- the order-`N^2` dynamic amplification requirement on parabolic windows.

Not established:

- an upper bound on the first or later generator time derivatives;
- exclusion of nonlinear large-amplitude amplification;
- optimization over all terminal weights near `|u(T)|`;
- critical `L^3` control, blow-up, or global regularity.

The next calculation should differentiate the exact pressure-only
generator at the reset. Its leading possible size is `N^5`, precisely the
threshold exposed by (7.1). The first jet must be separated into viscosity,
low-high transport, pressure response, and backward-weight advection before
any second-order computation is justified.

## 9. Reproducibility

Run:

```text
python work/ns_collision/scripts/deficit_retaining_annular_restart_gate_audit.py
```

The production record is
`results/deficit_retaining_annular_restart_gate_audit_v1.json`.
