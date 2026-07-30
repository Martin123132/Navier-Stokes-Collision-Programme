# Annular rho-zero first-jet gate

## 1. Result

The deficit-retaining restart theorem showed that the static-optimal annular
witness must amplify its normalized generator from order `N^3` to order
`N^5` on a parabolic window. This stage differentiates the exact `rho=0`
generator at the restart.

For

```text
u_N=h_N-a_N U,
lambda_N=t_N Phi_+++,
```

with `(a_N,t_N)` chosen by the static joint optimizer, the following are now
established.

1. The four directional first-variation formulas are exact.
2. A rectangular Fourier evaluator reproduces them without aliasing.
3. The complete first derivative is negative at
   `N=25,29,33,37,41`.
4. The viscous-pressure component has a rigorously negative `N^5` limit.

Numerically,

```text
D_u g_pressure[nu Delta u_N]/N^5
 -> -1.0442344590350905e-7/nu.                     (1.1)
```

The other first-jet components are much smaller on the audited carriers, but
their `o(N^5)` bounds are not yet proved. Therefore (1.1) is a theorem for
the dominant component, not yet a theorem for the total first jet or the
finite restart window.

## 2. Exact generator and directions

Remove the inessential outer factor three from the integrated replica
identity and write

```text
g_0(u,lambda)
 =integral[
    p(u) u dot grad lambda
    -nu lambda |grad u|^2
    -nu lambda |grad lambda|^2].                  (2.1)
```

For a velocity direction `v`, pressure linearizes as

```text
p'[u;v]=p[u,v]+p[v,u],

Delta p'[u;v]
 =-partial_i partial_j(u_i v_j+v_i u_j).
```

Direct differentiation gives

```text
D_u g_0[v]
 =integral[
    p'[u;v] u dot grad lambda
    +p v dot grad lambda
    -2nu lambda grad u:grad v].                   (2.2)
```

For a weight direction `mu`,

```text
D_lambda g_0[mu]
 =integral[
    p u dot grad mu
    -nu mu |grad u|^2
    -nu mu |grad lambda|^2
    -2nu lambda grad lambda dot grad mu].         (2.3)
```

At the restart the Navier-Stokes and backward-weight directions split into

```text
v_E  =-(u dot grad)u-grad p,
v_nu = nu Delta u,

mu_A  =-u dot grad lambda,
mu_nu =-nu Delta lambda.                          (2.4)
```

Thus

```text
d g_0/dt
 =D_u g_0[v_E]
  +D_u g_0[v_nu]
  +D_lambda g_0[mu_A]
  +D_lambda g_0[mu_nu].                           (2.5)
```

The symbolic audit expands (2.2) and (2.3) independently and leaves zero
residual.

## 3. Dealiased Fourier replay

The pressure variation in (2.2) can contain one velocity multiplied by a
quadratic velocity direction. Its intermediate support reaches three times
the one-field support. If

```text
K=(3N-1,(N-1)/2,(N-1)/2),
```

the evaluator takes each grid length at least `6K_j+1`, rounded upward to a
fast FFT length. This prevents wraparound before the final spatial mean.

Three checks are independent of the production rows.

- At `N=3`, all four analytic directions agree with central differences; the
  largest residual is `3.14e-10`.
- The `N=3` values on padding factors six and eight agree within
  `3.34e-16`.
- The Euler direction remains divergence free to numerical roundoff.

The base generator also replays the separate algebraic static objective:

```text
g_N
 =t_N[
    a_N |B_N|
    -nu(E_N+a_N^2/2)]
  -(nu/16)(75/256)t_N^3.                          (3.1)
```

The largest base-objective residual over the five production carriers is
below `7e-19`.

## 4. Exact heat-weighted HHL identity

Let `B_N<0` denote the static pressure-only HHL load with unit low amplitude
and unit coefficient scale. Every HHL monomial contains two high waves
`k_1,k_2` and one low wave `ell`.

Applying `nu Delta` to the three velocity slots multiplies that monomial by

```text
-nu(|k_1|^2+|k_2|^2+|ell|^2).
```

Define `B_heat,N` by inserting the positive sum of squares into the signed
static pressure load. Since the actual low field is `-a_N U`,

```text
D_u g_pressure[nu Delta u_N]
 =nu a_N t_N B_heat,N.                            (4.1)
```

This is a finite identity, not an asymptotic approximation. The audit
computes `B_heat,N` directly from the resonant HHL pairs and compares (4.1)
with the independently dealiased pressure variation in (2.2). The maximum
residual over all five carriers is

```text
1.96e-14.
```

At `N=25`, for example,

```text
B_heat,25                  =-307.1500278892374,
FFT viscous-pressure term =  -1.271415502175431,
HHL replay                =  -1.271415502175430.
```

## 5. Strictly negative continuum limit

Use the continuum variables

```text
xi=(x,y,z) in D=[2,3]x[-1/2,1/2]^2,
r^2=x^2+y^2+z^2,
S=sin(pi(x-2))sin(pi(y+1/2))sin(pi(z+1/2)),

V_y=-zy/r^3,
V_z=(x^2+y^2)/r^3.
```

The earlier pressure theorem gives

```text
B_N/N -> b_0
 =(sqrt(2)/20) integral_D S^2(V_y^2-V_z^2).       (5.1)
```

For the heat-weighted sum,

```text
(|k_1|^2+|k_2|^2+|ell|^2)/N^2
 ->2r^2
```

uniformly on the compact continuum domain. The high-high terms are Riemann
sums, while the weighted cross-pressure contribution is
`O(N^2)=o(N^3)`. Therefore

```text
B_heat,N/N^3 -> b_2
 =(sqrt(2)/10) integral_D
   S^2 r^2(V_y^2-V_z^2).                          (5.2)
```

The old pointwise estimate strengthens after multiplication by `r^2`:

```text
V_z^2-V_y^2 >=255/13718,
r^2(V_z^2-V_y^2)>=510/6859,
integral_D S^2=1/8.
```

Consequently

```text
b_0 <=-51sqrt(2)/438976<0,
b_2 <=-51sqrt(2)/54872<0.                         (5.3)
```

Order-64 tensor Gauss-Legendre quadrature, used only to report the limit
values and not to establish their signs, gives

```text
b_0=-0.0014065919385788297,
b_2=-0.017493957024435965.
```

## 6. Static optimizer and the N5 coefficient

Set `q=75/256`. The static ray optimizer obeys

```text
a_N/N -> |b_0|/nu,

t_N/N -> |b_0| sqrt(8/(3q))/nu
       =64|b_0|/(15sqrt(2)nu).                    (6.1)
```

Combining (4.1), (5.2), and (6.1) yields

```text
D_u g_pressure[nu Delta u_N]/N^5
 -> |b_0|^2 sqrt(8/(3q)) b_2/nu<0.                (6.2)
```

The quadrature value is (1.1). The purely analytic margins in (5.3) already
give the weaker but rigorous strict bound

```text
lim D_u g_pressure[nu Delta u_N]/N^5
 <=-1.0705252009222428e-10/nu<0.                  (6.3)
```

Thus the heat direction initially destroys, rather than amplifies, the
annular pressure escape at exactly the `N^5` scale demanded by the reset tax.

## 7. Complete finite first jet

For `nu=1`, the production rows are:

```text
 N    g_N             g'_N/N^5        pressure_nu/N^5   |remainder/pressure|
25   4.0077463e-5    -1.0968049e-7   -1.3019295e-7       0.15755
29   7.7003941e-5    -1.2392782e-7   -1.3488728e-7       0.08125
33   1.2107969e-4    -1.2817174e-7   -1.3433381e-7       0.04587
37   1.7377590e-4    -1.2874568e-7   -1.3243422e-7       0.02785
41   2.3652703e-4    -1.2795957e-7   -1.3029913e-7       0.01796
```

Here the remainder is the sum of the viscous Fisher, Euler, weight-advection,
and weight-antidiffusion contributions. It is positive on these rows but its
fraction of the negative pressure term decreases rapidly.

At `N=41`, the normalized decomposition is

```text
viscous pressure       -1.3029912962e-7
viscous Fisher         +1.8108444208e-9
Euler                  +4.3540527888e-10
weight advection       +8.1944590957e-12
weight antidiffusion   +8.5112272753e-11
total                  -1.2795957319e-7.
```

The linearized generator crossing occurs at
`N^2 tau=0.0234,...,0.0268`; the linearized time integral crosses at
`N^2 tau=0.0468,...,0.0536`. Hence the first-order replay is already negative
on the proposed scaled window `T=0.1`.

This last sentence is only a first-order diagnostic. A Taylor remainder
bound is required before it becomes a finite-window conclusion.

## 8. Scope and next gate

Established:

- exact velocity and weight first variations of the `rho=0` generator;
- independent finite-difference and padding validation;
- exact heat-weighted HHL replay of the viscous-pressure derivative;
- a strictly negative asymptotic `N^5` coefficient for that component;
- a negative complete first derivative on all five audited carriers.

Not established:

- `o(N^5)` bounds for every non-pressure first-jet component;
- a total asymptotic first-jet coefficient;
- a second-jet or Taylor-remainder bound on `T/N^2`;
- exclusion of the required nonlinear `N^2` amplification;
- optimization over dynamically relevant terminal weights;
- critical `L^3` control, blow-up, or global regularity.

The next theorem gate is therefore narrow. Prove uniform carrier bounds for
the four remainders, starting with the apparent `O(N^3)` Euler term and the
viscous weighted-Fisher commutator. If their sum is `o(N^5)`, (6.2) promotes
to the total first-jet limit. Only then is a second-time-jet calculation
worth launching.

## 9. Reproducibility

Run:

```text
python work/ns_collision/scripts/annular_rho_zero_first_jet_audit.py
```

The production record is
`results/annular_rho_zero_first_jet_audit_v1.json`.
